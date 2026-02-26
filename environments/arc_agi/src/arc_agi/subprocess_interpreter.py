"""Subprocess-based Python interpreter for RLM.

Replaces the Deno/Pyodide-based PythonInterpreter with a persistent CPython
subprocess. This enables native DSPy usage, numpy/scipy, and all installed
packages without WASM limitations.

Communication uses line-delimited JSON over stdin/stdout, following the same
protocol pattern as DSPy's PythonInterpreter.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import selectors
import subprocess
import sys
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CodeInterpreterError(Exception):
    """Error from the code interpreter."""

    pass


class HistoryReset:
    """Signal from interpreter that the agent wants to reset REPL history."""

    def __init__(self, summary: str):
        self.summary = summary


class FinalOutput:
    """Returned by interpreter.execute() when SUBMIT() is called.

    This signals that the code execution loop should terminate and return
    the contained output to the caller.
    """

    def __init__(self, output: Any):
        self.output = output

    def __repr__(self) -> str:
        return f"FinalOutput({self.output!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FinalOutput):
            return NotImplemented
        return self.output == other.output


# ---------------------------------------------------------------------------
# REPL loop script (runs inside the subprocess)
# ---------------------------------------------------------------------------

_REPL_SCRIPT = r'''
import json
import sys
import io
import traceback

# ---- Signal exceptions ----

class _FinalOutputSignal(Exception):
    def __init__(self, data):
        self.data = data

def _make_json_serializable(obj):
    """Recursively convert numpy arrays and other non-JSON types to serializable forms."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    return obj

class _ResetHistorySignal(Exception):
    def __init__(self, summary):
        self.summary = summary

class _ToolCallSignal(Exception):
    def __init__(self, name, args):
        self.name = name
        self.args = args

# ---- Globals for the sandbox ----

_tool_call_id = 0
_registered_tools = {}

def SUBMIT(**kwargs):
    """Submit final output and end the REPL session."""
    raise _FinalOutputSignal(kwargs)

def RESET_HISTORY(summary: str):
    """Compact REPL history into a summary to reduce noise (variables persist)."""
    raise _ResetHistorySignal(summary)

def _make_tool_proxy(tool_name):
    """Create a proxy function that calls a host-side tool via IPC."""
    def proxy(*args, **kwargs):
        global _tool_call_id
        _tool_call_id += 1
        raise _ToolCallSignal(tool_name, {"args": list(args), "kwargs": kwargs})
    proxy.__name__ = tool_name
    return proxy

# ---- REPL loop ----

namespace = {"SUBMIT": SUBMIT, "RESET_HISTORY": RESET_HISTORY, "__builtins__": __builtins__}

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

def recv():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line.strip())

# Process initial config
config = recv()

# Configure DSPy if requested
if config.get("dspy_lm"):
    try:
        import dspy
        lm_config = config["dspy_lm"]
        lm = dspy.LM(**lm_config)
        dspy.configure(lm=lm)
        namespace["dspy"] = dspy
    except Exception as e:
        pass  # DSPy config failure is non-fatal

# Register tools
for tool_info in config.get("tools", []):
    name = tool_info["name"]
    _registered_tools[name] = tool_info
    namespace[name] = _make_tool_proxy(name)

# Execute sandbox code if provided
if config.get("sandbox_code"):
    try:
        exec(config["sandbox_code"], namespace)
    except Exception as e:
        pass  # Sandbox code failure is non-fatal during setup

# Signal ready
send({"ready": True})

# Main REPL loop
while True:
    try:
        msg = recv()
    except (json.JSONDecodeError, EOFError):
        break

    if msg.get("shutdown"):
        break

    code = msg.get("code", "")
    variables = msg.get("variables", {})

    # Inject variables
    for k, v in variables.items():
        namespace[k] = v

    # Capture stdout
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured

    try:
        exec(code, namespace)
        sys.stdout = old_stdout
        output = captured.getvalue()
        send({"output": output})

    except _FinalOutputSignal as e:
        sys.stdout = old_stdout
        captured_output = captured.getvalue()
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = _make_json_serializable(e.data)
        send({"error": "FinalOutput", "errorType": "FinalOutput", "errorArgs": [serializable_data], "output": captured_output})

    except _ResetHistorySignal as e:
        sys.stdout = old_stdout
        send({"type": "reset_history", "summary": e.summary})

    except _ToolCallSignal as e:
        sys.stdout = old_stdout
        # Send tool call request to host and wait for response
        _tool_call_id += 1
        call_id = _tool_call_id
        send({"type": "tool_call", "id": call_id, "name": e.name, "args": e.args})

        # Wait for tool response
        response = recv()
        tool_result = response.get("result", "")
        tool_error = response.get("error")

        if tool_error:
            # Re-execute remaining code is not possible; report tool error as output
            output = captured.getvalue()
            send({"output": output + f"\n[Tool Error] {e.name}: {tool_error}"})
        else:
            # Inject tool result and let user access it
            # For simple tools, we store the result and report it
            output = captured.getvalue()
            send({"output": output + str(tool_result)})

    except SyntaxError as e:
        sys.stdout = old_stdout
        send({"error": str(e), "errorType": "SyntaxError"})

    except Exception as e:
        sys.stdout = old_stdout
        tb = traceback.format_exc()
        send({"error": tb, "errorType": type(e).__name__})
'''


# ---------------------------------------------------------------------------
# SubprocessInterpreter
# ---------------------------------------------------------------------------


class SubprocessInterpreter:
    """CodeInterpreter that runs a persistent CPython subprocess.

    Enables native DSPy usage, numpy/scipy, and all installed packages.
    State persists across execute() calls. Communication is JSON-RPC
    over stdin/stdout.

    Implements the CodeInterpreter protocol from DSPy.
    """

    def __init__(
        self,
        sandbox_code: str | None = None,
        tools: dict[str, Callable] | None = None,
        output_fields: list[dict] | None = None,
        dspy_lm: Any | None = None,
        allowed_modules: list[str] | None = None,
        timeout: float = 120.0,
    ):
        """
        Args:
            sandbox_code: Python code to inject into the subprocess namespace at startup.
            tools: Dictionary of tool name -> callable functions available via IPC.
            output_fields: Output field definitions for typed SUBMIT signature.
            dspy_lm: DSPy LM instance to configure in the subprocess. Its config
                     dict (model, temperature, etc.) is serialized and used to
                     create a matching LM in the subprocess.
            allowed_modules: Additional modules the subprocess is allowed to import.
            timeout: Per-execution timeout in seconds.
        """
        self._sandbox_code = sandbox_code
        self._tools: dict[str, Callable] = dict(tools or {})
        self._output_fields = output_fields
        self._dspy_lm = dspy_lm
        self._allowed_modules = allowed_modules
        self._timeout = timeout
        self._process: subprocess.Popen | None = None
        self._started = False
        # For compatibility with RLM's injection logic
        self._tools_registered = False

    @property
    def tools(self) -> dict[str, Callable]:
        return self._tools

    @tools.setter
    def tools(self, value: dict[str, Callable]) -> None:
        self._tools = value

    @property
    def output_fields(self) -> list[dict] | None:
        return self._output_fields

    @output_fields.setter
    def output_fields(self, value: list[dict] | None) -> None:
        self._output_fields = value

    def start(self) -> None:
        """Start the subprocess and configure it."""
        if self._started and self._process and self._process.poll() is None:
            return

        # Build environment - inherit current env plus any API keys
        env = os.environ.copy()

        self._process = subprocess.Popen(
            [sys.executable, "-c", _REPL_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,  # Line-buffered
        )

        # Send initial config
        config: dict[str, Any] = {}

        # DSPy LM config
        if self._dspy_lm is not None:
            lm_kwargs = self._extract_lm_config(self._dspy_lm)
            if lm_kwargs:
                config["dspy_lm"] = lm_kwargs

        # Tools info (names + parameter info for proxy creation)
        if self._tools:
            tools_info = []
            for name, fn in self._tools.items():
                tools_info.append(
                    {
                        "name": name,
                        "parameters": self._extract_parameters(fn),
                    }
                )
            config["tools"] = tools_info

        # Sandbox code
        if self._sandbox_code:
            config["sandbox_code"] = self._sandbox_code

        self._send(config)

        # Wait for ready signal
        response = self._recv()
        if not response.get("ready"):
            raise CodeInterpreterError(f"Subprocess failed to start: {response}")

        self._started = True
        self._tools_registered = True

    def shutdown(self) -> None:
        """Terminate the subprocess."""
        if self._process and self._process.poll() is None:
            try:
                self._send({"shutdown": True})
                assert self._process.stdin is not None
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
                self._process.wait()
        self._process = None
        self._started = False
        self._tools_registered = False

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute Python code in the subprocess.

        Returns:
            - FinalOutput if SUBMIT() was called
            - HistoryReset if RESET_HISTORY() was called
            - str with captured stdout otherwise

        Raises:
            CodeInterpreterError: On runtime errors
            SyntaxError: On syntax errors
        """
        if not self._started:
            self.start()

        variables = variables or {}

        # Serialize variables (only simple JSON-compatible types)
        serialized_vars = {}
        for k, v in variables.items():
            serialized_vars[k] = self._serialize_value(k, v)

        self._send({"code": code, "variables": serialized_vars})

        # Read and handle messages until we get output
        while True:
            result = self._recv()

            # Handle tool call requests
            if result.get("type") == "tool_call":
                self._handle_tool_call(result)
                continue

            # Handle RESET_HISTORY signal
            if result.get("type") == "reset_history":
                return HistoryReset(result.get("summary", ""))

            # Handle errors
            if "error" in result:
                error_msg = result["error"]
                error_type = result.get("errorType", "RuntimeError")
                error_args = result.get("errorArgs", [])

                if error_type == "FinalOutput":
                    final_data = error_args[0] if error_args else None
                    captured_output = result.get("output", "")
                    return FinalOutput(final_data), captured_output
                elif error_type == "SyntaxError":
                    raise SyntaxError(f"Invalid Python syntax: {error_msg}")
                else:
                    raise CodeInterpreterError(f"{error_type}: {error_msg}")

            # Normal output
            return result.get("output", "")

    # ---- Internal helpers ----

    def _send(self, msg: dict) -> None:
        """Send JSON message to subprocess stdin."""
        if self._process is None or self._process.poll() is not None:
            stderr = ""
            if self._process and self._process.stderr:
                try:
                    stderr = self._process.stderr.read()
                except Exception:
                    pass
            exit_code = self._process.returncode if self._process else None
            raise CodeInterpreterError(
                f"Subprocess is not running (exit code: {exit_code})" + (f". Stderr: {stderr}" if stderr else "")
            )
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(msg) + "\n")
        self._process.stdin.flush()

    def _recv(self, timeout: float | None = None) -> dict:
        """Read JSON message from subprocess stdout.

        Args:
            timeout: Timeout in seconds. If None, uses self._timeout.
        """
        if self._process is None or self._process.poll() is not None:
            stderr = ""
            if self._process and self._process.stderr:
                stderr = self._process.stderr.read()
            raise CodeInterpreterError(f"Subprocess is not running. Stderr: {stderr}")

        assert self._process.stdout is not None

        # Use selectors for timeout (supports >1024 file descriptors unlike select.select)
        timeout_secs = timeout if timeout is not None else self._timeout
        if timeout_secs:
            sel = selectors.DefaultSelector()
            sel.register(self._process.stdout, selectors.EVENT_READ)
            try:
                ready = sel.select(timeout=timeout_secs)
                if not ready:
                    # Timeout - kill the subprocess
                    self._process.kill()
                    self._process.wait()
                    raise CodeInterpreterError(f"Code execution timed out after {timeout_secs} seconds")
            finally:
                sel.close()

        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise CodeInterpreterError(f"No output from subprocess. Stderr: {stderr}")

        try:
            return json.loads(line.strip())
        except json.JSONDecodeError as e:
            raise CodeInterpreterError(f"Invalid JSON from subprocess: {line.strip()!r}") from e

    def _handle_tool_call(self, msg: dict) -> None:
        """Handle a tool call request from the subprocess."""
        tool_name = msg.get("name", "")
        call_id = msg.get("id", 0)
        args_info = msg.get("args", {})

        tool_fn = self._tools.get(tool_name)
        if tool_fn is None:
            self._send(
                {
                    "type": "tool_response",
                    "id": call_id,
                    "result": None,
                    "error": f"Unknown tool: {tool_name}",
                }
            )
            return

        try:
            # Call the tool with positional and keyword args
            positional = args_info.get("args", [])
            keyword = args_info.get("kwargs", {})
            result = tool_fn(*positional, **keyword)
            self._send(
                {
                    "type": "tool_response",
                    "id": call_id,
                    "result": str(result) if result is not None else "",
                    "error": None,
                }
            )
        except Exception as e:
            self._send(
                {
                    "type": "tool_response",
                    "id": call_id,
                    "result": None,
                    "error": str(e),
                }
            )

    @staticmethod
    def _serialize_value(name: str, value: Any) -> Any:
        """Serialize a value for JSON transport to subprocess."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, dict, tuple)):
            # Let json.dumps handle the validation
            try:
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                raise CodeInterpreterError(f"Variable '{name}' contains non-serializable types")
        if hasattr(value, "tolist"):
            # numpy arrays
            return value.tolist()
        raise CodeInterpreterError(f"Cannot serialize variable '{name}' of type {type(value).__name__}")

    @staticmethod
    def _extract_parameters(fn: Callable) -> list[dict]:
        """Extract parameter info from a callable for tool registration."""
        params = []
        try:
            sig = inspect.signature(fn)
            for p_name, p in sig.parameters.items():
                param_info: dict[str, Any] = {"name": p_name}
                if p.annotation != inspect.Parameter.empty:
                    type_name = getattr(p.annotation, "__name__", str(p.annotation))
                    param_info["type"] = type_name
                if p.default != inspect.Parameter.empty:
                    param_info["default"] = p.default
                params.append(param_info)
        except (ValueError, TypeError):
            pass
        return params

    @staticmethod
    def _extract_lm_config(lm: Any) -> dict[str, Any] | None:
        """Extract serializable config from a DSPy LM instance."""
        try:
            config: dict[str, Any] = {}
            if hasattr(lm, "model"):
                config["model"] = lm.model
            if hasattr(lm, "temperature"):
                config["temperature"] = lm.temperature
            if hasattr(lm, "max_tokens"):
                config["max_tokens"] = lm.max_tokens
            # Include cache setting
            if hasattr(lm, "cache"):
                config["cache"] = lm.cache
            return config if config else None
        except Exception:
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
