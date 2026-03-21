"""Code execution sandbox with restricted builtins and multiprocessing.

Ported from SDPO's code.py with simplifications for prime-rl training.
"""

from __future__ import annotations

import __future__ as __future_module__
import ast
import copy
import faulthandler
import io
import json
import multiprocessing
import re
import sys
import time
import traceback
from collections.abc import MutableMapping
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEOUT_MSG = "Time out"
ERROR_PREFIX = "Error: "
INCORRECT_FORMAT = "Incorrect format"
MAX_MEMORY_BYTES = 1024 * 1024 * 1024  # 1 GB
DEFAULT_TIMEOUT = 6.0
FILENAME = "Solution.py"

PROTECTED_GLOBAL_NAMES = {"__debug_buffer__", "debug_print"}

# Best-effort import to support setting memory limits on POSIX systems.
try:
    import resource as _resource  # type: ignore
except Exception:
    _resource = None  # type: ignore


# ---------------------------------------------------------------------------
# Security helpers (ported from SDPO)
# ---------------------------------------------------------------------------


def _set_memory_limits(maximum_memory_bytes: Optional[int] = MAX_MEMORY_BYTES) -> None:
    if maximum_memory_bytes is None or maximum_memory_bytes <= 0 or _resource is None:
        return
    try:
        if hasattr(_resource, "RLIMIT_AS"):
            _resource.setrlimit(_resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        for limit_name in ("RLIMIT_DATA", "RLIMIT_RSS"):
            if hasattr(_resource, limit_name):
                limit_const = getattr(_resource, limit_name)
                try:
                    _resource.setrlimit(limit_const, (maximum_memory_bytes, maximum_memory_bytes))
                except Exception:
                    pass
    except Exception:
        pass


def _build_restricted_builtins():
    """Create a minimal, safer builtins set for executing user code."""
    import builtins as _builtins

    allowed_names = {
        "abs": _builtins.abs,
        "all": _builtins.all,
        "any": _builtins.any,
        "bool": _builtins.bool,
        "bytes": _builtins.bytes,
        "callable": _builtins.callable,
        "chr": _builtins.chr,
        "dict": _builtins.dict,
        "enumerate": _builtins.enumerate,
        "filter": _builtins.filter,
        "float": _builtins.float,
        "format": _builtins.format,
        "frozenset": _builtins.frozenset,
        "hash": _builtins.hash,
        "hex": _builtins.hex,
        "int": _builtins.int,
        "isinstance": _builtins.isinstance,
        "issubclass": _builtins.issubclass,
        "iter": _builtins.iter,
        "len": _builtins.len,
        "list": _builtins.list,
        "map": _builtins.map,
        "max": _builtins.max,
        "min": _builtins.min,
        "next": _builtins.next,
        "object": _builtins.object,
        "ord": _builtins.ord,
        "pow": _builtins.pow,
        "print": _builtins.print,
        "range": _builtins.range,
        "repr": _builtins.repr,
        "reversed": _builtins.reversed,
        "round": _builtins.round,
        "set": _builtins.set,
        "slice": _builtins.slice,
        "sorted": _builtins.sorted,
        "str": _builtins.str,
        "sum": _builtins.sum,
        "tuple": _builtins.tuple,
        "zip": _builtins.zip,
        "input": _builtins.input,
        "type": _builtins.type,
        "vars": _builtins.vars,
        "getattr": _builtins.getattr,
        "setattr": _builtins.setattr,
        "hasattr": _builtins.hasattr,
        "delattr": _builtins.delattr,
        "id": _builtins.id,
        "dir": _builtins.dir,
        "bin": _builtins.bin,
        "oct": _builtins.oct,
        "complex": _builtins.complex,
        "divmod": _builtins.divmod,
        "property": _builtins.property,
        "staticmethod": _builtins.staticmethod,
        "classmethod": _builtins.classmethod,
        "super": _builtins.super,
        "memoryview": _builtins.memoryview,
        "bytearray": _builtins.bytearray,
        "BaseException": _builtins.BaseException,
        "Exception": _builtins.Exception,
        "ValueError": _builtins.ValueError,
        "TypeError": _builtins.TypeError,
        "KeyError": _builtins.KeyError,
        "IndexError": _builtins.IndexError,
        "AttributeError": _builtins.AttributeError,
        "RuntimeError": _builtins.RuntimeError,
        "StopIteration": _builtins.StopIteration,
        "StopAsyncIteration": _builtins.StopAsyncIteration,
        "ArithmeticError": _builtins.ArithmeticError,
        "ZeroDivisionError": _builtins.ZeroDivisionError,
        "OverflowError": _builtins.OverflowError,
        "FloatingPointError": _builtins.FloatingPointError,
        "LookupError": _builtins.LookupError,
        "AssertionError": _builtins.AssertionError,
        "NotImplementedError": _builtins.NotImplementedError,
        "IOError": _builtins.IOError,
        "OSError": _builtins.OSError,
        "EOFError": _builtins.EOFError,
        "ImportError": _builtins.ImportError,
        "NameError": _builtins.NameError,
        "SyntaxError": _builtins.SyntaxError,
        "IndentationError": _builtins.IndentationError,
        "TabError": _builtins.TabError,
        "SystemError": _builtins.SystemError,
        "UnicodeError": _builtins.UnicodeError,
        "UnicodeDecodeError": _builtins.UnicodeDecodeError,
        "UnicodeEncodeError": _builtins.UnicodeEncodeError,
        "UnicodeTranslateError": _builtins.UnicodeTranslateError,
        "RecursionError": _builtins.RecursionError,
        "True": True,
        "False": False,
        "None": None,
    }

    allowed_modules = {
        "math", "cmath", "itertools", "functools", "operator", "statistics",
        "random", "collections", "heapq", "bisect", "array", "string", "re",
        "typing", "json", "io", "fractions", "decimal", "dataclasses",
        "datetime", "time", "sys", "sortedcontainers", "numpy", "copy",
        "enum", "abc", "struct", "hashlib",
    }

    real_import = _builtins.__import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in allowed_modules:
            raise ImportError(f"Import of module '{name}' is not allowed")
        return real_import(name, globals, locals, fromlist, level)

    allowed_names["open"] = None  # deny file I/O
    allowed_names["__import__"] = restricted_import
    allowed_names["__build_class__"] = _builtins.__build_class__

    return allowed_names


def _reliability_guard():
    """Disable destructive functions in the subprocess."""
    faulthandler.disable()

    try:
        import warnings as _warnings
        _warnings.filterwarnings("ignore", category=SyntaxWarning)
    except Exception:
        pass

    import builtins
    _set_memory_limits(MAX_MEMORY_BYTES)

    builtins.exit = None
    builtins.quit = None
    builtins.open = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    for attr in [
        "kill", "system", "putenv", "remove", "removedirs", "rmdir",
        "fchdir", "setuid", "fork", "forkpty", "killpg", "rename",
        "renames", "truncate", "replace", "unlink", "fchmod", "fchown",
        "chmod", "chown", "chroot", "lchmod", "lchown", "getcwd", "chdir",
    ]:
        if hasattr(os, attr):
            setattr(os, attr, None)

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
    sys.modules["inspect"] = None
    sys.modules["ctypes"] = None
    sys.modules["threading"] = None
    sys.modules["multiprocessing"] = None
    sys.modules["socket"] = None
    sys.modules["ssl"] = None
    sys.modules["urllib"] = None
    sys.modules["requests"] = None

    try:
        sys._getframe = None  # type: ignore
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sandbox namespace
# ---------------------------------------------------------------------------


def _create_sandbox_namespace():
    """Return a fresh globals dict with restricted builtins."""
    ns = {"__builtins__": _build_restricted_builtins()}

    # Auto-import common typing aliases
    try:
        import typing as _typing
        for _name in (
            "Any", "Optional", "Union", "List", "Dict", "Set", "Tuple",
            "Callable", "Iterable", "Iterator", "Sequence", "Mapping",
            "MutableMapping", "MutableSequence", "MutableSet", "DefaultDict",
            "Deque", "FrozenSet", "Type", "TypeVar", "Generic", "Literal",
            "TypedDict", "NoReturn", "overload",
        ):
            if hasattr(_typing, _name):
                ns[_name] = getattr(_typing, _name)
    except Exception:
        pass

    ns["__name__"] = "__not_main__"
    return ns


class _GuardedLocals(MutableMapping):
    """Minimal locals proxy that blocks writes to protected names."""
    def __init__(self, backing):
        self._b = backing
    def __getitem__(self, key):
        return self._b[key]
    def __setitem__(self, key, value):
        if key in PROTECTED_GLOBAL_NAMES:
            return
        self._b[key] = value
    def __delitem__(self, key):
        if key in PROTECTED_GLOBAL_NAMES:
            return
        del self._b[key]
    def __iter__(self):
        return iter(self._b)
    def __len__(self):
        return len(self._b)


def _exec_with_guarded_locals(code_obj, globals_ns):
    exec(code_obj, globals_ns, _GuardedLocals(globals_ns))


# ---------------------------------------------------------------------------
# Short traceback helper
# ---------------------------------------------------------------------------


def _short_trace(e, limit=3):
    frames = traceback.extract_tb(e.__traceback__)
    solution_frames = [
        f for f in frames
        if isinstance(getattr(f, "filename", None), str) and f.filename == FILENAME
    ]
    tail = solution_frames[-limit:] if solution_frames else []
    lines = [f"{type(e).__name__}: {e}"]
    for f in tail:
        if f.line:
            lines.append(f"  {f.line}")
        lines.append(f"Line {f.lineno} in {f.name} ({FILENAME})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test runners (functional and stdin)
# ---------------------------------------------------------------------------


def _to_safe_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_safe_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_safe_jsonable(v) for k, v in value.items()}
    raise TypeError(f"Non-serializable result type: {type(value).__name__}")


def _run_test_functional(completion, test_input, test_output, fn_name, namespace):
    """Run a functional test: call fn_name with test_input args, compare to test_output."""
    code_obj = compile(
        completion, FILENAME, "exec",
        flags=__future_module__.annotations.compiler_flag,
        dont_inherit=True,
    )
    _exec_with_guarded_locals(code_obj, namespace)

    # Resolve function name
    if fn_name and fn_name in namespace and callable(namespace[fn_name]):
        func_name = fn_name
    else:
        # Infer from AST
        func_name = None
        try:
            tree = ast.parse(completion)
            if fn_name:
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                        func_name = fn_name
                        break
            if func_name is None:
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        break
        except Exception:
            pass
        if func_name is None or func_name not in namespace or not callable(namespace.get(func_name)):
            func_name = completion.split("(")[0].split()[-1]

    output = io.StringIO()
    sys.stdout = output

    try:
        if isinstance(test_input, dict):
            result_output = namespace[func_name](**test_input)
        elif isinstance(test_input, list):
            result_output = namespace[func_name](*test_input)
        else:
            test_input_args = [json.loads(x) for x in test_input.split()]
            result_output = namespace[func_name](*test_input_args)

        try:
            lhs = _to_safe_jsonable(result_output)
            rhs = _to_safe_jsonable(json.loads(test_output) if isinstance(test_output, str) else test_output)
            lhs_dump = json.dumps(lhs, sort_keys=True, separators=(",", ":"))
            rhs_dump = json.dumps(rhs, sort_keys=True, separators=(",", ":"))
            if lhs_dump != rhs_dump:
                return False, result_output
            return True, result_output
        except Exception as ser_err:
            return False, f"{ERROR_PREFIX}{ser_err}"

    except BaseException as e:
        return False, f"{ERROR_PREFIX}{_short_trace(e)}"
    finally:
        sys.stdout = sys.__stdout__


def _run_test_stdin(completion, test_input, test_output, namespace):
    """Run a stdin test: pipe test_input to stdin, capture stdout, compare to test_output."""
    # Normalize: test_input may be a string or a list (e.g. ['1 2\n'])
    if isinstance(test_input, list):
        test_input = "\n".join(str(x) for x in test_input)
    if isinstance(test_output, list):
        test_output = "\n".join(str(x) for x in test_output)
    output = io.StringIO()
    old_stdout, old_stdin = sys.stdout, sys.stdin
    try:
        sys.stdout = output
        sys.stdin = io.StringIO(test_input)
        code_obj = compile('__name__ = "__main__"\n' + completion, FILENAME, "exec")
        _exec_with_guarded_locals(code_obj, namespace)
        out = output.getvalue().strip().replace("\n", " ").replace("\r", "")
        expected = test_output.strip().replace("\n", " ").replace("\r", "")
        return out == expected, output.getvalue().strip()
    except BaseException as e:
        return False, f"{ERROR_PREFIX}{_short_trace(e)}"
    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


def _run_single_test(test_cases, completion, send_conn, test_idx):
    """Run a single test case in a subprocess. Sends result via pipe."""
    test_type = test_cases["testtype"]
    fn_name = test_cases.get("fn_name", "")
    namespace = _create_sandbox_namespace()
    _reliability_guard()

    test_input = test_cases["inputs"][test_idx]
    test_output = test_cases["outputs"][test_idx]

    try:
        time_start = time.time()
        try:
            if test_type == "functional":
                passed, output_value = _run_test_functional(
                    completion, copy.deepcopy(test_input),
                    copy.deepcopy(test_output), fn_name, namespace,
                )
            elif test_type == "stdin":
                # Normalize list inputs/outputs to strings
                t_in = test_input
                t_out = test_output
                if isinstance(t_out, list):
                    t_out = "\n".join(str(x) for x in t_out)
                if isinstance(t_in, list):
                    t_in = "\n".join(str(x) for x in t_in)
                clean_output = t_out.strip()
                if clean_output.endswith("-"):
                    clean_output = clean_output[:clean_output.rfind("-")].rstrip()
                passed, output_value = _run_test_stdin(
                    completion, copy.deepcopy(t_in),
                    copy.deepcopy(clean_output), namespace,
                )
            else:
                raise ValueError(f"Invalid test type: {test_type}")
        except BaseException as e:
            passed = False
            output_value = f"{ERROR_PREFIX}{_short_trace(e)}"
        finally:
            time_elapsed = time.time() - time_start

        record = {
            "test_idx": test_idx,
            "input": test_input,
            "expected": test_output,
            "actual": output_value,
            "passed": passed,
            "time": time_elapsed,
        }
        send_conn.send(record)

    except BaseException as outer_e:
        record = {
            "test_idx": test_idx,
            "input": test_input,
            "expected": test_output,
            "actual": f"ERROR: {_short_trace(outer_e)}",
            "passed": False,
            "time": float("inf"),
        }
        send_conn.send(record)
    finally:
        send_conn.close()


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def extract_code(response: str) -> str | None:
    """Extract code from the last ```python...``` block, or the last ``` block."""
    # Try python-specific blocks first
    python_blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if python_blocks:
        return _normalize_code(python_blocks[-1].strip())

    # Fall back to any code block
    blocks = re.findall(r"```\w*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return _normalize_code(max(blocks, key=len).strip())

    return None


def _normalize_code(code: str) -> str:
    """Normalize LeetCode-style class methods to standalone functions.

    Handles:
    - class Solution: wrapper (dedent methods)
    - self parameter in function signatures (strip it)
    """
    # Strip 'self' from function signatures (handles both class methods and bare methods)
    code = re.sub(r'\(\s*self\s*,\s*', '(', code)
    code = re.sub(r'\(\s*self\s*\)', '()', code)

    # Unwrap class Solution: / class Solution(object): wrappers
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.lower() in ('solution',):
                # Extract methods from the class and dedent them
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item)
                if methods:
                    lines = code.split('\n')
                    # Find the class body and dedent it
                    new_lines = []
                    in_class = False
                    class_indent = None
                    for line in lines:
                        stripped = line.lstrip()
                        if stripped.startswith('class ') and 'Solution' in stripped:
                            in_class = True
                            continue
                        if in_class:
                            if not stripped:
                                new_lines.append('')
                                continue
                            indent = len(line) - len(stripped)
                            if class_indent is None:
                                class_indent = indent
                            if indent >= class_indent:
                                new_lines.append(line[class_indent:])
                            else:
                                in_class = False
                                new_lines.append(line)
                        else:
                            new_lines.append(line)
                    return '\n'.join(new_lines)
    except SyntaxError:
        pass

    return code


# ---------------------------------------------------------------------------
# Feedback formatting (LeetCode-style, ported from SDPO)
# ---------------------------------------------------------------------------


def _format_test_feedback(
    records: list[dict],
    max_tests_to_show: int = 2,
    max_length: int = 2000,
    max_input_chars: int = 250,
    max_input_lines: int = 8,
    max_expected_chars: int = 250,
    max_actual_chars: int = 250,
) -> str:
    if not records:
        return "No test execution information available."

    def _trunc(value, max_chars):
        s = str(value) if not isinstance(value, str) else value
        return s[:max_chars] + "..." if len(s) > max_chars else s

    failing = [r for r in records if not r["passed"]]

    if not failing:
        return ""

    # Prioritize errors and timeouts
    error_rec = next(
        (r for r in failing if isinstance(r.get("actual"), str) and str(r["actual"]).startswith(ERROR_PREFIX)),
        None,
    )
    timeout_rec = next((r for r in failing if r.get("actual") == TIMEOUT_MSG), None)
    format_rec = next((r for r in failing if r.get("actual") == INCORRECT_FORMAT), None)

    selected = error_rec or timeout_rec or format_rec
    if selected is not None:
        failing = [selected]
    else:
        failing = sorted(failing, key=lambda x: len(str(x["input"])) + len(str(x["actual"])))
        failing = failing[:max_tests_to_show]

    parts: list[str] = []

    for r in failing:
        test_idx = r["test_idx"] + 1
        actual = r["actual"]
        expected = r["expected"]
        stdin = r["input"]

        is_error = isinstance(actual, str) and actual.startswith(ERROR_PREFIX)
        is_timeout = actual == TIMEOUT_MSG
        is_format = actual == INCORRECT_FORMAT

        if is_error:
            parts.append("Runtime Error")
            parts.append(actual[len(ERROR_PREFIX):])
            parts.append("")
            parts.append("Last Executed Input")
            if stdin is not None:
                lines = str(stdin).splitlines()
                for line in lines[:max_input_lines]:
                    parts.append(_trunc(line, max_input_chars))
                if len(lines) > max_input_lines:
                    parts.append(f"... ({len(lines) - max_input_lines} more lines)")
        elif is_timeout:
            parts.append("Time Limit Exceeded")
            parts.append("")
            parts.append("Last Executed Input")
            if stdin is not None:
                parts.append(_trunc(str(stdin), max_input_chars))
        elif is_format:
            parts.append("Incorrect Format: Put your code inside a ```python ... ``` block.")
        else:
            parts.append(f"Test Case {test_idx}: Wrong Answer")
            parts.append("")
            parts.append("Input")
            if stdin is not None:
                lines = str(stdin).splitlines()
                for line in lines[:max_input_lines]:
                    parts.append(_trunc(line, max_input_chars))
                if len(lines) > max_input_lines:
                    parts.append(f"... ({len(lines) - max_input_lines} more lines)")
            parts.append("")
            parts.append("Output")
            parts.append(_trunc(actual, max_actual_chars))
            if expected is not None:
                parts.append("")
                parts.append("Expected")
                parts.append(_trunc(expected, max_expected_chars))

        parts.append("")

    result = "\n".join(parts).rstrip()
    if len(result) > max_length:
        result = result[:max_length]
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_code(code: str, tests_json: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Execute code against test cases in sandboxed subprocesses.

    Args:
        code: Python source code to execute.
        tests_json: JSON string with keys: inputs, outputs, testtype, fn_name, time_limit.
        timeout: Per-test timeout in seconds.

    Returns:
        {
            "passed": bool,           # all tests passed?
            "num_passed": int,
            "num_total": int,
            "feedback": str,          # structured feedback string
            "results": list[dict],    # per-test results
        }
    """
    try:
        test_cases = json.loads(tests_json)
    except Exception:
        return {
            "passed": False,
            "num_passed": 0,
            "num_total": 0,
            "feedback": "Failed to parse test cases.",
            "results": [],
        }

    num_tests = len(test_cases.get("inputs", []))
    if num_tests == 0:
        return {
            "passed": False,
            "num_passed": 0,
            "num_total": 0,
            "feedback": "No test cases found.",
            "results": [],
        }

    timeout_per_test = float(test_cases.get("time_limit", timeout) or timeout)
    # Cap total time
    max_total_timeout = min(timeout_per_test * num_tests, 60.0)

    # Launch all tests in parallel
    process_data = []
    for test_idx in range(num_tests):
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(
            target=_run_single_test,
            args=(test_cases, code, child_conn, test_idx),
        )
        p.start()
        child_conn.close()
        process_data.append({"process": p, "parent_conn": parent_conn})

    # Collect results
    records = []
    start_time = time.time()
    for test_idx, data in enumerate(process_data):
        p = data["process"]
        parent_conn = data["parent_conn"]

        remaining = max(0, max_total_timeout - (time.time() - start_time))
        this_timeout = min(timeout_per_test + 1, remaining + 1)

        if parent_conn.poll(this_timeout):
            try:
                result = parent_conn.recv()
            except Exception:
                result = {
                    "test_idx": test_idx,
                    "input": test_cases["inputs"][test_idx],
                    "expected": test_cases["outputs"][test_idx],
                    "actual": f"{ERROR_PREFIX}Process communication error",
                    "passed": False,
                    "time": float("inf"),
                }
        else:
            result = {
                "test_idx": test_idx,
                "input": test_cases["inputs"][test_idx],
                "expected": test_cases["outputs"][test_idx],
                "actual": TIMEOUT_MSG,
                "passed": False,
                "time": float("inf"),
            }

        records.append(result)

        p.join(timeout=0)
        if p.is_alive():
            p.kill()
            p.join()

    num_passed = sum(1 for r in records if r["passed"])
    all_passed = num_passed == num_tests

    feedback = _format_test_feedback(records)
    if not feedback and all_passed:
        feedback = f"All {num_tests} test cases passed."

    return {
        "passed": all_passed,
        "num_passed": num_passed,
        "num_total": num_tests,
        "feedback": feedback,
        "results": records,
    }
