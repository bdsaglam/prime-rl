"""REPL-based environment for ARC-AGI.

Provides a persistent Python REPL with pre-loaded task data and utilities.
The model responds with structured sections (reasoning, code) and
code is executed in a subprocess interpreter.

Based on RLM v3 architecture from epiq.
"""

from __future__ import annotations

import json
import re
from typing import Any

import verifiers as vf

from ..subprocess_interpreter import CodeInterpreterError, FinalOutput, SubprocessInterpreter

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Pattern to extract sections: [[ ## section_name ## ]]
_SECTION_PATTERN = re.compile(
    r"\[\[\s*##\s*(\w+)\s*##\s*\]\]\s*\n(.*?)(?=\[\[\s*##|\Z)",
    re.DOTALL,
)

# Fallback: code fence extraction
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def parse_response(text: str) -> dict[str, str]:
    """Parse structured response into sections.

    Expected format:
        [[ ## reasoning ## ]]
        <thinking about the problem>

        [[ ## code ## ]]
        ```python
        <python code to execute>
        ```

    Multiple [[ ## code ## ]] sections are concatenated in order.

    Returns dict with 'reasoning' and 'code' keys (may be empty).
    """
    sections: dict[str, str] = {
        "reasoning": "",
        "code": "",
    }

    code_sections: list[str] = []

    matches = _SECTION_PATTERN.findall(text)
    for name, content in matches:
        name = name.lower().strip()
        if name == "code":
            code_sections.append(content.strip())
        elif name in sections:
            sections[name] = content.strip()

    # Extract code from within fences for each code section and concatenate
    if code_sections:
        all_code_blocks: list[str] = []
        for section_content in code_sections:
            code_matches = _CODE_BLOCK_PATTERN.findall(section_content)
            if code_matches:
                all_code_blocks.extend(m.strip() for m in code_matches)
            else:
                stripped = _strip_code_fences(section_content)
                if stripped:
                    all_code_blocks.append(stripped)
        sections["code"] = "\n".join(all_code_blocks)

    # Fallback: if no code section, try to extract code from markdown fences anywhere
    if not sections["code"]:
        code_matches = _CODE_BLOCK_PATTERN.findall(text)
        if code_matches:
            sections["code"] = "\n".join(m.strip() for m in code_matches)

    return sections


def _strip_code_fences(code: str | None) -> str:
    """Strip markdown code fences from code."""
    if not code:
        return ""
    result = code.strip()
    # Remove leading ```python or ```
    result = re.sub(r"^```(?:python|py)?\s*\n?", "", result)
    # Remove trailing ```
    result = re.sub(r"\n?```\s*$", "", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Sandbox setup code
# ---------------------------------------------------------------------------

ARC_SANDBOX_CODE = '''\
import json
import numpy as np

# --- ARC task data (auto-injected) ---
task = json.loads({task_json!r})

# --- Utility functions ---

def format_grid(grid):
    """Convert grid to string format."""
    if hasattr(grid, 'tolist'):
        grid = grid.tolist()
    return "\\n".join(" ".join(str(cell) for cell in row) for row in grid)

def accuracy(pred, expected):
    """Compute exact match accuracy: 1.0 if grids are identical, 0.0 otherwise."""
    pred = np.array(pred)
    expected = np.array(expected)
    if pred.shape != expected.shape:
        return 0.0
    return 1.0 if np.array_equal(pred, expected) else 0.0

def soft_accuracy(pred, expected):
    """Compute cell-level accuracy: fraction of cells that match (0.0 to 1.0)."""
    pred = np.array(pred)
    expected = np.array(expected)
    if pred.shape != expected.shape:
        return 0.0
    return float(np.mean(pred == expected))
'''

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert at solving Abstract Reasoning Corpus (ARC) tasks. Your goal is to analyze
input-output examples and produce correct output predictions for the test inputs.

**1. Analyze the Examples:**
  * Identify key objects in input/output grids (shapes, lines, regions) using `scipy.ndimage.label` etc.
  * Determine relationships between objects (spatial arrangement, color, size).
  * Identify operations that transform input to output (rotation, reflection, color change, addition/removal).
  * Consider grid dimensions, symmetries, and other visual features.

**2. Formulate a Hypothesis:**
  * Based on your analysis, formulate a transformation rule that works across all examples.
  * Prioritize simpler rules first.
  * **Generalisation Check:** Will your rule generalise to the test inputs?
  * **Generalisation Advice:**
    * **Orientation/Direction/Shape:** Ensure your hypothesis covers symmetric cases.
    * **Avoid Arbitrary Constants:** Don't rely on constants tuned to training examples (thresholds, offsets, dimensions).
  * Consider these transformation types:
    * **Object Manipulation:** Moving, rotating, reflecting, or resizing objects.
    * **Color Changes:** Changing colors of specific objects or regions.
    * **Spatial Arrangements:** Rearranging objects in specific patterns.
    * **Object Addition/Removal/Swapping:** Based on certain criteria.
    * **Global vs. Local:** Consider whether components are global or local.

**3. Construct Outputs Directly:**

Once you figured out the transformation rule in the puzzle, apply it to the test inputs to get the output grids.
You may write a general `transform()` or `solve()` function when the rule is simple and clearly algorithmic.
However, for visually complex puzzles, a complete general transformation function can be hard to implement robustly in pure Python. 
In those cases, build each output grid directly through step-by-step edits in the REPL.

**Why?** Visual pattern recognition (e.g., locating objects, understanding spatial relationships,
or identifying enclosed regions) is often easier by inspection than by writing full detection algorithms
(flood-fill, connected components, complex boundary logic). In those cases, use this loop:
  * **Inspect** printed grids and identify key regions/objects/patterns. You can zoom in on some parts of the grid to see more details using slicing.
  * **Edit/Apply** targeted numpy operations to construct the output
  * **Inspect again** to verify and iterate until the grid matches the rule

Example workflow:
[[ ## reasoning ## ]]
<Your step-by-step reasoning>

[[ ## code ## ]]
```python
# Start with a copy of the input (or create fresh grid)
pred = np.array(task["test"][0]["input"]).copy()

# Make targeted edits based on what you observe
pred[2:5, 3:6] = 4  # Fill region you identified visually
pred[0, :] = 1      # Change top row

# Print to verify
print(format_grid(pred))
```

**4. Test on Training Examples:**

Before submitting, verify your approach works on training examples:
[[ ## code ## ]]
```python
# Apply same strategy to a training input
train_in = np.array(task["train"][0]["input"])
train_pred = train_in.copy()
# ... apply your edits ...
print(f"Accuracy: {{accuracy(train_pred, task['train'][0]['output'])}}")
```

**5. Submit Predictions:**

Once verified, apply your approach to all test inputs and submit:
[[ ## code ## ]]
```python
SUBMIT(test=[pred_0, pred_1, ...])  # List of output grids
```

---

**Your Environment -- Interactive REPL:**

You work in a persistent Python REPL. Variables persist across iterations.
Pre-loaded: `np` (numpy), `task`, and helper functions.

This workflow is iterative by design: each code block executes, you inspect output,
and then decide what to do next. Do not try to solve everything in one step.

Pre-loaded variables:
- `task["train"]`: list of {{"input": grid, "output": grid}} dicts
- `task["test"]`: list of {{"input": grid}} dicts (no output - you must predict)

Available helpers:
- `format_grid(grid)` - format grid as string for printing
- `accuracy(pred, expected)` - 1.0 if identical, 0.0 otherwise
- `soft_accuracy(pred, expected)` - fraction of matching cells

**Grid formatting:** Always use `format_grid()` when printing grids.

**Execution discipline:**
- Explore first: inspect data before transforming it (print samples, shapes/types, lengths, unique values).
- Iterate in small steps: write short snippets, inspect output, then refine; reuse persistent state.
- Verify before submitting: if outputs look empty/zero/unexpected (patterns, shapes, objects, colors, etc.), reassess your hypothesis and code.
- Submit only after inspection: `SUBMIT(...)` ends the run immediately, so print/check first and submit in a later step if needed.

---

**Response Format:**

[[ ## reasoning ## ]]
Your step-by-step analysis. Keep concise but clear.

[[ ## code ## ]]
```python
# Python code to execute
```

---

**Example:**

[[ ## reasoning ## ]]
Let me examine the training examples to understand the transformation pattern.

[[ ## code ## ]]
```python
print("Example 1 Input:")
print(format_grid(task["train"][0]["input"]))
print("\\nExample 1 Output:")
print(format_grid(task["train"][0]["output"]))
```

---

**Rules:**
- Verify your approach on training examples before submitting.
- Apply the rule to test inputs, inspect outputs, and iterate until the result is stable.
- Call `SUBMIT(test=[...])` only when you are confident in the final test predictions.
""".strip()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ArcAgiReplEnv(vf.MultiTurnEnv):
    """Subprocess REPL-based environment for ARC-AGI.

    Uses SubprocessInterpreter for isolated Python execution with full
    numpy/scipy support. Based on RLM v3 architecture.

    The model responds with structured sections:
    - reasoning: Think step-by-step about the problem
    - code: Python code to execute in the REPL

    The environment stops when SUBMIT() is called in the code.
    """

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        timeout: float = 1200.0,
        max_output_chars: int = 50_000,
        **kwargs,
    ):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        state["submitted_answers"] = None
        state["iteration"] = 0

        # Build sandbox code with task data
        info = state["info"]
        task_json = json.dumps(
            {
                "train": info["train"],
                "test": [{"input": t["input"]} for t in info["test"]],
            }
        )
        sandbox_code = ARC_SANDBOX_CODE.format(task_json=task_json)

        # Create per-rollout subprocess interpreter
        interpreter = SubprocessInterpreter(
            sandbox_code=sandbox_code,
            timeout=self.timeout,
        )
        interpreter.start()
        state["_interpreter"] = interpreter

        return state

    def _format_output(self, output: str) -> str:
        """Format and truncate REPL output."""
        if not output:
            return "(no output - did you forget to print?)"
        if len(output) > self.max_output_chars:
            return output[: self.max_output_chars] + "\n... (truncated)"
        return output

    def _convert_submit_data(self, data: dict[str, Any]) -> dict[str, Any] | str:
        """Convert SUBMIT data to the expected format.

        Only 'test' predictions are required. 'train' is optional.
        Returns error string if data is invalid.
        """
        result = {}
        for key in ["test", "train"]:
            if key in data:
                grids = data[key]
                if not isinstance(grids, list):
                    return f"[Error] SUBMIT({key}=...) must be a list of grids, got {type(grids).__name__}"
                converted = []
                for i, grid in enumerate(grids):
                    if isinstance(grid, str):
                        return f"[Error] SUBMIT({key}[{i}]) is a string. Pass the numpy array or list directly, not format_grid() output."
                    if hasattr(grid, "tolist"):
                        grid = grid.tolist()
                    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
                        return f"[Error] SUBMIT({key}[{i}]) must be a 2D list/array, got {type(grid).__name__}"
                    try:
                        converted.append([[int(c) for c in row] for row in grid])
                    except (TypeError, ValueError) as e:
                        return f"[Error] SUBMIT({key}[{i}]) contains invalid cell values: {e}"
                result[key] = converted
        return result

    def _execute_code(self, code: str, state: vf.State) -> str:
        """Execute Python code in the subprocess REPL."""
        interpreter: SubprocessInterpreter = state["_interpreter"]

        # Strip code fences if present
        code = _strip_code_fences(code)
        if not code:
            return "[Error] No code provided."

        try:
            result = interpreter.execute(code)
        except CodeInterpreterError as e:
            error_msg = str(e)
            if "timed out" in error_msg.lower():
                return (
                    f"[Timeout Error] {error_msg}\n\n"
                    "Your code took too long to execute. This usually happens when:\n"
                    "- There's an infinite loop in your code\n"
                    "- The computation is too complex (e.g., very large grid operations)\n"
                    "- You're waiting for input that won't arrive\n\n"
                    "Please simplify your approach and try again with more efficient code."
                )
            return f"[Error] {e}"
        except SyntaxError as e:
            return f"[Error] {e}"

        # Handle FinalOutput (from SUBMIT() call)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], FinalOutput):
            final_output, captured = result
            submitted = self._convert_submit_data(final_output.output or {})
            output = captured if captured else ""

            # Check if conversion returned an error
            if isinstance(submitted, str):
                return output + "\n" + submitted if output else submitted

            if submitted and state.get("submitted_answers") is None:
                state["submitted_answers"] = submitted
            return output + "\nAnswers submitted successfully."

        if isinstance(result, FinalOutput):
            submitted = self._convert_submit_data(result.output or {})

            # Check if conversion returned an error
            if isinstance(submitted, str):
                return submitted

            if submitted and state.get("submitted_answers") is None:
                state["submitted_answers"] = submitted
            return "Answers submitted successfully."

        # Normal output
        output = result if isinstance(result, str) else str(result)
        return self._format_output(output)

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        """Parse model response, execute code, and provide feedback.

        Always returns a non-empty list of messages so the model gets feedback,
        even when parsing or execution fails unexpectedly.
        """
        state["iteration"] = state.get("iteration", 0) + 1

        try:
            return self._env_response_inner(messages, state)
        except Exception as e:
            return [
                {
                    "role": "user",
                    "content": (
                        f"[Environment Error] An unexpected error occurred while processing your response: {e}\n\n"
                        "Please try again with a new code block."
                    ),
                }
            ]

    def _env_response_inner(
        self,
        messages: vf.Messages,
        state: vf.State,
    ) -> vf.Messages:
        """Inner implementation of env_response."""
        # The last message is always the assistant's response
        # (get_prompt_messages concatenates [prev_prompt, prev_completion])
        assistant_msg = messages[-1]

        # Extract text content from the assistant message
        content = assistant_msg.content
        if isinstance(content, list):
            # Handle list of ContentPart objects (TextContentPart, etc.)
            content = " ".join(getattr(p, "text", str(p)) for p in content)
        if not content:
            return [
                {
                    "role": "user",
                    "content": "No assistant response found. Please provide your reasoning and code.",
                }
            ]

        # Parse structured response
        sections = parse_response(content)

        # Execute code if provided
        code = sections["code"]
        if not code:
            return [
                {
                    "role": "user",
                    "content": "No code provided in the [[ ## code ## ]] section. Please provide Python code to execute.",
                }
            ]

        # Execute the code
        output = self._execute_code(code, state)

        # Check if SUBMIT was called during execution
        if state.get("submitted_answers") is not None:
            state["final_env_response"] = [{"role": "user", "content": f"REPL Output:\n{output}"}]
            return []

        # Return output as feedback
        return [{"role": "user", "content": f"REPL Output:\n{output}"}]

    @vf.cleanup
    async def cleanup_interpreter(self, state: vf.State) -> None:
        """Shutdown the subprocess interpreter after rollout completes."""
        interpreter = state.get("_interpreter")
        if interpreter is not None:
            interpreter.shutdown()
