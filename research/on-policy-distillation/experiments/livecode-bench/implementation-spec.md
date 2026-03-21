# LiveCodeBench Environment Implementation Spec

## Goal
Create a `livecodebench` environment for prime-rl that enables RL training on competitive programming problems with rich feedback for OPD.

## Directory Structure
```
environments/livecodebench/
  pyproject.toml
  src/livecodebench/
    __init__.py          # exports load_environment, prepare_teacher_context
    env.py               # load_environment(), dataset loading, system prompt
    sandbox.py           # code execution with restricted builtins + multiprocessing
    teacher_context.py   # prepare_teacher_context() for OPD (SDPO-style PI)
```

## Reference Files (READ THESE CAREFULLY)
1. **AIME env** (pattern to follow): `environments/aime/src/aime/env.py`
2. **AIME SDPO teacher context** (PI pattern): `environments/aime_sdpo/src/aime_sdpo/teacher_context.py`
3. **SDPO data loading**: `research/on-policy-distillation/papers/sdpo/repo/data/utils/livecodebench.py`
4. **SDPO code execution** (port this): `research/on-policy-distillation/papers/sdpo/repo/verl/utils/reward_score/feedback/code.py`
5. **SDPO prompt template**: `research/on-policy-distillation/papers/sdpo/repo/data/format/prompts.py`

## Component Details

### 1. `env.py` — Dataset Loading + Environment

Follow AIME's `load_environment()` pattern exactly. The function signature must be:
```python
def load_environment(
    dataset_name: str = "livecodebench/code_generation_lite",
    split: str | None = None,
    eval_dataset: str | None = None,
    eval_split: str | None = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    train_start_index: int = 0,
    eval_start_index: int = 0,
    system_prompt: str = SYSTEM_PROMPT,
    version: str = "v5",          # v5 or v6
    difficulty: str | None = None, # easy, medium, hard, or None for all
    train_cutoff: str = "2025-02-01",  # date string for train/eval split
    **kwargs,
) -> vf.Environment:
```

**Dataset loading**:
- Source: `livecodebench/code_generation_lite` from HuggingFace, split="test", revision="refs/pr/6" (see SDPO's code)
- Filter by `contest_date` for train/eval split (default: train = before 2025-02-01, eval = after)
- Optionally filter by difficulty level from metadata
- Each row needs: `question` (problem text + starter code signature if any), `answer` (not used for string matching — use "N/A"), `info` (JSON with problem_id, tests encoded, test metadata)
- The `info` dict must contain `tests` (the private test cases JSON) and `problem_id` for grouping

**Private test cases decoding** (from SDPO):
```python
import base64, zlib, pickle, json
decoded = base64.b64decode(encoded_data)
decompressed = zlib.decompress(decoded)
original = pickle.loads(decompressed)
tests = json.loads(original)
# tests is list of {"input": ..., "output": ..., "testtype": "functional"|"stdin"}
```

**System prompt** (from SDPO):
```
You are an expert Python programmer. You will be given a question (problem specification) and need to generate a correct Python solution.

Read the problem carefully. Think step by step. Put your final solution within a code block:
```python
# your code here
```
```

**Rubric**: Create a custom `CodeRubric(vf.Rubric)` that:
1. Extracts code from the model's response (regex: last ```python...``` block)
2. Calls `sandbox.execute_code()` to run against test cases
3. Returns reward (1.0 if all pass, 0.0 otherwise) AND stores feedback in the rollout
4. Feedback format: structured string with pass/fail per test, error messages for failures

### 2. `sandbox.py` — Code Execution

Port SDPO's `code.py` but simplify. Key functions needed:

```python
def execute_code(code: str, tests_json: str, timeout: float = 6.0) -> dict:
    """Execute code against test cases in a sandboxed subprocess.

    Returns:
        {
            "passed": bool,           # all tests passed?
            "num_passed": int,
            "num_total": int,
            "feedback": str,          # structured feedback string
            "results": list[dict],    # per-test results
        }
    """
```

**From SDPO's code.py, port these key pieces**:
- `_build_restricted_builtins()` — restricted Python builtins (no file I/O, no subprocess)
- `set_memory_limits()` — RLIMIT_AS memory cap (1GB)
- `unsafe_execute()` — the core execution function using multiprocessing.Process
- `format_test_feedback()` — LeetCode-style feedback formatting
- Support both `functional` (call function with args) and `stdin` (pipe input to stdin) test types

**Simplifications vs SDPO**:
- Don't need the full `Solution.py` / `Tests.py` file-based approach
- Don't need debug_print infrastructure
- Focus on: extract code → run in subprocess with restricted builtins + memory/time limits → collect results

**Security**: Use multiprocessing.Process (not threading) with restricted builtins and resource limits. This is sufficient for training. NOT production-grade but good enough.

### 3. `teacher_context.py` — OPD Teacher Context

Follow `aime_sdpo/teacher_context.py` pattern exactly. Must export:
```python
def prepare_teacher_context(rollouts: list[dict]) -> None:
    """SDPO-style teacher context: test feedback + correct sibling."""
```

**PI components for incorrect rollouts**:
1. Test execution feedback (from sandbox results stored in info)
2. Correct sibling solution from batch (if available)
3. NO student's own attempt (per SDPO findings)

**PI for correct rollouts**: Minimal — "Solution passed all tests."

### 4. `pyproject.toml`

```toml
[project]
name = "livecodebench"
description = "LiveCodeBench competitive programming environment"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "verifiers>=0.1.9",
    "datasets",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 5. `__init__.py`

```python
from .env import load_environment
from .teacher_context import prepare_teacher_context

__all__ = ["load_environment", "prepare_teacher_context"]
```

## Key Constraints

1. **Follow existing patterns**: AIME env for `load_environment`, AIME SDPO for `prepare_teacher_context`
2. **The verifiers contract**: `load_environment()` returns `vf.Environment`. Dataset must have `question`, `answer`, `info` columns.
3. **Rubric must store feedback**: After scoring, the feedback string must be accessible for `prepare_teacher_context` later. Store it in rollout `info["feedback"]`.
4. **Sync `prepare_teacher_context`**: Must be `def prepare_teacher_context(rollouts: list[dict]) -> None` (sync, single arg, mutates in place)
5. **Code extraction**: Use regex to find last ```python...``` block. If no code block found, return 0.0 reward with "No code block found" feedback.
6. **Test timeout**: 6 seconds per test case (from SDPO). Total timeout per problem = 6 * num_tests, capped at 60s.
7. **No Docker dependency**: Use multiprocessing + restricted builtins, not Docker containers.

## Verification

After implementation, verify:
```bash
# Install
uv pip install -e environments/livecodebench/

# Test import
python -c "from livecodebench import load_environment, prepare_teacher_context; print('OK')"

# Test dataset loading (just load, don't run training)
python -c "
from livecodebench.env import load_environment
env = load_environment(version='v5', num_train_examples=5)
print(f'Loaded {len(env.dataset)} train examples')
print(f'First example question (truncated): {env.dataset[0][\"question\"][:200]}')
"

# Test code execution
python -c "
from livecodebench.sandbox import execute_code
import json
# Simple test: function that adds two numbers
tests = json.dumps({
    'inputs': [['1 2'], ['3 4']],
    'outputs': [['3'], ['7']],
    'testtype': 'stdin',
    'fn_name': '',
    'time_limit': 6,
})
code = 'a, b = map(int, input().split())\nprint(a + b)'
result = execute_code(code, tests)
print(f'Passed: {result[\"passed\"]}, {result[\"num_passed\"]}/{result[\"num_total\"]}')
print(f'Feedback: {result[\"feedback\"][:200]}')
"
```

Run these verification steps and fix any issues before considering the task complete.
