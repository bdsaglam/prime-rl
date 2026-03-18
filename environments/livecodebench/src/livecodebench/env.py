"""LiveCodeBench competitive programming environment for RLVR."""

from __future__ import annotations

import base64
import json
import logging
import pickle
import zlib
from datetime import datetime
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

from .sandbox import INCORRECT_FORMAT, execute_code, extract_code

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 6.0

SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and need to generate a correct Python solution.\n\n"
    "Read the problem carefully. Think step by step. "
    "Put your final solution within a code block:\n"
    "```python\n# your code here\n```"
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _parse_signature(starter_code: str) -> str:
    """Extract function signature from starter code."""
    return "def " + starter_code.split("def ")[1].split("Input\n")[0].strip()


def _decode_private_tests(encoded_data: str, fn_name: str) -> str:
    """Decode private test cases from base64+zlib+pickle encoding.

    Returns a JSON string with keys: inputs, outputs, testtype, fn_name, time_limit.
    """
    decoded = base64.b64decode(encoded_data)
    decompressed = zlib.decompress(decoded)
    original = pickle.loads(decompressed)
    tests = json.loads(original)
    return json.dumps(
        {
            "inputs": [t["input"] for t in tests],
            "outputs": [t["output"] for t in tests],
            "testtype": tests[0]["testtype"],
            "fn_name": fn_name,
            "time_limit": DEFAULT_TIMEOUT,
        },
        ensure_ascii=False,
    )


def _prepare_dataset(
    dataset_name: str,
    split: str | None,
    difficulty: str | None,
    cutoff_date: datetime,
    is_eval: bool,
) -> Dataset:
    """Load and normalize a LiveCodeBench dataset into (question, answer, info) rows."""
    # Always load from "test" split on HF, then filter by date
    raw = load_dataset(dataset_name, split="test", revision="refs/pr/6")

    # Filter by contest date for train/eval split
    if is_eval:
        raw = raw.filter(lambda ex: ex["contest_date"] >= cutoff_date)
    else:
        raw = raw.filter(lambda ex: ex["contest_date"] < cutoff_date)

    # Optionally filter by difficulty (uses the 'difficulty' column directly)
    if difficulty is not None:
        difficulty_lower = difficulty.lower()
        raw = raw.filter(
            lambda ex: (ex.get("difficulty", "") or "").lower() == difficulty_lower
        )

    rows: list[dict] = []
    for i, row in enumerate(raw):
        problem = row["question_content"]  # includes public test cases

        # Add function signature hint if starter code is provided
        starter_code = row.get("starter_code", "")
        if starter_code and starter_code.strip():
            try:
                sig = _parse_signature(starter_code)
                problem += f"\n\nYour solution should have the following signature: ```python\n{sig}\n```"
            except Exception:
                pass

        # Get function name from metadata
        metadata = row.get("metadata", "")
        fn_name = ""
        if metadata and metadata.strip():
            try:
                meta_dict = json.loads(metadata)
                fn_name = meta_dict.get("func_name", "")
            except Exception:
                pass

        # Decode private test cases
        try:
            tests_json = _decode_private_tests(row["private_test_cases"], fn_name)
        except Exception as e:
            logger.warning(f"Failed to decode tests for problem {i}: {e}")
            continue

        problem_id = row.get("question_id", f"lcb_{i}")

        info_dict = {
            "problem_id": str(problem_id),
            "tests": tests_json,
            "fn_name": fn_name,
            "contest_date": str(row.get("contest_date", "")),
        }

        rows.append(
            {
                "question": problem,
                "answer": "N/A",  # No string-match answer; we use code execution
                "info": json.dumps(info_dict),
            }
        )

    ds = Dataset.from_list(rows)
    # Sort by question text for deterministic ordering
    ds = ds.sort("question")
    return ds


# ---------------------------------------------------------------------------
# Custom Rubric
# ---------------------------------------------------------------------------


def _score_code(completion: Any, state: dict, **kwargs) -> float:
    """Reward function: extract code, execute against tests, return reward.

    Uses `state` (not `info`) so we can mutate info in place and have it
    propagate back to the rollout for prepare_teacher_context.
    """
    info = state.get("info", {})
    if isinstance(info, str):
        info = json.loads(info)
        state["info"] = info  # replace string with parsed dict

    # Extract completion text
    completion = state.get("completion", completion)
    if isinstance(completion, list):
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict):
                            parts.append(p.get("text", str(p)))
                        elif isinstance(p, str):
                            parts.append(p)
        completion_text = "\n".join(parts)
    else:
        completion_text = str(completion)

    # Extract code from response
    code = extract_code(completion_text)
    if code is None:
        info["feedback"] = "No code block found. Put your code inside a ```python ... ``` block."
        info["execution_result"] = {"passed": False, "num_passed": 0, "num_total": 0}
        return 0.0

    # Get test cases
    tests_json = info.get("tests", "")
    if not tests_json:
        info["feedback"] = "No test cases available."
        info["execution_result"] = {"passed": False, "num_passed": 0, "num_total": 0}
        return 0.0

    # Execute code against tests
    result = execute_code(code, tests_json)

    # Store feedback for teacher context (mutates info dict in state)
    info["feedback"] = result["feedback"]
    info["execution_result"] = {
        "passed": result["passed"],
        "num_passed": result["num_passed"],
        "num_total": result["num_total"],
    }

    return 1.0 if result["passed"] else 0.0


class CodeRubric(vf.Rubric):
    """Rubric that executes code against test cases for reward scoring."""

    def __init__(self):
        super().__init__()
        self.add_reward_func(_score_code)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


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
    version: str = "v5",
    difficulty: str | None = None,
    train_cutoff: str = "2025-02-01",
    **kwargs,
) -> vf.Environment:
    """Load a LiveCodeBench environment.

    Args:
        dataset_name: HuggingFace dataset path.
        split: Ignored (we always load "test" and filter by date).
        eval_dataset: Optional separate dataset for evaluation.
        eval_split: Evaluation split.
        num_train_examples: Number of train examples to use. -1 for all.
        num_eval_examples: Number of eval examples to use. -1 for all.
        train_start_index: Start index for train set slicing (after sort).
        eval_start_index: Start index for eval set slicing (after sort).
        system_prompt: System prompt for the model.
        version: Dataset version tag (e.g. "v5", "v6").
        difficulty: Filter by difficulty ("easy", "medium", "hard", or None for all).
        train_cutoff: Date string (YYYY-MM-DD) for train/eval split.
        **kwargs: Additional arguments passed to SingleTurnEnv.
    """
    cutoff = datetime.strptime(train_cutoff, "%Y-%m-%d")

    train_ds = _prepare_dataset(
        dataset_name, split, difficulty, cutoff, is_eval=False,
    )
    if train_start_index > 0 or num_train_examples > 0:
        start = train_start_index
        end = len(train_ds) if num_train_examples < 0 else min(start + num_train_examples, len(train_ds))
        train_ds = train_ds.select(range(start, end))

    eval_ds = None
    if eval_dataset is not None:
        eval_cutoff = cutoff
        eval_ds = _prepare_dataset(
            eval_dataset, eval_split, difficulty, eval_cutoff, is_eval=True,
        )
        if eval_start_index > 0 or num_eval_examples > 0:
            start = eval_start_index
            end = len(eval_ds) if num_eval_examples < 0 else min(start + num_eval_examples, len(eval_ds))
            eval_ds = eval_ds.select(range(start, end))

    rubric = CodeRubric()

    env = vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=system_prompt,
        rubric=rubric,
        **kwargs,
    )
    return env
