"""AIME math competition environment for RLVR."""

from __future__ import annotations

import json
from typing import Any, cast

import verifiers as vf
from datasets import Dataset, concatenate_datasets, load_dataset


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# Maps dataset name -> (HF repo, config, default split, preprocess fn)
_DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "aimo-validation-aime": {
        "repo": "AI-MO/aimo-validation-aime",
        "config": None,
        "default_split": "train",
        "fields": {"problem": "problem", "answer": "answer", "solution": "solution", "id": "id"},
    },
    "aime2024": {
        "repo": "HuggingFaceH4/aime_2024",
        "config": None,
        "default_split": "train",
        "fields": {"problem": "problem", "answer": "answer", "solution": None, "id": None},
    },
    "aime2025": {
        "repo": "opencompass/AIME2025",
        "config": ["AIME2025-I", "AIME2025-II"],
        "default_split": "test",
        "fields": {"problem": "question", "answer": "answer", "solution": None, "id": None},
    },
}


def _strip_non_numeric(text: str) -> str:
    return "".join(c for c in text if c.isdigit() or c == ".")


def _load_hf_dataset(dataset_name: str, split: str | None) -> Dataset:
    """Load a raw HF dataset by name."""
    info = _DATASET_REGISTRY.get(dataset_name)
    if info is None:
        # Treat as a raw HF repo path
        _split = split or "train"
        return cast(Dataset, load_dataset(dataset_name, split=_split))

    _split = split or info["default_split"]

    configs = info["config"]
    if isinstance(configs, list):
        parts = [cast(Dataset, load_dataset(info["repo"], cfg)[_split]) for cfg in configs]
        return concatenate_datasets(parts)

    return cast(Dataset, load_dataset(info["repo"], configs, split=_split))


def _prepare_dataset(dataset_name: str, split: str | None) -> Dataset:
    """Load and normalize an AIME dataset into (question, answer, info) rows."""
    raw = _load_hf_dataset(dataset_name, split)
    info = _DATASET_REGISTRY.get(dataset_name, {})
    fields = info.get("fields", {"problem": "problem", "answer": "answer", "solution": None, "id": None})

    rows: list[dict] = []
    for i, row in enumerate(raw):
        problem = row[fields["problem"]]
        answer_raw = row[fields["answer"]]
        # Normalize: AIME answers are always integers 000-999
        answer = _strip_non_numeric(str(answer_raw))
        try:
            answer = str(int(float(answer)))
        except (ValueError, TypeError):
            answer = str(answer_raw)

        sol_key = fields.get("solution")
        solution = row[sol_key] if sol_key and sol_key in row else ""
        id_key = fields.get("id")
        problem_id = str(row[id_key]) if id_key and id_key in row else f"aime_{i}"

        # Build teacher context for OPD
        teacher_context = f"The correct answer is: {answer}"
        if solution:
            teacher_context += f"\n\nReference solution:\n{solution}"

        row_info = {
            "problem_id": problem_id,
            "teacher_context": teacher_context,
        }
        if solution:
            row_info["solution"] = solution

        rows.append({
            "question": problem,
            "answer": answer,
            "info": json.dumps(row_info),
        })

    ds = Dataset.from_list(rows)
    # Sort by question text for deterministic ordering across different loading paths
    ds = ds.sort("question")
    return ds


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "Solve the following math problem. "
    "Show your work step by step and put your final answer in \\boxed{}."
)


def load_environment(
    dataset_name: str = "aimo-validation-aime",
    split: str | None = None,
    eval_dataset: str | None = None,
    eval_split: str | None = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    train_start_index: int = 0,
    eval_start_index: int = 0,
    system_prompt: str = SYSTEM_PROMPT,
    **kwargs,
) -> vf.Environment:
    """Load an AIME environment.

    Args:
        dataset_name: Dataset to load. Supported: "aimo-validation-aime" (90 AIME problems),
            "aime2024" (30 problems), "aime2025" (30 problems), or any HF dataset path
            with "problem"/"answer" columns.
        split: HF split. Defaults to dataset-specific default.
        eval_dataset: Optional separate dataset for evaluation.
        eval_split: Evaluation split.
        num_train_examples: Number of train examples to use. -1 for all.
        num_eval_examples: Number of eval examples to use. -1 for all.
        train_start_index: Start index for train set slicing (after sort).
        eval_start_index: Start index for eval set slicing (after sort).
        system_prompt: System prompt for the model.
        **kwargs: Additional arguments passed to SingleTurnEnv.
    """
    train_ds = _prepare_dataset(dataset_name, split)
    if train_start_index > 0 or num_train_examples > 0:
        start = train_start_index
        end = len(train_ds) if num_train_examples < 0 else min(start + num_train_examples, len(train_ds))
        train_ds = train_ds.select(range(start, end))

    eval_ds = None
    if eval_dataset is not None:
        eval_ds = _prepare_dataset(eval_dataset, eval_split)
        if eval_start_index > 0 or num_eval_examples > 0:
            start = eval_start_index
            end = len(eval_ds) if num_eval_examples < 0 else min(start + num_eval_examples, len(eval_ds))
            eval_ds = eval_ds.select(range(start, end))

    rubric = vf.MathRubric()

    env = vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=system_prompt,
        rubric=rubric,
        **kwargs,
    )
    return env
