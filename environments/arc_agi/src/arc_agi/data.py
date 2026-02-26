"""Data loading and formatting for ARC-AGI environments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

from datasets import Dataset

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Grid = list[list[int]]


class ArcTask(TypedDict):
    task_id: str
    train: list[dict]  # [{"input": Grid, "output": Grid}, ...]
    test: list[dict]  # [{"input": Grid, "output": Grid}, ...]


# ---------------------------------------------------------------------------
# Grid formatting
# ---------------------------------------------------------------------------


def format_grid(grid: Grid) -> str:
    """Convert a grid to a string (comma-separated digits)."""
    return "\n".join(",".join(str(c) for c in row) for row in grid)


def format_input_output_pair(input: Grid, output: Grid) -> str:
    return f"Input:\n{format_grid(input)}\nOutput:\n{format_grid(output)}"


def format_pairs(pairs: list[dict], split: str = "train") -> str:
    parts: list[str] = []
    title = "Example" if split == "train" else "Challenge"
    for i, pair in enumerate(pairs, 1):
        parts.append(f"{title} #{i}")
        parts.append(format_input_output_pair(pair["input"], pair["output"]))
    return "\n".join(parts)


def format_task_question(train_pairs: list[dict], test_inputs: list[Grid]) -> str:
    """Format an ARC task as a question string with grids."""
    parts: list[str] = []
    for i, pair in enumerate(train_pairs, 1):
        parts.append(f"Example #{i}")
        parts.append("Input:")
        parts.append(format_grid(pair["input"]))
        parts.append("Output:")
        parts.append(format_grid(pair["output"]))
        parts.append("")

    for i, inp in enumerate(test_inputs, 1):
        parts.append(f"Challenge #{i}")
        parts.append("Input:")
        parts.append(format_grid(inp))
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Split parsing
# ---------------------------------------------------------------------------


def parse_split(split_str: str) -> tuple[str, int | None, int | None, list[str] | None]:
    """Parse split string with optional range or task ID notation.

    Supports formats:
    - "training" -> ("training", None, None, None)
    - "training[:100]" -> ("training", None, 100, None)
    - "training[50:]" -> ("training", 50, None, None)
    - "training[50:100]" -> ("training", 50, 100, None)
    - "training[[abc123,def456]]" -> ("training", None, None, ["abc123", "def456"])

    Args:
        split_str: Split string with optional range or task IDs.

    Returns:
        Tuple of (split_name, start, end, task_ids).
    """
    # Try task ID pattern first: split[[id1,id2,id3]]
    match = re.match(r"^(\w+)\[\[([^\]]+)\]\]$", split_str)
    if match:
        split_name = match.group(1)
        task_ids = [p.strip() for p in match.group(2).split(",")]
        return split_name, None, None, task_ids

    # Try range pattern: split[start:end]
    match = re.match(r"^(\w+)(?:\[(\d*):(\d*)\])?$", split_str)
    if not match:
        return split_str, None, None, None

    split_name = match.group(1)
    start_str = match.group(2)
    end_str = match.group(3)

    start = int(start_str) if start_str else None
    end = int(end_str) if end_str else None

    return split_name, start, end, None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _resolve_data_folder(data_folder: str) -> Path:
    """Resolve a dataset folder name under environments/arc_agi/data."""
    if "/" in data_folder or "\\" in data_folder:
        raise ValueError(f"Expected data folder name only (e.g. 'arc-dummy'), got path-like value: {data_folder!r}")

    base = DATA_ROOT / data_folder
    if not base.is_dir():
        raise FileNotFoundError(f"Data folder not found: {base}")
    return base


def load_arc_tasks(data_folder: str, split: str) -> tuple[dict, dict]:
    """Read challenges + solutions JSON files for the given split."""
    base = _resolve_data_folder(data_folder)
    challenges_path = base / f"arc-agi_{split}_challenges.json"
    solutions_path = base / f"arc-agi_{split}_solutions.json"

    with open(challenges_path) as f:
        challenges = json.load(f)
    with open(solutions_path) as f:
        solutions = json.load(f)
    return challenges, solutions


def prepare_dataset(data_folder: str, split: str) -> Dataset:
    """Build a HuggingFace Dataset with one row per ARC task.

    Args:
        data_folder: Name of a folder under environments/arc_agi/data.
        split: Split string with optional range or task IDs.
            Examples:
            - "training" - all tasks
            - "training[:100]" - first 100 tasks
            - "training[50:100]" - tasks 50-100
            - "training[[abc123,def456]]" - specific task IDs

    Returns:
        HuggingFace Dataset with columns: question, answer, info.
    """
    # Parse split string for range/task filtering
    split_name, start, end, task_ids = parse_split(split)

    challenges, solutions = load_arc_tasks(data_folder, split_name)

    # Filter by task IDs if specified
    if task_ids:
        challenges = {k: v for k, v in challenges.items() if k in task_ids}
        solutions = {k: v for k, v in solutions.items() if k in task_ids}

    rows: list[dict] = []
    for task_id, task in challenges.items():
        train_pairs = task["train"]
        test_inputs = [t["input"] for t in task["test"]]
        test_outputs = solutions[task_id]
        test_pairs = [{"input": inp, "output": output} for inp, output in zip(test_inputs, test_outputs)]

        question = format_task_question(train_pairs, test_inputs)
        answer = json.dumps(test_outputs)

        info = ArcTask(
            task_id=task_id,
            train=train_pairs,
            test=test_pairs,
        )

        rows.append(
            {
                "task": task_id,
                "question": question,
                "answer": answer,
                "info": json.dumps(info),
            }
        )

    # Apply range slicing if specified
    if start is not None or end is not None:
        rows = rows[start:end]

    return Dataset.from_list(rows)
