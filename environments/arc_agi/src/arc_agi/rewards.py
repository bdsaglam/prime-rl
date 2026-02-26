"""Reward functions for ARC-AGI environments."""

from __future__ import annotations

import numpy as np
import verifiers as vf

from .data import Grid

# ---------------------------------------------------------------------------
# Submission parsing
# ---------------------------------------------------------------------------


def parse_submission(state, key: str = "test") -> list[Grid] | None:
    """Extract submitted grids from state.

    Args:
        state: The environment state.
        key: "test" or "train" to extract the corresponding submission.
    """
    submitted = state.get("submitted_answers")
    if submitted is None:
        return None
    if isinstance(submitted, dict):
        return submitted.get(key)
    # Legacy format: list of grids (test only)
    return submitted if key == "test" else None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_exact_match(predicted: list[Grid] | None, expected: list[Grid]) -> float:
    """1.0 only if ALL predicted grids exactly match ALL expected grids."""
    if predicted is None:
        return 0.0
    if len(predicted) != len(expected):
        return 0.0
    return 1.0 if all(g == e for g, e in zip(predicted, expected)) else 0.0


def _compute_cell_accuracy(predicted: list[Grid] | None, expected: list[Grid]) -> float:
    """Average cell-level accuracy."""
    if predicted is None:
        return 0.0
    if len(predicted) != len(expected):
        return 0.0

    total_cells = 0
    correct_cells = 0
    for pred, exp in zip(predicted, expected):
        pred_arr = np.array(pred)
        exp_arr = np.array(exp)
        if pred_arr.shape != exp_arr.shape:
            total_cells += exp_arr.size
            continue
        total_cells += exp_arr.size
        correct_cells += int(np.sum(pred_arr == exp_arr))

    return correct_cells / total_cells if total_cells > 0 else 0.0


def _compute_shape_match(predicted: list[Grid] | None, expected: list[Grid]) -> float:
    """Fraction of cases where predicted dimensions match expected."""
    if predicted is None:
        return 0.0
    if len(predicted) != len(expected):
        return 0.0

    matches = 0
    for pred, exp in zip(predicted, expected):
        pred_arr = np.array(pred)
        exp_arr = np.array(exp)
        if pred_arr.shape == exp_arr.shape:
            matches += 1
    return matches / len(expected)


def _compute_format(predicted: list[Grid] | None, expected: list[Grid]) -> float:
    """1.0 if valid submission with correct number of grids."""
    if predicted is None:
        return 0.0
    return 1.0 if len(predicted) == len(expected) else 0.0


# ---------------------------------------------------------------------------
# Test reward functions
# ---------------------------------------------------------------------------


def exact_match_reward(state, info, **kwargs) -> float:
    """1.0 only if ALL predicted test grids exactly match ALL expected test grids."""
    expected = [p["output"] for p in info["test"]]
    return _compute_exact_match(parse_submission(state, "test"), expected)


def cell_accuracy_reward(state, info, **kwargs) -> float:
    """Average cell-level accuracy across all test cases."""
    expected = [p["output"] for p in info["test"]]
    return _compute_cell_accuracy(parse_submission(state, "test"), expected)


def shape_match_reward(state, info, **kwargs) -> float:
    """Fraction of test cases where predicted dimensions match expected."""
    expected = [p["output"] for p in info["test"]]
    return _compute_shape_match(parse_submission(state, "test"), expected)


def format_reward(state, info, **kwargs) -> float:
    """1.0 if valid submission with correct number of test grids."""
    expected = [p["output"] for p in info["test"]]
    return _compute_format(parse_submission(state, "test"), expected)


# ---------------------------------------------------------------------------
# Train reward functions
# ---------------------------------------------------------------------------


def train_exact_match_reward(state, info, **kwargs) -> float:
    """1.0 only if ALL predicted train grids exactly match ALL expected train outputs."""
    expected = [p["output"] for p in info["train"]]
    return _compute_exact_match(parse_submission(state, "train"), expected)


def train_cell_accuracy_reward(state, info, **kwargs) -> float:
    """Average cell-level accuracy across all training cases."""
    expected = [p["output"] for p in info["train"]]
    return _compute_cell_accuracy(parse_submission(state, "train"), expected)


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

_REWARD_MODE_WEIGHTS: dict[str, list[float]] = {
    # [exact_match, cell_accuracy, shape_match, format]
    "binary": [1.0, 0.0, 0.0, 0.0],
    "partial": [0.0, 1.0, 0.0, 0.0],
    "combined": [0.5, 0.5, 0.0, 0.0],
    "balanced": [2.0, 0.5, 0.3, 0.1],
}


def ArcAgiRubric(parser=None, reward_mode: str = "binary", **kwargs):
    """Create an ARC-AGI rubric with configurable reward weighting."""
    funcs = [exact_match_reward, cell_accuracy_reward, shape_match_reward, format_reward]
    weights = _REWARD_MODE_WEIGHTS.get(reward_mode, _REWARD_MODE_WEIGHTS["binary"])
    # Normalize weights so rewards are in [0, 1]
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    return vf.Rubric(funcs=funcs, weights=weights, parser=parser, **kwargs)
