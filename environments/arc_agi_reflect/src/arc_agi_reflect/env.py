"""Main entry point for ARC-AGI Reflect environment."""

from __future__ import annotations

import verifiers as vf
from datasets import Dataset, concatenate_datasets

from arc_agi.data import prepare_dataset
from arc_agi.rewards import ArcAgiRubric

from .envs.repl import ArcAgiReflectEnv


def _load_dataset(dataset: str | list[str], split: str) -> Dataset:
    """Load one or more datasets and concatenate them."""
    data_folders = [dataset] if isinstance(dataset, str) else list(dataset)
    dataset_list = [prepare_dataset(folder, split) for folder in data_folders]
    return concatenate_datasets(dataset_list) if len(dataset_list) > 1 else dataset_list[0]


def load_environment(
    dataset_name: str | list[str] = "arc-prize-2025",
    split: str = "training",
    eval_dataset: str | list[str] | None = None,
    eval_split: str = "evaluation",
    reward_mode: str = "balanced",
    max_turns: int = 8,
    reflection_style: str = "structured",
    **kwargs,
) -> vf.Environment:
    """Load an ARC-AGI Reflect environment.

    Same as arc_agi but adds a structured reflection step after SUBMIT.
    The student is shown their result (correct/incorrect + expected output)
    and asked to reflect in a structured format for OPD signal.

    Args:
        dataset_name: ARC data folder name(s) from environments/arc_agi/data.
        split: Data split (training or evaluation).
        eval_dataset: Separate ARC data folder name(s) for evaluation.
        eval_split: Evaluation data split.
        reward_mode: Reward weighting - "binary", "partial", "combined", or "balanced".
        max_turns: Maximum interaction turns (including the reflection turn).
        reflection_style: Reflection prompt style ("structured" or "open").
        **kwargs: Additional arguments passed to the environment.

    Returns:
        Configured environment instance.
    """
    legacy_dataset = kwargs.pop("dataset", None)
    if legacy_dataset is not None:
        dataset_name = legacy_dataset

    train_ds = _load_dataset(dataset_name, split)

    eval_ds = None
    if eval_dataset is not None:
        eval_ds = _load_dataset(eval_dataset, eval_split)

    parser = vf.Parser()
    rubric = ArcAgiRubric(parser=parser, reward_mode=reward_mode)

    env = ArcAgiReflectEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        reflection_style=reflection_style,
        **kwargs,
    )

    return env
