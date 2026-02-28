"""Main entry point for ARC-AGI environments."""

from __future__ import annotations

import verifiers as vf
from datasets import Dataset, concatenate_datasets

from .data import prepare_dataset
from .rewards import ArcAgiRubric


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
    max_turns: int = 10,
    env_type: str = "repl",
    **kwargs,
) -> vf.Environment:
    """Load an ARC-AGI environment.

    Args:
        dataset_name: ARC data folder name, or list of folder names to concatenate
            from environments/arc_agi/data (e.g. ["arc-prize-2024", "arc-prize-2025"]).
        split: Data split (training or evaluation).
        eval_dataset: Separate ARC data folder name(s) for evaluation (optional).
        eval_split: Evaluation data split.
        reward_mode: Reward weighting - "binary", "partial", "combined", or "balanced".
        max_turns: Maximum interaction turns.
        env_type: Environment type (only "repl" is supported).
        **kwargs: Additional arguments passed to the environment.

    Returns:
        Configured environment instance.
    """
    # Backward compatibility for old configs/commands that still pass `dataset=...`.
    legacy_dataset = kwargs.pop("dataset", None)
    if legacy_dataset is not None:
        dataset_name = legacy_dataset

    train_ds = _load_dataset(dataset_name, split)

    eval_ds = None
    if eval_dataset is not None:
        eval_ds = _load_dataset(eval_dataset, eval_split)

    parser = vf.Parser()
    rubric = ArcAgiRubric(parser=parser, reward_mode=reward_mode)

    if env_type != "repl":
        raise ValueError(f"Unknown env_type: {env_type}. Must be 'repl'.")

    from .envs.repl import ArcAgiReplEnv

    env = ArcAgiReplEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        **kwargs,
    )

    return env
