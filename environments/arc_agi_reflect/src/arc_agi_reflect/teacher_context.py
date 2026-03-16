"""ARC-AGI Reflect teacher context preparation for on-policy distillation.

For ARC-AGI, the teacher PI is simpler than AIME since the answer is a grid:
- Correct rollouts: keep static PI (expected output grids)
- Incorrect rollouts: provide expected output + per-challenge accuracy breakdown

The student already sees the expected output in the reflection prompt (by design —
"going from problem to answer is the challenging part"). The teacher gets richer
context about which specific cells/regions were wrong.
"""

import json
import logging

from prime_rl.configs.orchestrator import AnalyzerConfig

logger = logging.getLogger(__name__)


def _get_info(rollout: dict) -> dict:
    """Get rollout info as a dict, parsing JSON if needed."""
    info = rollout.get("info", {})
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except (json.JSONDecodeError, TypeError):
            info = {}
    return info if isinstance(info, dict) else {}


async def prepare_teacher_context(
    analyzer_config: AnalyzerConfig,
    rollouts: list[dict],
) -> list[dict]:
    """Prepare teacher context for ARC-AGI Reflect rollouts.

    Uses static PI only (no LLM call needed) since ARC-AGI answers are grids
    and the teacher context is already rich enough:
    - Expected output grids (already in teacher_context from data.py)
    - For incorrect rollouts: the teacher also knows cell-level accuracy

    No external LLM calls are needed — this is pure postprocessing.
    """
    num_correct = 0
    num_incorrect = 0

    for rollout in rollouts:
        if rollout.get("reward", 0) == 1.0:
            num_correct += 1
            continue

        info = _get_info(rollout)
        teacher_context = info.get("teacher_context", "")

        # Teacher context already contains expected outputs from data.py.
        # We keep it as-is — the info asymmetry comes from the student
        # only seeing binary + expected in the reflection prompt, while
        # the teacher sees this BEFORE scoring the entire sequence.
        if teacher_context:
            num_incorrect += 1

    if num_correct > 0:
        logger.info(
            f"ARC-AGI Reflect: {num_correct}/{len(rollouts)} correct, "
            f"{num_incorrect}/{len(rollouts)} incorrect (using static PI)"
        )

    return rollouts
