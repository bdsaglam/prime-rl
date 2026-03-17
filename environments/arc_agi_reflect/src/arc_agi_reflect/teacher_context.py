"""ARC-AGI Reflect teacher context preparation for on-policy distillation.

For ARC-AGI, the teacher PI includes:
- Expected output grids (from dataset)
- Student's reflection text (from the reflection turn)
- A correct sibling solution (if available in the batch)

The student already sees execution feedback + expected output in the reflection
prompt. The teacher sees the same PLUS the student's own reflection and a correct
sibling, giving it richer context for scoring.
"""

import json
import logging

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


def _extract_text(completion) -> str:
    """Extract text content from a completion."""
    if isinstance(completion, str):
        return completion
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
        return "\n".join(parts)
    return str(completion)


def prepare_teacher_context(rollouts: list[dict]) -> None:
    """Prepare teacher context for ARC-AGI Reflect rollouts.

    Assembles PI from:
    1. Expected output grids (from dataset info)
    2. Student's reflection (last trajectory step completion)
    3. Correct sibling solution (if available)

    Mutates rollout info in place.
    """
    # Group by example_id for sibling matching
    by_example: dict[str, list[dict]] = {}
    for rollout in rollouts:
        eid = rollout.get("example_id", "")
        by_example.setdefault(eid, []).append(rollout)

    num_enriched = 0

    for eid, group in by_example.items():
        # Find a correct sibling's code solution (if any)
        correct_solution = None
        for r in group:
            if r.get("reward", 0) > 0:
                traj = r.get("trajectory", [])
                # Get all assistant turns except the last (reflection)
                solution_parts = []
                for step in traj[:-1] if len(traj) > 1 else traj:
                    comp = step.get("completion")
                    if comp:
                        text = _extract_text(comp)
                        if text.strip():
                            solution_parts.append(text)
                if solution_parts:
                    correct_solution = "\n".join(solution_parts)
                    break

        for rollout in group:
            info = _get_info(rollout)

            # Start with existing teacher_context (expected outputs from data.py)
            base_context = info.get("teacher_context", "")

            # Extract student's reflection (last trajectory step)
            trajectory = rollout.get("trajectory", [])
            reflection = ""
            if len(trajectory) >= 2:
                last_step = trajectory[-1]
                comp = last_step.get("completion")
                if comp:
                    reflection = _extract_text(comp)

            # Build enriched teacher context
            parts = []
            if base_context:
                parts.append(base_context)

            if reflection:
                parts.append(f"Your reflection on your previous attempt:\n{reflection}")

            # Add correct sibling for incorrect rollouts
            if correct_solution and rollout.get("reward", 0) <= 0:
                parts.append(f"A correct solution to this problem:\n{correct_solution}")

            if parts:
                info["teacher_context"] = "\n\n".join(parts)
                rollout["info"] = info
                num_enriched += 1

    logger.info(
        f"ARC-AGI Reflect: enriched {num_enriched}/{len(rollouts)} rollouts with teacher context"
    )
