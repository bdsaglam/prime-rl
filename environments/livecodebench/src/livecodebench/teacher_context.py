"""SDPO-style teacher context for LiveCodeBench: test feedback + correct sibling.

Implements the PI strategy from SDPO (Hubotter et al., 2026):
- Test execution feedback (from sandbox results stored in info)
- Correct sibling solution from the batch (dynamic, changes each step)
- NO student's own attempt (SDPO Table 6 shows this hurts)
"""

from __future__ import annotations

import json


def _extract_completion_text(rollout: dict) -> str:
    """Extract the text of the student's completion from a rollout."""
    completion = rollout.get("completion", "")
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
    """Assemble SDPO-style teacher context from batch data.

    For each rollout, builds teacher_context from:
    1. Test execution feedback (from sandbox results stored in info)
    2. A correct sibling rollout's solution (if one exists for same problem)

    Does NOT include:
    - Student's own attempt (SDPO finds this reduces exploration)
    - Deliberative analysis

    Mutates rollout info in place.
    """
    # Group rollouts by problem
    by_problem: dict[str, list[dict]] = {}
    for rollout in rollouts:
        info = rollout.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
            rollout["info"] = info
        problem_id = info.get("problem_id", rollout.get("example_id", ""))
        by_problem.setdefault(problem_id, []).append(rollout)

    for problem_id, group in by_problem.items():
        # Find a correct sibling solution
        correct_sibling_text = None
        correct_rollout_id = None
        for r in group:
            if r.get("reward", 0) > 0:
                correct_sibling_text = _extract_completion_text(r)
                correct_rollout_id = id(r)
                break

        for rollout in group:
            info = rollout["info"] if isinstance(rollout["info"], dict) else json.loads(rollout["info"])

            reward = rollout.get("reward", 0)
            feedback = info.get("feedback", "")

            parts = []

            if reward > 0:
                # Correct rollout: minimal PI
                parts.append("Solution passed all tests.")
            else:
                # Incorrect rollout: provide test feedback + correct sibling
                if feedback:
                    parts.append(f"Test execution feedback:\n{feedback}")
                else:
                    parts.append("Solution did not pass all tests.")

            # Add correct sibling (not the rollout's own solution)
            if correct_sibling_text and id(rollout) != correct_rollout_id:
                parts.append(f"Correct solution:\n{correct_sibling_text}")
            elif correct_sibling_text and id(rollout) == correct_rollout_id:
                # This rollout IS the correct one -- use its own solution as demo
                parts.append(f"Correct solution:\n{correct_sibling_text}")

            info["teacher_context"] = "\n\n".join(parts)
            rollout["info"] = info
