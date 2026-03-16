"""AIME math competition environment with reflection-in-sequence.

Multi-turn environment: student solves, gets weak feedback (correct/incorrect),
then reflects in structured format. Teacher scores the full sequence with
richer PI (answer + optional reference solution).

Based on the reflection-in-sequence signal measurement results showing
reflection tokens carry 2-7x stronger learning signal than solution tokens.
"""

from __future__ import annotations

import json
from typing import Any

import verifiers as vf
from datasets import Dataset, concatenate_datasets, load_dataset
from verifiers.parsers.maybe_think_parser import MaybeThinkParser
from verifiers.utils.data_utils import extract_boxed_answer


# ---------------------------------------------------------------------------
# Dataset loading (shared with aime env)
# ---------------------------------------------------------------------------

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
    info = _DATASET_REGISTRY.get(dataset_name)
    if info is None:
        _split = split or "train"
        return load_dataset(dataset_name, split=_split)

    _split = split or info["default_split"]
    configs = info["config"]
    if isinstance(configs, list):
        parts = [load_dataset(info["repo"], cfg)[_split] for cfg in configs]
        return concatenate_datasets(parts)
    return load_dataset(info["repo"], configs, split=_split)


def _prepare_dataset(dataset_name: str, split: str | None) -> Dataset:
    raw = _load_hf_dataset(dataset_name, split)
    info = _DATASET_REGISTRY.get(dataset_name, {})
    fields = info.get("fields", {"problem": "problem", "answer": "answer", "solution": None, "id": None})

    rows: list[dict] = []
    for i, row in enumerate(raw):
        problem = row[fields["problem"]]
        answer_raw = row[fields["answer"]]
        answer = _strip_non_numeric(str(answer_raw))
        try:
            answer = str(int(float(answer)))
        except (ValueError, TypeError):
            answer = str(answer_raw)

        sol_key = fields.get("solution")
        solution = row[sol_key] if sol_key and sol_key in row else ""
        id_key = fields.get("id")
        problem_id = str(row[id_key]) if id_key and id_key in row else f"aime_{i}"

        teacher_context = f"The correct answer is: {answer}"
        if solution:
            teacher_context += f"\n\nReference solution:\n{solution}"

        row_info = {
            "problem_id": problem_id,
            "teacher_context": teacher_context,
            "answer": answer,
        }
        if solution:
            row_info["solution"] = solution

        rows.append({
            "question": problem,
            "answer": answer,
            "info": json.dumps(row_info),
        })

    ds = Dataset.from_list(rows)
    ds = ds.sort("question")
    return ds


# ---------------------------------------------------------------------------
# Parser: extract answer from FIRST assistant message (solution turn, not reflection)
# ---------------------------------------------------------------------------


class SolutionTurnParser(MaybeThinkParser):
    """Extract \\boxed{} from the first assistant message, not the last.

    In multi-turn rollouts the completion contains:
      [assistant(solution), user(reflection_prompt), assistant(reflection)]
    The default parser picks the last assistant message (reflection) which
    has no \\boxed{}.  We need the first one.
    """

    def parse_answer(self, completion):
        if isinstance(completion, str):
            return self.parse(completion)
        assistant_messages = self.get_assistant_messages(completion)
        if not assistant_messages:
            return None
        # Use FIRST assistant message (solution turn)
        content = self._message_field(assistant_messages[0], "content", "") or ""
        return self.parse(self._content_to_text(content))


# ---------------------------------------------------------------------------
# Reflection prompts
# ---------------------------------------------------------------------------

# Student PI templates for incorrect rollouts
_STUDENT_PI_INCORRECT = {
    "none": (
        "Now step back and reflect on your solution above. "
        "How confident are you? Did you make any errors in your reasoning?"
    ),
    "binary": (
        "Your solution above is **incorrect**. The correct answer differs from yours.\n\n"
        "Reflect on your approach. Where did your reasoning go wrong? "
        "What was your key mistake?"
    ),
    "answer": (
        "Your solution above is **incorrect**. The correct answer is: {answer}\n\n"
        "Analyze where your reasoning diverged from the correct path. "
        "What was your key error?"
    ),
}

# Student PI templates for correct rollouts
_STUDENT_PI_CORRECT = {
    "none": (
        "Now step back and reflect on your solution above. "
        "How confident are you? Is there a cleaner or more rigorous approach?"
    ),
    "binary": (
        "Your solution above is **correct**. Well done.\n\n"
        "Reflect on your approach. Was your reasoning rigorous? "
        "Could you have been more efficient?"
    ),
    "answer": (
        "Your solution above is **correct**. The answer is: {answer}\n\n"
        "Reflect: was your approach the most efficient? "
        "Were there any steps where your reasoning was shaky?"
    ),
}

# Structured reflection format suffix (appended to student PI)
_STRUCTURED_FORMAT = (
    "\n\nRespond in this exact format:\n"
    "APPROACH: [one sentence describing what method/strategy you used]\n"
    "VERDICT: [correct/incorrect/unsure]\n"
    "CONFIDENCE: [high/medium/low]\n"
    "ERROR_TYPE: [computational/conceptual/approach/none]\n"
    "ERROR_LOCATION: [which step or reasoning segment]\n"
    "WHAT_WENT_WRONG: [one sentence]\n"
    "CORRECTION: [one sentence describing the fix]"
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Solve the following math problem. "
    "Show your work step by step and put your final answer in \\boxed{}."
)


class AimeReflectionEnv(vf.MultiTurnEnv):
    """AIME environment with reflection-in-sequence.

    Turn 1: Student solves the problem.
    Turn 2: Student reflects on its solution with structured format.

    The env_response between turns provides weak feedback (correct/incorrect)
    and asks for structured reflection. The teacher scores the full sequence
    with richer PI (answer + optional reference solution).
    """

    def __init__(
        self,
        student_pi: str = "binary",
        reflection_style: str = "structured",
        **kwargs,
    ):
        super().__init__(max_turns=2, **kwargs)
        self.student_pi = student_pi
        self.reflection_style = reflection_style

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> vf.Messages:
        """Return reflection prompt after student's solution."""
        # Only insert reflection prompt after turn 1 (solution)
        if len(state["trajectory"]) != 1:
            return []

        # Check correctness of solution
        answer = state["answer"]
        solution_text = ""
        last_msg = messages[-1]
        content = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")
        if isinstance(content, str):
            solution_text = content
        elif isinstance(content, list):
            solution_text = " ".join(
                getattr(p, "text", str(p)) if not isinstance(p, str) else p
                for p in content
            )

        is_correct = self._check_answer(solution_text, answer)

        # Build reflection prompt
        templates = _STUDENT_PI_CORRECT if is_correct else _STUDENT_PI_INCORRECT
        template = templates.get(self.student_pi, templates["binary"])
        prompt = template.format(answer=answer)

        if self.reflection_style == "structured":
            prompt += _STRUCTURED_FORMAT

        return [{"role": "user", "content": prompt}]

    def _check_answer(self, solution_text: str, expected_answer: str) -> bool:
        """Check if the student's solution contains the correct answer."""
        answer = vf.extract_boxed_answer(solution_text)
        if answer is None:
            return False
        answer_clean = _strip_non_numeric(str(answer))
        expected_clean = _strip_non_numeric(str(expected_answer))
        try:
            return str(int(float(answer_clean))) == str(int(float(expected_clean)))
        except (ValueError, TypeError):
            return answer_clean == expected_clean


# ---------------------------------------------------------------------------
# Teacher context preparation (called by orchestrator after rollout generation)
# ---------------------------------------------------------------------------


def _extract_text(completion) -> str:
    """Extract text content from a completion (Messages list or string)."""
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
                    # Handle content parts (e.g. [{"type": "text", "text": "..."}])
                    for p in content:
                        if isinstance(p, dict):
                            parts.append(p.get("text", str(p)))
                        elif isinstance(p, str):
                            parts.append(p)
        return "\n".join(parts)
    return str(completion)


def prepare_teacher_context(rollouts: list[dict]) -> None:
    """Assemble teacher context from available batch data.

    For each rollout, builds teacher_context from:
    1. The correct answer
    2. The student's own reflection (turn 2 completion)
    3. A correct sibling rollout's solution (if one exists for the same problem)

    Mutates rollout info in place.
    """
    # Group rollouts by problem for sibling matching
    by_problem: dict[str, list[dict]] = {}
    for rollout in rollouts:
        info = rollout.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
            rollout["info"] = info
        problem_id = info.get("problem_id", rollout.get("example_id", ""))
        by_problem.setdefault(problem_id, []).append(rollout)

    for problem_id, group in by_problem.items():
        # Find a correct sibling solution (if any)
        correct_solution = None
        for r in group:
            if r.get("reward", 0) > 0:
                traj = r.get("trajectory", [])
                if traj and traj[0].get("completion"):
                    correct_solution = _extract_text(traj[0]["completion"])
                    break

        for rollout in group:
            info = rollout["info"] if isinstance(rollout["info"], dict) else json.loads(rollout["info"])
            answer = info.get("answer", "")

            # Extract student's reflection (turn 2 completion)
            trajectory = rollout.get("trajectory", [])
            reflection = ""
            if len(trajectory) >= 2:
                completion = trajectory[1].get("completion")
                if completion:
                    reflection = _extract_text(completion)

            # Build teacher context
            parts = [f"The correct answer is: {answer}"]

            if reflection:
                parts.append(f"Your reflection on your previous attempt:\n{reflection}")

            # Add correct sibling (but not the rollout's own solution)
            if correct_solution and rollout.get("reward", 0) <= 0:
                parts.append(f"A correct solution to this problem:\n{correct_solution}")

            info["teacher_context"] = "\n\n".join(parts)
            rollout["info"] = info


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
    student_pi: str = "binary",
    reflection_style: str = "structured",
    **kwargs,
) -> vf.Environment:
    """Load an AIME reflection-in-sequence environment.

    Same dataset args as the standard AIME env, plus:
        student_pi: What the student sees before reflecting.
            "none" - just asked to reflect (blind)
            "binary" - told correct/incorrect (default)
            "answer" - told the correct answer
        reflection_style: Reflection format.
            "structured" - VERDICT/ERROR_TYPE/... format (default, best signal)
            "open" - free-form reflection
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

    rubric = vf.MathRubric(parser=SolutionTurnParser(extract_fn=extract_boxed_answer))

    env = AimeReflectionEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=system_prompt,
        rubric=rubric,
        student_pi=student_pi,
        reflection_style=reflection_style,
        **kwargs,
    )
    return env
