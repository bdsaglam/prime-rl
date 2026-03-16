"""ARC-AGI REPL environment with structured reflection step.

Extends ArcAgiReplEnv to add a final reflection turn after SUBMIT.
After the student submits their answer, the environment:
1. Evaluates correctness (exact match per test case)
2. Shows the student their result + expected output
3. Asks for structured reflection

This creates info asymmetry for OPD: the teacher can be given richer PI
(e.g., the full expected output, analysis of errors) while the student
only sees binary correctness + the expected output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import verifiers as vf

from arc_agi.data import format_grid
from arc_agi.envs.repl import ArcAgiReplEnv


# ---------------------------------------------------------------------------
# Reflection prompts
# ---------------------------------------------------------------------------

REFLECTION_PROMPT_STRUCTURED = """\
Your submission has been evaluated.

{result_summary}

Now reflect on your approach using this EXACT format:

VERDICT: [correct/incorrect/partially_correct]
ERROR_TYPE: [logic/implementation/pattern_recognition/none]
ERROR_LOCATION: [which step or iteration your reasoning went wrong, or "N/A"]
WHAT_WENT_WRONG: [one sentence describing your error, or "Nothing — solution is correct"]
LESSON: [one sentence — what you would do differently next time]

Be precise and concise. No extra commentary.\
"""

REFLECTION_PROMPT_OPEN = """\
Your submission has been evaluated.

{result_summary}

Reflect briefly on your approach:
- What was your reasoning strategy?
- Where did it succeed or fail?
- What would you do differently?

Keep your reflection concise (3-5 sentences).\
"""

REFLECTION_PROMPT_TIMEOUT = """\
You ran out of turns without submitting an answer.

Now reflect on your approach using this EXACT format:

VERDICT: incorrect
ERROR_TYPE: [logic/implementation/pattern_recognition/timeout]
ERROR_LOCATION: [which step or iteration your reasoning went wrong, or "N/A"]
WHAT_WENT_WRONG: [one sentence describing why you couldn't solve it in time]
LESSON: [one sentence — what you would do differently next time]

Be precise and concise. No extra commentary.\
"""

REFLECTION_PROMPTS = {
    "structured": REFLECTION_PROMPT_STRUCTURED,
    "open": REFLECTION_PROMPT_OPEN,
}


def _build_result_summary(
    submitted: list | None,
    expected: list,
) -> tuple[str, bool]:
    """Build a human-readable result summary comparing submission to expected.

    Returns (summary_text, is_fully_correct).
    """
    if submitted is None:
        return "You did not submit any answers.", False

    if len(submitted) != len(expected):
        return (
            f"You submitted {len(submitted)} grid(s) but {len(expected)} were expected.\n"
            "Your submission could not be evaluated."
        ), False

    all_correct = True
    parts = []

    for i, (pred, exp) in enumerate(zip(submitted, expected), 1):
        pred_arr = np.array(pred)
        exp_arr = np.array(exp)

        if pred_arr.shape == exp_arr.shape and np.array_equal(pred_arr, exp_arr):
            parts.append(f"Challenge #{i}: CORRECT")
        else:
            all_correct = False
            if pred_arr.shape != exp_arr.shape:
                cell_acc = 0.0
            else:
                cell_acc = float(np.mean(pred_arr == exp_arr))

            parts.append(
                f"Challenge #{i}: INCORRECT (cell accuracy: {cell_acc:.1%})\n"
                f"Your output:\n{format_grid(pred)}\n"
                f"Expected output:\n{format_grid(exp)}"
            )

    header = "All challenges correct!" if all_correct else "Some challenges were incorrect."
    return header + "\n\n" + "\n\n".join(parts), all_correct


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ArcAgiReflectEnv(ArcAgiReplEnv):
    """ARC-AGI REPL environment with a structured reflection step.

    After the student calls SUBMIT(), the environment evaluates the answer,
    shows the result (including expected output for incorrect cases), and
    asks for a structured reflection. The reflection is the final assistant
    turn — no more code execution happens after it.

    This creates rich OPD signal: the teacher scores both the solution turns
    AND the reflection turn, with info asymmetry on the reflection.
    """

    def __init__(
        self,
        reflection_style: str = "structured",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reflection_style = reflection_style

    @vf.stop
    async def max_turns_reached(self, state: vf.State) -> bool:
        """Override to reserve one extra turn for reflection on timeout.

        If we've hit max_turns but haven't reflected yet, allow one more
        turn by returning False. The env_response method will inject the
        reflection prompt on the next call.
        """
        at_limit = len(state["trajectory"]) >= self.max_turns and self.max_turns > 0
        if at_limit and not state.get("_reflection_received"):
            # Mark that we need a timeout reflection
            state["_needs_timeout_reflection"] = True
            # Allow one more turn for the reflection
            if not state.get("_awaiting_reflection"):
                return False
        return at_limit

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        state["_awaiting_reflection"] = False
        state["_reflection_received"] = False
        state["_needs_timeout_reflection"] = False
        return state

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        """Extended env_response that adds a reflection step after SUBMIT."""

        # If we're awaiting reflection, the student just provided it — we're done
        if state.get("_awaiting_reflection"):
            state["_reflection_received"] = True
            state["_awaiting_reflection"] = False
            # Store the final response so it's included in the trajectory
            state["final_env_response"] = [{"role": "user", "content": "Reflection received. Session complete."}]
            return []

        # Check if we need a timeout reflection (hit max_turns without SUBMIT)
        if state.get("_needs_timeout_reflection") and not state.get("_awaiting_reflection"):
            state["_awaiting_reflection"] = True
            return [{"role": "user", "content": REFLECTION_PROMPT_TIMEOUT}]

        # Normal REPL processing
        response = await super().env_response(messages, state, **kwargs)

        # Check if SUBMIT was called (super sets submitted_answers and returns [])
        if state.get("submitted_answers") is not None and not state.get("_reflection_received"):
            # SUBMIT happened — inject reflection prompt instead of ending
            info = state["info"]
            expected = [p["output"] for p in info["test"]]
            submitted = state["submitted_answers"].get("test") if isinstance(state["submitted_answers"], dict) else None

            result_summary, _ = _build_result_summary(submitted, expected)
            prompt_template = REFLECTION_PROMPTS.get(self.reflection_style, REFLECTION_PROMPTS["structured"])
            reflection_prompt = prompt_template.format(result_summary=result_summary)

            state["_awaiting_reflection"] = True

            # Build the response: include the REPL output from SUBMIT + reflection prompt
            final_response = state.get("final_env_response", [])
            repl_output = final_response[0]["content"] if final_response else "Answers submitted successfully."

            # Clear final_env_response so the framework doesn't stop the episode
            # before the reflection turn. The base env sets this on SUBMIT, but we
            # need one more turn for the student to reflect.
            state.pop("final_env_response", None)

            return [{"role": "user", "content": f"{repl_output}\n\n{reflection_prompt}"}]

        return response
