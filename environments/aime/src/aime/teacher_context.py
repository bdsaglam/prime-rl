"""AIME-specific teacher context preparation for on-policy distillation.

Generates adaptive privileged information (PI) tailored to each student rollout.
For correct rollouts (reward=1), skips the LLM call and uses static PI.
For incorrect rollouts, calls an external LLM with an AIME-specific prompt that
accounts for whether a reference solution is available.
"""

import asyncio
import json
import logging

import litellm

from prime_rl.configs.orchestrator import AnalyzerConfig

litellm.drop_params = True

logger = logging.getLogger(__name__)

# --- Analysis prompt styles ---
# Each style has informed (with solution) and blind (without solution) variants.
# Selection based on analyzer_config.analysis_style.

ANALYSIS_PROMPTS: dict[str, dict[str, str]] = {
    # Structured: short, constrained error report. Best discrimination (d=1.74 informed).
    "structured": {
        "informed": (
            "You are a math grading system. Analyze the student's attempt against the "
            "reference information and produce a SHORT, STRUCTURED error report.\n\n"
            "Format your response EXACTLY as:\n"
            "VERDICT: [correct/incorrect/partially_correct]\n"
            "ERROR_TYPE: [computational/conceptual/approach/notation/none]\n"
            "ERROR_LOCATION: [which step or reasoning segment contains the first error]\n"
            "WHAT_WENT_WRONG: [one sentence describing the error]\n"
            "SHOULD_HAVE: [one sentence describing what the correct step would be]\n\n"
            "Be precise and concise. No extra commentary."
        ),
        "blind": (
            "You are a math grading system. Analyze the student's attempt and "
            "produce a SHORT, STRUCTURED error report.\n\n"
            "Format your response EXACTLY as:\n"
            "VERDICT: [correct/incorrect/partially_correct]\n"
            "ERROR_TYPE: [computational/conceptual/approach/notation/none]\n"
            "ERROR_LOCATION: [which step or reasoning segment contains the first error]\n"
            "WHAT_WENT_WRONG: [one sentence describing the error]\n"
            "SHOULD_HAVE: [one sentence describing what the correct step would be]\n\n"
            "Be precise and concise. No extra commentary."
        ),
    },
    # Directive: guidance framed for the teacher. Best balance of signal and discrimination.
    "directive": {
        "informed": (
            "You will help a math teacher score a student's work. "
            "The teacher will read your guidance BEFORE scoring each token of the student's response.\n\n"
            "Write brief, actionable guidance for the teacher:\n"
            "- Where in the response does the reasoning go wrong?\n"
            "- What should the student have written instead?\n"
            "- Which parts are correct and should be reinforced?\n\n"
            "Keep it under 200 words. Be specific about locations in the response."
        ),
        "blind": (
            "You will help a math teacher score a student's work. "
            "The teacher will read your guidance BEFORE scoring each token of the student's response.\n\n"
            "Write brief, actionable guidance for the teacher:\n"
            "- Where in the response does the reasoning go wrong?\n"
            "- What should the student have written instead?\n"
            "- Which parts are correct and should be reinforced?\n\n"
            "Keep it under 200 words. Be specific about locations in the response."
        ),
    },
    # Verbose: multi-paragraph analysis. Highest |KL| but worst discrimination — NOT recommended.
    "verbose": {
        "informed": (
            "You are given a math competition problem, its correct answer, a reference solution, "
            "and an attempt at solving it.\n\n"
            "Generate concise problem notes highlighting:\n"
            "- Key pitfalls to avoid (informed by errors in the attempt, but phrased as general advice)\n"
            "- Critical intermediate steps that are easy to get wrong\n"
            "- Alternative approaches if the attempt's strategy is suboptimal\n\n"
            "Do NOT restate the answer or reproduce the reference solution. "
            "Focus only on insights that go beyond what the reference solution already provides.\n\n"
            "Write as concise problem notes — no preamble, no meta-commentary. "
            "Just the mathematical insights. Keep it to 2-4 sentences."
        ),
        "blind": (
            "You are given a math competition problem, its correct answer, and an attempt at solving it.\n\n"
            "Generate a short set of hints and notes about this problem that would help someone solve it correctly. "
            "Do NOT include the answer — just the insights needed to get there.\n\n"
            "Your hints should include:\n"
            "- A sketch of the correct solution approach\n"
            "- Key intermediate results or mathematical facts relevant to the problem\n"
            "- Specific pitfalls to avoid (informed by errors you see in the attempt, "
            "but phrased as general advice about the problem)\n"
            "- Alternative approaches if the attempt's strategy is suboptimal\n\n"
            "Write as concise problem notes — no preamble, no meta-commentary. "
            "Just the mathematical insights."
        ),
    },
    # Error point: minimal — just the single critical error.
    "error_point": {
        "informed": (
            "You are analyzing a math student's work. Your ONLY job is to identify "
            "the single most critical error.\n\n"
            "Respond with:\n"
            "1. The exact step where the error occurs (quote the student's text)\n"
            "2. Why it's wrong (one sentence)\n"
            "3. What the correct reasoning would be (one sentence)\n\n"
            "If the solution appears correct, say 'No critical error found.' Nothing else."
        ),
        "blind": (
            "You are analyzing a math student's work. Your ONLY job is to identify "
            "the single most critical error.\n\n"
            "Respond with:\n"
            "1. The exact step where the error occurs (quote the student's text)\n"
            "2. Why it's wrong (one sentence)\n"
            "3. What the correct reasoning would be (one sentence)\n\n"
            "If the solution appears correct, say 'No critical error found.' Nothing else."
        ),
    },
}

# Legacy aliases for backward compatibility
SYSTEM_PROMPT_WITH_SOLUTION = ANALYSIS_PROMPTS["verbose"]["informed"]
SYSTEM_PROMPT_WITHOUT_SOLUTION = ANALYSIS_PROMPTS["verbose"]["blind"]


def _extract_problem(rollout: dict) -> str:
    """Extract the problem text from a rollout's prompt messages."""
    trajectory = rollout.get("trajectory", [])
    if not trajectory:
        return ""
    for msg in trajectory[0].get("prompt", []):
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, str):
                return content
    return ""


def _extract_response(rollout: dict) -> str:
    """Extract the assistant's response from a rollout's completion messages."""
    parts = []
    for step in rollout["trajectory"]:
        for msg in step.get("completion", []):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
    return "\n".join(parts)


def _get_info(rollout: dict) -> dict:
    """Get rollout info as a dict, parsing JSON if needed."""
    info = rollout.get("info", {})
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except (json.JSONDecodeError, TypeError):
            info = {}
    return info if isinstance(info, dict) else {}


async def _call_llm(
    config: AnalyzerConfig,
    system_prompt: str,
    user_messages: list[str],
) -> list[str]:
    """Call the LLM for a batch of user messages with concurrency control."""
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def _single(user_content: str) -> str:
        async with semaphore:
            response = await litellm.acompletion(
                model=config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        return response.choices[0].message.content or ""

    return await asyncio.gather(*[_single(msg) for msg in user_messages])


async def prepare_teacher_context(
    analyzer_config: AnalyzerConfig,
    rollouts: list[dict],
) -> list[dict]:
    """Prepare teacher context for AIME rollouts.

    - Correct rollouts (reward=1): keep static PI (answer + ref solution)
    - Incorrect rollouts: call LLM for adaptive PI, then prepend answer
    """
    incorrect_indices = []
    incorrect_messages = []
    has_solution_flags = []

    for i, rollout in enumerate(rollouts):
        if rollout.get("reward", 0) == 1.0:
            continue
        if not rollout.get("trajectory"):
            continue

        info = _get_info(rollout)
        problem = _extract_problem(rollout)
        response = _extract_response(rollout)
        solution = info.get("solution", "")
        teacher_context = info.get("teacher_context", "")

        reference = teacher_context
        if solution and solution not in teacher_context:
            reference += f"\n\nReference solution:\n{solution}"

        incorrect_indices.append(i)
        has_solution_flags.append(bool(solution))
        incorrect_messages.append(
            f"## Problem\n{problem}\n\n"
            f"## Reference Information\n{reference}\n\n"
            f"## Attempt\n{response}"
        )

    num_correct = len(rollouts) - len(incorrect_indices)
    if num_correct > 0:
        logger.info(
            f"Skipping {num_correct}/{len(rollouts)} correct rollouts "
            f"(keeping static teacher_context)"
        )

    if not incorrect_indices:
        return rollouts

    # Group by prompt type (with/without solution) for batched calls
    with_solution = [
        (idx, msg) for idx, msg, has_sol
        in zip(incorrect_indices, incorrect_messages, has_solution_flags)
        if has_sol
    ]
    without_solution = [
        (idx, msg) for idx, msg, has_sol
        in zip(incorrect_indices, incorrect_messages, has_solution_flags)
        if not has_sol
    ]

    # Select prompts based on analysis_style (default: structured)
    style = getattr(analyzer_config, "analysis_style", "structured")
    style_prompts = ANALYSIS_PROMPTS.get(style, ANALYSIS_PROMPTS["structured"])

    analyses: dict[int, str] = {}

    if with_solution:
        prompt = analyzer_config.system_prompt or style_prompts["informed"]
        results = await _call_llm(
            analyzer_config, prompt, [msg for _, msg in with_solution]
        )
        for (idx, _), analysis in zip(with_solution, results):
            analyses[idx] = analysis

    if without_solution:
        prompt = analyzer_config.system_prompt or style_prompts["blind"]
        results = await _call_llm(
            analyzer_config, prompt, [msg for _, msg in without_solution]
        )
        for (idx, _), analysis in zip(without_solution, results):
            analyses[idx] = analysis

    # Update rollout info with adaptive teacher_context
    for idx, analysis in analyses.items():
        rollout = rollouts[idx]
        info = rollout["info"]
        if isinstance(info, str):
            info = json.loads(info)
            rollout["info"] = info
        if isinstance(info, dict):
            # Prepend the answer, then the adaptive analysis
            answer = info.get("teacher_context", "").split("\n")[0]  # "The correct answer is: X"
            info["teacher_context"] = f"{answer}\n\n{analysis}"

    logger.info(
        f"Generated {len(analyses)} adaptive teacher contexts "
        f"({len(with_solution)} with solution, {len(without_solution)} without)"
    )

    return rollouts
