import asyncio
import copy
import re
import time
from itertools import cycle
from pathlib import Path
from typing import Any, AsyncContextManager

import pandas as pd
import verifiers as vf
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage
from rich.console import Console
from rich.table import Table
from verifiers.utils.async_utils import maybe_semaphore
from verifiers.utils.client_utils import setup_openai_client

from prime_rl.configs.orchestrator import SamplingConfig
from prime_rl.transport import TrainingSample
from prime_rl.utils.utils import (
    format_num,
    format_time,
    get_broadcast_dir,
    get_ckpt_dir,
    get_step_path,
)

SEMAPHORE: AsyncContextManager | None = None


async def set_semaphore(limit: int):
    global SEMAPHORE
    SEMAPHORE = await maybe_semaphore(limit)


async def get_semaphore() -> AsyncContextManager:
    global SEMAPHORE
    assert SEMAPHORE is not None, "Semaphore not set"
    return SEMAPHORE


def get_sampling_args(sampling_config: SamplingConfig, temperature: float) -> dict:
    # Convert SamplingConfig to vLLM OAI sampling args
    # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters_2
    sampling_args = dict(sampling_config)
    sampling_args.pop("temp_scheduler", None)
    sampling_args["temperature"] = temperature
    sampling_args["top_p"] = 1.0
    sampling_args["logprobs"] = True
    sampling_args["extra_body"] = {
        **sampling_config.extra_body,
        "return_token_ids": True,  # Always return token IDs
        "top_k": -1,
        "min_p": 0.0,
    }
    sampling_args["extra_body"]["min_tokens"] = sampling_args.pop("min_tokens")
    sampling_args["extra_body"]["repetition_penalty"] = sampling_args.pop("repetition_penalty")
    return sampling_args


def parse_num_completion_tokens(responses: list[list[ChatCompletion]]) -> list[int]:
    """Parses the number of tokens from a list of chat completions returned by OAI API."""
    all_num_completion_tokens = []
    for response in responses:
        num_completion_tokens = 0
        for chat_completion in response:
            assert isinstance(chat_completion, ChatCompletion)
            assert chat_completion.usage is not None, "Usage should be present in the response"
            usage = chat_completion.usage
            assert isinstance(usage, CompletionUsage)
            num_completion_tokens += usage.completion_tokens
        all_num_completion_tokens.append(num_completion_tokens)
    assert len(all_num_completion_tokens) == len(responses), (
        "Number of completion tokens should be the same as the number of responses"
    )
    return all_num_completion_tokens


def parse_is_truncated_completions(responses: list[list[ChatCompletion]]) -> list[bool]:
    """Parses whether the completions were truncated from a list of (multi-turn) OAI chat completions"""
    all_is_truncated = []
    for response in responses:
        is_truncated = False
        for chat_completion in response:
            assert isinstance(chat_completion, ChatCompletion)
            assert len(chat_completion.choices) == 1, "Response should always have one choice"
            choice = chat_completion.choices[0]
            assert isinstance(choice, Choice)
            if choice.finish_reason == "length":
                is_truncated = True
        all_is_truncated.append(is_truncated)
    return all_is_truncated


def print_benchmark(history: dict[str, list[Any]]) -> None:
    """
    Print benchmark results as rich table. Shows formatted values for the
    inference throughput and overall step time. First first N rows show the
    per-step values, and the last row shows the mean, std, min, and max values.
    """
    history.pop("step")
    assert all(len(v) for v in history.values()), "All metrics must have logged the same number of steps"

    # Turn metric history into pd.DataFrame
    df = pd.DataFrame(dict(history.items()))
    columns = {
        "perf/throughput": "Throughput",
        "time/step": "Step Time",
    }
    df = df.rename(columns=columns)
    df = df[list(columns.values())]
    df = df.iloc[1:]  # Exclude first row

    # Setup console
    console = Console()
    table = Table(title="Benchmark")

    # Add columns
    table.add_column("Step", justify="right")
    for col in df.columns:
        table.add_column(col, justify="center", style="magenta")

    # Add formatted rows
    formatted_df = pd.DataFrame(columns=df.columns)
    formatted_df["Step Time"] = df["Step Time"].apply(format_time)
    formatted_df["Throughput"] = df["Throughput"].apply(format_num, precision=2)
    for step, row in formatted_df.iterrows():
        table.add_row(*([str(step)] + [str(x) for x in row]))

    # Separator
    num_table_columns = 1 + len(df.columns)
    table.add_row(*([""] * num_table_columns))

    # Add row for formatted, aggregated statistics
    mean_df = df.describe().loc[["mean", "std", "min", "max"], :]
    formatted_mean_df = pd.DataFrame(columns=mean_df.columns)
    formatted_mean_df["Step Time"] = mean_df["Step Time"].apply(format_time)
    formatted_mean_df["Throughput"] = mean_df["Throughput"].apply(format_num, precision=2)
    mean_row = ["Overall"] + formatted_mean_df.T.apply(
        lambda row: f"{row['mean']} ± {row['std']} [{row['min']}, {row['max']}]", axis=1
    ).tolist()
    table.add_row(*mean_row)

    # Display table
    console.print(table)


def build_teacher_prompt_ids(
    messages: list[dict],
    teacher_context: str,
    tokenizer,
    chat_template_kwargs: dict | None = None,
) -> list[int]:
    """Build teacher prompt tokens with privileged info appended to the first user message.

    Following SDPO/SDFT, PI is appended to the user message containing the problem,
    then followed by "Correctly solve the original question." The system message
    is left unchanged.

    Args:
        messages: Original OpenAI-format chat messages from the rollout.
        teacher_context: Privileged information string to inject.
        tokenizer: HuggingFace tokenizer (must match teacher model's tokenizer).
        chat_template_kwargs: Extra kwargs for apply_chat_template (e.g. enable_thinking=False).

    Returns:
        Token IDs for the teacher's privileged prompt.
    """
    modified = copy.deepcopy(messages)

    # Find the first user message (the problem) and append PI
    for msg in modified:
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                msg["content"] += (
                    f"\n\n{teacher_context}\n\n"
                    "Correctly solve the original question."
                )
            break

    return tokenizer.apply_chat_template(
        modified, tokenize=True, add_generation_prompt=True, return_dict=False, **(chat_template_kwargs or {})
    )


DEFAULT_DELIBERATIVE_PROMPT = """You are an expert math teacher analyzing a student's work.
Carefully read the problem and the student's attempt below. You do NOT know the correct answer.
Your job is to analyze the student's reasoning process in depth:
1. What approach/strategy did the student use?
2. Trace through the key reasoning steps — are they logically valid?
3. Identify any specific steps where errors might have occurred and why.
4. Assess the overall quality: Is the reasoning sound? Are there gaps?
5. What should the student have done differently?
Be specific about which steps are good and which are problematic."""

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return _THINK_RE.sub("", text).strip()


async def generate_deliberative_analyses(
    clients: list[vf.ClientConfig],
    model_name: str,
    rollouts: list[dict],
    system_prompt: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    max_model_len: int = 32768,
) -> list[str]:
    """Generate deliberative analyses for student rollouts using the teacher model.

    For each rollout, calls the teacher to analyze the student's work (blind — no answer).
    Returns a list of analysis strings, one per rollout.

    Think tags are stripped from the teacher's analysis output so that only the
    substantive analysis is used as privileged information. Student rollouts are
    sent to the teacher unmodified.
    """
    prompt = system_prompt or DEFAULT_DELIBERATIVE_PROMPT

    async def _generate_single(
        client_config: vf.ClientConfig,
        rollout: dict,
    ) -> str:
        client = setup_openai_client(client_config)

        # Extract problem and student response from rollout
        trajectory = rollout.get("trajectory", [])
        if not trajectory:
            return ""
        messages = trajectory[0].get("prompt", [])

        # Find the user message (problem)
        problem_text = ""
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, str):
                    problem_text = content
                break

        # Get student response from completion messages (all assistant turns concatenated)
        student_response = ""
        for step in trajectory:
            for msg in step.get("completion", []):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        student_response += content + "\n"

        # Truncate student response to fit within context. We need room for:
        # system prompt + problem + framing + max_tokens for output.
        # Use conservative ratio of 2 chars/token to avoid exceeding context.
        overhead_tokens = 512  # system prompt, problem, chat template overhead
        input_budget_tokens = max_model_len - max_tokens - overhead_tokens
        input_budget_chars = max(1000, input_budget_tokens * 2)  # 2 chars/token = very conservative
        if len(student_response) > input_budget_chars:
            student_response = student_response[:input_budget_chars] + "\n[... truncated for context length ...]"

        # Use conservative token estimate (2 chars/token) to cap max_tokens
        estimated_input_tokens = (len(prompt) + len(problem_text) + len(student_response) + 200) // 2
        effective_max_tokens = min(max_tokens, max(256, max_model_len - estimated_input_tokens - 100))

        async with await get_semaphore():
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Problem:\n{problem_text}\n\nStudent's attempt:\n{student_response}"},
                ],
                max_tokens=effective_max_tokens,
                temperature=temperature,
            )

        analysis = response.choices[0].message.content or ""
        # Strip <think>...</think> from teacher's analysis — use only the substantive output
        return _strip_think_tags(analysis)

    return await asyncio.gather(*[
        _generate_single(client, rollout)
        for client, rollout in zip(cycle(clients), rollouts)
    ])


async def compute_teacher_logprobs(
    clients: list[vf.ClientConfig],
    model_name: str,
    samples: list[TrainingSample],
    max_model_len: int | None = None,
    teacher_prompt_ids_list: list[list[int] | None] | None = None,
    use_tito: bool = True,
) -> list[list[float]]:
    """Compute teacher model logprobs for a batch of training samples via prefill.

    If teacher_prompt_ids_list is provided, uses privileged prompts instead of
    sample.prompt_ids (Phase 1 OPD). The returned logprobs are always aligned to
    the student's sequence length (len(prompt_ids) + len(completion_ids)).

    Args:
        use_tito: If True, use prime-rl's /chat/completions/tokens endpoint.
            If False, use standard /v1/completions with prompt_logprobs (for external vLLM servers).
    """

    async def _compute_single(
        client_config: vf.ClientConfig,
        sample: TrainingSample,
        teacher_prompt_ids: list[int] | None = None,
    ) -> list[float]:
        client = setup_openai_client(client_config)

        # Use privileged prompt if provided, otherwise fall back to student prompt
        prompt_ids = teacher_prompt_ids if teacher_prompt_ids is not None else sample.prompt_ids
        all_tokens = prompt_ids + sample.completion_ids
        student_full_len = len(sample.prompt_ids) + len(sample.completion_ids)

        # Truncate to fit within teacher's context window (need room for max_tokens=1)
        if max_model_len is not None and len(all_tokens) >= max_model_len:
            all_tokens = all_tokens[: max_model_len - 1]

        async with await get_semaphore():
            if use_tito:
                response = await client.post(
                    "/chat/completions/tokens",
                    body={
                        "model": model_name,
                        "messages": [{"role": "user", "content": ""}],
                        "tokens": all_tokens,
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "skip_special_tokens": False,
                        "prompt_logprobs": True,
                    },
                    cast_to=ChatCompletion,
                )
                raw_logprobs = [
                    0.0 if lp is None else float(next(iter(lp.values()))["logprob"])
                    for lp in getattr(response, "prompt_logprobs", [])
                ]
            else:
                # Standard vLLM completions endpoint — works with any vLLM server.
                # prompt_logprobs is a vLLM extension, so we access it from the
                # response's underlying model data via choices[0].
                response = await client.completions.create(
                    model=model_name,
                    prompt=all_tokens,
                    max_tokens=1,
                    temperature=1.0,
                    top_p=1.0,
                    extra_body={"prompt_logprobs": 1},
                )
                # vLLM puts prompt_logprobs on choices[0] as an extension field
                choice = response.choices[0]
                prompt_lps = getattr(choice, "prompt_logprobs", None) or []
                raw_logprobs = [
                    0.0 if lp is None else float(next(iter(lp.values()))["logprob"])
                    for lp in prompt_lps
                ]

        if teacher_prompt_ids is not None:
            # Phase 1: privileged prompt is longer — extract completion logprobs and align
            teacher_prompt_len = len(prompt_ids)
            completion_logprobs = raw_logprobs[teacher_prompt_len:]
            # Pad if truncation cut into completion tokens
            if len(completion_logprobs) < len(sample.completion_ids):
                completion_logprobs.extend([0.0] * (len(sample.completion_ids) - len(completion_logprobs)))
            # Build aligned result: zeros for student prompt + teacher's completion logprobs
            logprobs = [0.0] * len(sample.prompt_ids) + completion_logprobs
        else:
            # Phase 0: same prompt, logprobs align directly
            logprobs = raw_logprobs
            if len(logprobs) < student_full_len:
                logprobs.extend([0.0] * (student_full_len - len(logprobs)))

        return logprobs

    if teacher_prompt_ids_list is None:
        teacher_prompt_ids_list = [None] * len(samples)

    return await asyncio.gather(*[
        _compute_single(client, sample, tp_ids)
        for client, sample, tp_ids in zip(cycle(clients), samples, teacher_prompt_ids_list)
    ])


def get_weight_dir(output_dir: Path, step: int, check_exists: bool = True, wait_timeout: int | None = None) -> Path:
    """Get the weight directory for a given checkpoint step.

    Args:
        output_dir: The output directory for the run.
        step: The checkpoint step.
        check_exists: If True, raises FileNotFoundError if no weight directory exists.
            If False, returns the broadcast directory path without checking existence
            (useful for NCCL mode where weights are broadcasted, not stored on disk).
        wait_timeout: Maximum time in seconds to wait for a stable directory to appear.
            If None, no waiting is performed.
    """
    ckpt_weight_dir = get_step_path(get_ckpt_dir(output_dir), step) / "weight"
    broadcast_weight_dir = get_step_path(get_broadcast_dir(output_dir), step)

    def find_stable_dir() -> Path | None:
        # For checkpoint weights, check STABLE file in parent directory (checkpoints/step_{step}/STABLE)
        ckpt_step_dir = get_step_path(get_ckpt_dir(output_dir), step)
        if (ckpt_step_dir / "STABLE").exists() and ckpt_weight_dir.exists():
            return ckpt_weight_dir

        # For broadcast weights, check STABLE file in the broadcast directory itself
        if (broadcast_weight_dir / "STABLE").exists() and broadcast_weight_dir.exists():
            return broadcast_weight_dir

        return None

    # Check immediately, then wait if needed
    result = find_stable_dir()
    if result is None and wait_timeout:
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            time.sleep(1)
            result = find_stable_dir()
            if result:
                break

    if result:
        return result
    if not check_exists:
        return broadcast_weight_dir

    raise FileNotFoundError(f"No weight directory found for checkpoint step {step}")
