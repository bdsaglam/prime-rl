# Phase 1: Privileged-Info Teacher

Give the teacher model privileged information (ground-truth answers, reference solutions, hints, etc.) so it produces directional per-token guidance rather than Phase 0's blind prefill.

## How Phase 0 works (baseline)

The teacher sees the exact same token sequence as the student:

```
[system prompt] [task examples + test input] [student's REPL turns]
                 ^--- identical to student ---^
```

Teacher logprobs come from `compute_teacher_logprobs()` (`src/prime_rl/orchestrator/utils.py:146-190`) which sends `sample.prompt_ids + sample.completion_ids` to the teacher via `/chat/completions/tokens`. The teacher scores the student's completion tokens given the same context the student had. With a frozen same-model teacher, the KL is just a regularizer against drift.

## What Phase 1 changes

The teacher sees a modified prompt that includes privileged information:

```
[system prompt + privileged info] [task examples + test input] [student's REPL turns]
^--- teacher sees more info ------^                             ^--- same completion ---^
```

This creates an information asymmetry: the teacher "knows the answer" and assigns higher probability to completion tokens that move toward the correct solution. The student doesn't see the answer but gets pulled toward the teacher's distribution via the KL term.

## Why this helps

- **Phase 0 problem**: If the teacher is the same model (or comparable), it's roughly as confused as the student. The KL term just prevents drift — it doesn't provide directional signal.
- **Phase 1 solution**: A teacher with privileged info assigns higher logprobs to correct reasoning steps and lower logprobs to wrong ones. This gives the student dense, directional per-token feedback — not just "stay close to init" but "move toward correct behavior."
- **Works with any teacher**: The privileged prompt works whether the teacher is the same model (self-distillation) or a larger model (even stronger signal).

## Architecture

Privileged info is **data-driven** — each task provides its own teacher context (ground truth, reference solution, hints, etc.) via the dataset. The system is environment-agnostic: any environment can supply `teacher_context` in its `info` dict.

### Data flow

```
Dataset row                     Rollout                         Orchestrator
├─ info["teacher_context"]  →   rollout["info"]             →   build_teacher_prompt_ids()
│  (string, from data)          ["teacher_context"]             ├─ get messages from trajectory[0]["prompt"]
│                                                               ├─ inject teacher_context into system message
│                                                               ├─ tokenize with apply_chat_template()
│                                                               └─ return teacher_prompt_ids
│
└─ (rest of data unchanged)     rollout["trajectory"]       →   interleave_rollout() → TrainingSample
                                                                  (unchanged from Phase 0)

                                                            →   compute_teacher_logprobs()
                                                                  ├─ uses teacher_prompt_ids + completion_ids
                                                                  ├─ handles logprob alignment
                                                                  └─ returns N+M logprobs (same format as Phase 0)
```

### Key design decisions

1. **teacher_prompt_ids NOT added to TrainingSample.** TrainingSample is a `msgspec.Struct` serialized from orchestrator to trainer (`src/prime_rl/transport/types.py`). Adding long token lists wastes serialization bandwidth. Instead, teacher_prompt_ids are built and consumed entirely within the orchestrator, passed as a parallel list to `compute_teacher_logprobs()`.

2. **Use original chat messages, not decode→re-encode.** The rollout's `trajectory[0]["prompt"]` contains the original OpenAI-format messages. We modify the system message and re-tokenize via `tokenizer.apply_chat_template()`. This avoids lossy decode→encode roundtrips of token IDs.

3. **One-to-many rollout→sample mapping.** `interleave_rollout()` can produce multiple TrainingSamples from one rollout (if extension breaks). In practice for ARC-AGI REPL, extension almost always holds (one sample per rollout). The teacher prompt is built once per rollout and shared across all its samples. If a rollout has no teacher_context, its samples fall back to Phase 0 behavior.

## Implementation

### Files to modify

| File | Change | Lines of code |
|------|--------|--------------|
| `environments/arc_agi/src/arc_agi/data.py` | Add `teacher_context` to `info` dict | ~15 |
| `src/prime_rl/orchestrator/utils.py` | New `build_teacher_prompt_ids()`, modify `compute_teacher_logprobs()` | ~40 |
| `src/prime_rl/orchestrator/orchestrator.py` | Build teacher prompts, pass to `compute_teacher_logprobs()` | ~20 |

No changes to: trainer, loss function, TrainingSample, config schema, or inference server.

---

### 1. Data layer: `environments/arc_agi/src/arc_agi/data.py`

Add `teacher_context` to the `ArcTask` TypedDict and populate it in `prepare_dataset()`.

**Extend ArcTask** (line 21):
```python
class ArcTask(TypedDict):
    task_id: str
    train: list[dict]
    test: list[dict]
    teacher_context: str | None  # NEW: privileged info for teacher prompt
```

**Populate in `prepare_dataset()`** (after line 181):

The `teacher_context` comes from an optional data file: `arc-agi_{split}_teacher_context.json`. This file maps `task_id → string`. If the file doesn't exist, `teacher_context` defaults to a formatted version of the ground-truth test outputs.

```python
# Load optional teacher context
teacher_context_path = base / f"arc-agi_{split_name}_teacher_context.json"
if teacher_context_path.exists():
    with open(teacher_context_path) as f:
        teacher_contexts = json.load(f)  # {task_id: str}
else:
    teacher_contexts = None

# In the per-task loop:
if teacher_contexts and task_id in teacher_contexts:
    teacher_context = teacher_contexts[task_id]
else:
    teacher_context = _format_default_teacher_context(test_pairs)

info = ArcTask(
    task_id=task_id,
    train=train_pairs,
    test=test_pairs,
    teacher_context=teacher_context,
)
```

**Default teacher context** (new helper):
```python
def _format_default_teacher_context(test_pairs: list[dict]) -> str:
    """Format ground-truth test outputs as default teacher context."""
    parts = []
    for i, pair in enumerate(test_pairs, 1):
        parts.append(f"Challenge #{i} expected output:")
        parts.append(format_grid(pair["output"]))
    return "\n".join(parts)
```

This uses the same grid formatting as the existing `format_task_question()`, keeping the style consistent.

---

### 2. Teacher prompt construction: `src/prime_rl/orchestrator/utils.py`

**New function: `build_teacher_prompt_ids()`**

Constructs teacher prompt token IDs by injecting `teacher_context` into the system message of the original chat messages:

```python
def build_teacher_prompt_ids(
    messages: list[dict],
    teacher_context: str,
    tokenizer,
) -> list[int]:
    """Build teacher prompt tokens with privileged info injected into system message.

    Takes the original chat messages from the rollout's first trajectory step,
    appends teacher_context to the system message, and tokenizes.
    """
    import copy
    modified = copy.deepcopy(messages)

    # Find and modify system message
    for msg in modified:
        if msg["role"] == "system":
            if isinstance(msg["content"], str):
                msg["content"] += f"\n\n--- PRIVILEGED INFORMATION ---\n{teacher_context}"
            break
    else:
        # No system message found — prepend one
        modified.insert(0, {
            "role": "system",
            "content": f"--- PRIVILEGED INFORMATION ---\n{teacher_context}",
        })

    return tokenizer.apply_chat_template(modified, tokenize=True, add_generation_prompt=True)
```

**Modify `compute_teacher_logprobs()`** (line 146):

Add `teacher_prompt_ids_list` parameter. When provided, use privileged prompt tokens. Handle logprob alignment for different prompt lengths.

```python
async def compute_teacher_logprobs(
    clients: list[vf.ClientConfig],
    model_name: str,
    samples: list[TrainingSample],
    max_model_len: int | None = None,
    teacher_prompt_ids_list: list[list[int] | None] | None = None,  # NEW
) -> list[list[float]]:
    """Compute teacher model logprobs for a batch of training samples via prefill.

    If teacher_prompt_ids_list is provided, uses privileged prompts instead of
    sample.prompt_ids. The returned logprobs are always aligned to the student's
    sequence (len = len(prompt_ids) + len(completion_ids)).
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

        # Truncate to fit within teacher's context window
        if max_model_len is not None and len(all_tokens) >= max_model_len:
            all_tokens = all_tokens[: max_model_len - 1]

        async with await get_semaphore():
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

        if teacher_prompt_ids is not None:
            # Privileged prompt: extract completion logprobs and align to student sequence
            teacher_prompt_len = len(prompt_ids)
            completion_logprobs = raw_logprobs[teacher_prompt_len:]
            # Pad completion logprobs if truncation cut into them
            if len(completion_logprobs) < len(sample.completion_ids):
                completion_logprobs.extend([0.0] * (len(sample.completion_ids) - len(completion_logprobs)))
            # Build aligned result: zeros for student prompt + teacher's completion logprobs
            logprobs = [0.0] * len(sample.prompt_ids) + completion_logprobs
        else:
            # Phase 0 behavior: same prompt, logprobs align directly
            logprobs = raw_logprobs
            if len(logprobs) < student_full_len:
                logprobs.extend([0.0] * (student_full_len - len(logprobs)))

        return logprobs

    # Build per-sample teacher_prompt_ids (None = fall back to Phase 0)
    if teacher_prompt_ids_list is None:
        teacher_prompt_ids_list = [None] * len(samples)

    return await asyncio.gather(*[
        _compute_single(client, sample, tp_ids)
        for client, sample, tp_ids in zip(cycle(clients), samples, teacher_prompt_ids_list)
    ])
```

### Logprob alignment

In Phase 0, teacher and student see the same prompt, so logprobs align 1:1. In Phase 1, the teacher's prompt is longer (has privileged info appended to system message). The sequences:

```
Student: [prompt_ids (N tokens)] [completion_ids (M tokens)]     -> N+M total
Teacher: [teacher_prompt_ids (N+K tokens)] [completion_ids (M tokens)] -> N+K+M total
```

The trainer loss computes `teacher_kl = teacher_logprobs - trainer_logprobs` element-wise over the full `N+M` sequence. We return `N+M` logprobs aligned to the student:

- **Prompt positions (0..N-1):** Set to `0.0`. These are masked by `loss_mask` (prompt_mask is all `False`), so they contribute nothing to the loss.
- **Completion positions (N..N+M-1):** The teacher's logprobs for the same completion tokens, extracted from positions `N+K..N+K+M-1` in the teacher's output.

---

### 3. Orchestrator wiring: `src/prime_rl/orchestrator/orchestrator.py`

After building `train_examples` (line 528), before calling `compute_teacher_logprobs` (line 530), build the teacher prompt IDs:

```python
# Build privileged teacher prompts if teacher_context is available
teacher_prompt_ids_list: list[list[int] | None] | None = None
if config.teacher_model and teacher_inference_pool:
    teacher_prompt_ids_list = _build_teacher_prompts(
        train_rollouts, results, tokenizer
    )

# Compute teacher logprobs (existing code, with new parameter)
if config.teacher_model and teacher_inference_pool:
    logger.info(f"Computing teacher logprobs for {len(train_examples)} training examples")
    teacher_logprobs_start_time = time.perf_counter()
    teacher_logprobs_list = await compute_teacher_logprobs(
        clients=teacher_inference_pool.clients,
        model_name=config.teacher_model.model.name,
        samples=train_examples,
        max_model_len=config.seq_len,
        teacher_prompt_ids_list=teacher_prompt_ids_list,  # NEW
    )
    for train_example, teacher_logprobs in zip(train_examples, teacher_logprobs_list):
        train_example.teacher_logprobs = teacher_logprobs
    teacher_logprobs_time = time.perf_counter() - teacher_logprobs_start_time
    logger.debug(f"Computed teacher logprobs in {teacher_logprobs_time:.2f}s")
```

**New helper function `_build_teacher_prompts()`** (in orchestrator.py or utils.py):

Maps rollouts → samples → teacher_prompt_ids. One rollout can produce multiple samples; all samples from a rollout share the same teacher context.

```python
def _build_teacher_prompts(
    train_rollouts: list[vf.RolloutOutput],
    rollout_samples: list[list[TrainingSample] | None],
    tokenizer,
) -> list[list[int] | None]:
    """Build privileged teacher prompt IDs for all training samples.

    Returns a flat list aligned with train_examples (same order as the
    for rollout, samples in zip(train_rollouts, rollout_samples) loop).
    """
    teacher_prompt_ids_list = []

    for rollout, samples in zip(train_rollouts, rollout_samples):
        if samples is None:
            continue

        # Extract teacher_context from rollout info
        info = rollout["info"]
        if isinstance(info, str):
            info = json.loads(info)
        teacher_context = info.get("teacher_context")

        if teacher_context is None:
            # No privileged info — fall back to Phase 0 for all samples from this rollout
            teacher_prompt_ids_list.extend([None] * len(samples))
            continue

        # Get original messages from first trajectory step
        messages = rollout["trajectory"][0]["prompt"]

        # Build privileged teacher prompt token IDs
        teacher_prompt_ids = build_teacher_prompt_ids(messages, teacher_context, tokenizer)

        # All samples from this rollout get the same teacher prompt
        teacher_prompt_ids_list.extend([teacher_prompt_ids] * len(samples))

    return teacher_prompt_ids_list
```

---

## Considerations

- **Token budget**: The privileged prompt is longer (ground truth grids add ~50-200 tokens depending on grid size). With `max_model_len=32768`, this is negligible.
- **Tokenizer compatibility**: Teacher and student must share the same tokenizer so completion token IDs are valid for both. True for Qwen3 family, NOT for cross-family teachers (e.g., DeepSeek).
- **Same model vs different model**: Works with both. With the same model, this is OPSD-style self-distillation. With a larger teacher (32B), the signal is even stronger since the teacher is both more capable and more informed.
- **No training loop changes**: Like Phase 0, this only modifies data loading and the orchestrator. The trainer sees `teacher_logprobs` in the same format.
- **Graceful fallback**: If `teacher_context` is missing from the data, the system falls back to Phase 0 behavior automatically (no privileged info). This means existing configs work without changes.
- **Multi-turn extension**: For ARC-AGI REPL, extension almost always holds (one sample per rollout). The teacher prompt is built from the first trajectory step's messages, which contain the full initial prompt (system + user message with task).

## Verification

1. **Unit test**: Create a test that builds teacher prompt IDs and verifies they contain the privileged info tokens.
2. **Data test**: Verify `teacher_context` appears in the `info` dict after `prepare_dataset()`.
3. **Integration test**: Run training with `--dump-config` to validate the config, then run a few steps and check:
   - W&B `teacher_kl` metric is positive and differs from Phase 0 (the privileged teacher should produce different logprobs)
   - Teacher logprobs length matches sample length (packer validation at `src/prime_rl/trainer/rl/packer.py:151-155` will catch misalignment)
4. **Smoke test**: Run `uv run rl @ configs/arc_agi/opd-rl-qwen-8b.toml` for a few steps and monitor logs for errors in teacher logprob computation.
