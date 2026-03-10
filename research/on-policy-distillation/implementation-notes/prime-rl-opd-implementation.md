# Prime-RL On-Policy Distillation (OPD) Implementation

Research notes on the prime-rl OPD feature, merged via PR #1458 (Dec 31, 2025).

## Overview

On-policy distillation in prime-rl uses a teacher model to provide **dense token-level feedback** during RL training. The student model generates rollouts (on-policy), then the teacher scores those same rollouts by computing logprobs via a prefill pass. The teacher's token-level signal is blended into the loss function alongside (or instead of) reward-based advantages.

Key design: the teacher never generates text -- it only scores the student's existing completions.

## Configuration (TOML)

### Minimal Setup (Same Model as Teacher)

```toml
teacher_gpu_ids = [2, 3]

[trainer.loss]
teacher_tau = 0.5
```

When `teacher_gpu_ids` is set and no `[teacher_inference]` section exists, prime-rl deep-copies the main `[inference]` config, auto-assigns the next port, and launches a separate vLLM inference server on the specified GPUs for the teacher. The teacher model defaults to the same model as the student inference server.

### Custom Teacher Model

```toml
teacher_gpu_ids = [2, 3]

[teacher_inference.model]
name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

[trainer.loss]
teacher_tau = 0.5
```

### External Teacher Server (No Local GPUs)

```toml
[trainer.loss]
teacher_tau = 0.5

[orchestrator.teacher_model.client]
base_url = ["http://teacher-server:8000/v1"]

[orchestrator.teacher_model.model]
name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

No `teacher_gpu_ids` needed -- connects to an already-running vLLM-compatible server.

### Pure Distillation Mode (No RL Rewards)

```toml
teacher_gpu_ids = [2, 3]

[trainer.loss]
teacher_tau = 1.0
adv_tau = 0.0           # disable reward-based learning entirely

[orchestrator.buffer]
skip_verification = true  # skip environment reward computation
```

### Hybrid Mode (RL + Distillation)

```toml
teacher_gpu_ids = [2, 3]

[trainer.loss]
teacher_tau = 0.5   # distillation strength
adv_tau = 1.0       # RL advantage weight (default)
```

Both `teacher_tau > 0` and `adv_tau > 0` enables hybrid mode where the student learns from both the teacher's token-level signal AND reward-based advantages simultaneously.

## Configuration Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `teacher_gpu_ids` | `None` | GPU IDs for teacher inference server. Triggers auto-launch. |
| `teacher_inference` | `None` | Full inference config for teacher. Auto-derived from `inference` if absent. |
| `trainer.loss.teacher_tau` | `0.0` | Weight of teacher KL signal in the loss. Must be `> 0` to activate distillation. |
| `trainer.loss.adv_tau` | `1.0` | Weight of reward-based advantages. Set to `0` for pure distillation. |
| `trainer.loss.kl_tau` | `0.0` | Weight of KL divergence regularization against inference policy. |
| `orchestrator.teacher_model.client.base_url` | `[]` | URLs for external teacher server(s). |
| `orchestrator.teacher_model.model.name` | `""` | Teacher model name for API calls. |
| `orchestrator.buffer.skip_verification` | `false` | Skip reward computation (for pure distillation). |

## GPU Allocation

The `RLConfig` top-level class in `src/prime_rl/rl.py` manages GPU assignment:

```python
class RLConfig(BaseRLConfig):
    inference_gpu_ids: list[int] = [0]       # student rollout generation
    trainer_gpu_ids: list[int] = [1]         # gradient updates
    teacher_gpu_ids: list[int] | None = None # teacher inference (optional)
```

Example 4-GPU allocation:
- GPUs 0,1: student inference (rollout generation)
- GPU 2: teacher inference
- GPU 3: trainer

Example 8-GPU allocation with larger teacher:
- GPUs 0,1,2,3: student inference
- GPUs 4,5: teacher inference (TP=2 or DP=2)
- GPUs 6,7: trainer

The auto-setup validator (`auto_setup_teacher_inference`) handles:
1. Deep-copying the inference config if `teacher_inference` is not explicitly set
2. Assigning a non-conflicting port (inference port + 1)
3. Computing data parallelism: `dp = len(teacher_gpu_ids) // tp`
4. Wiring the orchestrator's `teacher_model.client` to point at the teacher server
5. Setting `CUDA_VISIBLE_DEVICES` when spawning the teacher process

## Loss Function Implementation

Source: `src/prime_rl/trainer/rl/loss.py`

### Core Loss Equation

The loss function is a masked importance-sampling policy gradient with three additive signal components:

```python
# line 115-116
log_importance_ratio = trainer_logprobs - inference_logprobs
teacher_kl = teacher_logprobs - trainer_logprobs  # if teacher_logprobs is not None

# line 148-150: combine advantages and teacher signal
advantages = adv_tau * advantages
if teacher_logprobs is not None:
    advantages = advantages + teacher_tau * teacher_kl.detach()

# line 151-152: final loss coefficient
coeff = importance_ratio * (advantages - kl_tau * log_importance_ratio)
loss = -(coeff.detach() * trainer_logprobs)[keep_mask].sum()
```

In mathematical notation, the effective per-token "advantage" signal is:

```
effective_advantage = adv_tau * A(s,a) + teacher_tau * (log p_teacher(a|s) - log p_student(a|s))
```

Where:
- `A(s,a)` is the reward-based advantage (from GRPO or similar)
- `log p_teacher - log p_student` is the per-token KL from student to teacher (positive when teacher assigns higher probability)
- The `teacher_kl` is `.detach()`-ed so it acts as a reward signal, not a direct gradient

The teacher KL term pushes the student toward tokens the teacher would prefer. When `teacher_tau` is high and `adv_tau` is 0, this becomes pure distillation: the student learns to match the teacher's distribution on student-generated rollouts.

### Importance Sampling and Masking

The loss includes sophisticated off-policy correction:
- **Token-level masking**: tokens with importance ratio outside `[token_mask_low, token_mask_high]` (default: `[0.125, 8.0]`) are masked
- **Sequence-level masking**: sequences with any token outside `[sequence_mask_low, sequence_mask_high]` are masked
- **Geometric sequence ratio masking**: based on geometric mean of token ratios
- Both `token` and `sequence` importance ratio types are supported (`ratio_type` config)

### Monitoring Metric: `teacher_kl`

```python
if teacher_kl is not None:
    metrics["teacher_kl"] = _safe_mean(teacher_kl, loss_mask)
```

`teacher_kl = log p_teacher - log p_student`, averaged over completion tokens. Lower values indicate the student is converging toward the teacher's behavior. This metric should trend downward during successful distillation.

## How Teacher Logprobs Are Computed

Source: `src/prime_rl/orchestrator/utils.py :: compute_teacher_logprobs()`

### The Prefill Approach

The teacher does NOT generate text. It scores the student's already-generated completions using the vLLM `/chat/completions/tokens` endpoint:

```python
async def compute_teacher_logprobs(
    clients: list[vf.ClientConfig],
    model_name: str,
    samples: list[TrainingSample],
) -> list[list[float]]:

    async def _compute_single(client_config, sample):
        client = setup_openai_client(client_config)
        async with await get_semaphore():
            response = await client.post(
                "/chat/completions/tokens",
                body={
                    "model": model_name,
                    "messages": [{"role": "user", "content": ""}],
                    "tokens": sample.prompt_ids + sample.completion_ids,  # full sequence
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "skip_special_tokens": False,
                    "prompt_logprobs": True,  # return per-token logprobs
                },
                cast_to=ChatCompletion,
            )
        return [
            0.0 if lp is None else float(next(iter(lp.values()))["logprob"])
            for lp in getattr(response, "prompt_logprobs", [])
        ]

    return await asyncio.gather(*[
        _compute_single(client, sample)
        for client, sample in zip(cycle(clients), samples)
    ])
```

Key details:
- Sends `prompt_ids + completion_ids` as the full token sequence
- Uses `prompt_logprobs=True` to get logprobs for every token in the sequence
- Uses vLLM's `/chat/completions/tokens` endpoint (a vLLM-specific extension, NOT standard OpenAI API)
- Requests are distributed across teacher clients in round-robin fashion (`cycle(clients)`)
- Runs asynchronously with semaphore-based concurrency control
- `temperature=1.0, top_p=1.0` ensures raw logprobs without temperature scaling

### Data Flow

1. Student generates rollouts via inference server
2. Rollouts are converted to `TrainingSample` objects (with `teacher_logprobs=None`)
3. Orchestrator calls `compute_teacher_logprobs()` on all training samples
4. Teacher logprobs are attached: `train_example.teacher_logprobs = teacher_logprobs`
5. Samples are packed into batches and sent to the trainer
6. Trainer unpacks teacher logprobs and passes them to `compute_loss()`

## Hybrid Mode (OPD + RL Rewards)

Yes, prime-rl fully supports combining distillation with RL rewards. The combination is controlled by `adv_tau` and `teacher_tau`:

| `adv_tau` | `teacher_tau` | Mode |
|-----------|---------------|------|
| 1.0 | 0.0 | Pure RL (default) |
| 0.0 | 1.0 | Pure distillation |
| 1.0 | 0.5 | Hybrid (RL + distillation) |
| 0.5 | 0.5 | Balanced hybrid |

The effective advantage is simply `adv_tau * advantages + teacher_tau * teacher_kl`, so both signals are additive. There is no special "hybrid mode" flag -- it emerges naturally from setting both tau values > 0.

## Self-Distillation Analysis

### Does prime-rl support self-distillation?

**Partially, with significant caveats.** There are two interpretations:

#### 1. Same model architecture, same weights (trivial self-distillation)
If `teacher_gpu_ids` is set without specifying `[teacher_inference.model]`, the teacher defaults to a deep-copy of the inference config, meaning the same model. However, this is **not useful** because:
- Teacher and student would have identical logprobs initially
- `teacher_kl` would be zero everywhere
- As the student trains, its weights diverge from the frozen teacher (teacher weights are NOT updated)
- This becomes "distillation from frozen initial checkpoint" -- potentially useful as a conservative regularizer

#### 2. Self-distillation with privileged information (same model, different context)
This is the more interesting case for ARC-AGI: using the same model as teacher but giving it privileged information (e.g., the correct answer, examples of solved puzzles). **This is NOT natively supported.** The current implementation:

- Teacher sees the exact same token sequence as the student (`sample.prompt_ids + sample.completion_ids`)
- There is no mechanism to inject a different prompt or additional context for the teacher
- The `/chat/completions/tokens` endpoint receives raw token IDs, not messages with different system prompts

To implement privileged-info self-distillation, you would need to modify `compute_teacher_logprobs()` to:
1. Construct a different prompt for the teacher (with privileged info)
2. Append the student's completion tokens
3. Score the student's completion under the teacher's augmented context

## Limitations and Caveats

1. **vLLM-specific endpoint**: `compute_teacher_logprobs()` uses `/chat/completions/tokens` with `prompt_logprobs=True`, which is a vLLM-specific extension. This will not work with other OpenAI-compatible servers.

2. **No teacher weight updates**: The teacher model is frozen. Its weights are never updated during training. If you need a co-evolving teacher, you'd need custom logic.

3. **Sequential bottleneck**: Teacher logprobs are computed AFTER student rollouts complete but BEFORE training begins. This adds latency to each training step proportional to `batch_size / teacher_throughput`.

4. **Token-level only**: The distillation signal is purely token-level logprobs. There is no sequence-level or trajectory-level distillation signal.

5. **No privileged-info routing**: The teacher sees the identical token sequence as the student. There is no mechanism for the teacher to receive additional context (e.g., ground-truth answers, demonstrations).

6. **Same tokenizer assumption**: Teacher and student must share the same tokenizer since raw token IDs are sent directly to the teacher.

7. **skip_verification constraints**: When `skip_verification = true`, several features are disabled:
   - `online_difficulty_filtering` must be off
   - `easy_threshold` and `hard_threshold` cannot be set
   - All rewards are set to 0

8. **Memory overhead**: The teacher inference server requires its own GPU memory allocation, separate from both the student inference and trainer.

## Modifications Needed for ARC-AGI REPL Environment

### Current ARC-AGI Setup (from our configs)
Our `arc-agi-qwen3-8b.toml` uses:
- 3 GPUs for inference, 1 for training
- 32K sequence length
- LoRA training (rank 32)
- Multi-turn REPL environment with tool calling

### Required Modifications

#### 1. GPU Budget Expansion
OPD requires additional GPUs for the teacher. Options:
- **Minimal (6 GPUs)**: 2 inference + 2 teacher + 1 trainer + 1 spare
- **Current hardware (4 GPUs)**: Could work with `[orchestrator.teacher_model.client]` pointing to an external teacher server (e.g., a larger model served on a separate machine or cloud API)

Example config addition for our 4-GPU setup using an external teacher:
```toml
# Keep existing GPU allocation
inference_gpu_ids = [0, 1, 2]
trainer_gpu_ids = [3]

# Use external teacher (no local teacher GPUs needed)
[trainer.loss]
teacher_tau = 0.3
adv_tau = 1.0

[orchestrator.teacher_model.client]
base_url = ["http://teacher-host:8000/v1"]

[orchestrator.teacher_model.model]
name = "Qwen/Qwen3-32B"
```

#### 2. Pure Distillation for ARC-AGI (Cheaper Exploration)
ARC-AGI REPL rollouts are expensive (multi-turn tool use, code execution). Pure distillation could skip reward verification:

```toml
teacher_gpu_ids = [4, 5]

[trainer.loss]
teacher_tau = 1.0
adv_tau = 0.0

[orchestrator.buffer]
skip_verification = true
```

**Caveat**: This completely removes the environment reward signal. The student learns ONLY from the teacher. This is only useful if the teacher is substantially better at ARC-AGI than the student.

#### 3. Privileged-Info Self-Distillation (Custom Implementation Required)
For the ARC-AGI use case, the most interesting variant would be:
- **Teacher**: Same model, but given the correct output grid in the prompt as an additional demonstration
- **Student**: Standard ARC-AGI REPL prompt (no answer)
- **Signal**: Teacher logprobs on student-generated code/reasoning are higher when the student's approach aligns with what works given the known answer

This requires modifying `compute_teacher_logprobs()` in the orchestrator. Specifically:
1. In `trajectories.py`, store the ground-truth answer alongside each `TrainingSample`
2. In `utils.py :: compute_teacher_logprobs()`, construct a teacher-specific prompt that includes the ground truth
3. Tokenize the augmented teacher prompt + student completion and send to the teacher for scoring
4. The returned logprobs cover only the student completion tokens (after the augmented prompt)

This modification is non-trivial because:
- The current API sends raw token IDs, not messages
- Different prompt lengths mean logprob alignment gets complex
- The teacher prompt must be carefully constructed to avoid information leakage into the token sequence

#### 4. Hybrid Mode Recommendation
For ARC-AGI, the most pragmatic approach is hybrid mode with a stronger teacher:
```toml
teacher_gpu_ids = [4, 5]

[teacher_inference.model]
name = "Qwen/Qwen3-32B"

[trainer.loss]
teacher_tau = 0.3    # moderate teacher guidance
adv_tau = 1.0        # keep full reward signal
kl_tau = 0.01        # light KL regularization

[orchestrator.buffer]
# keep verification enabled -- we want rewards
```

This lets the student learn from both:
- Environment rewards (did the code produce the correct grid?)
- Teacher token-level guidance (would a stronger model have written similar code/reasoning?)

## Key Source Files

| File | Purpose |
|------|---------|
| `src/prime_rl/rl.py` | Top-level config (`RLConfig`), GPU allocation, process spawning |
| `src/prime_rl/trainer/rl/loss.py` | Loss function with `teacher_tau`, `adv_tau`, `teacher_kl` |
| `src/prime_rl/trainer/rl/config.py` | `LossConfig` with tau parameters |
| `src/prime_rl/orchestrator/utils.py` | `compute_teacher_logprobs()` via vLLM prefill |
| `src/prime_rl/orchestrator/config.py` | `TeacherModelConfig`, `BufferConfig.skip_verification` |
| `src/prime_rl/orchestrator/orchestrator.py` | Teacher pool setup, logprob attachment to training samples |
| `src/prime_rl/orchestrator/trajectories.py` | `TrainingSample` creation with `teacher_logprobs=None` |
| `docs/on_policy_distillation.md` | Official documentation |

## Summary

Prime-rl's OPD is a well-designed, modular feature that:
1. **Supports hybrid mode natively** -- just set both `teacher_tau` and `adv_tau` > 0
2. **Handles GPU allocation automatically** via `teacher_gpu_ids`
3. **Uses prefill scoring** (not generation) for teacher logprobs, making it efficient
4. **Does NOT support privileged-info self-distillation** out of the box -- the teacher sees the same tokens as the student
5. **Requires vLLM** for the teacher server due to the `/chat/completions/tokens` endpoint dependency

For our ARC-AGI project, the most actionable path is hybrid mode with a larger teacher model (e.g., Qwen3-32B teaching Qwen3-8B), either on local GPUs or via an external server. Privileged-info self-distillation would require custom modifications to the orchestrator's teacher logprob computation pipeline.
