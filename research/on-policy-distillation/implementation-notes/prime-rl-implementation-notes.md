# prime-rl OPD Implementation Notes

Lessons learned from implementing Phase 0 (ARC-AGI REPL + OPD) in prime-rl that were not covered in the original planning docs.

## Bugs & Fixes in prime-rl

### 1. Validator ordering: `teacher_tau` + `num_teacher_gpus`

**File:** `src/prime_rl/configs/rl.py:326-336`

`validate_teacher_model` runs before `auto_setup_teacher_inference` in Pydantic's validator chain. So a config with `teacher_tau > 0` and `deployment.num_teacher_gpus = 1` (but no explicit `orchestrator.teacher_model`) fails validation — even though `auto_setup_teacher_inference` would have auto-configured it later.

**Fix:** Added an early return in the validator when `num_teacher_gpus > 0` on single-node deployments:

```python
if self.deployment.type == "single_node" and (self.deployment.num_teacher_gpus or 0) > 0:
    return self
```

### 2. Teacher prefill crashes on long sequences (`max_tokens=0`)

**File:** `src/prime_rl/orchestrator/utils.py:146-189`

`compute_teacher_logprobs` sends the full token sequence (`prompt_ids + completion_ids`) as the "prompt" to the teacher's `/chat/completions/tokens` endpoint with `max_tokens=1`. When the sequence length >= `max_model_len`, vLLM computes `effective_max_tokens = max_model_len - prompt_len = 0` and returns a 400 error.

This is especially likely with a cross-model teacher (e.g., 32B teacher with `max_model_len=32768` scoring 8B student sequences that can also be up to 32768 tokens).

**Fix:** Added `max_model_len` parameter. Truncates token sequences to `max_model_len - 1` before sending, then pads returned logprobs with `0.0` to match the full sequence length. The 0.0 padding is neutral for the loss: `teacher_kl = teacher_logprobs - trainer_logprobs`, so 0.0 contributes no gradient.

The caller passes `config.seq_len` as the bound.

### 3. Teacher inference port conflict

When you explicitly set `[teacher_inference]` in the config (needed for a different teacher model), the auto port assignment (`inference.server.port + 1`) doesn't apply. You must manually set:

```toml
[teacher_inference.server]
port = 8900  # or any port != inference port
```

Without this, both servers try to use port 8000 and validation fails.

## Dependency & Config Setup

### uv local path dependencies

Can't use PEP 508 `file:./path` syntax in `[project.dependencies]` with uv. Instead:

```toml
# In [project.dependencies]:
"arc-agi",

# In [tool.uv.sources]:
arc-agi = { path = "environments/arc_agi", editable = true }
```

### `load_environment()` arg names

The correct kwarg is `eval_dataset`, not `eval_dataset_name`. The old 32B config in rlvr used the wrong name — it silently went into `**kwargs` and was ignored (eval fell back to the train dataset).

### Config directory convention

Real training configs go in `configs/` (alongside gsm8k, deepscaler, etc.), not `examples/`. The `examples/` directory is for demo setups.

## Cross-Model Teacher (32B teacher, 8B student)

### Memory

Qwen3-32B at bf16 is ~64GB. Fits on a single A100 80GB with `gpu_memory_utilization=0.90`. KV cache is limited (~20-40% utilization observed) but sufficient since the teacher only does prefill scoring, not autoregressive generation.

### Throughput

Teacher logprob computation for 128 samples: ~10 minutes at ~2500 tokens/s on 1 GPU. This is the main throughput bottleneck — slower than rollout generation. The teacher processes requests sequentially (1-2 concurrent) due to limited KV cache.

### Tokenizer compatibility

Qwen3-8B and Qwen3-32B share the same tokenizer, so token IDs from the student can be sent directly to the teacher. This would NOT work with a teacher from a different model family (e.g., DeepSeek).

## Training Observations (First Run)

### Step 0 eval results

Eval on 8 examples (4 rollouts each): `Avg@4=0.8380` — partial credit, not solve rate. Average completion length ~9270 tokens. No truncation. This is the base model's performance before any training.

### Async rollout reuse

Step 1 showed `Throughput: 710552535.3 tokens/s` — impossibly high, indicating it reused rollouts from step 0's buffer rather than generating new ones. This is expected prime-rl async behavior (`max_off_policy_steps=8` allows reuse).

### Repetition filter

By step 2, the repetition filter flagged 1/128 rollouts. The model occasionally falls into repetitive patterns in the REPL. The filter is non-enforcing by default (just logs).

## Config Template (Phase 0, 4 GPUs)

Key settings that differ from standard RL configs:

```toml
# OPD-specific
[deployment]
num_teacher_gpus = 1          # Allocates 1 GPU for teacher inference

[trainer.loss]
teacher_tau = 0.3             # Distillation strength
adv_tau = 1.0                 # RL reward signal (hybrid: both active)

# Teacher model (different from student)
[teacher_inference]
gpu_memory_utilization = 0.90

[teacher_inference.server]
port = 8900

[teacher_inference.model]
name = "willcb/Qwen3-32B"    # Larger teacher
max_model_len = 32768
dtype = "bfloat16"
```

Everything else (orchestrator, env, eval) is the same as a standard RL config.
