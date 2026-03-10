# OPD Math Verification Experiment

**Goal:** Verify that our prime-rl OPD training pipeline actually works by testing it on a math benchmark where OPD is known to produce gains. If OPD works on math but not ARC-AGI, the problem is task difficulty. If OPD doesn't work on math either, our implementation is broken.

**Status:** Planning. Must run baseline eval before training.

---

## Motivation

We've run OPD training on ARC-AGI (Phase 1: 23 steps, Phase 1.5: 200 steps) with no clear improvement. Two possible explanations:

1. **ARC-AGI is too hard** for OPD at this scale (sparse rewards, multi-turn REPL, near-zero base success rate)
2. **Our OPD implementation/setup is broken** (loss function bug, teacher logprob pipeline issue, hyperparameter misconfiguration)

We can't distinguish these without testing on a benchmark where OPD is known to work. Multiple papers show OPD improving math performance with similar model sizes:

- **Thinking Machines blog:** Qwen3-8B + Qwen3-32B teacher, ~150 steps, 60% -> 74.4% AIME'24 (but started from 400K-prompt SFT checkpoint, so not cold-start)
- **OPSD (Zhao et al.):** Qwen3-8B self-distillation, up to 1500 steps, +0.9 avg over GRPO on AIME/HMMT (modest but consistent)
- **SDPO (Hubotter et al.):** Qwen3-8B self-distillation, ~6 hours on 4 GPUs, +7.6% on LiveCodeBench

Note: prime-rl's OPD feature (PR #1458) has **no public benchmark results** beyond a toy reverse-text test. We may be the first real users.

## Design Principles

- **Same models as ARC experiments.** Student: `willcb/Qwen3-8B`, Teacher: `willcb/Qwen3-32B`. Same LoRA rank, same training infrastructure. The only change is the environment.
- **Single-turn math.** Removes all multi-turn REPL complexity. Isolates the OPD mechanism.
- **Fast iteration.** 50-100 steps, 2-4 hours per run.
- **Baseline first.** Evaluate the pre-trained model on the eval set before any training. If baseline accuracy is >85%, the task is too easy and we need a harder subset.

## Step 0: Baseline Evaluation (MUST DO FIRST)

Before training, evaluate `willcb/Qwen3-8B` on the training and eval datasets to establish baselines and check difficulty.

### Start inference server

```bash
# Use all 4 GPUs for eval (we're not training yet)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-8B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 16384
```

Note: No `--reasoning-parser`, no `--enable-auto-tool-choice` needed for math. The math-env doesn't use tool calling. Check whether the model needs `enable_thinking` or not -- try both and see which gives better baseline.

### Evaluate on Hendrycks Math (training dataset)

```bash
# Sample of 50 problems from training set
uv run vf-eval math-env \
    -a '{"dataset_name": "PrimeIntellect/Hendrycks-Math", "dataset_subset": "default"}' \
    -m willcb/Qwen3-8B \
    -b http://0.0.0.0:8900/v1 \
    -n 50 \
    -t 4096
```

### Evaluate on Math500 (held-out eval)

```bash
uv run vf-eval math500 \
    -m willcb/Qwen3-8B \
    -b http://0.0.0.0:8900/v1 \
    -n 50 \
    -t 4096
```

### Evaluate on AIME 2024 (hard eval)

```bash
uv run vf-eval aime2024 \
    -m willcb/Qwen3-8B \
    -b http://0.0.0.0:8900/v1 \
    -t 4096
```

### Decision gate

| Hendrycks Math baseline | Action |
|------------------------|--------|
| < 50% | Good difficulty range. Proceed with Hendrycks Math. |
| 50-75% | Acceptable. Proceed. |
| 75-85% | Marginal. Consider filtering to harder subsets only. |
| > 85% | Too easy. Switch to AIME 2024 or Math500 as training set. |

Also check: does the 32B teacher score meaningfully higher than the 8B student? If not, the teacher signal will be weak. Evaluate `willcb/Qwen3-32B` on the same set (or look up known benchmarks for Qwen3-32B on Hendrycks Math).

### If `vf-eval` commands don't work

The eval tool names (`math-env`, `math500`, `aime2024`) may need to be prefixed with `primeintellect/`. Try:

```bash
uv run vf-eval primeintellect/math-env \
    -a '{"dataset_name": "PrimeIntellect/Hendrycks-Math", "dataset_subset": "default"}' \
    -m willcb/Qwen3-8B \
    -b http://0.0.0.0:8900/v1 \
    -n 50 \
    -t 4096
```

Or use `prime eval run` syntax:

```bash
prime eval run primeintellect/math-env \
    -x '{"dataset_name": "PrimeIntellect/Hendrycks-Math", "dataset_subset": "default"}' \
    -n 50 -r 1 \
    -m willcb/Qwen3-8B \
    -b http://0.0.0.0:8900/v1
```

## Step 1: Training Runs

Three runs, each 100 steps. Run sequentially (they share GPUs).

### GPU Allocation (all 3 runs)

```
GPU 0-1: Student inference (willcb/Qwen3-8B, DP=2)
GPU 2:   Trainer (LoRA r=32)
GPU 3:   Teacher inference (willcb/Qwen3-32B, bf16, gpu_mem_util=0.90)
```

For Run 1 (no teacher), GPU 3 is unused.

### Run 1: GRPO Baseline (no teacher)

Pure RL with reward signal only. This tells us: does standard GRPO work on this task with our setup?

Config: `configs/opd_math_verify/grpo-baseline.toml`

```toml
max_steps = 100
seq_len = 4096

[deployment]
num_train_gpus = 1
num_infer_gpus = 2

[model]
name = "willcb/Qwen3-8B"

[wandb]
project = "opd-math-verify"
name = "grpo-baseline"

# --- Trainer ---
[trainer.model.lora]
rank = 32
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

[trainer.model.ac]
freq = 1

[trainer.loss]
adv_tau = 1.0
teacher_tau = 0.0

[trainer.optim]
lr = 5e-5
weight_decay = 0.0

# --- Orchestrator ---
[orchestrator]
batch_size = 128
rollouts_per_example = 16
oversampling_factor = 2.0

[orchestrator.sampling]
max_tokens = 4096

[orchestrator.buffer]
easy_threshold = 1.0
hard_threshold = 0.0

[[orchestrator.env]]
id = "primeintellect/math-env"
name = "hendrycks-math"
args = { dataset_name = "PrimeIntellect/Hendrycks-Math", dataset_subset = "default", math_verify_max_workers = 128, math_verify_timeout = 60 }

[orchestrator.eval]
interval = 25
rollouts_per_example = 4

[[orchestrator.eval.env]]
id = "primeintellect/math500"
name = "math500"
num_examples = 30

# --- Inference ---
[inference]
gpu_memory_utilization = 0.90

[inference.model]
name = "willcb/Qwen3-8B"
max_model_len = 16384
dtype = "bfloat16"
```

### Run 2: Pure OPD (teacher only, no RL rewards)

Teacher provides the only learning signal. Tests: does the teacher KL mechanism work at all?

Config: `configs/opd_math_verify/opd-pure.toml`

Same as Run 1 but with these changes:

```toml
[deployment]
num_train_gpus = 1
num_infer_gpus = 2
num_teacher_gpus = 1

[wandb]
name = "opd-pure"

[trainer.loss]
adv_tau = 0.0       # no RL reward signal
teacher_tau = 1.0    # pure distillation

# Teacher inference
[teacher_inference]
gpu_memory_utilization = 0.90

[teacher_inference.model]
name = "willcb/Qwen3-32B"
max_model_len = 16384
dtype = "bfloat16"
```

### Run 3: Hybrid OPD + RL

Both teacher signal and rewards. Tests: does adding teacher to RL help?

Config: `configs/opd_math_verify/opd-hybrid.toml`

Same as Run 2 but with:

```toml
[wandb]
name = "opd-hybrid"

[trainer.loss]
adv_tau = 1.0
teacher_tau = 0.5
```

### Launch commands

```bash
# Kill any existing GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# Run 1 (no teacher, only needs 3 GPUs)
uv run rl @ configs/opd_math_verify/grpo-baseline.toml

# After Run 1 completes, kill processes and start Run 2
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
uv run rl @ configs/opd_math_verify/opd-pure.toml

# After Run 2 completes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
uv run rl @ configs/opd_math_verify/opd-hybrid.toml
```

### Monitoring

```bash
# Training progress
tail -F outputs/logs/orchestrator.stdout
tail -F outputs/logs/trainer.stdout

# Key metrics to watch in W&B (project: opd-math-verify)
# - reward/mean: should increase for Run 1 and Run 3
# - teacher_kl: should start positive and decrease for Run 2 and Run 3
# - mismatch_kl: policy divergence from inference snapshot
# - entropy: should decrease gradually, not collapse
# - is_masked: fraction of masked tokens (too high = training is stale)
```

## What Success Looks Like

| Run | Key Metric | Expected if OPD works | Expected if OPD broken |
|-----|-----------|----------------------|----------------------|
| 1 (GRPO) | reward/mean | Increases from baseline | May or may not work (tests RL, not OPD) |
| 2 (Pure OPD) | teacher_kl | Starts positive (~0.5-2.0), decreases toward 0 | Flat at ~0, erratic, or NaN |
| 2 (Pure OPD) | reward/mean | Increases (student learns from teacher) | Flat or decreasing |
| 3 (Hybrid) | reward/mean | >= Run 1 rewards, ideally better | Worse than Run 1 |
| 3 (Hybrid) | teacher_kl | Decreasing | Flat or erratic |

### Interpreting Results

| Outcome | Diagnosis | Next Step |
|---------|-----------|-----------|
| All 3 runs show improvement | Pipeline works. ARC-AGI is the hard part. | Return to ARC with confidence in the setup |
| Run 1 works, Runs 2-3 don't | OPD mechanism is broken | Debug loss function, teacher logprob pipeline |
| Run 2 works but Run 3 doesn't | Hybrid blending is wrong | Check adv_tau/teacher_tau interaction |
| None work | Broader training issue (LR, masking, model) | Debug training loop fundamentals |
| teacher_kl starts at ~0 | Teacher and student too similar | Check that teacher is actually Qwen3-32B, not 8B |
| teacher_kl stays high, never decreases | Student can't learn from teacher signal | Check loss gradient flow, masking |

## Hyperparameter Notes

These configs are deliberately close to our ARC setup to maximize the diagnostic value:

| Parameter | Value | Matches ARC? | Rationale |
|-----------|-------|-------------|-----------|
| Student model | willcb/Qwen3-8B | Yes | Same model |
| Teacher model | willcb/Qwen3-32B | Yes | Same teacher |
| LoRA rank | 32 | Yes | Same capacity |
| LR | 5e-5 | Yes | Same as Phase 1 ARC |
| teacher_tau | 0.5 (hybrid) / 1.0 (pure) | Close (0.5 in ARC) | Match ARC |
| batch_size | 128 | Yes | Same |
| rollouts_per_example | 16 | Yes | Same |
| seq_len | 4096 | No (ARC: 32768) | Math is shorter |
| max_tokens | 4096 | No (ARC: 2048) | Math needs longer responses |
| max_turns | 1 (implicit) | No (ARC: 10) | Single-turn math |

The key difference is `seq_len` and `max_tokens` — math solutions are shorter than ARC REPL sessions, so we use smaller values for memory efficiency. Everything else matches the ARC setup as closely as possible.

## Open Questions

1. **Does `willcb/Qwen3-8B` need `enable_thinking`?** Our ARC configs disable thinking (`enable_thinking = false`). For math, thinking mode might help. Try baseline eval with both settings. Training should probably match whatever gives a reasonable (not too high, not too low) baseline.

2. **Is Hendrycks Math the right difficulty?** If 8B already scores >80%, gains will be small and hard to measure. The baseline eval (Step 0) answers this.

3. **Should we use `orchestrator.buffer.skip_verification = true` for Run 2?** Pure distillation (adv_tau=0) doesn't need rewards, so skipping verification saves time. But keeping it on lets us track reward/mean as a diagnostic. Keep it on.

4. **Do we need `easy_threshold` / `hard_threshold`?** These filter out problems that are too easy or too hard for the student. The existing Hendrycks Math configs use `easy_threshold = 1.0, hard_threshold = 0.0` (filter out problems solved by all rollouts, keep problems solved by none). This is good — it focuses training on problems where there's variance.

## Files to Create

Before running, create the config files. The configs above are inline — extract them into:

```
configs/opd_math_verify/
    grpo-baseline.toml
    opd-pure.toml
    opd-hybrid.toml
```

These can share most content. The only differences are `wandb.name`, `trainer.loss` settings, `[deployment]` (teacher GPUs), and `[teacher_inference]` section.
