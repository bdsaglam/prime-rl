# prime-rl Training Management Guide

Practical guide for **actively managing** RL training runs with prime-rl — monitoring metrics, diagnosing crashes, fixing configs, recovering from failures, and re-launching until training succeeds.

Works for any task (math, code, ARC-AGI, reasoning), any model (Qwen, Llama, DeepSeek, etc.), and any training mode (pure RL, OPD, hybrid).

Adapted from [Weyaxi's GRPO+LoRA Engineering Handbook](https://huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl) with insights mapped to prime-rl's architecture and metric names.

**Battle-tested across multiple runs:**
- Frozen policy diagnosed (48 steps wasted — LR too low + 95% truncation)
- Teacher OOM, max_model_len mismatch, config propagation bugs fixed
- Qwen3 thinking mode, teacher GPU allocation, per-turn vs seq-level truncation resolved
- Healthy training achieved (mismatch_kl rising, MFU 64-69%)

---

## 1. Architecture Overview

prime-rl uses an async multi-process pipeline:

```
┌──────────────────────────────────────────────┐
│           RL LAUNCHER (uv run rl @)          │
│          Spawns & monitors processes         │
└──────────┬───────────────────────────────────┘
           │
   ┌───────┼────────┬──────────────────┐
   │       │        │                  │
   v       v        v                  v
[INFERENCE] [TRAINER] [ORCHESTRATOR] [TEACHER_INF]
  (vLLM)   (torchrun)  (rollouts)    (optional)
```

- **Inference server**: vLLM, serves student model, accepts weight updates
- **Orchestrator**: generates rollouts via environments, computes rewards, feeds trainer
- **Trainer**: consumes rollouts, computes RL/OPD loss, updates weights
- **Teacher inference** (optional): separate vLLM for teacher model (OPD)

All launched by a single command:
```bash
uv run rl @ configs/your_task/your_config.toml
```

---

## 2. Critical Metrics Dashboard

When a run starts, open these W&B panels side by side.

### Tier 1: Check Every 10 Minutes

| Metric | Where | Healthy Range | What It Tells You |
|--------|-------|---------------|-------------------|
| `reward/mean` | orchestrator | Rising, then plateau | Is the model learning at all? |
| `is_truncated/mean` | orchestrator | < 0.2 | Are rollouts completing? **If high, nothing else matters.** |
| `loss/mean` | trainer | Moving away from 0 (can be negative) | Is optimization working? |
| `mismatch_kl/mean` | trainer | 0.001 – 0.05 | Has the policy moved at all? < 0.001 = frozen policy |
| `teacher_kl/mean` | trainer | Negative and declining | Is distillation working? (OPD only) |
| `entropy/mean` | trainer | 0.3 – 2.0 for instruct models | Exploration vs collapse |
| `optim/grad_norm` | trainer | Stable, no spikes | Numerical stability |

### Tier 2: Check Every 30 Minutes

| Metric | Where | What It Tells You |
|--------|-------|-------------------|
| `trainer_probs/mean` vs `inference_probs/mean` | trainer | Gap = policy updating. If gap < 0.001, policy is frozen. |
| `time/wait_for_batch` | trainer | Pipeline sync health. Spikes = permanent throughput loss. |
| `perf/mfu` | trainer | GPU efficiency. Should stabilize at 60-70%. |
| `batch/solve_none` | orchestrator | Fraction of unsolvable tasks. If rising, model is regressing. |
| `batch/solve_all` | orchestrator | Fraction of trivially solved tasks. If rising, tasks too easy. |
| `batch/effective_batch_size` | orchestrator | `1 - solve_none - solve_all`. Maximize this. |
| `decode_len/mean` | orchestrator | Response length dynamics (see Section 5). |
| `num_turns/mean` | orchestrator | Multi-turn interaction depth (if applicable). |
| `metrics/*` | orchestrator | Task-specific metrics (exact_match, format_reward, etc.). |

### Tier 3: Periodic Health Checks

| Metric | Where | What It Tells You |
|--------|-------|-------------------|
| `perf/throughput` | trainer | Tokens/sec — watch for degradation |
| `perf/peak_memory` | trainer | Memory pressure (GiB) |
| `system/ckpt_disk_free_ratio` | trainer | Disk space — if < 0.15, crash at next checkpoint |
| `time/teacher_logprobs` | orchestrator | Teacher inference bottleneck (OPD only) |
| `time/generate_completions` | orchestrator | Generation time — varies with decode length |
| `error/mean` | orchestrator | Environment errors (timeouts, crashes) |
| `filter/total_detected_rate` | orchestrator | Gibberish/repetition filtering |
| `event_loop_lag/p90` | orchestrator | Orchestrator health |

---

## 3. Training Phases: What to Expect

### Phase A: Cold Start (Steps 0–20)

```
reward/mean:       Task-dependent baseline (partial credit if available)
entropy/mean:      Instruct models start LOW (~0.35–0.45), not high
mismatch_kl/mean:  ~0.0005 → should be rising by step 10
teacher_kl/mean:   Negative (if OPD) — see Section 4
grad_norm:         Task/LR dependent, should be stable
MFU:               Should stabilize at 60-70%
```

**By step 10, check the vitals:**
1. `is_truncated/mean` — if > 0.3, **stop and fix sequence lengths first**
2. `mismatch_kl/mean` — if < 0.001 after 10 steps, LR is too low
3. `trainer_probs/mean` vs `inference_probs/mean` — gap < 0.001 = policy frozen
4. If using teacher: tail `teacher_inference.stdout` for errors (OOM kills runs silently)

**Red flags:**
- `is_truncated/mean` > 0.5 → Fix `max_turns`, `max_tokens`, or `seq_len` before continuing
- `error/mean` high → Environment crashing. Check `outputs/logs/orchestrator.stdout`
- `mismatch_kl/mean` ≈ 0 after 20 steps → Increase LR by 5–10x

### Phase B: Rapid Learning (Steps 20–100)

*Only reachable if Phase A vitals pass.*

```
reward/mean:       Rising
entropy/mean:      May decline slightly or stay stable
mismatch_kl/mean:  Rising to 0.005–0.02 (policy exploring)
teacher_kl/mean:   Becoming more negative (student approaching teacher)
batch/solve_none:  Declining (model solving more tasks)
```

**What to do:** This is the productive phase. Monitor but don't intervene.

**Red flag:** Reward flat after 50 steps but mismatch_kl IS rising → model is changing but not improving. Consider reducing LR or checking reward function.

### Phase C: Plateau (Steps 100–300+)

```
reward/mean:       Flattening
entropy/mean:      Stable
mismatch_kl/mean:  Still rising slowly
teacher_kl/mean:   Flat (student learned what it can from teacher)
```

**What to do:** Check eval metrics — if plateaued for 50+ steps, consider early stopping. Most RL runs converge in 3–5 effective epochs (~100–200 steps depending on batch size).

---

## 4. OPD-Specific Metrics

If using On-Policy Distillation (teacher model), the loss has two components:

```
total_loss = adv_tau * GRPO_advantage_loss + teacher_tau * teacher_KL_loss
```

### `teacher_kl/mean` — The Distillation Signal

This is `mean(teacher_logprobs - student_logprobs)` across tokens. **It is typically NEGATIVE** because the teacher (usually larger) is more confident on the student's chosen tokens.

| Pattern | Meaning | Action |
|---------|---------|--------|
| Negative, becoming more negative | Student approaching teacher | Healthy |
| Negative, flat | Student not learning from teacher | Check LR, truncation |
| Becoming less negative (toward 0) | Student drifting from teacher | Increase `teacher_tau` |
| Near 0 or positive | Teacher logprobs may be broken | Check teacher server health |
| Not appearing | Teacher not configured | Check `num_teacher_gpus > 0` and `teacher_tau > 0` |

### `mismatch_kl/mean` — Policy Drift

KL between current policy and inference reference. Measures how much policy changed per step.

| Range | Meaning | Action |
|-------|---------|--------|
| 0.001 – 0.01 | Controlled updates | Healthy |
| 0.01 – 0.05 | Significant updates | Monitor |
| > 0.1 | Dangerous drift | Reduce LR |
| < 0.001 | Policy barely updating | **LR too low — increase 5-10x** |

**Key diagnostic:** Compare `trainer_probs/mean` vs `inference_probs/mean`. Gap < 0.001 means the policy is frozen regardless of other metrics.

### Importance Ratio Masking

- `is_masked/mean` — Fraction of tokens masked. If > 50%, losing most signal.
- `is_masked_high/mean` — Policy too confident vs reference
- `is_masked_low/mean` — Policy abandoned reference behavior
- `sequence_masked_high/mean`, `sequence_masked_low/mean` — Entire sequences discarded

If masking is aggressive: effective batch shrinks → noisy training. Reduce LR or checkpoint more frequently.

---

## 5. Response Length and Truncation

**Truncation is the #1 failure mode.** Track `decode_len/mean`, `num_turns/mean`, and `is_truncated/mean` together.

### Two Types of Truncation (OPPOSITE fixes!)

`is_truncated` is set when ANY single turn hits `finish_reason == "length"`. Even one truncated turn out of many flags the whole rollout.

| Type | Symptom | Cause | Fix |
|------|---------|-------|-----|
| **Seq-level** | Total sequence hits `seq_len`, long decode_len | Too many turns × tokens | Reduce `max_turns`, reduce `max_tokens` |
| **Per-turn** | Model can't finish output in one turn, reward ≈ 0 | `max_tokens` too low | **Increase** `max_tokens`, reduce `max_turns` |

**Diagnosis:** Compare `decode_len/mean` with `max_tokens × max_turns`. If decode_len is far below theoretical max but is_truncated is high → per-turn truncation (increase `max_tokens`). If decode_len is near `seq_len` → seq-level truncation (reduce turns/tokens).

**Length dynamics as diagnostic:**

| Pattern | Meaning |
|---------|---------|
| Length decreasing, reward rising | Model learning efficiency — good |
| Length decreasing, reward flat | Model finding shortcuts — check reward function |
| Length increasing, reward rising | Model reasoning deeper — watch truncation |
| Length increasing, reward flat | Verbose padding — consider KL budget |

### Eval Truncation ≠ Training Truncation

Eval often uses fewer turns, shorter sequences, and different sampling parameters. Eval can show 0% truncation while training is 95%+ truncated in the same run. Always check both independently.

---

## 6. Entropy: The Underrated Metric

| entropy/mean | Meaning | Risk |
|-------------|---------|------|
| > 2.0 | Very uncertain | Unusual for instruct models — check model loaded correctly |
| 0.5 – 2.0 | Good balance | Ideal range |
| 0.3 – 0.5 | Confident (typical for instruct models) | Fine if learning |
| < 0.2 | Possible mode collapse | Check outputs for repetition |
| Near 0 | Mode collapse | Stop, roll back |

Monitor `kl_ent_ratio/mean` — ratio of mismatch KL to entropy. If approaching 1.0, policy drift is dominating the output distribution. Stop or reduce LR.

---

## 7. Batch Quality Metrics

### `batch/effective_batch_size`

```
effective = 1 - solve_none - solve_all
```

| Pattern | Meaning | Action |
|---------|---------|--------|
| `solve_none` > 0.5 | Tasks too hard | Easier dataset, curriculum, or more rollouts |
| `solve_all` > 0.3 | Tasks too easy | Harder tasks, fewer rollouts |
| Both low | Good reward spread | Ideal — maximum learning signal |

### `batch/off_policy_level/mean`

How stale are the rollouts being trained on?
- 0 = perfectly on-policy (ideal)
- 1–2 = slightly stale (acceptable)
- 3+ = significantly off-policy → check orchestrator bottleneck

---

## 8. Common Failure Modes & Recovery

### Problem: Everything Flat — Policy Frozen

**Symptoms:** Reward oscillates but no trend after 30+ steps. `mismatch_kl` < 0.001. `trainer_probs` ≈ `inference_probs`. All metrics "stable."

**This is the most insidious failure** — stable ≠ learning.

**Cause:** Insufficient gradient signal. Usually:
1. LR too low for the model/LoRA rank/loss magnitude
2. Truncation killing reward signal → near-zero GRPO advantage

**Recovery:**
1. Fix truncation first (Section 5)
2. Increase LR by 5–10x
3. Both changes simultaneously are fine — they address independent problems
4. No need to roll back — policy hasn't diverged

### Problem: Reward Stuck at Zero

**Symptoms:** `reward/mean` ≈ 0 for 50+ steps, `batch/solve_none` ≈ 1.0

**Diagnosis:**
1. `error/mean` high → Fix environment errors
2. `decode_len/mean` near 0 → Model collapsed
3. Check inference logs — is vLLM healthy?
4. Run manual eval with known-good prompt

**Recovery:** Roll back, lower LR, increase temperature.

### Problem: KL Divergence Exploding

**Symptoms:** `mismatch_kl` > 0.1, `is_masked/mean` > 0.5, volatile reward

**Recovery:**
1. Lower LR
2. Decrease `teacher_tau`
3. Increase `rollouts_per_example`
4. Restart from checkpoint where KL was reasonable

### Problem: Entropy Collapse

**Symptoms:** `entropy/mean` → 0, identical outputs across rollouts

**Recovery:**
1. Increase temperature (try 1.1–1.2)
2. Increase LoRA dropout
3. Reduce `adv_tau` relative to `teacher_tau`
4. Roll back to checkpoint before collapse

### Problem: Thinking Mode Inflating Sequences (Qwen3)

**Symptoms:** `is_truncated` near 1.0 despite adequate `max_tokens`. Sequences are long but actual output is short.

**Root cause:** Qwen3 generates `<think>...</think>` blocks by default, consuming most of the token budget before producing actual content.

**Fix:** Disable thinking in both student sampling and teacher prompt tokenization:
```toml
[orchestrator.sampling.extra_body]
chat_template_kwargs = { enable_thinking = false }

[orchestrator.eval.sampling.extra_body]
chat_template_kwargs = { enable_thinking = false }

[orchestrator.teacher_model.chat_template_kwargs]
enable_thinking = false
```

Note: `reasoning_parser` only controls how thinking tokens are *parsed* — the model generates them regardless. You must disable via chat_template_kwargs.

### Problem: Teacher OOM During log_softmax

**Symptoms:** Teacher server crashes mid-step. Log shows `torch.OutOfMemoryError` during `log_softmax`. Orchestrator fails with connection error.

**Root cause:** `gpu_memory_utilization` allocates too much to KV cache, leaving nothing for activation tensors. `log_softmax` on the full vocab tensor needs ~1.2 GiB.

**Fix:** Reduce `gpu_memory_utilization` from 0.95 to 0.88. Note: reducing `max_model_len` alone does NOT free pre-allocated KV cache.

### Problem: Teacher max_model_len Mismatch

**Symptoms:** `openai.BadRequestError: max_tokens must be at least 1, got -XXXX`

**Root cause:** Orchestrator uses `config.seq_len` for truncation instead of teacher's actual `max_model_len`. Sequences between teacher's limit and seq_len get rejected.

**Fix:** Ensure `[orchestrator.teacher_model]` has `max_model_len` set to match `[teacher_inference.model]`. Verify with `--dump-config`.

### Problem: Teacher Context Window Too Small

**Symptoms:** Teacher logprobs increasingly padded with 0.0 as sequences grow. Mismatch_kl peaks then declines.

**Root cause:** Teacher `max_model_len` < student `seq_len`. Truncation logic pads with zeros.

**Fix:** Use more GPUs for teacher (TP=2) to fit larger context:
```toml
[deployment]
num_infer_gpus = 1      # down from 2
num_teacher_gpus = 2    # up from 1

[teacher_inference.model]
max_model_len = 32768   # match student seq_len
```

Slower throughput but correct training signal. **Slower is better than wrong.**

### Problem: Disk Full

**Symptoms:** "No space left on device" at checkpoint save

**Prevention:** Watch `system/ckpt_disk_free_ratio` — alert at < 0.15.

**Recovery:** Delete oldest checkpoints, reduce `ckpt.interval`, restart.

### Problem: Throughput Degradation

**Symptoms:** `perf/mfu` dropping, `time/step` increasing

**Diagnosis:**
1. `time/wait_for_batch` spiking → Trainer idle, orchestrator behind
2. `time/teacher_logprobs` growing → Teacher bottleneck (longer sequences)
3. `time/generate_completions` growing → Student inference slowing
4. `event_loop_lag/p90` > 1s → Orchestrator saturated
5. `batch/off_policy_level/mean` rising → Batches stale

---

## 9. Active Training Management

### 9.0 tmux Session Convention

Organize training into a tmux session with named windows:

| Window | Name | Contents |
|--------|------|----------|
| 0 | `train` | Main training process (`uv run rl @ ...`) |
| 1 | `orchestrator` | `tail -F outputs/logs/orchestrator.stdout` |
| 2 | `trainer` | `tail -F outputs/logs/trainer.stdout` |
| 3 | `teacher` | `tail -F outputs/logs/teacher_inference.stdout` |
| 4 | `inference` | `tail -F outputs/logs/inference.stdout` |
| 5 | `gpu` | `watch -n 5 nvidia-smi` |
| 6 | `disk` | `watch -n 60 'df -h / \| tail -1'` |

**Quick reference:**
```bash
tmux switch -t train            # Switch to training session
Ctrl-B w                        # List all windows
Ctrl-B 0–6                      # Jump to window
tmux capture-pane -t train:trainer -p -S -50  # Read logs programmatically
```

### 9.1 Pre-Launch Checklist

```bash
# 1. Kill zombie vLLM processes from previous crashed runs
# vLLM spawns child processes that persist after parent dies
fuser /dev/nvidia* 2>/dev/null  # List PIDs using GPUs
kill $(fuser /dev/nvidia* 2>/dev/null | tr -s ' ')
nvidia-smi  # Verify GPUs clean (0 MiB used)

# 2. Disk space (need >100GB free for checkpoints)
df -h /

# 3. Validate config
uv run rl @ configs/your_config.toml --dump-config /tmp/check
# Inspect resolved config:
#   - teacher_model.max_model_len matches teacher_inference.model.max_model_len
#   - For Qwen3: chat_template_kwargs has enable_thinking=false
#   - seq_len, max_tokens, max_turns look reasonable

# 4. Test environment loads
uv run python -c "import verifiers as vf; env = vf.load_environment('your-env', ...); print('OK')"

# 5. For re-launches: clean previous output
uv run rl @ configs/your_config.toml --clean-output-dir
```

**Critical:** Step 1 is essential. Without it, the new run OOMs because zombie processes hold GPU memory. Step 5 is needed on every re-launch after a crash.

### 9.2 Launching Training

```bash
# Create tmux session with training in window 0
tmux new-session -d -s train -n train \
  'uv run rl @ configs/your_config.toml; echo "=== EXITED (code: $?) ==="'

# Add monitoring windows (wait for log files)
tmux new-window -t train -n orchestrator 'sleep 5 && tail -F outputs/logs/orchestrator.stdout'
tmux new-window -t train -n trainer      'sleep 5 && tail -F outputs/logs/trainer.stdout'
tmux new-window -t train -n teacher      'sleep 5 && tail -F outputs/logs/teacher_inference.stdout'
tmux new-window -t train -n inference    'sleep 5 && tail -F outputs/logs/inference.stdout'
tmux new-window -t train -n gpu          'watch -n 5 nvidia-smi'
tmux new-window -t train -n disk         'watch -n 60 "df -h / | tail -1"'
tmux select-window -t train:train
```

### 9.3 Monitoring Cadence

- **Every 10 min:** Tier 1 metrics (reward, truncation, mismatch_kl)
- **Every 30 min:** All log panes, check for errors
- **Every hour:** Full health check — all tiers, GPU memory, disk space
- **No new step in 30 min:** Investigate immediately — process may be hung

### 9.4 Crash Recovery

When a crash is detected:

**Step 1 — Diagnose:** Read last ~100 lines from each log pane.

**Step 2 — Clean up:**
```bash
tmux kill-session -t train
kill $(fuser /dev/nvidia* 2>/dev/null | tr -s ' ')
sleep 2 && nvidia-smi  # Verify clean
```

**Step 3 — Fix and re-launch.** Diagnose the crash (see Section 8), apply fix, re-create tmux session.

**Key insight:** When one component (e.g., teacher) dies, others don't crash — they hang waiting forever. A "stuck" run isn't slow, it's dead. If no new step in >30 minutes, check logs.

---

## 10. Hyperparameter Tuning Decision Tree

### FIRST: Check the Fundamentals

```
is_truncated/mean > 0.3?
└── YES → STOP. Fix truncation before anything else.
    ├── decode_len near seq_len? → Seq-level truncation
    │   ├── Reduce max_turns
    │   └── Reduce max_tokens if per-turn output excessive
    ├── reward ≈ 0 but decode_len moderate? → Per-turn truncation
    │   ├── INCREASE max_tokens
    │   └── Reduce max_turns to compensate
    ├── Qwen3 thinking mode enabled? → Disable it
    │   └── chat_template_kwargs = { enable_thinking = false }
    └── Nothing else matters until rollouts complete.

mismatch_kl/mean < 0.001 after 20 steps?
└── YES → Policy frozen. LR too low.
    ├── Check trainer_probs vs inference_probs gap (< 0.001 = frozen)
    ├── Increase LR by 5-10x
    └── No need to roll back — policy hasn't diverged
```

### When Reward is Rising but Slowly

```
reward/mean rising < 0.01/step?
├── is_truncated/mean > 0.3? → Fix sequence lengths
├── batch/solve_none > 0.5? → Tasks too hard → easier dataset or curriculum
├── batch/solve_all > 0.3? → Tasks too easy → harder tasks
├── entropy/mean > 2.0? → Too much exploration → lower temperature
└── teacher_kl/mean flat? → Teacher not helping → check teacher, increase teacher_tau
```

### When Reward is Flat

```
reward/mean oscillating, no trend for 30+ steps?
├── mismatch_kl < 0.001? → Policy frozen → increase LR 5-10x
├── mismatch_kl rising but reward flat? → Model changing, not improving
│   → Reduce LR, check truncation
├── entropy < 0.2? → Mode collapse → see Section 8
└── eval metric also flat? → Training converged (or never started)
```

### When Reward Plateaus

```
reward/mean was rising, now flat 50+ steps?
├── eval metric also flat? → Converged. Stop or change approach.
│   └── eval declining? → Overfitting → more regularization, stop
├── mismatch_kl still rising? → Model changing, not improving → reduce LR
└── entropy < 0.2? → Mode collapse → see Section 8
```

### Adjusting teacher_tau vs adv_tau (OPD)

| Situation | Adjust |
|-----------|--------|
| Teacher KL declining but reward flat | Increase `adv_tau` (more RL signal) |
| Reward rising but teacher KL rising | Increase `teacher_tau` (more anchoring) |
| Both flat | Check if both signals flowing |
| Both declining | Healthy — don't touch |

---

## 11. W&B Dashboard Layout

**Row 1 — Big Picture:** `reward/mean`, eval metric, `loss/mean`

**Row 2 — Vitals:** `is_truncated/mean`, `mismatch_kl/mean`, `trainer_probs/mean` + `inference_probs/mean`

**Row 3 — OPD Health:** `teacher_kl/mean`, `entropy/mean`, `kl_ent_ratio/mean`

**Row 4 — Task-Specific:** Environment-dependent metrics (`metrics/*`)

**Row 5 — Batch Quality:** `batch/effective_batch_size`, `batch/solve_none` + `batch/solve_all`, `decode_len/mean`

**Row 6 — System:** `perf/mfu`, `time/wait_for_batch`, `optim/grad_norm`, `system/ckpt_disk_free_ratio`

---

## 12. Key Lessons

1. **Check fundamentals in first 10 steps.** Truncation and mismatch_kl are the two vitals. If either is broken, stop and fix.

1b. **Never let GPUs sit idle between experiments.** When a training run finishes, immediately launch the next planned experiment. Have the next config ready before the current run completes. GPU time is the bottleneck — every minute of idle GPUs is wasted compute.

2. **"Stable metrics" can mean "nothing is happening."** Flat ≠ healthy. The key tell: `trainer_probs` ≈ `inference_probs`.

3. **Truncation + OPD = vicious cycle.** Truncated rollouts → low rewards → near-zero advantage → teacher term dominates on truncated sequences → weak signal → no learning. Fix truncation first, always.

4. **`teacher_kl/mean` is negative, not positive.** More negative = student approaching teacher = healthy.

5. **Instruct models start with low entropy (~0.35–0.45).** Collapse threshold for instruct models is ~0.2, not ~0.5.

6. **`time/wait_for_batch` spikes permanently degrade throughput.** A single stall can desynchronize the pipeline and halve MFU for the rest of the run.

7. **Expect multiple crashes before a run succeeds.** Budget time for iteration. Each crash reveals a different issue.

8. **Zombie vLLM processes are the #1 re-launch blocker.** `fuser /dev/nvidia*` before every re-launch. Without it, immediate OOM.

9. **`gpu_memory_utilization` controls KV cache, not just memory.** Reducing it frees space for activation tensors. `max_model_len` alone does NOT free pre-allocated KV cache.

10. **Config propagation bugs are silent.** Always `--dump-config` and verify the resolved values.

11. **Eval truncation ≠ training truncation.** Different sampling params → different behavior. Check both.

12. **Per-turn truncation ≠ seq-level truncation.** One needs `max_tokens` increased, the other decreased. Diagnose which type before acting.

13. **A hung process looks like slow training, not a crash.** When one component dies, others wait forever. If no new step in 30 minutes, check logs.

14. **`--clean-output-dir` on every re-launch.** Without it: `FileExistsError` on startup.

15. **Slower is better than wrong.** More teacher GPUs (TP=2) halves inference throughput but gives correct teacher signal. Throughput loss is recoverable; wrong training dynamics are not.

16. **Early stopping is underrated.** Most gains happen in the first 30% of training. When eval is flat for 50+ steps, you're burning compute.

17. **Effective batch size is the real batch size.** If `solve_none = 0.7`, 70% of examples produce zero GRPO signal. Only examples with reward variance contribute to learning.

---

## Appendix: Config Quick Reference

### Minimal Config Template

```toml
max_steps = 200
seq_len = 4096

[wandb]
project = "my-project"
name = "experiment-1"

[model]
name = "Qwen/Qwen3-8B"

[deployment]
num_train_gpus = 1
num_infer_gpus = 2

[orchestrator]
batch_size = 256
rollouts_per_example = 8

[[orchestrator.env]]
id = "math-env"
args = { dataset_name = "hendrycks/math", max_turns = 1 }

[orchestrator.sampling]
temperature = 1.0
max_tokens = 2048

[trainer.loss]
adv_tau = 1.0
# teacher_tau = 0.3  # Uncomment for OPD

[trainer.optim]
lr = 5e-6

[trainer.model.lora]
rank = 32
alpha = 32
```

### Adding OPD (Teacher Distillation)

```toml
[deployment]
num_teacher_gpus = 1  # or 2 for TP=2

[teacher_inference]
gpu_memory_utilization = 0.88

[teacher_inference.model]
name = "Qwen/Qwen3-32B"
max_model_len = 32768
dtype = "bfloat16"

[trainer.loss]
teacher_tau = 0.3
adv_tau = 1.0
```

### Adding Eval

```toml
[orchestrator.eval]
interval = 10  # Every 10 steps

[[orchestrator.eval.env]]
id = "math-env"
args = { dataset_name = "hendrycks/math", split = "test" }

[orchestrator.eval.sampling]
temperature = 0.0
max_tokens = 2048
n = 4  # best-of-4 eval
```

### Key CLI Flags

```bash
uv run rl @ config.toml                    # Normal launch
uv run rl @ config.toml --dump-config /tmp  # Validate config without running
uv run rl @ config.toml --clean-output-dir  # Clean previous checkpoints
uv run rl @ config.toml --dry-run           # Validate everything without training
```

---

## Appendix: All Metric Names

<details>
<summary>Trainer Metrics</summary>

```
loss/mean, loss/std, loss/min, loss/max
entropy/mean, entropy/std, entropy/min, entropy/max
mismatch_kl/mean, mismatch_kl/std, mismatch_kl/min, mismatch_kl/max
masked_mismatch_kl/mean
unmasked_mismatch_kl/mean
teacher_kl/mean, teacher_kl/std, teacher_kl/min, teacher_kl/max
is_masked/mean, is_masked_low/mean, is_masked_high/mean
sequence_masked_low/mean, sequence_masked_high/mean
geo_masked_low/mean, geo_masked_high/mean, geo_seq_ratio/mean
trainer_probs/mean, inference_probs/mean
kl_ent_ratio/mean
optim/lr, optim/grad_norm
perf/throughput, perf/throughput_per_gpu, perf/mfu, perf/peak_memory
time/step, time/forward_backward, time/wait_for_batch, time/load_data
time/broadcast_weights, time/save_ckpt
system/ckpt_disk_free_gib, system/ckpt_disk_free_ratio
```
</details>

<details>
<summary>Orchestrator Metrics</summary>

```
reward/mean, reward/std, reward/min, reward/max, reward/median
val_reward/mean, val_reward/std, val_reward/min, val_reward/max
batch/solve_none, batch/solve_all, batch/effective_batch_size
batch/off_policy_level/mean, batch/cancelled_rollouts
batch/inflight_rollouts, batch/inflight_samples, batch/async_level
seq_len/mean, prefill_len/mean, decode_len/mean
num_turns/mean, is_truncated/mean
sampling/temperature
error/mean, error/{type}
progress/tokens, progress/total_tokens, progress/samples
time/step, time/generate_completions, time/teacher_logprobs
time/save_ckpt, time/parallel_preprocess
event_loop_lag/p90, event_loop_lag/max
filter/total_detected_rate, filter/total_enforced_rate
pool/easy, pool/normal, pool/hard
metrics/{env_specific}
```
</details>
