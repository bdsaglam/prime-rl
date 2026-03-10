# Phase 1.5b Handover — Hint-Free Curriculum Training

**Date:** 2026-03-06 22:30 UTC
**Last step completed:** Orchestrator step 148, Trainer step 148
**Training target:** Step 200
**Tmux session:** `opd` (window: `train`)
**Config:** `configs/arc_agi/opd-rl-qwen-8b-hint-curriculum-1.5b.toml`
**W&B:** project `arc-agi-opd`, run name `arc-agi-hint-curriculum-phase1.5b-*`

---

## What Is Running

Phase 1.5b training: hint-free fine-tuning of the Qwen3-8B student, resumed from Phase 1.5a checkpoint (step 60). The student does NOT see hints. The teacher (Qwen3-32B) still sees hints + ground-truth outputs as privileged information (PI) via `teacher_context`.

### Hardware Layout
- GPUs 0-1: Student inference (Qwen3-8B, DP=2)
- GPU 2: Trainer (LoRA r=32)
- GPU 3: Teacher inference (Qwen3-32B, bf16)

### Key Config Values
- `teacher_tau = 0.5` (distillation strength)
- `adv_tau = 1.0` (RL reward signal)
- `lr = 5e-6` in config **BUT actual LR = 1e-5** (checkpoint optimizer state overrides config — this is expected PyTorch behavior when resuming; see below)
- `max_steps = 200`
- `eval interval = 10`
- `ckpt interval = 10`
- `include_hint = false` in training env
- `include_hint = false` in eval env

### LR Override Issue
The config specifies `lr = 5e-6` but the actual training LR is `1e-5` because PyTorch's `optimizer.load_state_dict()` loads the checkpoint's optimizer state (which was 1e-5 from Phase 1.5a), overriding the config value. This is standard PyTorch behavior and NOT a bug. To actually change LR on resume, you'd need to either start fresh or manually modify the checkpoint. We decided to let it continue at 1e-5 since it was working fine.

---

## Training Progress Summary

### Eval Scores (Avg@4 — hint-free, 8 examples × 4 rollouts)

| Step | Avg@4 | Truncated | Notes |
|------|-------|-----------|-------|
| 0* | 0.2542 | 28% | Baseline (Phase 1.5a start, pre-training) |
| 70 | 0.1735 | 43.8% | First 1.5b eval (10 hint-free steps) |
| 80 | 0.1189 | 71.9% | Dip — model becoming verbose |
| 90 | 0.1169 | 81.2% | Bottom — 81% truncation |
| 100 | 0.1428 | 71.9% | Recovery begins |
| 110 | 0.1474 | 62.5% | Continuing recovery |
| 120 | 0.1638 | 53.1% | Good improvement |
| 130 | 0.1427 | 71.9% | Fluctuation |
| 140 | **0.1775** | 56.2% | **Best eval so far in 1.5b** |
| 150 | (pending) | | |

*Step 0 baseline is from Phase 1.5a's first eval (model before any hint training).*

### Training Reward Trend

| Step Range | Avg Reward | Notes |
|------------|-----------|-------|
| 60-69 | ~0.19 | First hint-free steps, model struggling |
| 70-79 | ~0.22 | Slight improvement |
| 80-89 | ~0.22 | Plateau |
| 90-99 | ~0.28 | Starting to climb |
| 100-109 | ~0.31 | Noticeable improvement |
| 110-119 | ~0.33 | Including 0.50 peaks |
| 120-129 | ~0.38 | Strong, with 0.47 peaks |
| 130-139 | **~0.48** | Peak period (0.67 max at step 138) |
| 140-148 | **~0.42** | Sustained high (0.56 at step 148) |

Training rewards show a clear upward trend from ~0.19 to ~0.45 average. The model IS learning to solve training puzzles without hints.

### Trainer Metrics (stable throughout)
- **Entropy:** 0.52 → 0.48 (slight decline, healthy range)
- **Mismatch KL:** 0.0002-0.0004 (very low, student-teacher aligned)
- **Grad Norm:** 0.017-0.028 (stable, no spikes)
- **Loss:** hovering near 0 (-0.0005 to 0.0005)
- **LR:** 1e-5 constant (see note above)

---

## Key Observations

1. **Training rewards are improving strongly** (0.19 → 0.45+ avg), but **eval improves slowly** (0.12 → 0.18). This gap suggests partial overfitting to training puzzles or high eval variance (only 8 examples).

2. **Truncation rate is the best health indicator.** When truncation is high (>70%), the model is being verbose and failing to converge. The best eval (step 140, 0.178) had 56% truncation. The worst (step 90, 0.117) had 81%.

3. **The U-shaped eval curve** (0.17 → 0.12 → 0.18) suggests the model initially got worse when hints were removed (steps 70-90), then recovered (steps 100-140). This is expected curriculum behavior.

4. **Sequence lengths declining** from ~14,000 to ~11,000 tokens/sample, indicating the model is becoming more efficient in its REPL interactions.

---

## What To Monitor

### Commands
```bash
# Check orchestrator steps
cat outputs/logs/orchestrator.stdout | sed 's/\x1b\[[0-9;]*m//g' | grep "SUCCESS.*Step" | tail -10

# Check trainer steps
cat outputs/logs/trainer.stdout | sed 's/\x1b\[[0-9;]*m//g' | grep "SUCCESS" | tail -5

# Check eval results
cat outputs/logs/orchestrator.stdout | sed 's/\x1b\[[0-9;]*m//g' | grep "Avg@"

# Check latest activity
tail -5 outputs/logs/orchestrator.stdout | sed 's/\x1b\[[0-9;]*m//g'

# GPU utilization
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

### What to watch for
- **Eval Avg@4:** Should continue recovering toward 0.20+. If it drops below 0.12, training may be degrading.
- **Truncation rate:** Should stay below 70%. If it goes above 80%, the model is becoming too verbose.
- **Entropy:** Should stay above 0.40. Below 0.35 indicates mode collapse risk.
- **Training reward:** Should stay above 0.30 on average. If it drops below 0.15, something is wrong.
- **Gibberish count:** Usually 1-3 per step. If it rises above 10, the model may be degenerating.
- **Crashes:** Check if tmux session `opd` is still alive. If training crashed, check the last log lines for errors.

### Remaining evals
- Step 150 (next eval)
- Step 160, 170, 180, 190, 200

### Checkpoints saved
Every 10 steps: step_60 through step_140 (plus step_150 when it saves). Located in `outputs/` directory.

---

## When Training Completes (Step 200)

The user's instruction was: *"I think you should periodically check training and if it's stuck, fix it. if it completes, start next training. there shouldn't be any end for your task as long as this thread is active."*

### Decision at step 200
Based on eval trends, decide whether to:

1. **Continue training** — If eval is still improving (Avg@4 trending up), extend max_steps by creating a new config with `resume_step = -1` and higher `max_steps`.

2. **Try a different approach** — If eval plateaued below 0.20, consider:
   - Lower learning rate (need fresh start since checkpoint overrides)
   - Different teacher_tau (try 0.7 or 1.0)
   - Different hint strategy

3. **Move to Phase 2** — If eval reaches or exceeds baseline (0.25+), the hint-free curriculum worked and we can proceed to the next phase in `tmp/on-policy-distillation/arc-agi-opd-plan.md`.

---

## File References

| File | Description |
|------|-------------|
| `configs/arc_agi/opd-rl-qwen-8b-hint-curriculum-1.5b.toml` | Phase 1.5b config (currently running) |
| `configs/arc_agi/opd-rl-qwen-8b-hint-curriculum.toml` | Phase 1.5a config (hint-assisted, completed) |
| `tmp/on-policy-distillation/phase1.5-hint-curriculum.md` | Phase 1.5 plan & results documentation |
| `tmp/on-policy-distillation/arc-agi-opd-plan.md` | Overall phased plan (Phase 0-3) |
| `environments/arc_agi/src/arc_agi/data.py` | Hint field in ArcTask, teacher_context |
| `environments/arc_agi/src/arc_agi/envs/repl.py` | include_hint flag, hint injection |
| `environments/arc_agi/src/arc_agi/env.py` | include_hint passthrough |

---

## Context from Phase 1.5a

Phase 1.5a (steps 0-60) trained WITH hints (`include_hint=true`). Key finding: **hint overfitting** — the model learned to depend on hints. Hint-free eval declined from 0.254 → 0.168 over 50 steps, even though hint-assisted training rewards were high (0.45-0.82).

Phase 1.5b was started from the step 60 checkpoint to wean the model off hints while keeping the teacher's privileged information (hints + ground truth) as a dense learning signal via OPD.
