# Multi-Turn Reflection OPD Experiments

**Goal:** Train with reflection-in-sequence: student solves → gets feedback → reflects → teacher scores full sequence with PI.

**Date:** 2026-03-14 to 2026-03-16

---

## Environment: `aime_mt`

Multi-turn AIME environment (`environments/aime_mt/src/aime_mt/env.py`):
- **Turn 1:** Student solves the problem (solution with `\boxed{}`)
- **Turn 2:** Student reflects on its solution after receiving feedback (correct/incorrect)
- **Teacher PI:** Answer + student's reflection + correct sibling solution (if available)
- **Reward:** Extracted from Turn 1 (solution), NOT Turn 2 (reflection)

### Bug Fixes During These Experiments

1. **`env_ids` shadowing bug (CRITICAL):** `orchestrator.py` line 280 overwrote train `env_ids` with eval env IDs, causing `prepare_teacher_context` to never be called for `aime_mt`. Fixed by renaming to `eval_env_ids`. This bug affected v1-v3 runs.

2. **Reward extraction bug:** Default `MathRubric` parser extracts `\boxed{}` from the *last* assistant message (reflection turn), which has no boxed answer. Created `SolutionTurnParser` to extract from the *first* assistant message (solution turn). Fixed for v5+.

---

## Experiment v4: Pure OPD, 8B Self-Teacher, Structured Reflection

**Config:** `configs/aime_mt/reflection-self-teacher-8b.toml`

| Parameter | Value |
|-----------|-------|
| Model | willcb/Qwen3-8B (self-teacher) |
| Loss | Pure OPD (adv_tau=0, teacher_tau=1) |
| LR | 1e-5 |
| Batch size | 32 |
| Max steps | 20 |
| Train set | AIME 2025 problems 0-21 (22 problems) |
| Eval set | AIME 2025 problems 22-29 (8 heldout) |
| Reflection | Structured format (VERDICT/ERROR_TYPE/etc.) |
| Student PI | Binary (told correct/incorrect) |

### Results

**Eval (single-turn AIME, no reflection):**

| Step | Heldout Avg@4 | Heldout Pass@4 | Train Avg@4 | Train Pass@4 |
|------|--------------|----------------|-------------|--------------|
| 10 (base) | 0.688 | 0.875 | 0.727 | 0.864 |
| 20 | **0.719 (+3.1%)** | 0.750 (-12.5%) | **0.659 (-6.8%)** | 0.682 (-18.2%) |

**Trainer metrics:**

| Step | Loss | Grad Norm | Mismatch KL |
|------|------|-----------|-------------|
| 0 | 0.0058 | 0.070 | 0.0007 |
| 10 | 0.0026 | 0.075 | 0.0007 |
| 19 | 0.0014 | 0.047 | 0.0008 |

**Train reward (broken metric — reflection turn extraction):** ~0.03 throughout (useless).

### Analysis

- Heldout Avg@4 improved +3.1%, but Pass@4 dropped significantly → mode collapse / diversity loss
- Train set *regressed* despite being the training data
- Mismatch KL barely moved (0.0007 → 0.0008) — weak learning signal
- Grad norms healthy (0.05-0.07), confirming PI was injected correctly (5x above broken v1-v3)
- The `env_ids` bug fix was validated: teacher context contained 1100-2200 chars (answer + reflection + sibling) vs 25-26 chars (answer only) in broken runs

---

## Experiment v5: Pure OPD, 8B Self-Teacher, Free Reflection, Higher LR, Larger Train Set

**Config:** `configs/aime_mt/reflection-self-teacher-8b-v5.toml`

Changes from v4:
- Free-form reflection (no structured format)
- LR 3e-5 (3x higher)
- 90 train problems (aimo-validation-aime, 4x more data)
- 40 steps (2x longer)
- Eval on full AIME 2025 (30 problems, completely heldout)

| Parameter | Value |
|-----------|-------|
| Model | willcb/Qwen3-8B (self-teacher) |
| Loss | Pure OPD (adv_tau=0, teacher_tau=1) |
| LR | 3e-5 |
| Batch size | 32 |
| Max steps | 40 |
| Train set | aimo-validation-aime (90 problems) |
| Eval set | AIME 2025 (30 problems, fully heldout) |
| Reflection | Open/free-form |
| Student PI | Binary |

### Results

**Eval (single-turn AIME):**

| Step | AIME 2025 Avg@4 | AIME 2025 Pass@4 | Train Avg@4 | Train Pass@4 |
|------|----------------|------------------|-------------|--------------|
| 0 (base) | **0.725** | 0.767 | **0.725** | 0.800 |
| 10 | 0.683 (-4.2%) | 0.767 | 0.725 (flat) | 0.833 |
| 20 | 0.583 (-14.2%) | 0.700 | 0.617 (-10.8%) | 0.800 |
| 30 | 0.575 (-15.0%) | 0.733 | 0.725 (recovered) | 0.867 |
| 40 | **0.592 (-13.3%)** | 0.667 | **0.667 (-5.8%)** | 0.800 |

**Trainer metrics — catastrophic divergence:**

| Step | Loss | Grad Norm | Entropy | Mismatch KL |
|------|------|-----------|---------|-------------|
| 0 | 0.008 | 0.062 | 0.32 | 0.0007 |
| 5 | 0.010 | 0.076 | 0.33 | 0.0009 |
| 10 | 0.010 | 0.061 | 0.37 | 0.0020 |
| 15 | 0.013 | 0.052 | 0.40 | 0.0024 |
| 20 | 0.016 | 0.075 | 0.46 | 0.0029 |
| 25 | 0.217 | 0.315 | 0.70 | 0.0023 |
| 30 | 0.250 | 0.205 | 0.76 | 0.0025 |
| 35 | 0.321 | 0.221 | 1.75 | 0.0032 |
| 39 | 0.692 | 0.367 | 2.98 | 0.0026 |

**Train reward:** 0.875 (step 0) → 0.281 (step 39) — model unlearned math.

### Analysis

- **Catastrophic divergence** starting around step 17-20: loss, entropy, and grad norms explode
- Entropy 0.32 → 2.98: model becomes increasingly random
- Loss 0.008 → 0.692: 85x increase
- LR 3e-5 was too aggressive — accelerated the instability
- **Root cause:** Pure OPD (adv_tau=0) with self-teacher creates a feedback loop. As the student degrades, the teacher (same weights) degrades too. No reward anchor to prevent drift.
- Free-form reflection may have contributed — less constrained output space for the model to explore

---

## 32B Self-Teacher Attempts (Failed — OOM)

Attempted to run 32B self-teacher on 4x A100 80GB. Multiple configurations tried:

| Config | GPUs | OOM at |
|--------|------|--------|
| TP=1 train, TP=2 infer | 1+2+1 | 79.2 GB (model alone fills GPU) |
| TP=2 train, TP=2 infer, 32K seq | 2+2 | 77.9 GB |
| TP=2 train, TP=2 infer, 16K seq | 2+2 | 78.9 GB |
| TP=2, LoRA r=8, optim_cpu_offload, 16K | 2+2 | 78.9 GB |

**Conclusion:** 32B training doesn't fit on 4x A100 80GB regardless of optimizations. The base model with TP=2 uses ~77GB per shard, leaving no room for activations/gradients. Would need 8+ GPUs or different hardware.

---

## Key Findings

1. **Pure OPD with self-teacher is unstable.** Without a reward anchor (adv_tau > 0), the model can diverge catastrophically. The teacher signal alone is insufficient to maintain stability when teacher weights are synced with student.

2. **The `prepare_teacher_context` pipeline works correctly** (after the env_ids bug fix). Teacher context includes answer + student reflection + correct sibling solutions. Grad norms 5x above broken baseline confirm PI signal is real.

3. **Reward extraction must match the environment.** Multi-turn environments need custom parsers to extract answers from the correct turn, not the default "last assistant message."

4. **Higher LR accelerates divergence** in pure OPD. The 3e-5 run collapsed faster and harder than 1e-5.

5. **Mismatch KL is the best early health signal.** In v4 it stayed flat (0.0007-0.0008) indicating weak learning. In v5 it moved to 0.002-0.003 showing the model was actually changing, but the changes were destructive.

## Next Experiment: Mixed OPD + GRPO

Based on these findings, the next run should combine:
- `adv_tau=0.5` (GRPO reward signal as stability anchor)
- `teacher_tau=1.0` (OPD for per-token teaching)
- `lr=1e-5` (conservative)
- Larger train set (90 problems)
- Free-form reflection

The GRPO signal provides direct pressure toward correct answers, preventing the entropy explosion seen in pure OPD. The OPD signal provides richer per-token feedback that GRPO alone cannot offer.
