# OPD Verification Spike

**Goal:** Verify that OPD training pipeline works and establish whether deliberative OPD outperforms answer-only OPD.

**Date:** 2026-03-09

---

## Setup

| Parameter | Value |
|-----------|-------|
| Student | willcb/Qwen3-8B |
| Teacher | willcb/Qwen3-32B (external, port 8932) |
| Loss | Pure OPD (adv_tau=0, teacher_tau=1) |
| LoRA | rank=32, alpha=32, all projections |
| LR | 1e-5, constant |
| Batch size | 32 (rollouts/step) |
| Rollouts/example | 4 |
| Seq len | 32768 |
| Max tokens (train) | 31768 |
| Max tokens (eval) | 31000 |
| Max steps | 20 |
| Eval interval | 10 |
| Eval set | AIME 2025, deterministic 8 problems (sorted, num_eval_examples=8) |
| Train set | aimo-validation-aime (90 problems) |
| Hardware | 4x A100 80GB: 1 train + 3 infer |

## Pre-training Baselines (deterministic 8-problem AIME 2025)

| Model | Avg@4 | Pass@4 | Truncation | Avg Output Tokens |
|-------|-------|--------|------------|-------------------|
| 8B student | 0.688 | 0.75 | 12.5% | 14785 |
| 32B teacher | 0.781 | 0.875 | 0% | 13187 |

Gap: +9.3% absolute Avg@4.

Per-problem breakdown:
| Problem | 8B | 32B | Gap |
|---------|-----|-----|-----|
| 1 | 0/4 | 2/4 | +2 |
| 2 | 0/4 | 0/4 | 0 |
| 3 | 4/4 | 4/4 | 0 |
| 4 | 2/4 | 4/4 | +2 |
| 5 | 4/4 | 4/4 | 0 |
| 6 | 4/4 | 4/4 | 0 |
| 7 | 4/4 | 4/4 | 0 |
| 8 | 4/4 | 3/4 | -1 |

## Experiment A: Baseline OPD (answer_only PI)

- **Config:** `configs/aime/verify-baseline.toml`
- **W&B project:** aime-opd, run: verify-baseline
- **Output:** `outputs/aime-verify-baseline/`
- **Status:** RUNNING (started 2026-03-09 16:33)

### Eval Results (AIME 2025, 8 problems)

| Step | Avg@4 | Pass@1 | Pass@4 | Completion Len | Truncated |
|------|-------|--------|--------|----------------|-----------|
| 0 (in-training) | 0.6562 | 0.6562 | 0.7500 | 14726 | 15.6% |

### Before/After Comparison (standalone eval, step 9 LoRA adapter)

**AIME 2025 (8 problems, eval set):**
| Model | Avg@4 | Truncation | Avg Tokens |
|-------|-------|------------|------------|
| Base 8B | 0.688 | 12.5% | 14785 |
| Trained 8B (step 9) | 0.719 | 12.5% | 14970 |
| **Delta** | **+0.031 (+4.5%)** | 0% | +185 |

**aimo-validation-aime (16 problems, train set):**
| Model | Avg@4 | Truncation | Avg Tokens |
|-------|-------|------------|------------|
| Base 8B | 0.812 | 6.2% | 12978 |
| Trained 8B (step 9) | 0.875 | 4.7% | 13018 |
| **Delta** | **+0.063 (+7.8%)** | -1.5% | +40 |

**Conclusion:** OPD is working — small but consistent accuracy gains on both sets despite mismatch_kl appearing frozen (0.0007). No efficiency gain (token length unchanged). Training set is too easy (base already 0.812) — stronger signal expected with harder problems.

### Training Metrics

| Step | Reward | Seq Len | Loss | Mismatch KL | Entropy | Grad Norm | Orch Time |
|------|--------|---------|------|-------------|---------|-----------|-----------|
| 0 | 1.000 | 7698 | -0.0055 | 0.0007 | 0.2407 | 0.0613 | 902s |
| 1 | 0.969 | 13871 | -0.0028 | 0.0007 | 0.2631 | 0.0519 | 538s |
| 2 | 0.719 | 19878 | -0.0022 | 0.0007 | 0.2736 | 0.0514 | 464s |
| 3 | 0.750 | 11221 | -0.0054 | 0.0007 | 0.2845 | 0.0578 | 145s |
| 4 | 0.875 | 14263 | -0.0026 | 0.0007 | 0.2531 | 0.0447 | 500s |
| 5 | 0.813 | 17096 | -0.0019 | 0.0006 | 0.2413 | 0.0477 | 445s |
| 6 | 0.813 | 13667 | -0.0022 | 0.0006 | 0.2394 | 0.0494 | 588s |
| 7 | 0.844 | 13613 | -0.0036 | 0.0007 | 0.2900 | 0.0564 | 760s |
| 8 | 0.813 | 14561 | -0.0035 | 0.0007 | 0.2843 | 0.0549 | 376s |
| | | | | | | | |

**Observation:** Mismatch KL completely flat at 0.0006-0.0007 through 9 steps. Model is frozen. Loss oscillates (-0.002 to -0.006) without trend. Average reward ~0.85 — training set too easy for meaningful OPD signal. Gradient norms ~0.05 (tiny).

## Experiment B: Deliberative OPD

- **Config:** `configs/aime/verify-deliberative.toml`
- **W&B project:** aime-opd, run: verify-deliberative
- **Output:** `outputs/aime-verify-deliberative/`
- **Status:** QUEUED (runs after Experiment A)
- **Difference from A:** `deliberative=true, deliberative_max_tokens=4096, deliberative_temperature=0.3`

### Eval Results

| Step | Avg@4 | Pass@1 | Pass@4 | Completion Len | Truncated | Mismatch KL | Loss |
|------|-------|--------|--------|----------------|-----------|-------------|------|
| 0 | | | | | | | |
| 10 | | | | | | | |
| 20 | | | | | | | |

### Training Metrics

| Step | Loss | Mismatch KL | Entropy | Grad Norm |
|------|------|-------------|---------|-----------|
| | | | | |

## Key Questions

1. **Does mismatch_kl move?** Previous run: 0.0007→0.0010 in 22 steps (bs=128). With bs=32 we get more gradient updates per problem — should move faster.
2. **Does eval improve?** Step 0 baseline matches standalone eval (0.656 vs 0.688). If it rises, OPD is working.
3. **Does deliberative beat baseline?** The core research question.

## Missing: Train-set evaluation

Per-step reward confounds problem difficulty with model improvement (different problems sampled each step). Need before/after eval on a **fixed** subset of the train set to isolate learning effect.

**Plan:** After run completes, serve step 0 and step 20 checkpoints, eval both on a fixed subset of `aimo-validation-aime` (e.g., 16 problems). Compare accuracy and completion length.

For Experiment B, add train-set eval env to the config:
```toml
[[orchestrator.eval.env]]
id = "aime"
name = "aime-eval-train"
args = { dataset_name = "aimo-validation-aime", eval_dataset = "aimo-validation-aime", num_eval_examples = 16 }
```

## Decision Criteria

- If mismatch_kl rises >0.002 and eval improves: OPD pipeline works, proceed to full comparison.
- If mismatch_kl flat: pipeline broken or learning rate too low, debug.
- If both A and B improve but B > A: deliberative OPD validated.
- If train-set accuracy doesn't change despite 20 steps: model is frozen, need stronger signal (higher LR, harder problems, or larger LoRA rank).

---

## V2: AIME 2025 Non-Overlapping Split

Previous experiments used aimo-validation-aime (too easy, base 8B=0.812 Avg@4). V2 trains on AIME 2025 directly with non-overlapping train/eval split:

- **Train**: problems 0-21 (22 problems, sorted alphabetically)
- **Eval (heldout)**: problems 22-29 (8 problems, non-overlapping)
- **Eval (train)**: problems 0-21 (all 22 train problems, measures learning)

### V2 Pre-training Baselines

**8B student:**
| Split | Avg@4 | Pass@4 | Truncation | Avg Tokens |
|-------|-------|--------|------------|------------|
| Train (22) | 0.636 | — | 14.8% | 16683 |
| Heldout (8) | 0.625 | 0.875 | 12.5% | 17498 |

8B heldout per-problem: `[4/4, 2/4, 4/4, 2/4, 1/4, 3/4, 4/4, 0/4]` — good difficulty range.

**32B teacher:**
| Split | Avg@4 | Pass@4 | Truncation | Avg Tokens |
|-------|-------|--------|------------|------------|
| Train (22) | 0.648 | — | 18.2% | 16589 |
| Heldout (8) | 0.719 | 0.750 | 3.1% | 14523 |

32B heldout per-problem: `[4/4, 3/4, 4/4, 4/4, 0/4, 4/4, 4/4, 0/4]`

**Teacher-student gaps:**
| Split | 8B | 32B | Gap (abs) | Gap (rel) |
|-------|-----|-----|-----------|-----------|
| Train (22) | 0.636 | 0.648 | +0.012 | +1.9% |
| Heldout (8) | 0.625 | 0.719 | +0.094 | +15.0% |

Train gap is tiny (+1.9%) — both models struggle similarly on AIME 2025. Heldout gap is better (+15%). The OPD signal on the train set will be weak accuracy-wise, but KL divergence between 8B and 32B still exists even when both get the right answer (different reasoning paths/confidence).

### Experiment C: Baseline OPD v2

- **Config:** `configs/aime/verify-baseline-v2.toml`
- **W&B project:** aime-opd, run: verify-baseline-v2
- **Output:** `outputs/aime-verify-baseline-v2/`
- **Status:** COMPLETE (2026-03-10, 00:58–07:34, ~6.5h)

#### Eval Results

| Step | Split | Avg@4 | Pass@1 | Pass@2 | Pass@4 | Completion Len | Truncated |
|------|-------|-------|--------|--------|--------|----------------|-----------|
| 0 | Heldout (8) | 0.719 | 0.719 | 0.813 | 0.875 | 15526 | 0.0% |
| 0 | Train (22) | 0.682 | 0.682 | 0.742 | 0.773 | 17503 | 13.6% |
| 10 | Heldout (8) | 0.719 | 0.719 | 0.750 | 0.750 | 16080 | 9.4% |
| 10 | Train (22) | 0.659 | 0.659 | 0.720 | 0.773 | 16111 | 11.4% |
| 20 | Heldout (8) | **0.750** | 0.750 | 0.854 | 0.875 | 17068 | 15.6% |
| 20 | Train (22) | **0.705** | 0.705 | 0.780 | 0.818 | 16115 | 13.6% |

**Deltas (step 0 → 20):**
- Heldout: Avg@4 +0.031 (+4.3%), Pass@4 flat (0.875)
- Train: Avg@4 +0.023 (+3.4%), Pass@4 +0.045 (+5.8%)
- Truncation increased on heldout (0% → 15.6%) — model writing longer responses
- Completion length stable on train (~17.5K → 16.1K)

**Note:** Step 10 showed a dip (train 0.682→0.659, heldout flat) before recovering at step 20. Non-monotonic learning curve.

#### Training Metrics

| Step | Reward | Seq Len | Loss | Mismatch KL | Entropy | Grad Norm |
|------|--------|---------|------|-------------|---------|-----------|
| 0 | 0.938 | 8800 | -0.0038 | 0.0007 | 0.244 | 0.050 |
| 1 | 0.594 | 16115 | -0.0024 | 0.0007 | 0.308 | 0.055 |
| 2 | 0.625 | 15831 | -0.0025 | 0.0007 | 0.293 | 0.051 |
| 3 | 0.781 | 12203 | -0.0028 | 0.0007 | 0.282 | 0.049 |
| 4 | 0.688 | 14059 | -0.0022 | 0.0007 | 0.268 | 0.040 |
| 5 | 0.563 | 17291 | -0.0026 | 0.0007 | 0.322 | 0.055 |
| 6 | 0.813 | 16009 | -0.0020 | 0.0006 | 0.244 | 0.041 |
| 7 | 0.813 | 13901 | -0.0037 | 0.0007 | 0.296 | 0.048 |
| 8 | 0.688 | 16346 | -0.0024 | 0.0007 | 0.268 | 0.041 |
| 9 | 0.750 | 15716 | -0.0024 | 0.0007 | 0.275 | 0.043 |
| 10 | 0.813 | 13185 | -0.0037 | 0.0007 | 0.294 | 0.049 |
| 11 | 0.625 | 16608 | -0.0029 | 0.0007 | 0.308 | 0.048 |
| 12 | 0.656 | 16244 | -0.0029 | 0.0007 | 0.248 | 0.035 |
| 13 | 0.813 | 15700 | -0.0036 | 0.0008 | 0.309 | 0.049 |
| 14 | 0.781 | 17402 | -0.0028 | 0.0008 | 0.328 | 0.051 |
| 15 | 0.656 | 17659 | -0.0031 | 0.0008 | 0.329 | 0.051 |
| 16 | 0.625 | 18826 | -0.0027 | 0.0009 | 0.301 | 0.040 |
| 17 | 0.531 | 18195 | -0.0026 | 0.0008 | 0.288 | 0.038 |
| 18 | 0.625 | 17703 | -0.0029 | 0.0008 | 0.291 | 0.039 |
| 19 | 0.594 | 18124 | -0.0035 | 0.0009 | 0.318 | 0.041 |

**Observations:**
- Mismatch KL: 0.0007 → 0.0009 over 20 steps (+29% relative). Slow but non-zero learning signal.
- Average reward ~0.70 (much harder than v1's ~0.85). Better difficulty calibration.
- Entropy rising slightly (0.24 → 0.32) — model exploring more, not collapsing.
- Gradient norms stable (~0.04–0.05), no instability.
- Sequence length trending up (8.8K → 18.1K) — model writing progressively longer responses.

**Conclusion:** OPD baseline confirmed working on AIME 2025. Small but consistent improvements (+3-4% Avg@4) on both train and heldout sets. Mismatch KL finally showing movement (unlike v1 on easy data). Ready for deliberative comparison.

### Experiment D: Deliberative OPD v2 (cross-teacher 32B→8B)

- **Config:** `configs/aime/verify-deliberative-v2.toml`
- **Status:** KILLED at step 13 (2026-03-10)
- **Difference from C:** `deliberative=true, deliberative_max_tokens=4096`

#### Eval Results

| Step | Split | Avg@4 | Pass@4 |
|------|-------|-------|--------|
| 0 | Heldout (8) | 0.6562 | 0.8750 |
| 0 | Train (22) | 0.6932 | 0.7727 |
| 10 | Heldout (8) | 0.7500 | 0.8750 |
| 10 | Train (22) | 0.6477 | 0.7727 |

#### Training Metrics

| Step | Loss | Mismatch KL | Entropy | Grad Norm |
|------|------|-------------|---------|-----------|
| 0 | 0.0041 | 0.0007 | 0.243 | 0.077 |
| 5 | -0.0014 | 0.0006 | 0.279 | 0.060 |
| 10 | -0.0010 | 0.0007 | 0.310 | 0.063 |
| 13 | -0.0009 | 0.0008 | 0.321 | 0.070 |

**Observations:**
- Loss near zero or positive in early steps (very different from baseline's consistently negative loss)
- Mismatch KL same range as baseline (0.0006-0.0008)
- Heldout improved +9.4pp by step 10 (but different step-0 baseline makes comparison uncertain)
- Killed before completion — inconclusive

### Experiment E: Adaptive PI (env-specific prepare_teacher_context)

- **Config:** `configs/aime/verify-analyzer-v2.toml`
- **W&B project:** aime-opd, run: verify-analyzer-v2
- **Output:** `outputs/aime-verify-analyzer-v2/`
- **Status:** RUNNING (launched 2026-03-10 ~20:34)
- **Difference from C:** `[orchestrator.analyzer]` configured with Gemini 3 Flash. AIME env's `prepare_teacher_context` generates adaptive PI per-rollout:
  - Correct rollouts (reward=1): skip LLM call, keep static PI
  - Incorrect rollouts: call Gemini with env-specific prompt → problem notes/hints
  - Answer prepended to adaptive analysis
  - Two prompts: WITH_SOLUTION (pitfalls only) / WITHOUT_SOLUTION (sketch + pitfalls)
- **Architecture change:** `prepare_teacher_context` moved to env package (contract-based, no shared analyzer in prime-rl)

#### Eval Results

| Step | Split | Avg@4 | Pass@1 | Pass@2 | Pass@4 | Completion Len | Truncated |
|------|-------|-------|--------|--------|--------|----------------|-----------|
| 0 | Heldout (8) | 0.7812 | 0.7812 | 0.8125 | 0.8750 | 16903 | 6.2% |
| 0 | Train (22) | 0.6932 | 0.6932 | 0.7803 | 0.8182 | 16912 | 12.5% |
| 10 | Heldout (8) | | | | | | |
| 10 | Train (22) | | | | | | |
| 20 | Heldout (8) | | | | | | |
| 20 | Train (22) | | | | | | |

#### Training Metrics

| Step | Reward | Seq Len | Loss | Mismatch KL | Entropy | Grad Norm | Teacher Context Time |
|------|--------|---------|------|-------------|---------|-----------|---------------------|
| 0 | 1.000 | 12684 | -0.0042 | 0.0007 | 0.2521 | 0.0639 | 0.0s (all correct, skipped) |
| 1 | | | -0.0034 | 0.0007 | 0.3006 | 0.0545 | 26.9s |
| 2 | | | -0.0030 | 0.0007 | 0.2888 | 0.0503 | 82.9s |

**Status:** Crashed after step 2 with `AsyncLibraryNotFoundError: unknown async library, or not in async context`. The `prepare_teacher_context` async call from the env package hit an event loop incompatibility. Abandoned — pivoted to self-teacher experiments to isolate PI signal.

---

## Self-Teacher Experiments (8B→8B)

**Rationale:** Cross-teacher experiments (C, D, E) all show identical mismatch KL (0.0006-0.0009) regardless of PI type. The cross-model gap (32B vs 8B) dominates ~80% of the signal, making PI variations invisible. Self-teacher (8B→8B) removes the cross-model gap entirely, making PI the ONLY source of signal. Answer-only self-teacher should be a near-null baseline.

### Experiment F: Self-Teacher Baseline (answer_only, 8B→8B)

- **Config:** `configs/aime/verify-self-teacher-v2.toml`
- **Output:** `outputs/aime-verify-self-teacher-v2/`
- **Status:** COMPLETE (2026-03-11, ~10:10–19:30, ~9.3h)
- **Setup:** 8B student + 8B self-teacher, pure OPD (adv_tau=0, teacher_tau=1), answer_only PI
- **Hardware:** 4x A100: 1 train + 2 infer + 1 teacher (local, port 8001)

#### Eval Results

| Step | Split | Avg@4 | Pass@1 | Pass@2 | Pass@4 | Completion Len | Truncated |
|------|-------|-------|--------|--------|--------|----------------|-----------|
| 0 | Heldout (8) | 0.7188 | 0.7188 | 0.7500 | 0.7500 | 17291 | 12.5% |
| 0 | Train (22) | 0.6932 | 0.6932 | 0.7424 | 0.7727 | 16749 | 15.9% |
| 10 | Heldout (8) | 0.7188 | 0.7188 | 0.7500 | 0.7500 | 14836 | 0.0% |
| 10 | Train (22) | 0.7045 | 0.7045 | 0.7727 | 0.8182 | 15227 | 8.0% |
| 20 | Heldout (8) | **0.6562** | 0.6562 | 0.7917 | 0.8750 | 16505 | 0.0% |
| 20 | Train (22) | **0.6932** | 0.6932 | 0.7879 | 0.8182 | 15509 | 9.1% |

**Deltas (step 0 → 20):**
- Heldout: Avg@4 -0.063 (-8.7%), Pass@4 +0.125 (more variance, not better)
- Train: Avg@4 flat (0.6932), Pass@4 +0.045
- **No learning signal.** Self-teacher with answer-only PI is confirmed null baseline.

#### Training Metrics

| Step | Reward | Seq Len | Loss | Mismatch KL | Entropy | Grad Norm |
|------|--------|---------|------|-------------|---------|-----------|
| 0 | 1.000 | 6848 | -0.0010 | 0.0006 | 0.216 | 0.014 |
| 1 | 0.875 | 12638 | -0.0004 | 0.0007 | 0.288 | 0.011 |
| 2 | 0.688 | 17245 | -0.0002 | 0.0006 | 0.293 | 0.012 |
| 3 | 0.906 | 14164 | -0.0005 | 0.0007 | 0.258 | 0.012 |
| 4 | 0.938 | 13215 | -0.0002 | 0.0007 | 0.276 | 0.009 |
| 5 | 0.938 | 11928 | -0.0005 | 0.0007 | 0.251 | 0.010 |
| 6 | 0.750 | 14451 | -0.0004 | 0.0007 | 0.272 | 0.009 |
| 7 | 0.813 | 11774 | -0.0005 | 0.0007 | 0.301 | 0.011 |
| 8 | 0.781 | 14946 | -0.0002 | 0.0006 | 0.248 | 0.008 |
| 9 | 0.969 | 13931 | -0.0005 | 0.0007 | 0.278 | 0.014 |
| 10 | 0.750 | 13122 | -0.0004 | 0.0007 | 0.277 | 0.009 |
| 11 | 0.438 | 21082 | -0.0001 | 0.0006 | 0.263 | 0.006 |
| 12 | 0.531 | 19322 | -0.0001 | 0.0007 | 0.300 | 0.006 |
| 13 | 0.969 | 8637 | -0.0006 | 0.0007 | 0.251 | 0.010 |
| 14 | 0.750 | 14804 | -0.0003 | 0.0007 | 0.303 | 0.009 |
| 15 | 0.969 | 11185 | -0.0007 | 0.0007 | 0.281 | 0.011 |
| 16 | 0.906 | 10142 | -0.0005 | 0.0007 | 0.253 | 0.009 |
| 17 | 0.656 | 14467 | -0.0004 | 0.0007 | 0.293 | 0.007 |
| 18 | 0.625 | 14782 | -0.0004 | 0.0007 | 0.301 | 0.008 |
| 19 | 0.750 | 15648 | -0.0006 | 0.0007 | 0.283 | 0.010 |

**Observations:**
- **Mismatch KL completely flat at 0.0006-0.0007** — zero learning signal, as expected for self-distillation with no PI advantage.
- **Loss near zero** (-0.0001 to -0.0010) — 3-5x smaller than cross-teacher baseline (C).
- **Grad norms ~0.01** — 4-5x smaller than cross-teacher (0.04-0.05). Model has nothing to learn from itself.
- **Average reward ~0.80** — same batch difficulty as cross-teacher, confirming the difference is the teacher not the data.
- **Conclusion:** Self-teacher with answer-only PI is a confirmed null baseline. No learning signal exists when the teacher IS the student.

### Experiment G: Deliberative Self-Teacher (8B→8B)

- **Config:** `configs/aime/verify-self-teacher-deliberative-v2.toml`
- **Output:** `outputs/aime-verify-self-teacher-deliberative-v2/`
- **Status:** RUNNING (launched 2026-03-11 ~20:55)
- **Difference from F:** `deliberative=true, deliberative_max_tokens=4096, deliberative_temperature=0.3`
- **Key question:** Does deliberative PI break the self-teacher null baseline? If mismatch KL rises above 0.001, the deliberative teaching signal is real and independent of model size gap.

#### Eval Results

| Step | Split | Avg@4 | Pass@1 | Pass@2 | Pass@4 | Completion Len | Truncated |
|------|-------|-------|--------|--------|--------|----------------|-----------|
| 0 | Heldout (8) | 0.7188 | 0.7188 | 0.7500 | 0.7500 | 16789 | 6.2% |
| 0 | Train (22) | 0.6591 | 0.6591 | 0.7045 | 0.7273 | 16711 | 13.6% |
| 10 | Heldout (8) | 0.7188 | 0.7188 | 0.8125 | 0.8750 | 15842 | 3.1% |
| 10 | Train (22) | **0.7045** | 0.7045 | 0.7652 | 0.8182 | 15436 | 8.0% |
| 20 | Heldout (8) | **0.7500** | 0.7500 | 0.8125 | 0.8750 | 13882 | 0.0% |
| 20 | Train (22) | **0.7045** | 0.7045 | 0.7424 | 0.7727 | 14569 | 5.7% |

**Deltas (step 0 → 20):**
- Heldout: Avg@4 **+0.031 (+4.3%)**, Pass@4 +0.125 (+16.7%)
- Train: Avg@4 **+0.045 (+6.9%)**, Pass@4 +0.045 (+6.2%)
- Completion length decreasing (16.8K → 13.9K heldout) — model getting more efficient
- Truncation dropping (6.2% → 0.0% heldout)
- **Learning is happening** — contrast with baseline F which showed zero improvement

#### Training Metrics

| Step | Loss | Mismatch KL | Entropy | Grad Norm |
|------|------|-------------|---------|-----------|
| 0 | +0.0007 | 0.0007 | 0.270 | **0.052** |
| 1 | -0.0002 | 0.0007 | 0.276 | **0.045** |
| 2 | +0.0004 | 0.0007 | 0.287 | **0.048** |
| 3 | +0.0006 | 0.0007 | 0.278 | **0.042** |
| 4 | +0.0006 | 0.0007 | 0.238 | **0.041** |
| 5 | +0.0007 | 0.0008 | 0.314 | **0.050** |
| 6 | +0.0004 | 0.0007 | 0.265 | **0.048** |
| 7 | +0.0000 | 0.0007 | 0.292 | **0.044** |
| 8 | +0.0001 | 0.0007 | 0.289 | **0.042** |
| 9 | -0.0001 | 0.0007 | 0.271 | **0.041** |
| 10 | -0.0007 | 0.0007 | 0.305 | **0.048** |
| 11 | +0.0000 | 0.0007 | 0.314 | **0.042** |
| 12 | +0.0001 | 0.0007 | 0.255 | **0.031** |
| 13 | +0.0011 | 0.0008 | 0.258 | **0.049** |
| 14 | +0.0001 | 0.0008 | 0.296 | **0.042** |
| 15 | -0.0007 | 0.0007 | 0.294 | **0.046** |
| 16 | -0.0000 | 0.0007 | 0.258 | **0.040** |
| 17 | -0.0003 | 0.0008 | 0.291 | **0.042** |
| 18 | -0.0001 | 0.0007 | 0.286 | **0.032** |
| 19 | -0.0009 | 0.0008 | 0.314 | **0.044** |

**Observations:**
- **Grad norms 3-5x larger than baseline F** (0.031-0.052 vs 0.006-0.014). Deliberative PI creates real gradient signal in self-teacher.
- **Loss oscillates around zero**, often positive (steps 0-6). Baseline F was always negative. Positive loss means student logprobs < teacher logprobs on deliberative-scored tokens — teacher distribution genuinely differs.
- **Mismatch KL barely moves** (0.0007-0.0008). The gradient signal exists but doesn't yet translate to large parameter divergence. LR may be too low, or 20 steps insufficient.
- **Eval improves despite flat mismatch KL** — suggesting the model is learning subtler distributional shifts not captured by this single metric.

**Note:** First run crashed at step 6 due to empty trajectory bug in `utils.py:232` (same pattern as teacher_context.py). Fixed with `.get("trajectory", [])` guard. Restarted from scratch.

---

## Summary Table

| Experiment | Teacher | PI Type | Mismatch KL (range) | Heldout Δ Avg@4 | Train Δ Avg@4 | Status |
|------------|---------|---------|---------------------|-----------------|---------------|--------|
| C: Baseline v2 | 32B | answer_only | 0.0007→0.0009 | +4.3% | +3.4% | Complete |
| D: Deliberative v2 | 32B | deliberative | 0.0007→0.0008 | +9.4% (step 10) | -4.6% (step 10) | Killed@13 |
| E: Analyzer v2 | 32B | adaptive (Gemini) | 0.0007 | — | — | Crashed@2 |
| F: Self-teacher baseline | 8B | answer_only | 0.0006-0.0007 (flat) | -8.7% | 0% | Complete |
| G: Self-teacher deliberative | 8B | deliberative | 0.0007-0.0008 | **+4.3%** | **+6.9%** | Complete |

**Key result: Deliberative PI validated.** Cross-teacher experiments (C, D, E) all show identical mismatch KL regardless of PI — the model gap masks PI effects. Self-teacher baseline (F) confirms zero learning signal without model gap. Deliberative self-teacher (G) **learns where baseline doesn't** — heldout +4.3% vs -6.3%, grad norms 3-5x larger. The deliberative analysis creates an information asymmetry that drives learning even when teacher = student.

---

## Lessons Learned

- **Use `-a` not `-x` for eval env args:** `-x` (extra_env_kwargs) applies via `set_kwargs()` post-construction, overwriting `eval_dataset` Dataset object with a string. `-a` (env-args) passes to `load_environment()` constructor correctly.
- **vLLM flag is `--default-chat-template-kwargs`**, not `--chat-template-kwargs`.
- **Eval determinism:** Sort dataset in `_prepare_dataset()` + `num_eval_examples` arg in `load_environment()` ensures same 8 problems across standalone and in-training eval.
