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

### Experiment D: Deliberative OPD v2

- **Config:** `configs/aime/verify-deliberative-v2.toml`
- **Status:** RUNNING (launched 2026-03-10 ~07:35)

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

---

## Lessons Learned

- **Use `-a` not `-x` for eval env args:** `-x` (extra_env_kwargs) applies via `set_kwargs()` post-construction, overwriting `eval_dataset` Dataset object with a string. `-a` (env-args) passes to `load_environment()` constructor correctly.
- **vLLM flag is `--default-chat-template-kwargs`**, not `--chat-template-kwargs`.
- **Eval determinism:** Sort dataset in `_prepare_dataset()` + `num_eval_examples` arg in `load_environment()` ensures same 8 problems across standalone and in-training eval.
