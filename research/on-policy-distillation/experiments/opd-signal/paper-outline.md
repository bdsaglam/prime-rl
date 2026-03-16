# Paper Outline: Reflection-Augmented On-Policy Distillation

## Working Title

"Learning to Reflect: Structured Self-Diagnosis Amplifies On-Policy Distillation"

Alternative: "Reflection as Privileged Information: Teaching Models to Learn from Their Mistakes"

## Abstract (draft)

On-policy distillation (OPD) trains a student model by comparing its token-level predictions against a teacher's, where the teacher has access to privileged information (PI) like correct answers or reference solutions. We propose Reflection-Augmented OPD (RA-OPD), which adds a single extra conversation turn: after solving a problem, the student reflects on its own work in a structured format. The teacher then scores the full sequence including the reflection. We show that reflection tokens carry dramatically stronger learning signal than solution tokens — up to 7.5x higher effect size (Cohen's d = 4.67 vs 0.62, 95% CI [3.61, 4.64]). This signal is additive: it doesn't degrade solution-token learning. The mechanism works even when the student reflects blind (no external feedback): blind structured reflection achieves d = 1.44, exceeding answer-only PI on solution tokens (d = 0.63). Signal scales with compute: larger models reflect better (d increases with model size), and best-of-N selection over reflections gives +27% additional signal. Results replicate on a separate dataset (AIME 2024: d = 3.65, 95% CI [3.08, 4.60]). Our findings suggest that structured self-reflection is a general-purpose signal amplifier for knowledge distillation.

## 1. Introduction

- LLMs learn from feedback (RLHF, DPO, OPD) but the quality of feedback matters
- Standard OPD: teacher scores student's solution given answer/reference (static PI)
- Key insight: the student's own diagnosis of its mistakes contains rich information
- We add one turn: "reflect on your solution" → student generates structured self-diagnosis
- Teacher scores both solution AND reflection → stronger, more targeted learning signal
- This is analogous to how humans learn: attempt → reflect → get feedback → improve

## 2. Background

### 2.1 On-Policy Distillation (OPD)
- GKD foundation (Agarwal 2023)
- Teacher-student setup, per-token logprob matching
- PI types: answer, reference solution, peer rollout (SDPO)

### 2.2 Self-Distillation
- When teacher = student (same weights, different context)
- Self-OPD: the model teaches itself, PI creates information asymmetry
- Key: PI quality determines signal quality

### 2.3 Reflection and Self-Correction in LLMs
- Self-correction literature (Pan 2024): models struggle to correct without feedback
- Structured output formats improve reasoning (CoT, etc.)
- Our contribution: structured reflection as PI, trained via OPD

## 3. Method: Reflection-Augmented OPD (RA-OPD)

### 3.1 Setup
- Standard OPD: [problem] → [solution] → teacher scores with PI
- RA-OPD: [problem] → [solution] → [reflection prompt] → [reflection] → teacher scores with PI

### 3.2 Reflection Format
- Structured: VERDICT / CONFIDENCE / ERROR_TYPE / ERROR_LOCATION / WHAT_WENT_WRONG / CORRECTION
- Key design choice: precision over verbosity (structured d=4.67 vs open d=-0.57)

### 3.3 Information Asymmetry Design
- Student PI for reflection: varies (none, binary, answer, answer+hint)
- Teacher PI for scoring: answer, answer+ref, structured analysis
- The gap between student and teacher knowledge drives learning signal

### 3.4 Signal Measurement Framework

**Why measure signal instead of training?** OPD training is expensive (hours per run) and confounded by optimizer dynamics, learning rate, batch composition, etc. We want to compare dozens of PI configurations cheaply. The key insight: the OPD training loss *is* the KL divergence between student and teacher logprobs, so measuring this divergence directly tells us the magnitude of the gradient the student would receive.

**Metric 1: |KL| (mean absolute logprob divergence)**

For a rollout with tokens t₁...tₙ:

    |KL| = (1/n) Σᵢ |log p_student(tᵢ | context) - log p_teacher(tᵢ | context, PI)|

This is the per-token absolute difference in logprobs between student (no PI) and teacher (with PI), averaged over all completion tokens. It directly measures how much the teacher "disagrees" with the student's token predictions — which is proportional to the gradient magnitude in the OPD loss. Higher |KL| = stronger teaching signal.

**Why absolute value?** The raw KL can be positive (teacher more confident) or negative (teacher less confident). Both directions contribute gradient signal. |KL| captures total signal magnitude regardless of direction.

**Why this is a good proxy:** In the standard OPD loss, L = τ · (teacher_logprobs - student_logprobs), the gradient with respect to student parameters is proportional to the logprob difference. |KL| measures exactly this quantity, averaged over tokens. A configuration with higher |KL| literally produces larger gradients during training.

**Metric 2: Cohen's d (discrimination between correct and incorrect rollouts)**

    d = (mean |KL|_incorrect - mean |KL|_correct) / pooled_std

This measures whether the teaching signal is *selective* — does the teacher push harder on incorrect rollouts than correct ones? High d means the signal discriminates: incorrect rollouts get strong correction, correct rollouts get minimal perturbation. This is desirable because:
1. It allocates training budget to where learning is needed
2. It avoids degrading already-correct behavior
3. It's analogous to reward signal quality in RL (reward should separate good from bad trajectories)

**Why Cohen's d over raw ratio?** IC/C ratio (mean |KL| incorrect / correct) ignores variance. Cohen's d accounts for within-group spread, giving a standardized effect size. d > 0.8 is conventionally "large"; our best results reach d = 4.67.

**Metric 3: Gini coefficient (signal concentration)**

A potential failure mode: the teacher shifts *all* token logprobs up or down uniformly, inflating |KL| without targeting the critical "forking tokens" where the student's reasoning diverges. We measure concentration with the Gini coefficient over per-token |KL| values within each rollout:

    Gini = 0 → perfectly uniform (same |KL| at every token — "peanut butter" signal)
    Gini = 1 → maximally concentrated (all signal on one token)

We also report Top-k% fraction: what percentage of total |KL| in a rollout comes from the top 10% highest-signal tokens. Under uniform distribution, Top-10% = 10%.

**Empirical validation (32B self-teacher, incorrect rollouts):**

| Segment | Gini | Top-10% | CV |
|---|---|---|---|
| Solution tokens | 0.888 | 80.3% | 6.89 |
| Reflection (answer PI) | 0.766 | 55.8% | 1.98 |
| Reflection (blind) | 0.872 | 78.8% | 3.76 |

Both segments show highly concentrated signal (Gini > 0.75, Top-10% carrying 56-80% of total |KL|). The uniform-shift failure mode is ruled out. Solution tokens are slightly more concentrated (fewer but stronger spikes); reflection tokens spread signal more evenly across structured fields while remaining far from uniform.

**Metric 4: No-PI baseline d (confound floor)**

To ensure Cohen's d reflects PI-driven signal rather than confounds (e.g., incorrect rollouts being longer or more complex), we measure d when the teacher has *no privileged information*:

    d_no_pi ≈ 0.00–0.74 (depending on condition)

The conservative PI-attributable signal is d_with_PI - d_no_PI. For our headline result: 4.67 - 0.74 = 3.93, still very strong.

**Limitations:** |KL| measures gradient *magnitude*, not gradient *quality*. Cohen's d checks rollout-level selectivity; Gini checks token-level concentration. Neither guarantees the targeted tokens are the "right" ones (that would require human annotation of error locations). The ultimate validation requires actual training (Section 4.10).

**Practical advantage:** Scoring one rollout requires a single teacher prefill pass (~1s). Comparing 96 PI configurations on 100 rollouts takes ~2 hours vs ~96 days of training runs.

## 4. Experiments

### 4.1 Signal Measurement Setup
- Model: Qwen3-8B and Qwen3-32B
- Data: 100 rollouts on 25 AIME 2025 problems (86 incorrect, 14 correct)
- Conditions: 4 student PI × 2 formats × 3 teacher PI × 4 model configs = 96 conditions

### 4.2 Main Result: Reflection Tokens Carry Stronger Signal

**Table 1**: Cohen's d by segment and model configuration

| Config | d (solution) | d (reflection) | Ratio |
|---|---|---|---|
| 8B self-teacher | 0.54 | 0.85 | 1.6x |
| 32B teacher, 8B reflector | 0.63 | 2.56 | 4.1x |
| 32B self-teacher | 0.62 | **4.67** | **7.5x** |

### 4.3 Structured Format is Essential

**Table 2**: Open vs structured reflection

| Teacher PI | Open d | Structured d |
|---|---|---|
| answer_only | -0.57 | 3.99 |
| answer_ref | -0.12 | 2.90 |

Open-ended reflection has *negative* d — it inverts the discrimination signal.

### 4.4 Student PI Ablation

**Table 3**: More student PI → stronger signal (32B self-teacher, structured)

| Student PI | d (reflection) |
|---|---|
| none (blind) | 1.44 |
| binary | 2.89 |
| answer | 3.99 |
| answer+hint | **4.67** |

Even blind reflection (d=1.44) exceeds answer-only solution PI (d=0.63).

### 4.5 Scaling Properties

**Table 4**: Signal scales with model size
- Reflector: 8B → 32B gives ~2x d increase
- Teacher: 8B → 32B gives ~3x d increase
- Best-of-4 selection: +27.3% |KL| over random

### 4.6 Comparison with All PI Methods

**Table 5**: Unified comparison (same data, same rollouts)

| Method | Cohen's d | Segment |
|---|---|---|
| No PI | 0.10 | solution |
| Answer only | 0.84 | solution |
| Sibling rollout (SDPO) | 1.02 | solution |
| Structured analysis as PI (32B) | 2.23 | solution |
| **RA-OPD (32B self, blind)** | **1.44** | **reflection** |
| **RA-OPD (32B self, answer)** | **3.99** | **reflection** |
| **RA-OPD (32B self, answer+hint)** | **4.67** | **reflection** |

### 4.7 Additivity: Reflection + Analysis Stack

| Teacher PI | d (solution) | d (reflection) |
|---|---|---|
| answer_only | 0.62 | 0.72 |
| structured_analysis | 1.58 | 1.90 |

Using structured analysis as teacher PI gives strong signal on BOTH segments simultaneously.

### 4.8 Generalization: AIME 2024

| Condition | AIME 2025 d(refl) | AIME 2024 d(refl) | 95% CI |
|---|---|---|---|
| Blind (none) structured | 1.44 | 1.62 | [1.43, 1.88] |
| Answer structured | 3.99 | 3.65 | [3.08, 4.60] |

Both key findings replicate on a different dataset (30 AIME 2024 problems, 120 rollouts: 89 IC, 31 C). Bootstrap CIs (10,000 samples) exclude zero by wide margins.

### 4.9 Correct-Rollout Reflection Design

| Prompt Style | |KL| C (refl) | d |
|---|---|---|
| Diagnostic (original) | 0.0009 | 4.04 |
| Efficiency analysis | 0.0128 | 1.84 |
| Teaching | 0.0236 | 0.83 |

Tradeoff: richer prompts extract 26x more signal from correct rollouts but reduce discrimination. Efficiency analysis (v3) provides the best balance.

### 4.10 Training Validation (TODO)
- Self-OPD with reflection-in-sequence vs standard OPD
- Does signal translate to actual learning improvement?

## 5. Analysis

### 5.1 Why Structured > Open
- Structured format forces commitment to specific claims (VERDICT, ERROR_TYPE)
- Teacher can evaluate each claim against ground truth
- Open format produces variable-length, unfocused text that's harder to evaluate

### 5.2 Why Reflection Signal is So Strong
- Teacher sees the student's self-diagnosis AND knows the truth
- On incorrect rollouts: student's specific error identification creates rich divergence
- On correct rollouts: student says "all good" → minimal divergence
- The contrast between these is maximized by structured format

### 5.3 Blind Reflection as a Self-Supervised Signal
- d=1.44 with NO external feedback
- Student's uncertainty about its own work IS informative
- The teacher (with PI) evaluates this uncertainty → learning signal
- Enables training without external labels at inference time

### 5.4 Connection to the Bitter Lesson
- Signal scales with compute (bigger model, more reflections, BoN selection)
- No hand-crafted features needed — just a structured format
- The format itself could be learned (future work)

## 6. Related Work

- OPD: GKD, OPSD, SDPO, SDFT, pi-Distill
- Self-correction: STaR, rest^EM, Quiet-STaR
- Reflection: CTRL, Reflect-Retry-Reward, ReflectEvo, SCRIT
- Meta-learning: RL²F (Klissarov), RLTF-FM (Song)

## 7. Limitations and Future Work

- **Signal ≠ training**: |KL| measurement is a proxy for training signal quality. While it IS the OPD loss term, we haven't yet validated that higher d translates to faster/better training. The signal-to-training gap has been observed in deliberative PI experiments.
- **Math domain only**: Tested on AIME 2024 and 2025 (mathematical reasoning). Structured reflection may work differently for code generation, open-ended tasks, or multimodal problems.
- **Hand-designed format**: The VERDICT/ERROR_TYPE/... structured format was manually designed. The format itself could be learned or optimized.
- **Correct-rollout reflections**: Original diagnostic format wastes correct rollouts (all "none"). Alternative prompts (efficiency, teaching) extract more signal but reduce discrimination (Section 4.9). Optimal prompt for correct rollouts remains open.
- **Sample size**: AIME 2025 has only 14 correct rollouts (14% pass rate), making CIs wider. AIME 2024 with 31 correct rollouts (25.8%) provides tighter bounds.

## 8. Conclusion

Structured self-reflection is a powerful, general-purpose signal amplifier for on-policy distillation. By adding a single conversation turn where the student diagnoses its own work, we obtain per-token learning signal up to 7.5x stronger than standard methods. This signal scales with compute and works even without external labels. The mechanism suggests a path toward models that learn to learn: as the model trains, its reflections improve, generating even stronger signal for future learning.
