# OPD Signal Measurement: Findings

## Experiment Design

We measure the teaching signal in On-Policy Distillation (OPD) by varying three orthogonal dimensions:

### Dimension 1: Teacher Model

The model that scores the student's rollout tokens.

- **Self-teacher**: Same model as student (e.g., 8B → 8B)
- **Cross-teacher**: Larger model (e.g., 8B → 32B)

### Dimension 2: Privileged Information (PI)

Text injected into the teacher's system prompt to condition its predictions.

- **None** (empty string) — **baseline, lower bound**
- Answer only
- Answer + reference solution
- Student self-reflection (confidence, diagnosis)
- Any other conditioning text

### Dimension 3: Teacher Lens (System Prompt)

The system prompt used by the teacher (beyond the PI).

- Default math prompt
- Custom prompts (e.g., "solve efficiently")
- **Initial results: marginal effect (r > 0.97 between lens variants). Deprioritized.**

**Lower bound**: teacher = student, PI = '' → KL ≈ 0. All signal is measured as gain from this baseline.

## Methodology

### Pipeline

1. **Student rollout generation**: Qwen3-8B generates 4 rollouts per problem (25 AIME problems = 100 rollouts)
2. **Student self-scoring**: Student scores its own tokens via prefill → p_student(token | problem)
3. **Teacher scoring**: Teacher scores the same tokens with PI → p_teacher(token | problem, PI)
4. **Signal**: Per-token KL = student_logprob - teacher_logprob

### Prefill Scoring

We use vLLM's prefill with echo (`echo=True`, `max_tokens=1`). The token sequence is:

```
[system prompt + PI] [user: problem] [assistant: student's rollout tokens]
```

We extract logprobs only for the completion tokens (student's rollout).

PI is injected as:

```
{system prompt}

--- PRIVILEGED INFORMATION ---
{PI text}
```

### Metrics

**|KL| (mean absolute KL divergence)**: `mean(|student_logprob(t) - teacher_logprob(t)|)` across all tokens, averaged across rollouts. Higher = stronger signal. Baseline (self-teacher, no PI) ≈ 0.005.

**IC/C Ratio**: `mean(|KL| for incorrect) / mean(|KL| for correct)`. Closer to 1.0 = more balanced signal.

**Cohen's d**: Effect size separating correct from incorrect rollouts by |KL|. Higher = better discrimination.

**Copy Artifact**: When student text appears in PI, teacher reads tokens from context → near-zero logprobs. Inflates |KL| without pedagogical value.

**TCE (Teacher Confidence Excess)**: `mean(teacher_lp - student_lp)` where teacher > student. Continuous copy metric.

## Results: Dimension 1 × Dimension 2

### 8B Self-Teacher (Qwen3-8B → Qwen3-8B)

Signal gain over baseline (no PI = 0.005):

| PI Condition | \|KL\| | Δ from baseline | Cohen's d | IC/C |
|---|---|---|---|---|
| **No PI (baseline)** | **0.005** | **0** | — | — |
| Answer only | 0.014 | +0.009 | 0.78 | 1.51x |
| Solve efficiently (instruction) | 0.035 | +0.030 | — | — |
| Blind diagnosis (self-summary) | 0.038 | +0.033 | 0.55 | 1.30x |
| Blind confidence (self-rating) | 0.042 | +0.037 | 0.55 | — |
| Answer + ref solution (standard OPD) | 0.063 | +0.058 | 0.44 | 1.22x |

### 32B Cross-Teacher (Qwen3-8B → Qwen3-32B)

The 32B teacher scores the 8B student's rollout tokens. The cross-model gap creates a large baseline |KL| even with no PI.

| PI Condition | \|KL\| | Δ from no_pi | Cohen's d | IC/C |
|---|---|---|---|---|
| **No PI (cross-model baseline)** | **0.097** | **0** | 0.72 | 1.35x |
| Answer only | 0.098 | +0.002 | 0.75 | 1.36x |
| Solve efficiently | 0.101 | +0.005 | 0.65 | 1.30x |
| Blind diagnosis | 0.104 | +0.007 | 0.70 | 1.34x |
| Blind confidence | 0.105 | +0.008 | 0.63 | 1.30x |
| Ans + confidence | 0.106 | +0.009 | 0.64 | 1.30x |
| Ans + diagnosis | 0.105 | +0.008 | 0.71 | 1.35x |
| Ans + self-correct | 0.105 | +0.008 | 0.69 | 1.33x |
| Ans + self-grade | 0.106 | +0.009 | 0.68 | 1.34x |
| Ans + ref + confidence | 0.116 | +0.019 | 0.61 | 1.28x |
| Ans + ref + diagnosis | 0.117 | +0.020 | 0.62 | 1.30x |
| Answer + ref solution | 0.119 | +0.022 | 0.67 | 1.33x |

**Key observation**: The cross-model gap (0.097) dominates. PI adds only marginal signal on top (max +0.022). The 32B teacher already "teaches" through its different distribution, regardless of PI.

### 32B Self-Teacher (Qwen3-32B → Qwen3-32B, scoring 8B rollouts)

Uses 32B no_pi logprobs as the "student" baseline, isolating the PI-only signal from the same model.

| PI Condition | \|KL\| | Cohen's d | IC/C |
|---|---|---|---|
| Answer only | 0.016 | 0.71 | 1.43x |
| Solve efficiently | 0.023 | 0.16 | 1.06x |
| Blind diagnosis | 0.039 | 0.54 | 1.33x |
| Ans + self-correct | 0.041 | 0.45 | 1.26x |
| Ans + diagnosis | 0.041 | 0.57 | 1.35x |
| Ans + self-grade | 0.044 | 0.45 | 1.26x |
| Blind confidence | 0.046 | 0.33 | 1.18x |
| Ans + confidence | 0.048 | 0.38 | 1.21x |
| Ans + ref + diagnosis | 0.064 | 0.45 | 1.21x |
| Answer + ref solution | 0.065 | 0.55 | 1.28x |
| Ans + ref + confidence | 0.066 | 0.43 | 1.19x |

**Key observation**: PI signal from 32B self-teacher closely matches 8B self-teacher (e.g., answer_ref: 0.065 vs 0.063, blind_confidence: 0.046 vs 0.042). The PI effect is model-size-invariant.

## Negative Results

### Copy Artifact: Student Text in PI is a Dead End

Including any student rollout text in PI — full or truncated — causes the teacher to read tokens from context rather than teach.

**Full rollout**: 91% of tokens have teacher logprob > -0.01. |KL| = 0.185 is meaningless copy signal.

**Truncated rollout**: Appeared promising (monotonic |KL| increase with length) but detailed analysis shows:

- Before truncation boundary: TCE = 0.20-0.27 (pure copy)
- After truncation boundary: TCE = 0.063, matching answer_ref baseline
- The rollout text contributes zero genuine signal beyond copy artifact
- Monotonic |KL| increase = more tokens copied, not more teaching

**Verdict**: Any PI that includes student's own rollout text is contaminated. All truncation conditions are excluded from main results.

### Teacher Lens (System Prompt Variation): Marginal Effect

Varying the teacher's system prompt produces nearly identical signal patterns (r > 0.97 between variants). The teacher's behavior is dominated by the PI content, not the system prompt instructions.

## Key Findings

### 1. Cross-Model Gap Dominates Over PI

The 32B teacher provides |KL| = 0.097 with **no PI at all** — 19x the 8B self-teacher baseline (0.005). PI adds at most +0.022 on top of this, meaning the cross-model distribution difference accounts for ~80% of total signal in cross-teacher OPD.

This suggests that in standard OPD (larger teacher + answer_ref), most of the training signal comes from the teacher being a different (better) model, not from the PI content.

### 2. PI Signal is Model-Size-Invariant

When isolating just the PI effect (32B self-teacher), results nearly match 8B self-teacher:


| PI Condition          | 8B self | 32B self |
| --------------------- | ------- | -------- |
| Answer only           | 0.014   | 0.016    |
| Blind diagnosis       | 0.038   | 0.039    |
| Blind confidence      | 0.042   | 0.046    |
| Answer + ref solution | 0.063   | 0.065    |


The PI effect is consistent across model sizes — it's an intrinsic property of the information content, not the model capacity.

### 3. Blind Confidence Rating is the Best Self-Generated PI

The student assesses its own confidence (1-10 scale) without seeing the answer. This becomes PI for the teacher.

- |KL| = 0.042-0.046 (8B/32B self-teacher) — **8-9x gain** over no-PI baseline
- Achieves 67-71% of answer_ref signal with **zero external knowledge**
- Confidence ratings are genuinely calibrated (r = 0.41 with correctness)
  - Mean confidence: correct = 8.5/10, incorrect = 6.0/10 (Δ = +2.5)
  - Top 10% by confidence: 50% correct (vs 14% base rate = 3.6x lift)

### 4. Multi-Lens Combination Closes the Gap to Oracle PI

Three ref-free lenses (confidence + diagnosis + efficiency), combined at token level, reach **95% of answer_ref signal** (0.062 vs 0.065) with zero external knowledge. The key insight: complementarity exists at the **token level** (r = 0.43-0.65 between lenses, 48% high-disagreement tokens), not at the rollout level.

Adding ref-free lenses on top of answer-based lenses gives +37% |KL| improvement (0.065 → 0.088), suggesting multi-lens is valuable even when external knowledge is available.

### 5. Combining PI in Single Prompt: Additive but Modest

Adding self-reflection to answer+ref *in a single prompt* gives small gains (32B self-teacher):

- answer_ref alone: 0.065
- ans_ref + confidence: 0.066 (+1.5%)
- ans_ref + diagnosis: 0.064 (-1.5%)

Adding self-reflection to answer-only is more impactful:

- answer_only: 0.016
- ans + confidence: 0.048 (+200%)
- ans + diagnosis: 0.041 (+156%)

Self-reflection partially substitutes for the reference solution. But combining PI types in separate teacher passes (multi-lens) is much more effective than combining them in a single prompt.

### 6. Answer Alone is Surprisingly Weak

Just providing the correct answer gives |KL| = 0.014-0.016 — only +0.009-0.016 over baseline. The reference solution is what makes standard OPD work (0.016 → 0.065 = +300% from adding ref solution).

Blind confidence (0.046) provides **~3x more signal** than the correct answer alone (0.016), despite using no external knowledge.

### 7. Process-Level PI is Complementary to Outcome-Level PI

Diagnostic summaries of student behavior create signal that is:

- **Genuinely different** from outcome-based PI (r = 0.84 between blind_diagnosis and answer_only)
- **More discriminative** (Cohen's d 0.54-0.55 vs 0.55 for answer_ref in self-teacher)
- **More balanced** across correctness (IC/C ratio 1.30-1.33x vs 1.43x for answer_only)

## Multi-Lens Combination: Token-Level Ensemble of Self-Generated PI

### Motivation

In many real-world scenarios, no ground truth or reference solution is available. We've shown that self-generated PI (blind confidence, blind diagnosis) extracts substantial signal from student rollouts. The multi-lens hypothesis asks: **can combining multiple self-generated PI types extract more signal than any single one?**

The original multi-lens idea (varying system prompts) failed because prompt variation barely changes distributions (r > 0.97). But PI variation is a much stronger knob — different PI types produce genuinely different per-token signals.

### Per-Token Correlation Between Ref-Free Lenses

| Lens Pair | Token-level r (32B) | Token-level r (8B) |
|---|---|---|
| blind_confidence vs blind_diagnosis | 0.65 | 0.61 |
| blind_confidence vs solve_efficiently | 0.48 | 0.46 |
| blind_diagnosis vs solve_efficiently | 0.46 | 0.43 |

For comparison, system prompt variation gives r > 0.97. PI-based lenses are **genuinely complementary** at token level.

48% of tokens show high disagreement (CV > 0.5) between lenses. Each lens "wins" (has highest |KL|) on different tokens: confidence 51%, diagnosis 30%, efficiency 19%.

### Oracle Upper Bound: Per-Token Max Across Lenses

Taking max(|KL|) across lenses at each token gives the theoretical maximum signal from combining these lenses.

**32B Self-Teacher (ref-free lenses only):**

| Method | |KL| | Cohen's d |
|---|---|---|
| blind_confidence (best single) | 0.046 | 0.33 |
| blind_diagnosis | 0.039 | 0.54 |
| solve_efficiently | 0.023 | 0.16 |
| Equal-weight mean | 0.036 | 0.40 |
| **Per-token max (oracle)** | **0.062** | **0.43** |
| answer_ref (for comparison) | 0.065 | 0.55 |

**3 ref-free lenses oracle-combined (0.062) nearly match answer_ref (0.065) — with zero external knowledge.** The oracle improvement is +36% over the best single ref-free lens.

**8B Self-Teacher:**

| Method | |KL| | Cohen's d |
|---|---|---|
| blind_confidence (best single) | 0.042 | 0.30 |
| Per-token max (oracle) | 0.055 | 0.37 |
| answer_ref (for comparison) | 0.063 | 0.44 |

Oracle improvement: +31% over best single lens. Reaches 87% of answer_ref signal.

### Mixed ref-free + answer-based lenses (32B self-teacher)

| Method | |KL| | Cohen's d |
|---|---|---|
| answer_ref (best single) | 0.065 | 0.55 |
| Per-token max (4 lenses) | 0.088 | 0.56 |

Adding ref-free lenses on top of answer-based lenses gives +37% |KL| improvement.

### Rollout-Level vs Token-Level Combination

Weighted rollout-level mixture (Σ w_k * mean_|KL|_k) does **not** improve over the best single lens — the optimal weight vector always concentrates on one lens. This is because the complementarity is at **token level**, not rollout level. Different lenses activate on different tokens within the same rollout.

This means the training approach should use **per-token selection or weighting**, not fixed rollout-level weights.

### Implications for Training

The multi-lens OPD training recipe:
1. Student generates rollout
2. Student generates K self-reflections (confidence, diagnosis, efficiency assessment, ...)
3. Teacher scores rollout K times, each conditioned on a different self-reflection
4. At each token, select or weight across lenses (e.g., take max |KL| or softmax-weighted)
5. Train with per-token KL target

This is a **token-level ensemble** of self-generated teaching signals that requires no external knowledge.

## Copy Artifact Analysis

We developed continuous metrics for detecting copy contamination:

**TCE (Teacher Confidence Excess)**: Per-token measure of how much more confident the teacher is than the student. Copy conditions show TCE = 0.15-0.23 (vs 0.005 baseline).

**Boundary Decay Analysis**: For truncated PI, we map the character truncation point to token space and analyze TCE before/after the boundary:

- Before boundary: TCE = 0.20-0.27 (copy zone)
- After boundary: TCE = 0.063 (matches answer_ref — no excess from rollout text)
- Transition: Sharp decay over ~20-30 tokens past boundary

**Copy Fraction Curve**: Position-dependent fraction of tokens with logprob > -0.01:

- no_pi: 56-72% (natural gradient, tokens later in sequence are more predictable)
- full_rollout: 73-95% (severe copy, increasing with position)
- truncated: elevated before boundary, baseline after

## Confidence Calibration Analysis


| Confidence | N   | % Correct |
| ---------- | --- | --------- |
| 2-5/10     | 34  | 0%        |
| 6/10       | 18  | 6%        |
| 7/10       | 9   | 11%       |
| 8/10       | 17  | 12%       |
| 9/10       | 22  | 45%       |


## Test-Time Compute for Teaching: The Reasoning Teacher Hypothesis

### The Analogy

There is a deep structural parallel between inference-time scaling (chain-of-thought, reasoning LLMs) and teaching signal quality:

**Inference**: Before CoT, we expected LLMs to solve hard problems in a single forward pass. CoT/reasoning LLMs showed that using tokens to explore the solution space — thinking, backtracking, verifying — dramatically improves performance on problems requiring deep reasoning. The model spends compute (tokens) to arrive at a better answer.

**Teaching**: In standard OPD, we expect the teacher to produce good per-token credit assignment in a single forward pass — conditioned on PI, it reads the student's rollout and outputs logprobs. But credit assignment is hard. The teacher must simultaneously understand the problem, the reference solution, the student's approach, where the student went wrong, what would have been better, and how each token contributed to the outcome. We're asking the teacher to do this *implicitly* through its attention patterns during prefill, with no scratch space.

**The insight**: Just as CoT gives the model compute budget to reason about *what answer to give*, we should give the teacher compute budget to reason about *what learning signal to give*. Teaching is at least as hard as solving — arguably harder, because it requires understanding both the solution and the student's specific failure mode.

### What the Teacher Actually Does

The teacher's job is **credit assignment**: for each token in the student's rollout, determine how much the student should have done differently. This is the logprob distribution p_teacher(token_t | context, PI).

In standard OPD, this credit assignment happens implicitly through one prefill pass. The teacher has no opportunity to:
- Analyze the student's reasoning strategy before scoring
- Identify the critical error point
- Consider what the student was trying to do vs what they should have done
- Allocate stronger signal to the tokens that actually matter

### Proposed: Deliberative Teaching

Instead of one-pass scoring, the teacher first *reasons* about how to teach, then scores:

```
Input:  problem + student rollout + PI
         ↓
Phase 1: Teacher generates reasoning trace (thinking tokens)
         - "The student started correctly but made an error at step 3..."
         - "The key insight they missed is..."
         - "Tokens 45-60 are where the reasoning breaks down..."
         ↓
Phase 2: Teacher scores rollout tokens, conditioned on its own analysis
         - p_teacher(token_t | problem, rollout, PI, reasoning_trace)
```

The reasoning trace becomes additional context for the prefill scoring. The teacher has literally "thought about" how to teach before producing the teaching signal.

### Connection to Multi-Lens

Our multi-lens results are evidence that this direction works, even in a crude form:

- Each lens (confidence, diagnosis, efficiency) is a **fixed, pre-specified reasoning template** — a frozen "thought" about one aspect of the student's work
- The token-level complementarity (r = 0.43-0.65) shows that different "thoughts" activate different tokens
- The oracle upper bound (+36% over best single lens) shows the value of combining multiple perspectives

But fixed templates are a poor substitute for actual reasoning. A deliberative teacher could:
1. **Dynamically generate** the right "lenses" for each specific rollout (not one-size-fits-all)
2. **Focus** on the specific failure mode present (not waste capacity on irrelevant criteria)
3. **Chain** insights (noticing the error leads to understanding why, which leads to better credit assignment)
4. **Scale**: more thinking tokens = better teaching, analogous to more reasoning tokens = better answers

### Spectrum of Teacher Compute

From cheapest to most expensive:

| Level | Teacher Compute | What We Know |
|---|---|---|
| **0. Bare prefill** | 1 forward pass, no PI | |KL| ≈ 0.005 (self), 0.097 (cross) |
| **1. Static PI** | 1 forward pass + fixed PI | |KL| = 0.014-0.065 depending on PI |
| **2. Self-generated PI** | Student generates PI, then 1 forward pass | |KL| = 0.042-0.046 (no external knowledge) |
| **3. Multi-lens (fixed)** | K forward passes with K fixed PI types | |KL| = 0.062 oracle (ref-free), 0.088 (mixed) |
| **4. Deliberative** | Teacher generates reasoning trace, then 1 forward pass | **CONFIRMED: blind 0.087 (asst_prefix), beats answer_ref (0.065)** |
| **5. Deliberative + best-of-N** | Generate N analyses, pick best | best-of-4 blind: 0.082 (8B self), 0.085 (32B self) |
| **6. Adaptive multi-lens** | Teacher generates K custom lenses per rollout, then K forward passes | Untested — predicted upper bound |

Each level trades more compute for better credit assignment. The question is whether the marginal signal quality justifies the cost — the same question that CoT answered affirmatively for inference.

### Experimental Results: Deliberative Teaching

The teacher generates a per-rollout analysis of the student's work (~1024 tokens), then scores with that analysis as PI. Tested with both 8B self-teacher (true self-OPD) and 32B.

**8B Self-Teacher (true self-OPD — same model generates, analyzes, and scores):**

| Condition | |KL| | vs answer_ref | Cohen's d |
|---|---|---|---|
| no_pi | 0.007 | 11% | 0.68 |
| answer_only | 0.013 | 21% | 0.69 |
| answer_ref (standard OPD) | 0.063 | 100% | 0.43 |
| informed_deliberative (with answer) | 0.073 | 117% | 0.34 |
| **blind_deliberative (no answer)** | **0.075** | **119%** | 0.40 |
| informed_delib + ref solution | 0.079 | 125% | 0.37 |
| **best-of-4 blind** | **0.082** | **131%** | 0.41 |

**Blind deliberative (0.075) beats answer_ref (0.063) by 19% — with zero external knowledge, using the same 8B model for everything.** Best-of-4 blind reaches 131% of answer_ref.

**32B Self-Teacher (PI effect only):**

| Condition | |KL| | vs answer_ref | Cohen's d |
|---|---|---|---|
| answer_only | 0.016 | 25% | 0.75 |
| answer_ref (standard OPD) | 0.065 | 100% | 0.54 |
| informed_deliberative (with answer) | 0.069 | 107% | 0.43 |
| **blind_deliberative (no answer)** | **0.072** | **111%** | 0.34 |
| informed_delib + ref solution | 0.075 | 116% | 0.40 |

**32B Cross-Teacher (8B student vs 32B teacher):**

| Condition | |KL| | Cohen's d |
|---|---|---|
| no_pi | 0.097 | 0.72 |
| answer_only | 0.098 | 0.74 |
| answer_ref | 0.119 | 0.67 |
| blind_deliberative | 0.117 | 0.51 |
| informed_deliberative | 0.117 | 0.61 |
| best-of-4 blind | 0.123 | 0.49 |
| informed_delib + ref | 0.121 | 0.60 |

**32B Self-OPD (32B generates, analyzes, and scores its own rollouts — true 32B self-OPD):**

90 AIME problems × 4 rollouts = 360 rollouts. 32B correctness: 23.3% (vs 8B: 14.0%).

| Condition | |KL| | vs answer_ref | Cohen's d |
|---|---|---|---|
| no_pi | 0.007 | 11% | 0.54 |
| answer_only | 0.016 | 25% | 0.62 |
| answer_ref (standard OPD) | 0.064 | 100% | 0.37 |
| informed_deliberative (with answer) | 0.071 | 112% | 0.11 |
| **blind_deliberative (no answer)** | **0.076** | **119%** | 0.03 |
| informed_delib + ref solution | 0.076 | 120% | 0.12 |
| **best-of-4 blind** | **0.085** | **133%** | 0.12 |

**32B self-OPD confirms the pattern**: blind deliberative (0.076) beats answer_ref (0.064) by 19%, matching the 8B result exactly. Best-of-4 blind reaches 133%.

**However**: Cohen's d drops dramatically — 0.03 for blind_deliberative (vs 0.40 for 8B). The 32B teacher amplifies signal uniformly across correct and incorrect rollouts. This may be because the 32B model is more capable (23.3% vs 14.0% correct) and its deliberative analyses are more "confident" regardless of correctness.

### Key Observations

1. **Blind deliberation beats oracle PI — on all three configurations**: 8B self-OPD: +19%. 32B self-teacher (scoring 8B rollouts): +11%. 32B self-OPD: +19%. The |KL| improvement is remarkably consistent.

2. **Informed deliberation barely helps over blind**: 8B: blind 0.075 vs informed 0.073. 32B self-teacher: blind 0.072 vs informed 0.069. 32B self-OPD: blind 0.076 vs informed 0.071. Knowing the answer doesn't significantly improve the analysis — the teacher's reasoning about the student's process is inherently informative.

3. **Best-of-N scales signal further**: 8B best-of-4 blind reaches 0.082 (131% of answer_ref). 32B self-OPD best-of-4 blind: 0.085 (133%). This validates inference-time scaling for teaching.

4. **Cohen's d trades off with |KL| — partially fixable via placement**: With system prompt placement, deliberative conditions have higher |KL| but lower Cohen's d (severe for 32B self-OPD: d=0.03). However, `assistant_prefix` placement recovers significant discrimination (d=0.49 for blind_deliberative, up from d=0.34 with system placement). The remaining gap vs answer_only (d=0.72) may reflect that deliberative analysis is genuinely more uniform, or may further improve with training.

5. **Fully self-supervised OPD is viable**: The entire pipeline uses a single model with no external knowledge: student generates rollout → same model reasons about rollout → same model scores with reasoning as context → student trains. Both 8B (0.075-0.082) and 32B (0.076-0.085) self-OPD exceed standard OPD with oracle PI.

6. **|KL| is model-size-invariant, Cohen's d is not**: The PI-driven |KL| gain is nearly identical across model sizes (~0.076 for deliberative, ~0.064 for answer_ref). But the discriminative power (Cohen's d) degrades significantly for the more capable model. This may reflect that the 32B model's deliberative analyses are equally detailed for correct and incorrect rollouts.

## PI Placement: Where Should Privileged Information Go?

### Motivation

In standard OPD, PI is appended to the system prompt — before the user's problem and the student's rollout. For static PI (answer, reference solution), this is natural. But for deliberative analysis — a commentary *on* the student's specific rollout — placing it before the problem is unnatural. The teacher reads an analysis of something it hasn't seen yet.

We tested four placement positions for PI in the prefill token sequence:

1. **system**: `[system: prompt + PI] [user: problem] [assistant: rollout]` — current default
2. **system_with_question**: `[system: prompt + problem + PI] [user: problem] [assistant: rollout]` — PI sees the problem in system context
3. **user**: `[system: prompt] [user: problem + PI] [assistant: rollout]` — PI after problem, before response
4. **assistant_prefix**: `[system: prompt] [user: problem] [assistant: PI + rollout]` — PI as preamble to the response

### Results (32B self-teacher, no_pi baseline)

| PI Condition | system | sys+question | user | asst_prefix |
|---|---|---|---|---|
| answer_only | 0.016 (d=0.72) | 0.023 (d=0.63) | 0.018 (d=0.73) | 0.029 (d=0.42) |
| answer_ref | 0.065 (d=0.54) | 0.066 (d=0.56) | 0.073 (d=0.55) | 0.068 (d=0.54) |
| **blind_deliberative** | 0.072 (d=0.34) | 0.069 (d=0.32) | 0.074 (d=0.24) | **0.087 (d=0.49)** |
| **informed_deliberative** | 0.070 (d=0.43) | 0.068 (d=0.35) | 0.073 (d=0.30) | **0.082 (d=0.46)** |

Token-level correlation between placements (for blind_deliberative): system↔user r=0.93, system↔asst_prefix r=0.19, user↔asst_prefix r=0.19. The assistant_prefix placement produces a fundamentally different per-token signal.

### Key Findings

1. **For deliberative PI, `assistant_prefix` is the clear winner**: +21% |KL| over system placement (0.087 vs 0.072) AND +44% Cohen's d (0.49 vs 0.34). Both signal strength and discrimination improve. This partially resolves the Cohen's d concern from earlier experiments.

2. **For static PI (answer_ref), placement barely matters**: All four placements land in 0.065-0.073 range. Factual PI (answer + solution) works similarly regardless of where it sits in the context.

3. **`system_with_question` doesn't help**: Slightly worse than plain `system` for deliberative PI. Repeating the problem in the system prompt alongside the analysis adds no value.

4. **`assistant_prefix` produces a fundamentally different signal**: Token-level correlation with system/user is only r=0.19, meaning the teacher activates on entirely different tokens when the analysis is part of the response vs part of the prompt. This makes sense mechanically — the analysis tokens are in the same "response generation" mode as the rollout tokens, so the teacher's attention flows directly from diagnostic reasoning into scoring.

5. **The placement effect is specific to analytical PI**: Static PI (answer_only, answer_ref) shows modest or no improvement from assistant_prefix. The benefit comes specifically from deliberative analysis — content that reasons about the student's work and naturally belongs as a preamble to the response.

### Implications for Training

The training pipeline should inject deliberative analysis as an **assistant response prefix**, not as a system prompt. The token sequence for teacher scoring becomes:

```
[system: math prompt] [user: problem] [assistant: {analysis}\n\n--- STUDENT RESPONSE ---\n{rollout}]
```

The teacher logprobs are extracted only for the student rollout tokens (after the analysis prefix), ensuring the training signal aligns with the student's actual token sequence.

## Open Questions

1. ~~**Deliberative teaching**: Does teacher reasoning improve credit assignment?~~ **YES — blind deliberative (0.072) beats answer_ref (0.065).**
2. ~~**Self-supervised loop**: Can the pipeline work without ground truth?~~ **YES — blind deliberative needs no external knowledge and exceeds oracle PI.**
3. **Downstream validation**: Does deliberative signal translate to actual training improvement? The |KL| is higher but Cohen's d is lower — does the student learn better? **This is the most important open question** — especially given the near-zero Cohen's d for 32B self-OPD.
4. **Scaling law**: Is there a smooth relationship between teacher analysis budget (tokens) and signal quality? We tested ~1024 tokens — what about 256, 512, 2048, 4096?
5. **Multi-lens + deliberative**: Can we combine deliberative analysis with multi-lens (generate K different analyses per rollout, use per-token max)? Predicted to further improve.
6. ~~**Cohen's d gap**: Is placement the main factor?~~ **PARTIALLY RESOLVED — assistant_prefix placement improves Cohen's d from 0.34 to 0.49 for blind_deliberative.** Remaining gap vs answer_only (d=0.72) needs training validation to determine if it matters.
7. **Teacher training**: Can we train the teacher to generate better analyses? (Distillation from oracle, proxy reward RL, self-refinement — see IDEATION.md)
8. **Multi-turn / other domains**: Does this apply to ARC-AGI / code generation?
9. **Model families**: All results are Qwen3. Generalization?

