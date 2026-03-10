# Multi-Lens Teacher Scoring for On-Policy Distillation

Research notes on the idea of conditioning the OPD teacher on multiple evaluation criteria (prompts/rubrics) to produce distinct per-token distributions, each targeting a different failure mode in the student's rollout.

Status: **Hypothesis stage** — needs empirical validation before any training experiments.

---

## 1. Motivation

### The Problem with Single-Lens Scoring

In standard on-policy self-distillation (e.g., OPSD), the teacher is conditioned on a single piece of privileged information — typically the ground-truth answer — and produces **one** token-level distribution over the student's rollout. This distribution implicitly conflates all the ways the rollout is deficient: it simultaneously tries to "fix" incorrect reasoning, unnecessary steps, suboptimal approaches, and missing edge cases.

But these are *different* failure modes with *different* remedies. A rollout might arrive at the correct answer via an unnecessarily long derivation. The teacher's single distribution must somehow encode "the reasoning is fine but you should skip steps 3-7" — a nuanced signal that may get averaged out or dominated by other corrections.

### The Proposal: Decomposed Teacher Scoring

Instead of one teacher pass, run **K teacher passes** with different scoring prompts (lenses), each emphasizing a different evaluation criterion:

```
Teacher_correctness(· | x, PI, "focus on reasoning validity")  → dist_1
Teacher_efficiency(· | x, PI, "focus on unnecessary steps")    → dist_2
Teacher_approach(· | x, PI, "focus on method choice")          → dist_3
```

Each pass produces a different per-token distribution. The training loss becomes:

```
Loss = Σ_k  w_k * KL(student || teacher_k)
```

where weights `w_k` may be fixed, learned, or curriculum-scheduled.

### Why This Could Work

1. **Criterion-specific credit assignment.** Different lenses activate at different tokens. A correctness lens has strong gradients at reasoning-error tokens. An efficiency lens has strong gradients at redundant-step tokens. The KL loss naturally focuses each criterion's gradient on the tokens where it matters — something a single blended distribution cannot do.

2. **Richer privileged information.** Ground-truth answers are weak PI. A reference solution (the canonical solution from the dataset) gives the teacher much richer context for scoring. The teacher can see not just *what* the answer is, but *how* a good solution arrives there.

3. **Controllable training dynamics.** You could emphasize correctness early (fix basic errors), then shift to efficiency and style once reasoning is sound. This is a curriculum over evaluation criteria.

### The Core Hypothesis

> **H1:** The same self-distillation teacher, conditioned on different scoring prompts (while holding PI fixed), produces meaningfully different per-token distributions over the student's rollout. These distributions diverge most at tokens corresponding to the criterion each prompt targets.

If H1 is false — if teacher distributions are nearly identical regardless of scoring prompt — the idea has no foundation and we stop here or rethink.

---

## 2. Background and Related Work

### On-Policy Self-Distillation (Context)

See [overview.md](../papers/overview.md) for the full landscape. The key papers for this idea:

- **OPSD** (Zhao et al., 2026): Frozen teacher conditioned on ground-truth answer. Single-pass scoring. Simplest self-distillation baseline.
- **SDFT** (Shenfeld et al., 2026): EMA teacher conditioned on ICL demos. Richer PI than answer-only, but still single evaluation lens.
- **GKD** (Agarwal et al., 2024): Flexible divergence framework (fwd KL, rev KL, JSD). Shows divergence choice matters. But doesn't vary what the teacher attends to.
- **RLAD** (Zhang et al., 2026): Shows unconditional teacher KL can *conflict* with reward. Selective imitation: only apply KL when it helps. Key insight: not all teacher signal is equally useful. Paper: arXiv:2602.22495.

### Rubric-Based Reward Models (Adjacent Field)

A very active area that decomposes monolithic evaluation into multi-criteria rubrics — but for **scalar RL rewards**, not distributional teacher signal:

- **Rubrics as Rewards (RaR)** (arXiv:2507.17746): Multi-criteria rubric feedback for RL beyond verifiable domains. Up to +31% on HealthBench vs. scalar Likert rewards.
- **Rubric-ARM** (arXiv:2602.01511): Joint optimization of rubric generator + judge. Shows fixed/frozen rubrics underperform learned ones. +4.7% average on reward benchmarks.
- **OpenRubrics** (arXiv:2510.07743): Contrastive Rubric Generation from preferred/rejected pairs. Rubric-RM outperforms baselines by 8.4% across 8 benchmarks.
- **RM-R1: Reward Modeling as Reasoning** (arXiv:2505.02387): Dynamically generates task-specific rubrics, scores against them. Closest to "prompt-conditioned" evaluation. But produces scalar rewards, not teacher distributions.
- **Chasing the Tail** (arXiv:2509.21500): Rubric-based rewards as remedy for reward over-optimization in high-reward tail.
- **Rubric-Scaffolded RL** (arXiv:2508.16949): Uses rubric scaffolding for exploration in open-ended tasks.

### Multi-Objective Reward Models

- **ArmoRM** (arXiv:2406.12845): Multi-dimensional reward heads (honesty, verbosity, safety, etc.) with MoE gating that weights criteria by context. Shows context-dependent weighting matters. But reward model, not distillation teacher.
- **MAH-DPO** (arXiv:2510.01167): Vectorized rewards where each dimension is a different objective. Multi-Action-Head DPO for fine-grained control. But DPO-based, not on-policy distillation.

### Other Related

- **Compute as Teacher (CaT)** (arXiv:2509.14234): Self-proposed rubrics scored by LLM judge as reference-free supervision. Closest to combining rubrics with distillation, but uses scalar rubric scores.
- **MR-GSM8K** (arXiv:2312.17080): Meta-reasoning benchmark where models score solutions (not just solve). Multi-criteria scoring with sub-metrics. Evaluation benchmark, not training method.

### What's Novel Here

The specific combination absent from the literature:

> Same model, same PI, but **multiple prompt-conditioned forward passes** that produce **different per-token distributions**, each emphasizing a different evaluation criterion. The student trains against a mixture (or curriculum) of these distributions.

This differs from:
- Multi-teacher KD (different model weights)
- Multi-objective RL rewards (scalar, not distributional)
- Rubric-based RL (scalar scores, not token distributions)
- ArmoRM-style multi-head rewards (different heads, but not KL-based distillation)

---

## 3. Experiment Protocol

### Guiding Principle

Start with the cheapest possible test of H1. Only proceed to training experiments if the hypothesis survives.

### Stage 0: Qualitative Pre-Test (Text-Level)

**Goal:** Sanity check — do different scoring prompts produce different *textual* critiques of the same rollout?

**Setup:**
- Pick 20 MATH problems (5 each from difficulty levels 2-5)
- Generate 4 rollouts per problem from Qwen3-8B (or similar)
- Select rollouts that are: (a) correct but verbose, (b) correct and clean, (c) incorrect, (d) partially correct
- For each rollout, prompt the teacher (same model + ground-truth answer + reference solution) with 4 different critique prompts:

```
Prompt A (Correctness):
"You are reviewing a student's math solution. The correct answer is {answer}.
Here is the reference solution: {ref_solution}
Here is the student's solution: {student_solution}
Identify any reasoning errors, invalid logical steps, or incorrect calculations."

Prompt B (Efficiency):
"You are reviewing a student's math solution. The correct answer is {answer}.
Here is the reference solution: {ref_solution}
Here is the student's solution: {student_solution}
Identify any unnecessary steps, redundant calculations, or places where the solution
could be significantly shorter while remaining correct."

Prompt C (Approach Quality):
"You are reviewing a student's math solution. The correct answer is {answer}.
Here is the reference solution: {ref_solution}
Here is the student's solution: {student_solution}
Evaluate whether the student chose the best mathematical technique for this problem.
Are there more elegant or standard approaches they should have used?"

Prompt D (Robustness):
"You are reviewing a student's math solution. The correct answer is {answer}.
Here is the reference solution: {ref_solution}
Here is the student's solution: {student_solution}
Check whether the solution properly handles edge cases, domain restrictions,
and implicit assumptions. Does it verify the answer?"
```

**What to look for:**
- Do different prompts identify different tokens/steps as problematic?
- Or do all critiques converge on the same issues regardless of prompt?
- Annotate manually: which steps does each critique flag?

**Decision gate:** If 4 prompts produce essentially identical critiques for >80% of rollouts, reconsider the approach. If critiques diverge meaningfully, proceed to Stage 1.

**Cost:** ~20 problems * 4 rollouts * 4 prompts = 320 LLM inference calls. Cheap.

### Stage 1: Distribution Divergence Measurement

**Goal:** Quantify whether different scoring prompts produce different *distributional* (logit-level) teacher signals.

**Dataset:** MATH-500 (standard 500-problem subset of Hendrycks MATH). Well-studied, used by OPSD, RLTF, and others. Mix of difficulty levels. Reference solutions available.

**Why MATH over alternatives:**
- Base solve rate for 8B models: ~70-80% → plenty of both correct and incorrect rollouts
- Solutions naturally vary along correctness, efficiency, and approach quality
- Reference solutions available for every problem (richer PI than answer-only)
- Directly comparable to OPSD, RLTF, GKD results
- Single-turn (no multi-turn confounds)

**Setup:**

1. **Student model:** Qwen3-8B (used by OPSD, SDPO, pi-Distill — well-calibrated baseline)
2. **Generate rollouts:** Sample N=8 completions per problem (temperature ~0.7), using the model's standard math prompt template
3. **Teacher scoring:** For each rollout, run the teacher (same model) in **prefill mode** (no generation, just compute logprobs over the student's tokens) under 4 conditioning prompts

The teacher prompt structure for each lens:

```
# Lens A: Correctness
System: "You are evaluating a math solution for correctness. Focus on whether
each reasoning step logically follows from the previous one."
User: "Problem: {problem}\nCorrect answer: {answer}\n\n
The student wrote the following solution:\n{student_rollout}"

# Lens B: Efficiency
System: "You are evaluating a math solution for efficiency. Focus on whether
there are unnecessary or redundant steps."
User: "Problem: {problem}\nReference solution: {ref_solution}\n\n
The student wrote the following solution:\n{student_rollout}"

# Lens C: Approach Quality
System: "You are evaluating a math solution for approach quality. Focus on
whether the mathematical technique chosen is the most appropriate."
User: "Problem: {problem}\nReference solution: {ref_solution}\n\n
The student wrote the following solution:\n{student_rollout}"

# Lens D: Answer-Only Baseline (standard OPSD)
System: "You are a math tutor."
User: "Problem: {problem}\nCorrect answer: {answer}\n\n
The student wrote the following solution:\n{student_rollout}"
```

Lens D is the OPSD baseline — minimal conditioning, just the answer. Comparing A/B/C against D tells us whether richer/targeted prompts add information beyond standard OPSD.

4. **Collect logprobs:** For each (rollout, lens) pair, extract the full vocabulary distribution at every token position.

**Metrics:**

For each token position `t` in each rollout:

| Metric | What It Tells Us |
|--------|-----------------|
| `JSD(p_A(t), p_B(t))` (pairwise across all lens pairs) | Are teacher distributions different under different lenses? |
| `mean_t[JSD(p_A(t), p_D(t))]` | Does a specific lens differ from the OPSD baseline? |
| `max_t[JSD(p_A(t), p_B(t))]` | Are there specific tokens where lenses sharply disagree? |
| `H(p_k(t)) - H(p_student(t))` | Does the teacher become more/less confident than the student? |
| Correlation of high-JSD positions across lens pairs | Do different lenses activate at different tokens? |

**Visualization:**
- Heatmap: rows = token positions, columns = lens pairs, color = JSD value
- Overlay with rollout text to see which *words/steps* correspond to high divergence
- Scatter: JSD(A,B) vs JSD(A,C) at each token — if correlated, lenses aren't providing independent signal

**Decision gate:**
- **Proceed** if mean pairwise JSD across lenses is >0.01 nats AND high-JSD tokens are non-randomly distributed (cluster at specific steps)
- **Stop/rethink** if JSD is uniformly near zero, or if high-JSD tokens are randomly scattered

**Cost:** 500 problems * 8 rollouts * 4 lenses = 16,000 forward passes. Feasible on a single GPU in a few hours with vLLM batched prefill.

### Stage 2: Ablation — Does Richer PI Help?

**Goal:** Separate two effects: (a) does the *scoring prompt* matter? (b) does having a *reference solution* (vs. just the answer) matter?

**Design:** 2x2 factorial:

| | Answer-only PI | Reference-solution PI |
|---|---|---|
| **Generic prompt** (OPSD baseline) | Condition A | Condition B |
| **Criterion-specific prompt** (e.g., correctness) | Condition C | Condition D |

Compare distributions across conditions. This tells us whether the benefit (if any) comes from the prompt, the PI, or the interaction.

### Stage 3: Training Experiment (Only if Stages 0-2 pass)

**Goal:** Does training with multi-lens KL loss outperform single-lens KL loss?

**Experimental conditions:**

| Condition | Description |
|-----------|------------|
| **Baseline: GRPO** | Standard RL, no teacher |
| **Single-lens OPD (OPSD)** | Teacher conditioned on answer only, single pass |
| **Single-lens OPD + ref solution** | Teacher conditioned on answer + reference solution, single pass |
| **Multi-lens OPD (equal weights)** | K=3 lenses, w_k = 1/K each |
| **Multi-lens OPD (curriculum)** | Phase 1: correctness only → Phase 2: add efficiency → Phase 3: add approach |
| **Multi-lens OPD (learned weights)** | Per-step or per-problem weighting (more complex) |

**Training setup:**
- Model: Qwen3-8B
- Training data: OpenThoughts or MATH train split (following OPSD)
- Evaluation: MATH-500, AIME24, HMMT25 (following OPSD)
- Compute: same total forward passes across conditions (multi-lens uses K passes but fewer training steps, to keep FLOPs comparable)

**Metrics:**
- Accuracy (pass@1, avg@8, avg@16) on eval sets
- Solution length (tokens) — does multi-lens produce shorter solutions?
- Entropy at decision points — does multi-lens maintain healthier exploration?
- Catastrophic forgetting: MMLU-Pro, IFEval (standard holdout checks)

---

## 4. Open Questions

### Fundamental

1. **Does prompt conditioning change distributions enough?** The LLM may largely ignore the scoring prompt in prefill mode, producing similar logprobs regardless. The teacher's distribution during prefill is determined by attention patterns, and a system prompt about "efficiency" may not meaningfully alter attention over the student's math tokens. This is the Stage 1 question.

2. **Is this better than just using richer PI?** Maybe the entire benefit comes from giving the teacher a reference solution (vs. answer-only), and the prompt conditioning adds nothing on top. The Stage 2 factorial design addresses this.

3. **Can you aggregate multiple distributions meaningfully?** Averaging KL losses from K distributions is not the same as KL from the average distribution. The gradients may conflict — lens A wants to increase P(token X) while lens B wants to decrease it. Is this a feature (regularization) or a bug (gradient noise)?

### Design Choices

4. **How many lenses?** More lenses = richer signal but more compute (K forward passes per rollout). Diminishing returns likely kick in fast. Start with K=3.

5. **Fixed vs. adaptive weights?** ArmoRM shows context-dependent weighting (MoE gating) outperforms fixed weights for multi-objective rewards. Same may apply here. But adds complexity.

6. **Curriculum over criteria?** Intuition: teach correctness first, then efficiency. But no evidence yet. Compare fixed-weight vs. curriculum vs. learned weights in Stage 3.

7. **Which divergence?** GKD shows divergence choice (fwd KL, rev KL, JSD) interacts with the task. Does it also interact with the scoring lens? Maybe correctness benefits from mode-seeking (rev KL) while efficiency benefits from mode-covering (fwd KL).

### Connections to Other Ideas

8. **Relationship to process reward models.** PRMs score intermediate steps. Multi-lens teacher scoring scores the *same* steps from *different angles*. Could a multi-lens teacher subsume PRMs? Or are they complementary (PRM for step-level, multi-lens for criterion-level)?

9. **Relationship to RLAD.** RLAD's selective imitation asks "does the teacher direction help the reward?" Multi-lens scoring could be more fine-grained: "does the teacher's *correctness* direction help? does its *efficiency* direction help?" You could selectively apply each lens only when it's beneficial — RLAD per criterion.

10. **Relationship to Idea D in [research-questions-opd.md](../research-questions-opd.md).** Reflective retry produces a corrected trajectory that naturally encodes multiple lessons (what was wrong, what was unnecessary, what approach works better). Multi-lens scoring is an alternative decomposition: instead of one trajectory that implicitly encodes everything, K distributions that explicitly separate each concern.

---

## 5. Practical Considerations

### Compute Cost

Multi-lens scoring requires K forward passes per rollout instead of 1. For K=3:
- Training time: ~3x the teacher scoring cost per step
- But teacher scoring (prefill) is fast compared to student generation (autoregressive sampling)
- In typical OPD, teacher prefill is ~10-20% of total step time → K=3 adds ~20-40% overhead
- Compare to the alternative: running K times more rollouts for RL. Multi-lens may be cheaper per bit of signal.

### Implementation in Existing Frameworks

The teacher scoring infrastructure in [prime-rl](prime-rl-opd-implementation.md) already supports:
- Separate teacher inference server
- Prefill-mode logprob extraction
- Blending teacher signal into loss via `teacher_tau`

Multi-lens extension would require:
- Multiple prefill calls per rollout (or batching K prompts)
- Aggregation of K loss terms with weights
- Logging per-lens KL values for monitoring

### Prompt Engineering Risk

The quality of this approach depends on prompt engineering — the scoring prompts must actually steer the teacher's attention differently. This is fragile and may not transfer across model families. A learned approach (fine-tuning the teacher to score along specific dimensions) would be more robust but requires additional training data.

---

## 6. Summary

| Aspect | Detail |
|--------|--------|
| **Core idea** | Condition the OPD teacher on K different scoring prompts to produce K distributions, each targeting a different failure mode |
| **Key hypothesis** | Different prompts produce meaningfully different distributions (H1) |
| **Dataset** | MATH-500 (Stage 0-2), OpenThoughts + MATH train (Stage 3) |
| **Model** | Qwen3-8B |
| **Cheapest test** | Stage 0: text-level critiques, 320 inference calls |
| **Kill criterion** | If teacher distributions are nearly identical across lenses (JSD ~0), stop |
| **Novelty claim** | Multi-criteria rubric scoring (proven for RL rewards) applied to distributional teacher signal in OPD |
| **Risk** | Prompt conditioning may not meaningfully change prefill-mode distributions |

---

## 7. References

### On-Policy Distillation Papers
- See [papers/overview.md](../papers/overview.md) for the full landscape and citations
- OPSD (Zhao et al., 2026): arXiv:2601.18734
- GKD (Agarwal et al., 2024): arXiv:2306.13649
- RLAD (Zhang et al., 2026): arXiv:2602.22495

### Rubric-Based Reward Models
- Rubrics as Rewards (RaR): arXiv:2507.17746
- Rubric-ARM: arXiv:2602.01511
- OpenRubrics: arXiv:2510.07743
- RM-R1: arXiv:2505.02387
- Chasing the Tail: arXiv:2509.21500
- Rubric-Scaffolded RL: arXiv:2508.16949

### Multi-Objective Rewards
- ArmoRM: arXiv:2406.12845
- MAH-DPO: arXiv:2510.01167

### Other
- Compute as Teacher (CaT): arXiv:2509.14234
- MR-GSM8K: arXiv:2312.17080
- Research questions context: [research-questions-opd.md](../research-questions-opd.md)
