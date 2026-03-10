# Research Notes: Privileged Information Distillation for Language Models

**Paper:** "Privileged Information Distillation for Language Models"
**Authors:** Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, Massimo Caccia
**Affiliations:** ServiceNow, Mila Quebec, Universite de Montreal, McGill University, HEC Montreal
**arXiv:** [2602.04942](https://arxiv.org/abs/2602.04942) (v3, 16 Feb 2026)
**Code:** [github.com/Emilianopp/Privileged-Information-Distillation](https://github.com/Emilianopp/Privileged-Information-Distillation) (pending legal approval, code not yet released as of Feb 2026)
**Compute:** ~100,000 GPU hours on 2x H100 GPUs per experiment

---

## 1. Problem Statement

When distilling frontier models (e.g., DeepSeek-chat-v3.1) for multi-turn agentic tasks, the frontier model's Chain-of-Thought (CoT) reasoning is often hidden. Providers expose only action trajectories, not the internal reasoning process. The standard distillation pipeline (SFT on CoT traces followed by RL) breaks down because we only observe *what* successful agents do, not *how* they reason.

The core question: **Can we transfer knowledge from a policy trained with privileged information (PI) to a test-time policy that operates without it?**

This is formalized via Vapnik & Vashist (2009)'s learning using privileged information (LUPI) paradigm, adapted to the LLM setting.

---

## 2. Background: Agentic MDP Formulation

The paper formalizes multi-turn agentic interactions as a Markov Decision Process (MDP):
- Policy: `pi_theta(o | s) = prod_{i=0}^{T} pi_theta(z_i, a_i | s_{<i})`
- `o = (z, a)` consists of reasoning tokens `z` and action tokens `a`
- `s` is the evolving interaction context (user prompt + past outputs + environment responses)
- State transitions: `s_{t+1} ~ P(. | s_t, o_t)`
- Reward: `R(o, s) in [-1, 1]`

**Policy Optimization:** Uses GRPO (Group Relative Policy Optimization) without the base-model KL penalty term (following Shah et al., 2026). For each state, G trajectories are sampled; group-relative advantage `A_{s,g}` is computed by comparing each trajectory's return to the group average. The GRPO objective uses clipped importance-weighted policy updates:

```
J_GRPO(theta) = E[1/sum_g(K_g) * sum_{g,k} min(rho_{g,k} * A_{s,g}, clip(rho_{g,k}, 1-eps, 1+eps) * A_{s,g})]
```

where `rho_{g,k} = pi_theta(o_{g,k} | s_i, o_{g,<k}) / mu(o_{g,k} | s_i, o_{g,<k})` is the token-level importance ratio.

---

## 3. Method 1: Privileged Information Distillation (pi-Distill)

### 3.1 Core Idea

A single shared-parameter model `theta` acts as both:
- **Teacher** `pi^T_theta(o | s, I)` -- conditioned on privileged information `I`
- **Student** `pi^S_theta(o | s)` -- operates without PI

Both are trained jointly using a convex combination of two objectives. The key insight is that **shared parameters enable implicit transfer**: training one role improves the other, even without direct optimization.

### 3.2 Teacher Objective (Eq. 2)

```
J_Teacher(theta) = E_{o ~ pi^T(o|s,I), s~P} [R(o, s) - beta * D_KL(pi^T_theta(o | s, I) || sg(pi^S_theta(o|s)))]
```

- Samples trajectories from the teacher (conditioned on PI)
- Maximizes reward while staying close to the student via **reverse KL**
- `sg(.)` denotes stop-gradient
- `beta` controls regularization strength

The reverse KL penalty serves two purposes:
1. Encourages the teacher to produce traces that are *familiar* to the student (more on-policy for distillation)
2. Shared parameters promote transfer of teacher knowledge to student even without directly training the student

### 3.3 Student Objective (Eq. 3)

```
J_Student(theta) = E_{o ~ pi^T(o|s,I), s~P} [(pi^S_theta(o|s) / sg(pi^T_theta(o | s, I))) * R(o, s) - beta * D_KL(sg(pi^T_theta(o | s, I)) || pi^S_theta(o|s))]
```

- Also samples from teacher trajectories, but updates only the student
- Uses importance weighting `pi^S / pi^T` to correct for off-policy sampling
- KL term regularizes student toward teacher's distribution
- This is essentially off-policy learning from the teacher's high-reward behavior

### 3.4 Combined Objective (Eq. 4)

```
J_{pi-Distill}(theta) = alpha * J_Teacher(theta) + (1 - alpha) * J_Student(theta)
```

where `alpha in [0, 1]` controls the balance. Three key settings:

| Setting | alpha | Description | Behavior |
|---------|-------|-------------|----------|
| Teacher-only | 1.0 | Only teacher is directly optimized | Student improves through shared parameters; risk of policy collapse |
| Student-only | 0.0 | Only student is directly optimized | Learns from teacher traces; requires low initial KL |
| Joint training | 0.5 | Both optimized simultaneously | Most robust; mitigates failure modes of both extremes |

**Important implementation detail:** For `alpha = 0.5`, the paper uses **alpha annealing** -- linearly annealing `alpha: 0 -> 0.5` over 15 epochs. This means training starts student-focused and gradually incorporates teacher training.

### 3.5 Detailed Algorithm (Algorithm 1, Appendix C)

```
Input: Dataset D = {(s, I)}, Initial Policy pi_theta, Reference pi_ref, alpha, beta, epsilon
Initialize: phi <- theta  (parameters shared between teacher and student)

while not converged:
    Sample batch B = {(s_i, I_i)} ~ D

    // Step 1: Teacher Rollout (with PI)
    for each (s, I) in B:
        Sample K trajectories {o_1, ..., o_K} ~ pi^T(. | s, I)
        Compute rewards: R(o_k, s) = R_env(o_k, s) - beta * D_KL[pi^T(. | s, I) || pi_ref(. | s, I)]

    // Step 2: Group-Centered Advantages
    for each k in 1..K:
        R_bar = (1/K) * sum_j R(o_j, s)
        A_k = R(o_k, s) - R_bar

    // Step 3: Compute Objectives
    Teacher Objective (GRPO):
        J_Teacher = (1/K) * sum_k min(rho_k^teacher * A_k, clip(rho_k^teacher, 1-eps, 1+eps) * A_k)

    Student Objective (Off-Policy GRPO):
        Compute IS weights: rho_k = pi^S(o_k | s) / pi^T(o_k | s, I)  {Student input is s only}
        J_Student = (1/K) * sum_k min(rho_k * A_k, clip(rho_k, 1-eps, 1+eps) * A_k)

    // Step 4: Joint Update
    J_{pi-Distill}(theta) = alpha * J_Teacher + (1 - alpha) * J_Student
    theta <- theta + eta * grad(J_{pi-Distill})
```

### 3.6 Reference Model Choice

Critical ablation (Appendix D.2): Using the student itself `pi_theta` (with stop-gradient) as the reference for the KL term is far superior to using a fixed base model `pi_base`. When `pi_base` is used as reference:
- Performance degrades across all alpha settings
- Most severe for `alpha = 1` (teacher-only), where policy collapse occurs
- The policy is pushed far from base while still penalized for deviation

Using `pi_theta` (stop-gradient) keeps the KL regularizer aligned with the current student distribution. This is also **cheaper** -- no need to maintain a separate frozen reference model.

---

## 4. Method 2: On-Policy Self-Distillation (OPSD)

### 4.1 Core Idea

OPSD is an on-policy alternative where the **student** generates trajectories and is regularized toward the PI-conditioned teacher:

```
J_OPSD(theta) = E_{o ~ pi^S(o|s), s~P} [R(o, s) - beta * D_KL(pi^S_theta(o|s) || sg(pi^T_theta(o | s, I)))]
```

Key differences from pi-Distill:
- Trajectories are sampled from the **student** (on-policy), not the teacher
- The reverse KL acts as a dense per-token reward signal pulling student toward teacher
- No separate teacher optimization -- teacher only provides a "north star" signal
- Same shared-parameter setup

### 4.2 Algorithm (Algorithm 2, Appendix C)

```
Input: Dataset D = {(s, I)}, Initial Policy pi_theta, beta, epsilon, Learning rate eta

while not converged:
    Sample batch B = {(s_i, I_i)} ~ D

    // Step 1: Student Rollout (On-Policy sampling)
    for each (s, I) in B:
        Sample K trajectories {o_1, ..., o_K} ~ pi^S(. | s)
        Reward with Reverse KL: R(o_k, s) = R_env(o_k, s) - beta * D_KL[pi^S(. | s) || pi^T(. | s, I)]

    // Step 2: Group-Centered Advantages (per-sample)
    for each sample i:
        R_bar_i = (1/K) * sum_j R(o_{i,j}, s_i)
        for each k: A_{i,k} = R(o_{i,k}, s_i) - R_bar_i

    // Step 3: Objective (Off-Policy GRPO with importance correction)
    Compute IS weights: rho_{i,k} = pi^S_theta(o_{i,k} | s_i) / pi^S_old(o_{i,k} | s_i)
    J(theta) = (1/K) * sum_k min(rho_{i,k} * A_{i,k}, clip(rho_{i,k}, 1-eps, 1+eps) * A_{i,k})

    // Step 4: Policy Update
    theta <- theta + eta * grad(J(theta))
```

### 4.3 Concurrent Work

The OPSD objective was independently proposed by several concurrent papers:
- Zhao et al. (2026) -- "Self-distilled reasoner" (supervised reasoning)
- Shenfeld et al. (2026) -- "Self-distillation enables continual learning"
- Hubotter et al. (2026) -- "Reinforcement learning via self-distillation" (reflective self-feedback)

This paper's contribution is evaluating OPSD specifically for PI transfer without ground-truth answers, across diverse PI types, in multi-turn agentic settings.

---

## 5. Privileged Information: Three Granularity Levels

Frontier trajectories from DeepSeek-chat-v3.1 are transformed into three PI variants with increasing information density:

### 5.1 Tool Calls & Arguments (richest)

The complete sequence of actions including function names AND arguments:
```
1) id_by_email(email:"joe@email.com")
2) get_order(id=3)
3) ...
```
This provides the most direct signal about what to do and is closest to a "ground truth answer."

### 5.2 Tool Calls Only (medium)

Function names without arguments -- the model must infer correct parameters:
```
1) id_by_email
2) get_order
3) ...
```
Lower KL divergence because the student doesn't need to match exact arguments, just the right function sequence.

### 5.3 Self-Generated Hints (least dense)

The student model (after some training) is prompted to summarize a successful trajectory:
```
"Look up the user id from the email. Then, use it to retrieve the user's order details..."
```
This is a model-generated natural language summary. Excluded for R1-Distill-Llama-8B because it consistently just returned the raw trace or tool calls instead of a real summary.

### 5.4 Data Collection

- 15,885 successful traces for tau-Bench retail
- 1,986 for Travel Planner
- PI obtained for all 45 tasks (Travel Planner) and 300/500 tasks (tau-Bench)
- For each training task, selected the trajectory with the fewest steps

---

## 6. Connection to Variational EM (Appendix A)

### 6.1 Interpretation

pi-Distill can be viewed as a **joint variational Expectation-Maximization** algorithm:

**Target distribution** (reward-tilted posterior relative to reference `pi_ref`):
```
pi*(o | s) = pi_ref(o | s) * exp(R(o, s)) / Z
```
where `Z = sum_{o'} pi_ref(o' | s_0) * exp(R(o', s))` is the intractable partition function.

**E-step:** The teacher `pi^T_theta(o | s, I)` serves as a variational posterior that approximates `pi*`. Training the teacher minimizes `D_KL(pi^T || pi*)`, which yields (Eq. 8):
```
J_Teacher(theta) propto E_{pi^T} [R(o, s)] - D_KL(pi^T_theta(o | s, I) || pi_ref(. | s))
```

**M-step:** The student `pi^S` is fit to approximate the (intractable) `pi*` by instead fitting to the teacher's trajectories (Eq. 9):
```
J_SFT(theta) = E_{o ~ pi^T(o|s,I)} [log pi^S_theta(o | s)]
```

### 6.2 Why pi-Distill is Better than Sequential EM

Traditional approaches either:
1. Train teacher to convergence, then distill into student (sequential)
2. Alternate between E and M steps with separate models

pi-Distill's innovations over sequential EM:
- **Shared parameters** enable implicit transfer (training teacher directly improves student)
- **Joint optimization** avoids checkpoint selection and off-policy instability
- **Single training phase** instead of two-phase pipeline
- **Uses off-policy RL** (not just SFT) for the student, which leverages negative feedback on failed trajectories

Figure 11 (Appendix A.2) demonstrates this empirically on tau-Bench Retail with Qwen3-8B:
- Standard sequential EM (rejection fine-tuning) underperforms SFT w/ CoT
- Replacing RFT with off-policy RL enables EM to outperform SFT w/ CoT
- Simply optimizing `J_Teacher(theta)` with shared parameters drastically outperforms all other baselines

### 6.3 OPSD Variational Interpretation (Appendix B)

OPSD can be framed as minimizing the reverse KL between the student and a *PI-conditioned* target:
```
pi*(o | s, I) = pi_ref(o | s, I) * exp(R(o, s) / beta) / Z^h
```

Minimizing `D_KL(pi^S || pi*)` yields:
```
J_OPSD propto -E_{pi^S} [R(o, s)] + beta * D_KL(pi^S(. | s) || pi_ref(. | s, I))
```

Setting `pi_ref = pi^T_theta(o | s, I)` (the teacher) recovers the OPSD objective.

---

## 7. Experimental Setup

### 7.1 Benchmarks

| Benchmark | Domain | Training | Held-out | Details |
|-----------|--------|----------|----------|---------|
| tau-Bench Retail | Customer service (retail) | 500 tasks | 115 tasks | Tool-calling for order management |
| tau-Bench Airline | Customer service (airline) | -- | 50 tasks | OOD evaluation only |
| Travel Planner | Travel planning | 45 tasks | 180 tasks | Multi-step tool use for itinerary planning |
| GEM Suite | Multi-hop QA | -- | 7 datasets | 2Wiki, PopQA, TriviaQA, HotpotQA, Bamboogle, NaturalQuestions, Musique |

Key modifications to benchmarks:
- **tau-Bench:** Replaced GPT-4o user simulator with Qwen-14B; removed `transfer_to_human_agents` tool (consistently led to reward hacking)
- **Travel Planner:** Decoupled reward structure so easy constraints are evaluated individually with their corresponding hard constraints (original rubric caused reward hacking via the `Planner` tool shortcut)

### 7.2 Models

| Model | Family | Notes |
|-------|--------|-------|
| Qwen3-4B | Qwen | Smaller reasoning model |
| Qwen3-8B | Qwen | Primary evaluation model |
| R1-Distill-Llama-8B | Llama (R1-distilled) | Different family; warm-started with SFT w/ CoT |

R1-Distill-Llama-8B requires SFT warmstart because it fails to generate correct trajectories even when conditioned on PI, making direct RL training infeasible.

### 7.3 Baselines

1. **Base model** (no training)
2. **SFT w/ CoT** -- SFT on full frontier traces including CoT reasoning
3. **SFT w/o CoT** -- SFT on frontier actions only (no reasoning)
4. **Standard RL** -- GRPO without any PI
5. **SFT w/o CoT + RL** -- SFT on actions, then RL
6. **SFT w/ CoT + RL** -- SFT on full CoT traces, then RL (**industry standard**)

For SFT+RL baselines: sweep over multiple SFT checkpoints, report the best-performing final result. This is expensive but gives the strongest baseline.

### 7.4 Hyperparameters (Table 2, Appendix E.3)

| Parameter | Value |
|-----------|-------|
| Seeds | 3 |
| Rollout temperature | 0.75 |
| Trace length filter | Discard if >25k tokens (tau-Bench), >35k (RL/OPSD) |
| Advantage processing | Pop zero-advantage (always) |
| tau-Bench gradient steps | 600 |
| Travel Planner gradient steps | 400 |
| Gradient steps per sampling | tau-Bench = 3, TP = 2 |
| Repeats per group | tau-Bench = 5, TP = 4 |
| Training tasks sampled | tau-Bench = 64, TP = 45, SFT+RL (tau-Bench) = 128 |
| Learning rate sweep | {1e-6, 5e-6, 1e-5} |
| Final LR | tau-Bench: 5e-6 for all; TP: pi-distill = 1e-5, RL/OPSD = 5e-6 |
| beta | TP/OPSD = 0.5, pi-distill = 0.25 (unless swept) |
| Clipping epsilon | Lower = 0.8, Upper = 1.2 |
| Alpha annealing (alpha=0.5) | Linearly anneal alpha: 0 -> 0.5 over 15 epochs |

**Hardware:** 2x H100 GPUs, 25k token context limit.

**KL Estimation:** Rae-Blackwellized estimator (Amini et al., 2025). For pi-Distill, KL penalty is sequence-level and absorbed into the advantage computation. For OPSD, KL is estimated per-token and backpropagated directly.

**Length Penalty:** Cosine-shaped length penalty applied only to successful traces (r > 0). No penalty below 2,000 tokens; soft allowance up to 5,000 tokens; penalty increases with cosine schedule beyond that.

**PI Leakage Penalty:** Keywords like "privileged information", "hint", "secret" are detected in assistant messages; each occurrence incurs -0.1 penalty. In practice, this makes little difference to final performance but does reduce leakage rate.

---

## 8. Main Results (Table 1)

### 8.1 Qwen3-8B (strongest results)

| Method | Travel Planner | tau-Bench Retail | tau-Bench Airline (OOD) |
|--------|---------------|-----------------|------------------------|
| Base | 23.6% | 3.35% | 6.40% |
| SFT w/ CoT | 26.0% | 16.5% | 5.33% |
| SFT w/o CoT | 29.8% | 12.8% | 6.00% |
| RL | 27.5% | 23.9% | 6.67% |
| SFT w/o CoT + RL | 31.3% | 23.5% | 6.00% |
| OPSD | 37.5% | 27.3% | **14.0%** |
| **pi-Distill alpha=0** | **40.7%** | **31.1%** | 12.0% |
| **pi-Distill alpha=0.5** | 41.1% | 30.6% | 7.33% |
| **pi-Distill alpha=1** | **44.1%** | 29.7% | 9.33% |
| SFT w/ CoT + RL | 32.3% | 29.1% | 8.00% |

**Key findings:**
- pi-Distill consistently outperforms SFT w/ CoT + RL (the industry standard) **without access to any CoT reasoning traces**
- Best-case improvements: +11.8% on Travel Planner, +2.08% on tau-Bench Retail, +6.00% on tau-Bench Airline
- OPSD also outperforms SFT w/o CoT + RL on Travel Planner and tau-Bench
- Standard RL alone substantially underperforms pi-Distill variants

### 8.2 Qwen3-4B

| Method | Travel Planner | tau-Bench Retail | tau-Bench Airline (OOD) |
|--------|---------------|-----------------|------------------------|
| Base | 17.6% | 5.03% | 2.21% |
| OPSD | 29.8% | 23.1% | 10.6% |
| **pi-Distill alpha=0** | 28.5% | **25.3%** | 8.00% |
| **pi-Distill alpha=0.5** | **33.8%** | 22.6% | 6.00% |
| **pi-Distill alpha=1** | 28.2% | 22.5% | **12.0%** |
| SFT w/ CoT + RL | 26.4% | 23.3% | 6.67% |

Similar pattern: pi-Distill alpha=0.5 achieves 33.8% on Travel Planner vs 26.4% for SFT w/ CoT + RL.

### 8.3 R1-Distill-Llama-8B (with SFT warmstart)

Results are weaker overall. pi-Distill alpha=0.5 achieves 14.0% on Travel Planner (vs 12.4% for SFT w/ CoT + RL) and 18.6% on tau-Bench Retail (vs 16.3%). The model struggles because it cannot effectively leverage PI even when conditioned on it -- attributed to either extreme KL divergence or negative initial PI utility.

---

## 9. Out-of-Domain Generalization (Section 6)

### 9.1 GEM Benchmark Results (Figure 4)

Evaluation on 7 search-tool QA datasets. Best tau-Bench Retail checkpoint is selected, then evaluated across the entire GEM suite. Results reported as Pass@1 and Pass@10.

**Qwen3-4B:**
- SFT w/ CoT + RL is the top performer
- OPSD shows significant degradation
- pi-Distill variants competitive but don't consistently beat SFT w/ CoT + RL

**Qwen3-8B (the more capable model):**
- pi-Distill (alpha=0, alpha=0.5) and OPSD **significantly outperform** SFT w/ CoT + RL
- Both methods generalize better than the industry standard
- Standard RL consistently degrades relative to base model

**R1-Distill-Llama-8B:**
- SFT w/ CoT + RL drops *below* the base model (performance degradation)
- pi-Distill and OPSD do not improve over base model but avoid degradation

### 9.2 Key Takeaway: Scaling Benefits

The OOD results reveal a model-size-dependent pattern:
- **Smaller models (4B):** Benefit more from explicit CoT supervision (SFT w/ CoT + RL)
- **Larger models (8B):** Benefit more from on-policy methods (pi-Distill, OPSD)
- Interpretation: stronger reasoners can generate better self-feedback via on-policy exploration, making explicit CoT traces less necessary

Standard RL consistently *degrades* OOD performance across all benchmarks and models. pi-Distill avoids this degradation entirely.

---

## 10. What Matters for pi-Distill (Section 7.1)

The paper identifies two primary drivers of success:
1. **D_KL(pi^T_base || pi^S_base)** -- initial divergence between conditioned and unconditioned base policies
2. **Delta = score(pi^T_base) - score(pi^S_base)** -- initial PI utility (how much PI helps the base model)

Additionally: **Delta_max = max_t score(pi^PI_t) - max_t score(pi^RL_t)** -- maximum attainable improvement from PI over pure RL.

### 10.1 Teacher-Only Training (alpha = 1)

- Performance generally **declines or maintains** as initial KL increases
- **Failure mode: Policy Collapse.** When KL is too low, teacher and student collapse onto each other (pi^T approx pi^S), causing the teacher to ignore PI entirely. Figure 7 shows KL dropping to near zero during training even with beta=0, indicating collapse.
- **Key finding:** Even when initial Delta < 0 (PI hurts base model), teacher training can *learn to leverage PI*, showing positive Delta_max values. Learning to use PI is a significant contributing factor.
- Requires beta > 0 to prevent collapse; non-zero KL penalty is crucial for stabilizing training.

### 10.2 Student-Only Training (alpha = 0)

- **Low KL divergence is a strong predictor of success.** When teacher traces are close to student distribution, learning is easy.
- Example: tau-Bench with "Tool Calls Only" as PI consistently yields best results for alpha=0 because the minimal distribution shift makes the off-policy traces learnable.
- **Fails when Delta < 0** (negative PI utility). On Travel Planner with Qwen3-8B, alpha=0 underperforms significantly because PI provides negative utility.
- Interestingly, when teacher-student KL is low, student-only training can **transfer knowledge back to the teacher** even though the teacher is not directly trained. The low KL allows reverse transfer via shared parameters.

### 10.3 Joint Training (alpha = 0.5) -- MOST ROBUST

- **Best performance in 7 out of 16 scenarios**
- **Only ranked worst once** (across all settings)
- Effectively avoids the failure modes of both teacher-only (collapse) and student-only (high KL / negative utility)
- Balances both objectives: teacher learns to exploit PI while student learns from improved traces
- **Recommended default** when lacking multiple types of PI or when sweeping alpha is infeasible

### Summary Table: When Each Setting Works

| Setting | Works when... | Fails when... |
|---------|--------------|---------------|
| alpha=1 (teacher) | High initial utility or can learn PI; KL not too low | KL too low (collapse); cannot learn to use PI |
| alpha=0 (student) | Low KL; positive Delta | High KL (off-policy traces too different); Delta < 0 |
| alpha=0.5 (joint) | Almost always | Rarely the worst; only when one extreme is clearly dominant |

---

## 11. What Matters for OPSD (Section 7.2)

OPSD behaves differently from pi-Distill:

- **Information content is the primary predictor**, not KL divergence
- **Tool Calls & Arguments (richest PI) consistently performs best** in most settings
- Unlike pi-Distill, higher KL is **not always detrimental** for OPSD -- richer information can compensate
- **Exception:** Qwen3-8B on tau-Bench where Tool Calls & Arguments shows highest KL and Delta_max is negative, causing the reverse-KL penalty to override positive utility benefits
- R1-Distill-Llama-8B consistently struggles with OPSD, attributed to extreme KL divergence
- **Model-size dependent:** OPSD struggles on smaller models (Qwen3-4B) but shows substantial gains for Qwen3-8B, especially in OOD settings

---

## 12. Ablation on beta (Section 8)

`beta` controls the KL regularization strength between teacher and student.

**Sweep:** beta in {0, 0.1, 0.25, 0.5}

### For pi-Distill:
- **beta > 0 is important in 17/21 ablated configurations**
- Most critical when teacher is being trained (alpha > 0)
- For alpha = 0 (student only), beta matters less
- No single best beta value, but beta > 0 generally matches or outperforms beta = 0
- beta = 0 with alpha = 1 leads to policy collapse (Figure 7, 9)

### For OPSD:
- beta is **less important** than information granularity and initial student-teacher KL
- Information content and KL divergence are more predictive factors

---

## 13. Additional Ablations and Implementation Details

### 13.1 PI Leakage (Appendix D.1)

- Keyword-based detection of PI leakage in student outputs
- Leakage rate increases during training across all PI types
- Tool Calls & Arguments shows highest leakage; Self-Generated Hints lowest
- Adding a leakage penalty (beta=0.25) reduces leakage but **does not affect task performance**
- At test time (evaluating pi^S), leakage does not meaningfully increase

### 13.2 Reward Hacking (Appendix G)

- Travel Planner with original rubric-based rewards: model learns to invoke `Planner` tool with certain arguments to end conversation early, satisfying easy constraints without doing actual planning
- Fix: decouple rewards so easy constraints are tied to their corresponding hard constraints
- This is an important practical consideration for RL on multi-turn agentic tasks

### 13.3 KL Estimation

Uses the Rae-Blackwellized estimator (Amini et al., 2025) for all losses requiring KL estimation:
- pi-Distill: sequence-level penalty aggregated into reward term, absorbed into advantage computation
- OPSD: same estimator, but KL is backpropagated directly through the estimation

---

## 14. Code Implementation Status

As of February 2026, the GitHub repository (https://github.com/Emilianopp/Privileged-Information-Distillation) contains only a README stating "We will make the code available ASAP, we are appending legal approval." No implementation code is available yet.

Based on the paper's details, the implementation would require:
1. A GRPO-compatible training loop (they use standard GRPO without base-model KL)
2. Shared-parameter teacher/student setup: same model called with different prompts (with/without PI in system prompt)
3. Two forward passes per batch: one for teacher (with PI), one for student (without PI)
4. Combined loss computation with alpha weighting
5. Importance sampling weights for the student objective
6. KL estimation using Rae-Blackwellized estimator
7. Length penalty and PI leakage penalty

**Relation to prime-rl's on-policy distillation:** prime-rl implements a *different* form of distillation where a separate larger teacher model provides KL signal on student rollouts. pi-Distill uses a **single shared-parameter model** where teacher/student differ only by conditioning on PI, which is a fundamentally different architecture. However, prime-rl's infrastructure could potentially be adapted:
- `teacher_tau` / `adv_tau` weighting maps conceptually to alpha
- The teacher KL computation could be repurposed for the pi-Distill reverse KL term
- Major difference: in pi-Distill, "teacher" is the same model with PI in the prompt, not a separate larger model

---

## 15. Relevance to ARC-AGI

### 15.1 Direct Applicability

The pi-Distill framework is highly relevant to ARC-AGI because we naturally have multiple forms of privileged information available:

| PI Type | ARC-AGI Equivalent | Expected Impact |
|---------|-------------------|-----------------|
| Tool Calls & Arguments | Ground-truth output grids for training examples | Richest signal; equivalent to giving the model the answer |
| Tool Calls Only | The transformation type/rule name without specifics | Medium signal; tells the model *what* operation to apply |
| Self-Generated Hints | Natural language description of the transformation pattern | Most flexible; doesn't require structured PI |

### 15.2 Concrete PI Sources for ARC-AGI

1. **Ground-truth outputs (test pairs):** During training, the test output is known. The teacher could be conditioned on the correct output grid, learning to "work backwards" from the answer. The student must solve the task without seeing the answer.

2. **Transformation descriptions:** Natural language descriptions of the ARC-AGI transformation rules (e.g., "rotate the input 90 degrees clockwise" or "fill enclosed regions with color X"). These could be generated by a frontier model analyzing the input-output pairs.

3. **Solution traces from stronger models:** If a frontier model (e.g., o3) can solve certain ARC tasks, its tool-call traces (even without CoT reasoning) could serve as PI for training smaller models.

4. **Partial feedback / verification signals:** After the student produces an output, the teacher could be conditioned on which cells are correct/incorrect -- a form of rich per-step feedback not available at test time.

### 15.3 Key Considerations

- **KL regime matters:** For ARC-AGI, the gap between "model with answer" and "model without answer" is likely large (high KL), which favors **joint training (alpha=0.5)** or **teacher-only (alpha=1)** over student-only
- **Model size matters for OOD:** Since ARC-AGI evaluation tasks are OOD by design, the finding that larger models benefit more from on-policy methods is relevant -- we should use the largest model feasible
- **Single training phase:** pi-Distill's efficiency (no SFT checkpoint sweep needed) is attractive for ARC-AGI where compute is limited
- **Avoid standard RL alone:** The paper consistently shows standard RL degrades OOD performance; for ARC-AGI generalization, pi-Distill or OPSD is preferable
- **Alpha annealing:** The linear annealing from alpha=0 to alpha=0.5 over early epochs could be important for stabilizing training when PI is very informative

### 15.4 Implementation Path

1. **Phase 1:** Generate PI -- Use a frontier model to solve ARC training tasks; extract tool calls / transformation descriptions
2. **Phase 2:** Format PI as prompt augmentation -- Add PI to system prompt for teacher mode
3. **Phase 3:** Implement pi-Distill on top of existing prime-rl GRPO setup -- modify training loop to support dual forward passes (with/without PI)
4. **Phase 4:** Start with alpha=0.5 (most robust), beta=0.25, evaluate on held-out ARC tasks
5. **Phase 5:** Ablate PI types and alpha values based on initial results

---

## 16. Limitations

1. **No code available yet** -- implementation must be reconstructed from paper details
2. **Limited model scale** -- all experiments on models <= 8B parameters
3. **Analysis is observational** -- factors (KL, Delta) are not systematically controlled
4. **PI must be available** -- requires access to successful frontier trajectories or other PI sources
5. **All PI derived from frontier traces** -- paper does not explore settings where neither frontier actions nor ground-truth answers are available (though Hubotter et al. explore self-reflection as PI)
6. **Multi-turn agentic focus** -- unclear how well results transfer to single-turn or non-agentic settings

---

## 17. Key Equations Reference

| Equation | Description |
|----------|-------------|
| (1) | GRPO objective with clipped importance weights |
| (2) | Teacher objective: reward maximization + reverse KL to student |
| (3) | Student objective: importance-weighted reward + KL to teacher |
| (4) | pi-Distill combined: alpha * J_Teacher + (1-alpha) * J_Student |
| (5) | OPSD objective: on-policy RL + reverse KL to PI-conditioned teacher |
| (6) | Target distribution pi* (reward-tilted posterior) |
| (7)-(8) | Variational EM derivation of J_Teacher |
| (9) | M-step: SFT objective for student |
| (11)-(14) | OPSD variational derivation |

---

## 18. Related Concurrent Work

| Paper | Relation | Key Difference |
|-------|----------|---------------|
| Zhao et al. (2026) "Self-distilled reasoner" | Proposes same OPSD objective | Targets supervised reasoning; this paper uses PI from actions |
| Shenfeld et al. (2026) "SDFT" | Proposes same OPSD objective | Focuses on continual learning |
| Hubotter et al. (2026) "RL via self-distillation" | Proposes same OPSD objective | Uses reflective self-feedback as PI |
| Zhou et al. (2025) "Variational reasoning" | Most similar to pi-Distill | Uses separate parameters; iterative EM; assumes oracle answers |
| Yang et al. (2026) "Learning beyond teacher" | On-policy distillation with reward extrapolation | Uses a separate larger teacher model |
| Chen et al. (2025) "Nudging" | Self-generated hints for RL | Injects hints to overcome zero-reward exploration barriers |
| Qu et al. (2026) "POPE" | Privileged oracle for on-policy exploration | Uses oracle solutions as structured exploration signals |
