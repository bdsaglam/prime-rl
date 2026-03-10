# On-Policy Distillation: Concepts and Ideas

A guide to on-policy distillation (OPD) for LLM post-training, synthesized from recent papers (2023-2026). This document covers the core ideas, variants, and their relationships.

## The Problem: Sparse Rewards and Off-Policy Mismatch

Two fundamental problems plague current LLM post-training:

**1. RL provides sparse feedback.** Standard RLVR (e.g., GRPO) assigns a single scalar reward per rollout. A model generating 16K tokens of reasoning gets exactly one bit of information: right or wrong. The model doesn't learn *where* it went wrong or *which* tokens were pivotal. Information-theoretically, RL teaches O(1) bits per episode regardless of token count. This makes RL sample-inefficient and particularly brutal when the base success rate is low -- you need ~1/epsilon_0 rollouts to estimate even one gradient direction (RLTF, Proposition 4.1).

**2. SFT/Off-policy distillation suffers from distribution mismatch.** Training on teacher-generated demonstrations is dense (O(N) bits per episode) but off-policy: the student learns in contexts the *teacher* visits, not ones the *student* will encounter at inference time. When the student makes an early mistake the teacher never makes, it diverges into unfamiliar territory and compounds errors -- the classic exposure bias / **DAgger** problem (DAgger = *D*ataset *Agg*regation; Ross, Gordon & Bagnell, AISTATS 2011).

The following table summarizes the landscape:

| Method | Sampling | Reward Signal | Teacher Required? |
|--------|----------|---------------|-------------------|
| SFT / Off-policy distillation | off-policy | dense | Yes (demonstrations) |
| RL (GRPO) | **on-policy** | sparse | No |
| **On-policy distillation** | **on-policy** | **dense** | Yes (larger model) |
| **Self-distillation** | **on-policy** | **dense** | **No** (same model) |

OPD combines the best of both worlds: on-policy sampling (learning from your own mistakes) with dense per-token supervision (knowing exactly where you went wrong).

---

## Core Mechanism: On-Policy Distillation

The foundational algorithm was formalized by Agarwal et al. (2024) as **Generalized Knowledge Distillation (GKD)** (ICLR 2024, arXiv:2306.13649). The idea is simple:

1. **Student generates** a rollout y given prompt x (on-policy sampling)
2. **Teacher scores** the student's rollout via a single forward pass (prefill), producing token-level probability distributions
3. **Compute divergence** between teacher and student distributions at each token position
4. **Update student** to reduce this divergence

The teacher never generates text -- it only evaluates the student's tokens. This is computationally cheap (one forward pass) and gives dense feedback (a learning signal at every token).

### The Loss Function

The per-token reverse KL divergence:

```
KL(pi_student || pi_teacher) = E_{x ~ pi_student} [ log pi_student(x_t | x_{<t}) - log pi_teacher(x_t | x_{<t}) ]
```

When the student generates a token the teacher considers unlikely, the KL is high at that position -- the student receives a strong push to change. When student and teacher agree, the KL is zero -- no wasted gradient signal.

### Why Reverse KL?

- **Mode-seeking**: The student focuses on the teacher's high-probability behaviors rather than spreading mass across the entire teacher distribution. This is important when the student has less capacity than the teacher.
- **RL synergy**: RL objectives naturally optimize reverse KL under the reward distribution. This makes OPD + RL combination seamless.
- **"Unhackable"**: Unlike scalar rewards, low KL always corresponds to good behavior from the teacher's perspective. No reward hacking.

GKD also supports forward KL (mode-covering, better for greedy sampling) and JSD (bounded, balanced). The choice is task-dependent -- see the detailed notes in `research-notes/gkd-agarwal-2023.md`.

### Key Properties

- **5% data beats 100% supervised**: GKD trained on 5% of data (on-policy) outperforms standard KD trained on 100% (off-policy). Fresh on-policy data targets current weaknesses.
- **50-100x compute savings over RL**: The Thinking Machines blog demonstrated that OPD reaches RL performance with 50-100x fewer FLOPs, because dense supervision provides O(N) bits per episode vs. O(1) for RL.
- **Seamless RL integration**: The combined objective is simply `adv_tau * RL_advantage + teacher_tau * teacher_KL`. Both signals use on-policy data and are additive in the loss.

---

## The Self-Distillation Revolution

A wave of concurrent papers (Jan-Feb 2026) independently discovered that **you don't need a separate, larger teacher model**. The same model can serve as both teacher and student by exploiting **information asymmetry**: the teacher sees privileged information the student doesn't.

This is a paradigm shift. Standard OPD requires hosting a larger teacher model (e.g., Qwen3-32B teaching Qwen3-8B), which is expensive. Self-distillation eliminates this cost entirely.

### How Information Asymmetry Works

The student sees only the problem:
```
Student prompt: "Solve: What is the derivative of f(x) = 3x^2 + 2x - 5 at x = 2?"
```

The teacher sees the problem plus privileged information (e.g., the correct answer):
```
Teacher prompt: "Solve: What is the derivative of f(x) = 3x^2 + 2x - 5 at x = 2?
Here is a reference solution: f'(x) = 6x + 2, so f'(2) = 14.
After understanding the reference solution, please solve this problem using your own approach below:"
```

Both prompts are fed to the **same model**. The teacher's next-token distributions are more informed because it has seen the answer. This gap between "knows the answer" and "doesn't know the answer" provides the dense training signal.

The central hypothesis: **rationalization is easier than generation**. When given the correct answer, even a model that couldn't solve the problem from scratch can produce better-informed token distributions. This has been empirically validated across multiple papers.

### Four Concurrent Approaches to Self-Distillation

| Paper | Approach | Privileged Info | Key Innovation |
|-------|----------|-----------------|----------------|
| **OPSD** (Zhao et al., UCLA/Meta) | Same model, teacher frozen to initial weights | Ground-truth solution | Simplest; 4-8x token efficiency vs GRPO |
| **SDFT** (Shenfeld et al., MIT) | Same model, EMA teacher | Expert demonstrations in-context | Continual learning without forgetting |
| **SDPO** (Hubotter et al., ETH Zurich) | Same model, EMA teacher | Successful peer rollouts + environment feedback | Works without ground-truth; uses env feedback |
| **pi-Distill** (Penaloza et al., ServiceNow/Mila) | Same model, joint teacher-student optimization | Action traces from frontier model | Joint training (alpha=0.5) most robust |

Let's examine each in detail.

---

## OPSD: On-Policy Self-Distillation (Zhao et al.)

**Paper**: "Self-Distilled Reasoner" (UCLA/Meta, 2026)
**Detailed notes**: `research-notes/opsd-zhao-2026.md`

### Method

1. Student generates on-policy rollout (no privileged info)
2. Teacher = same model but conditioned on ground-truth solution y*
3. Both evaluate the student's rollout at every token position
4. Minimize JSD between teacher and student distributions

The teacher is **frozen to the initial checkpoint** for stability. This means the teacher doesn't improve during training -- it provides a fixed, informed signal.

### Key Results

- **4-8x more token-efficient than GRPO**: OPSD uses 1 rollout of 2K tokens vs GRPO's 8 rollouts of 16K tokens
- **Single rollout sufficient**: Only 1 generation per problem (vs GRPO's group of 8)
- **Scale requirement**: Works at 4B+, marginal at 1.7B. The model must be capable enough to rationalize solutions when given the answer.

### When to Use

Best when you have **ground-truth answers** and want maximum simplicity. The frozen teacher is easy to implement but doesn't adapt as training progresses.

---

## SDFT: Self-Distillation Fine-Tuning (Shenfeld et al.)

**Paper**: "Self-Distillation Enables Continual Learning" (MIT, arXiv:2601.19897)
**Detailed notes**: `research-notes/sdft-shenfeld-2026.md`

### Method

Same core idea as OPSD but with two key differences:
1. **EMA teacher**: The teacher is an exponential moving average of the student, not frozen. This lets the teacher track the student's improvements: `phi <- alpha * theta + (1 - alpha) * phi`
2. **ICL-based privileged info**: Instead of just providing the answer, SDFT puts expert demonstrations in-context, leveraging the model's in-context learning ability.

### Key Results

- **Continual learning**: Train sequentially on Tool Use -> Science -> Medical without catastrophic forgetting. SFT shows severe oscillation; SDFT accumulates skills stably.
- **Better OOD generalization**: 98% OOD accuracy on knowledge acquisition vs 80% for SFT.
- **Preserves reasoning**: When fine-tuning a reasoning model (Olmo-3-7B-Think), SFT *degrades* performance to 23.5%; SDFT *improves* it to 43.7%.

### Scale Requirements

- **3B: too small** (ICL too weak for useful teacher signal)
- **7B: sweet spot** (+4 points over SFT)
- **14B: best** (+6.9 points over SFT)

### When to Use

Best for **continual learning** scenarios where you need to add new skills without forgetting old ones. Also good when you have demonstrations (not just answers) and want an evolving teacher.

---

## SDPO: RL via Self-Distillation (Hubotter et al.)

**Paper**: "Reinforcement Learning via Self-Distillation" (ETH Zurich, arXiv:2601.20802)
**Detailed notes**: `research-notes/sdpo-hubotter-2026.md`

### Method

SDPO's key innovation: it doesn't require ground-truth answers at all. The privileged information comes from **the environment itself**:

1. Student generates rollouts for a problem
2. Some rollouts succeed, others fail
3. The self-teacher is the same model conditioned on:
   - A successful peer solution (from the same batch)
   - Environment feedback (error messages, test results)
4. Re-evaluate the failed rollout under the teacher's informed context
5. The log-ratio `log p_teacher(token) - log p_student(token)` at each position gives per-token advantages

The teacher prompt looks like:
```
User: {problem}
Correct solution: {successful_peer_rollout}
Feedback from your unsuccessful earlier attempt: {environment_output}
Correctly solve the original question.
Assistant: {student's_original_response}  # re-evaluated, not generated
```

### Key Insight: Logit-Level Credit Assignment

SDPO provides the finest possible credit assignment -- not just per-token, but per-vocabulary-item at each position. This gives |V| * |y| unique advantage values per sequence, compared to GRPO's single scalar.

Three granularity levels, in order of effectiveness:
- **Logit-level** (full vocabulary at each position) -- best
- **Token-level** (only the sampled token at each position)
- **Sequence-level** (average all tokens) -- still beats GRPO

### Key Results

- **6x speedup on Chemistry**: SDPO at 1 hour matches GRPO at 5 hours
- **48.8% vs 41.2% on LiveCodeBench v6** (outperforms Claude Sonnet 4)
- **3-11x shorter generations** while achieving higher accuracy (no filler reasoning)
- **Test-time self-distillation**: Apply SDPO on single hard problems at inference time, solving problems that best-of-k and multi-turn cannot

### When to Use

Best when you **don't have ground-truth answers** but your environment provides **rich feedback** (error messages, test outputs, partial evaluations). Also uniquely valuable for test-time specialization on individual hard problems.

---

## pi-Distill (Penaloza et al.)

**Paper**: "Privileged Information Distillation" (ServiceNow/Mila, arXiv:2602.04942)
**Detailed notes**: `research-notes/pi-distill-penaloza-2026.md`

### Method

pi-Distill's distinctive feature: **joint optimization of both teacher and student**. Instead of freezing the teacher or using EMA, both are directly trained with a convex combination:

```
J = alpha * J_Teacher + (1 - alpha) * J_Student
```

- **alpha = 1 (teacher only)**: Teacher is trained with RL; student improves through shared parameters
- **alpha = 0 (student only)**: Student learns from teacher's (off-policy) traces
- **alpha = 0.5 (joint)**: Both optimized simultaneously -- most robust

### Three Sources of Privileged Information

1. **Tool Calls & Arguments** (richest): Full action sequence with parameters
2. **Tool Calls Only** (medium): Function names without arguments
3. **Self-Generated Hints** (least dense): Natural language summaries of successful strategies

### Key Results

- **Best in 7/16 scenarios, worst in only 1** (for alpha=0.5)
- **+11.8% on Travel Planner** over SFT w/ CoT + RL (the industry standard)
- **Can distill closed-source models** even when reasoning traces are hidden -- only action trajectories needed
- **Avoids RL's OOD degradation**: Standard RL consistently degrades out-of-distribution performance; pi-Distill avoids this

### KL Regime Analysis

| Setting | Works when... | Fails when... |
|---------|--------------|---------------|
| alpha=1 (teacher) | High initial utility; KL not too low | KL too low (collapse) |
| alpha=0 (student) | Low KL; positive PI utility | High KL; negative PI utility |
| alpha=0.5 (joint) | Almost always | Rarely worst |

### When to Use

Best when you have **multiple types of privileged information** and want robust performance without extensive hyperparameter search. Alpha=0.5 with annealing is a safe default.

---

## RLTF: RL from Text Feedback (Song et al.)

**Paper**: "Expanding the Capabilities of Reinforcement Learning via Text Feedback" (CMU, arXiv:2602.02482)
**Detailed notes**: `research-notes/rltf-song-2026.md`

### Two Complementary Methods

**RLTF-SD (Self-Distillation):**
- Model generates attempt -> receives text feedback -> generates better revision
- The feedback-conditioned revision is the implicit teacher
- Distill the second-turn quality into first-turn performance
- Key innovation: use *first-turn* baseline (not second-turn) to avoid gradient-signal collapse

**RLTF-FM (Feedback Modeling):**
- Train the model to *predict* what feedback it would receive (auxiliary cross-entropy loss)
- Acts as a "representation preconditioner" that identifies gradient directions invisible to reward-only RL
- Enables test-time self-critique without an external judge

### Key Results

- Knights & Knaves: 0.058 (base) -> 0.373 (GRPO) -> **0.880 (RLTF-FM)** -- a 2.4x improvement over GRPO
- Both methods consistently outperform all baselines across reasoning, math, and creative writing
- RLTF-FM uniquely enables test-time scaling via self-critique

### When to Use

Best when your environment naturally provides **structured text feedback** (error messages, test outputs, critiques). Uniquely powerful when the base success rate is very low -- this is exactly the regime where RLTF's theory predicts the largest gains.

---

## Meta-Learning from Language Feedback (Klissarov et al.)

**Paper**: "RL^2F" and "Social Meta-Learning" (Google DeepMind, arXiv:2602.16066, 2602.16488)
**Detailed notes**: `research-notes/meta-learning-klissarov-2026.md`

### Core Idea

Frame multi-turn interaction as RL^2 meta-learning:
- **Inner loop (in-context)**: Student learns to integrate feedback within a conversation
- **Outer loop (gradients)**: Optimize the student's weights to be better at in-context learning

Teacher and student are the same model; the teacher has privileged information (ground-truth) and provides corrective natural language feedback across multiple turns.

### Key Results

- **Gemini 2.5 Flash matches Pro** in multi-turn math after RL^2F training
- **Remarkable OOD transfer**: Training on math improves performance on Poker, Wordle, Maze Navigation, ARC-AGI
- **In-context plasticity**: Baseline models literally give up reasoning when given feedback; RL^2F-trained models actively integrate it

### Unique Contribution

The concept of **in-context plasticity** -- the learned ability to change predictions based on in-context information. This is a *skill*, not just an emergent property of scale. Baseline models (even frontier ones) are surprisingly bad at integrating feedback.

### When to Use

Best when you want to improve **multi-turn interactive reasoning** and have access to verifiable environments. The cross-domain transfer result suggests training this skill on cheap math/code data and transferring to harder domains.

---

## Relationships Between Methods

```
                          ┌─────────────────────────────┐
                          │     GKD (Agarwal 2024)       │
                          │  Foundational OPD framework   │
                          │  Separate teacher + student   │
                          └───────────┬─────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                   ▼
          ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
          │   Thinking   │   │  Self-Distill │   │   Feedback   │
          │  Machines /  │   │   Methods     │   │   Methods    │
          │   Qwen3      │   │              │   │              │
          │ (Larger      │   │              │   │              │
          │  teacher)    │   │              │   │              │
          └──────────────┘   └──────┬───────┘   └──────┬───────┘
                                    │                   │
                    ┌───────────────┼───────┐           │
                    ▼               ▼       ▼           ▼
              ┌──────────┐  ┌──────────┐ ┌───────┐ ┌───────┐
              │  OPSD    │  │  SDFT    │ │  pi-  │ │ RLTF  │
              │ (frozen  │  │ (EMA     │ │Distill│ │ (text │
              │ teacher, │  │ teacher, │ │(joint │ │ feed- │
              │ ground-  │  │ demos)   │ │train) │ │ back) │
              │ truth)   │  │          │ │       │ │       │
              └──────────┘  └──────────┘ └───────┘ └───────┘
                                                       │
                                                  ┌────┴────┐
                                                  ▼         ▼
                                              ┌───────┐ ┌───────┐
                                              │ SDPO  │ │ RL^2F │
                                              │(self- │ │(meta- │
                                              │teacher│ │learn) │
                                              │+env   │ │       │
                                              │feed.) │ │       │
                                              └───────┘ └───────┘
```

### Shared Principles

All methods share these core insights:

1. **On-policy sampling is critical**: Training on the student's own generations eliminates distribution mismatch
2. **Dense supervision beats sparse rewards**: Token-level signals provide O(N) bits vs O(1)
3. **Information asymmetry creates teachers**: A model conditioned on privileged info becomes a useful teacher for its unconditioned self
4. **Gradients don't flow through sampling**: The sampling process is non-differentiable; only the loss computation contributes gradients
5. **Model scale matters**: Self-distillation requires sufficient ICL capacity (typically 4B+ parameters)

### Key Differences

| Dimension | GKD/OPD | OPSD | SDFT | SDPO | pi-Distill | RLTF |
|-----------|---------|------|------|------|------------|------|
| Teacher model | Separate, larger | Same (frozen) | Same (EMA) | Same (EMA) | Same (joint) | Same (feedback-conditioned) |
| Privileged info | Teacher's capability | Ground-truth answer | Demonstrations | Peer solutions + env feedback | Action traces | Text feedback |
| Needs ground truth? | No | Yes | Yes (demos) | No | No (uses traces) | No |
| Teacher evolves? | No | No | Yes (EMA) | Yes (EMA) | Yes (joint training) | N/A |
| Works without rewards? | Yes | Yes | Yes | No (needs some successes) | No (needs rewards) | No |
| Multi-turn? | No | No | No | No | Yes | Yes (2-turn) |

---

## Practical Considerations

### Which Method to Choose?

Decision tree:

1. **Do you have a stronger teacher model available?**
   - Yes -> Standard OPD (GKD-style) is simplest. Use prime-rl's `teacher_tau`.
   - No -> Self-distillation variant.

2. **Do you have ground-truth answers?**
   - Yes -> OPSD or SDFT
   - No, but have successful traces -> pi-Distill or SDPO
   - No, but have env feedback -> SDPO or RLTF

3. **Is continual learning important?**
   - Yes -> SDFT (designed for it)
   - No -> Any method

4. **Is your base success rate very low?**
   - Yes -> RLTF (theory predicts largest gains) or standard OPD with stronger teacher
   - No -> OPSD or SDPO

5. **Do you need test-time adaptation?**
   - Yes -> SDPO (test-time self-distillation) or RL^2F (meta-learning)
   - No -> Any method

### Compute Requirements

| Method | Extra Forward Passes | GPU Overhead | Memory |
|--------|---------------------|-------------|--------|
| Standard OPD | 1 (teacher prefill) | Need separate teacher GPUs | High (two models) |
| Self-distillation | 1-2 (teacher prefill with different prompt) | Minimal (same model) | Moderate |
| SDPO | 1 (teacher prefill with reprompt) | +6-17% time | Low (top-K logits) |
| RLTF-FM | 1 (feedback prediction) | Minimal (auxiliary loss) | Low |

### Common Pitfalls

1. **Model too small**: Self-distillation needs 4B+ parameters. Below 3B, ICL is too weak.
2. **Teacher-student KL too low**: If privileged info barely changes the model's behavior, the training signal is too weak. This is the "policy collapse" failure mode.
3. **Teacher-student KL too high**: If the teacher's traces are too different from what the student would produce, off-policy learning from teacher traces fails.
4. **Forgetting to freeze/EMA**: Training both teacher and student without regularization leads to instability.
5. **Including student's response in teacher prompt**: SDPO found this reduces exploration. The teacher should not see what the student tried.

---

## References

| Paper | Key Contribution | Notes File |
|-------|-----------------|------------|
| GKD (Agarwal et al., 2024) | Foundational OPD framework | `research-notes/gkd-agarwal-2023.md` |
| Thinking Machines blog (Lu, 2025) | Practical OPD implementation, 50-100x savings | Parent folder: `On-Policy Distillation.md` |
| OPSD (Zhao et al., 2026) | Self-distillation with frozen teacher | `research-notes/opsd-zhao-2026.md` |
| SDFT (Shenfeld et al., 2026) | Continual learning via self-distillation | `research-notes/sdft-shenfeld-2026.md` |
| SDPO (Hubotter et al., 2026) | RL + self-distillation from env feedback | `research-notes/sdpo-hubotter-2026.md` |
| pi-Distill (Penaloza et al., 2026) | Joint teacher-student training with PI | `research-notes/pi-distill-penaloza-2026.md` |
| RLTF (Song et al., 2026) | Self-distillation from text feedback | `research-notes/rltf-song-2026.md` |
| RL^2F (Klissarov et al., 2026) | Meta-learning for in-context plasticity | `research-notes/meta-learning-klissarov-2026.md` |
| Qwen3 Technical Report (2025) | OPD at 10x lower cost than RL | Cited in Thinking Machines blog |

All PDFs are in `papers/` directory.
