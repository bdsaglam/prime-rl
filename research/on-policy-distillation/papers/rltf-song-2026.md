# RLTF: Expanding the Capabilities of Reinforcement Learning via Text Feedback

**Paper:** [arXiv:2602.02482](https://arxiv.org/abs/2602.02482) (v2, February 11, 2026)
**Authors:** Yuda Song\*, Lili Chen\*, Fahim Tajwar, Remi Munos, Deepak Pathak, J. Andrew Bagnell, Aarti Singh, Andrea Zanette
**Affiliations:** Carnegie Mellon University, Inria, Aurora Innovation
**Website:** https://rl-textfeedback.github.io/
**Code:** https://github.com/lili-chen/rltf

---

## 1. Core Problem: The Information Poverty of Scalar Rewards

Standard RL for LLM post-training relies on a single scalar reward (or one-bit preference label) per rollout. This creates a fundamental information bottleneck:

- **Sparse signal**: Each trajectory yields one bit of information (correct/incorrect), but tells the model nothing about *what went wrong* or *how to fix it*.
- **Exploration inefficiency**: When the base model success rate is low (epsilon_0), reliably estimating even a single gradient component requires on the order of 1/epsilon_0 rollouts (Proposition 4.1).
- **Geometric concentration**: Even at the population level, reward-weighted gradient signals concentrate on a small set of representation directions. There can exist a nontrivial low-signal subspace of directions that are *weakly identified* by reward-only updates under base-policy sampling.

At the other extreme, distillation from expert demonstrations provides dense supervision but requires costly human-generated solutions and does not scale.

**Text feedback occupies a middle ground.** It is richer than a scalar reward (it can localize errors, name violated constraints, suggest fixes) yet cheaper than complete demonstrations. Critically, text feedback is already abundant: users critique chatbot outputs, code execution produces error traces, symbolic checkers generate structured diagnostics.

---

## 2. Formalization: RL from Text Feedback (RLTF)

### 2.1 Setup

- **Prompt space** X, initial prompts X_0, distribution mu over X_0
- **Policy** pi: X -> Delta(Y) maps prompts to output distributions
- **Feedback provider** M: X x Y -> Delta(C) produces text feedback given prompt and output
- **Reward** R: X_0 x Y -> [0,1] evaluated on original prompt

### 2.2 Interaction Protocol

Multi-turn interaction with horizon H:
1. h=0: Sample prompt x_0, generate y_0 ~ pi(.|x_0), receive reward r_0 = R(x_0, y_0), get feedback c_0 ~ M(.|x_0, y_0)
2. h>0: Form augmented prompt x_h = f(x_{h-1}, y_{h-1}, c_{h-1}), generate y_h ~ pi(.|x_h), receive r_h = R(x_0, y_h), get feedback c_h

The paper focuses on the 2-turn case (H=2) for clarity and experiments.

### 2.3 The Key Asymmetry

A naive approach is standard multi-turn RL, optimizing cumulative reward J_MultiTurn(pi) = E[sum of r_h]. But this does not isolate text feedback as a learning signal -- the policy treats feedback as context and may learn to ignore it. Empirically, naive multi-turn RL improves second-turn performance but yields little gain on the first turn.

**The RLTF objective** is instead to improve single-turn performance:

```
J_SingleTurn(pi) = E_{x_0 ~ mu}[E_{y ~ pi(.|x_0)}[R(x_0, y)]]
```

The central research question: *Given access to feedback-augmented trajectories during training, how can we design learning objectives and algorithms that improve J_SingleTurn(pi)?*

---

## 3. Method 1: RLTF-SD (Self Distillation)

### 3.1 Core Idea

Text feedback often turns an incorrect first attempt into a correct second attempt. RLTF-SD exploits this by treating the feedback-conditioned second-turn policy as an implicit teacher:

1. **Generate first attempt**: y_0 ~ pi(.|x_0)
2. **Receive feedback**: c_0 ~ M(.|x_0, y_0)
3. **Form augmented prompt**: x_1 = f(x_0, y_0, c_0)
4. **Generate revised output**: y_1 ~ pi(.|x_1)
5. **Distill back**: Update pi(.|x_0) to produce outputs like y_1

This "compiles away" the need for feedback by turning test-time refinement into a training signal. The policy learns from corrected solutions rather than exploring from scratch.

### 3.2 The Distillation Objective

The RL-style distillation objective is:

```
l_distill(pi) = E_{x_1 ~ P^pi, y_1 ~ pi(.|x_1)} [ (pi(y_1|x_0) / pi_ref(y_1|x_1)) * A(x_0, y_1) ]
```

where:
- pi_ref is a reference distribution for importance-sampling correction
- A(x_0, y_1) is an advantage estimator of the reward R(x_0, y_1)

**Key insight (Eq. 4):** When pi_ref(.|x_1) = pi(.|x_1), this recovers an off-policy objective with importance-sampling correction. Setting A(y_1) = R(x_0, y_1) exactly recovers J_SingleTurn(pi) in expectation:

```
E_{y_1 ~ pi(.|x_1)} [ (pi(y_1|x_0) / pi(y_1|x_1)) * R(x_0, y_1) ] = E_{y ~ pi(.|x_0)}[R(x_0, y)] = J_SingleTurn(pi)
```

This means we can obtain an unbiased gradient for the single-turn objective using samples from the second-turn (feedback-conditioned) policy.

### 3.3 Gradient-Signal Collapse with Second-Turn Baselines

A natural baseline is the GRPO-style group-mean from second-turn rewards:

```
A_i^(1) := R(x_0, y_1^i) - (1/N) * sum_j R(x_0, y_1^j)
```

**Problem: gradient-signal collapse.** When feedback makes the second-turn policy highly reliable (p_1 -> 1), second-turn rewards become nearly constant across the group, so the centered advantages vanish and the update is approximately zero. The probability of a non-zero update scales as 1 - p_1^N ≈ N(1 - p_1), so there is no learning signal for the first turn even though the teacher is consistently correct.

### 3.4 First-Turn Baseline (Solution)

Use first-turn rewards instead:

```
b^(0) := (1/N) * sum_j R(x_0, y_0^j)
A_i^(0) := R(x_0, y_1^i) - b^(0)
```

This avoids collapse: when the first-turn policy is imperfect (b^(0) < 1) but the teacher is correct, the advantage A_i^(0) = R(x_0, y_1^i) - b^(0) != 0, providing a meaningful learning signal.

### 3.5 The Bias-Variance Tradeoff in Importance Weighting

Full importance weighting (pi(y_1|x_0) / pi(y_1|x_1)) is unbiased but high-variance because the ratio compounds across tokens, inducing heavy-tailed weights. Three options:

1. **Full importance sampling**: Unbiased but high variance; unstable in practice.
2. **CISPO-style clipping** (Eq. 8): `clip(pi(y_1|x_0)/pi_ref(y_1|x_1), 1-eps, 1+eps) * A(y_1)` -- controlled bias, reduced variance.
3. **AWR-style objective** (no importance weighting): `E_{y_1 ~ pi(.|x_1)}[A(y_1) * nabla log pi(y_1 | x_0)]` -- higher bias but low variance.

### 3.6 Final RLTF-SD Algorithm

The paper adopts:
1. **pi_ref(.|x_1) = pi(.|x_0)** -- this removes importance weighting entirely (AWR-style), since x_0 is a prefix of x_1. This eliminates the importance ratio and consistently improves stability and performance.
2. **First-turn mean baseline** b^(0) for advantage estimation.

The combined gradient update has two components:
- **Self-distillation gradient**: `(1/N) * sum_i A^{i,b} * nabla log pi(y_1^i | x_0)` where A^{i,b} = r_1^{i,b} - b^(0)
- **RL gradient** (standard multi-turn GRPO on both turns for reward optimization)

### 3.7 Connection to Rejection Sampling

Rejection Sampling (SFT on correct second-turn outputs) is a special case: binary advantage A(x_0, y_1) in {0,1}, pi_ref = sg[pi(.|x_0)]. But it underperforms methods with baselines because negative samples contribute nothing to learning.

---

## 4. Method 2: RLTF-FM (Feedback Modeling)

### 4.1 Core Idea

Instead of using feedback to generate better trajectories for distillation, RLTF-FM treats the critique itself as a supervision signal. The policy is trained to *predict* the feedback as an auxiliary objective.

### 4.2 Feedback Prediction Loss

Define a feedback-prediction distribution:

```
p_pi(c | x, y) := pi(c | f_FeeMol(x, y))
```

where f_FeeMol is a prompt template that elicits critique-style feedback given (x, y). The cross-entropy loss:

```
l_FeeMol(pi) := E_pi [ sum_{h=0}^{H-1} -log p_pi(c_h | x_h, y_h) ]
```

Note: y_h tokens are treated as constants (no gradient through response tokens) -- this is pure supervised learning on feedback tokens.

### 4.3 Joint Objective with RL

Feedback modeling is combined with standard multi-turn RL:

```
max_pi J_MultiTurn(pi) - lambda_FeeMol * l_FeeMol(pi)
```

where lambda_FeeMol >= 0 controls the strength of the auxiliary feedback loss.

### 4.4 Theoretical Analysis: Why Predicting Feedback Helps

The theoretical contribution is analyzed through the lens of representation learning in a frozen-rollout regime (batch RL with log-linear/softmax policy):

**Proposition 4.1 (Reward-only bottlenecks):**
Under sparse rewards with base success rate epsilon_0:
- (i) **Rare-event estimation**: The per-sample policy-gradient estimator has low SNR; for any direction, SNR scales at most as sqrt(epsilon_0). Reliably estimating a single gradient component requires O(1/epsilon_0) rollouts.
- (ii) **Weak identifiability**: The reward-weighted gradient signal concentrates on a small set of representation directions. There can exist a nontrivial low-signal subspace S_low that is weakly identified by reward-only updates.

**Proposition 4.2 (Feedback modeling yields well-conditioned signal):**
Under the same batch regime, RLTF-FM provides an additional supervised learning signal on the shared representation. Under mild coverage conditions on the feedback, RLTF-FM is *informative in representation directions that are weakly identified by sparse reward under base rollouts*. As a result, RLTF-FM can learn representation degrees of freedom that reward-only RL fails to identify early on.

**Intuition**: Reward-only RL acts like an effectively low-rank update. Feedback modeling acts as a "representation preconditioner" that fills in the missing representation directions, improving identifiability and conditioning. The feedback signal is dense (token-level gradients on every rollout, not just successful ones) and structured (natural language describes the error).

---

## 5. Test-Time Scaling via Self-Feedback

### 5.1 Mechanism

Because RLTF-FM trains the policy to predict critiques, the same model can be run in a "feedback mode" at inference time for iterative self-refinement:

1. Generate initial output: y_0 ~ pi_theta(.|x_0)
2. Generate self-critique: c_hat_0 ~ p_theta(.|x_0, y_0)
3. Form augmented prompt: x_1 = f(x_0, y_0, c_hat_0)
4. Generate refined output: y_1 ~ pi_theta(.|x_1)
5. Repeat for up to H rounds

This enables test-time scaling *without requiring a separate learned judge model*. The auxiliary training simply makes the policy's self-critique distribution more faithful to the external feedback channel.

### 5.2 Key Observation

RLTF-FM uniquely enables this because it explicitly trains the model to produce accurate critiques. Standard RL or RLTF-SD do not have this capability -- they never train the model to generate feedback.

---

## 6. Experimental Results

### 6.1 Setup

- **Feedback provider (judge)**: Qwen3-235B-A22B-Instruct-2507
- **Learner (student)**: Llama-3.1-8B-Instruct
- **RL algorithm**: GRPO with early termination
- **Horizon**: 2 turns

### 6.2 Benchmarks

Three domains:
- **Reasoning puzzles** (Reasoning Gym): Knights and Knaves, Binary Matrix, Shortest Path
- **Competition math**: MATH500, AIME24 (trained on DAPO and DeepMath)
- **Creative writing**: LitBench, WritingBench

### 6.3 Baselines

- Base Model (no training)
- GRPO Single Turn (J_SingleTurn only)
- GRPO Multi Turn (J_MultiTurn)
- Feedback Descent (Lee et al., 2025) -- text-space optimization via pairwise comparison

### 6.4 Main Results (Table 1)

| Task | Base | GRPO-ST | GRPO-MT | FeedbackDescent | **RLTF-SD** | **RLTF-FM** |
|------|------|---------|---------|-----------------|-------------|-------------|
| **Reasoning** | | | | | | |
| Knights & Knaves | 0.058 | 0.373 | 0.352 | 0.055 | **0.802** | **0.880** |
| Binary Matrix | 0.001 | 0.125 | 0.950 | 0.005 | **0.976** | **0.978** |
| Shortest Path | 0.034 | 0.385 | 0.384 | 0.035 | **0.830** | **0.905** |
| **Math (DAPO)** | | | | | | |
| MATH500 | 0.376 | 0.526 | 0.523 | 0.415 | **0.548** | **0.567** |
| AIME24 | 0.025 | 0.058 | 0.025 | 0.045 | **0.088** | **0.083** |
| **Math (DeepMath)** | | | | | | |
| MATH500 | 0.376 | 0.558 | 0.578 | 0.424 | **0.598** | **0.636** |
| AIME24 | 0.025 | 0.042 | 0.050 | 0.054 | **0.058** | **0.058** |
| **Creative Writing** | | | | | | |
| LitBench | 4.20 | 6.83 | 6.41 | 8.25 | **8.80** | 8.40 |
| WritingBench | 5.71 | 5.92 | 6.29 | 5.30 | 6.71 | 6.39 |

**Key observations:**
- Both RLTF-SD and RLTF-FM consistently outperform all baselines across all tasks.
- On reasoning puzzles, improvements are dramatic: Knights and Knaves goes from 0.373 (GRPO) to 0.880 (RLTF-FM), a +136% relative improvement.
- GRPO Multi-Turn performs similarly to GRPO Single-Turn in single-turn evaluation, confirming that naively incorporating feedback as context is insufficient to internalize it.
- Feedback Descent underperforms, indicating parameter-space optimization is more effective than text-space optimization.
- **RLTF-SD excels on creative writing** (where teacher-student distribution mismatch is small).
- **RLTF-FM excels on math and reasoning** (where feedback is more objective and the auxiliary prediction loss is easier to optimize).

### 6.5 Ablation: Design Choices for Self Distillation (RQ2)

- **First-turn baseline vs. GRPO-style second-turn baseline**: First-turn baseline is consistently better, confirming the gradient-signal collapse theory.
- **AWR (no importance weighting) vs. PPO clipping vs. CISPO clipping**: AWR without importance weighting is the most stable and performs best. Introducing importance weighting introduces training instability, even with clipping.
- **Rejection Sampling underperforms** both RLTF-SD and RLTF-SD with GRPO baseline, confirming the benefit of variance reduction via baselines.

### 6.6 Ablation: Rich Feedback vs. Correctness-Only (RQ3)

Replacing rich text feedback with just "Your previous answer was {correct/incorrect}" significantly hurts performance for RLTF-SD, confirming that semantically rich feedback is critical. The correctness-only baseline performs much worse, indicating the model genuinely learns from the content of the critique, not just the binary signal.

### 6.7 Test-Time Scaling (RQ4)

Evaluated on Knights and Knaves and MATH500 with up to 5 rounds of self-feedback:
- RLTF-FM with self-critique achieves significant improvement over 0-round baseline.
- GRPO with and without self-critique RL achieve similar performance, confirming that standard RL alone is not sufficient for learning useful self-critique.
- Adding RLTF-FM loss on top of self-critique RL training brings significant further improvement.
- Improvement saturates after a few rounds (diminishing returns), consistent with the self-improvement literature.

---

## 7. Algorithms Summary

### Algorithm 1: Self Distillation (RLTF-SD)

```
For each training step:
  Sample minibatch of prompts {x_0^b}
  For each prompt, for i = 1..N:
    Sample first-turn output y_0 ~ pi(.|x_0)
    Obtain feedback c_0 ~ M(x_0, y_0)
    Form x_1 = f(x_0, y_0, c_0)
    Sample second-turn output y_1 ~ pi(.|x_1)
    Get rewards r_0 = R(x_0, y_0) and r_1 = R(x_0, y_1)

  Compute baselines:
    b^(0) = mean(r_0^i)  [first-turn baseline]
    b^(R) = mean(R^i)    [return baseline]
    b^(1) = mean(r_1^i)  [second-turn baseline]

  Self-distillation advantages: A^i = r_1^i - b^(0)
  RL advantages: A_RL,0^i = R^i - b^(R), A_RL,1^i = r_1^i - b^(1)

  Self-distillation gradient: g = (1/N) sum_i A^i * nabla log pi(y_1^i | x_0)
  RL gradient: g_RL = (1/N) sum_i [A_RL,0^i * nabla log pi(y_0^i | x_0) + A_RL,1^i * nabla log pi(y_1^i | x_1)]

  Update: theta <- OPT(theta, eta, g + g_RL)
```

### Algorithm 2: Feedback Modeling with Test-time Self-Feedback (RLTF-FM Inference)

```
Given initial prompt x_0, number of self-critique steps H:
  For h = 1..H:
    Sample output y_h ~ pi_theta(.|x_{h-1})
    Generate self-critique c_hat_h ~ p_theta(.|x_{h-1}, y_h)
    Form x_h = f(x_{h-1}, y_h, c_hat_h)
  Return final output y_H
```

### Algorithm 3: Feedback Modeling (RLTF-FM Training)

Same as RLTF-SD but replaces self-distillation gradient with feedback modeling gradient:
```
  g = (1/N) sum_i nabla log pi(c_0^i | f_FeeMol(x_0, y_0^i))
```
Combined with the standard RL gradient.

---

## 8. Related Work and Context

- **Learning from text feedback in robotics/NLP**: Prior work used text corrections for robot control; RLTF is the first to formalize leveraging text feedback *as a training signal* specifically for RL in the LLM setting.
- **LLM distillation**: Connects to self-distillation (Askell et al., 2021; Snell et al., 2022) where teacher and student are the same model but the teacher has access to privileged information (here, the feedback).
- **Feedback Descent** (Lee et al., 2025): Optimizes directly in text space via pairwise comparison -- underperforms RLTF's parameter-space methods.
- **LLM world models**: Feedback modeling is related to learning environment dynamics (predicting what the environment/judge will say), connecting to the Dyna architecture and Code World Model.
- **Concurrent work**: Hubotter et al. (2026) study self-distillation from interpreter feedback; Zhao et al. (2026b), Shenfeld et al. (2026) explore related self-distillation ideas but with demonstrations rather than feedback.

---

## 9. Limitations

1. **Feedback quality**: Real-world feedback may be noisy or subjective; the paper uses a strong LLM judge (Qwen3-235B) which may not always be available.
2. **Horizon**: Results are for 2-turn interaction; truly long-horizon feedback may require summarization or other techniques to address distribution shift and context limits.
3. **Theoretical scope**: Theory focuses on representation learning near the base policy's distribution; a full end-to-end analysis would strengthen understanding.
4. **Process supervision**: Exploring interplay with process reward models (Lightman et al., 2023) is an open direction.

---

## 10. Relevance to ARC-AGI and Our REPL Environment

### 10.1 Why RLTF is Directly Applicable

Our ARC-AGI REPL environment is a near-perfect instantiation of the RLTF setup:

1. **The environment naturally provides rich text feedback.** In the iterative environment, when the model's `transform` function fails, it receives structured feedback: pass/fail per training example, diff visualizations showing expected vs. actual output. In the REPL environment, code execution produces Python tracebacks, assertion errors, and printed output. This is exactly the kind of "text feedback from tool-mediated workflows" that the paper highlights as a sweet spot.

2. **Feedback is available during training but not at test time** (in the standard evaluation protocol). This matches the RLTF asymmetry exactly: during RL training, the model gets multiple turns with feedback; at test time, we want good first-attempt performance.

3. **The feedback is semantically rich.** An error like `IndexError: list index out of range at line 12` or a diff showing `Expected [[1,0],[0,1]] but got [[1,1],[0,0]]` carries far more information than a binary reward of 0.

4. **Binary reward is extremely sparse for ARC-AGI.** Base model success rates on ARC tasks are often very low (our scratchpad shows extensive hyperparameter tuning challenges). This is precisely the regime where Proposition 4.1 predicts reward-only RL will be sample-inefficient and where RLTF's richer signal provides the biggest advantage.

### 10.2 Specific Application Strategies

**RLTF-SD for ARC-AGI:**
- Turn 1: Model generates a `transform` function
- Environment executes it, provides structured error feedback (pass/fail per example, diff visualization)
- Turn 2: Model revises its function given the feedback
- Distill: Train the single-turn policy to match the quality of second-turn outputs

This directly matches our iterative environment's existing multi-turn structure.

**RLTF-FM for ARC-AGI:**
- In addition to RL, train the model to predict what the execution feedback will be for a given (task, code) pair
- This forces the model to learn an internal model of code execution and grid transformations
- At test time, the model can self-critique: generate code, predict what errors it would produce, then revise

This is particularly appealing because code execution feedback in ARC-AGI is deterministic and structured -- exactly the conditions where feedback modeling should work best (objective, low-noise feedback).

### 10.3 Implementation Considerations for prime-rl

Our current stack uses prime-rl for RL training. Implementing RLTF would require:

1. **For RLTF-SD**: Modify the training loop to:
   - Collect two-turn rollouts with feedback
   - Compute the AWR-style distillation gradient on y_1 with respect to pi(.|x_0)
   - Use first-turn baseline for advantage estimation
   - This is complementary to the existing on-policy distillation support (which distills from a separate teacher model)

2. **For RLTF-FM**: Add an auxiliary SFT loss on feedback prediction:
   - Use the same rollout data but additionally train the model to predict execution output
   - Relatively straightforward since it is a standard cross-entropy loss on the feedback tokens
   - lambda_FeeMol hyperparameter controls the balance with RL

3. **Test-time self-feedback**: RLTF-FM enables a zero-cost test-time scaling strategy:
   - The model generates code, self-critiques, revises -- all without actual code execution
   - This could complement our existing multi-turn evaluation with actual execution

### 10.4 Unique Advantage for Code/ARC Settings

The paper notes (footnote 1) that code execution feedback is a special case where text feedback *is* available at test time (through tool use). This means for ARC-AGI, we can potentially get the benefits of both approaches:
- **During training**: Use RLTF-SD and/or RLTF-FM with actual execution feedback
- **During test-time**: Use actual code execution for feedback (multi-turn), but *also* use RLTF-FM's self-critique for additional reasoning before committing to execution

This dual-use makes RLTF particularly powerful for our setting compared to domains where feedback is only available during training.

### 10.5 Expected Impact

Given the paper's results (e.g., Knights and Knaves: 0.058 base -> 0.373 GRPO -> 0.880 RLTF-FM), and that ARC-AGI similarly has low base success rates with rich structured feedback, we should expect substantial improvements from incorporating text feedback into our RL pipeline. The reasoning puzzle results are most relevant to ARC-AGI since they involve structured logical reasoning with verifiable correctness.

---

## 11. Key Takeaways

1. **Text feedback is a powerful intermediate signal** between sparse rewards and full demonstrations, and it can be leveraged systematically through two complementary methods.

2. **Self-distillation (RLTF-SD)** turns feedback-conditioned refinement into a training signal by treating the model's own second-turn as a teacher. The AWR-style objective with first-turn baseline is the best design choice.

3. **Feedback modeling (RLTF-FM)** trains the model to predict feedback as an auxiliary task, providing dense token-level gradients and enabling test-time self-critique. Theoretically, it acts as a "representation preconditioner" that identifies representation directions invisible to reward-only RL.

4. **The two methods have complementary strengths**: RLTF-SD excels where teacher-student mismatch is small (creative writing); RLTF-FM excels where feedback is objective and predictable (math, reasoning).

5. **For ARC-AGI, RLTF is a natural fit**: our REPL environment already produces exactly the kind of structured text feedback the framework requires, and the sparse-reward regime is where the theory predicts the largest gains.
