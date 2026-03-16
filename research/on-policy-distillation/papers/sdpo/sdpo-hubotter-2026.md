# SDPO: Reinforcement Learning via Self-Distillation

**Paper:** "Reinforcement Learning via Self-Distillation" (arXiv:2601.20802v2, Feb 2026)
**Authors:** Jonas Hubotter, Frederike Lubeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, Andreas Krause
**Affiliations:** ETH Zurich, Max Planck Institute, MIT, Stanford
**Code:** https://github.com/lasgroup/SDPO (Apache-2.0, built on verl)
**Project page:** https://self-distillation.github.io/SDPO

---

## 1. Core Idea

SDPO introduces a simple but powerful modification to standard RLVR (Reinforcement Learning with Verifiable Rewards) pipelines: instead of learning only from a scalar outcome reward per rollout (as in GRPO), the model uses itself as a "self-teacher" by conditioning on rich environment feedback and/or successful peer solutions to re-evaluate its own failed attempts. This creates dense, per-token credit assignment without any external teacher model.

The key insight is that LLMs already possess a powerful mechanism for using feedback: **in-context learning**. When conditioned on feedback (e.g., "RuntimeError: division by zero at line 73"), the same model can often identify its plausible mistakes and propose corrections. SDPO exploits this by re-evaluating the log-probabilities of the original attempt under the feedback-augmented context, yielding a token-level advantage signal.

### Positioning relative to other methods

| Method | Sampling | Signal | Feedback |
|--------|----------|--------|----------|
| SFT / Distillation (Hinton et al.) | off-policy | rich | strong teacher required |
| On-Policy Distillation (Agarwal et al., GKD) | on-policy | rich | strong teacher required |
| RLVR (GRPO) | on-policy | weak (scalar) | environment |
| **SDPO (this paper)** | **on-policy** | **rich (logit-level)** | **environment (self-teacher)** |

SDPO is the only method that achieves rich, dense signal from on-policy training without requiring an external stronger teacher.

---

## 2. The RLRF Setting

The paper formalizes **Reinforcement Learning with Rich Feedback (RLRF)** as a generalization of RLVR. In RLVR, the agent receives only a scalar reward r (e.g., pass/fail). In RLRF, the environment additionally provides tokenized feedback f -- any sequence of tokens describing the outcome: runtime errors, failing unit tests, judge evaluations, etc.

This is not a new environment design; many existing verifiable environments already emit this information. SDPO simply makes use of what was previously discarded.

---

## 3. How SDPO Works

### 3.1 The Self-Teacher

The self-teacher is simply the current policy prompted with additional context. Given a question x and feedback f, the self-teacher is:

```
pi_theta(. | x, f)
```

where f can incorporate:
1. **Environment output** -- runtime errors, failed test cases, judge evaluations
2. **Sample solution** -- a successful attempt from another rollout in the same batch (if one exists)
3. **The student's original attempt** y (optionally included)

The self-teacher re-evaluates the student's original response y by computing the log-probabilities of each token in y under this feedback-augmented context. No additional sampling is needed -- this is a pure forward pass (prefill) on the existing response.

### 3.2 The Reprompting Template

The self-teacher receives a specific prompt structure (Table 2 in the paper):

```
User:   {prompt}
        Correct solution:
        {successful_previous_rollout}
        The following is feedback from your unsuccessful earlier attempt:
        {environment_output}
        Correctly solve the original question.
Assistant: {original_response}
```

Where:
- `{prompt}` = the original question
- `{successful_previous_rollout}` = a correct solution from the current batch (if available; otherwise this paragraph is skipped)
- `{environment_output}` = environment feedback like runtime errors (if the attempt was unsuccessful and no solution exists; otherwise skipped)
- `{original_response}` = the student's original attempt, re-evaluated under this augmented context

If the model's original attempt was already successful, it is passed as the correct solution.

### 3.3 The SDPO Loss

The core distillation loss minimizes the KL divergence between the student and self-teacher distributions at each token position:

```
L_SDPO(theta) = sum_t KL( pi_theta(. | x, y_{<t}) || stopgrad(pi_theta(. | x, f, y_{<t})) )
```

The `stopgrad` operator blocks gradients from flowing through the teacher, preventing the model from regressing toward ignoring the feedback. The teacher's role is purely evaluative: to determine where the student went wrong based on retrospection.

### 3.4 The SDPO Gradient and Advantage

The gradient of the SDPO loss (Proposition 2.1) is:

```
nabla L_SDPO(theta) = E_{y ~ pi_theta(.|x)} [ sum_t E_{y_hat_t ~ pi_theta(.|x,y_{<t})} [
    log( pi_theta(y_hat_t | x, y_{<t}) / pi_theta(y_hat_t | x, f, y_{<t}) ) . nabla_theta log pi_theta(y_hat_t | x, y_{<t})
]]
```

This reveals that SDPO is a **(negated) logit-level policy gradient** where advantages are estimated using the self-teacher. Comparing GRPO and SDPO advantages directly:

```
A_{i,t}^GRPO := r_i - mean{r_i}_{i=1}^G    (constant across all tokens in a rollout)

A_{i,t}^SDPO(y_hat_{i,t}) = log( pi_theta(y_hat_{i,t} | x, f_i, y_{i,<t}) / pi_theta(y_hat_{i,t} | x, y_{i,<t}) )
```

Key differences:
- **GRPO** advantages are constant within a rollout (one scalar per sequence) and only applied to the sampled token y_{i,t}
- **SDPO** advantages vary per-token AND per-vocabulary-item (logit-level), giving |V| * |y| unique advantage values per sequence
- SDPO advantages are **zero** where student and teacher agree, **positive** for tokens the teacher finds more likely (given feedback), and **negative** for tokens the teacher considers less likely
- This means SDPO assigns credit at the **logit level** -- the finest granularity possible

This tight connection to RLVR means SDPO can be implemented as a **drop-in replacement** in standard RLVR pipelines by simply swapping the advantage computation.

---

## 4. Credit Assignment Granularity

The paper ablates three levels of credit assignment in SDPO:

1. **Logit-level SDPO** (default): Credit assignment over the top-K most likely tokens (under the student) at each position. Assigns |y| * (K+1) unique advantages per sequence.
2. **Token-level SDPO**: Credit assignment only over the single most likely token at each position.
3. **Sequence-level SDPO**: Average all SDPO advantages across tokens to produce a single scalar per sequence (analogous to GRPO's approach, but using rich feedback).

Results (LCBv6): logit-level > token-level > sequence-level > GRPO. Even sequence-level SDPO significantly outperforms GRPO, showing that the rich feedback itself is valuable even without dense credit assignment.

---

## 5. Stability Improvements

Two modifications significantly enhance training stability:

### 5.1 Teacher Regularization

The self-teacher's parameters drift during training. Two strategies stabilize this:

**EMA (Exponential Moving Average):** The teacher is an EMA of the student parameters:
```
theta_teacher = (1 - tau) * theta_teacher + tau * theta_student
```
Default `teacher_update_rate` = 0.05. The EMA teacher runs on the RefWorker in the verl infrastructure.

**Trust-Region:** Instead of EMA, the current teacher is interpolated with the initial teacher:
```
pi_teacher = (1 - alpha_tr) * pi_theta_init + alpha_tr * pi_theta
```
This constrains how far the teacher can drift from the initial model.

Both strategies outperform an unregularized teacher (raw `q_theta`). Interestingly, the frozen initial teacher (`q_theta_ref`) also performs well, but trust-region and EMA are best overall (Table 4).

### 5.2 Symmetric Jensen-Shannon Divergence

Instead of asymmetric KL, SDPO uses JSD (alpha=0.5) for the distillation loss:

```
JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m),  where m = 0.5 * (p + q)
```

The `alpha` parameter controls the interpolation:
- alpha = 0.0: forward KL (mode-seeking)
- alpha = 0.5: JSD (symmetric, default)
- alpha = 1.0: reverse KL (mode-covering)

JSD has been shown to improve stability in on-policy distillation (Agarwal et al., 2024).

---

## 6. Compute Time and Memory

### 6.1 Time Overhead

The only additional computation vs GRPO is computing log-probabilities from the self-teacher (a forward pass on the reprompted sequences). This is parallelizable and substantially faster than sequential generation.

Measured overhead (Figure 5):
- Without code environment: **+5.8%** time per training step
- With code environment: **+17.1%** time per training step (longer reprompts)

### 6.2 Memory: Top-K Distillation

Naive KL computation requires full logits from both student and teacher (|V| ~ 150K for Qwen3). SDPO approximates this with **top-K distillation**:

- Compute only the top-K logits of the student and the corresponding teacher logits
- Add a "tail" probability bucket capturing the remaining probability mass
- Default K = 100, which captures most of the information while using <0.1% of the vocabulary

This avoids virtually any memory overhead while preserving the quality of the KL estimate.

Config: `distillation_topk = 100`, `distillation_add_tail = True`

---

## 7. Off-Policy Extension via Importance Sampling

The SDPO gradient naturally extends to off-policy data via PPO-style clipped importance sampling:

```
L_SDPO_offpolicy = sum_t ( pi_theta(y_t | x, y_{<t}) / pi_theta_old(y_t | x, y_{<t}) ) * A_t^SDPO
```

with clipping via `is_clip = 2.0` (default). This is important because in practice, SDPO generates a batch and then performs gradient updates, making the data slightly off-policy. The clipping prevents instability from large importance ratios.

---

## 8. Learning Without Rich Feedback (Section 3)

SDPO works even in standard RLVR environments that only return scalar rewards. In this mode, SDPO uses **successful attempts from the current batch as "feedback" for failed attempts on the same question**. If rollout group G for question x contains both successful and failed attempts, the successful attempt becomes the demonstration for the self-teacher when re-evaluating failed attempts.

### 8.1 Results on Reasoning Benchmarks

Tasks: Science Q&A (Chemistry, Physics, Biology, Materials science from SciKnowEval L3) and Tool Use (ToolAlpaca).
Models: Qwen3-8B, Olmo3-7B-Instruct.
Hardware: 4x NVIDIA GH200 per run, ~6 hours total.

**Table 3 highlights (avg@16 accuracy):**

| Task + Model | GRPO (5h) | SDPO (1h) | SDPO (5h) |
|---|---|---|---|
| Chemistry + Qwen3-8B | 74.5 | 73.2 | **80.9** |
| Chemistry + Olmo3-7B | 56.7 | **68.0** | **80.0** |
| Physics + Qwen3-8B | 72.7 | 66.6 | **75.6** |
| Biology + Qwen3-8B | 59.9 | **50.6** | 56.8 |
| Materials + Qwen3-8B | 77.1 | 72.1 | **78.4** |
| Tool use + Qwen3-8B | 67.7 | **68.0** | **68.5** |

Key findings:
- SDPO outperforms GRPO on almost all tasks, often substantially
- SDPO at 1 hour often matches or exceeds GRPO at 5 hours (up to **6x speedup** on Chemistry)
- On Chemistry with Olmo3-7B-Instruct, SDPO achieves the 5h GRPO accuracy in 50 minutes

### 8.2 Concise Reasoning

SDPO produces **3-11x shorter** generations than GRPO while achieving higher accuracy. On Chemistry with Olmo3-7B-Instruct, SDPO achieves an **11x reduction** in response length.

Qualitatively, GRPO responses contain "superficial" reasoning: filler phrases ("Hmm", "Wait"), circular logical loops, and repeated calculations. SDPO's dense credit assignment assigns specific advantages to each token, leading to sparse activation and avoiding these patterns.

---

## 9. Learning With Rich Environment Feedback (Section 4)

### 9.1 Code Generation on LiveCodeBench v6

Task: 131 competitive programming problems from LCBv6 (Feb-May 2025) with LeetCode-style feedback (runtime errors, wrong answers, failed test cases).
Model: Qwen3-8B (default), scaling study across Qwen3 family.
Setup: Public tests for training evaluation, private tests for validation. 4 rollouts per question.

**Key results:**
- **SDPO: 48.8%** vs **GRPO: 41.2%** final accuracy on LCBv6
- SDPO reaches GRPO's final accuracy in **4x fewer generations**
- SDPO outperforms the strongest instruct models on the public LCBv6 leaderboard: Claude Sonnet 4 (40.5%) and Claude Opus 4 (39.7%)

### 9.2 Scaling with Model Size

The gains from SDPO are **tightly coupled with model scale** (Figure 8):

| Model | Base | GRPO | SDPO |
|---|---|---|---|
| Qwen3-0.6B | ~0.05 | ~0.15 | ~0.18 |
| Qwen3-1.7B | ~0.13 | ~0.28 | ~0.33 |
| Qwen3-4B | ~0.25 | ~0.40 | ~0.44 |
| Qwen3-8B | ~0.28 | ~0.41 | **~0.49** |

SDPO significantly outperforms GRPO on larger models while only slightly improving on smaller ones. On models weaker than Qwen3-0.6B (e.g., Qwen2.5-1.5B), SDPO can actually **underperform GRPO**.

**Takeaway:** The self-teacher's ability to perform accurate retrospection is an emergent phenomenon that scales with the model's in-context learning ability.

### 9.3 The Self-Teacher Improves During Training

Unlike standard distillation where the teacher is frozen, the SDPO self-teacher improves throughout training because it shares (regularized) parameters with the student. Figure 10 (right) shows:
- The self-teacher's generative accuracy improves continuously
- The final student **surpasses** the initial teacher's accuracy
- This is true bootstrapping: a weak model improves itself into a strong one

### 9.4 Which Feedback Is Most Informative?

Ablation of feedback components (Table 6):

| Feedback f | Teacher accuracy | Student accuracy (SDPO) |
|---|---|---|
| output only | 32.5% | 39.9% |
| own solution only | 42.4% | 42.6% |
| output + own solution | **42.5%** | **48.3%** |
| y + output + own solution | 39.3% | 44.5% |

- Environment output and sample solutions are **complementary**
- Including the student's original attempt y in the teacher context **reduces exploration** (biases teacher toward student's distribution)
- Best combination: output + own solution, without student's original attempt in the prompt

### 9.5 Catastrophic Forgetting

SDPO maintains better performance on holdout tasks than GRPO (Table 5):

| Method | LCBv6 | IFEval | ArenaHard-v2 | MMLU-Pro | Avg (holdout) |
|---|---|---|---|---|---|
| Base | 27.9 | 83.9 | 13.9 | 62.5 | 43.5 |
| GRPO | 41.2 | 82.2 | 11.4 | 62.3 | 41.8 |
| **SDPO** | **48.8** | **83.2** | **11.7** | **62.9** | **42.4** |
| SFT on self-teacher | 42.7 | 83.7 | 10.1 | 61.9 | 41.4 |

SDPO achieves the best performance-forgetting tradeoff. SFT on the self-teacher (off-policy) significantly underperforms SDPO while also forgetting more.

---

## 10. Combining GRPO and SDPO (Section 4.5)

A hybrid approach is possible:

```
A_{i,t}^{SDPO+GRPO}(y_hat_{i,t}) := lambda * A_{i,t}^GRPO + (1 - lambda) * A_{i,t}^SDPO(y_hat_{i,t}),    lambda in [0, 1]
```

Results (Figure 11, lambda=0.9):
- SDPO+GRPO significantly outperforms both SDPO and GRPO on weak models (Qwen3-0.6B)
- SDPO+GRPO slightly **underperforms** pure SDPO on strong models (Qwen3-8B)

The rationale: GRPO advantages are unbiased Monte Carlo estimates of the reward objective, while SDPO advantages are biased (from feedback/bootstrapping) but lower variance. For weak models where SDPO advantages are unreliable, the GRPO signal helps stabilize training.

---

## 11. Test-Time Self-Distillation (Section 5)

SDPO can be applied at **test time** on individual hard questions. This is a form of test-time training (TTT) where the model specializes to a single question.

### 11.1 How It Works

1. The model attempts question x, generating y_1 and receiving feedback f_1
2. Instead of appending (y_1, f_1) to the context (multi-turn), SDPO distills pi_theta(. | x, (y_1, f_1)) into the weights: theta_1 -> theta_2
3. The updated model attempts x again, getting y_2, f_2
4. Repeat: SDPO distills the accumulated context into weights

This **compresses context into model weights**, avoiding the transformer context length bottleneck. Multi-turn sampling is limited by the context window (Qwen3-8B's 40k tokens), while SDPO can learn indefinitely.

### 11.2 Results on Hard LCBv6 Questions

**Very hard tasks** (pass@64 < 0.03, 9 questions):
- Best-of-k at k=2750: 41.5% discovery rate
- Multi-turn at k=2750: 35.6% discovery rate
- **SDPO at k=2750: 53.2% discovery rate**
- SDPO reaches 22% discovery with **3x fewer** attempts than alternatives

**Hard tasks** (pass@64 < 0.5, 19 questions):
- SDPO achieves 78% discovery@2750 vs 72.3% (best-of-k) and 68.4% (multi-turn)
- SDPO discovers solutions in 70% of cases within k=1000

**Remarkable finding:** SDPO uniquely solves Question 3, which neither best-of-k nor multi-turn can solve within 2750 attempts. SDPO discovers it after 321 attempts (20 SDPO iterations with batch size 16).

### 11.3 Why It Works

- The initial self-teacher accuracy is <1% on most hard questions (0% on 78% of them)
- Yet SDPO's credit assignment is sufficient to iteratively refine the policy
- Rich environment feedback provides signal **even before any solution is found** -- this is impossible in standard RLVR
- Context compression via distillation avoids the diminishing returns of multi-turn (context window saturation at ~1000 steps)

---

## 12. Implementation Details

### 12.1 Framework

Built on **verl** (https://github.com/verl-project/verl). SDPO is a drop-in modification:
- Set `actor.policy_loss.loss_mode = "sdpo"` (default: "vanilla" for GRPO)
- Configure `actor.self_distillation` settings
- The EMA teacher runs on the RefWorker (same infrastructure used for KL reference in GRPO)

### 12.2 Key Configuration Options

```yaml
actor:
  policy_loss:
    loss_mode: "sdpo"                           # Enable SDPO (vs "vanilla" for GRPO)

  self_distillation:
    # Core
    full_logit_distillation: true                # Full-logit KL (vs token-level only)
    alpha: 0.5                                   # 0.0=forward KL, 0.5=JSD, 1.0=reverse KL
    success_reward_threshold: 1.0                # Min reward to count as successful demo

    # Teacher regularization
    teacher_regularization: "ema"                # "ema" or "trust-region"
    teacher_update_rate: 0.05                    # EMA rate or trust-region mixing coeff

    # Memory optimization
    distillation_topk: 100                       # Top-K logits for KL approximation
    distillation_add_tail: true                  # Include tail probability bucket

    # Off-policy correction
    is_clip: 2.0                                 # Importance sampling ratio clip

    # Reprompting
    max_reprompt_len: 10240                      # Max tokens in reprompted prompt
    reprompt_truncation: "right"                 # "left", "right", or "error"
    dont_reprompt_on_self_success: true           # Skip self-demo if already correct
    remove_thinking_from_demonstration: true      # Strip <think>...</think> tags

    # Feedback
    include_environment_feedback: true            # Use env feedback (errors, test output)
    environment_feedback_only_without_solution: true  # Only use feedback when no solution exists

    # Templates (with placeholders)
    reprompt_template: "..."                      # {prompt}, {solution}, {feedback}
    solution_template: "..."                      # {successful_previous_attempt}
    feedback_template: "..."                      # {feedback_raw}
```

### 12.3 Training Setup

- **On-policy:** One gradient step per generation batch (strictly on-policy)
- **GRPO baseline comparison:** 4 off-policy mini-batch steps per generation batch
- **Hardware:** 4x NVIDIA GH200 per run
- **Training time:** ~6 hours total (including init/validation) per run
- **Rollout group size:** G rollouts per question (4 for LCBv6, 16 for science Q&A)
- **Metric:** avg@16 (average accuracy across 16 samples per question)

### 12.4 Data Pipeline

```bash
# Load dataset
python data/load_dataset.py --dataset_name Chemistry

# Split into train/test
python data/split_tasks.py --json_path datasets/chemistry.json --test_ratio 0.1

# Preprocess to parquet format (required by verl)
python data/preprocess.py --data_source DATASET_PATH
```

### 12.5 Running Experiments

```bash
# Without rich feedback
bash experiments/generalization/run_sdpo_all.sh          # SDPO
bash experiments/generalization/run_baseline_grpo_all.sh  # GRPO baseline

# With rich feedback (LCBv6)
bash experiments/rich_feedback/run_sdpo.sh                # SDPO
bash experiments/rich_feedback/run_baseline_grpo.sh       # GRPO baseline

# Test-time self-distillation
bash experiments/ttt/run_multiturn_all.sh                 # Multi-turn baseline
python baseline_multiturn/multiturn.py --data-dir=lcb_v6_singles/q_120
```

---

## 13. Limitations

1. **Model scale dependency:** SDPO requires sufficiently capable base models for accurate in-context retrospection. Below ~1.5B parameters, it can underperform GRPO.
2. **Feedback quality sensitivity:** Misleading or uninformative environment feedback degrades learning.
3. **Computational overhead:** While small (~6-17%), it is more pronounced for smaller models with short generations where generation time is comparatively small.
4. **Biased advantages:** SDPO advantages are biased with respect to the reward objective (unlike GRPO's unbiased Monte Carlo estimates). This is the classic bias-variance tradeoff.

---

## 14. Relevance to ARC-AGI REPL Environment

SDPO is directly relevant to our ARC-AGI setup for several reasons:

### 14.1 Rich Feedback Availability

Our REPL environment provides exactly the kind of rich feedback SDPO is designed for:
- **REPL output:** When the model executes code in the REPL, it gets stdout/stderr, including error messages, tracebacks, and intermediate outputs
- **Runtime errors:** Python exceptions with line numbers and stack traces (analogous to the LeetCode feedback in the paper)
- **Grid comparisons:** When the model's predicted grid doesn't match the expected output, the environment can show which cells differ
- **Verifiable rewards:** ARC tasks have ground-truth grids, providing binary correctness signals

### 14.2 Self-Distillation Without External Teacher

We don't have access to a stronger teacher model for ARC-AGI. SDPO's self-distillation approach is ideal because:
- The model teaches itself using its own in-context learning
- No need for a separate, larger model (unlike prime-rl's on-policy distillation which requires a teacher)
- The approach is self-bootstrapping: the teacher improves as the student improves

### 14.3 Test-Time Self-Distillation for Hard Tasks

ARC tasks are individually hard and diverse. Test-time self-distillation could be valuable:
- Each ARC task is unique; specializing the model per-task at test time makes sense
- The REPL provides rich per-attempt feedback (errors, partial outputs)
- Context compression avoids the limited context window problem in multi-turn approaches
- SDPO solved questions that multi-turn and best-of-k could not (Question 3 in the paper)

### 14.4 Implementation Compatibility

SDPO is built on verl, and the modification is minimal (swap advantage computation). Key considerations:
- Our current stack uses prime-rl; we would need to either port SDPO's advantage computation or switch to verl
- The core change is small: compute self-teacher log-probs on reprompted sequences, use the log-ratio as per-token advantage
- The reprompting template can be adapted for ARC: include REPL output, error messages, and successful grid solutions from peer rollouts

### 14.5 Specific Opportunities

1. **REPL error feedback as environment output:** When the model writes Python code to solve ARC tasks and gets a runtime error, this error message is the `{environment_output}` in the SDPO template.
2. **Successful peer solutions as demonstrations:** In a rollout group, if one attempt solves the ARC task, its solution becomes the `{successful_previous_rollout}` for failed attempts.
3. **Grid diff as feedback:** We could enrich the environment feedback by showing the diff between predicted and expected grids, giving the self-teacher precise information about what went wrong.
4. **Concise reasoning:** SDPO's tendency to produce shorter, more efficient reasoning traces (3-11x shorter than GRPO) could help with our context window constraints in the REPL.

### 14.6 Open Questions

- How well does SDPO work with the specific model sizes we're targeting? The paper shows gains scale with model size; sub-1.5B may not benefit.
- Can we adapt the reprompting template for ARC's multi-step REPL interactions (the model may make multiple REPL calls per attempt)?
- How should we handle the combination of REPL feedback and grid verification feedback?
- Would SDPO+GRPO hybrid be better for our setup, given we're likely using 7-8B models?
- Can test-time self-distillation be applied to ARC evaluation, where each task is seen only once?

---

## 15. Key Takeaways

1. **SDPO is a minimal modification to GRPO** that replaces scalar advantages with dense, logit-level advantages from a self-teacher. It can be implemented by swapping the advantage computation in any RLVR pipeline.

2. **Rich feedback is underutilized** in current RLVR. Even standard environments that only provide scalar rewards implicitly contain rich feedback in the form of successful peer attempts.

3. **Dense credit assignment matters.** Logit-level > token-level > sequence-level. The granularity of credit assignment directly impacts sample efficiency and final performance.

4. **Self-teaching scales with model capability.** The approach is emergent: small models barely benefit, but 8B+ models show substantial gains. This parallels the scaling of in-context learning.

5. **On-policy self-distillation avoids forgetting** better than off-policy SFT on the self-teacher's outputs.

6. **Test-time self-distillation is a powerful capability** that compresses context into weights, overcoming the context window bottleneck and solving problems that multi-turn and best-of-k cannot.

7. **For our ARC-AGI work:** SDPO is a strong candidate for improving our REPL-based training. The REPL naturally provides rich feedback, we lack an external teacher, and the test-time self-distillation capability aligns with ARC's per-task evaluation format.

---

## Citation

```bibtex
@article{hubotter2026reinforcement,
  title = {Reinforcement Learning via Self-Distillation},
  author = {H\"ubotter, Jonas and L\"ubeck, Frederike and Behric, Lejs and Baumann, Anton and Bagatella, Marco and Marta, Daniel and Hakimi, Ido and Shenfeld, Idan and Kleine Buening, Thomas and Guestrin, Carlos and Krause, Andreas},
  year = {2026},
  journal = {arXiv preprint arXiv:2601.20802},
}
```
