# On-Policy Self-Distillation (OPSD) -- Research Notes

**Paper:** "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"
**Authors:** Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, Aditya Grover
**Affiliations:** UCLA, HKU, Meta Superintelligence Labs
**PDF:** `../papers/OPSD.pdf`
**Blog:** https://siyan-zhao.github.io/blog/2026/opsd/

---

## 1. Core Idea

OPSD eliminates the need for a separate, larger teacher model in on-policy distillation. Instead, **the same model serves as both teacher and student**, differentiated only by their conditioning context:

- **Student policy** `p_S(. | x)`: sees only the problem `x` (matches inference-time conditions).
- **Teacher policy** `p_T(. | x, y*)`: sees the problem `x` AND the ground-truth solution `y*` (privileged information).

Both policies share identical parameters `theta`. The information asymmetry created by conditioning on `y*` is what makes the teacher's distribution more informative than the student's.

### The Key Hypothesis: Rationalization Is Easier Than Generation

The paper's central insight draws from a human learning analogy: when a student struggles with a problem, examining the correct solution and rationalizing *why* it works is significantly easier than generating the solution from scratch. For LLMs, prior work has established that **evaluation is easier than generation** (Sun et al., 2024; Naor, 1996). OPSD extends this: **rationalization -- explaining a given correct answer -- is similarly easier than generation**. When conditioned on `y*`, the model can produce better-informed next-token distributions even though it uses the same parameters.

---

## 2. Method: Step-by-Step Training Process

### Algorithm (from Algorithm 1 in the paper)

**Input:** Reasoning dataset `S = {(x_i, y*_i)}`, language model `p_theta`, divergence measure `D` (e.g., JSD_beta)

**For each training step:**

1. **Sample a minibatch** `B` from `S`.

2. **For each** `(x, y*)` in `B`:

   a. **Student generates on-policy rollout:** `y_hat ~ p_S(. | x)` -- the student samples a complete response given only the problem. This is a standard autoregressive generation with no access to `y*`.

   b. **Both policies evaluate the student's rollout:** At each token position `n` in the student-generated sequence `y_hat`, compute next-token distributions over the full vocabulary `V`:
      - Student: `p_S(y_n | x, y_hat_{<n})`
      - Teacher: `p_T(y_n | x, y*, y_hat_{<n})`

   c. **Compute token-wise divergence** along the student's rollout:
      ```
      l(x, y*) = D(p_T || p_S)(y_hat | x) = (1/|y_hat|) * sum_{n=1}^{|y_hat|} D(p_T(. | y_hat_{<n}, x, y*) || p_S(. | y_hat_{<n}, x))
      ```

3. **Aggregate batch loss:** `L_OPSD = (1/|B|) * sum l(x, y*)`

4. **Update parameters:** `theta <- theta - eta * grad_theta L_OPSD(theta)`
   - Critically, **gradients flow only through the student's logits**. The teacher `p_T` acts as a fixed distribution target conditioned on privileged information `(x, y*)`.

5. **Return** trained parameters `theta` for the inference-time student policy `p_S(. | x)`.

### Important Implementation Detail: Fixed Teacher Policy

The teacher policy is **fixed to the initial policy** (i.e., the pre-training checkpoint), rather than updating alongside the student. This stabilizes training and implicitly acts as regularization to prevent excessive deviation from the initial policy.

---

## 3. Prompt Templates: How Privileged Information Is Injected

### Student Prompt (inference-time conditions)
```
Problem: Find the derivative of f(x) = 3x^2 + 2x - 5 at x = 2

Answer:
```

### Teacher Prompt (privileged information injected)
```
Problem: Find the derivative of f(x) = 3x^2 + 2x - 5 at x = 2

Here is a reference solution:
First find f'(x) = 6x + 2, then evaluate at x = 2: f'(2) = 6(2) + 2 = 14

After understanding the reference solution, please try to solve this problem
using your own approach below:

Answer:
```

The key design choice: the teacher prompt includes the ground-truth solution `y*` as a "reference solution" and then asks the model to generate its own approach. This encourages the teacher to **rationalize** the solution -- to naturally evaluate the student's generation in light of the known correct answer. The rationalization happens **implicitly through one forward pass** (the teacher does not generate tokens, it only produces informed next-token distributions over the student's trajectory).

---

## 4. Training Objectives

### Primary: Full-Vocabulary Logit Distillation

The trajectory-averaged, token-wise divergence is defined as:

```
D(p_T || p_S)(y_hat | x) = (1/|y_hat|) * sum_{n=1}^{|y_hat|} D(p_T(. | x, y*, y_hat_{<n}) || p_S(. | x, y_hat_{<n}))
```

where `D` can be any distribution divergence measure. The paper uses **generalized Jensen-Shannon Divergence** with weight `beta`:

```
JSD_beta(p_T || p_S) = beta * D_KL(p_T || m) + (1 - beta) * D_KL(p_S || m)
```

where `m = beta * p_T + (1 - beta) * p_S` is the interpolated mixture distribution.

The paper uses `beta = 0.5` (standard JSD) in experiments.

The overall loss to minimize:

```
L(theta) = E_{(x,y*) ~ S} [ E_{y_hat ~ p_S(.|x)} [ D(p_T || p_S)(y_hat | x) ] ]
```

### Alternative: Sampled-Token Policy Gradient

Instead of matching full distributions, define a per-token advantage:

```
A_n(x, y_hat) = log p_T(y_hat_n | x, y*, y_hat_{<n}) - log p_S(y_hat_n | x, y_hat_{<n})
```

And optimize a policy-gradient-style objective:

```
L(theta) = -E_{(x,y*) ~ S} [ E_{y_hat ~ p_S(.|x)} [ (1/|y_hat|) * sum_n A_n(x, y_hat) * log p_S(y_hat_n | x, y_hat_{<n}) ] ]
```

Here `A_n` is treated as a constant (stop-gradient), so gradients take the standard REINFORCE form: `A_n * grad log p_S`.

### Ablation Result (Table 3, Qwen3-4B, pass@8)

| Method Variant                    | AIME25 | HMMT25 |
|-----------------------------------|--------|--------|
| OPSD w/ Full-vocabulary logit     | **84.1** | **60.0** |
| OPSD w/ Sampled-token distill     | 82.1   | 57.3   |

Full-vocabulary provides richer supervision but uses more memory.

---

## 5. Comparison with Other Training Methods

| Property               | SFT/Off-Policy Distill | GRPO  | On-Policy Distill | OPSD (Ours) |
|------------------------|------------------------|-------|-------------------|-------------|
| On-Policy Data         | No                     | Yes   | Yes               | Yes         |
| Dense Learning Signal  | Yes                    | No    | Yes               | Yes         |
| Low Sampling Cost      | Yes                    | No    | Yes               | Yes         |
| No External Teacher    | Yes                    | Yes   | No                | Yes         |

OPSD uniquely combines **all four** desirable properties.

### Key Advantages Over GRPO

1. **Dense vs. sparse signal:** GRPO assigns the same binary reward to all tokens in a response. OPSD provides per-token distributional guidance -- the teacher shows the student what the *entire* next-token distribution should look like at every position.

2. **No vanishing gradients:** GRPO's signal vanishes when all sampled responses are either all correct or all incorrect (the group-normalized advantage becomes zero). OPSD always provides a meaningful gradient because the teacher and student distributions differ.

3. **Single rollout:** OPSD needs only 1 rollout per problem. GRPO requires a group of rollouts (typically 8) to estimate advantages.

---

## 6. Token Efficiency: 4-8x Improvement vs. GRPO

This is one of the paper's strongest claims. Under the same effective training batch size on Qwen3-4B:

- **GRPO:** 8 rollouts per problem, each up to 16,384 tokens = ~131k tokens per problem.
- **OPSD:** 1 rollout per problem, capped at 2,048 tokens = ~2k tokens per problem.

The ratio is approximately **64x fewer generated tokens** per problem, though the paper conservatively reports **4-8x overall improvement** when accounting for the fact that OPSD requires computing full-vocabulary logits (teacher + student forward passes).

From Figure 3 in the paper: when plotted against total tokens generated, OPSD reaches higher average@16 accuracy with substantially fewer tokens. When plotted against gradient steps, OPSD and GRPO achieve comparable performance -- confirming the efficiency comes from needing far fewer tokens per step.

### Why It Works With Fewer Tokens

Dense token-level supervision from the teacher compensates for shorter generation lengths. Each generated token receives a distributional learning signal (teacher's full next-token distribution), rather than a single scalar reward applied uniformly.

---

## 7. Model Scale Requirements

Tested on Qwen3 family at three scales (Table 2, average@16 across AIME24, AIME25, HMMT25, AMO-Bench):

| Model     | Base   | + SFT  | + GRPO | + OPSD |
|-----------|--------|--------|--------|--------|
| Qwen3-8B  | 50.0   | 50.0   | 51.3   | **52.2** |
| Qwen3-4B  | 48.3   | 49.6   | 49.6   | **50.6** |
| Qwen3-1.7B | 28.8 | 28.0   | **30.5** | 30.4  |

### Scale-Dependent Effectiveness

- **At 8B:** OPSD clearly outperforms both GRPO and SFT across all benchmarks. The teacher policy benefits most from the model's capacity to rationalize solutions.
- **At 4B:** OPSD matches or exceeds GRPO. On HMMT25, OPSD achieves 45.8 vs. GRPO's 42.7. On AMO-Bench, OPSD gets 13.5 vs. GRPO's 12.8.
- **At 1.7B:** OPSD provides marginal gains, roughly matching GRPO's average. The paper explains: "conditioning on y* must produce a better-informed next-token distribution. When capacity is insufficient, the teacher signal is weak." The 1.7B model lacks the capacity to meaningfully rationalize solutions even when given the answer.

**Takeaway:** OPSD requires models of at least ~4B parameters for consistent improvements. Below that, the model's ability to create useful information asymmetry via conditioning breaks down.

---

## 8. Experimental Details

### Training Data

- Subset of **OpenThoughts** (Guha et al., 2025) -- up to 30K problem-solution pairs with chain-of-thought reasoning.

### Hyperparameters (from Table 6, Appendix)

| Parameter                      | GRPO          | OPSD          |
|-------------------------------|---------------|---------------|
| Learning Rate                 | 2e-5          | 2e-5          |
| Batch Size (per device)      | 1             | 1             |
| Gradient Accumulation Steps   | 4             | 4             |
| Effective Batch Size          | 32            | 32            |
| LoRA Rank (r)                 | 64            | 64            |
| LoRA Alpha                    | 128           | 128           |
| LoRA Target Modules           | q,k,v,o,gate,up,down_proj | q,k,v,o,gate,up,down_proj |
| Max Completion Length          | 16,000        | 2,048         |
| Generations per Prompt        | 8             | 1             |
| Temperature                   | 1.2           | 1.2           |
| KL Coefficient (beta)         | 0.0           | --            |

### Evaluation Parameters (Table 5)

- Max New Tokens: 38,912
- Thinking Mode: Enabled
- Top-p: 0.95
- Samples per Prompt: 16 (average@16 reported)

### Infrastructure

- 8x A100 GPUs with LoRA
- AdamW optimizer with cosine LR schedule, warmup ratio 0.1
- bfloat16 precision
- Gradient checkpointing + Flash Attention 2
- Divergence: JSD with beta=0.5

---

## 9. Generation Length Ablation (Figure 4)

On Qwen3-4B, varying the student's on-policy generation length:

| Generation Length | Effect on Pass@K |
|-------------------|-----------------|
| 1,024 tokens      | Baseline; lowest performance |
| 2,048 tokens      | Significant improvement over 1k |
| 4,096 tokens      | Further improvement; best at higher K values |

Longer generations provide more teacher feedback tokens per problem, but with diminishing returns. The paper hypothesizes that **early tokens are more important** for distillation as they represent more critical branching points in the reasoning chain.

---

## 10. Benchmarks and Detailed Results (Table 2)

### Qwen3-8B (average@16)

| Method | AIME24 | AIME25 | HMMT25 | AMO-Bench | Average |
|--------|--------|--------|--------|-----------|---------|
| Base   | 75.2   | 68.3   | 43.1   | 13.4      | 50.0    |
| + SFT  | 76.3   | 66.2   | 44.7   | 12.9      | 50.0    |
| + GRPO | 76.7   | 68.7   | 45.0   | 14.8      | 51.3    |
| **+ OPSD** | **77.5** | **69.8** | **47.1** | 14.3  | **52.2** |

### Qwen3-4B (average@16)

| Method | AIME24 | AIME25 | HMMT25 | AMO-Bench | Average |
|--------|--------|--------|--------|-----------|---------|
| Base   | 74.6   | 65.8   | 40.3   | 12.4      | 48.3    |
| + SFT  | 75.2   | 66.3   | 44.4   | 12.5      | 49.6    |
| + GRPO | 75.6   | 67.1   | 42.7   | 12.8      | 49.6    |
| **+ OPSD** | **76.0** | **66.9** | **45.8** | **13.5** | **50.6** |

---

## 11. Limitations and Future Directions

1. **No explicit correctness verification:** The current framework does not verify whether the student's generated output is actually correct. Incorporating outcome-based verification (as in GRPO) alongside distribution matching could yield further gains.

2. **Problem difficulty matters:** If reasoning problems exceed the model's comprehension threshold, the teacher cannot provide meaningful supervision even with access to ground-truth. This suggests **curriculum learning** -- gradually increasing difficulty -- could enhance training effectiveness.

3. **Scale ceiling unknown:** Experiments only go up to 8B. Whether the trend of increasing benefit with model scale continues at 70B+ is an open question.

4. **Memory cost of full-vocabulary logits:** Computing divergence over the entire vocabulary at every token position incurs high peak memory, creating a performance-memory trade-off compared to the sampled-token variant.

5. **Proposed future direction -- Group Self-Distillation:** Combining distribution matching with outcome-based verification using the model's own correct reasoning traces.

---

## 12. Relevance to ARC-AGI

OPSD is highly relevant to ARC-AGI training for several reasons:

### Direct Applicability

1. **We have ground-truth outputs for training examples.** ARC-AGI tasks come with input-output pairs. The ground-truth output grid can serve as the privileged information `y*` in the teacher prompt.

2. **Verifiable answers.** ARC outputs are exact grid matches -- even more cleanly verifiable than mathematical reasoning, making it possible to combine OPSD with outcome-based verification.

3. **Token efficiency matters.** ARC-AGI tasks involve complex spatial reasoning with potentially long chain-of-thought. OPSD's 4-8x token efficiency over GRPO directly translates to faster iteration cycles with our limited compute budget.

### Key Considerations for ARC-AGI Adaptation

1. **What is `y*` for ARC?** The privileged information could be:
   - The correct output grid itself
   - The correct output grid + a description of the transformation rule
   - Multiple training examples showing the pattern

2. **Teacher prompt template for ARC:** Would need adaptation, e.g.:
   ```
   Here are the training examples:
   [input_1] -> [output_1]
   [input_2] -> [output_2]

   Here is the correct output for the test input:
   [test_input] -> [correct_test_output]

   After understanding the transformation pattern from the correct output,
   please solve this problem using your own approach below:
   ```

3. **Model scale constraint:** The paper shows OPSD requires ~4B+ parameters for meaningful gains. We should prioritize experiments with 4B and 8B models rather than smaller ones.

4. **Combination with GRPO:** The paper's limitations section hints at combining distribution matching with outcome-based rewards. For ARC-AGI, a hybrid approach could use OPSD for dense token-level guidance and GRPO-style rewards for final answer correctness.

5. **Curriculum learning:** ARC tasks vary greatly in difficulty. Following the paper's suggestion, starting with easier ARC tasks and progressively increasing difficulty could improve training effectiveness with OPSD.

### Open Questions for Our Project

- How does the model's spatial reasoning capacity affect its ability to rationalize ARC transformations when given the correct output?
- Should `y*` include only the output grid, or also an explicit description of the transformation rule (if available)?
- Can we use the ARC training examples themselves as a form of in-context privileged information (showing more examples to the teacher than the student)?
- What is the right generation length for ARC reasoning? The paper's sweet spot was 2k-4k tokens for math; ARC reasoning might be shorter or longer.

---

## 13. Connections to Related Work

- **DAgger (Ross et al., 2011):** OPSD can be seen as a DAgger-style imitation learning approach where the teacher provides corrective supervision on states visited by the student.
- **GKD (Agarwal et al., 2024):** Generalized Knowledge Distillation -- OPSD's full-vocabulary variant is closely related but removes the external teacher requirement.
- **Lu & Lab (2025):** On-policy distillation from Thinking Machines -- OPSD's sampled-token variant follows this approach but uses same-model self-distillation instead of a separate teacher.
- **Context Distillation (Snell et al., 2022):** Shows models can internalize privileged context through SFT. OPSD extends this to on-policy training with dense token-level objectives.

---

*Last updated: 2026-02-22*
