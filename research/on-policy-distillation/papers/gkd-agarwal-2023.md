# Generalized Knowledge Distillation (GKD)

**Paper**: "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
**Authors**: Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem
**Affiliations**: Google DeepMind, Mila, University of Toronto
**Published**: ICLR 2024
**arXiv**: [2306.13649v3](https://arxiv.org/abs/2306.13649)
**PDF**: `papers/gkd-agarwal-2023.pdf`

---

## 1. Core Problem: Distribution Mismatch in Standard KD

Standard knowledge distillation (KD) for auto-regressive language models suffers from a fundamental **train-inference distribution mismatch**. The issue is:

- During training, the student is trained on output sequences from a fixed dataset -- either ground-truth targets or teacher-generated sequences.
- During inference, the student generates tokens auto-regressively, conditioned on its *own* previously generated tokens.
- The partial sequences the student encounters during inference can be very different from those seen during training.

This is the same **exposure bias** problem known from imitation learning (Pomerleau, 1991; Ross & Bagnell, 2010). Since each prediction is contingent on all prior steps, an error at an early step can cascade and compound, leading to poor-quality generation.

Additionally, the standard objective -- minimizing forward KL divergence -- requires the student to cover the *entire* support of the teacher's distribution. When the student has much lower model capacity than the teacher, this forces the student to assign probability mass to tokens the teacher considers low-probability, which can produce hallucinations and low-quality generations.

### The Imitation Learning Connection

GKD's key insight is recognizing that KD for auto-regressive models is fundamentally an **imitation learning** problem with an interactive expert (the teacher). Drawing from DAgger (Ross et al., 2011), the solution is to:

1. Collect sequences using the student's own policy (on-policy data).
2. Obtain expert labels (teacher probabilities) on those sequences.
3. Retrain the student on this augmented dataset.

---

## 2. The GKD Algorithm

### 2.1 Notation and Setup

- **Vocabulary** V of M tokens
- **Input** x, **output** y = (y_1, y_2, ..., y_n)
- **Teacher** p_T: a fixed, high-capacity model
- **Student** p_S^theta: a smaller model with learnable parameters theta
- **Dataset** (X, Y): input-output pairs (optionally available)
- y_{<n} = (y_1, ..., y_{n-1}): prefix of output sequence up to the n-th token
- Token-level policy: p(y_n | y_{<n}, x) -- next-token probability distribution

### 2.2 Token-Level Divergence

For any divergence D, the **token-level divergence** between teacher and student over a complete output sequence y given input x is defined as:

```
D(p_T || p_S^theta)(y|x) := (1/L_y) * sum_{n=1}^{L_y} D(p_T(. | y_{<n}, x) || p_S^theta(. | y_{<n}, x))
```

This averages the divergence at each token position, where at each step both models are conditioned on the same prefix y_{<n} (whether that prefix came from the student, the teacher, or the ground truth).

### 2.3 Existing Approaches (Baselines)

**Supervised Fine-Tuning (SFT):**
```
L_SFT(theta) = E_{(x,y)~(X,Y)} [ -log p_S^theta(y|x) ]
```
Minimizes negative log-likelihood on fixed dataset sequences.

**Supervised KD:**
```
L_SD(theta) = E_{(x,y)~(X,Y)} [ D_KL(p_T || p_S^theta)(y|x) ]
```
Trains student to match teacher's token-level distributions, but only on fixed dataset sequences. This still suffers from distribution mismatch.

**SeqKD** (Kim & Rush, 2016): SFT on teacher-generated outputs. Still uses fixed sequences.

### 2.4 On-Policy KD

The on-policy KD loss is:

```
L_OD(theta) = E_{x~X} [ E_{y~p_S(.|x)} [ D_KL(p_T || p_S^theta)(y|x) ] ]
```

Key properties:
- Output sequences y are **sampled from the student** p_S(.|x)
- Teacher provides token-level feedback on these student-generated sequences
- Gradients do **NOT** backpropagate through the student's sampling distribution p_S(.|x) -- this is critical for stability and computational efficiency (similar to REINFORCE)
- Temperature gamma = 1 is used during student sampling to encourage diversity
- Student sampling is cheaper than teacher sampling due to the model size difference

### 2.5 The Generalized KD Objective

GKD unifies supervised and on-policy approaches into a single framework with flexible divergence choice:

```
L_GKD(theta) = (1 - lambda) * E_{(x,y)~(X,Y)} [ D(p_T || p_S^theta)(y|x) ]
             + lambda * E_{x~X} [ E_{y~p_S(.|x)} [ D(p_T || p_S^theta)(y|x) ] ]
```

Where:
- **lambda in [0, 1]**: the **student data fraction** -- controls the mix of on-policy vs. fixed-data training
  - lambda = 0: Supervised KD (purely off-policy)
  - lambda = 0.5: Mixed (like ImitKD)
  - lambda = 1: Purely on-policy
- **D**: any divergence measure (not limited to forward KL)

Gradients are NOT backpropagated through the student's sampling distribution, keeping training stable and computationally efficient.

### 2.6 Algorithm 1: GKD (Step by Step)

```
Given:
  - Teacher model p_T
  - Student model p_S^theta
  - Dataset (X, Y) of (input, output) pairs
  - Hyperparameters: student data fraction lambda, divergence D, learning rate eta

For each training step k = 1, ..., K:
  1. Generate a random value u ~ Uniform(0, 1)
  2. If u <= lambda:
       # ON-POLICY: generate outputs from the student
       Sample inputs x from X
       Generate outputs y ~ p_S^theta(.|x)
       Form batch B = {(x_b, y_b)}_{b=1}^{B}
  3. Else:
       # OFF-POLICY: use fixed dataset
       Sample batch B = {(x_b, y_b)}_{b=1}^{B} from (X, Y)
  4. Compute divergence and update:
       theta <- theta - eta * (1/B) * sum_{(x,y) in B} grad_theta D(p_T || p_S^theta)(y|x)
```

**Critical implementation detail**: In step 2, when generating y ~ p_S^theta(.|x), the sampling is treated as a non-differentiable operation. The gradient in step 4 only flows through the divergence computation D(p_T || p_S^theta), not through the sampling process that produced y. This is analogous to how REINFORCE treats the policy rollout.

### 2.7 Prerequisite: Reasonable Student Initialization

GKD assumes access to a student that can already generate sequences of adequate quality -- sequences that the teacher can provide meaningful feedback on. In practice, the authors start from student models that have already undergone supervised fine-tuning. This is analogous to the two-stage RLHF pipeline: SFT first, then RL fine-tuning.

---

## 3. Divergence Functions: Forward KL, Reverse KL, and JSD

### 3.1 Forward KL (Mode-Covering)

```
D_KL(P || Q) = sum_c P(c) * log(P(c) / Q(c))
```

- Equivalent to minimizing the negative log-likelihood under the teacher distribution.
- **Mode-covering**: forces the student to assign mass wherever the teacher assigns mass.
- Can force the student to spread its limited capacity across the entire teacher support, including low-probability tokens.
- When the student lacks capacity, this can lead to hallucinations -- the student assigns mass to tokens the teacher considers unlikely.
- Works well with **greedy sampling** (temperature -> 0) at evaluation time.

### 3.2 Reverse KL (Mode-Seeking)

```
D_KL(Q || P) = sum_c Q(c) * log(Q(c) / P(c))
```

- **Mode-seeking**: concentrates on high-probability teacher modes.
- The student focuses on tokens where the teacher assigns high probability.
- Avoids low-quality/hallucinated generations at the cost of less diverse outputs.
- Better when the student has significantly less capacity than the teacher.
- Works especially well for **instruction tuning** where capturing core behaviors matters more than diversity.
- Natural synergy with RL (RLHF typically optimizes a form of reverse KL).

### 3.3 Generalized Jensen-Shannon Divergence (JSD(beta))

```
D_JSD(beta)(P || Q) = beta * D_KL(P || beta*P + (1-beta)*Q) + (1-beta) * D_KL(Q || beta*P + (1-beta)*Q)
```

Where beta in (0, 1) is an interpolation parameter:
- **lim_{beta->0}**: D_JSD(beta)(P||Q) / beta = D_KL(P||Q) -- behaves like **forward KL**
- **beta close to 1**: behaves like **reverse KL**
- JSD is **always bounded**, unlike KL divergence which can be unbounded for distributions with disjoint supports.

The paper evaluates JSD(0.1), JSD(0.5), and JSD(0.9).

### 3.4 Key Insight: Divergence Choice Is Task-Dependent

| Divergence  | Behavior        | Best for                                       |
|-------------|-----------------|------------------------------------------------|
| Forward KL  | Mode-covering   | Greedy sampling; tasks where coverage matters  |
| JSD(0.1)    | Near-forward KL | Similar to forward KL                          |
| JSD(0.5)    | Balanced        | General-purpose                                |
| JSD(0.9)    | Near-reverse KL | Temperature sampling; quality over diversity   |
| Reverse KL  | Mode-seeking    | Instruction tuning; capacity-limited students  |

As temperature sampling increases (gamma -> 1), mode-seeking divergences (reverse KL, JSD(0.9)) yield superior quality but less diversity. At lower temperatures, performance differences between divergences narrow.

---

## 4. How On-Policy Sampling Works

The on-policy mechanism is the core innovation. Here is the detailed flow:

### Step 1: Student Generates
Given input prompts x sampled from the dataset:
- The student model p_S^theta auto-regressively generates complete output sequences y.
- Generation uses temperature gamma = 1 (softmax temperature) to encourage exploration and diversity.
- This is a **non-differentiable** sampling process -- no gradients flow through it.

### Step 2: Teacher Scores
For each student-generated sequence y:
- The teacher model p_T performs a single forward pass to compute its token-level probability distributions p_T(. | y_{<n}, x) at every position.
- This is just a **prefill** operation -- the teacher does not generate anything, it only evaluates the student's tokens.
- This is computationally efficient: one forward pass per sequence.

### Step 3: Divergence Computation
At each token position n in the sequence:
- Compare p_T(. | y_{<n}, x) (teacher's next-token distribution) with p_S^theta(. | y_{<n}, x) (student's next-token distribution).
- Compute the chosen divergence D between these two distributions.
- Average across all token positions.

### Step 4: Gradient Update
- Compute gradients of the averaged divergence with respect to theta (student parameters).
- Update theta via gradient descent.
- Gradients only flow through the student's parameterized distribution p_S^theta, NOT through the sampling process that generated y.

### Why This Works

The student learns from its **own mistakes**. When the student generates a low-quality prefix, the teacher's distributions at subsequent positions reveal what the student should have done differently. This is precisely the feedback loop that eliminates the train-inference mismatch:

- If the student makes an error at step k, the remaining positions k+1, k+2, ... are all conditioned on this erroneous prefix.
- The teacher provides the correct distribution at every position, teaching the student to recover from errors it actually makes.
- As the student improves, it generates better sequences, shifting the training distribution to more challenging frontiers.

---

## 5. Integration with RLHF

### 5.1 The Combined Objective

GKD can be seamlessly combined with reinforcement learning from human feedback (RLHF) or AI feedback (RLAIF). The combined objective is:

```
E_{x~X} [ (1 - alpha) * E_{y~p_S^theta(.|x)} [r(y)]  -  alpha * E_{y~p_S(.|x)} [D(p_T || p_S^theta)(y|x)] ]
```

Where:
- **r(y)**: a scalar reward (e.g., human preference, factual consistency score)
- **alpha in [0, 1]**: controls the relative strength of distillation vs. reward maximization
  - alpha = 1: pure distillation (no reward optimization)
  - alpha = 0: pure RL (no distillation)
  - 0 < alpha < 1: joint optimization

### 5.2 Why This Combination Is Natural

- Both on-policy GKD and RL use **student-generated outputs** -- they share the sampling infrastructure.
- Adding GKD to RL requires only computing teacher log-probs on sequences already being generated for RL. Minimal additional overhead.
- Distillation can reduce the "alignment tax" -- the drop in general capabilities that often accompanies RLHF.
- For integrating with existing RL fine-tuning workflows, the authors recommend using **reverse KL** or **JSD(0.9)** as the divergence.

### 5.3 Experimental Validation (RLAIF + GKD on XSum)

Using textual entailment feedback as a reward (RLEF) combined with on-policy GKD with JSD(0.9):
- The combined approach improves both **ROUGE-2** (summarization quality) and **factual consistency** (entailment score).
- alpha controls the trade-off: lower alpha emphasizes reward (ROUGE-2 rises), higher alpha emphasizes distillation (factual consistency rises).
- On-policy GKD + RL outperforms both pure RL (RLEF*) and the teacher model alone on the quality-consistency frontier.

---

## 6. Key Experimental Results

### 6.1 Abstractive Summarization (XSum)

**Setup**: T5-XL (~3B params) as teacher, T5-small/base/large (77M/250M/800M) as students. ROUGE-2 metric.

**Results**:
- GKD achieves **2.1x improvement** over baseline KD methods (relative performance gain across model sizes).
- On-policy GKD with JSD(0.9) consistently outperforms supervised KD, SeqKD, ImitKD, and f-distill.
- **Remarkable data efficiency**: On-policy GKD trained on just 5% of the data (without ground-truth summaries) outperforms supervised KD and ImitKD trained on 100% of the data.
- Purely on-policy (lambda = 1) consistently beats mixed (lambda = 0.5) and supervised (lambda = 0) variants.

### 6.2 Machine Translation (WMT14 en-de)

**Setup**: T5-XL teacher (BLEU 28), T5-small/base students. BLEU metric with beam search.

**Results**:
- GKD achieves **1.7x improvement** over baselines.
- JSD divergences outperform pure forward/reverse KL.
- Purely on-policy data (lambda = 1) consistently best.
- Performance gap between divergences shrinks with larger students.

### 6.3 Arithmetic Reasoning (GSM8K with Chain-of-Thought)

**Setup**: FLAN T5-XL teacher (27.9% accuracy), T5-base student (10.16% baseline). Greedy sampling evaluation with calculator.

**Results**:
- On-policy GKD achieves **1.9x improvement** over baselines.
- **Forward KL** performs best here (greedy sampling evaluation).
- Performance improves consistently with more on-policy data (lambda >= 0.25).
- Purely student-generated CoTs (lambda = 1) outperform fixed CoT datasets or mixed approaches.
- On-policy GKD surpasses GPT-3 davinci-002 few-shot performance and approaches PaLM (540B) few-shot performance with a T5-base (250M) student.

### 6.4 Task-Agnostic Distillation (FLAN Instruction Tuning)

**Setup**: FLAN T5-XL teacher, FLAN T5-Base student. FLAN2021 dataset (5.36M examples, 62 tasks). Evaluated on held-out MMLU (57 tasks) and BBH (23 tasks).

**Results**:
- On-policy GKD with **reverse KL** yields **2% absolute MMLU** and **1% absolute BBH** accuracy improvements over baselines.
- Reverse KL substantially outperforms forward KL for instruction tuning.
- Hypothesis: reverse KL's mode-seeking nature ensures the model focuses on core behaviors specified by instructions, rather than spreading capacity across less relevant details.

### 6.5 Summary of Best Divergence per Task

| Task                     | Best Divergence | Evaluation Strategy |
|--------------------------|-----------------|---------------------|
| Summarization (XSum)     | JSD(0.9)        | Temperature sampling|
| Translation (WMT)        | JSD variants    | Beam search         |
| Reasoning (GSM8K)        | Forward KL      | Greedy sampling     |
| Instruction tuning (FLAN)| Reverse KL      | Few-shot prompting  |

### 6.6 Computational Overhead

- On-policy sampling adds **1.8-2.2x** computational cost during fine-tuning versus fixed-dataset approaches.
- The authors argue this is justified because training cost is minor relative to deployment/serving costs.
- When combined with RL (which already requires on-policy sampling), the additional overhead of computing teacher log-probs is minimal.

---

## 7. Design Insights and Practical Recommendations

### 7.1 On-Policy Data Is Consistently Better
Across all four tasks, using purely on-policy student-generated data (lambda = 1) matches or outperforms any mixture with fixed dataset sequences. This validates the core thesis: learning from your own mistakes is more effective than learning from others' demonstrations.

### 7.2 Divergence Selection Guidelines
- **Greedy sampling evaluation** -> Forward KL works well (it covers the teacher's mode that greedy decoding selects).
- **Temperature sampling evaluation** -> Mode-seeking divergences (reverse KL, JSD(0.9)) produce higher quality outputs.
- **Instruction tuning / behavior cloning** -> Reverse KL excels at capturing core behaviors.
- **When unsure** -> JSD(0.5) or JSD(0.9) provide a reasonable default.

### 7.3 Data Efficiency
GKD is remarkably data-efficient. On-policy GKD on 5% of labeled data outperforms supervised KD on 100%. This is because every training step uses fresh on-policy data that targets the student's current weaknesses, rather than repeatedly fitting to a fixed dataset.

### 7.4 Student Initialization Matters
The student should be pre-trained or fine-tuned enough to generate coherent sequences. Starting from a randomly initialized student would produce incoherent outputs that the teacher cannot meaningfully score.

### 7.5 No Gradient Through Sampling
Not backpropagating through the student's sampling distribution is critical for:
- **Stability**: Avoids high-variance gradient estimates that plague RL-style methods like MiniLLM.
- **Simplicity**: No need for stabilization tricks (reward hacking mitigation, length normalization, etc.).
- **Efficiency**: Sampling is treated as a simple forward pass with no gradient tape.

---

## 8. Comparison with Related Work

| Method   | On-Policy? | Divergence | Gradient Through Sampling? | RL Integration? |
|----------|-----------|------------|---------------------------|-----------------|
| Supervised KD | No  | Forward KL | N/A                      | No              |
| SeqKD    | No        | Forward KL | N/A                      | No              |
| ImitKD   | Mixed     | Forward KL | No                       | No              |
| f-distill| Mixed     | Total Var. | No                       | No              |
| MiniLLM  | Yes       | Reverse KL | **Yes** (policy gradient) | No              |
| **GKD**  | **Yes**   | **Any**    | **No**                   | **Yes**         |

GKD's advantages over MiniLLM:
- Simpler (no gradient through sampling => no need for stabilization tricks)
- More general (any divergence, not just reverse KL)
- Can use forward KL or JSD which sometimes outperform reverse KL
- Seamless RL integration

---

## 9. Relevance to Our ARC-AGI Use Case

### 9.1 Our Setup

We are training models (Qwen3-8B, Nemotron-Cascade-8B/14B, and others) to solve ARC-AGI visual pattern reasoning puzzles using reinforcement learning with verifiable rewards. Our current approach:

- **Environment**: Iterative code-writing -- the model writes a `transform` function, gets feedback on training examples, refines, and submits.
- **Reward**: Binary (exact grid match), partial (cell-level accuracy), or combined. These are **verifiable** -- no reward model needed.
- **Training**: GRPO via prime-rl with LoRA, on-policy rollouts, multi-turn interaction.
- **Teacher candidates**: Qwen3-32B, GPT-oss-120B, or other large models that score well on ARC-AGI.

### 9.2 Why GKD Is Directly Relevant

**The distribution mismatch problem is acute for ARC-AGI.** Our student generates multi-turn code-writing trajectories with up to 40 turns. Errors compound: a wrong pattern hypothesis at turn 1 leads to wasted turns of debugging an incorrect approach. Pure RL provides only a single sparse reward at the end of the entire multi-turn episode.

GKD-style on-policy distillation could provide **dense, per-token feedback** on the student's own multi-turn trajectories:

1. **Student** generates a multi-turn trajectory (prompt -> code -> feedback -> revised code -> ...).
2. **Teacher** (e.g., Qwen3-32B) evaluates the student's trajectory with a single forward pass, providing token-level distributions.
3. **Training** uses the KL divergence between teacher and student distributions at every token position.

This gives the student information about *where* it went wrong, not just *whether* the final grid matched.

### 9.3 Combining GKD with Verifiable Rewards

The GKD + RL objective (Section 5 of the paper) is directly applicable:

```
E_{x~X} [ (1 - alpha) * E_{y~p_S(.|x)} [r(y)]  -  alpha * E_{y~p_S(.|x)} [D(p_T || p_S)(y|x)] ]
```

For ARC-AGI:
- **r(y)**: the verifiable reward (binary/partial grid match) -- this is our existing signal.
- **D(p_T || p_S)(y|x)**: dense per-token distillation signal from a high-performing teacher.
- **alpha**: tune to balance "solve the puzzle" (reward) vs. "reason like the teacher" (distillation).

This combined objective could:
- Accelerate learning by providing dense feedback (O(N) bits per episode vs O(1) for pure RL).
- Prevent the student from developing degenerate strategies that hack the reward but don't generalize.
- Maintain general code-writing and reasoning capabilities (reduce "alignment tax").

### 9.4 Practical Considerations for Implementation

**prime-rl already supports this.** Per our prior research (`on-policy-distillation-tool-research.md`), prime-rl has merged on-policy distillation support. The configuration is straightforward:

```toml
[teacher_inference.model]
name = "Qwen/Qwen3-32B"

[trainer.loss]
teacher_tau = 1.0   # weight for teacher distillation
adv_tau = 1.0       # weight for RL advantage (verifiable reward)
```

**Divergence recommendation for ARC-AGI:**
- ARC-AGI evaluation is essentially binary (exact match) -- more like **greedy sampling** evaluation.
- The paper suggests **forward KL** works well for greedy/deterministic evaluation (GSM8K result).
- However, since our student has much less capacity than the teacher (8B vs 32B), **reverse KL** or **JSD(0.9)** might avoid forcing the student to cover reasoning strategies it cannot execute.
- **Recommendation**: Start with reverse KL (which prime-rl uses by default and which has natural RL synergy), then ablate with JSD(0.9) and forward KL.

**Student initialization requirement:**
- GKD requires the student to already generate adequate sequences. Our models are already instruction-tuned (Qwen3-8B-Instruct etc.) and can write code, so this prerequisite is met.

**Multi-turn complication:**
- GKD was designed for single-turn generation. Our ARC-AGI environment is multi-turn (up to 40 turns with environment feedback between turns).
- The teacher can still score each turn's generation, but we need to be careful about how environment feedback interleaves with the distillation signal.
- One approach: Apply GKD token-level distillation only to the model's generated tokens within each turn, treating environment feedback tokens as fixed context.

### 9.5 Expected Benefits

Based on the paper's findings:

| GKD Finding | Expected ARC-AGI Impact |
|-------------|------------------------|
| 2.1x improvement over baseline KD | Significant accuracy boost over pure RL on ARC-AGI |
| 5% data outperforms 100% supervised | Can train effectively with limited ARC puzzle data |
| Dense per-token feedback | Student learns *where* its code/reasoning went wrong, not just whether the grid matched |
| RL + GKD combination | Can jointly optimize for grid accuracy and reasoning quality |
| 1.8-2.2x compute overhead | Acceptable given our 4-GPU setup (3 inference + 1 training) |

### 9.6 Risks and Open Questions

1. **Multi-turn distillation**: The paper only studies single-turn generation. Multi-turn ARC-AGI trajectories with interleaved environment feedback are a novel application. The teacher may not produce meaningful distributions when conditioned on environment feedback it has never seen before in a student-specific error context.

2. **Teacher quality ceiling**: If no available teacher model reliably solves ARC-AGI puzzles, the distillation signal may be noisy or misleading. We need to verify the teacher's ARC-AGI performance first.

3. **Code generation specifics**: The paper studies summarization, translation, and math reasoning. Code generation for grid transformations has different error modes (syntax errors, logic errors, off-by-one in grid coordinates). It is unclear if per-token distillation captures these structural patterns effectively.

4. **Compute budget**: Running both a large teacher (32B) for log-prob computation and the student (8B) for training requires careful GPU memory management. Our current 4-GPU setup allocates 3 GPUs for inference and 1 for training -- the teacher model would need to share inference GPUs or use a separate deployment.

---

## 10. Key Takeaways

1. **On-policy data eliminates distribution mismatch**: Training on the student's own outputs with teacher feedback consistently outperforms training on fixed datasets, across all tasks studied.

2. **Divergence choice matters and is task-dependent**: There is no universally best divergence. Forward KL for greedy evaluation, reverse KL for instruction following, JSD for a balanced approach.

3. **Remarkable data efficiency**: On-policy GKD on 5% of data beats supervised KD on 100%.

4. **Seamless RL integration**: The combined GKD + RL objective is natural and effective, reducing alignment tax while improving task performance.

5. **Simplicity is a feature**: Not backpropagating through sampling keeps GKD stable and simple compared to policy-gradient approaches like MiniLLM.

6. **Dense feedback >> sparse rewards**: The Thinking Machines blog post (which builds on GKD) demonstrates 50-100x compute efficiency gains of on-policy distillation over pure RL, largely because distillation provides O(N) bits per episode vs. O(1) for RL.

---

## 11. Connection to Broader Research Context

GKD is the foundational paper for the on-policy distillation line of work. Subsequent papers and efforts that build on it include:

- **Qwen3 Technical Report** (2025): Used on-policy distillation to achieve 74.4% on AIME'24 at 10x lower cost than RL, citing GKD as inspiration.
- **Thinking Machines / Kevin Lu** (2025): Extended GKD with practical reverse-KL implementation, demonstrating 50-100x compute savings over RL for math reasoning and continual learning.
- **MiniLLM** (Gu et al., 2023): Concurrent work that also uses on-policy distillation but backpropagates through sampling (more complex, less stable).
- **Self-Distillation (SDFT, SDPO)**: Related approaches where the model distills from an earlier version of itself, useful for continual learning.
- **prime-rl**: Our training framework, which has implemented on-policy distillation with `teacher_tau` / `adv_tau` knobs for controlling the RL-distillation trade-off.

For our ARC-AGI project, GKD provides the theoretical foundation for combining dense teacher feedback with verifiable environment rewards -- exactly the hybrid signal we need for efficient training on this challenging reasoning task.
