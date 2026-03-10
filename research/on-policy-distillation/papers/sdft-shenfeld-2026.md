# Self-Distillation Fine-Tuning (SDFT): Research Notes

**Paper:** "Self-Distillation Enables Continual Learning"
**Authors:** Idan Shenfeld (MIT/Improbable AI Lab), Mehul Damani (MIT/Improbable AI Lab), Jonas Hubotter (ETH Zurich), Pulkit Agrawal (MIT/Improbable AI Lab)
**arXiv:** [2601.19897](https://arxiv.org/abs/2601.19897) (January 2026)
**Code:** [github.com/idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation)
**Project page:** [self-distillation.github.io/SDFT.html](https://self-distillation.github.io/SDFT.html)

---

## 1. Core Idea

SDFT is a method for **on-policy learning from demonstrations** that does not require an explicit reward function or a separate teacher model. The key insight is: a foundation model can serve as **both student and teacher** by exploiting the gap between its unconditional behavior and its behavior when conditioned on expert demonstrations via in-context learning (ICL).

- **Student:** The model given only the task input: `pi_theta(.|x)`
- **Teacher:** The *same* model conditioned on the task input *plus* an expert demonstration: `pi(.|x, c)`

The demonstration `c` acts as **privileged information** available only at training time. The teacher's distribution is close to the base model (preserving general capabilities) while producing high-quality task-specific outputs. Training distills this demonstration-conditioned behavior into the student's unconditional behavior via on-policy reverse KL minimization.

This is fundamentally different from standard on-policy distillation (e.g., Thinking Machines / Tinker approach), where a separate, larger teacher model grades student rollouts. Here, the model is its own teacher---the supervision signal comes from how much observing a demonstration shifts the model's own predictions.

---

## 2. Method Details

### 2.1 Loss Function

The objective minimizes the reverse KL divergence between student and teacher:

```
L(theta) = D_KL(pi_theta(.|x) || pi(.|x, c))
         = E_{y ~ pi_theta(.|x)} [log(pi_theta(y|x) / pi(y|x, c))]
```

This is decomposed into a token-level loss using the autoregressive structure. The gradient estimator used in practice is the **full analytic per-token KL estimator**, which marginalizes over the vocabulary at each timestep:

```
g_analytic = sum_t sum_{v in V} log(pi_theta(v|y_{<t}, x) / pi(v|y_{<t}, x, c)) * grad_theta log pi_theta(v|y_{<t}, x)
```

This estimator has strictly lower variance than the token-level (partial) estimator, though it is biased at the sequence level. In ablations, it consistently outperformed both the simpler token-level estimator (higher variance) and the more expensive Rao-Blackwellized estimator (no measurable gains).

### 2.2 Connection to Inverse RL

The authors formally show SDFT is equivalent to maximizing an implicit reward function. Starting from the trust-region RL formulation:

```
pi_{k+1} = argmax_pi E_{y ~ pi}[r(y, x)] - beta * D_KL(pi(.|x) || pi_k(.|x))
```

The optimal policy takes the form: `pi*_{k+1}(y|x) ~ pi_k(y|x) * exp(r(y,x) / beta)`.

The **In-Context Assumption** substitutes the unknown optimal policy with the demonstration-conditioned model:

```
pi*_{k+1}(y|x) ~ pi(y|x, c)
```

This yields the implicit reward: `r(y, x, c) = log pi(y|x, c) - log pi_k(y|x)`, which is the log-likelihood ratio between teacher and student. At the token level:

```
r_t(y_t | y_{<t}, x, c) = log pi(y_t | y_{<t}, x, c) / pi_k(y_t | y_{<t}, x)
```

The policy gradient under this reward is equivalent in expectation to the gradient of the reverse KL. Thus SDFT can be viewed as an on-policy RL algorithm that maximizes rewards inferred by comparing the student's current behavior to its own "wiser," demonstration-aware version.

### 2.3 Teacher Prompt Template

The teacher is constructed using a simple prompt format:

```
<Question>
This is an example for a response to the question:
<Demonstration>
Now answer with a response of your own, including the thinking process:
```

This prompt prevents the model from copying the demonstration verbatim, instead eliciting a response that reflects the model's *understanding* of the intent behind the demonstration. The model uses ICL to reconstruct the correct reasoning process rather than merely parroting.

### 2.4 Teacher Model: EMA

A critical design choice is how the teacher model parameters evolve during training. The paper ablates three options:

| Teacher Type | Behavior |
|---|---|
| **Frozen base model** | Stable but underperforms---fails to track student improvements |
| **Current student model** | Unstable---small fluctuations amplified by on-policy feedback loop |
| **EMA of student (chosen)** | Best of both: tracks progress while smoothing variance |

The EMA update rule: `phi <- alpha * theta + (1 - alpha) * phi`, with alpha swept over {0.01, 0.02, 0.05}.

### 2.5 ICL Assumption: Two Requirements

The quality of the teacher signal depends on two empirically verified conditions:

1. **Optimality:** The demonstration-conditioned model should achieve near-maximal task performance.
   - Validation: On ToolAlpaca, base Qwen-2.5-7B achieves 42% accuracy. With demonstrations in context: 100%.
   - Manually inspected 50 teacher reasoning traces---not only correct final answers but valid chain-of-thought.

2. **Minimal Deviation:** The teacher distribution should remain close to the base model in KL terms.
   - SFT model deviates from base by 1.26 nats, while teacher deviates by only 0.68 nats (nearly half).
   - This is crucial: staying close to the base policy means the trust-region constraint is satisfied and general capabilities are preserved.

---

## 3. Full Algorithm (Pseudocode)

```
Require: Dataset D = {(x_i, c_i)}, model pi_theta,
         student context Ctx_S(x), teacher context Ctx_T(x, c),
         batch size B, max generation length T, LR eta, EMA rate alpha

1: Set teacher weights phi = theta
2: for each training step do
3:     Sample minibatch B = {(x_i, c_i)} ~ D
4:     for all (x_i, c_i) in B in parallel do
5:         # Student rollout (on-policy):
6:         s_i <- Ctx_S(x_i)
7:         Sample y_i = (y_{i,1:T}) ~ P_sample(. | s_i)
8:         # Compute teacher and student token logprobs:
9:         t_i <- Ctx_T(x_i, c_i)
10:        l^S_{i,t} <- log pi_theta(y_{i,t} | y_{i,<t}, s_i)   [student logprobs]
11:        l^T_{i,t} <- log pi_phi(y_{i,t} | y_{i,<t}, t_i)     [teacher logprobs]
12:    end for
13:    # Gradient computation:
14:    g <- (1/B) sum_{i=1}^{B} g_analytic({l^S_i, l^T_i})
15:    # Optional: importance sampling correction for vLLM
16:    theta <- theta - eta * g
17:    phi <- alpha * theta + (1 - alpha) * phi   [EMA update]
18: end for
```

Key implementation detail: only **1 on-policy rollout per prompt** is used. Multiple trajectories per prompt produced negligible improvements while substantially increasing compute.

---

## 4. Experimental Results

All experiments use **Qwen2.5-7B-Instruct** as the base model unless stated otherwise. Training on a **single NVIDIA H200 GPU**. Built on the **HuggingFace TRL library**.

### 4.1 Skill Learning (New Task Accuracy vs. Prior Capability Retention)

Three domains tested: Science Q&A (SciKnowEval Chemistry L-3), Tool Use (ToolAlpaca), Medical (HuatuoGPT-o1).

**Figure 4 (Pareto frontiers):** SDFT consistently achieves the best trade-off between new-task accuracy and prior-capability retention across all three domains. Each point is a trained model; top-right is ideal (high on both axes).

| Method | New Task Performance | Prior Capability Retention |
|---|---|---|
| SFT | Moderate | Severe degradation (10-15 pts on benchmarks) |
| DFT (offline on-policy) | Better than SFT | Moderate degradation |
| SFT + Re-invoke | Better than SFT | Partial recovery |
| **SDFT** | **Highest** | **Minimal degradation** |

Prior capabilities measured across: HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval.

### 4.2 Knowledge Acquisition

Learning about 2025 natural disasters (post-training-cutoff Wikipedia articles):

| Method | Strict Accuracy | Lenient Accuracy | OOD Accuracy |
|---|---|---|---|
| Base | 0 | 0 | 0 |
| Oracle RAG | 91 | 100 | 100 |
| CPT (continual pretraining) | 9 | 37 | 7 |
| SFT | 80 | 95 | 80 |
| **SDFT** | **89** | **100** | **98** |

The OOD accuracy gap is especially significant: SFT achieves 80% but SDFT achieves 98%, nearly matching oracle RAG. This indicates genuine knowledge integration rather than narrow memorization.

### 4.3 Continual Learning (Sequential Multi-Task)

Training sequentially on Tool Use -> Science Q&A -> Medical:

- **SDFT (Figure 3a):** Stable accumulation of skills. Performance on each task remains high as new tasks are learned. All three curves rise and stay up.
- **SFT (Figure 3b):** Severe oscillatory behavior. When training shifts to a new task, performance on previous tasks rapidly degrades. Classic catastrophic forgetting.

### 4.4 Training Reasoning Models Without Reasoning Data

Fine-tuning **Olmo-3-7B-Think** on medical tasks with **answer-only supervision** (no chain-of-thought traces):

| Model | Accuracy | Avg. # of Tokens |
|---|---|---|
| Olmo-3-7B-Think (base) | 31.2 | 4612 |
| + SFT | 23.5 | 3273 |
| + **SDFT** | **43.7** | **4180** |

SFT *degrades* performance and collapses reasoning depth (token count drops significantly). SDFT preserves reasoning behavior because the demonstration-conditioned teacher maintains the model's internal reasoning style---it produces a reasoning-consistent target distribution even when the external data contains only final answers.

### 4.5 On-Policy Learning is Essential (Ablation)

Comparing three ways of using the same demonstration-conditioned teacher:

| Method | Accuracy (after 2000 generations) |
|---|---|
| SFT from teacher samples | ~42% |
| Offline distillation (KL on teacher samples) | ~52% |
| **SDFT (on-policy distillation)** | **~67%** |

Neither offline approach matches on-policy SDFT, confirming that the benefits come specifically from on-policy learning, not merely teacher quality.

### 4.6 Teacher Context Ablation

For knowledge acquisition, conditioning the teacher on different information:

| Context | Strict Accuracy |
|---|---|
| Only answers | 37% |
| Only text (article) | 75% |
| **Text + answers** | **89%** |

Full demonstration context (both source text and worked answer) is critical for effective knowledge transfer.

---

## 5. Scaling Behavior

Performance gap between SDFT and SFT on Science Q&A scales monotonically with model size:

| Model Size | SDFT - SFT Gap | Notes |
|---|---|---|
| **3B** | **Negative** (~-3.3) | ICL too weak to provide meaningful teacher signal. SDFT underperforms SFT. |
| **7B** | **+4.0** | Sweet spot for the method on current hardware |
| **14B** | **+6.9** | Advantage continues to grow |

The method fundamentally depends on the model's **in-context learning ability**. Smaller models (3B) have weak ICL and thus the demonstration-conditioned version cannot serve as a useful teacher. The authors project that even larger models with stronger ICL will benefit more.

**Critical implication for our work:** If using Qwen2.5, we need at least 7B. The 3B variant is insufficient for SDFT.

---

## 6. Hyperparameters

### Skill Learning Experiments

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| Learning rate | {5e-6, 1e-5, 5e-5} |
| Optimizer | AdamW |
| LR scheduler | Cosine with warmup |
| Warmup steps | 10 |
| Epochs | {1, 2} |
| Batch size | {16, 32, 64} |
| Max grad norm | 1 |
| bfloat16 | True |
| Weight decay | 0 |
| EMA alpha | {0.01, 0.02, 0.05} |
| Max generation length | 2048 |

### Knowledge Acquisition Experiments

Same as above except:
- Epochs: {1, 2, 4}
- Max generation length: 1024

---

## 7. Computational Cost

- SDFT requires **~2.5x FLOPs** and **~4x wall-clock time** compared to SFT.
- The overhead comes from on-policy generation during training.
- However, multi-stage baselines (e.g., SFT + Re-invoke for capability restoration) may actually cost more total compute while achieving worse results.
- Only **1 on-policy generation per prompt** is needed (single trajectory).

---

## 8. Implementation Details (GitHub Repo)

Repository: [github.com/idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation)

### Key Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: argument parsing, model loading, dataset prep, trainer init |
| `distil_trainer.py` | Core training loop: loss computation, generation, EMA updates |
| `distil_config.py` | Configuration dataclass extending TRL's TrainingArguments |

### Architecture

- Built on **HuggingFace TRL** (v0.24.0) library
- Uses **vLLM** (v0.12.0) for efficient on-policy generation
- **DeepSpeed** (v0.18.4) for training
- Single H200 GPU for all experiments

### Key Config Parameters

```python
# From main.py / distil_config.py
model_name = "Qwen/Qwen2.5-7B-Instruct"
learning_rate = 2e-5
num_train_epochs = 1
num_prompts_per_batch = 32
ref_model_mixup_alpha = 0.01  # EMA rate

# From distil_config.py defaults
max_prompt_length = 512
num_generations = 8  # per prompt (though paper uses 1 for main experiments)
max_completion_length = 256
temperature = 1.0
```

### Loss Computation

The `_compute_loss` method in `distil_trainer.py` supports:
- Forward KL (alpha=0)
- Reverse KL (alpha=1) -- used in the paper
- Mixtures

Loss is masked to exclude padding tokens. An optional `num_loss_tokens_to_skip` parameter masks initial tokens to avoid learned linguistic artifacts from the teacher.

### EMA Update

```python
# MemoryEfficientSyncRefModelCallback
# Parameter-by-parameter sync for DeepSpeed ZeRO-3 compatibility
ref_param.data.mul_(1.0 - alpha).add_(model_param.data, alpha=alpha)
```

### Importance Sampling

When vLLM is used for generation (separate inference engine from training model), importance sampling correction compensates for distribution mismatch:

```python
importance_sampling_ratio = torch.clamp(
    torch.exp(old_per_token_logps - sampling_per_token_logps),
    max=self.vllm_importance_sampling_cap
)
```

### Data Format

The `load_tooluse_dataset()` function creates two prompt formats per example:
- **Student prompt:** Task input only (query)
- **Teacher prompt:** Task input + golden response demonstration

### Dependencies

```
datasets==4.3.0, torch==2.9.0, transformers==4.57.1, accelerate==1.11.0,
peft==0.17.1, vllm==0.12.0, trl==0.24.0, deepspeed==0.18.4,
flashinfer-python==0.5.3, wandb==0.22.2
```

---

## 9. Known Limitations

1. **Linguistic artifacts:** The student may inherit spurious phrases from the teacher (e.g., "Based on the text..."). Mitigated by masking loss over initial tokens.

2. **Model size requirement:** 3B is too small. The method requires sufficient ICL capability (7B+ for Qwen2.5 family).

3. **Cannot induce fundamental behavioral shifts:** SDFT excels at gradual capability acquisition but struggles when the desired behavior requires fundamentally different generation patterns (e.g., converting a non-reasoning model to explicit chain-of-thought).

4. **Depends on demonstration quality:** The privileged information (demonstrations) must actually improve the model's behavior when provided in context.

---

## 10. Relevance to Our ARC-AGI Setting

### Direct Applicability

Our ARC-AGI environment has a natural structure that maps onto SDFT:

1. **Privileged information is available:** For each ARC task, we have ground-truth test outputs (the `solutions` in `data.py`). These could serve as the demonstration `c` in SDFT. At training time, we can condition the model on the correct test output grid to create the teacher signal.

2. **The ARC task format already includes demonstrations:** Each ARC task provides training input-output pairs (typically 2-5 examples) that demonstrate the transformation rule. The test input is given, and the model must produce the test output. The ground-truth test output is the privileged information.

3. **Teacher prompt construction for ARC:** We could construct the teacher prompt as:
   ```
   [Standard ARC task prompt with training examples + test input]
   Here is the correct output for the challenge:
   [Ground-truth test output grid]
   Now write Python code to solve this transformation:
   ```
   The teacher version of the model sees the answer and can reason backward from it about the transformation rule, while the student must discover the rule independently.

### Potential Advantages for ARC

- **Preserving general coding ability:** Our ARC models use a REPL environment where the model writes Python code. SFT on ARC-specific demonstrations risks degrading general coding capability. SDFT's on-policy nature should preserve these broader skills.

- **Continual learning across ARC task distributions:** If we train sequentially on different ARC puzzle distributions (e.g., from arc-prize-2024 and arc-prize-2025), SDFT could accumulate pattern recognition skills without forgetting.

- **No explicit reward engineering needed:** Currently our `rewards.py` uses exact_match, cell_accuracy, shape_match, and format rewards. SDFT bypasses this entirely---the supervision signal comes from how the demonstration shifts the model's predictions.

### Considerations and Challenges

1. **ICL quality for ARC:** The critical question is whether showing the model the correct ARC test output in context actually enables it to produce a high-quality reasoning trace / code solution. ARC tasks require understanding abstract spatial transformations---simply seeing the answer may not help the model reason about *how* to get there, especially for complex multi-step transformations. This needs empirical validation (analogous to the authors' ToolAlpaca validation of the ICL assumption).

2. **Model size constraints:** The paper shows 3B is too small, 7B+ works. Our scratchpad shows we work with Qwen2.5-7B-Instruct, Qwen3-8B, and larger models (14B, 32B). The 7B models should be sufficient based on the paper's findings, but ARC-specific ICL quality needs testing.

3. **Multi-turn environment:** Our ARC environment is multi-turn (up to 10 turns with REPL interaction). SDFT as described operates on single-turn generation. Adapting it to multi-turn would require either:
   - Treating each turn independently (losing trajectory-level coherence)
   - Extending the method to score full multi-turn trajectories (more complex)
   - Using a simplified single-turn variant of the ARC environment

4. **Complementarity with existing RL setup:** The paper explicitly states SDFT is not a replacement for RL but complementary. Our existing GRPO/DR-GRPO training (visible in `scratchpad.md`) could be combined with SDFT. SDFT could serve as initialization (learning from demonstrations) before RL fine-tuning (optimizing the reward function).

5. **Integration with prime-rl:** Our existing infrastructure uses prime-rl for on-policy distillation with a separate teacher model. SDFT's self-distillation approach could be simpler to set up since it does not require a separate, larger teacher---just the same model with demonstrations in context. However, the TRL-based SDFT implementation is separate from prime-rl's infrastructure.

### Proposed Experiment Design

A concrete first experiment could be:

1. Take Qwen2.5-7B-Instruct as base model.
2. For each ARC training task, construct:
   - **Student prompt:** Standard ARC format (training examples + test input) + system prompt for code generation
   - **Teacher prompt:** Same, but appended with the correct test output grid as demonstration
3. Run SDFT with the paper's default hyperparameters (lr=2e-5, EMA alpha=0.01, batch_size=32).
4. Evaluate on held-out ARC tasks measuring exact_match and cell_accuracy.
5. Compare against SFT baseline (training on expert code solutions) and our current GRPO approach.

---

## 11. Relationship to Other On-Policy Distillation Work

| Method | Teacher | Student | On-Policy? | Reward? |
|---|---|---|---|---|
| Standard RL (GRPO) | N/A | Same model | Yes | Explicit (scalar) |
| SFT / Off-policy distillation | Separate large model | Student | No | Dense (teacher logprobs) |
| On-policy distillation (Tinker/Qwen3) | Separate large model | Student | Yes | Dense (KL) |
| **SDFT (this paper)** | **Same model + demo in context** | **Same model (no demo)** | **Yes** | **Dense (self-KL)** |
| DFT (Wu et al. 2025b) | Same or separate | Student | Approx on-policy | Dense |
| Re-invoke (Lu & Lab 2025) | Base model | Fine-tuned model | On-policy (post-hoc) | Dense (KL) |

SDFT's unique contribution is eliminating the need for a separate teacher while still achieving on-policy learning with dense supervision. The supervision comes from the model's own ICL capabilities, which are "free" given a sufficiently capable base model (7B+).

The connection to Re-invoke (from Thinking Machines / Kevin Lu) is particularly interesting: Re-invoke performs on-policy distillation *after* SFT to restore degraded capabilities. SDFT does this *during* training by design, avoiding the two-stage process entirely.

---

## 12. Summary

SDFT is a clean, practical method for learning from demonstrations without catastrophic forgetting. Its core innovation---using the model's own ICL as the teacher signal---eliminates the need for a separate teacher model while retaining the benefits of on-policy learning with dense per-token supervision.

For our ARC-AGI work, the most promising application is using ground-truth test outputs as privileged in-context demonstrations. The method's ability to preserve general capabilities while learning new skills is directly relevant to training ARC solvers that maintain coding ability. The main open question is whether seeing a correct ARC output in context gives the model enough of a signal to reconstruct the transformation logic---this is the ICL assumption that must be validated empirically for our specific domain.
