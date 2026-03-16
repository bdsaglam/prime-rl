# Literature Connections: Methods for Learning to Reflect

## The Problem

In self-OPD, three components share the same weights but only one gets explicit gradient signal:

| Component | Trained? | Signal |
|---|---|---|
| Student (solving) | Yes | OPD loss on solution tokens |
| Teacher (using PI to score) | No | Only incidental via shared weights |
| Analyzer (generating PI) | No | Only incidental via shared weights |

We want ALL three to improve. The model should learn to solve, learn to score well given PI, and learn to generate better PI (reflect better).

## Key Methods from Our Literature

### 1. RLTF-FM (Song 2026) — Auxiliary Feedback Prediction Loss

**Paper**: `papers/rltf-song-2026.md`

**Mechanism**: Train the model to *predict* feedback/critique as an auxiliary objective alongside the primary RL loss. The model generates both solutions AND critiques.

**Key theoretical result**: Feedback modeling acts as a "representation preconditioner" — it provides learning signal in representation directions that reward-only RL fails to identify from sparse signal. This is exactly the kind of signal enrichment we're after.

**How it maps to us**: Add a second loss term where the model, given a problem + its failed rollout, must predict a structured analysis. The target analysis comes from a stronger model or from best-of-N selection. This explicitly trains the reflection skill.

```
Total Loss = OPD_loss(solution_tokens) + λ * FM_loss(analysis_tokens)
```

**Critical insight**: RLTF-FM uniquely enables test-time self-critique because it explicitly trains the model to produce accurate feedback. Standard RL/OPD never trains this ability.

### 2. RL²F / Meta-Learning (Klissarov 2026) — Learning In-Context Plasticity

**Paper**: `papers/meta-learning-klissarov-2026.md`

**Mechanism**: Frame multi-turn interaction as RL² meta-learning. Inner loop: model learns in-context from feedback within a conversation. Outer loop (gradients): optimize the model to be better at this in-context learning.

**Key finding**: "The ability to learn from feedback is itself a learnable skill." After RL²F training:
- Models develop "in-context plasticity" — they actively reason about feedback
- Base model ignores feedback, repeats mistakes; trained model uses thinking traces to derive new approaches
- **The autodidact agent (self-critiques at inference) outperforms the version with a privileged external teacher**

**How it maps to us**: The self-OPD setup IS a meta-learning setup. The "inner loop" is the model using PI to score better. The "outer loop" is weight updates from OPD loss. If we structure training so the model learns to extract more from PI over time, we get RL²F's benefits.

**Practical path**: Train on full sequences including analysis + solution. The model learns that good analysis → good solution → high reward. Over time, it develops better "in-context plasticity" for using its own reflections.

### 3. pi-Distill (Penaloza 2026) — Joint Teacher-Student Optimization

**Paper**: `papers/pi-distill-penaloza-2026.md` (referenced in `research-notes/opd-concepts.md`)

**Mechanism**: Joint optimization of both teacher and student:
```
J = α * J_Teacher + (1 - α) * J_Student
```
- α=1: Only teacher learns (student improves through shared params)
- α=0: Only student learns from teacher
- α=0.5: Both optimized simultaneously — most robust (+11.8% on Travel Planner)

**How it maps to us**: In self-OPD, teacher and student are the same model. pi-Distill shows that explicitly training both roles simultaneously is better than training just one and hoping the other improves incidentally. We could add a "teacher loss" that optimizes the model's scoring quality given PI.

### 4. Reflective Retry (from `open-questions.md`)

**Our own earlier idea**: Student fails → reveal answer → student reflects and retries → corrected trajectory becomes PI.

**Key insight**: "The trajectory starts from the student's own reasoning. It shares the same initial exploration, the same hypotheses, the same code style. The divergence point is exactly the decision that matters."

**Connection to education research**: "test-enhanced learning" (Roediger & Karpicke 2006) — students who attempt retrieval and get corrective feedback learn better than those who study passively.

### 5. Klissarov's World-Modeling Objective

**From meta-learning paper**: Train on the FULL interactive dialogue including the teacher's critiques. But crucially, train the model to predict teacher's critiques "as if unconditioned on privileged information" — requiring the model to INFER the critique logic from context alone.

**This is exactly "learning to reflect blind"**: The model learns to produce informed-quality critiques without actually having the answer. This could close our blind→informed gap (d=1.48 → d=2.23 for structured analysis).

## Synthesis: The Missing Method

No existing paper combines all three pieces:
1. **Auxiliary feedback prediction** (RLTF-FM) — explicitly train reflection
2. **Meta-learning for in-context plasticity** (RL²F) — learn to USE reflection better
3. **Self-OPD with structured analysis** (ours) — per-token gradient signal from reflection

The combination would be:
- Self-OPD on solution tokens (learn to solve)
- Feedback modeling loss on analysis tokens (learn to reflect)
- Meta-learning structure so both improve together (learn to learn)
- Structured format constrains analysis to be precise (d=2.23)

## Practical Implementation Path

**Phase 1**: Reflection-in-sequence
- Modify prompt: "Here's your previous attempt. Analyze what went wrong, then solve correctly."
- Model generates [analysis][solution] as one sequence
- Standard OPD loss on the full sequence — gradients flow through analysis tokens naturally
- Teacher (same model + answer PI) scores both analysis quality and solution quality

**Phase 2**: Add RLTF-FM auxiliary loss
- Separate training objective: predict structured analysis given (problem, failed rollout)
- Target analyses from stronger model or best-of-N
- This explicitly trains the reflection skill, not just incidentally

**Phase 3**: World-modeling (Klissarov-style)
- Train to predict informed analysis from blind context
- Model learns to infer what the right critique would be without seeing the answer
- Closes the blind→informed performance gap

## External Methods (from literature search)

### 6. CTRL — Teaching Language Models to Critique via RL (2025)

**Paper**: [arxiv 2502.03492](https://arxiv.org/abs/2502.03492)

**Mechanism**: Two-stage: (1) synthesize critiques by reasoning about execution feedback, (2) refine the critic via RL where reward = whether the critique enables a fixed generator to correct its output. Trains critique quality end-to-end.

**How it maps to us**: This is the closest existing method to what we need. Train the analyzer so its analyses maximize the teacher-student signal (or downstream student improvement). The reward is: "did this analysis actually help?"

### 7. Reflect, Retry, Reward (2025)

**Paper**: [arxiv 2505.24726](https://arxiv.org/abs/2505.24726)

**Mechanism**: Trains a model to generate better self-reflections using RL, requiring only binary success/failure signal from a verifier. The reflection itself is the learned artifact.

**How it maps to us**: Directly optimizes reflection quality. Our binary signal could be: did the student improve on this problem after using the analysis-based PI?

### 8. ReflectEvo — Learning Self-Reflection (2025)

**Paper**: [arxiv 2505.16475](https://arxiv.org/abs/2505.16475), ACL 2025

**Mechanism**: Learns high-quality reflection using only binary correct/incorrect feedback. Shows flawed reflection leads to repeated errors, while high-quality reflection improves Acc@t1 by 13%+.

**Key evidence**: Reflection quality directly determines downstream improvement. Supports our finding that best-of-4 selection (quality proxy) boosts signal by +31-33%.

### 9. Quiet-STaR — Learning to Think Before Speaking (Zelikman 2024)

**Paper**: [arxiv 2403.09629](https://arxiv.org/abs/2403.09629)

**Mechanism**: Model generates internal "thoughts" before each token, trained via REINFORCE. Uses a **mixing head** (shallow MLP) that interpolates between with-thought and without-thought logits: `log p = w * log p_base + (1-w) * log p_thought`. Since thought tokens are discrete, can't backprop → uses REINFORCE where reward = improvement in future token prediction.

**How it maps to us**: Our analysis is a "thought" that happens before teacher scoring. Quiet-STaR proves you can train discrete thought generation to improve downstream predictions. The mixing head idea is particularly relevant — we could interpolate between with-analysis and without-analysis teacher logprobs, and train the analysis to maximize the improvement.

### 10. ML³ — Meta-Learning via Learned Loss (Bechtle et al., ICML)

**Paper**: [arxiv 1906.05374](https://arxiv.org/abs/1906.05374)

**Mechanism**: Encodes learning strategies into a parametric loss function. The meta-loss is trained to provide strong learning signal across tasks. After training, the task-specific losses are no longer needed — the meta-loss alone drives optimization. 5x more sample efficient.

**How it maps to us**: The analysis-as-PI is essentially a learned component of the loss computation pipeline. ML³ shows this is viable — you CAN meta-learn the thing that generates the training signal, and it generalizes.

### 11. SCRIT — Self-Evolving Critique (2025)

**Paper**: [arxiv 2501.05707](https://arxiv.org/abs/2501.05707)

**Mechanism**: LLM self-validates generated critiques by checking whether proposed corrections lead to valid solutions. Critiques producing consistent corrections are kept as training data.

**How it maps to us**: Self-validation for filtering analysis quality without external labels. Could use: generate analysis → use as PI → if student improves, keep analysis as positive training example.

### 12. Solver-Verifier Gap Theory (2025)

**Paper**: [arxiv 2507.00075](https://arxiv.org/abs/2507.00075)

**Key insight**: Self-improvement is possible when verification is easier than generation. Our deliberative PI widens the solver-verifier gap by making the teacher/verifier stronger (conditioned on analysis), creating more learning signal. This provides theoretical grounding for why the approach works.

### Other Connections

- **Hindsight Experience Replay** — relabeling failures. Conceptually related but uses fixed relabeling strategy; doesn't learn to relabel better.
- **Expert Iteration** — generate→evaluate→distill. Applied to reflection: generate analyses → evaluate → distill good patterns.
- **Differentiable Communication (DIAL, CommNet)** — Learning to send useful messages between agents. Analysis is a "message" from analyzer to teacher. DIAL uses backprop through continuous messages during training, discrete at execution.
- **Amortized Inference** — Internalizing iterative optimization into a forward pass. Our analysis could be seen as amortized introspection — the model internalizes reflection patterns so deeply it doesn't need explicit analysis at deployment.

## Revised Synthesis

The missing method that nobody has built:

**Self-OPD with learned reflection**, combining:
1. Standard OPD loss on solution tokens (from existing literature)
2. RLTF-FM auxiliary loss on analysis tokens (Song 2026) — explicitly trains reflection
3. CTRL-style end-to-end optimization (2025) — reward = downstream learning improvement
4. Quiet-STaR's mixing mechanism (2024) — interpolate with/without analysis
5. Structured analysis format (our finding) — constrains output for precision

The closest existing systems are CTRL (trains critique quality) and Reflect,Retry,Reward (trains reflection quality), but neither operates within the self-distillation framework, and neither uses the per-token logprob signal that OPD provides. Our unique position: we can measure the quality of ANY analysis via |KL| and Cohen's d in a single forward pass, without actually training the student. This makes the reward signal for the analyzer cheap and informative.
