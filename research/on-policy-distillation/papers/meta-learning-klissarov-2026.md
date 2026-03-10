# Meta-Learning from Language Feedback: Research Notes

**Papers reviewed:**
1. **"Improving Interactive In-Context Learning from Natural Language Feedback"** (arxiv:2602.16066)
   - Klissarov\*, Cook\*, Antognini\*, Sun, Li, Jaques, Musat, Grefenstette -- Google DeepMind, Feb 2026
2. **"Learning to Learn from Language Feedback with Social Meta-Learning"** (arxiv:2602.16488)
   - Cook, Antognini, Klissarov, Musat, Grefenstette -- Google DeepMind, Feb 2026

Both papers are closely related (overlapping author teams, same core idea). Paper 1 (2602.16066) is the flagship technical paper introducing RL^2F on thinking models (Gemini 2.5 Flash/Pro). Paper 2 (2602.16488) formalizes the same idea under the name "Social Meta-Learning" (SML) and runs more thorough ablations on open-weight models (Gemma-3-12B-IT, Qwen3-8B), including offline vs. online training, Q-priming, and behavioural analysis. This document synthesizes both.

---

## 1. Core Idea: Self-Play with Information Asymmetry

The central insight is deceptively simple: **you can turn any single-turn verifiable task into a multi-turn didactic interaction by introducing information asymmetry between a "teacher" and a "student" -- both instantiated by the same model.**

- **Student**: Receives the problem statement. Attempts to solve it. Has no access to the ground-truth solution or privileged verification information.
- **Teacher**: Receives the same problem statement PLUS privileged information (e.g., the ground-truth solution for math, or unit test outputs for code). Provides natural language feedback to guide the student without revealing the answer directly.

Key design properties:
- The teacher does NOT need to be a larger/stronger model. **Information asymmetry, not model capability asymmetry, drives the feedback quality.** The same base model can play both roles. This is verified: the teacher reveals the solution in only ~0.3% of cases.
- This is a **cooperative game**, not adversarial. Both agents benefit when the student succeeds.
- The setup is a **scalable data generation method**: any existing single-turn verifiable dataset (math, code, logic puzzles) can be automatically converted into multi-turn interactive training data at negligible cost.

---

## 2. Formal Framework: POMDP and RL^2 Connection

### 2.1 POMDP Formulation

The interactive learning process is modeled as a Partially Observable Markov Decision Process (POMDP):

```
<S, A, O, T, R, gamma>
```

- **State** `s_t = (k_t, o_t)`: Public conversation history `o_t` + teacher's private knowledge `k_t`
- **Student observation**: Only `o_t` (conversation history) -- cannot see `k_t`
- **Student action** `a_t = u_t^S`: A natural language utterance (solution attempt, question, etc.)
- **Teacher action** `u_{t+1}^T ~ pi_T(.|s_t, k_t)`: Natural language feedback conditioned on privileged info
- **Transition**: Deterministic appending to conversation history; stochastic through teacher's policy
- **Reward** `R(s_t, a_t)`: Sparse, binary -- +1 if student's answer is correct, 0 otherwise. Only awarded at conversation-level (not per-turn).
- **Episode**: Terminates when student produces correct answer OR max turns `T_max` reached.

### 2.2 The RL^2 / Black-Box Meta-Learning Connection

This is explicitly identified as analogous to **RL^2** (Duan et al., 2016; Wang et al., 2016) and first-order meta-learning (Nichol et al., 2018):

- **Inner loop (in-context)**: Within a single episode/conversation, the student learns to integrate feedback and improve its answer. No weight updates happen here -- this is pure in-context learning.
- **Outer loop (weight updates)**: Across episodes, gradient updates optimize the student's weights to be better at learning from feedback in-context.

The language feedback from the teacher can be interpreted as an **augmented reward observation** -- the LLM is effectively implementing an in-context RL algorithm whose weights are optimized via outer-loop RL. This is why the method is called **RL^2F** (Reinforcement Learning with Language Feedback).

The critical difference from standard RL^2: instead of scalar rewards as the "augmented observation," the agent receives **rich natural language feedback** that contains structured information about what went wrong and how to fix it. This makes the inner-loop learning signal dramatically more informative.

---

## 3. Training: How Corrective Feedback Creates Multi-Turn Trajectories

### 3.1 Episode Structure (RL^2F / Paper 1)

```
Turn 1: Student receives problem -> attempts solution -> automatic verification
  If WRONG: Teacher provides language feedback (conditioned on privileged info)
Turn 2: Student sees feedback -> revises solution -> automatic verification
  If WRONG: Teacher provides more feedback
Turn 3: Student sees all prior feedback -> revises again -> verification
  If CORRECT: R = +1, episode ends
  ...
Turn T_max: If still wrong -> R = 0, episode ends
```

The student model's multi-turn context accumulates: `[problem, attempt_1, feedback_1, attempt_2, feedback_2, ...]`. This growing context IS the "memory" of the inner-loop learner.

### 3.2 Training Approaches (Paper 2: SML)

Paper 2 systematically compares training strategies:

**Offline RL (SFT on filtered data):**
- Generate conversational rollouts using the initial student policy
- Filter to keep only successful trajectories (where student eventually solves the problem)
- SFT on the student's turns from these successful dialogues

**Online RL (GRPO):**
- Generate groups of `g` conversational trajectories per problem
- Each trajectory gets trajectory-level binary reward
- Use GRPO with group-normalized advantages: `A_k = (r_k - mean(r)) / std(r)`
- Apply reward discounting over turns with factor `gamma = 0.7`
- No KL penalty (`beta = 0`)

**Key finding: Online RL is substantially more effective than offline SFT.** Online RL generalises better to test problems and to longer conversations than were used during training (e.g., training on 4-turn conversations enables learning from feedback for up to 10 turns at test time). This mirrors the broader finding that RL generalises where SFT memorises (Chu et al., 2025; Kirk et al., 2024).

### 3.3 Q-Priming (Paper 2)

SML alone does not explicitly promote **question-asking** -- a desirable behavior for navigating ambiguity. Paper 2 introduces **Q-priming**: a preliminary SFT stage that explicitly teaches the model to ask clarifying questions.

- During data generation, if a student's turn is incorrect, with probability `P_Q(t) = 0.75^t * I[R(s_t, a_t) = 0]`, replace the response with a generated question
- The question is generated by providing the student model with both its prior attempt AND the teacher's private knowledge, then prompting it to formulate an informative query
- The probability decays exponentially over turns, encouraging exploration early in conversations
- After Q-priming SFT, continue with online RL training

Result: Q-priming models make **5x fewer premature answer attempts** on ambiguous tasks and are far more likely to ask clarifying questions. This represents a shift from "presumptive guessing" to "proactive enquiry."

---

## 4. How the Student Learns to Leverage Past Mistakes (In-Context Adaptation)

### 4.1 The Plasticity Problem

A key empirical finding across both papers: **current frontier models (Gemini 2.5 Pro, GPT-5, Flash, Flash-Lite) are remarkably poor at learning from corrective feedback.** Evaluated across HardMath2, ARC-AGI, Codeforces, and BIG-Bench Extra Hard:

- Performance improves only modestly across multiple turns of teacher feedback
- Larger models tend to improve more (GPT-5 > Gemini 2.5 Pro > Flash > Flash-Lite)
- But even the best models leave enormous room for improvement

The baseline failure mode is vivid (see Figure 6 in Paper 1): when receiving precise mathematical corrections, the baseline Gemini 2.5 Flash **refuses to update its stance, repeats identical incorrect code, and eventually stops using thinking tokens entirely**. The model literally gives up on reasoning rather than integrating the feedback.

### 4.2 In-Context Plasticity

The authors define **"in-context plasticity"** as the ability of a neural network to change its predictions in response to new information provided in-context (by analogy to "plasticity" in continual learning, which refers to changing predictions in response to new weight updates).

After RL^2F training:
- The fine-tuned model demonstrates **dramatically increased in-context plasticity**
- It uses its thinking traces not merely to justify prior output, but to **reason about the teacher's hints** and successfully integrate them
- Paper 2 shows this quantitatively: the loss on the correct answer decreases steadily across conversation turns for SML-trained models, while the base model shows no clear trend

This is perhaps the most important conceptual contribution: **the ability to learn from feedback is itself a learnable skill**, not just an emergent property of scale.

### 4.3 Loss Dynamics (Paper 2, Figure 3)

Paper 2 measures the average loss on the correct answer following each teacher turn:
- **SML-trained model**: Clear and steady loss reduction as conversation progresses (the model gets better at predicting the right answer after each piece of feedback)
- **Base model**: No clear trend -- feedback does not systematically reduce loss on the correct answer

This is direct evidence that SML teaches models **how** to learn from language feedback.

---

## 5. Results

### 5.1 Headline Result: Flash Matching Pro (Paper 1)

On HardMath2 (advanced, multi-turn math):
- **Gemini 2.5 Flash + RL^2F** nearly matches **Gemini 2.5 Pro** in multi-turn performance
- Single-turn RL only slightly improves baseline Flash's ability to learn from feedback across turns
- RL^2F continuously widens the gap relative to baselines as turns increase

This is significant because mathematics is a domain where Pro typically excels over Flash by a full tier. The training procedure bridges this gap through improved interactivity alone.

### 5.2 Non-Thinking Models (Paper 1 & 2)

On Gemma-3-12B-IT (non-thinking model):
- **Omni-MATH**: RL^2F is the most effective strategy. Standard RL and SFT match single-turn performance but only RL^2F continuously widens the gap over multiple turns.
- The improvement over baselines grows with the number of interaction turns, confirming that the method specifically improves the ability to **accumulate information** across a conversation.

### 5.3 Out-of-Distribution Generalization

This is one of the most striking findings: **training on math interactions transfers to completely different domains.**

**Paper 1 -- Gemini 2.5 Flash (trained on math only):**

| Task | RL^2F | Single-turn RL | Baseline Flash |
|---|---|---|---|
| Multi-turn ARC-AGI | **23.56** | 20.47 | 20.47 |
| Multi-turn Codeforces | **37.03** | 32.77 | 33.33 |
| Multi-turn Linguini | **56.00** | 42.35 | 42.00 |
| **Average** | **38.86** | 31.86 | 31.93 |

RL^2F shows ~7% average cross-domain improvement. Single-turn RL shows virtually zero transfer. This confirms that RL^2F teaches a **general interactive reasoning capability**, not domain-specific math tricks.

**Paper 1 -- Beyond Didactic Interactions (Gemini 2.5 Flash, math-trained):**

Evaluated on 10 diverse out-of-distribution multi-turn agentic tasks (no teacher involved -- the model just acts within environments):

| Task | RL^2F | Single-turn RL | Baseline Flash |
|---|---|---|---|
| Maze Navigation | **87.50** | 78.35 | 75.00 |
| Only Connect Wall | **72.00** | 44.75 | 53.00 |
| Poker | **38.71** | 36.95 | 36.82 |
| Wordle | **59.03** | 57.42 | 56.72 |
| **Average (10 tasks)** | **51.65** | 46.54 | 46.92 |

Maze Navigation (+12.5%) and Only Connect Wall (+19%) show the largest gains. These are tasks requiring multi-step reasoning and environmental feedback integration -- precisely the skills that RL^2F trains.

**Paper 2 -- Cross-domain transfer (Gemma-3-12B-IT):**
- Math-to-Code: Training SML on Omni-MATH, evaluating on LiveCodeBench -- notable multi-turn performance gain
- Code-to-Math: Training SML on OpenCodeInstruct, evaluating on Omni-MATH -- similar transfer
- Both directions work, confirming the skill is domain-general

**Paper 2 -- Feedback Generalisation (Lost-in-Conversation benchmark):**
- Even though SML trains on fully-specified problems with corrective feedback, it transfers to **underspecified "sharded" problems** where information is revealed incrementally
- MT-RL achieves 69.9% on sharded math vs. 62.7 baseline, and 54.8% on sharded code vs. 51.5 baseline

---

## 6. Pathway to Self-Improvement (Paper 1, Section 4)

A remarkable extension: the feedback loop can be **internalized**.

### 6.1 World-Modeling Objective

Rather than training only on the student's turns, train on the **full interactive dialogue**, including the teacher's critiques. Crucially, train the model to predict the teacher's turns **as if they were unconditioned on privileged information** -- requiring the model to infer the logic of the critique solely from the context of the error.

This is analogous to an **auxiliary world-modeling objective**: the model learns to predict what feedback it would receive, effectively modeling the "feedback environment."

### 6.2 Self-Improvement at Inference

At inference time, the model interacts with **itself** -- generating both student attempts and self-critiques. The autodidact agent:
- Generates a loop of self-critiques and self-refinements
- Does NOT condition the self-critique on privileged information
- Gives itself feedback solely by observing its past attempts

Result: **The autodidact agent outperforms the agent evaluated through didactic interactions with a privileged teacher.** The external training signal prevents degenerate self-feedback loops that typically plague self-improvement methods.

---

## 7. "In-Context Plasticity" -- Concept Deep Dive

Plasticity (Dohare et al., 2024; Lyle et al., 2023; Nikishin et al., 2022) is well-studied in continual RL as the ability of a neural network to change its predictions in response to new data through weight updates. The authors identify an analogous phenomenon for **in-context learning**:

**Lack of in-context plasticity**: Despite receiving precise corrective feedback, baseline models:
- Refuse to update their stance
- Repeat identical incorrect code/solutions
- Eventually cease using thinking tokens (give up on reasoning)
- Explicitly acknowledge feedback but fail to integrate it (Paper 1, Figure 8 -- the model says "the teacher is right" but then repeats the same answer)

**Enhanced in-context plasticity** (post-RL^2F/SML training):
- Models use thinking traces to actively reason about teacher hints
- Successfully derive new, correct approaches based on feedback
- Show greater task decomposition
- Better integration leads to improved final performance

This is a first clear characterization of this phenomenon, with the authors noting it requires deeper investigation.

---

## 8. Technical Details

### 8.1 Paper 1 (RL^2F on Gemini)
- **Base model**: Gemini 2.5 Flash (thinking model)
- **Training data**: Private set of mathematics questions (harder than Omni-MATH)
- **Evaluation**: HardMath2 (in-domain), ARC-AGI, Codeforces, Linguini, + 10 agentic tasks (OOD)
- **Teacher**: Same model with privileged info; teacher leaks solution only ~0.3% of the time
- **Max turns**: T_max (not specified, appears to be 3 for eval)
- **Verification**: String matching + LLM judge pipeline to detect solution leakage

### 8.2 Paper 2 (SML on Open Models)
- **Base model**: Gemma-3-12B-IT (primary), Qwen3-8B (appendix)
- **Training data**: 2,000 problems from Omni-MATH (math), OpenCodeInstruct (code)
- **RL algorithm**: GRPO with group size `g = 8`, no KL penalty (`beta = 0`)
- **Reward discounting**: `gamma = 0.7` over conversational turns
- **Max turns**: N = 4 during training, evaluated at N = 10
- **Teacher**: Same Gemma-3-12B-IT model (shown: stronger teacher during training has minimal impact; information asymmetry is what matters, not teacher quality)
- **Training**: Single epoch, three seeds, 95% confidence intervals reported
- **Q-priming**: SFT stage with question injection probability `P_Q(t) = 0.75^t`

---

## 9. Relevance to ARC-AGI and Our REPL Environment

### 9.1 Our Setup is Already Multi-Turn

Our ARC-AGI REPL environment (`/home/baris/repos/rlvr/environments/arc_agi/src/arc_agi/envs/repl.py`) is inherently multi-turn:

- The model operates in an iterative REPL loop: reason -> write code -> observe output -> reason again -> write more code -> ...
- Each REPL iteration provides **environmental feedback** (code execution results, printed grids, error messages)
- The model must learn to leverage this feedback to iteratively refine its solution
- The episode terminates only when `SUBMIT()` is called

This is structurally similar to the RL^2F/SML setup, where the student gets multiple turns of feedback before the episode ends. The REPL output IS the "teacher's feedback" in our case.

### 9.2 What These Papers Suggest We Could Do

**Opportunity 1: Teacher-Augmented Feedback During Rollouts**

Currently, our REPL environment returns raw execution output (stdout, stderr, errors). The papers suggest we could augment this with **teacher feedback** that has access to privileged information:

- The teacher could see the ground-truth output grid and the student's current prediction
- Instead of just "REPL Output: [grid]", the teacher could provide targeted hints: "Your prediction has the wrong color in the top-left 3x3 region" or "The transformation rule works for training example 1 but fails on example 2 -- check your handling of rotated inputs"
- The teacher can be the same model with privileged info (ground-truth test outputs)

This would create richer feedback signals during training rollouts without requiring a stronger model.

**Opportunity 2: Training the "Learn from Feedback" Skill**

The most transferable insight: **training on didactic interactions in one domain improves multi-turn performance in unrelated domains.** Concretely:

- We could train on math/code didactic interactions (cheap, abundant data) and expect transfer to ARC-AGI multi-turn REPL performance
- The RL^2F-trained Gemini 2.5 Flash already showed improvement on multi-turn ARC-AGI (23.56 vs 20.47 baseline) despite being trained only on math
- This suggests a pre-training/warm-up stage: train the interactive learning skill on easy-to-generate math/code interactions before fine-tuning on ARC-AGI specifically

**Opportunity 3: Structured Multi-Turn RL Training**

Our current RL training likely treats the full REPL episode as a single trajectory. The papers suggest:

- Explicitly modeling the multi-turn structure with reward discounting (`gamma = 0.7`)
- Using GRPO with trajectory-level rewards and group normalization
- Training with a maximum of N = 4 turns but evaluating at N = 10 (the skill generalizes to longer interactions)

**Opportunity 4: Self-Improvement Through Internalized Critique**

Paper 1's Section 4 shows the model can internalize the teacher's feedback capability:
- Train the model to predict what feedback a teacher WOULD give
- At inference, the model self-critiques without privileged information
- This could be integrated into our REPL loop: after each code execution, the model could generate an internal critique before its next attempt

**Opportunity 5: Q-Priming for ARC-AGI**

ARC-AGI tasks are inherently underspecified from the model's perspective (it must discover the transformation rule). Q-priming's insight -- teaching models to ask clarifying questions and explore before committing to an answer -- maps naturally to our REPL:
- Instead of immediately trying to solve, the model could first explore by printing grids, computing statistics, testing partial hypotheses
- Q-priming could encourage this exploratory behavior during early REPL iterations

### 9.3 Key Differences from Our Setup

Important distinctions to keep in mind:

1. **Our "teacher" is the environment, not an LLM**: In our REPL setup, feedback comes from code execution (deterministic), not from a language model teacher (stochastic). This is potentially easier to work with since the feedback is objective and not subject to LLM hallucination.

2. **Our tasks are harder to verify incrementally**: In math, you can check if the final answer is correct. In ARC-AGI, intermediate steps (understanding the pattern, implementing the transformation) are harder to verify automatically. The REPL execution output provides some signal but not direct correctness feedback until `SUBMIT()`.

3. **Our privileged information is different**: For ARC-AGI, the privileged info is the test output grids. For code, it is unit test results. A teacher with access to test outputs could provide feedback like "your prediction for test case 0 has accuracy 0.4 -- the bottom half is correct but the top half has wrong colors."

4. **Soft accuracy as intermediate reward**: Our environment provides `soft_accuracy()` (cell-level matching). This could serve as a natural intermediate signal for teacher feedback: "Your current prediction matches 73% of cells in test case 0" -- providing a gradient-like signal through language.

---

## 10. Open Questions and Future Directions

1. **Curriculum learning**: Neither paper explores curriculum over problem difficulty. The teacher does not select problems based on the student's current level. This is explicitly noted as promising future work.

2. **Safety concerns**: Enhanced in-context adaptability could also increase sycophancy or susceptibility to manipulation. The authors flag this but do not address it.

3. **Mixed-motive settings**: The cooperative teacher-student setup could be extended to competitive or mixed-motive scenarios (debate, negotiation) which might yield different cognitive capabilities.

4. **Continual learning**: The papers improve in-context (short-term) adaptation but do not address how to consolidate these transient improvements into permanent capabilities.

5. **Dense rewards**: Both papers use sparse, conversation-level rewards. Turn-level rewards (e.g., using the teacher as a judge of progress) could improve sample efficiency.

6. **Dynamic tasks**: All experiments assume a static underlying task. Extending to settings where user intent, knowledge, or preferences evolve during conversation is noted as important future work.

---

## 11. Summary Table

| Aspect | Paper 1 (RL^2F) | Paper 2 (SML) |
|---|---|---|
| **Focus** | Thinking models, flagship results | Open models, thorough ablations |
| **Models** | Gemini 2.5 Flash/Pro | Gemma-3-12B-IT, Qwen3-8B |
| **Method name** | RL^2F | SML |
| **Training** | Online RL (details sparse) | GRPO (g=8, beta=0, gamma=0.7) |
| **Key innovation** | Self-improvement via world modeling | Q-priming for question-asking |
| **OOD eval** | ARC-AGI, Codeforces, Poker, Wordle, etc. | LiveCodeBench, Lost-in-Conversation |
| **Transfer** | Math -> 10 diverse agentic tasks | Math <-> Code bidirectional |
| **Headline** | Flash matches Pro | Online RL >> SFT; Q-priming works |

---

## References

- Klissarov, M., Cook, J., Antognini, D., Sun, H., Li, J., Jaques, N., Musat, C., & Grefenstette, E. (2026). Improving Interactive In-Context Learning from Natural Language Feedback. arXiv:2602.16066.
- Cook, J., Antognini, D., Klissarov, M., Musat, C., & Grefenstette, E. (2026). Learning to Learn from Language Feedback with Social Meta-Learning. arXiv:2602.16488.
- Duan, Y., Schulman, J., Chen, X., Bartlett, P.L., Sutskever, I., & Abbeel, P. (2016). RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning. arXiv.
- Wang, J.X., et al. (2016). Learning to reinforcement learn. arXiv.
- Nichol, A., Achiam, J., & Schulman, J. (2018). On first-order meta-learning algorithms. arXiv:1803.02999.
