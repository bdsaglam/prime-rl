# Open Research Questions: On-Policy Distillation for Reasoning Tasks

These are working notes on fundamental questions about OPD/SDFT, particularly for tasks requiring genuine exploration and reasoning (ARC-AGI, math proofs, code generation with debugging).

---

## 1. The Exploration Problem

### The Student Analogy

A student who reads solutions thinks they understand the material. They follow each step, it all makes sense. Then they sit the exam and can't solve anything independently. Why?

Because **reading a solution exercises a different cognitive process than generating one**. Solving requires:
- Generating candidate approaches (exploration)
- Hitting dead ends and recognizing them (backtracking)
- Managing uncertainty about whether you're on the right path
- Recovering from partial failures mid-trajectory

A solution trace is a clean path from A to Z. Real reasoning is A → D → dead end → back to B → F → wrong → C → E → ... → Z. The messy process IS the skill.

### How This Applies to OPD

In OPD, the teacher sees privileged information and produces a "clean" token distribution — it knows where to go, so its distribution is confidently peaked on correct next tokens. The student learns to match this distribution.

But matching a confident, privileged distribution may teach the student to be **overconfident without the privilege**. Concretely:

```
Teacher (sees the transformation rule):
  Token distribution at decision point:
  "np.rot90" → 0.85 probability    ← teacher is sure, it knows the rule
  "np.flipud" → 0.10
  "np.fliplr" → 0.05

Student (trained to match teacher):
  Learns to also peak on "np.rot90" early in the trajectory
  But WITHOUT the rule, it has no basis for this confidence
  → It becomes a pattern-matcher that looks confident but isn't actually reasoning
```

This is distinct from the task-difficulty problem (Module 5). Even when the privileged info IS sufficient for the teacher, the learned behavior may be **brittle** — the student mimics the teacher's conclusions without learning the teacher's reasoning process (which relied on the privilege).

### The Distributional Mismatch

There may be a fundamental mismatch between:
- **Teacher distribution**: Shaped by certainty (knows the answer/rule)
- **Optimal student distribution**: Should reflect genuine uncertainty, maintain exploration

The optimal policy for a student who doesn't know the answer should have **higher entropy at decision points** — it should hedge, try things, and use REPL feedback to narrow down. The teacher's low-entropy distribution may be the wrong target entirely.

**Question**: Is there a way to construct a teacher distribution that reflects "how I would reason if I were good at this" rather than "how I would reason if I already knew the answer"?

---

## 2. Beyond Task Difficulty: Three Axes of OPD Applicability

Module 5 discusses the difficulty axis. But there are at least three:

### Axis 1: Reasoning Gap (from Module 5)
Does the privileged info let the teacher produce a better distribution?
- Small gap → weak signal (answer-only for hard ARC)
- Large gap → strong signal (frontier analysis for ARC)

### Axis 2: Exploration Dependency
Does the task require exploration/backtracking to solve?
- Low exploration: ToolAlpaca (one correct tool call, no dead ends)
- High exploration: ARC via REPL (iterate, inspect, debug, backtrack)
- Very high: Mathematical proof discovery, open-ended research

For high-exploration tasks, even a perfect teacher signal may teach the wrong thing — it teaches the **destination** but not the **navigation skill**.

### Axis 3: Process vs Outcome
Is the process deterministic given the goal, or is there essential path diversity?
- Deterministic: "Sort this list" — knowing the output essentially determines the code
- Path-diverse: "Solve this ARC puzzle" — many valid code approaches for the same output
- Highly path-diverse: "Write a research paper" — the process IS the product

OPD works best in the bottom-left corner: low exploration, deterministic process. It becomes increasingly questionable as you move toward high exploration and path diversity.

```
                    Low exploration          High exploration
                    ──────────────          ────────────────
Deterministic      OPD works well           OPD teaches destination
process            (tool-use, simple        but not navigation
                    code generation)         (math proofs?)

Path-diverse       OPD picks ONE path       OPD is questionable
process            (may be fine, may         (creative tasks,
                    miss better ones)         open-ended reasoning)
```

---

## 3. What Would Fix This?

### Idea A: Process-Aware Teacher

Instead of conditioning on the answer/analysis, condition on **successful trajectories**. The teacher is the same model, but it has seen an example of a full reasoning trace (including backtracking) that leads to success:

```
Teacher prompt: [task + "Here is a successful solving session:\n" + full_trajectory]
```

This preserves the messiness of real reasoning. The teacher's distribution reflects how to reason, not just what to conclude. But it requires having successful trajectories — which is a chicken-and-egg problem (you need a policy that can solve puzzles to generate training data for the policy).

One resolution: use the frontier LLM to generate full multi-turn solving sessions, not just analyses.

### Idea B: Entropy-Regularized OPD

Add an entropy bonus to the OPD loss to prevent the student from becoming overconfident:

```
Loss = KL(student || teacher) - beta * H(student)
```

This encourages the student to match the teacher's relative preferences (rot90 > flipud) while maintaining enough entropy to explore. The student learns "rot90 is promising" without learning "rot90 is certain."

### Idea C: Advantage-Weighted OPD

Only apply the teacher signal on tokens where the student's choice actually matters. Weight the KL by the advantage:

```
weighted_kl_t = advantage_t * KL_t(student || teacher)
```

Tokens where all choices lead to similar outcomes (low advantage) get low weight — the student doesn't need to match the teacher there. Tokens at critical decision points (high advantage) get high weight. This focuses distillation on the decisions that matter.

Requires running enough rollouts to estimate advantages, which partially merges OPD with GRPO.

### Idea D: Reflective Retry as Privileged Information (Baris's idea)

The core problem with Ideas A-C is that the teacher's privilege comes from *external* information (answers, analyses, trajectories). This produces a clean distribution that doesn't reflect the student's actual reasoning process. What if instead, the privilege is the **student's own corrected reasoning**?

The setup:

```
1. Student generates trajectory on-policy (may fail)
   Turn 1: explore → hypothesis A
   Turn 2: code based on A → wrong output
   Turn 3: try hypothesis B → still wrong
   Turn 4: SUBMIT(wrong) → reward = 0

2. Reveal the answer to the SAME agent, ask it to reflect:
   "Your submitted answer was wrong. The correct output is: [grid]
    Look back at your trajectory. Where did you go wrong?
    Now retry from scratch (or from a specific earlier point)."

3. Agent retries WITH the answer visible, producing a corrected trajectory:
   Turn 1': explore → hypothesis A (same start, natural)
   Turn 2': "wait, knowing the answer, A doesn't produce [grid]..."
   Turn 3': pivot to hypothesis C → correct!
   Turn 4': verify → SUBMIT(correct)

4. The corrected trajectory becomes the teacher's privileged info
```

Why this is better than a clean frontier analysis:

- **The trajectory starts from the student's own reasoning.** It shares the same initial exploration, the same hypotheses, the same code style. The divergence point — where the student backtracks after seeing the answer — is exactly the decision that matters.

- **It preserves the messy exploration structure.** The teacher trajectory has dead ends, backtracking, and recovery. It teaches navigation, not just destination. The student learns "when you see output X and it doesn't match, pivot to approach C" rather than "always do approach C."

- **The privilege is minimal and targeted.** The agent doesn't get a full analysis — it gets its own failed trajectory + the answer. The correction it produces is grounded in its own experience. This satisfies the SDFT "minimal deviation" property better than injecting external analysis.

- **It solves the chicken-and-egg problem from Idea A.** You don't need successful trajectories from a strong model. You use the student's own failures, enhanced by hindsight.

Implementation in the REPL environment:

```python
# Phase 1: Student rollout (on-policy, no privilege)
trajectory = student.rollout(task)  # multi-turn REPL interaction
reward = exact_match(trajectory.submission, ground_truth)

if reward < 1.0:
    # Phase 2: Reflective retry (with answer revealed)
    retry_prompt = (
        f"Your attempt failed. The correct output is:\n{format_grid(ground_truth)}\n\n"
        f"Here is your previous attempt:\n{format_trajectory(trajectory)}\n\n"
        "Reflect on where you went wrong. Identify the incorrect assumption or bug. "
        "Then solve the puzzle again, starting from scratch or from the point "
        "where you diverged from the correct approach."
    )
    corrected_trajectory = student.rollout(task, prefix=retry_prompt)

    # Phase 3: Use corrected trajectory as teacher signal
    # Teacher = same model conditioned on (task + failed_attempt + answer + reflection)
    # Student = same model conditioned on (task) only
    # Loss = KL(student || teacher) on the corrected trajectory's tokens
```

This is a form of **hindsight-conditioned self-improvement**: the model learns from its own mistakes, corrected by minimal privileged information (just the answer, not a full analysis). The reflection forces the model to articulate what went wrong, which is itself a reasoning skill.

**Key variants:**

- **Retry from scratch**: Full re-solve after reflection. Teaches the complete reasoning process.
- **Retry from branch point**: Resume from the last correct turn, only re-do the part that went wrong. More targeted signal, less compute.
- **Multiple retries with decreasing privilege**: First retry sees the answer, second retry sees only "your first retry was closer but still wrong," etc. Fading scaffolding.

**Open question**: Does the reflection itself need to be high quality? If the 7B model's reflection is shallow ("I made an error"), the corrected trajectory may not be much better. May need the frontier LLM for the reflection step, or may need to filter for retries that actually succeed.

**Connection to education research**: This is essentially the "test-enhanced learning" or "retrieval practice" effect (Roediger & Karpicke, 2006). Students who attempt retrieval and then get corrective feedback learn better than students who study the material passively. The failed attempt creates a "desirable difficulty" that the correction can build on.

### Idea E: Staged Curriculum Within OPD

Start with the teacher having VERY strong privilege, then gradually weaken it:

```
Phase 1: Teacher sees full successful trajectory → strong signal, teaches process
Phase 2: Teacher sees frontier analysis → medium signal, teaches reasoning
Phase 3: Teacher sees answer + own failed attempt (Idea D) → teaches self-correction
Phase 4: Teacher sees answer only → weak signal, teaches verification
Phase 5: No teacher (pure RL) → learns from own outcomes
```

Each phase relies on less privilege, forcing the student to internalize more of the reasoning. Ideas A-D slot naturally into different phases of this curriculum.

---

## 4. Related Work to Investigate

- **Process Reward Models (PRM)**: Score intermediate reasoning steps, not just outcomes. Relevant because they address credit assignment in reasoning chains. Could PRMs provide a better teacher signal than OPD?

- **Expert Iteration (ExIt)**: Generate solutions, filter for successful ones, SFT on those. Crude but sidesteps the exploration problem by learning from the student's own successful trajectories. Compare: ExIt teaches from student's messy process; OPD teaches from teacher's clean distribution.

- **MCTS-guided training** (AlphaProof-style): Use search to find solutions, then distill the search policy. The search process IS the exploration. Relevant comparison point.

- **Curriculum learning for RL**: Starting with easy tasks and progressing to hard ones. Complementary to OPD — you could do OPD on easy puzzles (where the signal is good) and RL on hard ones (where OPD signal degrades).

- **Hindsight Experience Replay (HER)**: From robotics RL. When a trajectory fails to reach the goal, relabel it as if a different goal was intended. Philosophically related: using alternative interpretations of trajectories as training signal.

- **The "illusion of competence" in education research**: Bjork & Bjork's desirable difficulties framework. Students who study solutions perform worse on tests than those who struggle through problems. This is exactly the exploration problem applied to human learning.

---

## 5. Concrete Next Steps for ARC-AGI

1. **Validate the ICL assumption empirically**: For 50 ARC puzzles at varying difficulty, compare:
   - Base model solve rate (no privilege)
   - Base model + answer grid (naive SDFT privilege)
   - Base model + frontier analysis (rich privilege)
   - Base model + full successful trajectory (process privilege)

   Measure the gap. If answer-only doesn't improve solve rate much, naive SDFT is dead on arrival.

2. **Generate frontier analyses**: Run Claude on the full ARC training set. Store (task_id, analysis, confidence). Filter for high-confidence analyses. This is the dataset for OPD Config 2.

3. **A/B test**: Train two models with same compute budget:
   - Model A: GRPO only (sparse reward, many rollouts)
   - Model B: OPD Phase 2 → GRPO Phase 3 (dense signal first, then RL)

   Compare on held-out ARC tasks. If B > A, the dense signal from OPD is worth the frontier-LLM cost.

4. **Monitor exploration metrics during training**: Track entropy of the policy at decision points. If OPD causes entropy to collapse (student becomes overconfident), that's the exploration problem manifesting. Compare entropy trajectories between pure GRPO and OPD→GRPO.
