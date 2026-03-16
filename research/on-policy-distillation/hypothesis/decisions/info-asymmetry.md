# Information Asymmetry Design in RA-OPD

What the student sees vs what the teacher sees during scoring.

## The Core Principle

The OPD learning signal comes from the GAP between student and teacher knowledge. If both see the same thing, KL ≈ 0 → no signal. If teacher sees everything and student nothing, signal is maximal but may be noisy. The art is finding the right asymmetry.

## What the Student Sees During Reflection

The student solves the problem first (no PI), then receives feedback before reflecting:

| Student PI Level | What Student Knows | Use Case |
|---|---|---|
| **none** (blind) | Nothing — just reflects on its own work | Fully self-supervised |
| **binary** | "Your answer is correct/incorrect" | Cheapest useful PI (free from reward) |
| **answer** | The correct answer (not solution path) | Standard PI |
| **answer + hint** | Answer + explanation of why it matters | Richest student PI |

## What the Teacher Sees During Scoring

The teacher scores the ENTIRE sequence (solution + reflection) with richer PI:

| Teacher PI | What Teacher Knows | d (reflection) |
|---|---|---|
| none | Nothing (matched with student) | 0.37 |
| binary (matched) | Same as student | 4.47 |
| answer | Correct answer | 2.89 |
| **student's own reflection** | The structured reflection text | **5.25** |
| **answer + student's reflection** | Both | **5.30** |
| answer + ref solution | Answer + worked solution | 3.13 |

## The Best Configuration: binary → answer+reflection

**Student PI**: binary (correct/incorrect) — this is free from the reward signal, costs nothing.
**Teacher PI**: answer + student's own structured reflection.

This gives: **d=1.48 on solution, d=5.30 on reflection**.

## Why This Works

1. **Binary is the minimum useful student PI**: The student needs SOME feedback to produce informative reflections. Blind reflection (d=1.19 on reflection with teacher having reflection as PI) is much weaker than binary-informed (d=5.25).

2. **Reflection as teacher PI outperforms answer**: The student's structured self-diagnosis (K: binary→reflection, d=1.48 on solution) beats answer-only (J: binary→answer, d=0.63 on solution) by 2.3x. Why? The reflection is a compressed map of the student's reasoning — the teacher evaluates structured claims about where the student went wrong, which is more informative than just knowing the right answer.

3. **Answer is redundant on top of reflection**: K→L is 5.25→5.30 — the reflection already captures the relevant information about whether the answer matches. But answer is free, so include it.

4. **Reference solution hurts**: M (answer+ref, d=3.13) < K (reflection, d=5.25). The reference solution is generic; the student's reflection is specific to THIS rollout.

5. **Even matched PI creates signal**: When teacher PI = student PI = binary (condition I), d=4.47 on reflection. This is because the reflection CONTENT differs between correct and incorrect rollouts, and the teacher evaluates these different structured claims.

## The Self-Teaching Loop

In self-OPD (student = teacher, same weights), the student's own reflection becomes its teaching context. This is the model re-reading its structured self-diagnosis while knowing the ground truth:

```
Training time:
  Student mode: solve → told binary → reflect (with student weights)
  Teacher mode: score full sequence with answer + student's reflection (same weights, more context)
  Loss: KL between student and teacher logprobs on solution + reflection tokens

The gradient teaches:
  - Solution tokens: "here's how to solve better"
  - Reflection tokens: "here's how to reflect better"
```

Better reflections → better PI for the teacher → stronger signal → better solutions AND better reflections. Virtuous cycle.

## Connection to SDPO

SDPO (Hubotter 2026) creates asymmetry by giving the teacher a CORRECT PEER ROLLOUT as PI. The student's bad rollout is scored against what a correct rollout would look like. Our approach is fundamentally different: the PI is the student's OWN reflection on its work, not a peer's correct solution.

SDPO's sibling rollout gives d=1.02. Our reflection-as-PI gives d=5.30 on reflection tokens (and d=1.48 on solution tokens vs SDPO's d=1.02). The student's self-diagnosis contains more information than a peer solution because it's specific to the student's reasoning path.

## Key References

- Full info asymmetry table: `research/on-policy-distillation/experiments/opd-signal/FINDINGS.md` (section "Info Asymmetry: Reflection as Memory")
- Reflection-in-sequence results: `research/on-policy-distillation/experiments/opd-signal/reflection-in-seq-results.md`
