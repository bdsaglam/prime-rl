# Correct Rollout Handling in RA-OPD

## The Question
When the student solves correctly, what should the reflection look like? The standard diagnostic prompt ("what went wrong?") is awkward for correct solutions — there's no error to diagnose.

## The Problem
With the standard diagnostic prompt, correct rollouts produce near-zero |KL| on reflection tokens:
- Correct rollout reflection |KL|: 0.0009
- Incorrect rollout reflection |KL|: 0.035+

The student writes "Nothing went wrong, solution is correct" → teacher agrees → KL ≈ 0. These tokens carry almost no learning signal.

## Alternative Prompts for Correct Rollouts

We tested specialized prompts that extract learning from successes:

| Prompt Style | |KL| (correct, reflection) | Cohen's d | Description |
|---|---|---|---|
| Diagnostic (default) | 0.0009 | 4.04 | "What went wrong?" — near-zero for correct |
| **Efficiency analysis** | **0.0128** | **1.84** | "Could you have solved this more efficiently?" |
| Teaching | 0.0236 | 0.83 | "Explain prerequisites and common pitfalls" |

## The Tradeoff
Richer correct-rollout prompts generate 14-26x more |KL| but reduce discrimination (d drops from 4.04 to 1.84 or 0.83). This makes sense: when you force the model to generate substantial text for correct rollouts, the IC/C ratio narrows.

## Current Decision: Use efficiency analysis prompt for correct rollouts

**Rationale**:
- 14x more signal than diagnostic (0.0128 vs 0.0009)
- Still good discrimination (d=1.84)
- Teaches the model to reason about solution quality, not just correctness
- "Could you have been more efficient?" always has a meaningful answer, even for correct solutions

## Implementation
In practice, the reflection prompt is chosen based on the binary feedback:
- **Incorrect**: standard diagnostic prompt (VERDICT/ERROR_TYPE/ERROR_LOCATION/WHAT_WENT_WRONG/LESSON)
- **Correct**: efficiency analysis prompt ("Your solution is correct. Reflect on whether your approach was efficient...")

This is implemented per-environment. See:
- AIME: `environments/aime/src/aime/teacher_context.py`
- ARC-AGI: `environments/arc_agi_reflect/src/arc_agi_reflect/envs/repl.py`

## Open Question
Should we skip correct rollouts entirely? In standard GRPO, correct rollouts get positive advantage, providing positive signal. Adding reflection on top may be unnecessary overhead. But the efficiency analysis does teach the model to reason about solution quality, which could improve long-term performance.

## Key References
- Correct-rollout analysis: `research/on-policy-distillation/experiments/opd-signal/FINDINGS.md` (section "Correct-Rollout Reflection Design")
