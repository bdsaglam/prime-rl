# Cross-Experience Connections

## Idea

Current analysis operates on single (problem, rollout) pairs in isolation. Richer reflection would connect across experiences: "this is the same conceptual error as problem X" or "problems like this require technique Y."

## Motivation

- Human experts build pattern libraries from experience — they don't analyze each problem from scratch
- A batch of rollouts on related problems contains shared failure modes
- Identifying cross-problem patterns could create more generalizable learning signal

## Possible Implementations

- **Batch-level analysis**: Analyzer sees multiple rollouts before producing per-rollout analysis
- **Pattern memory**: Maintain a running summary of common error types across training
- **Problem clustering**: Group problems by type, generate cluster-level insights
- **Curriculum-aware analysis**: "You've made this error 3 times in the last batch — here's the pattern"

## Challenges

- Context window limitations for batch-level analysis
- How to inject cross-problem insight into per-rollout PI
- Risk of generic advice that doesn't help on specific problems

## Status

Idea stage. Most speculative of the tangents.
