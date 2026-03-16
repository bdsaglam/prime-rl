# Reflection on Successes

## Idea

Currently we only analyze incorrect rollouts (reward=0). But humans learn from successes too. Analyzing correct rollouts could reinforce *why* they worked and promote better reasoning patterns.

## Motivation

- A correct rollout might arrive at the right answer through luck, compensating errors, or suboptimal approaches
- Reflection on "what went right and what could be better" promotes deeper understanding
- In GRPO, correct rollouts get positive advantage — but this is uniform across all tokens
- Analysis could create *non-uniform* positive signal: "these steps were strong, these were unnecessary"

## Possible Analyses

- **Efficiency analysis**: Could the solution be shorter/more elegant?
- **Robustness check**: Did the approach work for the right reasons, or was it fragile?
- **Generalization**: What broader principle does this solution exemplify?
- **Risk identification**: Where could this approach fail on similar problems?

## Connection to Training

Current OPD skips teacher context for correct rollouts (reward=1). We could instead generate "positive analysis" PI that helps the teacher give more nuanced credit to different parts of the correct solution.

## Status

Idea stage. Would need to change the `prepare_teacher_context` contract to not skip correct rollouts.
