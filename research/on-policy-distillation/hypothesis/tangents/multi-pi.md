# Multi-PI: Multiple Targeted Analyses as PI

## Idea

Instead of one analysis, generate multiple short, structured analyses from different angles and apply all as PI. Each captures a different aspect of the student's experience.

## Motivation

- Verbose analysis fails because it tries to cover everything in one pass → noise
- Structured analysis succeeds because it's precise about one thing → signal
- Multiple structured analyses could be precise about *many* things while staying concise

## Possible Lenses

1. **Error classification** (structured): VERDICT/ERROR_TYPE/ERROR_LOCATION
2. **Approach comparison**: What strategy did student use vs what would work better?
3. **Key insight**: What mathematical fact or technique is the student missing?
4. **Step-by-step trace**: Which specific step first diverges from correct reasoning?

## Implementation Options

- Concatenate all as one PI string (simplest)
- Separate PI injections at different points in the prompt
- Weighted combination in the loss function

## Connection to Findings

- Best-of-4 blind already showed +31-33% over single analysis ([`FINDINGS.md`](../../experiments/opd-signal/FINDINGS.md))
- That was selection (pick best); this is combination (use all)
- Structured format keeps each piece short → less noise risk from concatenation

## Status

Idea stage. Needs experiment design.
