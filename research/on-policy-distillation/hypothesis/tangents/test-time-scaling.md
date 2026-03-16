# Test-Time Scaling via Self-Analysis Loop

Iterative self-improvement without external labels: generate rollouts → self-analyze → gradient update → retry.

Works in no-ground-truth regime where no other OPD method can help. Blind structured analysis (d=1.48 with 32B analyzer) provides meaningful signal from scratch.

Full notes: [`research-notes/test-time-scaling-idea.md`](../../research-notes/test-time-scaling-idea.md)

## Status

Idea stage. Needs single-problem gradient update loop.
