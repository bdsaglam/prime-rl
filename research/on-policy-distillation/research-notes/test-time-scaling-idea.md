# Test-Time Scaling via Self-Analysis Loop

Date: 2026-03-12

## Foundational Hypothesis

Agent experiences contain rich learning signal that can be extracted through reflection — analogous to human introspection. Humans reflect on both successes and failures, distill lessons, create connections to prior knowledge, and update neural pathways to promote better skills and reasoning. We don't just learn from outcomes; we learn from *analyzing* our experiences.

Our signal measurement validates this: structured reflection (identifying the precise error, what should have changed) produces dramatically more precise learning signal (d=1.74) than unfocused analysis (d=0.13). The lesson isn't "less text = better" — it's that **good reflection isolates the actionable insight**, just as human introspection works best when it's targeted rather than rumination.

## Core Idea

Use structured analysis as a **test-time scaling strategy**: the model works on a single problem through iterative self-improvement without external labels.

### Loop

1. Generate N rollouts for a problem
2. Self-analyze: model generates structured analysis of its own attempts (blind — no ground truth needed)
3. Gradient update using analysis as PI (self-teacher with structured analysis)
4. Generate new rollouts with updated model
5. Repeat until convergence or budget exhausted

### Why This Could Work

- **Structured analysis has d=0.55-0.61 even blind** — no external verification needed
- **Analysis is the strongest PI in no-ground-truth regime** — blind directive gives 0.048 |KL| vs 0.002 for no_pi, a 24x improvement
- **Self-reward / self-verification**: The model's own analysis serves as a verification signal. If it can identify errors in its own reasoning, it can learn from them
- **Low-label regime**: Only need problems, not solutions. Works for novel/hard problems where no reference exists

### Verification Challenge

The model needs some way to assess whether its rollouts are improving. Options:
- **Self-reward**: Model scores its own outputs (risk of reward hacking)
- **Consistency voting**: If multiple rollouts agree, higher confidence
- **Analysis confidence**: Structured analysis includes VERDICT field — can use as soft label
- **Outcome-based**: For math, can verify final answer if answer is known; analysis helps even when answer isn't known

### Connection to Existing Work

- **STaR (Zelikman et al.)**: Iterative self-improvement, but uses ground-truth filtering
- **Self-Play Fine-Tuning (Chen et al.)**: Self-play but needs a judge
- **Our angle**: Uses *structured per-token PI* via teacher logprobs, not just reward filtering. The gradient signal is richer — it tells the model *which tokens* to change, not just whether the whole rollout was good

### Key Experiment

Simplest test: take a hard problem the model always fails on (0% in current batch). Run the self-analysis loop for K iterations. Does accuracy improve?

Metrics: accuracy@N, |KL| between iterations, analysis consistency across iterations.

### Prerequisites

- Structured analysis prompt (done — now default in `AnalyzerConfig`)
- Self-teacher scoring pipeline (done — `build_teacher_prompt_ids` with `user_sdpo` placement)
- Single-problem gradient update loop (need to build — adapt from training loop)
