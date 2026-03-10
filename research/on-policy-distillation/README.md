# On-Policy Distillation for ARC-AGI

Research docs for training ARC-AGI REPL agents with on-policy distillation (OPD).

## Experiments

### [`experiments/arc-agi-opd/`](experiments/arc-agi-opd/)

ARC-AGI training with OPD — the main experiment track.

| File | Phase | Status | Summary |
|------|-------|--------|---------|
| [`arc-agi-opd-plan.md`](experiments/arc-agi-opd/arc-agi-opd-plan.md) | 0-3 | Phases 0-1 done | Original phased plan. Phase 0 (standard OPD) through Phase 3 (RLTF auxiliary loss). |
| [`phase1-privileged-teacher.md`](experiments/arc-agi-opd/phase1-privileged-teacher.md) | 1 | Done (no improvement) | Privileged-info teacher with ground-truth outputs. 23/200 steps, rewards flat at 0.15-0.20. |
| [`phase1.5-hint-curriculum.md`](experiments/arc-agi-opd/phase1.5-hint-curriculum.md) | 1.5 | Done | Hint-assisted curriculum: 1.5a (hints ON, 60 steps) then 1.5b (hints OFF, 140 steps). |
| [`handover-phase1.5b.md`](experiments/arc-agi-opd/handover-phase1.5b.md) | 1.5b | Handover doc | State snapshot at step 148. Config, W&B links, hardware layout. |

### [`experiments/opd-repro/`](experiments/opd-repro/)

Verify the OPD pipeline works on a known-good benchmark before debugging ARC further.

| File | Status | Summary |
|------|--------|---------|
| [`opd-math-verification.md`](experiments/opd-repro/opd-math-verification.md) | Planning | Test OPD on Hendrycks Math. 3 runs: GRPO baseline, pure OPD, hybrid. |

### Other

| File | Description |
|------|-------------|
| [`prime-rl-training-management-guide.md`](experiments/prime-rl-training-management-guide.md) | Training management. Metrics to watch, healthy vs unhealthy patterns, failure modes, hyperparameter tuning. |

## Implementation Notes

Framework analyses, setup guides, and implementation details in [`implementation-notes/`](implementation-notes/).

| File | Description |
|------|-------------|
| [`prime-rl-implementation-notes.md`](implementation-notes/prime-rl-implementation-notes.md) | Lessons from Phase 0. Bugs found & fixed (validator ordering, teacher prefill crash, port conflicts). |
| [`prime-rl-opd-implementation.md`](implementation-notes/prime-rl-opd-implementation.md) | How prime-rl's native OPD works: config, loss function, teacher logprob computation. |
| [`prime-rl-extensibility.md`](implementation-notes/prime-rl-extensibility.md) | Extensibility for all 5 OPD variants. Difficulty ratings, LOC estimates, code snippets. |
| [`opentinker-extensibility.md`](implementation-notes/opentinker-extensibility.md) | OpenTinker framework assessment. |
| [`verl-sdpo-extensibility.md`](implementation-notes/verl-sdpo-extensibility.md) | veRL framework assessment. |
| [`external-teacher-setup.md`](implementation-notes/external-teacher-setup.md) | Hosting the teacher model on a remote server via SSH tunnel. |
| [`syncing-fork.md`](implementation-notes/syncing-fork.md) | Syncing our fork with upstream prime-rl. |

## Research Notes

Conceptual docs and design explorations in [`research-notes/`](research-notes/).

| File | Description |
|------|-------------|
| [`opd-concepts.md`](research-notes/opd-concepts.md) | OPD tutorial. Core mechanism, GKD foundation, all self-distillation variants, decision tree. |
| [`multi-lens-teacher-scoring.md`](research-notes/multi-lens-teacher-scoring.md) | Multi-lens teacher scoring design exploration. |

## Papers

Paper summaries and PDFs in [`papers/`](papers/). See [`papers/overview.md`](papers/overview.md) for the full list.

## Other

| File | Description |
|------|-------------|
| [`agent-workflow.md`](agent-workflow.md) | Agent workflow. tmux conventions, heartbeat pattern, background task management. |
