# On-Policy Distillation Research

Research docs for on-policy distillation (OPD) applied to reasoning tasks (AIME math, ARC-AGI).

## Key Result: Deliberative Teaching

A teacher model that first reasons about a student's rollout before scoring it produces **19% stronger learning signal** than standard OPD with oracle PI (answer + reference solution) -- with zero external knowledge. See [FINDINGS.md](experiments/opd-signal/FINDINGS.md) for the full results.

---

## Experiments

### [`experiments/opd-signal/`](experiments/opd-signal/) -- Signal Measurement (Main Contribution)

Measures OPD learning signal quality without training. Explores PI types, multi-lens combination, deliberative teaching, and PI placement.

| File | Summary |
|------|---------|
| [FINDINGS.md](experiments/opd-signal/FINDINGS.md) | Complete results: 15+ PI conditions across 3 teacher configs. Deliberative teaching, multi-lens, copy artifact analysis, PI placement. |
| [HANDOVER.md](experiments/opd-signal/HANDOVER.md) | Implementation plan for training validation. File map of scripts, result files, infrastructure. |

### [`experiments/training-verification/`](experiments/training-verification/) -- Training Validation

Verifying that signal measurement results translate to actual training improvements.

| File | Summary |
|------|---------|
| [verification-spike.md](experiments/training-verification/verification-spike.md) | Training experiment tracking. Baseline OPD v1/v2, deliberative OPD, adaptive PI. Eval results, training metrics, lessons learned. |
| [gap-analysis.md](experiments/training-verification/gap-analysis.md) | Why signal measurement results don't directly translate: cross-model gap dominance, PI placement, correct rollout skipping. Recommendations. |
| [pivot-strategy.md](experiments/training-verification/pivot-strategy.md) | Pivot options when AIME 2025 proved too easy: harder filtering, ARC-AGI, harder math, non-verifiable tasks. |

### [`experiments/arc-agi-opd/`](experiments/arc-agi-opd/) -- ARC-AGI Training

ARC-AGI REPL agent training with OPD. The original application domain.

| File | Summary |
|------|---------|
| [arc-agi-opd-plan.md](experiments/arc-agi-opd/arc-agi-opd-plan.md) | Phased plan (Phase 0-3). Phase 0 (standard OPD) through Phase 3 (RLTF auxiliary loss). |
| [phase1-privileged-teacher.md](experiments/arc-agi-opd/phase1-privileged-teacher.md) | Phase 1: privileged-info teacher with ground-truth outputs. 23/200 steps, rewards flat. |
| [phase1.5-hint-curriculum.md](experiments/arc-agi-opd/phase1.5-hint-curriculum.md) | Phase 1.5: hint-assisted curriculum (hints ON then OFF). |
| [handover-phase1.5b.md](experiments/arc-agi-opd/handover-phase1.5b.md) | State snapshot at step 148. Config, W&B links, hardware layout. |

### [`experiments/opd-repro/`](experiments/opd-repro/) -- Reproduction on Known Benchmarks

| File | Summary |
|------|---------|
| [opd-math-verification.md](experiments/opd-repro/opd-math-verification.md) | Plan to test OPD on Hendrycks Math. 3 runs: GRPO baseline, pure OPD, hybrid. |

### Other Experiment Docs

| File | Summary |
|------|---------|
| [literature-review-1.md](experiments/literature-review-1.md) | Comprehensive review: deliberative teaching, credit assignment optimization in OPD. Covers CaT, OPSD, COMPACT, ThinkPRM, and more. |
| [literature-review-2.md](experiments/literature-review-2.md) | Focused novelty assessment: 11 related papers with threat levels. OPSD, SDFT, GKD, SKD, pi-Distill, RLAD, ThinkPRM, Rubric-ARM, COMPACT, Mind's Mirror, SCoTD. |
| [analyzer-prompt-comparison.md](experiments/analyzer-prompt-comparison.md) | Side-by-side comparison of analyzer prompts on specific AIME problems. |
| [prime-rl-training-management-guide.md](experiments/prime-rl-training-management-guide.md) | Training management. Metrics to watch, healthy vs unhealthy patterns, failure modes, hyperparameter tuning. |

---

## Research Notes

Conceptual docs and design explorations in [`research-notes/`](research-notes/).

| File | Summary |
|------|---------|
| [opd-concepts.md](research-notes/opd-concepts.md) | OPD tutorial. Core mechanism, GKD foundation, all self-distillation variants, decision tree. |
| [multi-lens-teacher-scoring.md](research-notes/multi-lens-teacher-scoring.md) | Multi-lens teacher scoring design exploration. |
| [open-questions.md](research-notes/open-questions.md) | Fundamental questions about OPD for reasoning tasks: exploration problem, teacher confidence, multi-turn trajectories. |

---

## Implementation Notes

Framework analyses, setup guides, and implementation details in [`implementation-notes/`](implementation-notes/).

| File | Summary |
|------|---------|
| [prime-rl-implementation-notes.md](implementation-notes/prime-rl-implementation-notes.md) | Lessons from Phase 0. Bugs found and fixed (validator ordering, teacher prefill crash, port conflicts). |
| [prime-rl-opd-implementation.md](implementation-notes/prime-rl-opd-implementation.md) | How prime-rl's native OPD works: config, loss function, teacher logprob computation. |
| [prime-rl-extensibility.md](implementation-notes/prime-rl-extensibility.md) | Extensibility for all 5 OPD variants. Difficulty ratings, LOC estimates, code snippets. |
| [opentinker-extensibility.md](implementation-notes/opentinker-extensibility.md) | OpenTinker framework assessment. |
| [verl-sdpo-extensibility.md](implementation-notes/verl-sdpo-extensibility.md) | veRL framework assessment. |
| [external-teacher-setup.md](implementation-notes/external-teacher-setup.md) | Hosting the teacher model on a remote server via SSH tunnel. |
| [syncing-fork.md](implementation-notes/syncing-fork.md) | Syncing our fork with upstream prime-rl. |

---

## Papers

Paper summaries and PDFs in [`papers/`](papers/). See [`papers/overview.md`](papers/overview.md) for the full list.

Key papers:
- GKD (Agarwal 2023) -- On-policy distillation foundation
- OPSD (Zhao 2026) -- Self-distilled reasoner, on-policy self-distillation
- SDFT (Shenfeld 2026) -- Self-distillation for continual learning
- SDPO (Hubotter 2026) -- Self-distillation as on-policy preference optimization
- pi-Distill (Penaloza 2026) -- Privileged information distillation
- RLTF (Song 2026) -- Reinforcement learning with token-level feedback
- Meta-Learning (Klissarov 2026) -- Meta-learning perspective on distillation

---

## Directory Structure

```
research/on-policy-distillation/
├── README.md                          # This file
├── experiments/
│   ├── opd-signal/                    # Signal measurement (main contribution)
│   │   ├── FINDINGS.md                # Full results: deliberative teaching, multi-lens, placement
│   │   └── HANDOVER.md                # Implementation plan, file map, infrastructure
│   ├── training-verification/         # Training validation experiments
│   │   ├── verification-spike.md      # Experiment tracking (baseline, deliberative, adaptive)
│   │   ├── gap-analysis.md            # Signal-to-training gap analysis
│   │   └── pivot-strategy.md          # Pivot options (harder tasks, new domains)
│   ├── arc-agi-opd/                   # ARC-AGI OPD training (Phase 0-1.5)
│   ├── opd-repro/                     # Hendrycks Math reproduction plan
│   ├── literature-review-1.md         # Comprehensive literature review
│   ├── literature-review-2.md         # Focused novelty assessment (11 papers)
│   ├── analyzer-prompt-comparison.md  # Analyzer prompt examples
│   └── prime-rl-training-management-guide.md
├── research-notes/                    # Conceptual docs
│   ├── opd-concepts.md                # OPD tutorial and variants
│   ├── multi-lens-teacher-scoring.md  # Multi-lens design exploration
│   └── open-questions.md              # Fundamental research questions
├── implementation-notes/              # Framework and setup docs
│   ├── prime-rl-implementation-notes.md
│   ├── prime-rl-opd-implementation.md
│   ├── prime-rl-extensibility.md
│   ├── external-teacher-setup.md
│   ├── syncing-fork.md
│   ├── opentinker-extensibility.md
│   └── verl-sdpo-extensibility.md
└── papers/                            # Paper summaries + PDFs
    ├── overview.md
    ├── gkd-agarwal-2023.md (+pdf)
    ├── opsd-zhao-2026.md (+pdf)
    ├── sdft-shenfeld-2026.md (+pdf)
    ├── sdpo-hubotter-2026.md (+pdf)
    ├── pi-distill-penaloza-2026.md (+pdf)
    ├── rltf-song-2026.md (+pdf)
    ├── meta-learning-klissarov-2026.md (+pdfs)
    └── 2603.05433/ (overview.md + paper.pdf)
```
