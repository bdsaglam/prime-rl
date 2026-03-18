# AIME Ablation Study: Self-Reflection OPD

## Goal
Prove that student self-reflection creates a virtuous learning cycle: reflecting → better teaching signal → better solving AND better reflecting. Compare with deliberative OPD (external analysis) and GRPO baseline.

## Paper Story
1. **A (GRPO)**: RL reward signal only — standard baseline
2. **B (Answer OPD)**: Static PI (answer+solution) — shows OPD works but signal is weak with same-size teacher
3. **C (Deliberative OPD)**: Teacher analyzes student mistakes → richer PI → stronger signal. But analysis is discarded after scoring — no feedback loop.
4. **D (Self-Reflection OPD)**: Student reflects in-sequence → teacher scores both solution AND reflection tokens → student improves at solving AND reflecting → **self-improving loop**

## Setup
- **Model**: Qwen3-8B (willcb/Qwen3-8B), LoRA rank 32
- **Training data**: aimo-validation-aime (90 AIME problems with reference solutions)
- **Eval**: AIME 2025 (30 problems, OOD) + training set sample (30 problems)
- **Steps**: 50, eval every 10
- **Hardware**: 4x A100 80GB (1 train + 2 infer + 1 teacher)

## Conditions

| Config | Env | Teacher | PI Type | Config File |
|--------|-----|---------|---------|-------------|
| **A** | aime | None | None (GRPO only) | `configs/aime/ablation-A-grpo-only-2025.toml` |
| **B** | aime | 8B self | Answer + ref solution (static from dataset) | `configs/aime/ablation-B-answer-opd-2025.toml` |
| **C** | aime | 8B self | Structured deliberative analysis + answer | `configs/aime/ablation-C-deliberative-opd.toml` |
| **D** | aime_mt | 8B self | Student reflection + answer + correct sibling | `configs/aime_mt/ablation-D-self-reflection-2025.toml` |
| **E** | aime_sdpo | 8B self | Correct sibling + answer (SDPO-style, no student attempt) | `configs/aime/ablation-E-sdpo-pi-2025.toml` |

All use: adv_tau=1.0, teacher_tau=0.5 (B/C/D/E), lr=1e-5, batch_size=32

### Key Comparisons
- **D vs E**: Does student self-reflection add value beyond SDPO's correct-sibling PI? (core hypothesis)
- **E vs B**: Does dynamic sibling selection beat static reference solutions?
- **D vs C**: Self-reflection (student learns to reflect) vs deliberative (teacher analyzes externally)
- **B/C/D/E vs A**: Does OPD add value beyond GRPO?

## Key Hypothesis (D vs C)
- C: Teacher generates analysis externally. Analysis improves scoring but student never learns to analyze.
- D: Student reflects in-sequence. Teacher scores reflection tokens too. Student learns BOTH solving and reflecting. Over time, better reflection → richer PI → even stronger teaching signal.

## Key Metrics
- **teacher_kl/mean**: OPD signal strength (added to stdout)
- **Eval Avg@4 on AIME 2025**: OOD generalization
- **Training reward**: Solve rate on training problems

## Run Order (Automated Chain)
1. **C** — running now on 90-problem set (deliberative OPD, exploratory)
2. **D** — auto-launches after C (self-reflection OPD on AIME 2025, core hypothesis)
3. **E** — auto-launches after D (SDPO-style OPD on AIME 2025)
4. **B** — auto-launches after E (answer-only OPD on AIME 2025)
5. **A** — auto-launches after B (GRPO only on AIME 2025)

Automation: `launch-next.sh` (C→D), `launch-chain.sh` in tmux `chain` session (D→E→B→A)

## Dataset Decision
- 90-problem set (aimo-validation-aime): ~80% solve rate — too easy, little room for improvement
- AIME 2025 (30 problems): ~72% solve rate — better difficulty sweet spot
- **Decision**: D/B/A all use AIME 2025. C on 90 problems is exploratory/pilot.
- Config files: `configs/aime/ablation-{A,B}-*-2025.toml`, `configs/aime_mt/ablation-D-self-reflection-2025.toml`

## Results

### Config C: Deliberative OPD on 90 problems (RUNNING, step 22/50)
| Step | AIME 2025 Avg@4 | AIME 2025 Pass@4 | Train Avg@4 | Train Pass@4 |
|------|-----------------|-------------------|-------------|--------------|
| 0    | 0.717           | 0.800             | 0.725       | 0.833        |
| 10   | 0.725           | 0.833             | 0.725       | 0.800        |
| 20   | 0.683           | 0.833             | 0.717       | 0.800        |

- teacher_kl: -0.010 ± 0.003 (consistent negative, deliberative PI working)
- Training reward: ~80% (dataset too easy for Qwen3-8B)
- Entropy: 0.24-0.31 (stable, no divergence)
- Conclusion: Flat/noisy eval — confirms dataset is too easy

### Config D: Self-Reflection OPD on AIME 2025 (NEXT)
(pending — auto-launches after C)

### Config E: SDPO-style OPD on AIME 2025 (QUEUED)
(pending — auto-launches after D)

### Config B: Answer OPD on AIME 2025
(pending — auto-launches after E)

### Config A: GRPO Only on AIME 2025
(pending — auto-launches after B)

## Code Changes
1. `train.py`: Added teacher_kl to stdout step message
2. `utils.py`: Updated DEFAULT_DELIBERATIVE_PROMPT to informed structured format
3. `utils.py`: Deliberative analysis includes answer + solution when available
4. `utils.py`: Correct rollouts skip LLM call (static analysis)
5. `orchestrator.py`: Fixed async prepare_teacher_context handling + TypeError fallback
