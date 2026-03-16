# Reflection-Augmented On-Policy Distillation (RA-OPD)

Self-contained reference for the RA-OPD hypothesis and method. All results from this repository unless otherwise noted.

---

## 1. Method Overview

### Standard OPD (baseline)

On-Policy Distillation (OPD) trains a student model by comparing its per-token predictions against a teacher's. The teacher has access to **privileged information (PI)** — the correct answer, a reference solution, or both — that the student does not see. The per-token logprob difference (teacher - student) drives the training gradient: where the teacher disagrees with the student, the student learns.

Standard OPD token sequence:

```
[system prompt + PI] [user: problem] [assistant: student's solution]
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                      teacher scores these tokens via prefill
```

Loss: `L = adv_tau * GRPO_advantage + teacher_tau * (teacher_logprobs - student_logprobs)`

### RA-OPD (our method)

RA-OPD adds **one extra conversation turn**: after solving, the student reflects on its own work in a structured format. The teacher then scores the **full sequence** — both solution and reflection tokens — with richer PI than the student had.

RA-OPD token sequence:

```
[system prompt] [user: problem] [assistant: student's solution]
                [user: reflection prompt + student PI] [assistant: structured reflection]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                teacher scores with teacher PI (richer than student PI)
```

What makes this work:

1. **Information asymmetry**: The student reflects with limited information (e.g., only told "correct" or "incorrect"). The teacher evaluates the reflection with richer PI (e.g., the correct answer, reference solution, or the student's own reflection content). The gap between what the student knows and what the teacher knows creates learning signal.

2. **Structured format**: The student must commit to specific claims in fixed fields (VERDICT, ERROR_TYPE, ERROR_LOCATION, WHAT_WENT_WRONG, LESSON). This creates tokens the teacher can evaluate precisely. Open-ended reflection fails (see Section 2).

3. **No new loss functions**: The standard OPD loss applies unchanged. The only difference is prompt engineering — appending a reflection turn to the student's rollout before teacher scoring.

4. **Additive signal**: Reflection tokens provide additional learning signal without degrading the signal on solution tokens. The model receives gradient pressure on both *how it solves* and *how it reflects*.

### The structured reflection format

```
VERDICT: correct / incorrect
CONFIDENCE: 1-10
ERROR_TYPE: (e.g., algebraic, conceptual, computational)
ERROR_LOCATION: (which step went wrong)
WHAT_WENT_WRONG: (1-2 sentence diagnosis)
LESSON: (what to do differently)
```

For correct rollouts, all diagnostic fields are "none" by default (can use alternative prompts — see Section 2).

---

## 2. Key Results

All results from AIME 2025 (25 problems, 100 rollouts: 86 incorrect, 14 correct) unless noted. Models: Qwen3-8B and Qwen3-32B. Signal measured via teacher prefill scoring — no training required.

**Metrics**: |KL| = mean |student_logprob - teacher_logprob| per token (gradient magnitude). Cohen's d = effect size separating incorrect from correct rollouts by |KL| (signal selectivity; d > 0.8 is conventionally "large").

### 2.1 Reflection tokens carry 2-7x stronger signal than solution tokens

| Config | d (solution) | d (reflection) | Ratio |
|---|---|---|---|
| 8B self-teacher | 0.54 | 0.85 | 1.6x |
| 32B teacher, 8B reflector | 0.63 | 2.56 | 4.1x |
| 32B self-teacher | 0.62 | 4.67 | 7.5x |

The reflection turn creates a new segment of tokens where the teacher's disagreement with the student is far more selective than on solution tokens.

### 2.2 Student PI scales signal monotonically

More information given to the student before reflecting produces stronger signal on reflection tokens (32B self-teacher, structured format, teacher PI = answer_only):

| Student PI | d (reflection) |
|---|---|
| none (blind) | 1.44 |
| binary ("you're incorrect") | 2.89 |
| answer (told correct answer) | 3.99 |
| answer + hint | 4.67 |

Even blind reflection (d=1.44) exceeds answer-only PI on solution tokens (d=0.63). The student's uncertainty about its own work IS informative — the teacher with PI evaluates this uncertainty.

### 2.3 Best overall configuration: binary -> answer+reflection (d=5.30)

The student gets binary feedback (correct/incorrect — free from reward signal), then reflects. The teacher gets the correct answer plus the student's own reflection as PI.

| Condition | Student PI | Teacher PI | d (solution) | d (reflection) |
|---|---|---|---|---|
| K: binary -> reflection | binary | student's reflection | 1.48 | 5.25 |
| **L: binary -> answer+refl** | binary | answer + student's reflection | **1.48** | **5.30** |
| M: binary -> answer+ref | binary | answer + ref solution | 0.43 | 3.13 |

Key observations:
- The student's own reflection as teacher PI gives d=1.48 on solution tokens — better than answer-only (d=0.63). The reflection is a compressed map of the student's reasoning that the teacher can evaluate.
- Reference solution as teacher PI (d=3.13) is weaker than the student's own reflection (d=5.25). Generic solutions are less informative than rollout-specific self-diagnosis.
- Adding the answer to reflection PI is nearly redundant (5.25 -> 5.30), but the answer is free.

### 2.4 Structured format is essential

Open-ended reflection gives **negative** Cohen's d:

| Teacher PI | Student PI | Open d | Structured d |
|---|---|---|---|
| answer_only | binary | -0.69 to -1.26 | 0.56 to 2.89 |
| answer_ref | answer | -0.12 to -1.51 | 0.60 to 2.90 |

**Why negative d?** Incorrect rollouts trigger substantive freeform reflection ("I made an error at step X"), which aligns with what an informed teacher expects. Correct rollouts trigger bland "I'm confident" text that diverges more from teacher expectations. This inverts the signal: incorrect rollouts get *less* KL divergence than correct ones. Structured format avoids this by constraining both correct and incorrect reflections to the same fields — the teacher evaluates matching structured claims, not text quality.

### 2.5 Replication and cross-domain generalization

**AIME 2024** (30 problems, 120 rollouts: 89 IC, 31 C, 32B self-teacher):

| Condition | AIME 2025 d(refl) | AIME 2024 d(refl) | 95% CI |
|---|---|---|---|
| Blind structured | 1.44 | 1.62 | [1.43, 1.88] |
| Answer structured | 3.99 | 3.65 | [3.08, 4.60] |

**ARC-AGI single-turn** (visual reasoning, 198 rollouts, 50 tasks, 32B self-teacher):

| Segment | Cohen's d | |KL| IC | |KL| C |
|---|---|---|---|
| Solution | -0.09 | 0.093 | 0.097 |
| **Reflection** | **2.29** | **0.114** | **0.044** |

**ARC-AGI multi-turn REPL+Reflect** (30 tasks × 4 rollouts = 120, 32B self-teacher, `arc_agi_reflect` env):

Multi-turn REPL where student writes Python code, debugs iteratively, then SUBMIT triggers structured reflection. 0 exact matches, 99/120 with structured reflection. Teacher PI = expected output grids.

| Split | Sol d | Refl d | Ratio |
|---|---|---|---|
| Format (submitted vs not) | 1.05 | **3.40** | 3.2x |
| Shape match (yes/no) | 0.54 | **1.28** | 2.4x |
| Cell accuracy (median) | 0.50 | **0.76** | 1.5x |

Raw |KL|: solution 0.058, reflection 0.147 (2.5x ratio). Positive correlation between reward and |KL| (r=0.547 on reflection) — opposite of AIME, because better attempts engage more with the transformation the teacher knows.

The core finding replicates across domains and interaction modes: reflection tokens carry 2-3x more signal than solution tokens in both single-turn reasoning (AIME, ARC-AGI single-turn) and multi-turn agentic code execution (ARC-AGI REPL).

### 2.6 Training validation (8B self-teacher, deliberative PI)

Tested with the pre-reflection deliberative PI approach (teacher generates analysis of student's work before scoring):

| Experiment | Heldout Acc | Train Acc | Grad Norms |
|---|---|---|---|
| Deliberative PI (self-teacher) | **+4.3%** | **+6.9%** | **3-5x baseline** |
| Answer-only baseline (self-teacher) | -6.3% | flat | 1x |
| Cross-teacher baseline (32B->8B) | +4.3% | — | — |

The answer-only self-teacher shows zero learning — confirming that the signal from PI (not just model gap) matters. Deliberative PI produces measurable learning where the baseline produces none.

### 2.7 Signal is additive

Adding the reflection turn does NOT degrade solution-token signal. Solution d stays at 0.44-0.63 regardless of what happens in the reflection turn. The teacher independently evaluates both segments.

### 2.8 Signal concentration

The signal is not uniform ("peanut butter") — it is concentrated on specific tokens:

| Segment | Gini Coefficient | Top-10% tokens carry |
|---|---|---|
| Solution tokens (IC) | 0.888 | 80.3% of signal |
| Reflection tokens, answer PI (IC) | 0.766 | 55.8% of signal |
| Reflection tokens, blind (IC) | 0.872 | 78.8% of signal |

### 2.9 Best-of-N reflections (more compute -> more signal)

| Method | Mean |KL| | vs Random |
|---|---|---|
| Random (avg of 4) | 0.0324 | -- |
| Best-of-4 | 0.0413 | +27.3% |
| Worst-of-4 | 0.0249 | -23.2% |

Higher temperature also helps: T=0.9 gives |KL|=0.0368 vs T=0.3 gives |KL|=0.0299.

### 2.10 Correct-rollout reflection design

Alternative prompts extract learning from correct rollouts (default diagnostic produces near-zero signal on correct rollouts):

| Prompt Style | |KL| C (refl) | Cohen's d |
|---|---|---|
| Diagnostic (original, all "none") | 0.0009 | 4.04 |
| Efficiency analysis | 0.0128 | 1.84 |
| Teaching (prerequisites, pitfalls) | 0.0236 | 0.83 |

Tradeoff: richer correct-rollout prompts produce 14-26x more |KL| on correct rollouts but reduce discrimination. Efficiency analysis provides the best balance.

---

## 3. Design Decisions

Each decision is documented in a separate file in `decisions/` (to be created as needed). Summary of rationale:

### PI Placement — `decisions/pi-placement.md`

Where PI is injected in the token sequence matters for deliberative (analytical) PI but not for static PI (answer/reference).

| Placement | Static PI (answer_ref) | Deliberative PI (blind) |
|---|---|---|
| system | 0.065 (d=0.54) | 0.072 (d=0.34) |
| user | 0.073 (d=0.55) | 0.074 (d=0.24) |
| **assistant_prefix** | 0.068 (d=0.54) | **0.087 (d=0.49)** |

For deliberative analysis, `assistant_prefix` is best: +21% |KL| and +44% Cohen's d over system placement. The analysis tokens are in the same "response generation" mode as the rollout tokens, so the teacher's attention flows directly from diagnostic reasoning into scoring.

For RA-OPD (reflection-in-sequence), PI placement is less relevant because the student generates the reflection — the PI is in the prompt preceding the reflection turn, not injected into the scoring context.

### Reflection Format — `decisions/reflection-format.md`

Structured format (VERDICT/ERROR_TYPE/ERROR_LOCATION/WHAT_WENT_WRONG/LESSON) is essential. Open-ended reflection inverts the signal (negative Cohen's d). The structured format works because:

1. Forces commitment to specific evaluable claims
2. Constrains both correct and incorrect rollouts to the same fields (prevents length/style confounds)
3. Produces short, precise text that the teacher can evaluate token-by-token
4. Mirrors the finding that |KL| and discrimination are inversely correlated with text length

### Information Asymmetry — `decisions/info-asymmetry.md`

The gap between student PI and teacher PI drives the learning signal. Recommended configurations:

| Student PI | Teacher PI | d (reflection) | Use Case |
|---|---|---|---|
| binary | answer + reflection | 5.30 | Best overall (binary is free from reward) |
| answer | answer_only | 3.99 | When answer is available |
| none (blind) | answer_ref | 1.44 | No external feedback available |

Binary feedback (correct/incorrect) is the minimum useful student PI — it is free from the reward signal and produces 2x more signal than blind (2.89 vs 1.44).

### Correct-Rollout Handling — `decisions/correct-rollout-handling.md`

Options for extracting signal from correct rollouts:

- **Diagnostic (default)**: All fields = "none", 36 tokens, |KL|=0.0009. Maximizes discrimination (d=4.04) but wastes correct rollouts.
- **Efficiency analysis**: "How could you solve this more efficiently?", 59 tokens, |KL|=0.0128. Best balance: 14x more signal, d=1.84.
- **Teaching**: "Explain prerequisites and common pitfalls", 163 tokens, |KL|=0.0236. Most signal but low discrimination (d=0.83).

Recommendation: Use efficiency analysis for correct rollouts.

---

## 4. Comparison to Related Work

| Method | Paper | PI Source | Signal Type | Key Difference from RA-OPD |
|---|---|---|---|---|
| GKD | Agarwal 2023 | Teacher demonstration | Imitation | Teacher generates tokens; student imitates. RA-OPD has teacher score student's own tokens. |
| SDPO | Hubotter 2026 | Correct peer rollout | Social learning | Uses another rollout as PI, not student self-reflection. PI is external. |
| SDFT | Shenfeld 2026 | Golden response | Imitation | Supervised fine-tuning on teacher demonstrations. No per-token credit assignment. |
| pi-Distill | Penaloza 2026 | Answer + ref solution | Static lookup | PI is fixed per problem, not adaptive to the student's specific reasoning. |
| RLTF | Song 2026 | Feedback prediction loss | Auxiliary objective | Trains reflection via a separate loss function; RA-OPD uses the same OPD loss. |
| User Interactions | Anthropic internal | Hindsight user context | Relabeling | Post-hoc context from real users; RA-OPD uses student-generated reflection. |
| STaR / Quiet-STaR | Zelikman 2022/2024 | Self-generated rationales | Rationalization | Generates reasoning for correct answers; not per-token teacher scoring. |
| **RA-OPD (ours)** | -- | **Student's own structured reflection** | **Self-diagnosis scored by teacher** | **Student generates the signal source; teacher evaluates it with PI advantage. No new loss functions.** |

The fundamental distinction: in all other methods, PI comes from external sources (answers, reference solutions, peer rollouts, user context). In RA-OPD, the student generates the PI source (its own reflection), and the information asymmetry between student and teacher creates the learning signal. The student's self-diagnosis — even when wrong — is informative because the teacher knows the truth.

---

## 5. Architecture and Code

### Core implementation files

| Component | Path |
|---|---|
| OPD loss function | `src/prime_rl/trainer/rl/loss.py` |
| Teacher logprobs + PI injection | `src/prime_rl/orchestrator/utils.py` |
| Orchestrator (env dispatch) | `src/prime_rl/orchestrator/orchestrator.py` |
| AnalyzerConfig (PI settings) | `src/prime_rl/configs/orchestrator.py` |
| AIME teacher context | `environments/aime/src/aime/teacher_context.py` |
| ARC-AGI reflect environment | `environments/arc_agi_reflect/src/arc_agi_reflect/` |
| ARC-AGI reflect teacher context | `environments/arc_agi_reflect/src/arc_agi_reflect/teacher_context.py` |
| Training entry point | `src/prime_rl/entrypoints/rl.py` |

### How the pipeline works

1. **Student generates rollouts** on training problems via vLLM inference.
2. **(Optional) `prepare_teacher_context`** generates adaptive PI per rollout. Each environment implements this function independently (see contract below).
3. **Teacher scores** the same token sequence via prefill (no generation), with PI injected into the prompt.
4. **Loss computation**: `adv_tau * GRPO_advantage + teacher_tau * (teacher_logprobs - student_logprobs)`

### Environment contract

Each environment implements:

```python
# e.g., environments/aime/src/aime/teacher_context.py
async def prepare_teacher_context(
    analyzer_config: AnalyzerConfig,
    rollouts: list[dict]
) -> list[dict]
```

The orchestrator discovers this function via importlib and dispatches per environment. Environments own their PI generation entirely — there is no shared analyzer module.

### Key config: AnalyzerConfig

`AnalyzerConfig.analysis_style` controls the reflection/analysis format. Default is `"structured"`. Options: `structured`, `directive`, `verbose`, `error_point`. The AIME environment selects informed/blind variants based on whether a reference solution is available.

---

## 6. Open Questions

1. **Training validation at scale**: Signal measurement (d=5.30) does not guarantee training improvement. The 8B self-teacher deliberative training showed +4.3% heldout, but reflection-in-sequence training has not been validated. The signal-to-training gap is the most important open question.

2. **Learnable reflection format**: The VERDICT/ERROR_TYPE/... format is hand-designed. Can the model discover a better reflection structure through OPD training on reflection tokens? The gradient pressure on reflection tokens provides exactly this learning signal, but whether it converges to something useful is unknown.

3. **Multi-turn reflection (answered)**: Tested with `arc_agi_reflect` — multi-turn REPL with code execution + structured reflection. Reflection d=3.40 (format split), d=1.28 (shape match), with reflection |KL| 2.5x solution |KL|. Signal pattern generalizes to agentic environments. Interesting finding: positive correlation between reward and |KL| (r=0.547), opposite of AIME — better code attempts create more divergence from the PI-informed teacher.

4. **Test-time scaling via self-analysis loop (partially validated)**: Proxy experiment confirms the prerequisite: 32B blind self-analysis ranks rollouts with rho=0.47 (code_quality vs reward, p<0.0001). Best-of-4 self-selection gives +6.3% over random (oracle: +30.4%). The model can identify its better attempts without ground truth, but the gap to oracle is large. Full gradient-update loop is untested. See `tmp/on-policy-distillation/experiments/arc-signal/results/test_time_ranking_summary.md`.

5. **Cross-model family**: All results are Qwen3 (8B and 32B). Generalization to Llama, Gemma, or other architectures is untested.

6. **Optimal student PI**: Binary feedback (correct/incorrect) is cheap (free from reward signal) and effective (d=2.89 vs blind d=1.44). Is there a sweet spot between binary and full answer that balances signal strength against information leakage during training?

7. **Learning to reflect (virtuous cycle)**: Since reflection tokens receive OPD gradients, the model should learn to reflect more accurately over training. Better reflections create better PI (K/L results show reflection as PI outperforms answer as PI). This predicts a virtuous cycle: reflect -> learn to reflect better -> stronger signal -> faster learning. Whether this cycle actually materializes in training is unvalidated.

8. **Correct-rollout utilization**: The default diagnostic prompt wastes correct rollouts (|KL|=0.0009). Efficiency analysis extracts 14x more signal (|KL|=0.0128) but reduces discrimination. The optimal correct-rollout strategy may depend on the training phase and curriculum.

---

## 7. File Index

### Research documentation

| File | Description |
|---|---|
| `research/on-policy-distillation/hypothesis/README.md` | This document |
| `research/on-policy-distillation/hypothesis/tangents/` | Tangent hypotheses (multi-PI, test-time scaling, success reflection, etc.) |
| `research/on-policy-distillation/experiments/opd-signal/FINDINGS.md` | Full signal measurement results (all PI types, placements, deliberative teaching) |
| `research/on-policy-distillation/experiments/opd-signal/reflection-in-seq-results.md` | Detailed reflection-in-sequence results (all model configs, info asymmetry patterns) |
| `research/on-policy-distillation/experiments/opd-signal/reflection-in-sequence.md` | Experiment design for reflection-in-sequence |
| `research/on-policy-distillation/experiments/opd-signal/paper-outline.md` | Paper outline with abstract draft |
| `research/on-policy-distillation/experiments/training-verification/verification-spike.md` | Training validation results (8B self-teacher experiments A-G) |
| `research/on-policy-distillation/experiments/training-verification/gap-analysis.md` | Analysis of signal-to-training gap |
| `research/on-policy-distillation/experiments/training-verification/pivot-strategy.md` | Strategy for self-teacher pivot |
| `research/on-policy-distillation/research-notes/sdpo-placement-pi-content-results.md` | PI placement and content results (SDPO-style, structured analysis) |
| `research/on-policy-distillation/research-notes/test-time-scaling-idea.md` | Test-time scaling via self-analysis loop |
| `research/on-policy-distillation/research-notes/self-distillation-papers-review.md` | Literature review of self-distillation papers |
| `research/on-policy-distillation/experiments/prime-rl-training-management-guide.md` | Training operations guide |

### Code

| File | Description |
|---|---|
| `src/prime_rl/trainer/rl/loss.py` | OPD loss function |
| `src/prime_rl/orchestrator/utils.py` | Teacher logprob scoring + PI injection |
| `src/prime_rl/orchestrator/orchestrator.py` | Main orchestrator loop, env dispatch |
| `src/prime_rl/configs/orchestrator.py` | AnalyzerConfig (analysis_style, PI settings) |
| `src/prime_rl/entrypoints/rl.py` | Training entry point |
| `environments/aime/src/aime/teacher_context.py` | AIME teacher context (PI generation) |
| `environments/arc_agi_reflect/src/arc_agi_reflect/env.py` | ARC-AGI reflect environment |
| `environments/arc_agi_reflect/src/arc_agi_reflect/teacher_context.py` | ARC-AGI reflect teacher context |

### Configs

| File | Description |
|---|---|
| `configs/aime/verify-self-teacher-deliberative-v2.toml` | Self-teacher with deliberative PI |
| `configs/aime/verify-self-teacher-v2.toml` | Self-teacher baseline (answer-only) |
| `configs/aime/verify-analyzer-v2.toml` | Cross-teacher with analyzer |

### Data (experiment outputs)

| File | Description |
|---|---|
| `tmp/on-policy-distillation/experiments/arc-signal/` | ARC-AGI signal measurement data |
| Results JSON files referenced in `experiments/opd-signal/reflection-in-seq-results.md` | Raw reflection scoring data |

All paths are relative to the repository root (`/home/baris/repos/prime-rl/`).
