# Reflection-in-Sequence: Signal Measurement Results

## Setup

Multi-turn sequence: student solves → user asks for reflection (with varying PI) → student reflects. Teacher (with richer PI) scores the full sequence via prefill. We measure |KL| and Cohen's d separately on solution tokens and reflection tokens.

**Independent variables:**
- Student PI in reflection prompt: none, binary ("incorrect"), answer, answer+hint
- Reflection format: open (freeform) vs structured (VERDICT/CONFIDENCE/ERROR_TYPE/...)
- Teacher PI: no_pi, answer_only, answer_ref
- Reflector model: 8B or 32B (generates the reflection)
- Teacher model: 8B or 32B (scores the sequence)

**Data:** 100 rollouts (86 incorrect, 14 correct), 25 AIME 2025 problems.

## Key Result: Massive Signal on Reflection Tokens

Reflection tokens carry dramatically stronger discrimination signal than solution tokens when using structured format:

### Best Cohen's d on reflection tokens (per config)

| Teacher | Reflector | Best Condition | d (reflection) | d (solution) | Ratio |
|---|---|---|---|---|---|
| 8B | 8B | answer_ref / none__structured | 0.85 | 0.41 | 2.1x |
| 8B | 32B | answer_ref / answer__structured | 1.42 | 0.44 | 3.2x |
| 32B | 8B | answer_ref / answer__structured | 2.56 | 0.44 | 5.8x |
| **32B** | **32B** | **answer_only / answer_hint__structured** | **4.67** | 0.62 | **7.5x** |

For comparison, our previous best signal was d=2.23 (structured analysis as PI for solution tokens). Reflection tokens reach d=4.67 — more than double.

### The structured format is essential

Open-ended reflection produces **negative** Cohen's d in most conditions:

| Teacher PI | Student PI | Open d | Structured d | Gap |
|---|---|---|---|---|
| answer_only | binary | -0.69 to -1.26 | 0.56 to 2.89 | +1.25 to +4.15 |
| answer_ref | answer | -0.12 to -1.51 | 0.60 to 2.90 | +0.72 to +4.41 |
| answer_ref | answer_hint | -1.07 to -2.18 | 0.12 to 3.57 | +1.19 to +5.75 |

**Why negative d on open reflection?** Incorrect rollouts trigger substantive freeform reflection ("I made an error at step X"), which aligns well with what an informed teacher expects. Correct rollouts trigger bland "I'm confident" responses that are less distinctive. The teacher with PI sees *less* divergence from incorrect reflections, inverting the signal. Structured format avoids this by constraining both correct and incorrect reflections to the same fields.

### Info asymmetry patterns

**Student PI = none (blind reflection), structured format:**

| Teacher PI | 8B×8B | 32B×8B | 32B×32B |
|---|---|---|---|
| answer_only | 0.24 | -0.01 | 1.44 |
| answer_ref | 0.85 | 1.24 | 3.55 |

**Student PI = binary ("you're incorrect"), structured format:**

| Teacher PI | 8B×8B | 32B×8B | 32B×32B |
|---|---|---|---|
| answer_only | -1.13 | 0.56 | 2.89 |
| answer_ref | -0.02 | 1.56 | 3.13 |

**Student PI = answer (told correct answer), structured format:**

| Teacher PI | 8B×8B | 32B×8B | 32B×32B |
|---|---|---|---|
| answer_only | -1.28 | 0.93 | 3.99 |
| answer_ref | 0.60 | 2.56 | 2.90 |

**Student PI = answer+hint, structured format:**

| Teacher PI | 8B×8B | 32B×8B | 32B×32B |
|---|---|---|---|
| answer_only | -1.28 | -0.24 | 4.67 |
| answer_ref | -0.13 | 0.48 | 3.57 |

### Key patterns

1. **32B reflector >> 8B reflector** for discrimination (32B×32B d=4.67 vs 32B×8B d=2.56). A more capable reflector produces reflections that are more informative for discriminating correct vs incorrect.

2. **32B teacher >> 8B teacher**. 32B teacher with answer_only is often sufficient — adding answer_ref sometimes reduces d (e.g., 32B×32B answer__structured: answer_only d=3.99, answer_ref d=2.90).

3. **More student PI → more signal (with structured format).** For 32B×32B: none d=1.44, binary d=2.89, answer d=3.99, answer_hint d=4.67. Giving the student more info makes its structured reflection more informative about its understanding.

4. **answer_only teacher PI is surprisingly strong with 32B.** The 32B teacher only needs the answer to evaluate structured reflections effectively. The structured format makes the reflection so precise that the answer alone provides sufficient PI.

5. **Solution tokens are unaffected by reflection.** Solution d stays at ~0.44-0.63 regardless of what happens in the reflection turn. The reflection is purely additive signal.

## Implications for Training

### Reflection tokens as a free lunch

Adding a reflection turn provides additional trainable tokens with 2-7x stronger signal than solution tokens, without degrading solution signal. In self-OPD (same model for teacher and student), the model learns to both solve AND reflect better.

### Optimal configuration for self-OPD

For a self-teacher setup (same model), the recommended config is:
- **Student PI**: answer (tells the student the correct answer before reflecting)
- **Reflection format**: structured (VERDICT/CONFIDENCE/ERROR_TYPE/ERROR_LOCATION/WHAT_WENT_WRONG/CORRECTION)
- **Teacher PI**: answer_only or answer_ref (answer may suffice)
- This achieves d=3.99 (answer_only teacher) on reflection tokens with 32B self-teacher

### The "learning to reflect" mechanism works

Since reflection tokens get strong OPD signal, the model receives direct gradient pressure on HOW it reflects. Over training, the model should learn to produce more accurate structured reflections — e.g., correctly classifying error types, pinpointing error locations. This is "learning to reflect" without any auxiliary loss.

### Correct rollout reflections: prompt design tradeoff

Current diagnostic prompts produce "none" for all fields on correct rollouts (36 tokens, |KL|=0.0009). We tested 5 alternative prompts for correct rollouts:

| Prompt Style | |KL| C (refl) | N tokens | Cohen's d |
|---|---|---|---|
| v1 diagnostic (original) | 0.0009 | 36 | 4.04 |
| v5 blind diagnostic | 0.0019 | 36 | 3.82 |
| v3 efficiency analysis | 0.0128 | 59 | 1.84 |
| v2 meta-analysis | 0.0220 | 139 | 1.03 |
| v4 teaching | 0.0236 | 163 | 0.83 |

**Key tradeoff**: Richer correct-rollout reflections (v2, v4) generate 4-5x more trainable tokens with 26x more |KL|, but reduce Cohen's d because correct and incorrect reflections become more similar. The teaching prompt (v4) asks for prerequisites, common pitfalls, verification methods — substantive content that the teacher can evaluate.

**Recommendation**: Use v3 (efficiency) as the best balance — 14x more correct signal than v1 while maintaining strong d (1.84). For pure discrimination, keep v1.

## Additional Results

### Best-of-4 Reflections (Bitter Lesson: More Compute → More Signal)

| Method | Mean |KL| | vs Random |
|---|---|---|
| Random (avg of 4) | 0.0324 | — |
| Best-of-4 | 0.0413 | +27.3% |
| Worst-of-4 | 0.0249 | -23.2% |

Higher temperature also helps: T=0.9 gives |KL|=0.0368 vs T=0.3 gives |KL|=0.0299.

### Combined Reflection + Structured Analysis (32B self-teacher)

When the teacher gets structured analysis as PI AND scores the reflection:

| Teacher PI | d (solution) | d (reflection) |
|---|---|---|
| answer_only | 0.62 | 0.72 |
| answer_ref | 0.44 | 2.41 |
| structured_analysis | **1.58** | **1.90** |

Structured analysis as teacher PI gives strong signal on BOTH segments. The teacher with full error analysis can evaluate both the solution AND the reflection effectively.

### Blind Verdict Accuracy (Student Self-Assessment)

| Condition | IC Detection | C Detection |
|---|---|---|
| 32B blind (none) | 43% | 100% |
| 32B binary (told wrong) | 100% | 100% |
| 8B blind (none) | 67% | 100% |
| 32B answer (told answer) | 100% | 100% |

The blind student only detects its own errors 43-67% of the time. The teacher (with PI) always knows. This asymmetry drives the learning signal on reflection tokens.

### Generalization: AIME 2024

Tested key conditions on AIME 2024 (30 problems, 120 8B rollouts: 89 incorrect, 31 correct) with 32B self-teacher:

| Condition | AIME 2025 d(refl) | AIME 2024 d(refl) |
|---|---|---|
| Blind (none) structured | 1.44 | **1.62** |
| Answer structured | 3.99 | **3.65** |

**Both key findings replicate on a different dataset.** Blind structured reflection (d=1.62) and answer-informed reflection (d=3.65) show consistent signal amplification, confirming these are not dataset-specific artifacts.

### Cross-Domain: ARC-AGI Multi-Turn REPL+Reflect

Tested on ARC-AGI visual reasoning tasks using the `arc_agi_reflect` environment — a multi-turn REPL where the student writes Python code to transform grids, then SUBMIT triggers a structured reflection step.

**Setup:** 30 ARC-Prize-2025 training problems × 4 rollouts = 120 rollouts. Qwen3-32B self-teacher, max 10 turns. Structured reflection (VERDICT/ERROR_TYPE/ERROR_LOCATION/WHAT_WENT_WRONG/LESSON). Teacher PI = expected output grids. 0 exact matches, 99/120 with structured reflection.

**Key difference from AIME:** Multi-turn REPL (code execution + iterative debugging) vs single-turn math reasoning. No correct rollouts, so Cohen's d uses reward-based splits instead of binary correct/incorrect.

| Split | N_hi / N_lo | Sol d | Refl d | Ratio |
|---|---|---|---|---|
| Format (submitted vs not) | 99 / 21 | 1.05 | **3.40** | 3.2x |
| Shape match (yes/no) | 84 / 36 | 0.54 | **1.28** | 2.4x |
| Cell accuracy (median) | 60 / 60 | 0.50 | **0.76** | 1.5x |

**Raw signal:**
- Solution |KL| = 0.058, Reflection |KL| = 0.147 (2.5x ratio)
- Control (no PI): |KL| ≈ 0 on both segments — signal is entirely PI-driven
- Solution Gini = 0.918 (concentrated signal), Reflection Gini = 0.835 (more distributed)

**Inverted correlation (opposite of AIME):**
- Reward vs |KL|: r=0.271 (solution), r=0.547 (reflection) — both **positive**
- Better rollouts have MORE |KL| divergence, not less
- This is because the teacher PI (expected grids) is most relevant when the student's code engages with the correct transformation — closer attempts create more divergence from the PI-informed teacher

**Comparison with single-turn ARC-AGI:**
- Single-turn (binary correct/incorrect): d=2.29 on solution
- Multi-turn REPL+Reflect: d=3.40 on reflection (format split), d=1.28 (shape match)
- Note: not directly comparable since single-turn has actual correct rollouts for the split

**Key finding:** The RA-OPD signal pattern generalizes to multi-turn agentic environments with code execution. Reflection tokens carry 2-3x more signal than solution tokens even in REPL settings, confirming this is not an artifact of single-turn math reasoning.

## Raw Data Files

- `results/reflection_in_seq_8b.json` — 8B model reflections (800 reflections)
- `results/reflection_in_seq_32b.json` — 32B model reflections (800 reflections)
- `results/reflection_scores_8bt_8br.json` — 8B teacher × 8B reflector scoring
- `results/reflection_scores_8bt_32br.json` — 8B teacher × 32B reflector scoring
- `results/reflection_scores_32bt_8br.json` — 32B teacher × 8B reflector scoring
- `results/reflection_scores_32bt_32br.json` — 32B teacher × 32B reflector scoring
- `tmp/on-policy-distillation/experiments/arc-signal/results/arc_signal_repl_32b.json` — ARC-AGI multi-turn REPL+Reflect signal (120 rollouts, 32B self-teacher)
- `tmp/on-policy-distillation/experiments/arc-signal/outputs/evals/arc_agi_reflect--Qwen--Qwen3-32B/fc491a75/` — Raw eval output (post-fix, with structured reflection)
