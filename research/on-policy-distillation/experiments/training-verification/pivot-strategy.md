# Adaptive PI: Pivot Strategy

## Current Status

Training Experiment E (adaptive PI via Gemini 3 Flash on AIME 2025) is running. Early signs:
- Step 0: all rewards=1.0, PI generation skipped (correct rollouts don't need adaptive PI)
- AIME 2025 may be too easy for 8B student to show meaningful improvement
- Mismatch KL flat at 0.0007 (same as baseline)

## Core Hypothesis

Standard OPD provides static PI (answer + ref solution). Adaptive PI tailors hints to the specific mistakes in each student rollout. This should produce stronger, more targeted KL divergence — the teacher's logprobs shift more where the student actually struggles.

## Decision Criteria (after 20 steps)

**Success signals:**
- Mismatch KL rises faster than baseline (0.0009 at step 20)
- Eval accuracy delta > baseline's +3-4%
- W&B samples show teacher_context that's actually adaptive (different per rollout)

**Failure signals:**
- Mismatch KL identical to baseline
- No eval improvement
- All training rewards ~1.0 (too easy)

## Pivot Options (ordered by proximity)

### Pivot 1: Harder AIME filtering
- Filter out problems where 8B gets >75% correct across rollouts
- Keep only problems where student genuinely struggles
- Pro: Minimal code change, same env
- Con: May reduce training set too much (22 → ~10 problems)

### Pivot 2: ARC-AGI
- Already has env + teacher_context infrastructure
- Much harder for LLMs — base accuracy ~20-40%
- Would need `prepare_teacher_context` for arc_agi env
- Pro: Hard problems with clear ground truth, visual reasoning
- Con: Different modality, may need different PI strategy

### Pivot 3: Harder math benchmarks
- MATH-500 (Hendrycks), AMC/USAMO, or competition math datasets
- More problems available, better difficulty calibration
- Pro: Same domain as AIME, easier to compare
- Con: May need new env implementation

### Pivot 4: Non-verifiable tasks (NEW ANGLE)
- **Key insight**: For verifiable tasks (math), the answer itself is already strong PI. Adaptive PI adds marginal value over answer_only. The real value of adaptive PI may be on tasks where ground truth isn't a simple answer.
- **Examples:**
  - Code generation (correctness is verifiable but solution paths are many)
  - Open-ended reasoning / writing (no single correct answer)
  - Tasks where reference solution quality varies
- **PI alternatives without ground truth:**
  - Pairwise comparison: generate multiple rollouts, compare quality
  - Self-reflection: model critiques its own work
  - Preference model: learned reward model provides signal
  - Consensus: multiple models agree/disagree on approach
- **Why this matters for a paper:** "Adaptive PI for non-verifiable tasks" is a much more novel contribution than "adaptive PI for math" where answer_only already works well
- Pro: Novel, potentially high impact, addresses real limitation of OPD
- Con: Requires new environments, evaluation methodology, may be scope-creepy

## Gap Analysis (why signal measurement ≠ training)

See `gap-analysis.md` for full analysis. Key issues:

1. **Cross-model gap dominates**: 32B teacher provides 80% of signal from model difference alone. PI variation is marginal on top. **Self-teacher (8B→8B) isolates the PI effect.**
2. **We're not doing deliberative teaching**: Signal measurement had the teacher itself reasoning about the student's work. Current setup uses Gemini to generate generic problem hints — fundamentally different.
3. **PI placement**: System prompt (current) vs assistant_prefix (optimal for analytical PI: +21% |KL|, +44% Cohen's d)
4. **Correct rollout skipping**: Deliberative PI has signal for correct rollouts too.

**Priority order**: Self-teacher first (isolate PI effect) → exact reproduction of signal setup → placement fix → skip policy fix.

## Implementation Notes

- `prepare_teacher_context` contract already supports any env — just implement the function
- AnalyzerConfig (litellm routing) works for any external LLM
- The correct-rollout-skipping pattern generalizes: only spend compute on rollouts that need help
- For self-teacher deliberative: can use the teacher model itself (via litellm openai/ prefix to local vLLM) instead of Gemini
