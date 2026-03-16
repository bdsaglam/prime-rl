# Gap Analysis: Signal Measurement vs Training Results

## Resolution (2026-03-12)

**Hypothesis #1 (cross-model gap dominates) confirmed.** Self-teacher experiments prove it:

| Experiment | Teacher | PI | Heldout Δ | Train Δ | Grad Norms |
|------------|---------|-----|-----------|---------|------------|
| C: Cross-teacher baseline | 32B | answer_only | +4.3% | +3.4% | 0.04-0.05 |
| D: Cross-teacher deliberative | 32B | deliberative | inconclusive | inconclusive | 0.06-0.08 |
| F: Self-teacher baseline | 8B | answer_only | **-6.3%** | **0%** | 0.006-0.014 |
| G: Self-teacher deliberative | 8B | deliberative | **+4.3%** | **+6.9%** | **0.031-0.052** |

Self-teacher baseline (F) = null baseline (no learning). Deliberative self-teacher (G) = **learning where none existed**. The PI signal IS real — it was just invisible in cross-teacher because the model gap dominates.

Full results: `verification-spike.md`

---

## Original Problem (pre-resolution)

Signal measurement (FINDINGS.md) showed blind deliberative |KL| = 0.075 beats answer_ref |KL| = 0.063 by 19%. But in training:
- Baseline v2 (answer_only): mismatch KL = 0.0007 → 0.0009 over 20 steps
- Adaptive PI (Experiment E): mismatch KL = 0.0007, flat after 2 steps

Why isn't the stronger signal translating to faster/better learning?

## Possible Explanations

### 1. Cross-model gap dominates (most likely)

Signal measurement finding #1: "The cross-model gap (0.097) dominates. PI adds only marginal signal on top (max +0.022)."

In our training setup (8B student, 32B teacher), the cross-model gap accounts for ~80% of total signal. PI variation (answer_only vs adaptive hints) contributes at most +0.022 on top of 0.097. This is a ~2% relative difference in total |KL|.

**The mismatch KL we're measuring in training (0.0007) may already be overwhelmed by the cross-model gap, making PI variations invisible in this metric.**

To test this: run self-teacher OPD (8B→8B) where PI is the ONLY source of signal. In self-teacher, no_pi gives |KL|≈0.005, while adaptive PI should give 0.075 — a 15x difference.

### 2. We're NOT doing deliberative teaching

The signal measurement tested **deliberative teaching**: the teacher reads the student's rollout, generates an analysis of the student's specific reasoning, then uses that analysis as PI when scoring.

Current Experiment E does something different: an **external model** (Gemini 3 Flash) generates "problem hints/notes" informed by the student's attempt. This is:
- A different model (Gemini, not Qwen3)
- A different prompt (problem notes, not reasoning analysis)
- A different output (hints about the problem, not analysis of the student's process)

The original signal measurement had the teacher itself reasoning about the student's work. That's key — the teacher's own understanding of the student's reasoning path is what creates the better credit assignment.

### 3. PI placement is suboptimal

FINDINGS showed assistant_prefix >> system for deliberative PI:
- system: |KL|=0.072, Cohen's d=0.34
- assistant_prefix: |KL|=0.087, Cohen's d=0.49

Current training uses system placement. For static PI this doesn't matter much, but for analytical PI it's a 21% |KL| loss.

### 4. Correct rollout skipping may lose signal

We skip LLM calls for reward=1 rollouts. But deliberative PI creates signal even for correct rollouts — the teacher can still identify inefficiencies, redundancies, or near-misses in correct solutions. By skipping, we lose this signal for ~50-100% of rollouts (step 0 had reward=1.0).

### 5. The "adaptive hints" prompt isn't producing the right kind of PI

The v4 prompt generates structured problem notes (solution sketch, pitfalls, alternatives). This is similar to **answer_ref** with extra commentary — not like the deliberative analysis from signal measurement.

The deliberative analysis was free-form reasoning about the student's specific approach:
- "The student started correctly but made an error at step 3..."
- "The key insight they missed is..."
- "Tokens 45-60 are where the reasoning breaks down..."

Our current prompt produces generic problem notes that happen to be informed by the attempt.

## Recommendations (ordered by expected impact)

### A. Switch to self-teacher (8B→8B) — isolate PI effect
- Removes cross-model gap (80% of signal)
- PI becomes the ONLY source of teaching signal
- If adaptive PI > answer_only in self-teacher training, hypothesis confirmed
- Fastest to test: just change teacher model config

### B. Match the signal measurement setup exactly
- Use the SAME model (Qwen3-8B or 32B) as analyzer, not Gemini
- Use the SAME prompt as FINDINGS (blind deliberative: read rollout, analyze reasoning)
- Place analysis in assistant_prefix, not system prompt
- This is the closest reproduction of what we measured

### C. Don't skip correct rollouts
- Deliberative PI has signal for correct rollouts too
- Only skip if the rollout has very high reward AND no interesting reasoning patterns

### D. Use assistant_prefix placement for analytical PI
- Requires changes to `build_teacher_prompt_ids()` in utils.py
- Only matters for analytical PI (not static answer_only)

### E. Measure the actual |KL| we're producing
- Log per-step mean |KL| between student and teacher logprobs
- Compare with signal measurement predictions
- If |KL| matches predictions but mismatch_kl doesn't move → learning rate / optimization issue
- If |KL| is lower than predicted → PI generation or placement issue

## Priority

**Do A first** (self-teacher). It's the cleanest test of the hypothesis. If adaptive PI beats answer_only in 8B self-teacher training, the method works and we know the cross-model gap was masking the effect.

If A shows no improvement, do B (exact reproduction of signal measurement setup) to rule out prompt/model differences.

If B also shows no improvement, the signal doesn't translate to training gains — time to investigate why (is |KL| actually higher but not helping? Is Cohen's d the real predictor of training success? etc.)
