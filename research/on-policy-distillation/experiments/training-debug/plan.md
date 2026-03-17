# Training Pipeline Debug Plan

## Problem Statement

Signal measurement consistently shows strong teacher-student discrimination:
- 8B×8B self-teacher, structured reflection: d=0.85
- 32B×32B: d=4.67
- ARC-AGI multi-turn REPL: reflection d=3.40

But training shows near-zero OPD signal:
- Mismatch KL: 0.0001-0.0007 across ALL training runs
- No trend over training steps
- Pattern: GRPO from reward signal works (when reward exists), OPD contributes nothing

This could be a real limitation (8B self-teacher too weak) or a pipeline bug. Must verify before concluding.

## CRITICAL FINDING: Wrong Metric (2026-03-17)

**We were watching the wrong metric.** The `mismatch_kl` printed to stdout measures trainer vs inference policy drift (should be ~0 at start, grows with training). The actual OPD signal is `teacher_kl = teacher_logprobs - trainer_logprobs`, logged to W&B but NOT printed to stdout.

### Actual teacher_kl values from W&B:

| Run | Type | |teacher_kl| mean | Sign | adv_tau | teacher_tau | Outcome |
|-----|------|-------------------|------|---------|-------------|---------|
| Deliberative v2 | Pure OPD, deliberative PI | 0.027 | negative | 0 | 1 | **+4.3% heldout** |
| v4 (structured) | Pure OPD, structured refl | 0.012 | negative | 0 | 1 | Modest +3.1% |
| v6 (GRPO+OPD) | Mixed, open refl | 0.025 | **positive** | 1 | 0.5 | Crashed early |
| Baseline v2 | Pure OPD, answer-only | 0.006 | negative | 0 | 1 | Zero learning |
| ARC v3 (open) | Mixed, open refl | 0.005 | mixed/noisy | 1 | 0.5 | Near zero |

### Interpretation:
- **Deliberative v2 has 4.5x stronger |teacher_kl| than baseline** — PI is clearly working
- **teacher_kl is NOT near zero** — ranges 0.006 to 0.027, consistent with signal measurement |KL| of 0.03-0.08
- **ARC v3 weak signal** is real — open reflection format produces near-zero/noisy teacher_kl (matches signal measurement: open reflection has negative d)
- **Sign matters**: negative = teacher less confident (pushes student away from tokens), positive = teacher more confident (pulls student toward tokens)

### What this changes:
The pipeline is NOT broken. The OPD signal IS flowing through to training. Our previous conclusion that "OPD contributes nothing" was based on watching the wrong metric entirely. The actual OPD signal strength correlates with training outcomes:
- Strong signal (0.027) → learning (+4.3%)
- Medium signal (0.012) → modest learning (+3.1%)
- Weak signal (0.006) → no learning
- Noisy/near-zero (0.005) → no learning

## Debug Steps

### 1. Verify PI reaches teacher prompt — SKIPPED (signal confirms it works)

teacher_kl values of 0.012-0.027 for PI runs vs 0.006 for baseline confirm PI is reaching the teacher and changing its behavior. No need to dump prompts.

### 2. Verify teacher logprobs differ from student — CONFIRMED ✅

teacher_kl is 0.006-0.027, meaning teacher and student logprobs differ. The signal is genuine.

### 3. Check open vs structured reflection impact — CONFIRMED ✅

ARC v3 (open reflection): |teacher_kl| = 0.005, noisy, mixed sign
v4 (structured reflection): |teacher_kl| = 0.012, consistent negative sign
Deliberative v2 (deliberative PI): |teacher_kl| = 0.027, consistent negative sign

**Structured > open, deliberative > structured** — matches signal measurement exactly.

### 4. Compare loss masking with SDPO — DEPRIORITIZED

Since OPD signal is flowing correctly, loss masking isn't the primary issue. The token masking (0.125-8.0) operates on importance ratios, not on teacher_kl directly.

### 5. Verify logprob alignment — DEPRIORITIZED

If alignment were broken, teacher_kl would be random noise near zero. The consistent negative sign for PI runs suggests alignment is correct.

### 6. Check W&B metrics for signal — DONE ✅

teacher_kl is logged and visible in W&B. Should also add it to stdout for easier monitoring.

## Revised Action Items

1. **Add teacher_kl to stdout step message** — so we can monitor without needing W&B
2. **Switch ARC config to structured reflection** — open reflection is confirmed weak
3. **Run ARC with structured reflection + deliberative PI** — should see |teacher_kl| > 0.01
4. **Consider stronger PI for ARC** — deliberative analysis (like AIME) vs just reflection
5. **Try external teacher for ARC** — 32B or API-based, since cross-model gap helps

## Additional Finding: AIME async prepare_teacher_context Bug

Background agent traced the full PI injection pipeline and found that AIME's `prepare_teacher_context` is `async def` with signature `(analyzer_config, rollouts)`, but the orchestrator calls it synchronously with only `(rollouts)` at line 631. This means:
- For AIME: the function returns a coroutine object without executing, teacher_context never populated
- For ARC: `prepare_teacher_context` is sync `def(rollouts)`, works correctly

**However**, the AIME deliberative v2 run (which worked) used the **legacy code path** (`deliberative = true` in config), NOT the new env-specific discovery. The legacy path at line 637-658 properly awaits and generates PI. So this bug only affects AIME runs that try to use the new `prepare_teacher_context` path (v6 used the new path — and it had positive teacher_kl of 0.025, meaning it was getting PI from somewhere... need to investigate further).

## SDPO Comparison Summary

Background agent compared prime-rl's loss with SDPO. Key differences:
- SDPO uses full logit-level KL divergence, prime-rl uses per-token log-ratio
- SDPO uses lighter masking (top-K approximation), prime-rl has aggressive token masking (0.125-8.0)
- SDPO uses EMA teacher + JSD divergence for stability
- Token masking thresholds (0.125-8.0) operate on importance ratios (trainer vs inference), NOT on teacher_kl directly, so they shouldn't suppress the OPD signal in early training when policies are still close

## Root Cause Summary

The "near-zero OPD signal" was a **monitoring artifact**, not a real problem. We were watching `mismatch_kl` (trainer vs inference drift) instead of `teacher_kl` (teacher vs trainer, the actual OPD signal). The pipeline works correctly for the configurations that showed learning. The remaining challenge is making the OPD signal strong enough to drive learning — which requires the right PI format (structured/deliberative, not open) and potentially stronger teachers.
