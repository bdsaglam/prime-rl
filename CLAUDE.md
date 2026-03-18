# prime-rl: Self-Reflection On-Policy Distillation

Fork of [PrimeIntellect/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Core Research Hypothesis

Student self-reflection creates a virtuous learning cycle. Student solves → gets feedback → reflects in structured format → teacher (same model, with PI) scores the FULL sequence including reflection tokens. By teaching on both solution AND reflection tokens, the student learns to solve better AND reflect better. Better reflection → richer PI → stronger teaching signal → self-improving loop.

**This is different from deliberative OPD** where an external analysis is generated and discarded after scoring. In self-reflection OPD, the student's own reflection becomes part of what's learned.

## Current Task: AIME Ablation Study

**Plan**: `research/on-policy-distillation/experiments/aime-ablation/README.md`

**Ablation conditions (all on AIME 2025 except C on 90-problem set):**
- **A**: GRPO only (baseline) — `configs/aime/ablation-A-grpo-only-2025.toml`
- **B**: GRPO + answer-only OPD (static PI from dataset) — `configs/aime/ablation-B-answer-opd-2025.toml`
- **C**: GRPO + deliberative OPD (teacher generates analysis) — `configs/aime/ablation-C-deliberative-opd.toml`
- **D**: GRPO + self-reflection OPD (student reflects in-sequence) — `configs/aime_mt/ablation-D-self-reflection-2025.toml`
- **E**: GRPO + SDPO-style OPD (correct sibling + answer, no student attempt) — `configs/aime/ablation-E-sdpo-pi-2025.toml`

**Status**: Run C at step 22/50 (flat results, dataset too easy). Automated chain: C→D→E→B→A.
- D/E/B/A all use AIME 2025 (30 harder problems). C used 90-problem set (exploratory).
- Automation: `launch-next.sh` (C→D), tmux `chain` session runs `launch-chain.sh` (D→E→B→A)
- E uses `aime_sdpo` env (`environments/aime_sdpo/`) with SDPO-style PI (correct sibling, no student attempt)

## CRITICAL: mismatch_kl ≠ teacher_kl

- `mismatch_kl` (in stdout) = trainer vs inference drift. Always ~0.0007. **Irrelevant to OPD.**
- `teacher_kl` (now in stdout + W&B) = `teacher_logprobs - trainer_logprobs`. **This is the actual OPD signal.**
- The pipeline is NOT broken. teacher_kl ranges from 0.006-0.027 and correlates with training outcomes.
- Deliberative v2 (teacher_kl=0.027) → +4.3% heldout. Baseline (teacher_kl=0.006) → zero learning.
- Full debug writeup: `research/on-policy-distillation/experiments/training-debug/plan.md`

## How OPD Works

1. Student generates rollouts (single-turn for `aime`, multi-turn with reflection for `aime_mt`)
2. `prepare_teacher_context` assembles PI per rollout (per-env, discovered via importlib)
3. Teacher scores same token sequence via prefill, PI injected into first user message
4. Loss: `adv_tau * advantages + teacher_tau * (teacher_logprobs - trainer_logprobs)`
5. teacher_kl is logged to W&B and stdout

**Two code paths for PI generation:**
- **Env-specific**: `prepare_teacher_context(rollouts)` in env module (sync for aime_mt, broken async for aime)
- **Legacy deliberative**: `deliberative = true` in teacher_model config → `generate_deliberative_analyses()` in `utils.py`

Key code:
- Loss: `src/prime_rl/trainer/rl/loss.py` (line 116: `teacher_kl = teacher_logprobs - trainer_logprobs`)
- PI injection: `src/prime_rl/orchestrator/utils.py` (`build_teacher_prompt_ids`, `compute_teacher_logprobs`)
- Self-reflection env: `environments/aime_mt/src/aime_mt/env.py` (2-turn: solve + reflect)
- Self-reflection PI: `environments/aime_mt/src/aime_mt/env.py` → `prepare_teacher_context()`

## Signal Measurement Results

- 8B×8B self-teacher, structured reflection: d=0.85
- 32B×32B, structured: d=4.67
- Open reflection: NEGATIVE d → must use structured format
- Reflection tokens carry 2-7x stronger signal than solution tokens
- Full results: `research/on-policy-distillation/experiments/opd-signal/reflection-in-seq-results.md`

## Key Commands

```bash
# Training
python -m prime_rl.entrypoints.rl @ configs/aime/ablation-C-deliberative-opd.toml

# Monitor
tail -F outputs/<run>/logs/trainer.stdout   # teacher_kl visible
tail -F outputs/<run>/logs/orchestrator.stdout  # eval results, reward

# W&B metrics
python research/on-policy-distillation/experiments/aime-ablation/analyze.py
```

## Training Lessons

- **Qwen3**: `chat_template_kwargs = {enable_thinking = false}`
- **OOM with 32K**: `fused_lm_head_chunk_size = 8192`
- **32B OOM on 4xA100 80GB**: Infeasible
- **Health check**: teacher_kl should be non-zero; entropy > 0.5 = diverging
- **Kill zombie vLLM** before re-launch
- **aimo-validation-aime** (90 problems): ~80% solve rate — may be too easy for 8B
- **AIME 2025** (30 problems): ~72% solve rate — harder, standard benchmark
