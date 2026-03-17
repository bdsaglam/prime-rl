# prime-rl: Reflection-Augmented On-Policy Distillation

Fork of [PrimeIntellect/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Core Research Hypothesis

Agent trajectories contain rich learning signal extractable through reflection. Analogous to human introspection: we reflect on experiences (successes and failures), distill lessons, and update behavior — even without external reward. Our method: student solves → gets feedback → reflects in structured format → teacher (with PI) scores the full sequence including reflection tokens. The reflection tokens carry 2-7x stronger OPD signal than solution tokens.

## Current Task: Debug Training Pipeline

**Problem**: Signal measurement shows strong discrimination (d=0.85-4.67 on reflection tokens) but training shows near-zero mismatch KL (~0.0003). Something is broken between measurement and training.

**Debugging plan**: `research/on-policy-distillation/experiments/training-debug/plan.md`

**Key suspects**:
1. PI not reaching teacher (verify teacher prompts contain PI)
2. Teacher logprobs not different from student (verify per-token KL on known-good batches)
3. Loss masking killing weak signal (compare with SDPO implementation)
4. Open reflection format (NEGATIVE d) — must use structured

## How OPD Works

1. Student generates rollouts
2. `prepare_teacher_context` assembles PI per rollout (env-specific)
3. Teacher scores same token sequence via prefill, PI injected into first user message
4. Loss = `adv_tau * GRPO_advantage + teacher_tau * (teacher_logprobs - student_logprobs)`

Key code:
- Loss: `src/prime_rl/trainer/rl/loss.py`
- Teacher logprobs + PI injection: `src/prime_rl/orchestrator/utils.py` (`build_teacher_prompt_ids`, `compute_teacher_logprobs`)
- Orchestrator: `src/prime_rl/orchestrator/orchestrator.py`
- PI contract: `def prepare_teacher_context(rollouts: list[dict]) -> None` (per env, discovered via importlib)

## Key Commands

```bash
# Training
python -m prime_rl.entrypoints.rl @ configs/aime/opd-self-teacher-8b.toml

# Evaluation
prime eval run aime -a '{"dataset_name":"aime2025"}' -n 8 -r 4 -m MODEL -b URL -t MAX_TOKENS -T 0.6 --skip-upload -d

# Logs
tail -F outputs/<run>/logs/orchestrator.stdout
```

## Training Lessons

- **Qwen3**: `chat_template_kwargs = {enable_thinking = false}`
- **OOM with 32K**: `fused_lm_head_chunk_size = 8192`
- **ARC sandbox**: `timeout = 30` to prevent infinite loop hangs
- **32B OOM on 4xA100 80GB**: Infeasible (77GB/shard with TP=2)
- **Health check**: truncation + mismatch_kl in first 10 steps; entropy > 0.5 = diverging
- **Kill zombie vLLM** before re-launch

## Documentation Index

- **Signal measurement results**: `research/on-policy-distillation/experiments/opd-signal/reflection-in-seq-results.md`
- **Training experiments**: `research/on-policy-distillation/experiments/training-verification/`
- **Training debug plan**: `research/on-policy-distillation/experiments/training-debug/plan.md`
- **Full research index**: `research/on-policy-distillation/README.md`
- **Detailed findings & history**: See memory files in `.claude/projects/*/memory/`
