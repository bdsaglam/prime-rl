# prime-rl: ARC-AGI On-Policy Distillation

Fork of [PrimeIntellect/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) for training ARC-AGI REPL agents with on-policy distillation (OPD).

## Project Goal

Fine-tune Qwen3-8B as an ARC-AGI REPL agent. The model writes Python code in a multi-turn REPL to analyze grid patterns and produce predictions. Standard RL (GRPO) with sparse rewards doesn't work well for ARC — OPD provides dense per-token feedback from a teacher model to stabilize training.

## Current State

**Phase 0 is running.** Qwen3-8B student with Qwen3-32B frozen teacher on 4 GPUs (2 inference + 1 trainer + 1 teacher). Training config: `configs/arc_agi/opd-rl-qwen-8b.toml`. W&B project: `arc-agi-opd`.

Phase 0 required 3 bug fixes in prime-rl (documented in `tmp/on-policy-distillation/prime-rl-implementation-notes.md`):
1. Validator ordering for `teacher_tau` + `num_teacher_gpus` (`src/prime_rl/configs/rl.py:326-336`)
2. Teacher prefill crash on long sequences — truncation + padding (`src/prime_rl/orchestrator/utils.py:146-190`)
3. Teacher inference port conflict when using explicit `[teacher_inference]` config

**Next:** Phase 1 — privileged-info teacher (inject ground truth into teacher prompt). Plan: `tmp/on-policy-distillation/phase1-privileged-teacher.md`.

## Documentation

- `tmp/on-policy-distillation/README.md` — Full index of all OPD docs
- `tmp/on-policy-distillation/phase1-privileged-teacher.md` — Phase 1 plan (next step)
- `tmp/on-policy-distillation/prime-rl-implementation-notes.md` — Lessons from Phase 0
- `tmp/on-policy-distillation/arc-agi-opd-plan.md` — Original phased plan (Phase 0–3)
- `tmp/on-policy-distillation/opd-concepts.md` — OPD tutorial, all self-distillation variants
- `tmp/on-policy-distillation/research-notes/` — Paper summaries and framework analyses

## Key Commands

```bash
# Install dependencies
uv sync

# Validate config without training
uv run rl @ configs/arc_agi/opd-rl-qwen-8b.toml --dump-config .pydantic_config/arc_test

# Run training (requires 4 GPUs: 2 inference + 1 trainer + 1 teacher)
uv run rl @ configs/arc_agi/opd-rl-qwen-8b.toml

# Check logs while running
tail -F outputs/logs/orchestrator.stdout
tail -F outputs/logs/trainer.stdout
tail -F outputs/logs/inference.stdout
tail -F outputs/logs/teacher_inference.stdout

# Test environment loads
uv run python -c "import verifiers as vf; env = vf.load_environment('arc-agi', dataset_name='arc-dummy', env_type='repl'); print(type(env))"
```

## How OPD Works in prime-rl

1. Student generates rollouts (multi-turn REPL interactions)
2. Teacher scores the same token sequence via prefill (no generation)
3. Loss = `adv_tau * GRPO_advantage + teacher_tau * (teacher_logprobs - student_logprobs)`
4. The teacher KL term provides dense per-token guidance

Key code:
- Loss: `src/prime_rl/trainer/rl/loss.py:107-173`
- Teacher logprobs: `src/prime_rl/orchestrator/utils.py:146-190`
- Orchestrator call site: `src/prime_rl/orchestrator/orchestrator.py:530-544`
- Multi-turn interleaving: `src/prime_rl/orchestrator/trajectories.py`
- TrainingSample struct: `src/prime_rl/transport/types.py:5-22`

## Local Modifications to prime-rl

1. **`src/prime_rl/configs/rl.py:326-336`** — Fixed validator ordering bug. `validate_teacher_model` now allows `teacher_tau > 0` when `deployment.num_teacher_gpus > 0`.
2. **`src/prime_rl/orchestrator/utils.py:146-190`** — Added `max_model_len` param to `compute_teacher_logprobs`. Truncates sequences exceeding teacher context window, pads logprobs with 0.0.
3. **`pyproject.toml`** — Added `arc-agi` dependency with local path source in `[tool.uv.sources]`.
4. **`environments/arc_agi/`** — ARC-AGI REPL environment (copied from rlvr).
5. **`configs/arc_agi/opd-rl-qwen-8b.toml`** — Phase 0 training config.

## Repository Structure

```
prime-rl/
├── src/prime_rl/
│   ├── configs/              # Pydantic config models (rl.py, trainer.py, orchestrator.py, inference.py)
│   ├── entrypoints/rl.py     # Main entry — launches all processes
│   ├── trainer/rl/           # Training loop + loss function
│   ├── orchestrator/         # Orchestrator loop, teacher logprobs, trajectories
│   ├── inference/            # vLLM inference server
│   └── transport/types.py    # TrainingSample, TrainingBatch structs
├── environments/arc_agi/     # ARC-AGI REPL environment
│   ├── src/arc_agi/          # env.py, data.py, rewards.py, envs/repl.py
│   └── data/                 # ARC datasets (2024, 2025, dummy)
├── configs/arc_agi/          # Training configs
└── tmp/on-policy-distillation/  # Research docs, plans, notes
```

## Config Reference

Configs use TOML, passed via `@` syntax: `uv run rl @ config.toml`.

Key OPD settings in `configs/arc_agi/opd-rl-qwen-8b.toml`:
- `trainer.loss.teacher_tau = 0.3` — Distillation strength (0 = pure RL)
- `trainer.loss.adv_tau = 1.0` — RL reward signal (hybrid: both active)
- `deployment.num_teacher_gpus = 1` — Enables teacher inference server
- `[teacher_inference]` — Separate config for teacher model (Qwen3-32B, port 8032)
- `orchestrator.env[].args.env_type = "repl"` — Multi-turn REPL environment
- `orchestrator.env[].args.reward_mode = "balanced"` — Balanced reward weighting

## ARC-AGI Environment

The REPL env (`environments/arc_agi/src/arc_agi/envs/repl.py`):
- Model receives ARC task (training I/O pairs + test input) as system prompt
- Each turn: model writes Python code, env executes it, returns stdout/stderr
- Model calls `SUBMIT(test=[...])` to submit predictions
- Reward: grid accuracy (balanced = binary + partial credit)
- Ground truth is in `info["test"][i]["output"]` — available for Phase 1 privileged teacher

Environment args: `dataset_name`, `env_type` ("repl"/"iterative"), `reward_mode` ("binary"/"partial"/"balanced"), `max_turns`, `eval_dataset`, `eval_split`.

## Hardware

4 GPUs (A100 80GB):
- GPUs 0-1: Student inference (Qwen3-8B, DP=2)
- GPU 2: Trainer (LoRA r=32)
- GPU 3: Teacher inference (Qwen3-32B, bf16, gpu_mem_util=0.90)
