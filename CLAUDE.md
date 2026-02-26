# prime-rl: ARC-AGI On-Policy Distillation

Fork of [PrimeIntellect/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) for training ARC-AGI REPL agents with on-policy distillation (OPD).

## Project Goal

Fine-tune Qwen3-8B as an ARC-AGI REPL agent. The model writes Python code in a multi-turn REPL to analyze grid patterns and produce predictions. Standard RL (GRPO) with sparse rewards doesn't work well for ARC — OPD provides dense per-token feedback from a teacher model to stabilize training.

## Research Documentation

All OPD research, paper summaries, and implementation plans are in `tmp/on-policy-distillation/`. Start with:

- `tmp/scratchpad.md` — Scratchpad for training ARC-AGI
- `tmp/training-prime-rl.md` — Training guide for prime-rl
- `tmp/on-policy-distillation/README.md` — Full index of all docs
- `tmp/on-policy-distillation/arc-agi-opd-plan.md` — Phased plan (Phase 0–3) for prime-rl
- `tmp/on-policy-distillation/handover.md` — Self-contained briefing with full context
- `tmp/on-policy-distillation/opd-concepts.md` — Tutorial on OPD, all self-distillation variants
- `tmp/on-policy-distillation/sdpo-implementation.md` — Alternative SDPO/verl approach (fallback)
- `tmp/on-policy-distillation/research-notes/` — Deep-dive paper summaries and framework analyses

## Current State: Phase 0 (OPD with frozen teacher)

Phase 0 uses prime-rl's native OPD support with zero training loop modifications. The frozen teacher is the same Qwen3-8B checkpoint — its KL divergence acts as a regularizer preventing drift.

**What's done:**
- ARC-AGI environment copied into `environments/arc_agi/` and registered as dependency
- Training config at `configs/arc_agi/opd-rl-qwen-8b.toml`
- Config validates with `--dump-config`
- Bug fix in `src/prime_rl/configs/rl.py:326-336` — validator ordering for `teacher_tau` + `num_teacher_gpus`

**What's next:**
- Run Phase 0 training: `uv run rl @ configs/arc_agi/opd-rl-qwen-8b.toml`
- Monitor W&B: teacher_kl, reward, entropy
- If signal is weak (teacher_kl near zero), try larger teacher (Qwen3-32B on 4 GPUs)
- Phase 1: Privileged-info self-distillation (teacher sees ground truth)

## Repository Structure

```
prime-rl/
├── src/prime_rl/
│   ├── configs/           # Pydantic config models
│   │   ├── rl.py          # RLConfig (top-level, deployment, shared settings)
│   │   ├── trainer.py     # TrainerConfig, LossConfig, LoRAConfig, OptimizerConfig
│   │   ├── orchestrator.py # OrchestratorConfig, EnvConfig, EvalConfig, SamplingConfig
│   │   └── inference.py   # InferenceConfig, vLLM server settings
│   ├── entrypoints/
│   │   └── rl.py          # Main entry point — launches all processes
│   ├── trainer/rl/
│   │   ├── train.py       # Training loop
│   │   └── loss.py        # Loss function (teacher_tau KL + adv_tau GRPO)
│   ├── orchestrator/
│   │   ├── orchestrator.py     # Orchestrator main loop
│   │   ├── trajectories.py     # interleave_rollout() — multi-turn merging
│   │   └── utils.py            # Teacher logprob computation (lines 146-176)
│   └── inference/
│       └── server.py      # vLLM inference server
├── environments/
│   └── arc_agi/           # ARC-AGI REPL environment (copied from rlvr)
│       ├── src/arc_agi/
│       │   ├── env.py     # load_environment() entry point
│       │   ├── data.py    # Dataset loading
│       │   ├── rewards.py # Reward functions (binary, partial, balanced)
│       │   ├── subprocess_interpreter.py  # REPL subprocess
│       │   ├── sandbox.py # Iterative env sandbox
│       │   └── envs/
│       │       ├── repl.py      # ArcAgiReplEnv (primary)
│       │       └── iterative.py # ArcAgiIterativeEnv
│       ├── data/           # ARC datasets (2024, 2025, dummy)
│       └── tests/
├── configs/
│   └── arc_agi/
│       └── rl.toml        # Phase 0 OPD training config
└── tmp/
    └── on-policy-distillation/  # Research docs, paper summaries, plans
```

## Key Commands

```bash
# Install dependencies
uv sync --extra flash-attn

# Validate config without training
uv run rl @ configs/arc_agi/opd-rl-qwen-8b.toml --dump-config .pydantic_config/arc_test

# Run training (requires 4 GPUs: 2 inference + 1 trainer + 1 teacher)
uv run rl @ configs/arc_agi/opd-rl-qwen-8b.toml

# Test environment loads
uv run python -c "import verifiers as vf; env = vf.load_environment('arc-agi', dataset_name='arc-dummy', env_type='repl'); print(type(env))"

# Run ARC-AGI environment tests
uv run pytest environments/arc_agi/tests/
```

## Config Reference

Config files use TOML and are passed via `@` syntax: `uv run rl @ config.toml`.

Key OPD settings in `configs/arc_agi/opd-rl-qwen-8b.toml`:
- `trainer.loss.teacher_tau = 0.3` — OPD distillation strength (0 = pure RL, higher = more teacher influence)
- `trainer.loss.adv_tau = 1.0` — RL reward signal strength (hybrid mode: RL + distillation)
- `deployment.num_teacher_gpus = 1` — Enables teacher inference server (auto-configured)
- `orchestrator.env[].args.env_type = "repl"` — Uses multi-turn REPL environment
- `orchestrator.env[].args.reward_mode = "balanced"` — Balanced reward weighting

## How OPD Works in prime-rl

1. Student generates rollouts (multi-turn REPL interactions)
2. Teacher (frozen checkpoint) scores the same token sequence via prefill
3. Loss = `adv_tau * GRPO_advantage + teacher_tau * (teacher_logprobs - student_logprobs)`
4. The teacher KL term provides dense per-token guidance, stabilizing training

Key code:
- Loss computation: `src/prime_rl/trainer/rl/loss.py:107-173`
- Teacher logprobs: `src/prime_rl/orchestrator/utils.py:146-176`
- Multi-turn interleaving: `src/prime_rl/orchestrator/trajectories.py`

## Local Modifications to prime-rl

1. **`src/prime_rl/configs/rl.py:326-336`** — Fixed validator ordering bug. `validate_teacher_model` now allows `teacher_tau > 0` when `deployment.num_teacher_gpus > 0` (auto-setup runs later).
2. **`pyproject.toml`** — Added `arc-agi` dependency with local path source.
3. **`environments/arc_agi/`** — Copied from rlvr repo.
4. **`configs/arc_agi/opd-rl-qwen-8b.toml`** — New training config.

## ARC-AGI Environment

The REPL env (`environments/arc_agi/src/arc_agi/envs/repl.py`) provides multi-turn code execution:
- Model receives ARC task (training I/O pairs + test input) as system prompt
- Each turn: model writes Python code, env executes it, returns stdout/stderr
- Model calls `SUBMIT(test=[...])` to submit final predictions
- Reward: grid accuracy (balanced mode = mix of binary + partial credit)
- `SubprocessInterpreter` runs code in isolated subprocess with numpy available

Environment args (passed via config `args`):
- `dataset_name`: ARC data folder (e.g., "arc-prize-2024")
- `env_type`: "repl" (primary) or "iterative"
- `reward_mode`: "binary", "partial", or "balanced"
- `max_turns`: Max REPL interaction turns (default 10)

## Hardware

4 GPUs for Qwen3-8B with LoRA:
- GPUs 0-1: Inference (DP=2, rollout generation)
- GPU 2: Trainer (LoRA, FSDP)
- GPU 3: Teacher inference (frozen checkpoint)
