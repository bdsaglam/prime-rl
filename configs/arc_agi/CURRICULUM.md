# ARC-AGI Curriculum Learning: 2024 → 2025

Two-phase training using checkpoint resume. Phase 0 trains on ARC-AGI 2024, Phase 1 resumes on ARC-AGI 2025.

Both phases use difficulty filtering (`online_difficulty_filtering`, `easy_threshold`, `hard_threshold`) to evict solved/impossible problems from the sampling pool during training.

## Usage

Phases must be run sequentially — both share `output_dir = "outputs"` (default) because `resume_step` only looks in its own `output_dir` for checkpoints.

```bash
# Phase 0: train on 2024 (steps 0–500)
uv run rl @ configs/arc_agi/curriculum-2024.toml

# Phase 1: resume on 2025 (steps 500–1000, loads Phase 0 checkpoint)
uv run rl @ configs/arc_agi/curriculum-2025.toml
```

## Key differences between phases

| Setting | Phase 0 (2024) | Phase 1 (2025) |
|---|---|---|
| `max_steps` | 500 | 1000 |
| `dataset_name` | `arc-prize-2024` | `arc-prize-2025` |
| `ckpt.resume_step` | — | `-1` (latest) |
| `ckpt.skip_buffer` | — | `true` |

Phase 1 loads model weights and optimizer state from the latest Phase 0 checkpoint but starts a fresh buffer (old 2024 pool assignments don't apply to 2025 tasks).

## Tuning

- Adjust `max_steps` in each phase to control how long to train on each dataset
- Use `ckpt.resume_step = 500` instead of `-1` to pin to a specific checkpoint
- Set `ckpt.skip_progress = true` in Phase 1 to reset the step counter to 0
