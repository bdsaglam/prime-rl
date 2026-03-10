# Syncing Fork with Upstream

## Branch Strategy

- **`main`** — always mirrors upstream `PrimeIntellect-ai/prime-rl`
- **`opd-arc-agi`** — our working branch with all OPD modifications

## Setup (one-time)

```bash
git remote add upstream https://github.com/PrimeIntellect-ai/prime-rl.git
```

## Sync Process

```bash
# 1. Fetch upstream
git fetch upstream

# 2. Fast-forward main
git checkout main
git merge upstream/main --ff-only

# 3. Merge main into working branch
git checkout opd-arc-agi
git merge main
```

## Resolving Conflicts

Typical conflict points with our modifications:

| File | Our changes | Likely conflict |
|------|------------|-----------------|
| `pyproject.toml` | `arc-agi` in deps + sources | Near other env package entries in `[tool.uv.sources]` |
| `uv.lock` | Generated from our deps | Accept theirs (`git checkout --theirs uv.lock`), regenerate with `uv sync` |
| `src/prime_rl/orchestrator/orchestrator.py` | Teacher logprobs, `_build_teacher_prompts` | Import section (both sides add imports) |
| `src/prime_rl/configs/rl.py` | `validate_teacher_model` fix | Field reordering may shift lines |

Files that should always merge cleanly (no upstream changes to our code):
- `src/prime_rl/orchestrator/utils.py` — teacher prefill truncation/padding
- `src/prime_rl/trainer/rl/loss.py` — OPD loss function
- `src/prime_rl/configs/orchestrator.py` — `TeacherModelConfig` additions

## After Merge

```bash
# Regenerate lockfile with updated deps
uv sync

# Verify config still validates
uv run rl @ configs/arc_agi/opd-rl-qwen-8b-teacher-context.toml --dry-run
```
