# Phase 1.5: Hint-Assisted Curriculum Learning

## Problem

Phase 1 OPD training (steps 0-23) shows the student model (Qwen3-8B) is not improving on ARC-AGI puzzles. Rewards oscillate around 0.15-0.20 with no upward trend, grad norm is decreasing, and mismatch KL is flat at ~0.0012. The task appears too difficult for the student to learn from sparse RL rewards alone, even with teacher distillation.

**Root cause hypothesis**: ARC puzzles require discovering the transformation rule before writing code. Without understanding the rule, the student generates random code that rarely succeeds. The teacher (Qwen3-32B) sees ground-truth outputs via `teacher_context` but the student never sees any hint — the information gap is too large for the KL signal to bridge.

## Goal

Use **curriculum learning** with hints to bootstrap the student:

1. **Phase 1.5a** (hint-assisted training): Give the student a text hint describing the transformation rule as part of its prompt. The student learns to execute puzzle solutions given hints — a much easier task than discovering rules from scratch.

2. **Phase 1.5b** (hint-free fine-tuning): Resume training without hints. The student, having learned good REPL patterns and code structure from Phase 1.5a, should now be better equipped to discover rules on its own.

This is analogous to supervised pre-training followed by RL fine-tuning, but done entirely within the OPD framework.

## Current Architecture

### Data flow

The dataset (`data.py:prepare_dataset`) produces rows with:
- `question`: formatted grid pairs (train examples + test inputs)
- `answer`: JSON test outputs
- `info`: JSON dict containing `ArcTask` fields including `teacher_context`

The `teacher_context` field is currently used ONLY by the teacher (injected into teacher's system prompt in `orchestrator/utils.py:build_teacher_prompt_ids`). The student never sees it.

### Hint data

Hints are stored as `arc-agi_{split}_hints.json` files under `environments/arc_agi/data/{dataset}/`. Currently only `arc-prize-2024/arc-agi_training_hints.json` exists (276 hints covering ~69% of 400 training tasks). Each hint is a text description of the transformation rule (avg ~2300 chars).

The hints file is loaded in `data.py:prepare_dataset` (line 182-187) into `teacher_contexts`. Tasks without hints are skipped (line 206-207).

### Environment

`envs/repl.py:ArcAgiReplEnv` builds the prompt as:
- System message: `SYSTEM_PROMPT` (static instructions)
- User message: `question` (grid data from dataset)
- The `info` dict is available via `state["info"]` but only used for reward computation and teacher prompt building

## Implementation Plan

### Step 1: Add `hint` field to data pipeline

**File: `environments/arc_agi/src/arc_agi/data.py`**

- Rename `teacher_context` to `hint` in `ArcTask` TypedDict
- Add a new `hint` field to `ArcTask` that stores the raw hint text (currently stored in `teacher_context`)
- Keep `teacher_context` as a separate field for the teacher's privileged prompt (which may include more than just hints, e.g., ground-truth outputs)
- Update `prepare_dataset` to populate both fields:
  - `hint`: the raw hint text from the hints JSON file (no suffix appended)
  - `teacher_context`: hint + ground-truth outputs + suffix (as it is now)

**Rationale**: Separating `hint` from `teacher_context` allows independent control — the student can see the hint while the teacher sees hint + ground truth.

### Step 2: Add `include_hint` flag to environment

**File: `environments/arc_agi/src/arc_agi/envs/repl.py`**

- Add `include_hint: bool = False` parameter to `ArcAgiReplEnv.__init__`
- In `setup_state`, if `include_hint=True` and `info["hint"]` is non-empty, append the hint to the user message (the question/grid data)

Format for hint injection into user message:
```
{question}

---
Hint: Here is an analysis of the transformation rule for this puzzle:
{hint}

Use this hint to guide your solution. Write Python code to implement the described transformation.
```

**File: `environments/arc_agi/src/arc_agi/env.py`**

- Add `include_hint: bool = False` parameter to `load_environment`
- Pass it through to `ArcAgiReplEnv`

### Step 3: Evaluate baseline performance with hints

Before training, evaluate both Qwen3-8B and Qwen3-32B with and without hints to quantify the improvement.

**Use tmux** to manage long-running inference servers and eval runs:

```bash
# --- Terminal 1 (tmux session: inference) --- Start vLLM server for 8B ---
tmux new-session -d -s inference
tmux send-keys -t inference 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-8B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.4 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '"'"'{"enable_thinking": true}'"'"' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes' Enter

# Wait for server to be ready (check with: curl http://0.0.0.0:8900/health)

# --- Terminal 2 (tmux session: eval) --- Run evals ---
tmux new-session -d -s eval

# 8B without hints (baseline)
vf-eval arc-agi -a '{"dataset_name":"arc-prize-2024"}' -n 8 -r 3 -m Qwen/Qwen3-8B -b http://0.0.0.0:8900/v1 --debug

# 8B with hints (after Step 2 is implemented — pass include_hint in env args)
vf-eval arc-agi -a '{"dataset_name":"arc-prize-2024","include_hint":true}' -n 8 -r 3 -m Qwen/Qwen3-8B -b http://0.0.0.0:8900/v1 --debug

# Kill 8B server, start 32B server, repeat evals
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# 32B server (higher gpu-memory-utilization)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-32B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

# 32B without hints
prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m willcb/Qwen3-32B -b http://0.0.0.0:8900/v1

# 32B with hints
prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024","include_hint":true}' -n 4 -r 3 -m willcb/Qwen3-32B -b http://0.0.0.0:8900/v1
```

See `scratchpad.md` for more model-specific server configs and eval recipes.

The key metric is the reward (balanced: exact_match + cell_accuracy + shape_match + format).

**Success criterion**: If 8B-with-hints significantly outperforms 8B-without-hints (e.g., 2x+ reward), proceed with hint-assisted training. If the improvement is marginal, hints may not help and we should try other approaches.

**Results (Qwen3-8B on arc-prize-2024 training split, 8 tasks, 3 rollouts):**
- Without hints: avg reward = 0.106
- With hints: avg reward = 0.436
- **4.1x improvement** — success criterion met, proceed with training.

Note: Use `-a` (not `-x`) to pass `dataset_name`/`include_hint` to `vf-eval`. The `-a` flag maps to `--env-args` which goes to `load_environment()`, while `-x` maps to `--extra-env-kwargs` which goes to the environment constructor.

### Step 4: Create hint-assisted training config

**File: `configs/arc_agi/opd-rl-qwen-8b-hint-curriculum.toml`**

Based on the current config but with:
```toml
# Phase 1.5a: hint-assisted training
[[orchestrator.env]]
id = "arc-agi"
name = "arc-agi-repl"
args = { dataset_name = "arc-prize-2024", eval_dataset = "arc-prize-2024", eval_split = "evaluation", max_turns = 10, include_hint = true }

# Eval still without hints (to measure generalization)
[[orchestrator.eval.env]]
id = "arc-agi"
name = "arc-agi-eval"
args = { dataset_name = "arc-prize-2024", split = "evaluation", max_turns = 10, include_hint = false }
```

Hyperparameter considerations:
- `max_steps = 100` — shorter than full 200, just enough to learn execution patterns
- `teacher_tau = 0.3` — slightly lower since hints make the task easier
- `lr = 1e-5` — lower LR since we expect faster convergence with easier task
- Keep LoRA rank=32, same memory optimizations

### Step 5: Run hint-assisted training (Phase 1.5a)

Use tmux to manage the training process. Follow the patterns in `tmp/on-policy-distillation/prime-rl-training-management-guide.md` for crash recovery, monitoring, and diagnostics.

**Pre-launch checklist** (from the management guide Section 10.1):
```bash
# 1. Kill zombie GPU processes from previous runs
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
nvidia-smi  # Verify GPUs are clean (0 MiB used)

# 2. Check disk space (need >100GB free for checkpoints)
df -h /

# 3. Validate config
uv run rl @ configs/arc_agi/opd-rl-qwen-8b-hint-curriculum.toml --dump-config /tmp/check

# 4. Clean output dir from previous runs
# (needed on every re-launch — see guide Section 10.1)
```

**Launch training in tmux** (from guide Section 10.2):
```bash
# Create tmux session with training + monitoring windows
tmux new-session -d -s opd -n train \
  'uv run rl @ configs/arc_agi/opd-rl-qwen-8b-hint-curriculum.toml --clean-output-dir; echo "=== TRAINING EXITED (code: $?) ==="'

tmux new-window -t opd -n orchestrator 'sleep 5 && tail -F outputs/logs/orchestrator.stdout'
tmux new-window -t opd -n trainer      'sleep 5 && tail -F outputs/logs/trainer.stdout'
tmux new-window -t opd -n teacher      'sleep 5 && tail -F outputs/logs/teacher_inference.stdout'
tmux new-window -t opd -n inference    'sleep 5 && tail -F outputs/logs/inference.stdout'
tmux new-window -t opd -n gpu  'watch -n 5 nvidia-smi'
tmux new-window -t opd -n disk 'watch -n 60 "df -h / | tail -1"'
tmux select-window -t opd:train
```

**Monitor using tmux** (read output programmatically):
```bash
tmux capture-pane -t opd:orchestrator -p -S -50  # Recent orchestrator output
tmux capture-pane -t opd:trainer -p -S -50       # Recent trainer output
tmux capture-pane -t opd:gpu -p -S -10           # GPU memory
```

**What to monitor** (from guide Sections 1-2):
- Reward should improve faster than Phase 1 (target: 0.3+ average within 20 steps)
- Mismatch KL should be higher (teacher signal is more useful when student understands the hint)
- Entropy should stay healthy (0.3-0.6)
- `is_truncated/mean` < 0.2 (if high, fix sequence lengths — see guide Section 4)
- Check `trainer_probs/mean` vs `inference_probs/mean` gap > 0.001 (confirms policy is updating)

**If training crashes**, follow guide Section 10.4:
1. Read last ~100 lines from each tmux log pane to diagnose
2. Kill session and GPU processes: `tmux kill-session -t opd && nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9`
3. Fix the issue, then re-launch with the same tmux setup above

For the full decision tree on hyperparameter tuning and common failure modes, see `tmp/on-policy-distillation/prime-rl-training-management-guide.md` Sections 8-9.

**Phase 1.5a Early Training Results (Steps 0-3):**

W&B run: `ynrsokah` (orchestrator), project `arc-agi-opd`.

Orchestrator (rollout rewards):
| Step | Reward | Seq Length | Notes |
|------|--------|-----------|-------|
| 0 | 0.7008 | 7627 | Fresh rollouts |
| 1 | 0.4475 | 11561 | Buffer (harder puzzles) |
| 2 | 0.4747 | 10091 | Buffer (harder puzzles) |
| 3 | 0.7585 | 7529 | Fresh rollouts, policy updated |

Trainer:
| Step | Loss | Entropy | Mismatch KL | Grad Norm |
|------|------|---------|-------------|-----------|
| 0 | -0.0003 | 0.2819 | 0.0007 | 0.0352 |
| 1 | -0.0001 | 0.2977 | 0.0006 | 0.0244 |
| 2 | -0.0002 | 0.2872 | 0.0006 | 0.0227 |

Eval progression (without hints — measures generalization):
| Step | Avg@4 | Truncated | Notes |
|------|-------|-----------|-------|
| 0 | 0.2542 | 28.1% | Baseline |
| 25 | 0.2260 | 28.1% | Slight decline |
| 50 | 0.1684 | 43.8% | **Declining — hint dependency growing** |

Training metrics at step 50:
- Entropy: 0.45 (healthy)
- Mismatch KL: 0.0004 (student converging toward teacher)
- Grad norm: 0.009 (very low — plateauing)
- Training reward: 0.45-0.77 range (hint-assisted, consistently high)

**Critical finding**: Hint-free eval is declining while hint-assisted training rewards stay high. The model is overfitting to the hint format — learning to execute given hints rather than learning transferable reasoning patterns. Truncation rising (28% → 44%) suggests the model generates longer, unfocused responses without hints.

**Implication for Phase 1.5b**: Earlier checkpoints (step 10-30) may be better starting points for hint-free fine-tuning than later ones. The 1.5b config should try resuming from step 20 or 30 rather than the final step.

### Step 6: Resume without hints (Phase 1.5b)

After Phase 1.5a converges (or reaches max_steps), create a new config that:
- Starts from the Phase 1.5a checkpoint
- Sets `include_hint = false` in env args
- Uses higher `adv_tau` (rely more on RL reward now)
- Uses lower `teacher_tau` (less hand-holding needed)

```toml
# Phase 1.5b: hint-free fine-tuning from Phase 1.5a checkpoint
[ckpt]
resume_step = <last_step_from_1.5a>

[[orchestrator.env]]
args = { ..., include_hint = false }
```

Use the same tmux-based launch and monitoring workflow as Step 5.

## Key Files to Modify

1. `environments/arc_agi/src/arc_agi/data.py` — Add `hint` field to `ArcTask`, update `prepare_dataset`
2. `environments/arc_agi/src/arc_agi/envs/repl.py` — Add `include_hint` flag, inject hint into prompt
3. `environments/arc_agi/src/arc_agi/env.py` — Pass `include_hint` through to env
4. `configs/arc_agi/opd-rl-qwen-8b-hint-curriculum.toml` — New training config (copy from current)

## Reference Documents

- `tmp/on-policy-distillation/prime-rl-training-management-guide.md` — **Essential reading** for managing training runs. Covers crash recovery, metric dashboards, failure mode diagnosis, hyperparameter tuning decision trees, and tmux session conventions. Battle-tested across Phase 1 and Phase 2 runs.
- `scratchpad.md` — Inference server commands (`vllm serve`) and eval recipes (`prime eval run`) for various models (Qwen3-8B/14B/32B, Devstral, GLM, etc.)
- `configs/arc_agi/opd-rl-qwen-8b-teacher-context.toml` — Phase 1 config (base for Phase 1.5 config)
- `CLAUDE.md` — Project overview, key commands, architecture summary

## Risks and Mitigations

- **Hint dependency**: Student may become overly reliant on hints and fail without them. Mitigation: Phase 1.5b specifically trains without hints; also run eval without hints during 1.5a to track generalization.
- **Hint quality**: The 276 available hints cover only 69% of training tasks. Tasks without hints are already skipped. This is fine — we train on the 276 tasks with hints.
- **Overfitting to hint format**: Student may learn to pattern-match on "Hint:" prefix rather than understanding content. Mitigation: Monitor eval-without-hints reward during training.

## Relationship to Other Phases

- **Phase 0** (blind teacher, same model): Completed, showed KL is just regularization.
- **Phase 1** (privileged teacher with ground truth): Current failing run. Teacher sees answers but student doesn't benefit enough.
- **Phase 1.5** (this plan): Student sees hints too. Curriculum approach to bootstrap learning.
- **Phase 2** (teacher generates reference solutions): Future — teacher generates full solution traces as privileged info.

## W&B Tracking

- Project: `arc-agi-opd`
- Run names: `arc-agi-hint-curriculum-phase1.5a`, `arc-agi-hint-curriculum-phase1.5b`
- Key metrics to compare across phases: `reward/mean`, `mismatch_kl`, `entropy`, `is_truncated/mean`
