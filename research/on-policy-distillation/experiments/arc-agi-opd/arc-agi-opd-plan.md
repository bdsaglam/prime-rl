# ARC-AGI On-Policy Distillation: Implementation Plan

This document lays out a phased implementation plan for training ARC-AGI models with on-policy distillation (OPD) using prime-rl. Each phase builds on the previous one, starting with the easiest (native prime-rl OPD) and progressing to more advanced self-distillation variants.

## Hardware

Two servers, each with 4x NVIDIA A100 80GB (8 GPUs total). Currently, one server runs a single training job with 3 inference GPUs + 1 trainer GPU. The second server is available for the teacher model or additional training.

## Summary of Phases

| Phase | Method | Teacher | Needs Code Changes? | Risk | Expected Payoff |
|-------|--------|---------|---------------------|------|-----------------|
| 0 | Standard OPD | Qwen3-32B on server 2 | No (native prime-rl) | Low | Moderate |
| 1 | Privileged-info self-distillation | Same model + ground truth | Yes (orchestrator) | Medium | High |
| 2 | SDPO-style env feedback | Same model + REPL feedback | Yes (orchestrator + loss) | Medium-high | High |
| 3 | RLTF-FM auxiliary loss | N/A (auxiliary task) | Yes (trainer) | Medium | High |

---

## Phase 0: Standard OPD with External Teacher

**Goal:** Get OPD working end-to-end using prime-rl's native support. The student (Qwen3-8B or 32B) learns from a stronger teacher model.

### Why start here

- Zero code changes required -- prime-rl supports this natively via `teacher_tau`
- Validates the full pipeline (teacher serving, logprob computation, loss blending) before adding complexity
- Even without self-distillation, a stronger teacher provides dense token-level guidance that our sparse RL rewards cannot

### Teacher model selection

For the teacher to be useful, it needs to be better at ARC-AGI than the student. Options:

| Option | Model | Hosting | Pros | Cons |
|--------|-------|---------|------|------|
| A | Qwen3-32B (same as student) | Server 2, 4 GPUs | Same tokenizer, no compatibility issues | Same model -- only useful as regularizer against initial checkpoint |
| B | Qwen3-235B-A22B (MoE) | API (e.g., Fireworks, Together) | Much stronger, true teacher signal | API latency, cost per token, rate limits |
| C | DeepSeek-R1 (671B MoE) | API | Strong reasoning, different approach | Different tokenizer (incompatible!) |
| D | Qwen3-32B fine-tuned on ARC | Server 2, 4 GPUs | Domain-specialized teacher | Requires pre-training a teacher first |

**Recommendation: Option A first (Qwen3-32B self-teaching), then Option B (API teacher) if the signal is too weak.**

Option A is the simplest starting point. Even though it's the same model, the teacher is frozen at the initial checkpoint while the student trains. This acts as a conservative regularizer -- the student gets pushed back toward the pre-trained distribution at every token. This may actually help in our setting: our RL runs showed the model not learning, possibly because it drifts into bad regions. The teacher KL acts as a stabilizing anchor.

If the teacher KL is near zero (same model, weak signal), switch to Option B with an API-hosted 235B model.

### GPU allocation

**Server 1 (student):**
- GPUs 0, 1, 2: Student inference (rollout generation)
- GPU 3: Trainer

**Server 2 (teacher):**
- GPUs 0, 1, 2, 3: Teacher inference server (vLLM)

The teacher runs as an external vLLM server. prime-rl connects via `orchestrator.teacher_model.client.base_url`.

### Config: `configs/prime-rl/arc-agi-qwen3-32b-opd.toml`

```toml
# ==============================================================================
# Core
# ==============================================================================

max_steps = 500
seq_len = 32768
inference_gpu_ids = [0, 1, 2]
trainer_gpu_ids = [3]
# NOTE: no teacher_gpu_ids -- teacher is external

[model]
name = "willcb/Qwen3-32B"

# ==============================================================================
# Teacher (external server on server 2)
# ==============================================================================

[trainer.loss]
teacher_tau = 0.3    # start conservative; tune up if teacher_kl stays high
adv_tau = 1.0        # keep full RL reward signal (hybrid mode)

[orchestrator.teacher_model.client]
base_url = ["http://<server-2-ip>:8000/v1"]

[orchestrator.teacher_model.model]
name = "willcb/Qwen3-32B"

# ==============================================================================
# Trainer
# ==============================================================================

[trainer.optim]
lr = 1e-6
weight_decay = 0.0

[trainer.tokenizer]
name = "willcb/Qwen3-32B"

[trainer.model.ac]
freq = 1

[trainer.model.lora]
rank = 32
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ==============================================================================
# Orchestrator
# ==============================================================================

[orchestrator]
batch_size = 128
rollouts_per_example = 16
oversampling_factor = 2.0
max_concurrent = 64

[orchestrator.sampling]
max_tokens = 16384
temperature = 0.6

[orchestrator.wandb.log_extras]
samples = true
distributions = true
interval = 10

# ==============================================================================
# Environment
# ==============================================================================

[[orchestrator.env]]
id = "arc-agi"
args = { dataset_name = "arc-prize-2024", eval_dataset_name = "arc-prize-2024", eval_split = "evaluation", reward_mode = "balanced", max_turns = 40 }

# ==============================================================================
# Evaluation
# ==============================================================================

[orchestrator.eval]
interval = 100
rollouts_per_example = 4
num_examples = 16

[[orchestrator.eval.env]]
id = "arc-agi"
args = { dataset_name = "arc-prize-2024", eval_dataset_name = "arc-prize-2024", eval_split = "evaluation", reward_mode = "balanced", max_turns = 40 }

# ==============================================================================
# Inference (student)
# ==============================================================================

[inference]
gpu_memory_utilization = 0.9

[inference.model]
name = "willcb/Qwen3-32B"
max_model_len = 32768
dtype = "bfloat16"
enable_auto_tool_choice = true
tool_call_parser = "hermes"
reasoning_parser = "qwen3"

# ==============================================================================
# Weights & Biases
# ==============================================================================

[wandb]
project = "rlvr-arc-agi"
name = "opd-qwen3-32b-phase0"
```

### Launch procedure

1. Start teacher vLLM on server 2:
```bash
# On server 2
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-32B \
    --port 8000 \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3
```

2. Start training on server 1:
```bash
# On server 1
uv run rl @ configs/prime-rl/arc-agi-qwen3-32b-opd.toml
```

### Key metrics to watch

- `teacher_kl`: Should start positive and trend downward. If it stays near 0, the teacher signal is too weak (student = teacher). If it stays very high, consider reducing `teacher_tau`.
- `reward/mean`: Should improve faster than pure RL baseline (if we had one working).
- `entropy`: Should decrease gradually, not collapse.

### Hyperparameter sweep

| Parameter | Values to try | Rationale |
|-----------|--------------|-----------|
| `teacher_tau` | 0.1, 0.3, 0.5, 1.0 | Balance teacher guidance vs. exploration |
| `adv_tau` | 0.0, 0.5, 1.0 | 0.0 = pure distillation (no rewards needed), 1.0 = hybrid |
| `lr` | 1e-6, 5e-6 | Higher LR may be needed with dense teacher signal |

**Important first experiment:** Try `adv_tau = 0.0` (pure distillation, no rewards). This completely bypasses the sparse reward problem. The student learns purely from the teacher's token-level guidance. This only works if the teacher is better at ARC-AGI than the student, but it eliminates the zero-variance reward issue entirely.

---

## Phase 1: Self-Distillation with Privileged Information (OPSD-style)

**Goal:** Use the same model as both teacher and student, where the teacher is conditioned on the ground-truth output grid (privileged information). This eliminates the need for a separate stronger teacher.

### Why this matters

The ARC-AGI training data includes ground-truth output grids for test inputs. We can construct a teacher prompt that includes the correct output:

```
Student sees:
  "Here are training examples: [in1 -> out1], [in2 -> out2]. Predict the test output for [test_in]."

Teacher sees:
  "Here are training examples: [in1 -> out1], [in2 -> out2]. The correct test output is [test_out].
   After understanding the transformation pattern from the correct output, solve this problem:"
```

The teacher's next-token distributions are more informed because it knows the answer. This gap provides the training signal.

### Code changes required

prime-rl's `compute_teacher_logprobs()` in `src/prime_rl/orchestrator/utils.py` sends the exact same token sequence to the teacher as the student saw. We need to modify this to:

1. **Construct a different prompt for the teacher** that includes the ground-truth output
2. **Tokenize the augmented teacher prompt** separately
3. **Append the student's completion tokens** to the teacher prompt tokens
4. **Score the full sequence** via the teacher vLLM server
5. **Extract only the completion logprobs** (aligned to the student's completion)

#### Modification 1: Store ground truth in TrainingSample

In `src/prime_rl/orchestrator/trajectories.py`, the `TrainingSample` dataclass needs a new field:

```python
@dataclass
class TrainingSample:
    prompt_ids: list[int]
    completion_ids: list[int]
    reward: float
    # ... existing fields ...
    teacher_logprobs: list[float] | None = None
    privileged_info: str | None = None  # NEW: ground truth for teacher prompt
```

The orchestrator already has access to the environment's `info` dict (which contains the full task including `test[i].output`). We need to thread this through to the TrainingSample.

#### Modification 2: Construct teacher prompt

Create a helper that builds the teacher's augmented prompt:

```python
def build_teacher_prompt(
    student_messages: list[dict],  # the chat messages the student saw
    ground_truth_output: list[list[int]],  # the correct grid
    tokenizer,
) -> list[int]:
    """Build a teacher prompt that includes the ground truth answer."""
    # Take the student's system prompt and user message
    # Inject the ground truth output into the system prompt or first user message
    teacher_system = student_system + f"""

PRIVILEGED INFORMATION (for teacher scoring only):
The correct output for the test input is:
{format_grid(ground_truth_output)}

After understanding the correct transformation, evaluate the following attempt:
"""
    # Tokenize the augmented prompt
    teacher_prompt_ids = tokenizer.apply_chat_template(teacher_messages, ...)
    return teacher_prompt_ids
```

#### Modification 3: Modified compute_teacher_logprobs()

```python
async def compute_teacher_logprobs_privileged(
    clients, model_name, samples, tokenizer
):
    async def _compute_single(client_config, sample):
        if sample.privileged_info:
            # Build augmented teacher prompt
            teacher_prompt_ids = build_teacher_prompt(
                sample.messages, sample.privileged_info, tokenizer
            )
        else:
            teacher_prompt_ids = sample.prompt_ids

        # Score: teacher_prompt + student_completion
        full_ids = teacher_prompt_ids + sample.completion_ids
        response = await client.post(
            "/chat/completions/tokens",
            body={
                "model": model_name,
                "messages": [{"role": "user", "content": ""}],
                "tokens": full_ids,
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "skip_special_tokens": False,
                "prompt_logprobs": True,
            },
        )
        # Extract logprobs for ONLY the completion tokens
        all_logprobs = [...]
        # Offset by teacher prompt length
        completion_logprobs = all_logprobs[len(teacher_prompt_ids):]
        return completion_logprobs
```

### Config changes

Same as Phase 0, but the teacher serves the same model and the code handles privileged info injection:

```toml
# Teacher is the same model (self-distillation)
[orchestrator.teacher_model.client]
base_url = ["http://<server-2-ip>:8000/v1"]

[orchestrator.teacher_model.model]
name = "willcb/Qwen3-32B"

# Use a custom flag to enable privileged-info mode
[orchestrator.teacher_model]
privileged_info = true  # NEW: signals the code to inject ground truth

[trainer.loss]
teacher_tau = 0.5   # can be higher since teacher is genuinely more informed
adv_tau = 0.5       # blend with RL rewards
```

### Design decisions for ARC-AGI privileged info

What ground truth to give the teacher:

| Option | Content | Signal Strength | Risk |
|--------|---------|-----------------|------|
| A | Just the output grid | Moderate | Model may not understand spatial transformation from grid alone |
| B | Output grid + transformation description (if available) | Strong | Descriptions not available for all tasks |
| C | All training I/O pairs + test output | Strong | Long context, may exceed teacher's capacity to reason about |
| D | Output grid shown inline with test input | Moderate-strong | Natural format, model can pattern-match |

**Recommendation: Start with Option D** -- show the test output alongside the test input in the same format as training examples. This is the most natural privileged info:

```
Teacher prompt addition:
"The correct test output is:
[test_input] -> [correct_test_output]

After understanding the transformation pattern, solve this problem using your own approach:"
```

This leverages the model's existing ability to learn ARC patterns from input-output demonstrations. The teacher has one extra demonstration (the test pair), making rationalization straightforward.

### Teacher freezing strategy

Following OPSD (Zhao et al.), freeze the teacher at the initial checkpoint. The teacher vLLM server loads the base model once and never updates. This is the simplest approach and provides stable regularization.

If training stalls (teacher_kl goes to 0 because the student matches the frozen teacher), consider:
- EMA teacher (SDFT-style): periodically update teacher weights as EMA of student
- This requires more infrastructure (teacher weight refresh) but adapts to student improvements

---

## Phase 2: SDPO-Style Environment Feedback

**Goal:** Combine self-distillation with the rich feedback our REPL environment naturally provides. Instead of (or in addition to) ground-truth output, use REPL execution feedback (error messages, grid diffs, successful peer solutions) as the privileged information for the teacher.

### Why this is powerful for ARC-AGI

Our REPL environment already produces rich feedback:
- **Python tracebacks** when code fails
- **Printed grid output** the model can compare against expected
- **Successful peer solutions** from other rollouts in the same batch
- **soft_accuracy scores** showing partial correctness

SDPO (Hubotter et al.) showed that this environment feedback, combined with successful peer solutions, creates an effective self-teacher *without needing ground-truth answers*. For ARC-AGI, we have both -- we can use ground truth AND env feedback.

### Reprompting template for ARC-AGI

Adapt SDPO's template for our REPL environment:

```
User: {original_arc_prompt}

{if successful_peer_solution exists:}
A correct solution for this task:
{successful_peer_code}

{if environment_feedback exists and no peer solution:}
Feedback from an earlier unsuccessful attempt:
{repl_error_output}

After studying the above, correctly solve the original task.
```

The key insight from SDPO: the teacher prompt should NOT include the student's failed attempt. Including it biases the teacher toward the student's distribution and reduces exploration.

### Implementation approach

This phase requires the deepest modifications to prime-rl. There are two paths:

**Path A: Extend prime-rl's OPD** (recommended)
- Reuse the Phase 1 `compute_teacher_logprobs_privileged()` modification
- Instead of injecting ground-truth output, inject env feedback + peer solutions
- The orchestrator already has access to all rollout results in a batch (it needs this for GRPO advantage computation), so it can identify successful peers

**Path B: Port SDPO's advantage computation into prime-rl**
- SDPO uses `log p_teacher(token) - log p_student(token)` as per-token advantage (same formula as prime-rl's teacher_kl)
- The main addition is the reprompting logic and the top-K logit distillation for memory efficiency
- SDPO's EMA teacher adds complexity: need to periodically update teacher weights

### Data flow

```
1. Student generates N rollouts for problem x
2. Environment returns rewards + REPL feedback for each rollout
3. Orchestrator identifies:
   - Successful rollouts (reward > threshold)
   - Failed rollouts with feedback
4. For each failed rollout:
   a. Pick a successful peer (if any) as the demonstration
   b. Construct teacher prompt: problem + peer solution + env feedback
   c. Compute teacher logprobs on the failed rollout's completion
5. teacher_kl = teacher_logprobs - student_logprobs (per-token advantage)
6. Loss = adv_tau * rl_advantage + teacher_tau * teacher_kl
```

### When no successful peer exists

If all N rollouts for a problem fail (likely early in training for hard ARC tasks), there's no peer solution to use. Options:
- **Skip teacher signal for this problem** -- fall back to pure RL (teacher_kl = 0)
- **Use env feedback only** -- the teacher sees error messages but no correct solution. Still provides some signal.
- **Use ground truth (Phase 1 style)** -- fall back to privileged-info self-distillation

**Recommendation:** Combine Phases 1 and 2. When a successful peer exists, use it (SDPO-style). When no peer exists, use ground-truth privileged info (OPSD-style). This gives the strongest possible teacher signal in all cases.

### Config additions

```toml
[trainer.loss]
teacher_tau = 0.5
adv_tau = 1.0

[orchestrator.teacher_model]
privileged_info = true
use_peer_solutions = true          # NEW: use successful peers as teacher context
use_env_feedback = true            # NEW: include REPL output in teacher prompt
fallback_to_ground_truth = true    # NEW: use y* when no peer available
```

---

## Phase 3: RLTF-FM Feedback Modeling (Auxiliary Loss)

**Goal:** Train the model to predict REPL execution feedback as an auxiliary task, complementing the RL/distillation objectives.

### Why this matters

From RLTF (Song et al.): when base success rate is very low, reward-only RL suffers from weak identifiability -- the gradient signal concentrates on a small set of representation directions (Proposition 4.1). Feedback modeling acts as a "representation preconditioner" that fills in the missing directions.

This is exactly our situation: ARC-AGI has near-zero base success rate, so RL gradients are degenerate. Training the model to predict "what would the REPL output be if I ran this code?" forces it to learn an internal model of code execution and grid transformation -- representation knowledge that RL alone cannot identify.

### Implementation

This is the most independent phase -- it's an auxiliary loss that can be added to any of the previous phases.

#### Feedback prediction template

```
Given this ARC-AGI task and the following Python code, predict the output:

Task: {task_description}
Code:
```python
{model_generated_code}
```

Predicted execution output:
```

The model is trained to predict the actual REPL output (stdout + stderr) via standard cross-entropy loss on the feedback tokens.

#### Loss combination

```
total_loss = adv_tau * rl_loss + teacher_tau * distillation_loss + lambda_fm * feedback_modeling_loss
```

Where `feedback_modeling_loss` is standard next-token prediction (SFT-style) on the feedback tokens only.

#### Code modification

In `src/prime_rl/trainer/rl/loss.py`, add the auxiliary loss:

```python
def compute_loss(self, ...):
    # Existing RL + distillation loss
    rl_distill_loss = ...  # (existing code)

    # Feedback modeling auxiliary loss
    if feedback_ids is not None and lambda_fm > 0:
        fm_logits = model(feedback_prompt_ids)
        fm_loss = cross_entropy(fm_logits, feedback_ids)
        total_loss = rl_distill_loss + lambda_fm * fm_loss
    else:
        total_loss = rl_distill_loss

    return total_loss
```

#### Data pipeline

The feedback data comes naturally from the rollout process:
1. Student generates code in the REPL
2. REPL executes code and returns output (stdout/stderr)
3. Store (code, repl_output) pairs alongside the training samples
4. During training, predict repl_output from code

This requires storing the REPL feedback in the `TrainingSample` and threading it through to the trainer.

### Test-time self-feedback

Once trained with RLTF-FM, the model can self-critique at test time:
1. Generate code for an ARC task
2. Predict what the REPL output would be (without executing)
3. If the predicted output suggests errors, revise before executing
4. Execute and get actual feedback
5. Repeat

This is free test-time compute that doesn't require actual code execution.

### Config additions

```toml
[trainer.loss]
lambda_fm = 0.1      # feedback modeling loss weight
adv_tau = 1.0
teacher_tau = 0.5
```

---

## Phase Progression: What to Run and When

### Experiment 1: Pure distillation baseline (Phase 0)

**Config:** `teacher_tau = 1.0, adv_tau = 0.0` with Qwen3-32B teacher on server 2.

**Why:** Completely bypasses sparse rewards. If the teacher is better than the student at ARC, this should work even with zero environment rewards. This tests the OPD pipeline end-to-end.

**Success criterion:** `teacher_kl` decreases over training. Model generates ARC solutions that are more similar to what the teacher would produce.

**Expected time:** 1-2 days (500 steps).

### Experiment 2: Hybrid OPD + RL (Phase 0)

**Config:** `teacher_tau = 0.3, adv_tau = 1.0` with same teacher.

**Why:** Adds environment rewards back in. The teacher provides a dense learning signal while the binary reward grounds the optimization in actual task performance.

**Success criterion:** Higher reward mean than pure RL baseline. `teacher_kl` decreases AND reward improves.

### Experiment 3: Privileged-info self-distillation (Phase 1)

**Config:** Same model as teacher, ground-truth output as privileged info. `teacher_tau = 0.5, adv_tau = 0.5`.

**Why:** Tests whether the model can create a useful information gap by conditioning on the answer. This is the core OPSD hypothesis: rationalization is easier than generation.

**Prerequisite:** Phase 1 code changes implemented.

**Success criterion:** `teacher_kl` starts positive (teacher genuinely has different distribution from student) and decreases. Reward improves faster than Phase 0.

### Experiment 4: Combined self-distillation + env feedback (Phase 1 + 2)

**Config:** Privileged info + REPL feedback + peer solutions. `teacher_tau = 0.5, adv_tau = 1.0`.

**Why:** The strongest teacher signal: ground truth when no peer exists, successful peer solutions when available, plus error messages from the REPL.

**Prerequisite:** Phase 2 code changes implemented.

### Experiment 5: Add feedback modeling (Phase 3)

**Config:** Everything from Experiment 4 + `lambda_fm = 0.1`.

**Why:** The auxiliary feedback modeling loss acts as a representation preconditioner, particularly helpful when base success rate is near zero. This should improve the model's internal understanding of code execution on ARC grids.

**Prerequisite:** Phase 3 code changes implemented.

---

## Key Design Decision: Student Model Size

Our current config trains Qwen3-32B. The research shows:

- **OPSD requires 4B+ for meaningful gains** (Zhao et al.)
- **SDPO scales with model capability** -- 8B+ shows substantial gains
- **RLTF works at 8B** (tested with Llama-3.1-8B)

For self-distillation (Phases 1-3), the model must be capable enough to rationalize when given the answer. Qwen3-32B should be well above the threshold.

For standard OPD (Phase 0), student can be smaller (Qwen3-8B) if we use a stronger teacher (Qwen3-32B or API model).

| Setup | Student | Teacher | Use case |
|-------|---------|---------|----------|
| A | Qwen3-8B | Qwen3-32B (server 2) | Cheapest, good for iteration |
| B | Qwen3-32B | Qwen3-32B (server 2, frozen initial) | Self-regularization |
| C | Qwen3-32B | Qwen3-32B (self, privileged info) | Self-distillation (Phase 1+) |
| D | Qwen3-8B | API (Qwen3-235B) | Strongest teacher, API cost |

**Recommendation:** Start with Setup B for Phase 0 (quick, no code changes, validates pipeline). Move to Setup C for Phase 1+ (requires code changes but most promising).

---

## Risk Mitigation

### Risk 1: Teacher KL is too low (same model, no useful signal)

**Mitigation:** Monitor `teacher_kl` metric. If consistently < 0.01, the teacher and student distributions are too similar. Escalation path:
1. Increase privileged info richness (add more context to teacher prompt)
2. Switch to a stronger teacher model (API)
3. Try RLTF-FM instead (doesn't need a teacher at all)

### Risk 2: Teacher logprob computation is too slow

The teacher prefill pass adds latency. For multi-turn ARC rollouts (up to 40 turns), each rollout can be 32K tokens.

**Mitigation:**
- Teacher scoring is a single forward pass (prefill), not autoregressive generation -- much faster
- Use TP=2 on the teacher server for faster prefill
- The teacher only scores the final completion, not intermediate REPL turns

### Risk 3: Tokenizer mismatch

prime-rl sends raw token IDs to the teacher. If the teacher has a different tokenizer, logprobs are meaningless.

**Mitigation:** Only use Qwen3 family models as teacher (shared tokenizer). DeepSeek-R1 is incompatible.

### Risk 4: Multi-turn rollout complicates teacher prompt construction

ARC-AGI rollouts have multiple assistant turns (reasoning + code) interleaved with REPL output. The teacher needs to score the full sequence, but the privileged-info prompt changes the beginning.

**Mitigation:** For Phase 1, inject privileged info into the system prompt only. The rest of the conversation (assistant turns + REPL output) stays identical between teacher and student. This means the teacher prompt is:
```
[system prompt + privileged info] + [user message] + [assistant turn 1] + [repl output 1] + [assistant turn 2] + ...
```
The logprob offset is just the difference in system prompt length.

### Risk 5: Code changes to prime-rl break other things

**Mitigation:**
- Phase 0 requires zero code changes
- For Phases 1-3, fork prime-rl or use a feature branch
- Add the privileged info path as opt-in (gated behind config flags)
- Keep the original `compute_teacher_logprobs()` as fallback

---

## Implementation Effort Estimates

| Phase | Code Changes | Files Modified | Complexity |
|-------|-------------|----------------|------------|
| 0 | Config only | 0 | Trivial |
| 1 | Orchestrator | 2-3 files (trajectories.py, utils.py, config.py) | Moderate |
| 2 | Orchestrator + loss | 4-5 files (above + loss.py, orchestrator.py) | Moderate-high |
| 3 | Trainer | 2-3 files (loss.py, config.py, data pipeline) | Moderate |

Phases 1 and 2 share most code (both modify the teacher logprob pipeline). Phase 3 is relatively independent.

---

## Monitoring and Evaluation

### W&B Metrics to Track

| Metric | What it tells you | Healthy range |
|--------|-------------------|---------------|
| `teacher_kl` | How different teacher and student distributions are | Starts positive, trends down |
| `reward/mean` | Task completion rate | Should increase |
| `entropy` | Exploration level | Should decrease gradually, not collapse |
| `loss` | Training loss | Should decrease |
| `teacher_kl` + `reward/mean` together | Whether distillation signal helps task performance | Both should improve |

### Evaluation Protocol

1. **Online eval** (during training): 16 ARC tasks from validation set, 4 rollouts each, every 100 steps
2. **Offline eval** (after training): Full ARC-Prize-2024 evaluation set, 16 rollouts per task
3. **Comparison baselines:**
   - Pure RL (existing config, `adv_tau = 1.0, teacher_tau = 0.0`)
   - SFT on teacher demonstrations (upper bound if teacher is strong)
   - Pure distillation (no rewards)
   - Hybrid (distillation + rewards)

---

## Quick Reference: Which Experiment to Run When

```
Can't get RL to work (sparse rewards, zero variance)?
  ├── Have a stronger teacher model?
  │   ├── Yes → Phase 0: Pure distillation (adv_tau=0, teacher_tau=1)
  │   └── No → Phase 1: Self-distillation with ground truth
  │
  ├── RL works but is slow/sample-inefficient?
  │   ├── Have teacher → Phase 0: Hybrid (adv_tau=1, teacher_tau=0.3)
  │   └── No teacher → Phase 2: SDPO-style with env feedback
  │
  └── Base success rate is extremely low (< 1%)?
      └── Phase 3: Add RLTF-FM auxiliary loss (representation preconditioner)
```

Our current situation: **RL doesn't work due to sparse rewards**. We should start with Phase 0 pure distillation, then progress to Phase 1 self-distillation with ground truth.