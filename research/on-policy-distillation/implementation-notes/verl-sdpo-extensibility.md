# verl + SDPO Extensibility Analysis for On-Policy Distillation

**Date:** 2026-02-22
**Repos examined:**
- verl (stock): `/home/baris/repos/rlvr/tmp/repos/verl/` (commit `37ff251e`, ~2114 commits)
- SDPO fork: `/home/baris/repos/rlvr/tmp/repos/sdpo/` (commit `c52586b`, 7 commits on fork)

---

## A. Architecture Overview

### A1. Overall Architecture

verl is a Ray-based distributed RL training framework using a **single-controller** pattern. A driver process (the `RayPPOTrainer`) orchestrates the training loop on a single CPU node and dispatches work to Ray-managed GPU worker groups via RPC.

**Key architectural layers:**

| Layer | Location | Purpose |
|---|---|---|
| Single Controller | `verl/single_controller/ray/` | Ray worker group management, dispatch, collocation |
| Trainer | `verl/trainer/ppo/ray_trainer.py` | Training loop orchestration (driver process) |
| Core Algorithms | `verl/trainer/ppo/core_algos.py` | Loss functions, advantage estimators, KL penalties |
| Workers | `verl/workers/` | Actor, Critic, Rollout, Reward workers |
| Engine | `verl/workers/engine/` | FSDP, Megatron, TorchTitan backends |
| Agent Loop | `verl/experimental/agent_loop/` | Multi-turn rollout orchestration |
| Protocol | `verl/protocol.py` | DataProto -- the universal data exchange format |

**File:** `verl/trainer/ppo/ray_trainer.py:221-261` -- `RayPPOTrainer.__init__`

### A2. Training Loop Structure

The training loop is PPO-style but algorithm-agnostic. Flow in `RayPPOTrainer.fit()` (line 1218):

```
for epoch:
  for batch in train_dataloader:
    1. gen_batch = prepare batch for generation
    2. gen_batch_output = async_rollout_manager.generate_sequences(gen_batch)  # vLLM/SGLang
    3. batch = batch.union(gen_batch_output)  # merge rollout into batch
    4. reward_tensor = extract_reward(batch)  # rule-based or RM scoring
    5. old_log_prob = actor_rollout_wg.compute_log_prob(batch)  # recompute logprobs
    6. ref_log_prob = ref_policy_wg.compute_ref_log_prob(batch)  # if KL enabled
    7. values = critic_wg.compute_values(batch)  # if critic enabled
    8. compute_advantage(batch)  # on driver (lightweight)
    9. critic_wg.update_critic(batch)  # if critic
    10. actor_rollout_wg.update_actor(batch)  # policy gradient step
    11. checkpoint_manager.update_weights()  # sync weights to rollout replicas
```

**File:** `verl/trainer/ppo/ray_trainer.py:1218-1606`

### A3. Model Inference during Training

verl uses a **hybrid engine** approach -- the same model is used for both training (FSDP/Megatron) and inference (vLLM/SGLang). The system:

1. **Rollout phase:** Weights are synced to vLLM/SGLang replicas which run as separate async servers via Ray actors
2. **Training phase:** The rollout replicas are "slept" (freed from GPU) while the training FSDP model does forward/backward passes
3. **Weight sync:** After actor update, `checkpoint_manager.update_weights()` pushes new weights to rollout replicas

Rollout backends supported:
- vLLM (async server): `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- SGLang (async server): `verl/workers/rollout/sglang_rollout/async_sglang_server.py`
- TensorRT-LLM: `verl/workers/rollout/trtllm_rollout/`
- Naive HF: `verl/workers/rollout/hf_rollout.py`

### A4. Supported RL Algorithms

Advantage estimators (registered in `verl/trainer/ppo/core_algos.py:88-109`):
- **GAE** (PPO with value network)
- **GRPO** (group relative policy optimization)
- **REINFORCE++** (with and without baseline)
- **REMAX**
- **RLOO** (leave-one-out)
- **OPO** (online policy optimization)
- **GRPO_PASSK** (pass@k variant)
- **GPG**
- **Optimal Token Baseline**
- **TIR Optimal Token Baseline**

Policy loss functions (registered in `verl/trainer/ppo/core_algos.py`):
- `vanilla` (standard PPO clip)
- `gspo`, `sapo`, `gpg`, `clip_cov`, `kl_cov`, `geo_mean`, `cispo`, `bypass_mode`

Both registries are extensible via decorators (`@register_adv_est`, `@register_policy_loss`).

### A5. Comparison with prime-rl

| Aspect | verl | prime-rl |
|---|---|---|
| Orchestration | Ray single-controller (driver + worker groups) | Similar Ray-based |
| Env decoupling | Partial -- `AgentLoop` + `BaseInteraction` for multi-turn; reward functions pluggable | Full decoupling via `verifiers` library |
| Rollout engine | vLLM/SGLang async servers (built-in) | vLLM via verifiers |
| Training backend | FSDP, Megatron, TorchTitan | FSDP |
| Loss extensibility | Registry-based policy loss + advantage estimator | Less modular |
| Data protocol | `DataProto` (TensorDict + non_tensor_batch + meta_info) | Custom |
| Multi-turn | Native `ToolAgentLoop` with tool calling | verifiers REPL environment |
| Teacher/distillation | Not in stock verl (only in SDPO fork) | Not built-in |

---

## B. Environment / Multi-turn Support

### B1. Environment Handling

verl does NOT have a clean gym-like environment abstraction like prime-rl + verifiers. Instead:

1. **Single-turn (default):** Prompts go in, responses come out. Rewards computed by rule-based functions or reward models. The reward functions are registered per `data_source` in the reward manager (`verl/workers/reward_manager/`).

2. **Multi-turn (experimental):** The `AgentLoop` system (`verl/experimental/agent_loop/`) handles multi-turn interactions. There are two key abstractions:
   - `AgentLoopBase` (abstract): Defines the rollout loop per request
   - `BaseInteraction` (`verl/interactions/base.py`): Environment-side interface

### B2. Multi-turn Support

Yes, verl supports multi-turn natively via:

- `ToolAgentLoop` (`verl/experimental/agent_loop/tool_agent_loop.py:96`): Handles tool-calling multi-turn conversations
- `SingleTurnAgentLoop`: Standard single-turn
- `BaseInteraction` interface for environment-side responses

The `AgentLoopManager` manages rollout workers and dispatches generation requests. The agent loop tracks:
- `prompt_ids`, `response_ids`, `response_mask` (1 for LLM tokens, 0 for tool/env tokens)
- `response_logprobs` (only for LLM-generated tokens)
- `num_turns`
- `reward_score`

**File:** `verl/experimental/agent_loop/agent_loop.py:132-150` -- `AgentLoopOutput` dataclass

### B3. Plugging in Custom Environment (ARC-AGI REPL)

To plug in a custom environment, you would:

1. **Implement `BaseInteraction`** (`verl/interactions/base.py:20`):
   - `start_interaction()` -- create session
   - `generate_response(instance_id, messages)` -- process tool call, return (should_terminate, response, score, extra_data)
   - `calculate_score()` -- turn-level scoring
   - `finalize_interaction()` -- cleanup

2. **Register it** via the interaction registry (`verl/interactions/utils/interaction_registry.py`)

3. **Configure** the agent loop to use `tool_agent` mode with your interaction

This is **less clean** than prime-rl + verifiers because:
- The environment is tightly coupled to the agent loop's message format
- No gymnasium-like step/reset interface
- Tool parsing is built into the agent loop rather than being environment-side
- The interaction interface returns text strings, not structured observations

### B4. Environment Abstraction

verl assumes a **text-in, text-out** tool-calling pattern rather than a general RL environment. The `BaseInteraction` is the closest to an environment abstraction, but it's designed around chat messages, not general state/action/reward tuples.

---

## C. Reference Model / KL Infrastructure

### C1. Reference Model Support

Yes. verl has full reference model support:

- Reference model is loaded as a separate FSDP model (`ref_module_fsdp`) colocated on the same GPU workers
- Or, if using LoRA, the reference is the base model without LoRA adapters applied (`ref_in_actor` mode)
- Controlled by `config.algorithm.use_kl_in_reward` (KL penalty in reward) or `config.actor_rollout_ref.actor.use_kl_loss` (KL loss term)

**File:** `verl/trainer/ppo/utils.py:72-76` -- `need_reference_policy()`
**File:** `verl/trainer/ppo/ray_trainer.py:1099-1124` -- `_compute_ref_log_prob()`

### C2. Reference Logprob Computation

Reference logprobs are computed via `ref_policy_wg.compute_ref_log_prob(batch)`, which:
1. Does a forward pass through the reference model
2. Extracts per-token log probabilities
3. Returns them as `ref_log_prob` in a DataProto

**File:** `verl/workers/actor/dp_actor.py:424-506` -- `DataParallelPPOActor.compute_log_prob()`

### C3. KL Divergence Types

Supported KL penalty types (in `verl/trainer/ppo/core_algos.py:1841-1902`):

| Type | Formula | Notes |
|---|---|---|
| `kl` / `k1` | `logprob - ref_logprob` | Forward KL estimate |
| `abs` | `|logprob - ref_logprob|` | Absolute difference |
| `mse` / `k2` | `0.5 * (logprob - ref_logprob)^2` | MSE / unbiased gradient |
| `low_var_kl` / `k3` | `exp(ref-log) - (ref-log) - 1` | Low-variance KL approx |
| `full` | Full distribution KL | **NOT IMPLEMENTED** |
| `k1+`, `k3+` etc | Straight-through trick for unbiased gradients | |

These all operate on **sampled-token log probabilities only**, not full distributions.

### C4. Teacher Model Infrastructure

Stock verl has NO built-in teacher model concept. The reference model infrastructure supports:
- Same architecture as the actor
- Separate weights (full copy or LoRA base)
- Colocated on same GPUs

To add a different-size teacher, you would need to:
1. Create a new worker group for the teacher model
2. Allocate separate GPU resources
3. Implement forward pass for teacher logprob extraction

This is a **significant engineering effort** (~500+ LOC).

---

## D. Loss Function and Extensibility

### D1. Policy Gradient Loss Computation

Policy loss is computed in `DataParallelPPOActor.update_policy()` (`verl/workers/actor/dp_actor.py:508-676`):

1. Split batch into mini-batches (PPO epochs x mini_batch_size)
2. For each micro-batch:
   - Forward pass to get `log_prob` (and optionally entropy)
   - Call `policy_loss_fn(old_log_prob, log_prob, advantages, response_mask, ...)`
   - Optionally add entropy bonus and KL loss
   - Backward pass + gradient accumulation
3. Optimizer step with grad clipping

### D2. Custom Loss Terms

Yes, custom losses can be added via:

1. **Policy loss registry:** `@register_policy_loss("my_loss")` decorator in `core_algos.py`. The loss function receives `old_log_prob`, `log_prob`, `advantages`, `response_mask`, `config`, `rollout_is_weights`.

2. **Additional loss terms in update_policy:** The `update_policy` method already supports:
   - Entropy bonus (`entropy_coeff`)
   - KL loss (`use_kl_loss` + `kl_loss_coef`)
   - These are added to `policy_loss` before `loss.backward()`

3. **To add a new auxiliary loss**, you would modify `dp_actor.py:update_policy()` to compute and add the auxiliary term. This requires modifying the actor worker code (~30-50 LOC).

### D3. Access to Full Logits

**Stock verl:** The `_forward_micro_batch` method computes logits internally but only returns:
- `log_probs` (per sampled token)
- `entropys` (optional, from full logits)
- `sum_pi_squared` (optional, from full logits)

The full logits are available inside `_forward_micro_batch` (line 258: `logits_rmpad = output.logits`) but are not exposed to the caller.

**SDPO fork:** Modified `_forward_micro_batch` to optionally return:
- `all_logps` (full vocabulary log probs per token)
- `topk_logps` + `topk_indices` (top-k log probs)

This is controlled by `return_all_logps` and `distill_topk` parameters.

### D4. Modularity of Loss Computation

The loss computation is **reasonably modular**:

- **Advantage estimators**: Fully pluggable via registry (`@register_adv_est`)
- **Policy losses**: Fully pluggable via registry (`@register_policy_loss`)
- **KL penalties**: Computed as separate terms, configurable type
- **The forward pass**: Requires modification to expose new outputs (logits, teacher logprobs)
- **The training loop**: Changes to `update_policy()` in `dp_actor.py` for adding loss terms

The main weakness is that the `_forward_micro_batch` method is monolithic (~270 lines) and modifications require careful handling of padding, sequence parallelism, and fused kernels.

---

## E. Data Pipeline

### E1. Data Structures

**`DataProto`** (`verl/protocol.py:328-339`) is the universal data exchange format:

```python
@dataclass
class DataProto:
    batch: TensorDict        # PyTorch TensorDict for tensors (same batch size)
    non_tensor_batch: dict    # numpy arrays for non-tensor data (strings, dicts, etc.)
    meta_info: dict           # metadata (config, timing, etc.)
```

Key tensor fields that flow through the pipeline:
- `input_ids`, `attention_mask`, `position_ids` -- model inputs
- `prompts`, `responses` -- separated prompt/response
- `response_mask` -- mask for response tokens (1=LLM, 0=env)
- `old_log_probs`, `ref_log_prob` -- logprobs
- `values`, `advantages`, `returns` -- critic outputs
- `token_level_scores`, `token_level_rewards` -- rewards

Key non-tensor fields:
- `uid` -- unique ID per prompt (for GRPO grouping)
- `data_source` -- reward function routing
- `reward_model` -- ground truth, extra info
- `multi_modal_inputs` -- vision data
- `raw_prompt` -- original chat messages (SDPO addition)

### E2. Threading Extra Data

Yes, the `DataProto` design makes it straightforward to thread extra data:

- **Add tensor data:** `batch.batch["teacher_logprobs"] = teacher_logprobs_tensor`
- **Add non-tensor data:** `batch.non_tensor_batch["privileged_info"] = np.array([...])`
- **Union operation:** `batch = batch.union(extra_data)` merges two DataProtos
- **Pop/select:** `batch.pop(batch_keys=[...])` removes keys, `batch.select(batch_keys=[...])` keeps only specified keys

This is a **major strength** of verl's design -- SDPO leverages this extensively to thread `teacher_input_ids`, `teacher_attention_mask`, `teacher_position_ids`, `self_distillation_mask` through the pipeline.

### E3. Batching/Packing

- **Standard batching:** Left-padded prompts + right-padded responses
- **Dynamic batching:** `use_dynamic_bsz=True` packs sequences to maximize GPU utilization
- **Remove padding:** `use_remove_padding=True` uses flash attention varlen
- **Sequence length balancing:** Reorders batch so each DP rank gets similar total tokens
- **Ulysses sequence parallelism:** Splits long sequences across GPUs

---

## F. SDPO Integration Points

### F1. What SDPO Modifies in verl

SDPO's fork modifies the following verl files:

| File | Changes |
|---|---|
| `verl/workers/actor/dp_actor.py` | Added `teacher_module` field, `_update_teacher()` EMA method, modified `_forward_micro_batch()` to accept `return_all_logps`/`distill_topk`/`module` params, modified `update_policy()` for SDPO loss path |
| `verl/workers/config/actor.py` | Added `SelfDistillationConfig` dataclass |
| `verl/trainer/ppo/core_algos.py` | Added `compute_self_distillation_loss()` function (~100 LOC) |
| `verl/trainer/ppo/ray_trainer.py` | Added `_maybe_build_self_distillation_batch()`, `_collect_feedback()`, `_collect_solutions_by_uid()`, `_get_solution()`, `_remove_thinking_trace()` methods |
| `verl/trainer/main_ppo.py` | Added logic to detect SDPO mode and force `ActorRolloutRef` role |
| `verl/trainer/config/actor/actor.yaml` | Added `self_distillation` config block, `sdpo` as loss_mode option |
| `verl/trainer/config/sdpo.yaml` | New SDPO-specific config file |
| `verl/workers/fsdp_workers.py` | Teacher module initialization (reuses ref model as teacher, or creates TrustRegionTeacher) |
| `verl/utils/reward_score/feedback/` | New directory with feedback computation for math, code, GPQA, MCQ, tooluse |

**Estimated total SDPO modifications: ~800-1000 LOC** across all files.

### F2. How SDPO Implements Self-Teacher

The SDPO self-teacher mechanism works as follows:

1. **Teacher model = reference model** by default. In `fsdp_workers.py:905`, `self.actor.teacher_module = self.ref_module_fsdp`. The ref model is already loaded and colocated.

2. **EMA update** (optional): After each actor update, `_update_teacher()` does exponential moving average: `teacher = (1-rate)*teacher + rate*student` (`dp_actor.py:132-151`).

3. **Trust-region teacher** (alternative): `TrustRegionTeacher` wraps both ref and student, mixing their outputs with a coefficient.

4. **Reprompting:** The driver builds a teacher prompt by:
   - Taking the original problem prompt
   - Appending a successful peer solution (from same group) if available
   - Appending environment feedback (from reward computation) if available
   - Using configurable templates (`reprompt_template`, `solution_template`, `feedback_template`)

5. **Teacher forward pass:** During `update_policy()`, the teacher model processes the reprompted input + student response to get teacher logprobs on the same response tokens, but conditioned on privileged info.

**File:** `sdpo/verl/trainer/ppo/ray_trainer.py:672-796` -- `_maybe_build_self_distillation_batch()`
**File:** `sdpo/verl/workers/actor/dp_actor.py:808-848` -- teacher forward pass in update_policy

### F3. Privileged Info Injection

SDPO injects privileged info by constructing **different input sequences** for the teacher:

```
Teacher input: [system_prompt] + [reprompted_user_msg] + [student_response]
                                  ^--- contains solution + feedback
Student input: [system_prompt] + [original_user_msg] + [student_response]
```

Both teacher and student share the same `response` tokens. The teacher processes a longer prefix (due to solution/feedback text) but evaluates logprobs on the same response tokens.

This is implemented by:
- `teacher_input_ids`: Tokenized reprompted prompt + response
- `teacher_attention_mask`: Corresponding mask
- `teacher_position_ids`: Recomputed positions
- `self_distillation_mask`: Per-sample mask (1 if teacher has privileged info, 0 otherwise)

These are threaded through `DataProto.batch` and consumed in `dp_actor.py:update_policy()`.

### F4. Diff Summary: Stock verl vs SDPO

Files **only in SDPO** (not in stock verl):
- `verl/utils/reward_score/feedback/` -- environment feedback computation
- `verl/trainer/config/sdpo.yaml` -- SDPO config
- `verl/utils/memory_buffer.py` -- memory buffer utility

Files **modified** by SDPO:
- `verl/workers/actor/dp_actor.py` -- ~250 LOC added
- `verl/workers/config/actor.py` -- ~70 LOC added (SelfDistillationConfig)
- `verl/trainer/ppo/core_algos.py` -- ~110 LOC added (compute_self_distillation_loss)
- `verl/trainer/ppo/ray_trainer.py` -- ~200 LOC added (self-distillation batch building)
- `verl/trainer/main_ppo.py` -- ~10 LOC added
- `verl/workers/fsdp_workers.py` -- ~15 LOC added
- `verl/trainer/config/actor/actor.yaml` -- ~75 LOC added

### F5. Could SDPO's Modifications be Applied to Latest verl?

**Partially.** The SDPO fork is based on an older verl version. Key differences:

1. **verl has evolved significantly** -- new files like `engine_workers.py`, `torchtitan/`, QAT support, etc. are not in SDPO.
2. **SDPO requires legacy worker implementation** -- the `main_ppo.py` modification explicitly checks: `"SDPO requires the legacy worker implementation to colocate the teacher."` (line 134-136).
3. **The core SDPO logic is modular** -- the `compute_self_distillation_loss`, config dataclass, and batch building can be ported.
4. **The `_forward_micro_batch` changes** are the trickiest to port as verl's version continues to evolve.

**Recommendation:** Rather than rebasing SDPO, port the SDPO concepts (~800 LOC) to latest verl. The design patterns are well-isolated.

---

## G. Code Quality and Maturity

### G1. Codebase Size

- **107,916 lines** of Python across **407 files** in `verl/`
- **26,887 lines** across **148 test files**
- **2,114 commits** total

### G2. Documentation Quality

- Extensive README with architecture diagrams
- readthedocs site referenced in code
- Good inline docstrings on public methods
- Hydra config files are well-commented (especially in SDPO fork)
- Architecture docs reference: `verl.readthedocs.io`

### G3. Test Coverage

- 148 test files covering core functionality
- Tests for protocol, workers, algorithms
- Integration tests for Ray distributed training
- Not comprehensive -- many experimental features lack tests

### G4. Community Size and Activity

- ByteDance/Volcengine backed
- 2114 commits (active development)
- Multiple contributors (SGLang Team, ModelBest Inc.)
- Active GitHub issues and PRs
- Used as base by SDPO (Hubotter et al., 2026)

---

## Assessment: OPD Variant Implementation Difficulty

### Variant 1: Standard OPD (Teacher-Student)

**Description:** Larger teacher model scores student rollouts with per-token logprobs. Student learns from KL divergence to teacher.

**Difficulty: HARD (~600-800 LOC)**

Challenges:
- verl's reference model infrastructure assumes **same architecture** as actor
- A larger teacher requires a **separate worker group** with different GPU allocation
- Need to modify `ResourcePoolManager` to allocate teacher GPUs
- Need new worker type for teacher forward passes
- Teacher logprobs must be threaded through `DataProto`
- Loss computation needs teacher logprobs (can reuse SDPO's `compute_self_distillation_loss`)

Key modifications:
- `main_ppo.py`: Add teacher worker creation
- `ray_trainer.py`: Add teacher logprob computation step in training loop
- `dp_actor.py`: Add teacher KL loss term in `update_policy()`
- New teacher worker class or reuse ref policy worker with different model
- Config changes for teacher model path, GPU allocation

### Variant 2: Privileged-Info Self-Distillation (OPSD-style)

**Description:** Same model as teacher+student. Teacher sees privileged info (correct answer). Different prompts.

**Difficulty: MODERATE (~400-500 LOC)**

This is **exactly what SDPO implements**. The approach:
- Teacher = actor model (or EMA copy)
- Teacher prompt includes privileged info (correct answer, peer solutions)
- Student prompt is the original problem
- Loss = KL(student || teacher) on response tokens

Can be implemented by porting SDPO's modifications directly:
- `SelfDistillationConfig` dataclass
- `_maybe_build_self_distillation_batch()` -- modify for ARC-AGI specific privileged info
- `compute_self_distillation_loss()`
- `_forward_micro_batch` modifications for full logit access
- Teacher module initialization in `fsdp_workers.py`

### Variant 3: SDPO-style Env Feedback Distillation

**Description:** Teacher re-prompted with environment feedback + peer solutions.

**Difficulty: MODERATE (~500-600 LOC)**

This is SDPO itself. Required:
- Everything from Variant 2
- Plus: environment feedback collection (`verl/utils/reward_score/feedback/`)
- Plus: solution collection from peer group (`_collect_solutions_by_uid`)
- Plus: template-based reprompting

For ARC-AGI, the feedback would come from the REPL environment (grid comparison, test case results). Need to:
- Implement ARC-AGI specific feedback in the reward manager
- Thread feedback through `reward_extra_infos_dict`
- Build ARC-AGI specific reprompt templates

### Variant 4: RLTF-FM Auxiliary Loss

**Description:** Auxiliary cross-entropy loss to predict environment feedback.

**Difficulty: EASY-MODERATE (~200-300 LOC)**

verl's loss computation in `dp_actor.py:update_policy()` is already structured to add auxiliary losses (entropy, KL loss are examples). Adding a feedback prediction loss:

1. Thread feedback tokens through `DataProto.batch` (as `feedback_input_ids`, `feedback_labels`)
2. In `update_policy()`, after the policy forward pass, compute:
   ```python
   feedback_logits = model(feedback_input_ids)
   aux_loss = F.cross_entropy(feedback_logits, feedback_labels)
   policy_loss += aux_loss * feedback_loss_coef
   ```
3. Config addition for `feedback_loss_coef`

Key modifications:
- `ray_trainer.py`: Build feedback prediction data in training loop
- `dp_actor.py`: Add auxiliary loss term (~30 LOC)
- Config changes

### Variant 5: Full-Vocabulary JSD Distillation

**Description:** JSD over full logit distributions (not just sampled-token logprobs).

**Difficulty: MODERATE (~300-400 LOC)**

SDPO already implements this:
- `compute_self_distillation_loss()` in `core_algos.py:1085-1188` supports:
  - Forward KL (`alpha=0.0`)
  - Reverse KL (`alpha=1.0`)
  - Generalized JSD (`0 < alpha < 1`)
  - Full-logit mode (`full_logit_distillation=True`)
  - Top-k approximation (`distillation_topk`)
  - Tail bucket correction (`distillation_add_tail`)

The `_forward_micro_batch` modification to return `all_logps` or `topk_logps` is already in SDPO.

Key concern: **Memory.** Full vocabulary log probs are `(batch, seq_len, vocab_size)` -- for a 152k vocab model, this is ~2.3 GB per batch of 8 sequences with 2048 response length. SDPO's top-k approximation (`distillation_topk=100`) reduces this by ~1500x.

---

## Comparison with prime-rl for ARC-AGI Use Case

| Aspect | verl | prime-rl + verifiers |
|---|---|---|
| **Multi-turn ARC-AGI REPL** | Possible via `BaseInteraction` + `ToolAgentLoop`, but requires adaptation. Less clean than verifiers. | Native via verifiers `ReasoningEnv`. Already working. |
| **Environment decoupling** | Partial. Environment logic lives inside verl's agent loop. | Full. Environments are verifiers objects, completely separate from training. |
| **Distillation support** | SDPO fork provides a solid foundation for OPSD-style distillation. Stock verl has ref model + KL only. | No built-in distillation. Would need custom implementation from scratch. |
| **Full-logit access** | SDPO fork already implements this. Stock verl needs modification. | Would need modification. |
| **Teacher model support** | Ref model infrastructure exists (same architecture). Different-size teacher needs new worker group. | No infrastructure for teacher models. |
| **Loss extensibility** | Registry-based, well-structured. Easy to add new losses. | Less modular. |
| **Scalability** | Battle-tested at ByteDance scale. Megatron + FSDP + multi-node. | Smaller scale. |
| **Operational complexity** | Higher. Ray cluster, worker groups, hybrid engine. | Lower. Simpler setup. |
| **Migration effort** | Must re-implement ARC-AGI REPL env as `BaseInteraction`. Port verifiers environment logic. | Already working. |

---

## Recommendation

### For ARC-AGI + On-Policy Distillation

**Recommended approach: Implement OPD in prime-rl, borrowing SDPO's loss function design.**

Rationale:

1. **We already have a working ARC-AGI REPL environment** in prime-rl + verifiers. Migrating to verl would require re-implementing this as a `BaseInteraction`, which is non-trivial and provides no benefit.

2. **SDPO's core distillation logic is portable.** The `compute_self_distillation_loss()` function (~100 LOC), the reprompting batch builder (~200 LOC), and the teacher forward pass logic (~50 LOC) can be ported to any framework. These are not deeply coupled to verl's architecture.

3. **verl's main advantages** (scale, Megatron, hybrid engine) are not critical for our current ARC-AGI experiments (single-node, small models).

4. **verl's main disadvantage** is that it is a large, complex codebase (108K LOC) with significant operational overhead. Debugging issues in the Ray-based distributed system adds friction.

### If We Must Use verl

If we need verl's scale or community, the path is:

1. **Start with SDPO's fork** as the base (not stock verl)
2. **Implement ARC-AGI REPL** as a `BaseInteraction` class
3. **Port the verifiers environment logic** into verl's interaction framework
4. **Use SDPO's distillation infrastructure** directly for OPSD variants
5. **Estimate: 2-3 weeks** of engineering work for the full migration

### What to Port from SDPO to prime-rl

If staying with prime-rl, port these specific components:

1. **`compute_self_distillation_loss()`** from `sdpo/verl/trainer/ppo/core_algos.py:1085-1188`
   - Supports forward KL, reverse KL, JSD
   - Top-k and full-vocabulary modes
   - IS ratio clipping
   - ~100 LOC, framework-agnostic

2. **Reprompting logic** from `sdpo/verl/trainer/ppo/ray_trainer.py:672-796`
   - Peer solution collection from rollout group
   - Environment feedback integration
   - Template-based teacher prompt construction
   - ~130 LOC, framework-agnostic

3. **EMA teacher update** from `sdpo/verl/workers/actor/dp_actor.py:132-151`
   - Simple exponential moving average of model weights
   - ~20 LOC, completely portable

4. **SelfDistillationConfig** from `sdpo/verl/workers/config/actor.py:39-99`
   - Configuration dataclass with validation
   - ~60 LOC

**Total portable code: ~310 LOC** -- everything else is verl-specific plumbing.
