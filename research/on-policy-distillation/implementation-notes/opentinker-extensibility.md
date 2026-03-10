# OpenTinker Extensibility for On-Policy Distillation (OPD)

Research notes on the OpenTinker framework's architecture, existing KL/distillation support, and extensibility for implementing various OPD variants for ARC-AGI training.

Repository: https://github.com/open-tinker/OpenTinker
Paper: arXiv:2601.07376
Last commit examined: 2026-02-20 (0b31a0c)

---

## A. Current Architecture

### A1. Overall Architecture: Client-Server Separation on verl + Ray + vLLM

OpenTinker's key architectural innovation is a **client-server separation** for agentic RL. The training server (GPU cluster) runs verl-based PPO/GRPO training, while environment clients run on separate machines and communicate via HTTP.

```
Client (CPU)                    Server (GPU cluster)
+-----------------+             +-------------------+
| Environment     | -- HTTP --> | HTTP Training     |
| Data Generator  |             | Server (FastAPI)  |
| Reward Fn       |             |   |               |
+-----------------+             |   v               |
                                | RayPPOTrainer     |
                                |   |               |
                                |   +-> Actor/Rollout (vLLM)
                                |   +-> Critic       |
                                |   +-> RefPolicy    |
                                |   +-> RewardModel  |
                                +-------------------+
```

The verl submodule (`volcengine/verl`) provides the core distributed training infrastructure. OpenTinker patches certain verl files in `opentinker/backend_patch/verl/` rather than forking verl directly.

**Key files:**
- Server entry: `opentinker/server/launch_http_server.py` (line 13-222)
- HTTP server: `opentinker/server/http_training_server.py` (line 586-1280)
- Ray trainer: `opentinker/backend_patch/verl/trainer/ppo/ray_trainer.py` (line 325-1764)

### A2. RL Algorithms Supported

From the config and code, OpenTinker supports these advantage estimators:

| Algorithm | Config Value | Notes |
|-----------|-------------|-------|
| PPO (GAE) | `adv_estimator: gae` | Standard PPO with GAE, requires critic |
| GRPO | `adv_estimator: grpo` | Group Relative Policy Optimization, no critic needed |
| GRPO Per-Step | `adv_estimator: grpo_per_step` | Custom OpenTinker addition, per-turn credit assignment for multi-turn (GTPO-inspired) |
| REINFORCE++ | `adv_estimator: reinforce_plus_plus` | Via verl's core_algos |
| REMAX | `adv_estimator: remax` | Requires baseline generation |
| DAPO | Via verl | Supported through verl's advantage estimator registry |

**Code reference:** `opentinker/backend_patch/verl/trainer/ppo/ray_trainer.py:256-322` (compute_advantage function)

The `grpo_per_step` advantage estimator is OpenTinker's novel contribution, implemented in `opentinker/backend_patch/verl/trainer/ppo/per_step_core_algos.py`. It computes cumulative returns from each turn for fine-grained multi-turn credit assignment.

### A3. Training Loop Structure

The training loop follows the standard verl PPO flow (`ray_trainer.py:1295-1764`):

1. Load batch from dataloader (or HTTP client)
2. Generate rollouts (via vLLM, sync or async)
3. Compute response_mask (distinguishing LLM tokens from env tokens)
4. Compute reward (rule-based, model-based, or remote API)
5. Compute old_log_probs (current policy forward pass)
6. **Compute ref_log_prob** (if `use_kl_in_reward` or `use_kl_loss` enabled)
7. Compute values (if critic enabled)
8. Apply KL penalty to rewards (if `use_kl_in_reward`)
9. Compute advantages
10. Update critic
11. Update actor (policy gradient step)

### A4. Multi-Turn Environment Support

OpenTinker has **first-class multi-turn support** via its "agent loop" system:

- **GenericAgentLoop** (`opentinker/server/generic_agent_loop.py`): State machine for LLM-environment interaction (PENDING -> GENERATING -> INTERACTING -> TERMINATED)
- **PerTurnAgentLoopManager** (`opentinker/backend_patch/verl/experimental/agent_loop/per_turn_agent_loop.py`): Expands multi-turn episodes into individual per-turn training samples
- **AbstractGame** (`opentinker/environment/base_game.py`): Clean game abstraction with `reset()`, `step()`, `get_system_prompt()` methods
- **SandboxTool** (`opentinker/server/sandbox_tool.py`): Code execution via external sandbox server

Supported environment types (from README):
| Environment | Type | Multi-turn |
|------------|------|------------|
| Math (single-turn) | Data-dependent | No |
| Math (multi-turn) | Data-dependent | Yes, with code interpreter |
| Geo3K (VLM) | Data-dependent | Both |
| Gomoku | Data-free | Yes |
| AlfWorld | Data-free | Yes |
| Android World | Data-free | Yes |

The response_mask system handles multi-turn training correctly: mask=1 for LLM-generated tokens, mask=0 for environment observation tokens. This is critical for loss computation.

---

## B. Existing Distillation/KL Support

### B1. Reference Model Infrastructure

OpenTinker (via verl) has a **fully functional reference policy** infrastructure:

- **RefPolicy worker**: Separate Ray actor for reference model inference
  - Code: `opentinker/server/http_training_server.py:483-498` (conditional creation)
  - Config: `opentinker/server/config/ref/dp_ref.yaml`
- **When enabled**: Reference policy is created when `algorithm.use_kl_in_reward=True` OR `actor_rollout_ref.actor.use_kl_loss=True`
- **ref_in_actor**: For LoRA models, the reference is the base model without LoRA applied (no separate worker needed)
- **Separate model path**: The ref config explicitly supports different model paths:
  ```yaml
  # ref model is assumed to be identical to actor model. Specify model.path for using a different ref model.
  # Potential use case involves on policy distillation where we calculate KL divergence between student actor
  # and teacher ref
  model: null
  ```
  This comment (`dp_ref.yaml:12-14`) **explicitly acknowledges OPD as a use case** for the reference model infrastructure.

### B2. KL Divergence Usage

Two KL mechanisms exist:

**1. KL-in-Reward (PPO-style):**
- Config: `algorithm.use_kl_in_reward: True`
- Implemented in: `apply_kl_penalty()` at `ray_trainer.py:160-204`
- Computes: `kld = core_algos.kl_penalty(old_log_probs, ref_log_prob)` then `reward = score - beta * kld`
- Supports adaptive KL coefficient via `AdaptiveKLController`
- KL types: "kl", "abs", "mse", "low_var_kl", "full"

**2. KL-in-Loss (GRPO-style):**
- Config: `actor_rollout_ref.actor.use_kl_loss: True`
- KL loss is computed in verl's actor worker (not in OpenTinker's patches)
- Config parameters: `kl_loss_coef` (default 0.001), `kl_loss_type` (default "low_var_kl")

Both mechanisms use **per-token log probabilities** from the reference model (`ref_log_prob`), computed via `ref_policy_wg.compute_ref_log_prob(batch)` or `actor_rollout_wg.compute_ref_log_prob(batch)`.

### B3. Distillation-Specific Code

**There is no dedicated distillation pipeline.** However:

- The infrastructure for scoring student rollouts with a separate model exists (RefPolicy worker)
- The data pipeline (DataProto) can carry `ref_log_prob` alongside `old_log_probs`
- The loss function already supports KL terms (both in-reward and in-loss variants)

What's missing compared to prime-rl's OPD:
- No teacher model that can be a **different** model from the reference
- No `teacher_tau` / `adv_tau` blending weights
- No full-logit distribution access (only per-token logprobs)
- No privileged-information prompting for the teacher
- No dedicated teacher inference server management

### B4. Computing Logprobs from Reference/Teacher

Yes, this is fully supported. The flow:

```python
# ray_trainer.py:1549-1562
if self.use_reference_policy:
    if not self.ref_in_actor:
        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
    else:
        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
    batch = batch.union(ref_log_prob)
```

The `compute_ref_log_prob` method does a forward pass through the reference model on the student's rollout tokens, returning per-token logprobs. This is exactly the "prefill pass" mechanism needed for OPD.

---

## C. Extensibility for OPD

### C1. Adding a Teacher Model That Scores Student Rollouts

**Effort: Medium (3-5 days)**

The path is relatively clear:

1. **Reuse RefPolicy infrastructure**: The `RefPolicy` role already supports a different `model.path` (`dp_ref.yaml:15: model: null`). Setting a different model path would create a separate model for the teacher.

2. **Modify worker creation**: In `http_training_server.py:483-498`, add logic to create a teacher worker (similar to RefPolicy but with its own config section).

3. **Add teacher scoring step**: After student rollout generation, add a `teacher_wg.compute_log_prob(batch)` call. The infrastructure for this already exists -- just need to add it to the training loop at `http_training_server.py:1095-1102`.

4. **Store teacher logprobs**: Add `teacher_log_prob` to the batch via `batch = batch.union(teacher_log_prob)`.

**Challenge**: GPU memory management. The teacher model consumes additional GPU memory. OpenTinker's resource pool system (`ResourcePoolManager`) handles colocated models, but adding a 4th model (actor + rollout + critic + teacher) may require careful memory planning. OpenTinker's `free_cache_engine` mechanism for vLLM could help.

### C2. Loss Function Modularity

**Reasonably modular, but changes needed in verl internals.**

The loss function lives in verl's actor worker (not directly in OpenTinker). OpenTinker patches the trainer loop but the actual policy gradient computation (`update_actor`) happens inside verl's FSDP/Megatron worker code.

To add teacher KL terms:

1. The batch already carries `ref_log_prob` -- adding `teacher_log_prob` follows the same pattern.
2. The actor worker's policy loss would need modification to include: `loss = adv_loss + teacher_tau * KL(teacher || student)`
3. This requires patching verl's actor worker code (e.g., `verl/workers/fsdp_workers/actor.py`), similar to how OpenTinker already patches `ray_trainer.py` and `per_step_core_algos.py`.

**Current loss modes** (`opentinker/server/config/actor/actor.yaml:46-66`):
- `vanilla`: Standard policy gradient
- `clip-cov`: Coverage-based clipping
- `kl-cov`: KL-based coverage
- `gpg`: Generalized Policy Gradient

Adding a `distillation` mode or a `teacher_kl_coef` parameter would fit naturally into this system.

### C3. Data Pipeline Extensibility

**Highly extensible.** The `DataProto` (from verl) is the universal data container with:

- `batch` (TensorDict): Tensor data (input_ids, responses, logprobs, etc.)
- `non_tensor_batch` (dict): Non-tensor data (uid, reward_model, extra_info, etc.)
- `meta_info` (dict): Metadata

The pipeline already carries:
- `old_log_probs` (current policy)
- `ref_log_prob` (reference policy)
- `rollout_log_probs` (from rollout engine, optional)
- `turn_scores` (per-turn rewards for multi-turn)

Adding `teacher_log_prob` or `privileged_info` is straightforward:
```python
batch.batch["teacher_log_prob"] = teacher_log_prob_tensor  # for token-level data
batch.non_tensor_batch["privileged_info"] = privileged_info_array  # for prompt/metadata
```

### C4. Comparison with prime-rl's OPD Support

| Feature | prime-rl | OpenTinker |
|---------|----------|------------|
| Teacher model inference | Built-in (`teacher_gpu_ids`, separate vLLM server) | Must repurpose RefPolicy infrastructure |
| Teacher scoring (prefill logprobs) | Native (`teacher.score_completions()`) | Exists as `compute_ref_log_prob()`, needs extension |
| `teacher_tau` / `adv_tau` blending | Native config parameters | Must be added |
| Privileged-info prompting | Not built-in (but teacher has separate prompt) | Not built-in, requires prompt routing |
| Full-logit distribution | Not supported (per-token logprobs only) | Not supported (same limitation) |
| KL divergence types | Reverse KL on sampled tokens | Multiple: "kl", "abs", "mse", "low_var_kl", "full" |
| Multi-turn environment | Not supported | Native support (agent loops, per-turn training) |
| Code execution/REPL | Not built-in | Built-in (SandboxTool, math tool env) |
| External teacher server | Yes (`orchestrator.teacher_model.client.base_url`) | Not supported |
| GPU allocation | Explicit 3-way split (inference/trainer/teacher) | Resource pools, but no teacher allocation |
| Configuration | TOML-based, simple | Hydra/OmegaConf YAML, complex but flexible |

**Key advantage of prime-rl**: OPD is a first-class feature with ~3 lines of config to enable.
**Key advantage of OpenTinker**: Multi-turn environment support is first-class (critical for ARC-AGI).

---

## D. SDPO Connection

### D1. Shared Infrastructure with SDPO

The SDPO paper (Hubotter et al., 2025) is built on verl. OpenTinker also uses verl as its core training backend. This means:

- Same `DataProto` data container
- Same Ray-based distributed training
- Same FSDP/vLLM integration
- Same actor/critic/reference worker architecture

However, OpenTinker patches verl rather than using it as-is, so there may be version incompatibilities. OpenTinker's verl submodule points to `volcengine/verl.git` without a pinned commit.

### D2. Porting SDPO to OpenTinker

**Effort: Medium (3-5 days), depending on SDPO's verl version.**

SDPO's key requirements:
1. Teacher and student share the same model (different prompts) -- RefPolicy can handle this
2. Teacher sees privileged info (correct answer) -- requires prompt routing in the agent loop
3. KL divergence loss from teacher to student -- existing KL infrastructure covers this

The main work would be:
1. Modifying the rollout generation to produce two passes (teacher prompt + student prompt)
2. Routing the teacher's logprobs into the loss function
3. Handling the privileged information in the prompt construction

### D3. References to SDPO in the Codebase

**None found.** No references to "sdpo", "self-distillation", or "Hubotter" in the codebase.

The comment in `dp_ref.yaml:13-14` is the only explicit mention of distillation:
```yaml
# Potential use case involves on policy distillation where we calculate KL divergence between student actor
# and teacher ref
```

---

## E. Multi-Turn Environment Support

### E1. Multi-Turn Support for ARC-AGI REPL

**Strong native support.** OpenTinker's multi-turn architecture directly applies to ARC-AGI REPL:

- **Agent Loop State Machine** (`generic_agent_loop.py`): PENDING -> GENERATING -> INTERACTING -> TERMINATED maps cleanly to the ARC-AGI task flow (think -> code -> execute -> observe -> iterate)
- **Response Mask**: Properly distinguishes LLM tokens (mask=1) from environment feedback (mask=0)
- **Per-Turn Training**: The `PerTurnAgentLoopManager` can expand multi-turn episodes into individual training samples, solving context length issues
- **Turn-Level Rewards**: `turn_scores` field supports per-turn reward signals

### E2. Tool Calling / Code Execution

- **SandboxTool** (`opentinker/server/sandbox_tool.py`): Executes Python code via external sandbox server, returns stdout+stderr
- **Math Code Interpreter**: Full example with code execution for math problem solving
- **BaseInteraction**: Abstract interaction interface via verl's `BaseInteraction` class
- Tool config is specified via YAML (`interaction_config_path`)

### E3. Environment Abstraction

**Clean abstraction via AbstractGame:**

```python
class AbstractGame(ABC):
    def reset(self, **kwargs) -> str: ...
    def step(self, action: str) -> StepResult: ...
    def get_system_prompt(self) -> str: ...
    def get_initial_user_message(self) -> str: ...
    def generate_initial_state(self) -> Dict[str, Any]: ...
```

This is similar to the Gymnasium interface. Creating an ARC-AGI game environment would require:
1. Subclass `AbstractGame`
2. Implement `reset()` to present a puzzle
3. Implement `step()` to execute code and return grid state
4. Implement `get_system_prompt()` with ARC-AGI instructions

The `GameDataGenerator` wraps games into training data generation automatically.

---

## F. Code Quality and Maturity

### F1. Codebase Size

- **101 Python files**, ~25,920 lines of Python code (opentinker/ directory)
- Plus the verl submodule (not included in count, shared dependency)
- Relatively compact for a full RL training framework

### F2. Contributors and Activity

- **6 contributors**: zhusq20 (33 commits), lwaekfjlk (32), yau (31), Haofei Yu (4), JackSimbol (2), Chen Yi-Shing (1)
- **81 total commits** (as of 2026-02-20)
- **Active**: Latest commit 2026-02-20, actively merging PRs
- Core team appears to be 3 people (UIUC-affiliated based on paper)

### F3. Documentation Quality

- **README**: Good overview with quick-start examples, environment taxonomy, installation
- **Per-example docs**: Each task (math, gomoku, alfworld, etc.) has a dedicated markdown guide
- **Code comments**: Generally well-commented, especially the backend patches
- **Deep Wiki**: External documentation available at https://deepwiki.com/open-tinker/OpenTinker
- **Missing**: No architecture diagram, no API reference, no developer guide for extending

### F4. Test Coverage

- **Minimal**: One test file (`opentinker/tests/test_per_step_core_algos.py`, 283 lines)
- Tests cover the per-step advantage computation (OpenTinker's novel contribution)
- No integration tests, no training loop tests, no HTTP API tests
- Test infrastructure: pytest

---

## G. Assessment: OPD Variant Implementation Effort

### Variant 1: Standard OPD (Teacher-Student)

**Effort: 1-2 weeks**

What exists:
- RefPolicy worker can serve as teacher (different model path supported)
- `compute_ref_log_prob()` scores student rollouts with teacher logprobs
- KL computation infrastructure (multiple divergence types)

What needs building:
- Teacher config section (model path, GPU allocation, temperature)
- `teacher_tau` / `adv_tau` blending in the loss function (requires verl actor worker patch)
- Teacher inference server management (auto-launch like prime-rl, or manual)
- Integration into the HTTP training server's `train_step()` method

### Variant 2: Privileged-Info Self-Distillation

**Effort: 1-2 weeks (incremental on Variant 1)**

Additional work beyond Variant 1:
- Prompt routing: Teacher sees `[system + problem + answer]`, student sees `[system + problem]`
- Modify agent loop to support dual-prompt generation
- The RefPolicy worker already supports same-model inference, so no new model needed
- Key challenge: The prompt routing must happen during rollout, and the teacher forward pass must use the modified prompt

### Variant 3: SDPO-Style Env Feedback Distillation

**Effort: 2-3 weeks**

This is the most natural fit for OpenTinker because it already has multi-turn environments:
- Teacher is re-prompted with environment feedback (error messages, peer solutions)
- OpenTinker's agent loop already collects environment feedback (`turn_scores`, `env_info`)
- Need to: (1) re-prompt teacher with feedback, (2) compute teacher logprobs, (3) add KL loss
- The GenericAgentLoop's state machine can be extended to include a "teacher re-prompting" state

### Variant 4: RLTF-FM Auxiliary Loss

**Effort: 1-2 weeks**

- Add an auxiliary cross-entropy loss head to predict environment feedback tokens
- This requires modifying the actor model's forward pass (verl worker level)
- The environment feedback is already available in the batch (`non_tensor_batch`)
- Relatively self-contained: add auxiliary loss term, add config parameters

### Variant 5: Full-Vocabulary JSD Distillation

**Effort: 2-3 weeks (most difficult)**

- Requires full logit distributions, not just per-token logprobs
- Neither OpenTinker nor verl exposes full logits during training (only log_probs of sampled tokens)
- Would need to modify the rollout engine (vLLM) to return full logits
- JSD computation over full vocabulary is memory-intensive
- May require chunked computation or approximations

This is the hardest variant for any framework that uses vLLM for inference, because vLLM optimizes by only returning logprobs for sampled tokens.

---

## H. Comparison Summary: OpenTinker vs. prime-rl for OPD

| Criterion | OpenTinker | prime-rl |
|-----------|-----------|----------|
| **OPD readiness** | Partial (infrastructure exists, no dedicated pipeline) | Native (first-class feature, 3-line config) |
| **Multi-turn support** | Native, excellent (agent loops, per-turn training) | None |
| **Code execution/REPL** | Built-in (SandboxTool) | None |
| **ARC-AGI fit** | High (multi-turn + tool calling + per-turn rewards) | Low (single-turn only) |
| **Time to basic OPD** | 1-2 weeks | 0 (already works) |
| **Time to privileged-info OPD** | 2-3 weeks | 1-2 weeks (prompt routing needed) |
| **Time to SDPO-style OPD** | 2-3 weeks | 3-4 weeks (need multi-turn first) |
| **Full-logit JSD** | 2-3 weeks | Same difficulty |
| **Maturity** | Early (v0.1.0, 81 commits, 6 contributors) | More mature (~2 years, larger community) |
| **Test coverage** | Minimal (1 test file) | Better (but not extensive) |
| **Documentation** | Adequate for examples, sparse for internals | Better API docs |
| **Architecture complexity** | Higher (client-server, HTTP, Hydra) | Lower (single process, TOML) |

---

## I. Recommendation

### Is OpenTinker viable for OPD?

**Yes, conditionally.**

**For ARC-AGI specifically, OpenTinker has a significant advantage over prime-rl**: its native multi-turn environment support, code execution infrastructure, and per-turn training are exactly what ARC-AGI REPL needs. prime-rl would require building all of this from scratch.

**However, OPD requires 1-2 weeks of development work** to extend OpenTinker's existing KL/reference model infrastructure into a proper teacher-student distillation pipeline. The key modifications are:

1. Add a `teacher` config section (model path, GPU allocation)
2. Add a teacher scoring step in the training loop
3. Add `teacher_tau` / `adv_tau` blending in the loss function
4. Patch verl's actor worker to accept the teacher KL term

### Recommended approach: Hybrid strategy

1. **Use OpenTinker as the base** for ARC-AGI training (multi-turn + REPL is essential)
2. **Port prime-rl's OPD concepts** into OpenTinker:
   - Teacher model scoring via prefill pass (reuse RefPolicy infrastructure)
   - `teacher_tau` / `adv_tau` loss blending
   - Support for external teacher servers (for using larger teacher models)
3. **Start with Variant 2 (privileged-info self-distillation)** as it's most natural:
   - Same model as teacher and student
   - Teacher sees the correct answer (available in ARC-AGI data)
   - No additional GPU memory for a separate model
4. **Iterate toward SDPO-style** (Variant 3) using environment feedback from the REPL

### Risk factors

- **verl version compatibility**: OpenTinker patches verl files, which may break with verl updates
- **Limited test coverage**: Changes to the training loop need careful manual testing
- **Small team**: Only 3 active contributors; bug fixes may be slow
- **Early maturity**: v0.1.0 with 81 commits; expect rough edges

### Estimated total effort

| Phase | Effort | Deliverable |
|-------|--------|-------------|
| ARC-AGI environment | 1 week | AbstractGame subclass for ARC-AGI REPL |
| Basic OPD (self-distillation) | 1-2 weeks | Privileged-info teacher using RefPolicy |
| SDPO-style feedback OPD | 1-2 weeks | Teacher re-prompted with env feedback |
| Full OPD pipeline | 1 week | Config-driven, tested, documented |
| **Total** | **4-6 weeks** | Full OPD for ARC-AGI on OpenTinker |

Compare with prime-rl: OPD works out of the box, but multi-turn environment support would take 4-6 weeks to build from scratch. **The total effort converges** -- the question is whether you'd rather build OPD on top of multi-turn (OpenTinker) or multi-turn on top of OPD (prime-rl).

**Recommendation: Use OpenTinker as the base**, because multi-turn REPL is harder to retrofit than OPD.
