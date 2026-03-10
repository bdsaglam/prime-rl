# On-Policy Distillation Papers: Comparative Overview

One core insight: **train on the student's own outputs with dense per-token supervision**. This document maps the full landscape — 7 papers with deep notes, ~13 more from a literature review, plus foundational lineage.

---

## At a Glance: Core Papers (Deep Notes Available)

| Paper | Year | Shorthand | Core Idea | Needs External Teacher? | Needs Ground Truth? | Multi-Turn? |
|-------|------|-----------|-----------|------------------------|--------------------:|-------------|
| [GKD](gkd-agarwal-2023.md) (Agarwal et al.) | 2024 | GKD | On-policy sampling + flexible divergence | Yes (larger model) | No | No |
| [OPSD](opsd-zhao-2026.md) (Zhao et al.) | 2026 | OPSD | Same model as teacher via ground-truth conditioning | No | Yes | No |
| [SDFT](sdft-shenfeld-2026.md) (Shenfeld et al.) | 2026 | SDFT | EMA teacher + ICL-based demos; continual learning | No | Yes (demos) | No |
| [SDPO](sdpo-hubotter-2026.md) (Hubotter et al.) | 2026 | SDPO | Self-teacher from env feedback + peer solutions | No | No | No |
| [pi-Distill](pi-distill-penaloza-2026.md) (Penaloza et al.) | 2026 | pi-Distill | Joint teacher-student optimization with PI | No | No (uses traces) | Yes (agentic) |
| [RLTF](rltf-song-2026.md) (Song et al.) | 2026 | RLTF | Distill from text feedback; feedback modeling aux loss | No | No | Yes (2-turn) |
| [RL^2F / SML](meta-learning-klissarov-2026.md) (Klissarov et al.) | 2026 | RL^2F | Meta-learning for in-context plasticity | No (same model) | No | Yes (multi-turn) |

### Wider Landscape (From Literature Review)

| Paper | Year | Shorthand | Core Idea | Priority |
|-------|------|-----------|-----------|----------|
| MiniLLM (Gu et al.) | 2024 | MiniLLM | Reverse-KL via policy gradient; gradients through sampling | HIGH |
| ImitKD (Lin & Xia) | 2020 | ImitKD | Imitation learning for autoregressive KD; GKD baseline | LOW |
| Context Distillation (Snell et al.) | 2022 | CtxDistill | Internalize privileged context (CoT) via SFT; precursor to OPSD | MEDIUM |
| Qwen3 Technical Report | 2025 | Qwen3-OPD | OPD at 10x lower cost than RL on AIME'24 | MEDIUM |
| Thinking Machines blog (Lu) | 2025 | Tinker | Practical reverse-KL OPD; 50-100x compute savings claim | MEDIUM |
| POPE (Qu et al.) | 2026 | POPE | Privileged oracle prefixes for on-policy RL exploration | MEDIUM |
| Nudging / NuRL (Chen et al.) | 2025 | Nudging | Self-generated hints for zero-reward problems | LOW-MED |
| Feedback Descent (Lee et al.) | 2025 | FeedDesc | Text-space optimization via pairwise comparison (no weight updates) | LOW |
| Variational Reasoning (Zhou et al.) | 2025 | VarReason | CoT as latent variables in VI/ELBO framework; unifies KD+RL views | MEDIUM |
| ExOPD (Yang et al.) | 2026 | ExOPD | Reward extrapolation — student surpasses teacher via scaled KL | HIGH |
| GAD (Ye et al.) | 2025 | GAD | GAN-based on-policy distillation from black-box teacher (no logits) | HIGH |
| RLAD (Zhang et al.) | 2026 | RLAD | Trust-region selective imitation — apply teacher KL only when helpful | HIGH |
| OVD (Xiong et al.) | 2026 | OVD | Verbal (trajectory-level) scores replace token-level KL; cuts memory | HIGH |
| X-KD (Cai & Yuan) | 2026 | X-KD | Inverse RL to recover teacher's reward, then reward-regularized KD | MEDIUM |

See [literature-review-report.md](literature-review-report.md) for full details on each.

---

## The Evolutionary Tree

```
Foundational Lineage
  DAgger (2011) — On-policy imitation learning; eliminates train-test mismatch
  SeqKD (2016) — Sequence-level KD; off-policy baseline all OPD papers compare against
  LUPI (2009) — Learning using privileged information; theoretical framework for teacher-student asymmetry
  RL^2 (2016) — Meta-RL via slow outer loop; inspires RL^2F's meta-learning framing
  Born-Again Networks (2018) — Same-architecture distillation can surpass the teacher
  HER (2017) — Hindsight relabeling of failures; inspires POPE/Nudging

Core OPD Methods
  GKD (2024) — Foundational framework
    │  On-policy sampling + teacher scores + flexible divergence
    │
    ├── MiniLLM (2024) — Concurrent. Reverse-KL via policy gradient (gradients through sampling)
    │     More complex but explicitly mode-seeking. Microsoft code available.
    │
    ├── Standard OPD (Qwen3 report, Tinker blog, 2025)
    │     Separate larger teacher, 10-100x savings over RL
    │
    ├── Self-Distillation wave (Jan-Feb 2026) — No external teacher
    │     │  Key insight: information asymmetry replaces capability asymmetry
    │     │
    │     ├── OPSD — Frozen teacher, ground-truth conditioning
    │     ├── SDFT — EMA teacher, demo-based ICL conditioning
    │     ├── SDPO — EMA teacher, env feedback + peer solutions
    │     ├── pi-Distill — Joint optimization, action traces as PI
    │     ├── RLTF — Feedback-conditioned self-distillation + feedback modeling
    │     └── RL^2F — Meta-learning: train the "learning from feedback" skill itself
    │
    └── Extensions & Alternatives (2025-2026)
          ├── ExOPD — Reward extrapolation: student surpasses teacher via scaled KL
          ├── RLAD — Selective imitation: apply teacher KL only when helpful (trust-region)
          ├── OVD — Trajectory-level verbal scores instead of token-level KL
          ├── GAD — GAN-based OPD when teacher logits unavailable (black-box)
          ├── X-KD — Inverse RL to recover teacher's reward function
          ├── POPE — Oracle solution prefixes for hard-problem exploration
          └── Nudging — Self-generated hints to unlock zero-reward problems

Precursors
  Context Distillation (2022) — Offline self-distillation; internalize CoT into weights
  ImitKD (2020) — Imitation learning for autoregressive KD; mixed on/off-policy
```

---

## What Each Paper Uniquely Contributes

### GKD — The Foundation
- **Unique insight:** KD for autoregressive models is an imitation learning problem (DAgger connection). On-policy data eliminates distribution mismatch.
- **Unique contribution:** Flexible divergence framework. Forward KL (mode-covering, good for greedy decoding), reverse KL (mode-seeking, good for instruction tuning), JSD (bounded, balanced). Task-dependent choice matters.
- **Key number:** 5% on-policy data outperforms 100% supervised data.
- **Limitation:** Requires a separate, larger teacher model.

### OPSD — Simplest Self-Distillation
- **Unique insight:** Rationalization is easier than generation. Conditioning on the ground-truth answer makes the same model a useful teacher.
- **Unique contribution:** Frozen teacher for stability; only 1 rollout per problem (vs GRPO's 8). 4-8x token efficiency over GRPO.
- **Key number:** Qwen3-8B avg@16: GRPO 51.3 → OPSD 52.2 on math benchmarks.
- **Limitation:** Requires ground-truth answers. Breaks down below ~4B parameters (model can't meaningfully rationalize).

### SDFT — Continual Learning Champion
- **Unique insight:** An EMA teacher tracks student improvements, avoiding the staleness of a frozen teacher. ICL-based conditioning (demos in context) preserves closeness to the base model's distribution.
- **Unique contribution:** The only paper that directly demonstrates **catastrophic forgetting prevention**. Sequential training on Tool Use → Science → Medical accumulates skills without oscillation.
- **Key number:** 98% OOD accuracy on knowledge acquisition vs 80% for SFT. Preserves reasoning: SFT degrades Olmo-3-7B-Think to 23.5%; SDFT improves it to 43.7%.
- **Limitation:** Requires 7B+ for ICL to be strong enough. Demonstrations must actually improve the model's in-context behavior.

### SDPO — No Ground Truth Needed
- **Unique insight:** Rich environment feedback (error messages, test outputs) + successful peer solutions from the same batch provide enough information asymmetry. No ground truth, no external teacher.
- **Unique contribution:** Logit-level credit assignment (per-token AND per-vocabulary-item). **Test-time self-distillation** — apply SDPO on individual hard problems at inference time, compressing context into weights. Solves problems that best-of-k and multi-turn cannot.
- **Key number:** 48.8% on LiveCodeBench v6 vs GRPO's 41.2%. 3-11x shorter generations. On hard problems, SDPO discovers solutions that neither best-of-k nor multi-turn find within 2750 attempts.
- **Limitation:** Needs at least some successful rollouts in each batch for the peer-solution signal. Model scale dependent (weak below ~1.5B).

### pi-Distill — Joint Training & PI Transfer
- **Unique insight:** Joint optimization of teacher and student (alpha=0.5) is more robust than training either alone. Can distill closed-source frontier models even when reasoning traces are hidden — only action trajectories needed.
- **Unique contribution:** Three PI granularities (tool calls+args, tool calls only, self-generated hints) with analysis of when each works. Variational EM interpretation connecting pi-Distill to latent variable models. KL regime analysis: alpha=0.5 is best in 7/16 scenarios, worst in only 1.
- **Key number:** +11.8% on Travel Planner over SFT w/ CoT + RL (industry standard). Standard RL consistently degrades OOD performance; pi-Distill avoids this.
- **Limitation:** No code released. Complex to implement (dual forward passes, alpha annealing, IS correction).

### RLTF — Leveraging Text Feedback
- **Unique insight:** Text feedback occupies a rich middle ground between sparse scalar rewards and full demonstrations. Two complementary mechanisms: self-distillation from corrected outputs (RLTF-SD) and feedback modeling as auxiliary loss (RLTF-FM).
- **Unique contribution:** Theoretical analysis of why sparse rewards fail (gradient-signal collapse, representation bottleneck). RLTF-FM acts as a "representation preconditioner" that identifies gradient directions invisible to reward-only RL. Enables test-time self-critique.
- **Key number:** Knights & Knaves: base 0.058 → GRPO 0.373 → RLTF-FM **0.880** (+136% over GRPO).
- **Limitation:** Requires a strong feedback provider (uses Qwen3-235B judge). Only tested with 2-turn horizon.

### RL^2F / SML — Meta-Learning for Interactive Reasoning
- **Unique insight:** The ability to learn from feedback is itself a learnable skill, not an emergent property of scale. Baseline frontier models (GPT-5, Gemini 2.5 Pro) are remarkably bad at integrating corrective feedback.
- **Unique contribution:** **In-context plasticity** as a concept. Cross-domain transfer: training on math interactions improves performance on Poker, Wordle, Maze Navigation, ARC-AGI. **Self-improvement pathway**: the model can internalize the teacher and self-critique without privileged information.
- **Key number:** Gemini 2.5 Flash matches Pro in multi-turn math. +7% average cross-domain improvement on 10 diverse agentic tasks (trained only on math).
- **Limitation:** Proprietary models (Gemini). Open-model results (Gemma-3-12B-IT) are weaker. Q-priming for question-asking adds complexity.

---

## Overlapping Themes

### 1. Dense Supervision >> Sparse Rewards
Every paper demonstrates this in its own setting. The consensus is clear: per-token teacher signal provides O(N) bits per episode vs O(1) for RL. The practical impact: 4-100x compute savings depending on the setting.

| Paper | Reported Speedup Over GRPO/RL |
|-------|------------------------------|
| GKD | 2.1x improvement; 5% data beats 100% supervised |
| OPSD | 4-8x token efficiency |
| SDPO | 6x speedup (1h SDPO ≈ 5h GRPO on Chemistry) |
| RLTF | +136% relative improvement on reasoning puzzles |
| RL^2F | Flash → Pro performance gap closed |

### 2. On-Policy Sampling Is Essential
GKD's lambda=1 (purely on-policy) consistently beats mixed or off-policy variants. SDFT's ablation: on-policy distillation reaches ~67% vs ~52% for offline distillation vs ~42% for SFT from teacher samples. SDPO: SFT on self-teacher outputs significantly underperforms on-policy SDPO.

### 3. Model Scale Matters for Self-Distillation
All self-distillation papers report a minimum model size threshold:
- **OPSD:** Marginal at 1.7B, clear gains at 4B+
- **SDFT:** Negative at 3B, positive at 7B+
- **SDPO:** Can underperform GRPO below ~1.5B
- **pi-Distill:** R1-Distill-Llama-8B struggles; Qwen3-8B works well

The bottleneck is **in-context learning ability**: the model must be capable enough that conditioning on privileged information actually shifts its predictions in a useful direction.

### 4. No Gradient Through Sampling (With One Exception)
Almost all papers treat the student's on-policy generation as a non-differentiable process. Gradients flow only through the loss computation, not through the sampling that produced the rollout. This is key for stability and efficiency. **MiniLLM is the exception** — it backpropagates through sampling via policy gradient, requiring stabilization tricks (teacher-mixed sampling, single-step regularization, length-normalized rewards). GKD's simpler approach generally wins.

### 5. Teacher Regularization
When the teacher shares parameters with the student (all self-distillation papers), regularization prevents drift/collapse:
- **OPSD:** Frozen teacher (simplest)
- **SDFT:** EMA (tracks improvements, smooths variance)
- **SDPO:** EMA or trust-region
- **pi-Distill:** Joint optimization with KL penalty; alpha annealing

### 6. Catastrophic Forgetting Resistance
Multiple papers show on-policy distillation preserves general capabilities better than SFT or pure RL:
- **SDFT:** Explicit continual learning experiments (no forgetting across 3 sequential tasks)
- **SDPO:** Better holdout performance than GRPO (42.4 vs 41.8 avg on IFEval/ArenaHard/MMLU-Pro)
- **pi-Distill:** Standard RL degrades OOD performance; pi-Distill/OPSD avoid this
- **GKD:** Reduces "alignment tax" when combined with RLHF

---

## Where They Diverge

### Source of Privileged Information

```
Ground truth ──────────── OPSD, SDFT, POPE (as prompt prefix)
Peer solutions ─────────── SDPO
Action traces ──────────── pi-Distill
Text feedback ──────────── RLTF, SDPO (env output)
Language feedback ──────── RL^2F (teacher with privileged info provides NL hints)
Teacher's capability ───── GKD, MiniLLM (separate larger model)
Self-generated hints ───── Nudging (model generates hints given gold answer)
Teacher's text output ──── GAD (black-box, no logits — uses discriminator)
Verbal scores ─────────── OVD (teacher gives trajectory-level 0-9 rating)
```

### Teacher Architecture

| Paper | Teacher Type | Evolves During Training? |
|-------|-------------|------------------------|
| GKD | Separate larger model | No |
| MiniLLM | Separate larger model | No |
| OPSD | Same model, frozen | No |
| SDFT | Same model, EMA | Yes (smoothly) |
| SDPO | Same model, EMA | Yes (smoothly) |
| pi-Distill | Same model, joint optimization | Yes (directly trained) |
| RLTF | Same model, feedback-conditioned | Implicitly |
| RL^2F | Same model with privileged info | Implicitly |
| GAD | Black-box teacher + learned discriminator | Discriminator evolves |
| OVD | Teacher as verbal scorer (no logits needed) | No |

### Divergence / Loss Choice

| Paper | Default Loss | Why |
|-------|-------------|-----|
| GKD | Task-dependent (fwd KL, rev KL, JSD) | Comprehensive ablation |
| MiniLLM | Reverse KL via policy gradient | Mode-seeking; gradients through sampling |
| OPSD | JSD(0.5) | Bounded, balanced |
| SDFT | Reverse KL | Mode-seeking, connects to inverse RL |
| SDPO | JSD(0.5) | Symmetric, stable |
| pi-Distill | Reverse KL (in KL penalty term) | Keeps teacher close to student |
| RLTF | AWR-style (no importance weighting) | Stability over correctness |
| RL^2F | GRPO (group-normalized advantages) | Standard RL + multi-turn structure |
| RLAD | Trust-region likelihood ratio (PPO-style) | Only imitate when helpful |
| OVD | Policy optimization on verbal scores | Avoids token-level KL memory cost |
| ExOPD | Scaled KL reward (factor >1) | Student can surpass teacher |
| GAD | GAN discriminator reward | No logits needed |

### Single-Turn vs Multi-Turn

| Single-Turn Focus | Multi-Turn Focus |
|---|---|
| GKD, OPSD, SDFT | RL^2F (meta-learning across turns) |
| SDPO (per-rollout, but test-time multi-step) | RLTF (2-turn distillation) |
| | pi-Distill (multi-turn agentic trajectories) |

---

## Synergies: What Could Be Combined

### SDPO + GRPO Hybrid
Already demonstrated in the SDPO paper. lambda=0.9 (mostly GRPO, some SDPO) helps weak models where SDPO advantages are unreliable. For strong models, pure SDPO is better.

### GKD + RL (Verifiable Rewards)
GKD Section 5 shows how to combine: `(1-alpha) * RL_reward + alpha * teacher_KL`. Both use on-policy data; adding teacher KL requires only one extra forward pass. This is what prime-rl implements with `teacher_tau` / `adv_tau`.

### RLAD-Style Selective Imitation + Any OPD
RLAD (Zhang et al., 2026) shows that applying teacher KL *unconditionally* can hurt when it conflicts with reward. Their trust-region selective imitation — only imitate when it improves the student's update — could be layered on top of any OPD method. Particularly relevant when combining distillation with RL rewards.

### RLTF-FM + Any Self-Distillation
RLTF's feedback modeling (predicting what the environment will say) could be added as an auxiliary loss to any other method. It provides representation-level benefits without changing the main training objective. Particularly promising with SDPO, where environment feedback is already collected.

### ExOPD: Surpassing the Teacher
ExOPD (Yang et al., 2026) shows the teacher need not be a performance ceiling. By scaling the KL reward factor >1, the student extrapolates beyond the teacher. Could be combined with any OPD method where the teacher is known to be suboptimal.

### RL^2F Pre-Training + Task-Specific OPD
RL^2F's cross-domain transfer result suggests a two-stage approach: (1) train the generic "learn from feedback" skill on cheap math/code data using RL^2F, then (2) apply task-specific OPD or RL on the target domain. The model arrives with better in-context plasticity.

### Staged Curriculum Across Methods
Progressively weaken the privileged information:
1. **Strong teacher (GKD):** External model teaches fundamentals
2. **Rich self-distillation (OPSD/SDFT):** Ground-truth conditioning teaches reasoning
3. **Feedback-based (SDPO/RLTF):** Environment feedback teaches self-correction
4. **Pure RL (GRPO):** Sparse rewards for final refinement

---

## The Information Asymmetry Spectrum

All self-distillation papers create a teacher-student gap via information asymmetry. The spectrum from most to least informative:

| Privileged Information | Example | Papers Using It |
|----------------------|---------|----------------|
| Full correct trajectory | Complete solving session with backtracking | RL^2F (teacher turns) |
| Oracle solution prefix | Answer prepended to prompt for RL rollouts | POPE |
| Ground-truth answer + demonstrations | Correct output grid + worked examples | SDFT |
| Ground-truth answer only | "The derivative is 14" | OPSD, Context Distillation |
| Successful peer solution | Another rollout that solved it | SDPO |
| Action trace (no reasoning) | Sequence of tool calls without CoT | pi-Distill |
| Structured text feedback | Error messages, test diffs | RLTF, SDPO |
| Teacher's text output (no logits) | Discriminator distinguishes teacher vs student text | GAD |
| Natural language hints | "Check your handling of rotated inputs" | RL^2F (teacher feedback) |
| Self-generated hints | Model generates hint given gold answer, re-rolls | Nudging |
| Trajectory-level verbal score | Teacher rates output 0-9 | OVD |
| Self-generated hints (weakest) | Model's own summary of what worked | pi-Distill (weakest variant) |

Richer privilege → stronger teacher signal → faster learning, but also risk of teaching "what to conclude" rather than "how to reason." The exploration problem (see [[research-questions-opd]]) remains open.

Note: the spectrum also has a **signal type** axis orthogonal to richness. Token-level logit supervision (GKD, OPSD, SDFT, SDPO) gives the densest gradient but requires memory for full distributions. Trajectory-level signals (OVD, GAD) are sparser but cheaper and work in black-box settings. RLAD's insight: the optimal signal type depends on whether the teacher's direction aligns with the reward — when they conflict, dense token-level imitation can actively hurt.

---

## Gaps and Open Questions Across Papers

### 1. Multi-Turn Self-Distillation
GKD, OPSD, and SDFT all operate on single-turn generation. Real reasoning tasks (ARC-AGI REPL, agentic tool use, code debugging) are inherently multi-turn. RL^2F and RLTF address multi-turn but with different framings (meta-learning vs 2-turn distillation). **No paper fully addresses on-policy self-distillation across extended multi-turn trajectories with environment interaction.**

### 2. Curriculum Over Difficulty
No paper systematically varies task difficulty during training. OPSD, SDFT, and SDPO all note that the teacher signal degrades on problems beyond the model's comprehension. A curriculum (easy → hard) is noted as future work in multiple papers but not implemented. POPE and Nudging partially address this by unlocking zero-reward problems, but don't offer a principled curriculum.

### 3. Process vs Outcome Teaching
All methods distill what the teacher concludes (per-token distributions), not how the teacher reasons (exploration, backtracking, hypothesis testing). See [[research-questions-opd]] Idea D: using the student's own corrected trajectory as PI could teach the reasoning process, not just conclusions.

### 4. Combination of Multiple Feedback Sources
Each paper uses one type of privileged information. In practice, you often have multiple signals simultaneously (ground-truth answers + environment feedback + peer solutions + text critiques). **No paper systematically combines multiple PI sources.**

### 5. Beyond 8B Scale
All experiments cap at 8B-14B parameters. Whether self-distillation benefits continue growing at 70B+ is unknown. GKD shows the performance gap between divergences shrinks with larger students, suggesting diminishing returns.

### 6. The O(N) Bits Claim
The "dense supervision provides O(N) bits per episode" claim (Thinking Machines blog) is repeated across papers but likely overstated. Not all tokens are equally informative. At most positions, teacher and student agree (zero gradient). The actual information content is concentrated at decision points. Worth quantifying empirically.

### 7. When On-Policy KL Hurts
RLAD (Zhang et al., 2026) shows that naive KL imitation applied unconditionally can *conflict* with reward maximization — sometimes the teacher's direction is worse than what RL alone would find. Their selective imitation (only apply KL when it helps) outperforms fixed-weight OPD. **The narrative is shifting from "on-policy is always better" to "on-policy is powerful but must be applied carefully."**

### 8. Black-Box Distillation
Most methods require teacher logits. In practice, frontier models (GPT-5, Claude) are black boxes. GAD (Ye et al., 2025) addresses this via a learned discriminator, and OVD (Xiong et al., 2026) uses verbal scores instead of logits. But the black-box setting remains underexplored relative to the white-box case.

### 9. Unified Theoretical Framework
Variational Reasoning (Zhou et al., 2025) and pi-Distill's EM interpretation both attempt to unify KD and RL under a single probabilistic framework. But we lack guarantees on when on-policy strictly dominates off-policy, how to optimally set the KL weight, or how self-distillation's information asymmetry interacts with model capacity.

---

## Quick Decision Guide

**Which paper to start with for implementation?**

- **Have a larger teacher model (white-box)?** → GKD (simplest, most proven)
- **Have a larger teacher model (black-box, no logits)?** → GAD (learned discriminator)
- **Have ground-truth answers, want simplicity?** → OPSD (frozen teacher, 1 rollout)
- **Need continual learning?** → SDFT (EMA teacher, no forgetting)
- **No ground truth, environment gives feedback?** → SDPO (env feedback + peer solutions)
- **Have action traces from frontier model, no CoT?** → pi-Distill (joint optimization)
- **Want to leverage text feedback explicitly?** → RLTF (self-distillation + feedback modeling)
- **Want general multi-turn interactive reasoning?** → RL^2F (meta-learning)
- **Want test-time adaptation on hard problems?** → SDPO (test-time self-distillation)
- **Want the student to surpass the teacher?** → ExOPD (reward extrapolation)
- **Combining OPD with RL and they conflict?** → RLAD (selective imitation)
- **Token-level KL too expensive (memory)?** → OVD (verbal trajectory scores)
- **Base success rate is near zero?** → POPE or Nudging (privileged exploration)

---

## Recommended Next Reads

Top 5 papers not yet in our deep notes, ranked by relevance (from [literature review](literature-review-report.md)):

1. **OVD** (Xiong et al., ICML 2026) — Verbal scores replace token-KL. Solves the memory bottleneck. +25.7% on math.
2. **GAD** (Ye et al., 2025) — Black-box OPD via GAN. 14B student matches GPT-5-Chat on QA.
3. **RLAD** (Zhang et al., 2026) — Trust-region selective imitation. Fixes the "KL hurts RL" problem.
4. **ExOPD** (Yang et al., 2026) — Reward extrapolation. Student surpasses teacher.
5. **Variational Reasoning** (Zhou et al., 2025) — Unifies KD+RL under VI/ELBO. Principled theoretical framework.

---

## References

### Core Papers (Deep Notes Available)

| Shorthand | Full Citation |
|-----------|--------------|
| GKD | Agarwal et al. (2024). "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes." ICLR 2024. arXiv:2306.13649 |
| OPSD | Zhao et al. (2026). "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models." UCLA/Meta. arXiv:2601.18734 |
| SDFT | Shenfeld et al. (2026). "Self-Distillation Enables Continual Learning." arXiv:2601.19897 |
| SDPO | Hubotter et al. (2026). "Reinforcement Learning via Self-Distillation." arXiv:2601.20802 |
| pi-Distill | Penaloza et al. (2026). "Privileged Information Distillation for Language Models." arXiv:2602.04942 |
| RLTF | Song et al. (2026). "Expanding the Capabilities of RL via Text Feedback." arXiv:2602.02482 |
| RL^2F | Klissarov et al. (2026). "Improving Interactive In-Context Learning from Natural Language Feedback." arXiv:2602.16066 |
| SML | Cook et al. (2026). "Learning to Learn from Language Feedback with Social Meta-Learning." arXiv:2602.16488 |

### Wider Landscape

| Shorthand | Full Citation |
|-----------|--------------|
| MiniLLM | Gu et al. (2024). "Knowledge Distillation of Large Language Models." ICLR 2024. arXiv:2306.08543 |
| ImitKD | Lin & Xia (2020). "Autoregressive Knowledge Distillation through Imitation Learning." EMNLP 2020. |
| CtxDistill | Snell et al. (2022). "Learning by Distilling Context." NeurIPS 2022. arXiv:2209.15189 |
| Qwen3-OPD | Qwen Team (2025). "Qwen3: A System Report." arXiv:2505.09388 |
| Tinker | Lu (2025). "On-Policy Distillation." Thinking Machines blog. |
| POPE | Qu et al. (2026). "POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration." arXiv:2601.18779 |
| Nudging | Chen et al. (2025). "Nudging the Boundaries of LLM Reasoning." ICLR 2026. arXiv:2509.25666 |
| FeedDesc | Lee et al. (2025). "Feedback Descent: Open-Ended Text Optimization via Pairwise Comparison." arXiv:2511.07919 |
| VarReason | Zhou et al. (2025). "Variational Reasoning for Language Models." arXiv:2509.22637 |
| ExOPD | Yang et al. (2026). "Learning beyond Teacher: Generalized OPD with Reward Extrapolation." arXiv:2602.12125 |
| GAD | Ye et al. (2025). "Black-Box On-Policy Distillation of Large Language Models." arXiv:2511.10643 |
| RLAD | Zhang et al. (2026). "Reinforcement-aware Knowledge Distillation for LLM Reasoning." arXiv:2602.22495 |
| OVD | Xiong et al. (2026). "OVD: On-Policy Verbal Distillation." ICML 2026. arXiv:2601.21968 |
| X-KD | Cai & Yuan (2026). "X-KD: General Experiential Knowledge Distillation for LLMs." arXiv:2602.12674 |

### Foundational

| Shorthand | Full Citation |
|-----------|--------------|
| DAgger | Ross, Gordon & Bagnell (2011). "A Reduction of Imitation Learning to No-Regret Online Learning." AISTATS. |
| SeqKD | Kim & Rush (2016). "Sequence-Level Knowledge Distillation." EMNLP. arXiv:1606.07947 |
| Born-Again | Furlanello et al. (2018). "Born-Again Neural Networks." ICML. |
| RL^2 | Duan et al. (2016). "RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning." |
| LUPI | Vapnik & Vashist (2009). "A New Learning Paradigm: Learning Using Privileged Information." |
| HER | Andrychowicz et al. (2017). "Hindsight Experience Replay." NeurIPS. |
