# Handover: OPD Signal Measurement → Deliberative Teaching Training

## TL;DR

We discovered that a teacher model reasoning about a student's rollout before scoring it produces better learning signal than standard OPD with oracle PI (answer + reference solution) — with zero external knowledge. The next step is to validate this in actual training.

## What We Did

### Phase 1: Signal Measurement Infrastructure

Built a pipeline to measure OPD learning signal quality without training:
1. Student (Qwen3-8B) generates rollouts on AIME problems
2. Student scores its own tokens via prefill → baseline logprobs
3. Teacher scores the same tokens conditioned on privileged information (PI) → teacher logprobs
4. Per-token KL divergence = learning signal. Higher |KL| = stronger signal.

### Phase 2: PI Exploration (Dimension 2)

Tested ~15 PI conditions across three configurations. Key results:

**What works:**
- `answer_ref` (answer + reference solution): |KL| ≈ 0.063 (8B self) / 0.065 (32B self) — the standard OPD baseline
- `blind_confidence` (student self-rates 1-10, no answer): |KL| ≈ 0.042-0.046 — 70% of oracle with zero external knowledge
- `blind_diagnosis` (student self-analyzes errors): |KL| ≈ 0.038-0.039

**What doesn't work:**
- System prompt variation: r > 0.97 between variants (negligible effect)
- Student rollout text in PI: copy artifact (teacher reads tokens from context, inflates |KL| without pedagogical value)
- Answer alone: surprisingly weak (|KL| ≈ 0.014-0.016)

**Key insight:** PI signal is model-size-invariant. 8B and 32B self-teachers produce nearly identical |KL| for the same PI.

### Phase 3: Multi-Lens Combination

Different PI types are complementary at the **token level** (r = 0.43-0.65 between lenses). Oracle per-token max of 3 ref-free lenses (0.062) nearly matches answer_ref (0.065). But this requires K scoring passes per rollout.

### Phase 4: Deliberative Teaching (Main Result)

**Core idea:** Give the teacher compute budget to reason about *how to teach*, analogous to CoT for inference.

Pipeline:
1. Teacher generates ~1024-token analysis of student's work (blind — no answer provided)
2. Analysis becomes additional PI context for the scoring pass
3. Teacher's attention flows through its own analysis → better credit assignment

**Results across all three configurations:**

| Config | answer_ref |KL| | blind_delib |KL| | Δ | Cohen's d (delib) | best-of-4 |KL| |
|---|---|---|---|---|---|
| 8B self-OPD (8B→8B) | 0.063 | 0.075 | +19% | 0.40 | 0.082 (+31%) |
| 32B self-teacher (32B scoring 8B rollouts) | 0.065 | 0.072 | +11% | 0.34 | — |
| 32B self-OPD (32B→32B) | 0.064 | 0.076 | +19% | 0.03 | 0.085 (+33%) |

**Key findings:**
1. Blind deliberative beats oracle PI (answer_ref) by ~19% — consistently across all configs
2. Informed deliberation barely helps over blind — knowing the answer doesn't improve the analysis
3. Best-of-N scales: picking best of 4 candidate analyses gives +31-33% over answer_ref
4. Fully self-supervised: single model generates rollout → analyzes → scores. No external knowledge.

**Open concern — Cohen's d:** Deliberative conditions amplify signal uniformly (both correct and incorrect rollouts) rather than selectively. Cohen's d drops from 0.62 (answer_only) to 0.03 (blind_delib) for 32B self-OPD. **Partially resolved by PI placement** (see Phase 5 below).

### Phase 5: PI Placement (Where PI Goes in the Token Sequence)

All prior results used system prompt placement (PI appended to system message). We tested 4 positions:

| PI Condition | system | sys+question | user | **asst_prefix** |
|---|---|---|---|---|
| answer_ref | 0.065 (d=0.54) | 0.066 (d=0.56) | 0.073 (d=0.55) | 0.068 (d=0.54) |
| **blind_deliberative** | 0.072 (d=0.34) | 0.069 (d=0.32) | 0.074 (d=0.24) | **0.087 (d=0.49)** |

**`assistant_prefix` is the clear winner for deliberative PI**: +21% |KL| AND +44% Cohen's d over system placement. The analysis sits as a preamble to the teacher's response, right before the student's rollout tokens — attention flows directly from diagnostic reasoning into scoring. Static PI (answer_ref) is largely placement-insensitive.

**For training**: Inject deliberative analysis as assistant response prefix:
```
[system: prompt] [user: problem] [assistant: {analysis}\n\n--- STUDENT RESPONSE ---\n{rollout}]
```

## Next Step: Training Validation

### Recommended Experiment

**Compare deliberative OPD vs standard OPD in actual student training:**

1. **Baseline**: Standard self-OPD with `answer_only` as PI
   - Why answer_only not answer_ref: isolates the PI effect. answer_only is the minimal external knowledge baseline. If blind_deliberative (zero knowledge) beats answer_only (knows the answer), that's the strongest result.

2. **Treatment**: Self-OPD with `blind_deliberative` as PI
   - Teacher generates analysis of student rollout (no answer), uses as PI for scoring

3. **Stretch**: Cross-teacher OPD with 32B teacher, both conditions

### What to Measure
- Student accuracy on held-out problems over training steps
- Whether the higher |KL| translates to faster/better learning
- Whether the low Cohen's d matters (uniform vs discriminative signal)

### Self-Teacher vs Cross-Teacher
- **Self-teacher (recommended first)**: Cleaner comparison, isolates the PI effect. No confound from cross-model distribution gap. Also more practical (no need for a larger model).
- **Cross-teacher**: The cross-model gap (|KL| = 0.097 with no PI) dominates over PI effects (+0.022 for answer_ref). Deliberative PI may have less marginal impact here, but worth testing.

## Working Conventions

**Use tmux for all commands.** Run everything in named tmux sessions/windows so that:
- Commands persist if the agent's connection drops
- The user can attach to any session and watch live (`tmux switch -t <session>`, `Ctrl-B w`)
- The agent can inspect output programmatically (`tmux capture-pane -t <session>:<window> -p -S -50`)

**Convention:** Use a tmux session called `opd` for all OPD work. Create named windows for each long-running process (training, vLLM servers, log tailing, GPU monitoring). See the training management guide (Section 9) for the full tmux layout.

**Example:**
```bash
# Start a vLLM server in a named window
tmux new-window -t opd -n vllm-8b 'uv run vllm serve Qwen/Qwen3-8B --port 8900 ...'

# Run a script
tmux new-window -t opd -n scoring 'cd /path/to/experiments && uv run python score.py'

# Check output from agent code
tmux capture-pane -t opd:scoring -p -S -50
```

## File Map

### Core Documents
- `FINDINGS.md` — Complete results with tables, methodology, analysis
- `IDEATION.md` — 8 ideas for future work, prioritization, teacher learning approaches
- `HANDOVER.md` — This file
- `../../implementation/prime-rl-training-management-guide.md` — **Training management guide**: metrics dashboard, failure diagnosis, crash recovery, hyperparameter tuning, tmux conventions. Read this before launching any training run.

### Scripts (in order of typical pipeline execution)

**Rollout generation:**
- `rollouts.py` — Generate student rollouts via vLLM
- `problems.py` — Load AIME problems from HuggingFace

**Student self-scoring:**
- `compute_student_logprobs.py` — Compute student's own logprobs via prefill (generic utility)
  - Args: `--model`, `--rollouts`, `--output`, `--concurrency`

**PI generation:**
- `generate_student_reflections.py` — Student self-assessment (confidence, diagnosis)
- `summarize_rollouts.py` — Generate compressed summaries of rollouts
- `generate_deliberative_pi.py` — **Main new script.** Generates per-rollout deliberative analyses
  - Args: `--model`, `--rollouts`, `--output`
  - Generates: blind_deliberative, informed_deliberative, 4 blind candidates for best-of-N
  - Temperature: 0.3 for singles, 0.7 for candidates

**Scoring:**
- `score_deliberative.py` — Score rollouts with deliberative PI conditions
  - Args: `--rollouts`, `--student-logprobs`
  - Conditions: no_pi, answer_only, answer_ref, blind_deliberative, informed_deliberative, informed_delib_ref, blind_candidate_0..3
  - Outputs: per-condition |KL|, Cohen's d, correct/incorrect breakdown, best-of-N analysis
- `run_experiment.py` — Original TOML-config-driven scoring (for non-deliberative conditions)
- `scorers.py` — Core prefill scoring logic (shared)

**PI placement experiment:**
- `score_pi_placement.py` — Test PI in system/user/assistant_prefix positions
- `score_pi_placement_extra.py` — Supplementary: system_with_question placement
- `token_utils.py` — Updated with `pi_placement` parameter (system, system_with_question, user, assistant_prefix)

**Analysis:**
- `analyze_32b.py` — Cross-teacher / self-teacher comparison tables
- `analyze_multi_lens.py` — Token-level lens correlation, oracle upper bound
- `analyze_copy_artifact.py` — Copy artifact detection (TCE, boundary analysis)

**Full pipeline scripts:**
- `run_32b_self_opd.sh` — End-to-end 32B self-OPD (generate → logprobs → analyses → scoring)

### Result Files

```
results/
├── deliberative_pi.json          # 32B analyses of 8B rollouts (100 rollouts)
├── deliberative_scores.json      # 32B scoring 8B rollouts with deliberative PI
├── deliberative_pi_8b.json       # 8B analyses of 8B rollouts (100 rollouts)
├── deliberative_scores_8b.json   # 8B scoring 8B rollouts with deliberative PI
├── 32b-self-opd/
│   ├── rollouts_32b.json         # 360 rollouts from 32B (90 AIME × 4)
│   ├── student_logprobs_32b.json # 32B student self-logprobs
│   ├── deliberative_pi_32b_self.json    # 32B analyses of 32B rollouts
│   └── deliberative_scores_32b_self.json # 32B self-OPD scores
├── aime-32b-clean/
│   ├── teacher_32B.json          # 32B scoring 8B rollouts across 12 PI conditions
│   └── analysis.json             # Pre-computed analysis
├── summaries_all.json            # All summary types for 8B rollouts
└── [other experiment directories from earlier phases]
```

### Upstream Data
- `../multi-lens-stage0/rollouts_aime_selected.json` — 25 AIME problems, 100 rollouts (8B)
- `../multi-lens-stage0/kl_student_logprobs.json` — 8B student self-logprobs (reusable)

## Infrastructure

- **vLLM server**: Qwen3-32B on port 8900, TP=4 on 4×A100 80GB, tmux session `opd` window `vllm-32b`
  - To switch to 8B: kill 32B server, start 8B with TP=1 on port 8900
- **Prefill throughput**: ~1.3k tokens/s for 32B, concurrency=16
- **Key dependencies**: vllm, transformers, datasets (HuggingFace), scipy

## Integration with prime-rl Training

To use deliberative OPD in actual training, the training loop needs:

1. **Per-rollout analysis generation**: After student generates rollouts, teacher generates ~1024-token analysis for each. This is a standard generation call (not prefill).

2. **Modified PI injection — assistant_prefix placement**: The analysis must be injected as an **assistant response prefix**, not in the system prompt. Instead of the current `build_teacher_prompt_ids()` which appends PI to the system message, the new approach prepends the analysis to the completion tokens:
   ```
   teacher_completion = analysis_text + "\n\n--- STUDENT RESPONSE ---\n" + student_rollout
   ```
   This requires modifying `build_teacher_prompt_ids()` in `src/prime_rl/orchestrator/utils.py` and the alignment logic in `compute_teacher_logprobs()` to account for the longer completion (extract logprobs only for the student rollout portion, not the analysis prefix).

3. **Scoring**: Standard prefill scoring, but logprobs are extracted from `completion[len(analysis_prefix):]` to align with the student's actual token sequence.

**Why assistant_prefix?** Placement experiment showed +21% |KL| and +44% Cohen's d over system prompt placement for deliberative PI. The analysis as response preamble lets the teacher's attention flow directly from diagnostic reasoning into token scoring.

The deliberative analysis adds one vLLM generation call per rollout per training step. At ~6s per analysis (32B), this is ~10 minutes for 100 rollouts (with concurrency=16). Whether this overhead is acceptable depends on the training step duration.

## Concrete Implementation Plan

### Step 1: Add analysis generation to the orchestrator

In `src/prime_rl/orchestrator/orchestrator.py`, after student rollouts are collected but before teacher logprobs are computed, add a step that generates a blind deliberative analysis for each rollout. This is a standard chat completion call to the teacher vLLM server:

```python
# Prompt for blind analysis (from generate_deliberative_pi.py)
BLIND_ANALYSIS_PROMPT = """You are an expert math teacher analyzing a student's work.
Carefully read the problem and the student's attempt below. You do NOT know the correct answer.
Your job is to analyze the student's reasoning process in depth:
1. What approach/strategy did the student use?
2. Trace through the key reasoning steps — are they logically valid?
3. Identify any specific steps where errors might have occurred and why.
4. Assess the overall quality: Is the reasoning sound? Are there gaps?
5. What should the student have done differently?
Be specific about which steps are good and which are problematic."""

# For each rollout, call: teacher.chat.completions.create(
#   messages=[{"role": "system", "content": BLIND_ANALYSIS_PROMPT},
#             {"role": "user", "content": f"Problem:\n{problem}\n\nStudent's attempt:\n{rollout}"}],
#   max_tokens=1024, temperature=0.3)
```

### Step 2: Modify teacher prompt construction for assistant_prefix

Current `build_teacher_prompt_ids()` in `utils.py` appends PI to system message. For assistant_prefix:

```python
# Instead of modifying the system message, keep it unchanged.
# Prepend the analysis to the completion tokens:
analysis_prefix_ids = tokenizer.encode(
    analysis_text + "\n\n--- STUDENT RESPONSE ---\n",
    add_special_tokens=False
)
all_tokens = prompt_ids + analysis_prefix_ids + completion_ids
# prompt_len stays the same (system + user + gen_prompt marker)
# But we need to track analysis_prefix_len for logprob extraction
```

### Step 3: Modify logprob alignment in `compute_teacher_logprobs()`

Currently extracts completion logprobs from position `len(teacher_prompt_ids)`. With assistant_prefix, need to skip the analysis prefix tokens too:

```python
# Phase 1 with assistant_prefix:
teacher_prompt_len = len(prompt_ids)  # same as student prompt
analysis_prefix_len = len(analysis_prefix_ids)
# Logprobs for student rollout start at: teacher_prompt_len + analysis_prefix_len
completion_logprobs = raw_logprobs[teacher_prompt_len + analysis_prefix_len:]
```

### Step 4: Config and TOML

Add to orchestrator config:
```toml
[orchestrator.teacher_model]
deliberative = true           # Enable analysis generation before scoring
analysis_max_tokens = 1024    # Token budget for analysis
analysis_temperature = 0.3
# pi_placement = "assistant_prefix"  # could make configurable
```

### Step 5: Training experiment

Run two configs:
1. **Baseline**: `answer_only` PI in system prompt (existing behavior)
2. **Treatment**: `blind_deliberative` PI in assistant_prefix (new behavior)

Both self-teacher (8B→8B). Compare on AIME eval accuracy over training steps.

## Key Open Questions for Training

1. **Does higher |KL| = better training?** Deliberative has 19% more |KL| but lower Cohen's d. The signal is stronger but less discriminative. Training will answer this.

2. **Is blind or informed better for training?** In signal measurement, blind ≈ informed. But in training, knowing the answer might still help (even if it doesn't change |KL| magnitude).

3. **Self-teacher or cross-teacher?** Self-teacher isolates the PI effect cleanly. Cross-teacher has the practical advantage of a better base distribution. Both worth testing.

4. **Analysis budget scaling**: We used ~1024 tokens. Is there a sweet spot?

5. **Staleness**: As the student improves during training, analyses generated from old rollouts become stale. How often to regenerate?
