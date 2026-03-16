# prime-rl: On-Policy Distillation Research

Fork of [PrimeIntellect/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) for research on adaptive privileged information (PI) in on-policy distillation (OPD).

## Research Focus

**Adaptive PI for OPD**: Standard OPD gives the teacher a static advantage (answer + reference solution). We explore *deliberative teaching* — the teacher generates an analysis of the student's specific mistakes before scoring, creating stronger, more targeted per-token feedback.

**Key signal measurement results**: Analysis prompt style matters dramatically — structured format (short VERDICT/ERROR_TYPE/ERROR_LOCATION) achieves d=1.74 (informed) vs d=0.25 for verbose. |KL| and discrimination are inversely correlated: shorter analysis = more precise signal. Sibling rollout (SDPO-style) remains strongest single PI at d=1.02. Details: `research/on-policy-distillation/research-notes/sdpo-placement-pi-content-results.md`

**Analysis style**: `AnalyzerConfig.analysis_style` controls prompt format. Default is `"structured"` (best discrimination). Options: structured, directive, verbose, error_point. AIME env (`environments/aime/src/aime/teacher_context.py`) selects informed/blind variant based on whether reference solution is available.

**Training validation (2026-03-12)**: Deliberative PI confirmed working in self-teacher (8B→8B) setting. Heldout +4.3%, train +6.9%, grad norms 3-5x baseline — while answer-only self-teacher shows zero learning. Cross-model gap masks PI in cross-teacher setups. Details: `research/on-policy-distillation/experiments/training-verification/verification-spike.md`.

**Future direction**: Test-time scaling via self-analysis loop — model iteratively analyzes its own rollouts and updates. See `research/on-policy-distillation/research-notes/test-time-scaling-idea.md`.

## Architecture: `prepare_teacher_context` Contract

Each environment owns its teacher context preparation. The orchestrator discovers `prepare_teacher_context` via importlib:

```python
# Contract (implemented per env, e.g. environments/aime/src/aime/teacher_context.py)
async def prepare_teacher_context(analyzer_config: AnalyzerConfig, rollouts: list[dict]) -> list[dict]
```

- Env decides what PI to generate based on available data (ref solution, answer, rollout quality)
- Correct rollouts can be skipped (no LLM call needed)
- Uses litellm for external LLM calls (Gemini, etc.)
- No shared analyzer module in prime-rl — envs own this entirely

## Key Commands

```bash
# Training
python -m prime_rl.entrypoints.rl @ configs/aime/opd-self-teacher-8b.toml

# Evaluation
prime eval run aime -a '{"dataset_name":"aime2025"}' -n 8 -r 4 -m MODEL -b URL -t MAX_TOKENS -T 0.6 --skip-upload -d

# Logs
tail -F outputs/logs/orchestrator.stdout
```

## How OPD Works

1. Student generates rollouts on training problems
2. (Optional) `prepare_teacher_context` generates adaptive PI per rollout
3. Teacher scores the same token sequence via prefill (no generation), with PI injected into prompt
4. Loss = `adv_tau * GRPO_advantage + teacher_tau * (teacher_logprobs - student_logprobs)`

Key code:
- Loss: `src/prime_rl/trainer/rl/loss.py`
- Teacher logprobs + PI injection: `src/prime_rl/orchestrator/utils.py`
- Orchestrator (env dispatch): `src/prime_rl/orchestrator/orchestrator.py`
- Analyzer config: `src/prime_rl/configs/orchestrator.py`

## Repository Structure

```
prime-rl/
├── src/prime_rl/
│   ├── configs/              # Pydantic config models
│   ├── entrypoints/rl.py     # Main entry — launches all processes
│   ├── trainer/rl/           # Training loop + loss function
│   ├── orchestrator/         # Orchestrator loop, teacher logprobs, env dispatch
│   ├── inference/            # vLLM inference server
│   └── transport/types.py    # TrainingSample, TrainingBatch structs
├── environments/
│   ├── aime/                 # AIME math competition env + teacher_context.py
│   └── arc_agi/              # ARC-AGI REPL environment
├── configs/
│   ├── aime/                 # AIME training configs (self-teacher, deliberative, etc.)
│   └── arc_agi/              # ARC-AGI training configs
├── research/                 # Organized research docs, papers, notes
└── tmp/on-policy-distillation/  # Active experiment logs and analysis
```

## Research Documentation

See `research/on-policy-distillation/README.md` for the full index. Key docs:

- **Signal measurement**: `research/on-policy-distillation/experiments/opd-signal/FINDINGS.md`
- **Gap analysis**: `research/on-policy-distillation/experiments/training-verification/gap-analysis.md`
- **Pivot strategy**: `research/on-policy-distillation/experiments/training-verification/pivot-strategy.md`
- **Training management**: `research/on-policy-distillation/experiments/prime-rl-training-management-guide.md`
- **Literature reviews**: `research/on-policy-distillation/experiments/literature-review-{1,2}.md`

## Training Lessons

- **Qwen3**: Disable thinking mode via `chat_template_kwargs = {enable_thinking = false}`
- **OOM with 32K**: Use `fused_lm_head_chunk_size = 8192` in `[trainer.model]`
- **16K too short for AIME**: 80% truncation. 32K gives 20% truncation
- **Check first 10 steps**: truncation rate + mismatch_kl are early health signals
- **Kill zombie vLLM**: Always kill old vLLM processes before re-launching
