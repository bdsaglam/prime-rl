# LiveCodeBench as a Training Benchmark for RL with OPD

**Date:** 2026-03-18
**Status:** Investigation complete, recommending adoption

---

## 1. LiveCodeBench Overview

**LiveCodeBench** (LCB) is a contamination-free benchmark for evaluating LLM coding capabilities. It continuously collects fresh competitive programming problems from LeetCode, AtCoder, and Codeforces, making it resistant to data contamination -- a major issue with older benchmarks like HumanEval (saturated by frontier models).

### Versions and Problem Counts

| Version | Time Period | Total Problems |
|---------|------------|----------------|
| v1 | May 2023 -- Mar 2024 | 400 |
| v2 | May 2023 -- May 2024 | 511 |
| v3 | May 2023 -- Jul 2024 | 612 |
| v4 | May 2023 -- Sep 2024 | 713 |
| v5 | May 2023 -- Jan 2025 | 880 |
| v6 | May 2023 -- Apr 2025 | 1,055 |

Note: These are cumulative. LCBv6 problems released after Feb 2025 form the "v6-only" subset used by SDPO (131 questions, Feb--May 2025). Problems are tagged with difficulty (easy/medium/hard) and release date, enabling filtered evaluation windows.

### Dataset Availability

- HuggingFace: `livecodebench/code_generation_lite` (recommended, pruned test cases)
- Each problem: description, input/output examples, public test cases (in description), private test cases (encoded)
- GitHub: `github.com/LiveCodeBench/LiveCodeBench`

### Evaluation Capabilities

LCB evaluates four capabilities beyond basic code generation:
1. **Code generation** -- solve the problem from a description
2. **Self-repair** -- fix code given error feedback
3. **Code execution** -- predict output of code
4. **Test output prediction** -- predict what tests will produce

### Feedback Richness (Critical for RL/OPD)

LCB problems provide rich, structured feedback per test case:
- **Runtime errors**: exception type, traceback, line numbers (e.g., "RuntimeError: division by zero at line 73")
- **Wrong answers**: input, expected output, actual output
- **Time limit exceeded**: which test case timed out
- **Format errors**: missing code block

This is exactly what makes it excellent for SDPO/OPD: the feedback is machine-readable, informative, and directly useful for dense credit assignment. Math benchmarks like AIME only provide binary correct/incorrect.

---

## 2. Qwen3-8B and 32B Performance on LiveCodeBench

### Qwen3-8B

| Source | Version | Mode | Score |
|--------|---------|------|-------|
| Qwen3 tech report (official) | v5 | Thinking | ~57.5--57.8 |
| Reproduced (post-LCB bug fixes) | v5 | Thinking | ~57.8 |
| Reproduced (pre-bug fixes) | v5 | Thinking | 38.3 (broken -- `###` truncation bug) |
| Third-party estimate (CRITIQUE-CODER paper) | v5 | Non-thinking | ~57.5 |
| SDPO paper (Hubotter et al.) | v6-only (131q) | Non-thinking (base) | ~28% |
| SDPO paper | v6-only (131q) | After GRPO training | 41.2% |
| SDPO paper | v6-only (131q) | After SDPO training | 48.8% |
| Aggregate review (2026) | mixed | Thinking | ~60.2 |

**Important note on LCB evaluation bugs (Aug 2025):** Three bugs in the official LCB eval pipeline were discovered:
1. `###` used as stop token, truncating solutions with markdown headers
2. Backtick extraction without verifying code content
3. Hard-coded chat templates misaligned with model training

These bugs inflated some scores by up to 50%. Post-fix, Qwen3-8B official and reproduced scores converge at ~57.8 on LCBv5.

### Qwen3-32B

| Source | Version | Mode | Score |
|--------|---------|------|-------|
| Qwen3 tech report | v5 | Thinking | ~65--67 (estimated from relative positioning) |

The exact Qwen3-32B LCBv5 score is not clearly reported in accessible sources, but it sits between Qwen3-8B (~57.8) and the flagship Qwen3-235B-A22B (70.7) on LCBv5.

### Qwen3 Flagship

- Qwen3-235B-A22B: 70.7 on LCBv5 (thinking mode)
- Qwen3-Coder: 70.6 on LCB (highest open-source on leaderboard)

### Current Leaderboard Leaders (for context)

- Gemini 3 Pro Preview: 91.7%
- DeepSeek V3.2 Speciale: 89.6%

---

## 3. Existing Environment in prime-rl

### Environments Present

```
environments/
  aime/           -- AIME math (single-turn, MathRubric)
  aime_mt/        -- AIME multi-turn
  aime_sdpo/      -- AIME SDPO variant
  arc_agi/        -- ARC-AGI REPL (multi-turn, sandbox-based)
  arc_agi_reflect/ -- ARC-AGI with reflection
```

**No LiveCodeBench environment exists.** Grepping the entire repo for `livecodebench`, `lcb`, or `livecode` returns zero hits in the main codebase (only in the SDPO paper's repo copy at `research/on-policy-distillation/papers/sdpo/repo/`).

### Verifiers Framework

The prime-rl training stack uses the `verifiers` package (`vf`), which provides:
- `vf.SingleTurnEnv` -- used by AIME (question in, answer out, rubric scores)
- `vf.Environment` -- base class
- `SandboxEnv` / `PythonEnv` -- sandbox-backed environments using `prime_sandboxes` for safe code execution (used by ARC-AGI REPL)
- `vf.MathRubric` -- math answer checking
- `vf.Rubric` -- base rubric class (custom rubrics like `ArcAgiRubric`)

The `PythonEnv` class already provides a persistent Python REPL in a sandboxed container, which could potentially be adapted for code execution. However, LCB problems need a different execution model (compile and run against test cases, not interactive REPL).

### SDPO Repo Reference Implementation

The SDPO repo (`research/on-policy-distillation/papers/sdpo/repo/`) contains a complete LCB implementation:
- **Data loading**: `data/utils/livecodebench.py` -- loads from HuggingFace, splits train/test by date cutoff (Feb 2025)
- **Prompt formatting**: `data/format/code.py` + `data/format/prompts.py` -- wraps problem in a code-specific system prompt
- **Code execution**: `verl/utils/reward_score/feedback/code.py` -- multiprocess sandboxed execution with:
  - Restricted builtins (no file I/O, no subprocess, no network)
  - Memory limits (1GB via `RLIMIT_AS`)
  - Time limits per test case (6 seconds)
  - Support for both `functional` (function call) and `stdin` (IO-based) test types
- **Feedback generation**: `format_test_feedback()` -- LeetCode-style formatted feedback (runtime errors, wrong answers with input/expected/actual)
- **Reward**: Binary (all tests pass = 1.0) or partial (fraction of tests passed)

---

## 4. Implementation Plan

### 4.1 Reference Architecture

**AIME pattern (simple, single-turn):**
- `env.py`: loads HF dataset, formats as `(question, answer, info)`, creates `vf.SingleTurnEnv` with `MathRubric`
- `teacher_context.py`: `prepare_teacher_context()` generates PI for the teacher

**ARC-AGI pattern (complex, multi-turn, sandboxed):**
- `env.py`: loads dataset, creates custom `ArcAgiReplEnv` (extends `SandboxEnv`) with custom `ArcAgiRubric`
- Multi-turn REPL interaction with code execution in sandbox
- `teacher_context.py`: PI generation

### 4.2 LCB Environment Design

For LCB, we likely want a **hybrid approach**: single-turn code generation (student writes complete solution) with rich feedback from test execution.

#### Option A: Single-Turn with Rich Feedback (Recommended for v1)

Like AIME but with code execution for reward:

```
environments/livecodebench/
  pyproject.toml
  src/livecodebench/
    __init__.py
    env.py            -- load_environment(), dataset loading
    rubric.py         -- CodeRubric: execute code, compute reward + feedback
    teacher_context.py -- prepare_teacher_context() for OPD
    sandbox.py         -- code execution (adapted from SDPO's code.py)
    data.py           -- dataset loading/formatting from HuggingFace
```

**Key components:**

1. **Dataset loading** (`data.py`):
   - Source: `livecodebench/code_generation_lite` from HuggingFace
   - Use `version_tag` to select version (recommend v5 or v6)
   - Split: by date cutoff (e.g., train on problems before date X, eval on after)
   - Fields: `question_content`, `starter_code`, `private_test_cases`, `metadata`
   - SDPO's `data/utils/livecodebench.py` is a direct reference

2. **Code execution sandbox** (`sandbox.py`):
   - **Option 1**: In-process with restricted builtins + multiprocessing (SDPO approach). Simpler, faster, but less secure.
   - **Option 2**: Use `prime_sandboxes` (already available in verifiers). More secure, but higher latency.
   - **Recommendation**: Start with Option 1 (SDPO approach) for speed. The restricted builtins + memory limits + time limits are sufficient for training. Can upgrade to Docker-based later.
   - SDPO's `verl/utils/reward_score/feedback/code.py` is a complete reference (~870 lines)

3. **Rubric** (`rubric.py`):
   - Execute extracted code against test cases
   - Compute reward: binary (pass all) or partial credit (fraction passed)
   - Generate structured feedback string for failed tests
   - Return feedback in rollout `info` for teacher context

4. **Teacher context for OPD** (`teacher_context.py`):
   - `prepare_teacher_context(rollouts)` -- generates PI per rollout
   - For **correct** rollouts: minimal PI ("Solution is correct")
   - For **incorrect** rollouts: PI includes:
     - Environment feedback (runtime error, wrong answer details)
     - Correct sibling solution (from same batch, if one exists) -- per SDPO's finding that output + own solution is best combination
     - The correct answer/test output for the failing case
   - This maps directly to SDPO's reprompting template

5. **Environment** (`env.py`):
   - Use `vf.SingleTurnEnv` for v1 (student generates one complete solution)
   - System prompt adapted from SDPO: "You are a coding expert..."
   - Could later extend to multi-turn (student gets feedback, iterates)

#### Option B: Multi-Turn with REPL (Future extension)

Like ARC-AGI, student can iterate:
1. Student writes code
2. Environment runs public tests, returns feedback
3. Student revises
4. Repeat up to N turns
5. Final submission evaluated against private tests

This is more complex but matches how humans actually solve coding problems and enables richer learning signal.

### 4.3 Reward Function Design

Based on SDPO's findings and our OPD needs:

| Mode | Reward | When |
|------|--------|------|
| Binary (sparse) | 1.0 if all tests pass, 0.0 otherwise | Default for training |
| Partial credit | Fraction of tests passed | Alternative, may help with very hard problems |
| With feedback | Reward + structured feedback string | For OPD teacher context |

SDPO uses binary rewards for validation and sparse rewards for training on the test split. For public tests during training, they allow partial credit.

### 4.4 Prompt Design

From SDPO's `prompts.py`:
```
You are a coding expert. You will be given a coding problem, and you need to
write a correct Python program that matches the specification and passes all
tests. The time limit is 1 second. You may start by outlining your thought
process. In the end, please provide the complete code in a code block
enclosed with ``` ```.

{problem}
```

For Qwen3, ensure `enable_thinking = false` in non-thinking mode.

---

## 5. Difficulty Assessment

### Qwen3-8B Base Rate on LCB

| Benchmark | Qwen3-8B Base Rate | Sweet Spot? |
|-----------|-------------------|-------------|
| AIME (aimo-validation) | ~72--80% | Too easy |
| LCBv6-only (131q, SDPO) | ~28% | Yes -- ideal |
| LCBv5 (full, thinking) | ~57.8% | Moderate |
| LCBv5 (full, non-thinking) | ~40--50% (est.) | Yes |

**The LCB difficulty is in our sweet spot (20--50%)**, especially:
- The v6-only subset (newest, hardest problems): ~28% base rate
- LCBv5 in non-thinking mode: ~40--50%
- Hard problems within any version: very low pass rates

### Difficulty Filtering

LCB problems are tagged with difficulty (easy/medium/hard):
- Easy: near-perfect for frontier models, 60--80% for 8B models
- Medium: 30--60% for 8B models
- Hard: 5--20% for 8B models

We can filter to medium+hard for optimal training difficulty. The v6-only subset naturally skews harder (newest problems, less contamination).

### Comparison with AIME

| Aspect | AIME | LiveCodeBench |
|--------|------|---------------|
| Qwen3-8B base rate | 72--80% | 28--50% (version dependent) |
| Feedback richness | Binary (correct/incorrect) | Rich (errors, test results, IO) |
| Problem count | 90 (aimo-validation) | 131 (v6-only) to 1055 (full v6) |
| Domain | Math | Competitive programming |
| Contamination risk | High (well-known problems) | Low (continuously updated) |
| Verification | String match on answer | Code execution against tests |
| OPD signal potential | Limited (answer only) | High (errors, correct solutions, test output) |

---

## 6. Recommendation

### Verdict: Strong YES -- adopt LiveCodeBench

LCB is an excellent next benchmark for our RL+OPD pipeline for the following reasons:

### 6.1 Why LCB

1. **Ideal difficulty**: 28% base rate on v6-only (vs 72--80% on AIME). This is squarely in the RL sweet spot where there's room to improve but enough signal to learn.

2. **Rich feedback for OPD**: This is the killer feature. LCB provides structured, informative feedback per test case (runtime errors, wrong answers with I/O, time limits). This is exactly what SDPO showed drives dense credit assignment. Our OPD pipeline can use this feedback as privileged information for the teacher, similar to how SDPO uses it for the self-teacher.

3. **Proven by SDPO**: Hubotter et al. demonstrated 48.8% (from 28% base) on LCBv6 with Qwen3-8B using self-distillation. This validates that the benchmark responds well to distillation-based RL training.

4. **Large, contamination-free dataset**: 1055 problems (v6), continuously updated. Can split by date for train/eval without contamination concerns.

5. **Existing reference implementation**: The SDPO repo provides a complete, tested implementation of dataset loading, code execution, feedback generation, and reward computation that we can adapt.

### 6.2 Recommended Version

**LCBv5 or v6** with a date-based train/test split:
- Train: problems before Feb 2025 (~880 problems from v5)
- Eval: problems Feb--May 2025 (the v6-only 131-problem subset, same as SDPO)
- This gives a large, diverse training set and a held-out eval set

### 6.3 Risks and Challenges

1. **Code execution security**: Running model-generated code is inherently risky. SDPO's approach (restricted builtins, memory/time limits, multiprocessing) is sufficient for training but not production-grade. For higher security, use Docker/`prime_sandboxes`.

2. **Execution latency**: Code execution adds latency to the reward computation pipeline. SDPO reports +17% overhead with code environments. With multiprocessing this should be manageable.

3. **LCB evaluation bugs**: The `###` truncation bug and template issues (discovered Aug 2025) can cause misleading results. Must use corrected evaluation or our own code extraction (SDPO's `extract_code()` is simple regex-based and avoids this).

4. **Domain shift**: Moving from math to coding changes what the model learns. May want to run mixed-domain training (AIME + LCB) to avoid catastrophic forgetting on math.

5. **Single-turn limitation**: Initial implementation is single-turn (generate one solution). Multi-turn iteration would be more powerful but significantly more complex.

### 6.4 Estimated Implementation Effort

| Component | Effort | Notes |
|-----------|--------|-------|
| Dataset loading | 1 day | Adapt SDPO's `livecodebench.py` |
| Code execution sandbox | 1--2 days | Port SDPO's `code.py` (~870 lines, well-structured) |
| Rubric (reward + feedback) | 1 day | Wrap execution in verifiers Rubric interface |
| Environment (vf.SingleTurnEnv) | 0.5 days | Similar to AIME env.py |
| Teacher context (OPD PI) | 1 day | Adapt AIME's teacher_context.py pattern |
| Config + integration | 0.5 days | TOML config, test run |
| Testing + debugging | 1--2 days | End-to-end training validation |
| **Total** | **~5--7 days** | |

### 6.5 Comparison with ARC-AGI as Alternative

| Aspect | LiveCodeBench | ARC-AGI |
|--------|--------------|---------|
| Difficulty (Qwen3-8B) | 28--50% | ~2--5% (too hard for 8B) |
| Feedback richness | Excellent (test errors) | Good (REPL output) |
| Problem count | 131--1055 | ~400 training, 100 eval |
| Implementation complexity | Moderate | Already done |
| RL training signal | Strong (proven by SDPO) | Weak (too hard, sparse reward) |
| Domain diversity | High (algorithmic coding) | Low (visual pattern) |
| SDPO applicability | Directly validated | Untested |

**LCB is the better next benchmark**: ideal difficulty range, richer feedback, larger dataset, and direct validation by SDPO.

---

## Key References

- LiveCodeBench website: https://livecodebench.github.io/
- HuggingFace dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite
- SDPO paper: https://arxiv.org/abs/2601.20802
- SDPO repo: https://github.com/lasgroup/SDPO (local copy at `research/on-policy-distillation/papers/sdpo/repo/`)
- SDPO LCB data loading: `research/on-policy-distillation/papers/sdpo/repo/data/utils/livecodebench.py`
- SDPO code execution: `research/on-policy-distillation/papers/sdpo/repo/verl/utils/reward_score/feedback/code.py`
- SDPO feedback formatting: `format_test_feedback()` in same file
- LCB evaluation bug report: https://blog.collinear.ai/p/lcb-bug-fixes
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- Qwen3 blog: https://qwenlm.github.io/blog/qwen3/
