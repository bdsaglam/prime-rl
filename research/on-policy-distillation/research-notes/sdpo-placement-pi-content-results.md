# SDPO-Style Placement + PI Content Comparison

Date: 2025-03-12

## Experiment Design

Comprehensive signal measurement comparing PI content types under SDPO-style placement (`user_sdpo`: PI appended to user message + "Correctly solve the original question.").

**Dimensions:**

- Student: Qwen3-8B (100 rollouts, 25 AIME problems, 14% correct)
- Teacher: 8B self-teacher, 32B cross-teacher
- Analysis model: 8B, 32B (for deliberative PI generation)
- PI content: 11 conditions
- PI placement: user_sdpo (primary), assistant_prefix (secondary)

**Data reuse:** Existing rollouts, student logprobs, deliberative analyses (8B+32B), blind summaries.

## Placement Effect (user_sdpo vs system)

First validated that SDPO-style placement improves over our previous system placement:

| PI Condition | system |KL| | user_sdpo |KL| | Δ |
|---|---|---|---|
| answer_only | 0.0136 | 0.0150 | **+10.3%** |
| answer_ref | 0.0630 | 0.0700 | **+11.0%** |
| blind_diagnosis | 0.0340 | 0.0346 | +1.6% |
| blind_confidence | 0.0421 | 0.0429 | +1.7% |

Static PI (answer, ref) benefits most from proximity to response tokens. Self-generated summaries are less sensitive to placement.

## Main Results: PI Content (user_sdpo, 8B self-teacher)

### |KL| Table


| PI Content            | 8B→8B, 8B analysis | 8B→8B, 32B analysis | 8B→32B, 32B analysis | 8B→32B, 8B analysis |
| --------------------- | ------------------ | ------------------- | -------------------- | ------------------- |
| no_pi                 | 0.0024             | 0.0023              | 0.0966               | 0.0966              |
| answer_only           | 0.0150             | 0.0150              | 0.0997               | 0.0997              |
| answer_ref            | 0.0700             | 0.0699              | 0.1230               | 0.1230              |
| blind_diagnosis       | 0.0346             | 0.0345              | 0.1050               | 0.1050              |
| blind_confidence      | 0.0428             | 0.0428              | 0.1077               | 0.1077              |
| sibling_rollout       | 0.0835             | 0.0835              | 0.1120               | 0.1120              |
| answer_sibling        | 0.0833             | 0.0833              | 0.1108               | 0.1108              |
| blind_deliberative    | 0.0738             | 0.0743              | 0.1206               | 0.1180              |
| informed_deliberative | 0.0744             | 0.0739              | 0.1224               | 0.1195              |
| informed_delib_ref    | 0.0857             | 0.0844              | 0.1268               | 0.1247              |
| best-of-4 blind       | 0.0807             | 0.0844              | 0.1272               | 0.1224              |


### Cohen's d Table


| PI Content            | 8B→8B, 8B analysis | 8B→8B, 32B analysis | 8B→32B, 32B analysis | 8B→32B, 8B analysis |
| --------------------- | ------------------ | ------------------- | -------------------- | ------------------- |
| no_pi                 | 0.10               | 0.08                | 0.76                 | 0.76                |
| answer_only           | 0.84               | 0.85                | 0.81                 | 0.81                |
| answer_ref            | 0.58               | 0.58                | 0.72                 | 0.72                |
| blind_diagnosis       | 0.39               | 0.39                | 0.71                 | 0.71                |
| blind_confidence      | 0.24               | 0.23                | 0.63                 | 0.63                |
| sibling_rollout       | **1.02**           | **1.02**            | **1.07**             | **1.07**            |
| answer_sibling        | **1.07**           | **1.08**            | **1.09**             | **1.09**            |
| blind_deliberative    | 0.32               | 0.22                | 0.48                 | 0.59                |
| informed_deliberative | 0.23               | 0.31                | 0.55                 | 0.55                |
| informed_delib_ref    | 0.39               | 0.32                | 0.53                 | 0.58                |
| best-of-4 blind       | 0.32               | 0.15                | 0.43                 | 0.60                |


### PI-only Signal (|KL| minus no_pi baseline)


| PI Content         | 8B→8B  | 8B→32B |
| ------------------ | ------ | ------ |
| answer_only        | +0.013 | +0.003 |
| answer_ref         | +0.068 | +0.026 |
| sibling_rollout    | +0.081 | +0.015 |
| blind_deliberative | +0.071 | +0.024 |
| informed_delib_ref | +0.083 | +0.030 |


## Key Findings

### 1. Sibling rollout is the strongest PI for self-teacher (|KL| × Cohen's d)

Sibling rollout (a correct peer rollout from the same batch — SDPO's approach) achieves:

- |KL| = 0.084, +19% over answer_ref (0.070)
- Cohen's d = 1.02, **3x better** than deliberative (0.32) and **1.8x better** than answer_ref (0.58)
- Zero cost (no LLM call, just reuse existing correct rollouts)

### 2. Deliberative beats answer_ref in |KL| but not in discrimination

Under user_sdpo placement:

- blind_deliberative: +5.5% |KL| over answer_ref, but d drops from 0.58 to 0.32
- Previously with system placement: +19% |KL| over answer_ref

The gap narrowed because answer_ref benefited more from the placement change (+11%) than deliberative did.

### 3. Analysis model (8B vs 32B) barely matters

Across all conditions, 8B and 32B analyses produce near-identical results. The analysis model is not a bottleneck.

### 4. Cross-model gap still dominates for 32B teacher

PI-only signal with 32B teacher: max +0.030 (informed_delib_ref). Self-teacher: +0.083. Cross-model gap accounts for ~80% of total signal, confirming earlier findings.

### 5. informed_delib_ref is the |KL| ceiling

Combining answer + ref solution + deliberative analysis gives 0.086 — the highest raw signal. But sibling_rollout achieves 97% of that (0.084) with far better discrimination.

## assistant_prefix Results (8B self-teacher, 8B analysis)

| PI Content | user_sdpo |KL| | asst_prefix |KL| | user_sdpo d | asst_prefix d |
|---|---|---|---|---|
| answer_only | 0.015 | **0.361** | 0.84 | 0.80 |
| answer_ref | 0.070 | **0.943** | 0.58 | 0.12 |
| blind_deliberative | 0.074 | **0.581** | 0.32 | 0.20 |
| sibling_rollout | 0.084 | **0.289** | 1.02 | 1.03 |

**WARNING**: assistant_prefix |KL| values are 5-10x higher than user_sdpo. This is almost certainly a **copy artifact** — text in the assistant turn gets "read" by the model, inflating logprobs without pedagogical value. answer_ref goes to 0.94 with d=0.12 (near-zero discrimination), confirming this. Sibling rollout is the exception — it maintains d=1.03, suggesting the model processes it differently.

## Implications

1. **For training**: user_sdpo placement is the safe choice. assistant_prefix is contaminated for most PI types.
2. **SDPO's sibling rollout is hard to beat** when correct peers exist. It's the natural "here's what you should have done" signal with excellent discrimination.
3. **Deliberative PI has a niche**: When no correct sibling exists (69% of our rollouts), deliberative analysis is the best available PI. This is the case for hard problems where the student always fails.
4. **Hybrid strategy**: Use sibling rollout when available, fall back to deliberative analysis when no peer succeeded. This combines SDPO's free PI with our adaptive PI for the hard tail.

## Sibling-Conditioned Analysis: Is Analysis Additive?

Generated analyses where the analyzer sees the correct sibling + student attempt, then tested whether adding analysis on top of sibling rollout improves signal.

**8B self-teacher, N=31 rollouts with correct sibling:**

| PI Content | |KL| | Cohen's d | vs sibling_only |
|---|---|---|---|
| no_pi | 0.003 | 0.84 | baseline |
| answer_ref | 0.060 | 1.04 | -29% |
| sibling_only | 0.084 | 1.02 | — |
| analysis_only_informed | 0.056 | 0.91 | -33% |
| **sibling + informed analysis (8B)** | **0.089** | 0.91 | **+6.1%** |
| **sibling + informed analysis (32B)** | **0.089** | 0.94 | **+7.1%** |
| **sibling + blind analysis (8B)** | **0.086** | 0.91 | **+3.1%** |

**32B cross-teacher, same 31 rollouts:**

| PI Content | |KL| | Cohen's d |
|---|---|---|
| no_pi | 0.079 | 1.03 |
| answer_ref | 0.102 | 1.12 |
| sibling_only | 0.112 | 1.07 |
| sibling + informed (32B) | **0.112** | 1.01 |

**Findings:**

1. **Analysis IS additive** for self-teacher: +6-7% |KL| on top of sibling rollout
2. Cohen's d drops slightly (1.02 → 0.91-0.94) but stays well above standalone deliberative (0.32)
3. Analysis model (8B vs 32B) barely matters — 0.089 vs 0.089
4. For 32B cross-teacher, analysis adds nothing on top of sibling (model gap dominates)
5. Standalone analysis (0.056) is weaker than sibling (0.084) — it's complementary, not a replacement

### Analyzer with Answer + Sibling (v2)

Tested giving the analyzer the correct answer in addition to sibling + student rollout.

**8B self-teacher, N=31, user_sdpo:**

| PI Content | |KL| | Cohen's d |
|---|---|---|
| sibling_only | 0.084 | 1.02 |
| sibling + blind analysis | 0.085 | 0.89 |
| sibling + informed analysis (no answer) | 0.090 | 0.94 |
| sibling + answer analysis (8B) | 0.089 | **1.00** |
| sibling + answer analysis (32B) | 0.089 | **0.98** |
| sibling + informed (32B) | 0.090 | **1.03** |

**Standalone analysis (no sibling in teacher PI):**

| Analysis type | |KL| | Cohen's d |
|---|---|---|
| analysis_only_blind (saw sibling) | 0.053 | 0.96 |
| analysis_only_informed (saw sibling) | 0.060 | 0.93 |
| analysis_only_answer (saw sibling + answer) | 0.059 | **1.17** |

**Key finding**: Giving the analyzer the answer significantly improves discrimination:

- `analysis_only_answer` d=1.17 — best discrimination of any analysis condition
- `sibling_plus_answer` d=1.00 — recovers sibling's d=1.02 while adding +6% |KL|
- The answer grounds the analysis: analyzer knows definitively which reasoning works

**Best combo**: sibling + informed analysis (32B) at |KL|=0.090, d=1.03.

**Recommended PI strategy:**

- When correct sibling exists: sibling + answer-informed analysis = best (|KL| 0.089, d 1.00)
- When no sibling: deliberative analysis alone = 0.074 (still beats answer_ref = 0.070)
- Self-teacher is the right setup (cross-teacher gap masks PI effects)
- Analysis model (8B vs 32B) barely matters for |KL|; 32B slightly better for d

## Analysis Prompt Style Variants

Tested 4 different analysis prompt styles inspired by the three Self-Distillation papers:

- **verbose**: Multi-paragraph analysis (our baseline — same as blind_deliberative)
- **structured**: Short structured error report — VERDICT/ERROR_TYPE/ERROR_LOCATION/WHAT_WENT_WRONG/SHOULD_HAVE (inspired by SDPO's rule-based env feedback)
- **directive**: Guidance framed for the teacher — "where does reasoning go wrong, what should student have written instead" (inspired by User Interactions' hindsight context)
- **error_point**: Minimal — just identify the single critical error + one-sentence fix

### 8B Self-Teacher Results — Cohen's d by Analysis Model

| Variant | 8B blind | 8B informed | 32B blind | 32B informed |
|---|---|---|---|---|
| verbose | 0.25 | 0.13 | 0.11 | 0.19 |
| directive | 0.61 | 0.45 | 0.68 | 0.39 |
| error_point | 0.35 | 0.25 | 0.68 | 0.44 |
| **structured** | **0.55** | **1.74** | **1.48** | **2.23** |

### 8B Self-Teacher Results — |KL| by Analysis Model

| Variant | 8B blind | 8B informed | 32B blind | 32B informed |
|---|---|---|---|---|
| verbose | 0.0772 | 0.0728 | 0.0778 | 0.0750 |
| directive | 0.0476 | 0.0501 | 0.0533 | 0.0534 |
| error_point | 0.0361 | 0.0354 | 0.0366 | 0.0384 |
| structured | 0.0295 | 0.0287 | 0.0260 | 0.0303 |

**Critical finding**: Analysis model DOES matter for structured format. 32B informed structured achieves d=2.23 — the highest discrimination measured in any experiment. 32B blind structured (d=1.48) also far exceeds 8B blind (d=0.55). The earlier finding that "analysis model barely matters" was specific to verbose prompts. Structured format benefits enormously from a more capable analyzer because it requires precise error classification.

### 32B Cross-Teacher Results (8B analysis model)

| Variant | Blind |KL| | Blind d | Informed |KL| | Informed d |
|---|---|---|---|---|
| no_pi | 0.0966 | 0.76 | 0.0966 | 0.76 |
| verbose | 0.1188 | 0.53 | 0.1169 | 0.53 |
| directive | 0.1098 | 0.75 | 0.1112 | 0.69 |
| error_point | 0.1050 | 0.69 | 0.1049 | 0.69 |
| structured | 0.1045 | **0.79** | 0.1040 | **0.86** |

### Key Findings

1. **|KL| and Cohen's d are inversely correlated across prompt styles.** Verbose shifts logprobs most but discriminates least. Structured shifts least but discriminates best. This is a fundamental insight — long analysis text acts as noise that shifts all rollouts uniformly.
2. **Informed + structured is an outlier: d=1.74.** This is the highest discrimination measured for any analysis-based PI — higher than sibling rollout (d≈1.02). The structured format with answer knowledge produces extremely precise signal. The short, constrained output forces the analyzer to be specific.
3. **Verbose is the worst for discrimination despite highest |KL|.** d=0.25 (blind) / 0.13 (informed). The long text shifts all rollouts similarly — this is the "copy artifact" of analysis text. High |KL| from verbose may actually be harmful for training.
4. **Directive is the best balance.** Good |KL| (0.048-0.050) with strong d (0.45-0.61). The "guidance for the teacher" framing from User Interactions works well.
5. **For training, prefer structured or directive over verbose.** The previous blind_deliberative results (d=0.32) used verbose prompts — switching to structured could dramatically improve discrimination while maintaining signal.

## Revised Recommendations

**PI strategy by data regime:**

| Data Available | Best PI | |KL| | Cohen's d |
|---|---|---|---|
| Correct sibling + answer | sibling + informed analysis (32B) | 0.090 | 1.03 |
| Correct sibling only | sibling rollout | 0.084 | 1.02 |
| Answer + ref solution | answer_ref | 0.070 | 0.58 |
| Answer only | informed structured analysis | 0.029 | **1.74** |
| No ground truth | blind directive analysis | 0.048 | 0.61 |

**Analysis prompt selection:**

- When answer is known: use **structured** format (d=1.74, the precision of knowing right/wrong + constrained output = best discrimination)
- When blind: use **directive** format (d=0.61, best balance of signal strength and discrimination)
- **Never use verbose** — it has highest |KL| but worst discrimination, likely harmful for training

## Open Questions

1. Does the Cohen's d gap matter for training? High |KL| + low d might still train well if signal is directionally correct.
2. Can we generate useful "synthetic siblings" via the teacher for problems where no real sibling exists?
3. Training validation: does the +6-7% additive signal translate to faster/better learning?
4. Can we combine structured analysis with sibling rollout? (structured analysis of sibling vs student, then sibling + structured analysis as PI)
5. Is the d=1.74 for informed+structured robust or a fluke? Should validate with more rollouts.

