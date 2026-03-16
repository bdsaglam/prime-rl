# Experiment: Reflection-in-Sequence OPD

## Core Idea

Append a reflection turn to the student's rollout sequence. The student solves, then reflects on its attempt. The teacher (same model + richer PI) scores the ENTIRE sequence including reflection tokens. OPD gradients flow through both solution and reflection tokens, teaching the model to both solve and reflect better.

```
Turn 1 (user):    Solve this problem: [problem]
Turn 2 (student): [solution attempt]                    ← standard OPD tokens
Turn 3 (user):    [reflection prompt + student PI]      ← triggers reflection
Turn 4 (student): [reflection/analysis]                 ← NEW: OPD tokens on reflection
--- teacher scores turns 2+4 with teacher PI (richer than student PI) ---
```

## Why This Works

1. **Reflection tokens get gradient signal**: Teacher with full PI produces logprobs on reflection tokens. If student's reflection misidentifies the error, teacher's logprobs diverge → OPD loss pushes student toward better reflection.

2. **Info asymmetry preserved**: Student gets light PI (binary correct/incorrect, maybe the answer). Teacher gets full PI (answer + structured analysis or reference solution). The gap between student and teacher logprobs on reflection tokens = learning signal for reflection quality.

3. **No new machinery**: Standard OPD loss on a longer sequence. No auxiliary losses, no outer loops. Just a prompt change.

4. **Self-teacher compatible**: When teacher = student (same weights), the model learns to reflect better through the shared-weight mechanism. The PI makes the model a better reflector in teacher mode; OPD transfers that to student mode.

## Experiment Design

### Phase A: Signal Measurement (no training)

Measure whether reflection tokens carry meaningful OPD signal. Use existing signal measurement infrastructure.

**Setup**: For each (problem, incorrect rollout) pair:
1. Build multi-turn sequence: problem → solution → reflection_prompt → reflection
2. Score with teacher (same model + PI) via prefill
3. Measure |KL| and Cohen's d on:
   - Solution tokens only (baseline comparison)
   - Reflection tokens only (new signal)
   - Full sequence (combined)

**Independent variables**:

| Variable | Levels | Rationale |
|---|---|---|
| Student PI in reflection prompt | none, binary ("incorrect"), answer, answer+hint | How much to reveal before reflection |
| Teacher PI | answer_only, answer_ref, structured_analysis | Standard PI comparison |
| Reflection prompt style | open ("reflect on your approach"), structured ("classify your error"), guided ("what went wrong at step X?") | Mirrors analysis prompt style findings |
| Whether student knows it's wrong | yes (told incorrect) vs no (just "reflect") | Does knowing the verdict help? |

**Key metrics**:
- |KL| on reflection tokens: raw signal magnitude
- Cohen's d on reflection tokens: discrimination quality
- Correlation between reflection quality and solution token signal
- Comparison: does adding reflection tokens increase total signal vs solution-only?

**Predictions**:
- Reflection tokens should carry signal when there's info asymmetry (teacher knows more than student)
- Structured reflection prompt should outperform open-ended (consistent with analysis prompt findings)
- Student PI = binary should be the sweet spot (enough to trigger meaningful reflection, not so much it closes the gap with teacher)

### Phase B: Training Experiments

If Phase A shows signal, run training to validate.

**Experiment B1: Reflection-in-sequence vs standard OPD**
- Control: standard self-OPD (solution only, answer PI)
- Treatment: reflection-in-sequence (solution + reflection, answer PI)
- Same model, same data, same compute
- Measure: eval accuracy over training steps

**Experiment B2: Student PI ablation**
- All use reflection-in-sequence
- Vary student PI: none vs binary vs answer
- Teacher PI fixed at answer_ref or structured_analysis
- Measure: which student PI level produces best learning

**Experiment B3: Does reflection quality improve over training?**
- Track reflection quality metrics across training:
  - Does the model's reflection become more accurate?
  - Does error classification improve?
  - Does the model learn to identify specific mistake locations?
- This is the "learning to reflect" signal — does OPD actually teach introspection?

## Implementation Notes

### Multi-turn sequence construction

The sequence is a standard multi-turn chat:

```python
messages = [
    {"role": "user", "content": problem_text},
    {"role": "assistant", "content": student_solution},      # from rollout
    {"role": "user", "content": reflection_prompt},           # new turn
    {"role": "assistant", "content": student_reflection},     # new generation
]
```

For the teacher, PI is injected as usual (system prompt or user_sdpo placement).

### Reflection prompt templates

**Student PI = none (blind reflection)**:
```
Reflect on your solution attempt above. What is your confidence level?
If you made any errors, identify where your reasoning went wrong.
```

**Student PI = binary (told incorrect)**:
```
Your solution is incorrect. The correct answer is different from what you provided.
Reflect on your approach. Where did your reasoning go wrong?
What would you do differently?
```

**Student PI = answer (told correct answer)**:
```
Your solution is incorrect. The correct answer is {answer}.
Analyze where your reasoning diverged from the correct path.
What was your key error?
```

**Student PI = answer + hint**:
```
Your solution is incorrect. The correct answer is {answer}.
Hint: {brief_hint_from_ref_solution}
Identify your error and explain what the correct approach should be.
```

### Structured reflection format (recommended based on prior findings)

```
Respond in this exact format:
VERDICT: [correct/incorrect/unsure]
CONFIDENCE: [high/medium/low]
ERROR_TYPE: [computational/conceptual/approach/none]
ERROR_LOCATION: [which step or reasoning segment]
WHAT_WENT_WRONG: [one sentence]
CORRECTION: [one sentence describing the fix]
```

### What to measure on reflection tokens

For signal measurement, score reflection tokens separately:
- Compute logprobs for turn 4 (reflection) tokens only
- Teacher has full PI, student has limited PI
- |KL| = mean |teacher_logprob - student_logprob| on reflection tokens
- Cohen's d = (mean_incorrect - mean_correct) / pooled_std (across problems)

### Handling correct rollouts

For correct rollouts, the reflection prompt changes:
- Student PI: "Your solution is correct." or nothing
- Student should still reflect: "Verify your solution. Is there a simpler approach?"
- Teacher PI: same as incorrect case
- Signal on correct rollouts may be weaker (less to correct) — measure this

## Connection to Existing Work

- **Analysis prompt style findings**: Structured > verbose for analysis. Expect same for reflection.
- **Signal measurement infra**: Reuse `score_analysis_variants.py` with multi-turn sequence builder.
- **Klissarov RL2F**: Their autodidact (self-critique) outperforms privileged teacher. Our reflection-in-sequence is a concrete implementation of this principle.
- **Test-time scaling**: Trained reflection enables inference-time self-improvement loops without any labels.

## Key Questions

1. **Is there signal on reflection tokens?** If teacher with full PI scores reflection tokens differently than student with limited PI, there's learning signal. This is Phase A.
2. **Does the signal translate to learning?** If training with reflection-in-sequence improves eval accuracy more than standard OPD, the idea works. This is Phase B.
3. **Does reflection quality improve over training?** If the model's reflections become more accurate over time, we've achieved "learning to reflect." This is the holy grail.
4. **What's the optimal info asymmetry?** Too little student PI → reflection is random. Too much → no gap with teacher. Binary correct/incorrect is the hypothesis.
5. **Token budget tradeoff**: Reflection tokens take context window space from solution tokens. Is the tradeoff worth it? At 32K this may not matter much.

## Phase A Results (COMPLETED)

Signal measurement confirmed: reflection tokens carry dramatically stronger signal than solution tokens.

**Best Cohen's d on reflection tokens:**

| Teacher | Reflector | Best Condition | d (reflection) | d (solution) |
|---|---|---|---|---|
| 8B | 8B | answer_ref / none__structured | 0.85 | 0.41 |
| 8B | 32B | answer_ref / answer__structured | 1.42 | 0.44 |
| 32B | 8B | answer_ref / answer__structured | 2.56 | 0.44 |
| **32B** | **32B** | **answer_only / answer_hint__structured** | **4.67** | 0.62 |

Key findings:
1. Structured format essential (open-ended reflection has negative d)
2. More student PI → more signal when using structured format
3. 32B reflector >> 8B reflector (better reflections = more discriminative)
4. Solution tokens unaffected — reflection is purely additive signal

Full results: [`reflection-in-seq-results.md`](reflection-in-seq-results.md)

**Phase B (training) is the next step.** The signal is strong and consistent. Training should validate whether this translates to actual learning improvement.

## Priority

Phase A complete. Phase B (training experiments) ready to begin once GPU resources are available.
