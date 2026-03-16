# Decision: Structured Reflection Format

## The Question

What format should the student's reflection take? Open-ended ("reflect on your approach") vs structured (fixed fields)?

## The Format

```
VERDICT: [correct/incorrect/partially_correct]
ERROR_TYPE: [logic/implementation/pattern_recognition/none]
ERROR_LOCATION: [which step went wrong, or "N/A"]
WHAT_WENT_WRONG: [one sentence]
LESSON: [one sentence]
```

For AIME (math), the full format also includes CONFIDENCE and CORRECTION fields.

## Why Structured Beats Open-Ended

**Experimental evidence** (32B self-teacher, AIME 2025):

| Format | d (reflection) | d (solution) |
|---|---|---|
| Open-ended | -0.57 to -1.51 | ~0.4-0.6 |
| Structured | 2.89 to 4.67 | ~0.4-0.6 |

Open-ended reflection produces **negative** Cohen's d on reflection tokens.

## Why Negative d on Open-Ended?

The mechanism:
1. **Incorrect rollouts** → student writes substantive freeform text: "I made an error at step 3 where I confused the quadratic formula..."
2. **Correct rollouts** → student writes bland text: "I'm confident my solution is correct."
3. **Teacher with PI** evaluates both → the teacher AGREES more with the substantive incorrect reflection (it aligns with what the PI-informed teacher expects to see)
4. Result: |KL| is LOWER for incorrect rollouts → inverted discrimination

**Structured format fixes this** by constraining both correct and incorrect reflections to the SAME fields. Both must fill in VERDICT, ERROR_TYPE, etc. The teacher's PI-informed evaluation of these fixed fields creates consistent discrimination.

## The Analogy

This mirrors human learning: targeted reflection ("I confused X with Y") beats unfocused rumination ("let me think about what happened...").

Across all analysis styles tested:

| Style | |KL| | Cohen's d | Description |
|---|---|---|---|
| Verbose | 0.077 | 0.13-0.25 | Multi-paragraph analysis |
| Directive | 0.048 | 0.61 | Teacher-style guidance |
| Structured | 0.029 | 1.74 | Short error report |

**|KL| and d are inversely correlated**: shorter, more precise reflection = lower total signal but MUCH better discrimination.

## Domain Adaptation

The structured format may need domain-specific fields:
- **AIME (math)**: ERROR_TYPE includes logic, computation, conceptual
- **ARC-AGI (visual reasoning)**: ERROR_TYPE includes pattern_recognition, implementation
- **Code**: Could include categories like algorithm_choice, edge_case, off_by_one

The VERDICT/ERROR_TYPE/ERROR_LOCATION/WHAT_WENT_WRONG/LESSON skeleton is domain-general. ERROR_TYPE values should be domain-specific.

## Open Question: Learnable Format

The structured format is hand-designed. Can the model learn optimal reflection structure through OPD training? The gradient on reflection tokens provides exactly this pressure, but we haven't tested whether the format evolves meaningfully during training.

## Key References

- Full style comparison: `research/on-policy-distillation/research-notes/sdpo-placement-pi-content-results.md`
- Reflection-in-sequence results: `research/on-policy-distillation/experiments/opd-signal/reflection-in-seq-results.md`
