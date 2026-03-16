# PI Placement: Where to Inject Privileged Information in OPD

## The Question

In OPD, the teacher scores the student's rollout tokens via prefill. PI (answer, analysis, reflection) must be injected somewhere in the token sequence. Where should it go?

The placement choice determines how the teacher's attention flows between the PI context and the rollout tokens being scored. Different placements activate different attention patterns in the transformer, which directly affects the quality of per-token signal.

## Four Placement Options Tested

1. **system**: `[system: prompt + PI] [user: problem] [assistant: rollout]` — current default for static PI
2. **system_with_question**: `[system: prompt + problem + PI] [user: problem] [assistant: rollout]` — PI sees problem context
3. **user** (SDPO-style): `[system: prompt] [user: problem + PI] [assistant: rollout]` — PI after problem
4. **assistant_prefix**: `[system: prompt] [user: problem] [assistant: PI + rollout]` — PI as response preamble

## Experimental Results (32B self-teacher)

| PI Condition | system | sys+question | user | asst_prefix |
|---|---|---|---|---|
| answer_only | 0.016 (d=0.72) | 0.023 (d=0.63) | 0.018 (d=0.73) | 0.029 (d=0.42) |
| answer_ref | 0.065 (d=0.54) | 0.066 (d=0.56) | 0.073 (d=0.55) | 0.068 (d=0.54) |
| blind_deliberative | 0.072 (d=0.34) | 0.069 (d=0.32) | 0.074 (d=0.24) | **0.087 (d=0.49)** |
| informed_deliberative | 0.070 (d=0.43) | 0.068 (d=0.35) | 0.073 (d=0.30) | **0.082 (d=0.46)** |

Token-level correlation: system<>user r=0.93, system<>asst_prefix r=0.19. Assistant_prefix produces fundamentally different per-token signal.

## Decision: Use `assistant_prefix` for analytical PI, `system` for static PI

**Rationale**:

1. For deliberative/analytical PI, `assistant_prefix` wins on BOTH |KL| (+21%) AND Cohen's d (+44%)
2. The analysis tokens sit in "response generation" attention mode — attention flows directly from diagnostic reasoning into scoring the rollout
3. For static PI (answer, ref solution), placement barely matters (all within 0.065-0.073)

The low correlation (r=0.19) between system and assistant_prefix placements is the most striking finding. It means the teacher is not just scoring the same tokens differently in magnitude — it is scoring *different tokens* as important. The assistant_prefix placement causes the teacher to attend to rollout tokens through the lens of the immediately preceding analysis, rather than through distant system-level context.

## Connection to SDPO and SDFT

**SDPO (Hubotter 2026)** uses a "user_sdpo" placement — the correct peer rollout goes into the user message. This is similar to our "user" placement. Our results show user placement gives +10-11% |KL| for static PI vs system, consistent with SDPO's approach. However, for analytical PI, assistant_prefix is far superior.

**SDFT (Shenfeld 2026)** doesn't explicitly discuss placement since it uses supervised fine-tuning on golden responses, not prefill scoring. The golden response IS the assistant content. Our assistant_prefix approach is conceptually similar — the analytical PI becomes part of the assistant response context.

## Implementation

In `src/prime_rl/orchestrator/utils.py`, PI is injected based on placement config. For RA-OPD with reflection-in-sequence, the reflection itself IS part of the sequence (not injected PI), so placement only matters for the teacher's additional PI context.

For the teacher scoring in RA-OPD:
```
Student view: [system] [user: problem] [asst: solution] [user: reflection_prompt + binary] [asst: reflection]
Teacher view: [system] [user: problem] [asst: solution] [user: reflection_prompt + answer+reflection_as_PI] [asst: reflection]
```

The info asymmetry is in the user message (reflection prompt), not in PI placement. The teacher sees richer context in the reflection prompt itself.

## Key Reference

Full results: `research/on-policy-distillation/research-notes/sdpo-placement-pi-content-results.md`
