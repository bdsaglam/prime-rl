# Aligning Language Models from User Interactions

**Authors:** Thomas Kleine Buening (ETH Zurich), Jonas Hübotter (ETH Zurich), Barna Pasztor (ETH Zurich), Idan Shenfeld (MIT), Giorgia Ramponi (University of Zurich), Andreas Krause (ETH Zurich)

**Year:** 2026

**ArXiv ID:** Not yet available (placeholder "arXiv:XXX" on project page)

**Project page:** https://self-distillation.github.io/user_interactions.html

**Code:** https://github.com/lasgroup/user_interactions

**PDF:** paper.pdf (local copy)

## Abstract

The paper proposes a method for training language models by learning from multi-turn user interactions. The approach uses self-distillation where models revise their behavior in hindsight after observing user follow-ups, creating training signals without explicit rewards or preference labels. Training on 14,000 real-world conversations from WildChat improved performance across alignment and instruction-following benchmarks while enabling continual personalization.

## Method (SDPO)

The core mechanism: sample an assistant turn, wait for the next user turn, then recompute the assistant's logits in hindsight with the user turn in context, and distill this corrected next-token distribution into the policy.

Key properties:
- The learning signal is local and intuitive -- user follow-ups naturally encode implicit feedback
- No explicit rewards or preference labels are required
- Uses self-distillation: the model teaches itself by comparing its original output distribution to the hindsight-informed distribution

## Key Results

- Improvements across alignment, instruction-following, reasoning, and creative writing benchmarks
- No regression in other capabilities
- Enables continual personalization without catastrophic forgetting
- Trained on 14k conversations from the WildChat dataset

## Relevance to Our Work

This paper is from the same self-distillation research group behind SDFT and SDPO. Their core idea of using hindsight context to create a better token-level distribution for distillation is closely related to our adaptive PI approach for on-policy distillation. In both cases, privileged information (their: user follow-up; ours: deliberative analysis of student mistakes) is injected to produce a stronger teacher signal that is then distilled into the student policy.
