# Optimal Analysis Generation

## Idea

Search over analysis space to find the analysis that maximizes learning signal. Current approach generates one analysis per prompt style. What if we could optimize the analysis itself?

## Motivation

- Prompt style dramatically affects signal quality (d ranges from 0.11 to 2.23)
- Even within a style, different analyses of the same rollout produce different signals
- Best-of-N analysis selection already shows +31-33% improvement
- There may be an "ideal analysis" for each rollout that maximizes learning

## Approaches

1. **Best-of-N selection**: Generate N analyses, score each as PI, select highest |KL| × d
2. **Analysis refinement**: Generate analysis → score → critique analysis → regenerate
3. **RL on the analyzer**: Train the analysis model to maximize downstream learning signal
4. **Prompt optimization**: Search over prompt space for the prompt that produces best analyses

## Connection to Multi-PI

Multi-PI (use all analyses) and optimal analysis (select best) are complementary:
- Generate N diverse analyses
- Score each individually
- Select top-K and combine as multi-PI

## Status

Best-of-N already tested in signal measurement. Refinement and RL on analyzer are unexplored.
