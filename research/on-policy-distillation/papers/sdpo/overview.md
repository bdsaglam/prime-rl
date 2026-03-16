# Reinforcement Learning via Self-Distillation (SDPO) - AlphaXiv Overview

Source: https://alphaxiv.org/abs/2601.20802
arXiv: https://arxiv.org/abs/2601.20802

## Core Concept

The paper introduces **Self-Distillation Policy Optimization (SDPO)**, a method that addresses a critical limitation in reinforcement learning with verifiable rewards (RLVR). Current RLVR methods rely on scalar outcome rewards, creating significant bottlenecks in credit assignment. SDPO leverages rich textual feedback -- such as runtime errors or judge evaluations -- that naturally explain failure causes.

## Key Innovation

Rather than requiring an external teacher or explicit reward model, SDPO treats "the current model conditioned on feedback as a self-teacher" and distills its feedback-informed predictions back into the policy. This enables the model to retrospectively identify its own mistakes through in-context learning.

## Demonstrated Results

Testing across multiple domains -- scientific reasoning, tool use, and competitive programming on LiveCodeBench v6 -- shows SDPO improves both sample efficiency and final accuracy compared to strong RLVR baselines. Notably, it outperforms baselines even in standard scalar-feedback environments by leveraging successful rollouts as implicit guidance.

## Practical Efficiency

At test time, applying SDPO to difficult binary-reward tasks accelerates discovery, achieving comparable results to best-of-k sampling or multi-turn conversations using approximately 3x fewer attempts.
