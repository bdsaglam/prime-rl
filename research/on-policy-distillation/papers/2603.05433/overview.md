# On-Policy Self-Distillation for Reasoning Compression (OPSDC)

**Paper:** [arXiv:2603.05433](https://arxiv.org/abs/2603.05433)

## Authors

- Hejian Sang (Iowa State University)
- Yuanda Xu (Princeton University)
- Zhengze Zhou (Cornell University)
- Ran He (Columbia University)
- Zhipeng Wang (Rice University)
- Jiachen Sun (University of Michigan)

## Abstract

Reasoning models think out loud, but much of what they say is noise. OPSDC (On-Policy Self-Distillation for Reasoning Compression) teaches models to reason more concisely by distilling their own concise behavior back into themselves. The approach minimizes per-token reverse KL divergence between a student model and a teacher version of itself conditioned on conciseness instructions. Without ground-truth answers or token budgets, OPSDC achieves 57-59% token reduction on MATH-500 with 9-16 point accuracy improvements. On AIME 2024, the 14B model gains 10 points with 41% compression.

## Key Contributions

- **Reconceptualization of Verbosity**: Challenges the assumption that verbose reasoning improves accuracy; demonstrates that excessive deliberation introduces compounding errors.
- **Ground-Truth Free Compression**: Eliminates dependence on external supervision or reward models. Requires only problem statements and simple conciseness instructions.
- **Adaptive Compression**: Automatically adjusts compression levels by problem difficulty — stronger compression for easier problems, preserving deliberation for hard ones.
- **Entropy Preservation**: Maintains stable model entropy throughout training, preserving exploratory capacity essential for complex reasoning.

## Methodology

### Core Design

- **Student model**: Generates reasoning without compression instructions
- **Teacher model**: Same model but conditioned on a conciseness instruction

### Training Objective

Minimizes reverse KL divergence between student and teacher distributions on student-generated rollouts. Teacher parameters refreshed every 50 training steps.

### Key Properties

- **On-policy training** prevents distribution shift
- **Reverse KL** provides self-regularization and training stability
- **Periodic teacher updates** enable progressive compression
- **Implicit reward function** favors tokens preferred by the concise teacher

## Results


| Benchmark         | Token Reduction | Accuracy Change          |
| ----------------- | --------------- | ------------------------ |
| MATH-500 (8B/14B) | 57-59%          | +9 to +16 pp             |
| AIME 2024 (14B)   | 41%             | +10.5 pp (65.8% → 76.3%) |
| AIME 2024 (8B)    | 35.4%           | -2.9 pp                  |
| AIME 2025         | ~35%            | ~-5 pp                   |


- MMLU scores fully preserved for both model sizes
- No degradation of general knowledge or reasoning ability

