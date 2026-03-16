# Learning to Reflect: Meta-Learning Introspection

## Idea

The model learns not just from reflection, but learns to **reflect better**. This isn't a separate "teaching" skill — it's the same agent improving its own introspection over time. A novice produces verbose, unfocused post-mortems (d=0.13). An expert immediately identifies the critical mistake and the fix (d=2.23). The skill of reflection itself is learnable.

In the self-teacher setup, the analyzer IS the student (same model). So improving the analyzer's reflection directly improves the model's capacity for self-directed learning. The loop is:

1. Attempt → 2. Reflect → 3. Learn from reflection → 4. Get better at reflecting → repeat

This is "learning to learn" in the most literal sense: the model develops better introspection, which produces richer learning signal, which accelerates future learning.

## Why This Matters

Our prompt style experiments show dramatic variation (d=0.11 to d=2.23) from the same underlying information, just presented differently. This means:
- The *content* of PI matters less than how it's *framed*
- There's a large space of possible PI formulations
- The optimal PI likely varies per problem, per student error type, per training stage
- Hand-designing prompts can't capture this variation — we need the analyzer to adapt

## Conceptual Framework

```
Student rollout + available labels (answer, ref, sibling, env feedback, ...)
        ↓
   Analyzer (learned)
        ↓
   PI (whatever format maximizes learning)
        ↓
   Teacher scores with PI → per-token gradients → student update
        ↓
   Did student improve? (next batch performance)
        ↓
   Signal back to analyzer: "your PI worked / didn't work"
```

The outer loop trains the analyzer; the inner loop trains the student.

## How Could This Work?

### Option A: RL on the Analyzer
- Reward = student improvement on held-out problems after K steps
- Analyzer generates PI → student trains → measure improvement → reward signal
- Challenge: very long horizon, expensive inner loop

### Option B: Proxy Reward
- Instead of actual training improvement, use signal metrics as proxy
- Reward = |KL| × Cohen's d (or some combination)
- Much cheaper — can evaluate in one forward pass
- Risk: proxy might not correlate with actual learning

### Option C: In-Context Meta-Learning
- Don't fine-tune the analyzer — give it examples of what worked
- "Here are 5 analyses that produced high learning signal, and 5 that didn't. Generate an analysis for this new rollout."
- Leverages the LLM's in-context learning ability
- Cheapest to try, no training loop needed

### Option D: Evolutionary / Best-of-N with Feedback
- Generate N candidate PIs per rollout
- Score each (|KL|, d, or actual training signal)
- Use winners as few-shot examples for future generation
- Over time, the analyzer's prompt evolves toward better PI
- Simple, iterative, doesn't require differentiable path through analyzer

### Option E: Distill from Search
- Run expensive search (many candidates, score all) offline
- Collect (rollout, best_PI) pairs
- Fine-tune analyzer on these pairs
- Resulting analyzer directly generates good PI without search at inference time

## Connection to Literature

- **Klissarov 2026 (Meta-Learning perspective)**: Frames distillation as meta-learning. Teacher's role is to provide learning signal that transfers. Our angle: make the *generation* of that signal learnable.
- **RLTF (Song 2026)**: Token-level feedback from environment. We're proposing learned token-level feedback from an analyzer.
- **Reward modeling literature**: Analyzer is essentially a learned "teaching reward model" — but instead of scalar reward, it produces rich text PI.

## What's Different From Optimal Analysis?

[`optimal-analysis.md`](optimal-analysis.md) searches over analysis *space* (best-of-N, refinement). Learning-to-teach goes further: it changes the analyzer's *policy* to directly generate optimal PI. Search is inference-time; meta-learning is training-time.

## Practical First Step

Option C (in-context meta-learning) is the easiest to test:
1. Take our existing scored analyses (we have |KL| and d for hundreds of (rollout, analysis) pairs)
2. Select high-signal and low-signal examples
3. Add them as few-shot context when generating new analyses
4. Measure if the few-shot-guided analyzer produces better PI

This requires no training, just prompt engineering with empirical examples.

## Open Questions

- What's the right reward signal for the analyzer? |KL|? d? Actual eval improvement?
- How to handle the credit assignment problem — which part of the PI helped?
- Can a smaller analyzer learn to teach as well as a larger one with the right training?
- Does the optimal PI change as the student improves? (curriculum over PI)
- How to prevent reward hacking — analyzer learns to game the proxy metric
