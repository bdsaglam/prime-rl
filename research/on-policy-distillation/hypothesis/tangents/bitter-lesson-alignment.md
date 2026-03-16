# Bitter Lesson Alignment: What Scales?

## The Bitter Lesson Applied to OPD

Sutton's bitter lesson: general methods that leverage computation (search, learning) ultimately outperform methods that leverage human knowledge (hand-crafted features, domain expertise). Applied to our setting:

**Hand-crafted (doesn't scale):**
- Specific analysis prompts per error type
- Human-designed PI templates
- Domain-specific reflection rubrics
- Fixed information asymmetry levels

**Scales with compute:**
- Generating MORE reflections and selecting the best (search)
- Training the model to reflect better over time (learning)
- Larger models producing better reflections (scale)
- More rollouts generating more diverse experiences (data)

## What We've Already Found That's Bitter-Lesson Aligned

1. **Structured format scales with model size.** 32B reflector d=4.67 vs 8B reflector d=0.85. No domain engineering — just more compute for reflection.

2. **More student PI → more signal (automatically).** answer_hint > answer > binary > none. The model self-selects what to reflect on given more information. No hand-crafting needed.

3. **Reflection-in-sequence is general.** The mechanism (solve → reflect → learn from reflection) works for any task with a verifier. No domain-specific analysis needed.

## The Scalable Method: Compute-Driven Reflection

The bitter-lesson-aligned version of our approach:

### Level 1: Scale reflections per problem
- Generate N reflections per rollout (different temperatures, prompts)
- Score each via teacher prefill (cheap)
- Select top-K by |KL| or Cohen's d
- Train on the best reflections

This is "search over reflection space" — exactly the bitter lesson. No human knowledge about what makes a good reflection. Just compute to generate many and select the best.

### Level 2: Scale the reflection model
- Bigger model → better reflections → stronger signal
- Our data: 8B d=0.85, 32B d=4.67 on same problem set
- Prediction: 70B or larger would push even higher
- At some point, the reflection model could be an external stronger model during training

### Level 3: Train the reflection skill
- Self-OPD with reflection-in-sequence
- Reflection tokens get gradient signal (d=4.67)
- Over training steps, model learns to produce better reflections
- Better reflections → stronger signal → faster learning → even better reflections
- This is a positive feedback loop that scales with training compute

### Level 4: Scale the experience
- More rollouts per problem → more diverse errors → richer reflections
- Harder problems → more substantive reflections → stronger signal
- Cross-problem reflection: "This error is similar to what I did on problem X"
- Scales with data and problem diversity

## The Minimal Scalable Method

Strip away all domain knowledge. What's the simplest method that scales?

```
repeat:
    1. Student generates rollouts on problems
    2. For each incorrect rollout:
       a. Tell student "you're wrong, answer is X"
       b. Student reflects in structured format
    3. Teacher (same model + ref solution) scores full sequence
    4. OPD loss on solution + reflection tokens
    5. Update weights
```

That's it. No analyzer model, no external LLM calls, no best-of-N selection, no curriculum over PI. Just one extra turn in the conversation with a structured reflection prompt. The method scales because:

- Bigger model → better reflections → more signal (proven)
- More training → better reflection skill → more signal (theoretical, needs validation)
- More problems → more experiences → more reflection opportunities (trivially true)
- Longer rollouts → more complex errors → richer reflections (probably true)

## What Doesn't Scale (Tempting but Wrong)

1. **Hand-crafted error taxonomies.** Our ERROR_TYPE field (computational/conceptual/approach/notation) is human-designed. A truly scalable method would let the model discover its own error categories.

2. **Fixed structured format.** The VERDICT/ERROR_TYPE/... template works well now but may be suboptimal for different domains. A scalable approach would learn the optimal reflection format.

3. **Fixed info asymmetry.** Telling the student "you're wrong, answer is X" is a hand-crafted choice. A scalable approach might learn what to reveal.

4. **Single reflection per rollout.** Best-of-N is better (proven +31-33%) but we're not doing it because it's more compute. Bitter lesson says: spend the compute.

## Concrete Next Experiment: Minimal Scalable Reflection-OPD

Test the simplest version that scales:

1. Self-OPD with 8B model
2. Student generates rollout
3. If incorrect: append "Your answer is incorrect. The correct answer is {answer}. Reflect:\nVERDICT:\nERROR_TYPE:\nERROR_LOCATION:\nWHAT_WENT_WRONG:\nCORRECTION:"
4. Student generates structured reflection (max 200 tokens)
5. Teacher (same 8B model + answer_ref PI) scores full sequence
6. OPD loss on solution + reflection tokens
7. Compare against standard OPD (solution only, answer_ref PI)

**Why this is the right first experiment:**
- Minimal change from standard OPD (one extra turn)
- No external dependencies (no separate analyzer model)
- Tests the core mechanism: does training with reflection improve over training without?
- Success would validate the entire research direction
- Failure would tell us the signal-to-training gap persists, pointing to next diagnostics

## Connection to Existing Bitter-Lesson Successes

- **AlphaGo**: Search (MCTS) + learning (self-play) beat domain expertise
- **GPT**: Scale (more parameters, more data) beat hand-crafted NLP pipelines
- **Self-play**: Generating your own training data from experience, scaling with compute
- **Ours**: Generating your own learning signal from reflection on experience, scaling with compute

The key insight: **reflection is a form of search over the error space.** When the student reflects "my error was at step 3, I should have used technique Y," it's doing a search over possible explanations for its failure. Structured format constrains this search to be efficient. Training on the results of this search (via OPD) is learning from search — the core of the bitter lesson.
