# Self-Distillation Research Group — Paper Review

Date: 2025-03-11
Source: https://self-distillation.github.io/
Papers saved: `research/papers/{sdft,sdpo,user-interactions}/` (each has paper.pdf, overview.md, repo/)

## Three Papers, One Core Pattern

All three papers from this group share the same self-distillation architecture:
1. **Student** = `pi_theta(y | x)` — model conditioned on input only
2. **Teacher** = `pi_theta(y | x, PI)` — same model conditioned on input + privileged info
3. **Loss** = minimize KL between student and teacher token distributions
4. **PI is prompt-level** — no architectural changes, just different text in the prompt

The only thing that varies is what the PI is.

---

## Paper Summaries

### SDFT: Self-Distillation Fine-Tuning (arxiv 2601.19897)

**PI**: Expert demonstration (full correct response to the task). Teacher sees query + demo, student sees query only.

**PI injection**: User prompt prefix — "This is an example for a response to the question: {demo}. Now answer with a response of your own."

**Loss**: Reverse KL `KL(student || teacher)`. Full-vocabulary analytic per-token estimator (sums over entire vocab V at each position). Empirically best despite theoretical bias at sequence level.

**Teacher**: Same model with EMA weights. EMA > frozen base (can't track improvement) > current student (instabilities).

**Models**: Qwen2.5-7B-Instruct (primary), scaling study with 3B/7B/14B. OLMo-3-7B-Think for reasoning. GPT-4o for demo generation, GPT-5/5-mini for data gen and judging.

**Datasets**: Custom skill learning tasks (Tool Use, Science Q&A, Medical), Knowledge Acquisition (Wikipedia).

### SDPO: Self-Distillation Policy Optimization (arxiv 2601.20802)

**PI**: Environment feedback (runtime errors, test failures, format issues) + successful sibling rollout from same batch. No external LLM or dataset references — fully self-supervised.

**PI injection**: User message, appended after original problem:
```
{original problem}

Correct solution:
{successful sibling rollout}

The following is feedback from your unsuccessful earlier attempt:
{env feedback}

Correctly solve the original question.
```

**Loss**: Forward KL `KL(student || teacher)` — in practice Jensen-Shannon for stability. Token-level with top-K=100 approximation over vocabulary. Per-token advantage = `log pi(y|x,f) - log pi(y|x)`.

**Teacher**: Same model, stop-gradient. Regularized via EMA or trust-region. Student can surpass initial teacher (true bootstrapping).

**Models**: Qwen3-8B (primary), Olmo3-7B-Instruct. Scaling: Qwen3 0.6B/1.7B/4B/8B + Qwen2.5-Instruct 1.5B/3B/7B.

**Datasets**: SciKnowEval L3 (science Q&A), ToolAlpaca (tool use), LiveCodeBench v6 (competitive programming with rich env feedback).

**Key detail**: "Correct solutions" are NEVER from the dataset. Always student-generated rollouts from the same batch that happened to succeed. If no rollout succeeded for a problem, the solution section is skipped.

### User Interactions: Aligning LMs from User Interactions (no arxiv ID yet)

**PI**: User's next follow-up message in a multi-turn conversation. Follow-ups carry implicit feedback (corrections, complaints, revision requests).

**PI injection**: User prompt suffix — "The following is a future user message. Use this to guide your answer to the user prompt: {follow-up}"

**Loss**: Reverse KL `KL(student || teacher)` at token level. Equivalent to policy gradient with per-token log-ratio advantages.

**Teacher**: Same model, stop-gradient (no separate weights, just detached). No EMA.

**Models**: Qwen3-4B/8B, Olmo3-7B-Instruct (SFT + DPO). Claude Haiku 4.5 as user simulator/judge. GPT-4 Turbo and GPT-4.1 as benchmark judges.

**Datasets**: WildChat/WildFeedback (real user conversations, off-policy from GPT-3.5/GPT-4), simulated personalization experiments.

---

## Comparison Table

| | SDFT | SDPO | User Interactions |
|---|---|---|---|
| PI content | Expert demo | Env feedback + peer rollout | User follow-up |
| PI location | User prefix | User message (end) | User suffix |
| KL direction | Reverse | Forward (JSD in practice) | Reverse |
| Token granularity | Full vocab analytic | Top-K=100 approx | Per-token log-ratio |
| Teacher weights | EMA | EMA or trust-region | Stop-gradient (same theta) |
| External LLM needed | Yes (demo generation) | No | No |
| Primary model size | 7B | 8B | 4B-8B |

---

## Relevance to Our OPD Work

**SDPO is the most relevant paper.** Reasons:

1. PI type matches — env feedback / analysis of what went wrong, same as our deliberative PI
2. Same training pattern — on-policy rollouts, score with PI-conditioned teacher via prefill, token-level logprob differences
3. The advantage formula `log pi(y|x,f) - log pi(y|x)` is structurally identical to our `teacher_logprobs - student_logprobs`
4. Validates self-teacher (same model, no larger teacher) — student surpasses initial teacher
5. Uses JSD instead of raw forward KL for stability (potential fix if our signal doesn't translate to gains)

**Key gap**: SDPO gets rich structured feedback for free from code environments. AIME only gives binary correctness. Our deliberative PI synthesizes the rich feedback that SDPO gets for free — stronger contribution story if it works.

---

## Decision: PI Placement Change

**Problem**: Our PI was injected into the system message (`--- PRIVILEGED INFORMATION ---`), placing it at the very start of context, potentially thousands of tokens from the response. Our own signal measurement had already shown assistant_prefix >> system placement (+21% |KL|, +44% Cohen's d).

**SDPO's approach**: PI appended to the user message, immediately before the assistant generation marker.

**Change made** (`src/prime_rl/orchestrator/utils.py:build_teacher_prompt_ids`):
- Before: PI appended to system message
- After: PI appended to last user message + "Correctly solve the original question."

Teacher now sees:
```
System: <original system prompt, unchanged>
User:   <problem text>

        The correct answer is: 42

        Reference solution:
        <solution text>

        Correctly solve the original question.
Assistant: <student's response tokens — scored via prefill>
```

This keeps PI close to the response tokens, maximizing its influence on per-token scoring.
