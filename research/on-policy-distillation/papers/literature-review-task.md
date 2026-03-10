# Task: Comprehensive Literature Review on On-Policy Distillation

## Objective

Conduct a systematic literature review on **on-policy distillation (OPD) for LLM post-training**, covering the period **March 2025 – March 2026**, plus foundational older papers that are heavily cited by recent work. The output should be a structured document identifying papers we don't already cover, summarizing their contributions, and mapping the landscape.

## Context

We already have detailed notes on 7 papers (see `papers/` directory):

| Paper | arXiv ID | Year | Already Covered |
|-------|----------|------|-----------------|
| GKD (Agarwal et al.) | 2306.13649 | 2024 | Yes |
| OPSD (Zhao et al.) | — | 2026 | Yes |
| SDFT (Shenfeld et al.) | 2601.19897 | 2026 | Yes |
| SDPO (Hubotter et al.) | 2601.20802 | 2026 | Yes |
| pi-Distill (Penaloza et al.) | 2602.04942 | 2026 | Yes |
| RLTF (Song et al.) | 2602.02482 | 2026 | Yes |
| RL^2F / SML (Klissarov et al.) | 2602.16066, 2602.16488 | 2026 | Yes |

We also have a comparative overview (`papers/overview.md`) and a concepts doc (`opd-concepts.md`). **Do not re-summarize these 7 papers.** Focus on finding what we're missing.

## Search Scope

### 1. Papers We Know We're Missing (Mentioned but No Notes)

These are referenced in our existing notes but we don't have dedicated coverage. Find and summarize each:

- **MiniLLM** (Gu et al., 2023) — On-policy distillation with gradients through sampling (policy gradient). Concurrent with GKD. How does it compare?
- **ImitKD** — Imitation-based knowledge distillation. Mixed on-policy/off-policy baseline in GKD.
- **Context Distillation** (Snell et al., 2022) — Shows models can internalize privileged context through SFT. Precursor to self-distillation.
- **Qwen3 Technical Report** (2025) — Used OPD to achieve 74.4% on AIME'24 at 10x lower cost than RL.
- **Thinking Machines / Kevin Lu blog post** (2025) — Practical reverse-KL OPD implementation, 50-100x compute savings claim.
- **POPE** (Qu et al., 2026) — Privileged oracle for on-policy exploration.
- **Nudging** (Chen et al., 2025) — Self-generated hints for RL to overcome zero-reward exploration barriers.
- **Feedback Descent** (Lee et al., 2025) — Text-space optimization via pairwise comparison. Baseline in RLTF.
- **Variational reasoning** (Zhou et al., 2025) — Most similar to pi-Distill. Separate parameters, iterative EM, assumes oracle answers.
- **Learning beyond teacher** (Yang et al., 2026) — On-policy distillation with reward extrapolation.

### 2. Broader Search Queries

Search arXiv, Semantic Scholar, and Google Scholar for papers published **March 2025 – March 2026** matching these queries:

- "on-policy distillation" language model
- "self-distillation" LLM reasoning
- "knowledge distillation" on-policy autoregressive
- "privileged information" language model training
- "self-teaching" LLM reinforcement learning
- "dense credit assignment" LLM RL
- on-policy KL divergence student teacher LLM
- "learning from feedback" LLM meta-learning
- DAgger language model imitation learning
- process reward model distillation

### 3. Foundational Papers (Older but Heavily Cited)

Confirm we understand the lineage. For each, provide a 2-3 sentence summary of its relevance to OPD:

- **DAgger** (Ross, Gordon & Bagnell, AISTATS 2011)
- **SeqKD** (Kim & Rush, 2016)
- **Born-Again Networks** (Furlanello et al., 2018)
- **RL^2** (Duan et al., 2016; Wang et al., 2016)
- **LUPI** (Vapnik & Vashist, 2009) — Learning using privileged information
- **Hindsight Experience Replay** (Andrychowicz et al., 2017)

## Output Format

Produce a single markdown file at `papers/literature-review.md` with this structure:

```
# OPD Literature Review (March 2025 – March 2026)

## 1. Papers We Were Missing
For each paper found from the "known missing" list:
- Full citation (authors, title, venue/arXiv, date)
- 3-5 sentence summary of the core contribution
- How it relates to our 7 existing papers (overlaps, extends, contradicts)
- Whether it warrants a full deep-dive note

## 2. New Papers Discovered
For each new paper found through search:
- Full citation
- 3-5 sentence summary
- Relevance to OPD (direct, tangential, foundational)
- Priority: HIGH (should read in full) / MEDIUM (skim abstract+results) / LOW (aware of it)

## 3. Foundational Lineage
2-3 sentences per foundational paper explaining its connection to modern OPD.

## 4. Landscape Summary
- How many total papers now exist in this space?
- What clusters/subfields are emerging?
- What are the biggest open problems across the literature?
- Any papers that challenge or contradict the "on-policy is always better" narrative?

## 5. Recommended Reading List
Top 5 papers we should read next (not already in our notes), ranked by relevance.
```

## Quality Criteria

- **Completeness:** We want to be confident we haven't missed a major paper in this space.
- **Recency bias:** Strongly prefer papers from 2025-2026. Older papers only if they are foundational or heavily cited.
- **No redundancy:** Don't re-summarize our existing 7 papers. Reference them by shorthand (GKD, OPSD, SDFT, SDPO, pi-Distill, RLTF, RL^2F).
- **Practical focus:** For each paper, note whether it has released code and on what framework (verl, TRL, prime-rl, custom).
- **Honest assessment:** If a paper is incremental or has weak experiments, say so. We want signal, not noise.
