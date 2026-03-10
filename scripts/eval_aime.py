"""Evaluate a model on AIME 2025 via a vLLM server.

Usage:
    python scripts/eval_aime.py --base-url http://localhost:8900/v1 --model willcb/Qwen3-8B --max-tokens 15000 --num-rollouts 4
    python scripts/eval_aime.py --base-url http://localhost:8932/v1 --model willcb/Qwen3-32B --max-tokens 31000 --num-rollouts 4
"""

import argparse
import asyncio
import re
from collections import defaultdict

from datasets import concatenate_datasets, load_dataset
from openai import AsyncOpenAI


def extract_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer, normalized to integer."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        ans = matches[-1].strip()
        numeric = "".join(c for c in ans if c.isdigit() or c == ".")
        if numeric:
            try:
                return str(int(float(numeric)))
            except (ValueError, TypeError):
                pass
        return ans
    return None


def normalize_answer(raw: str) -> str:
    """Normalize a ground-truth answer to comparable form."""
    numeric = "".join(c for c in raw if c.isdigit() or c == ".")
    try:
        return str(int(float(numeric)))
    except (ValueError, TypeError):
        return raw


async def eval_single(
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    problem: str,
    answer: str,
    idx: int,
    rollout_idx: int,
    max_tokens: int,
    temperature: float,
) -> tuple[int, int, bool, str | None, str]:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a math competition solver. Solve the problem step by step and give your final answer in \\boxed{}.",
                    },
                    {"role": "user", "content": problem},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            pred = extract_answer(text)
            correct = str(pred).strip() == answer if pred else False
            return (idx, rollout_idx, correct, pred, answer)
        except Exception as e:
            print(f"  ERROR problem {idx} rollout {rollout_idx}: {type(e).__name__}: {str(e)[:200]}")
            return (idx, rollout_idx, False, None, answer)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate model on AIME 2025")
    parser.add_argument("--base-url", required=True, help="vLLM server base URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--max-tokens", type=int, default=15000, help="Max generation tokens (must fit in model context with input)")
    parser.add_argument("--num-rollouts", type=int, default=4, help="Rollouts per problem")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds")
    parser.add_argument("--num-examples", type=int, default=-1, help="Number of problems to eval (-1 = all)")
    args = parser.parse_args()

    # Load dataset
    ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    ds = concatenate_datasets([ds1, ds2])
    if args.num_examples > 0:
        ds = ds.select(range(min(args.num_examples, len(ds))))
    print(f"Dataset: {len(ds)} AIME 2025 problems")
    print(f"Model: {args.model} @ {args.base_url}")
    print(f"Config: {args.num_rollouts} rollouts, max_tokens={args.max_tokens}, temp={args.temperature}, concurrency={args.concurrency}")

    client = AsyncOpenAI(base_url=args.base_url, api_key="empty", timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)

    # Verify server is up
    try:
        models = await client.models.list()
        available = [m.id for m in models.data]
        print(f"Server models: {available}")
        assert args.model in available, f"{args.model} not found in {available}"
    except Exception as e:
        print(f"FATAL: Cannot reach server at {args.base_url}: {e}")
        return

    # Build tasks
    tasks = []
    for i, row in enumerate(ds):
        problem = row["question"]
        answer = normalize_answer(str(row["answer"]).strip())
        for r in range(args.num_rollouts):
            tasks.append(eval_single(client, args.model, semaphore, problem, answer, i, r, args.max_tokens, args.temperature))

    print(f"\nRunning {len(tasks)} evaluations...")
    results = await asyncio.gather(*tasks)

    # Check for errors
    errors = sum(1 for _, _, _, pred, _ in results if pred is None)
    if errors > 0:
        print(f"\nWARNING: {errors}/{len(results)} requests failed!")

    # Aggregate
    problem_results = defaultdict(list)
    for idx, rollout_idx, correct, pred, answer in results:
        problem_results[idx].append(correct)

    avg_scores = []
    for idx in sorted(problem_results.keys()):
        corrects = problem_results[idx]
        avg_scores.append(sum(corrects) / len(corrects))

    n = args.num_rollouts
    total_avg = sum(avg_scores) / len(avg_scores)
    pass_n = sum(1.0 if any(problem_results[i]) else 0.0 for i in sorted(problem_results)) / len(problem_results)

    print(f"\n{'='*50}")
    print(f"  {args.model} on AIME 2025 ({len(ds)} problems)")
    print(f"{'='*50}")
    print(f"  Avg@{n}:  {total_avg:.4f}")
    print(f"  Pass@{n}: {pass_n:.4f}")
    print(f"  Errors:  {errors}/{len(results)}")
    print(f"\n  Per-problem:")
    for idx in sorted(problem_results.keys()):
        corrects = problem_results[idx]
        print(f"    {idx:2d}: {sum(corrects)}/{len(corrects)}")


if __name__ == "__main__":
    asyncio.run(main())
