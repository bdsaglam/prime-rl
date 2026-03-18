"""Analyze AIME ablation study results from W&B.

Usage:
    python research/on-policy-distillation/experiments/aime-ablation/analyze.py
"""

import wandb
import numpy as np

PROJECT = "bdsaglam/aime-opd"

# Map condition names to W&B run display names (trainer runs only)
ABLATION_RUNS = {
    "A: GRPO Only": "ablation-A-grpo-only-2025-trainer",
    "B: Answer OPD": "ablation-B-answer-opd-2025-trainer",
    "C: Deliberative OPD": "ablation-C-deliberative-opd-trainer",
    "D: Self-Reflection OPD": "ablation-D-self-reflection-2025-trainer",
    "E: SDPO-style OPD": "ablation-E-sdpo-pi-2025-trainer",
}

ORCHESTRATOR_RUNS = {
    "A: GRPO Only": "ablation-A-grpo-only-2025-orchestrator",
    "B: Answer OPD": "ablation-B-answer-opd-2025-orchestrator",
    "C: Deliberative OPD": "ablation-C-deliberative-opd-orchestrator",
    "D: Self-Reflection OPD": "ablation-D-self-reflection-2025-orchestrator",
    "E: SDPO-style OPD": "ablation-E-sdpo-pi-2025-orchestrator",
}


def find_run(display_name: str) -> wandb.apis.public.Run | None:
    """Find a W&B run by display name."""
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={"display_name": display_name}, order="-created_at")
    return runs[0] if runs else None


def get_trainer_metrics(run) -> dict:
    """Pull trainer metrics (teacher_kl, loss, entropy, masking)."""
    h = run.history(
        keys=["teacher_kl/mean", "mismatch_kl/mean", "loss/mean", "entropy/mean", "is_masked/mean"],
        pandas=False,
    )
    return {
        "steps": [r.get("_step") for r in h],
        "teacher_kl": [r.get("teacher_kl/mean", 0) for r in h],
        "mismatch_kl": [r.get("mismatch_kl/mean", 0) for r in h],
        "loss": [r.get("loss/mean", 0) for r in h],
        "entropy": [r.get("entropy/mean", 0) for r in h],
        "masked_pct": [r.get("is_masked/mean", 0) for r in h],
    }


def get_orchestrator_metrics(run) -> dict:
    """Pull orchestrator metrics (reward, eval)."""
    h = run.history(
        keys=["reward/mean", "metrics/correct_answer", "step"],
        pandas=False,
    )
    return {
        "steps": [r.get("step") for r in h],
        "reward": [r.get("reward/mean", 0) for r in h],
        "correct_pct": [r.get("metrics/correct_answer", 0) for r in h],
    }


def summarize_condition(name: str, trainer_run, orch_run) -> dict:
    """Summarize a single ablation condition."""
    tm = get_trainer_metrics(trainer_run) if trainer_run else {}
    om = get_orchestrator_metrics(orch_run) if orch_run else {}

    teacher_kl = tm.get("teacher_kl", [])
    reward = om.get("reward", [])

    return {
        "name": name,
        "steps": len(teacher_kl),
        "teacher_kl_mean": np.mean(teacher_kl) if teacher_kl else None,
        "teacher_kl_std": np.std(teacher_kl) if teacher_kl else None,
        "teacher_kl_abs_mean": np.mean(np.abs(teacher_kl)) if teacher_kl else None,
        "entropy_mean": np.mean(tm.get("entropy", [])) if tm.get("entropy") else None,
        "reward_mean": np.mean(reward) if reward else None,
        "reward_last5": np.mean(reward[-5:]) if len(reward) >= 5 else None,
    }


def main():
    api = wandb.Api()
    print("=" * 80)
    print("AIME ABLATION STUDY: Deliberative Self-Teaching")
    print("=" * 80)

    for name in ABLATION_RUNS:
        trainer_name = ABLATION_RUNS[name]
        orch_name = ORCHESTRATOR_RUNS[name]

        trainer_run = find_run(trainer_name)
        orch_run = find_run(orch_name)

        status = "FOUND" if trainer_run else "NOT FOUND"
        print(f"\n{'─' * 60}")
        print(f"{name}: {status}")

        if trainer_run:
            print(f"  Trainer: {trainer_run.id} ({trainer_run.state})")
            summary = summarize_condition(name, trainer_run, orch_run)
            print(f"  Steps completed: {summary['steps']}")
            if summary["teacher_kl_mean"] is not None:
                print(f"  Teacher KL: {summary['teacher_kl_mean']:.4f} ± {summary['teacher_kl_std']:.4f} (|mean|={summary['teacher_kl_abs_mean']:.4f})")
            if summary["entropy_mean"] is not None:
                print(f"  Entropy: {summary['entropy_mean']:.4f}")
            if summary["reward_mean"] is not None:
                print(f"  Reward (all): {summary['reward_mean']:.3f}")
            if summary["reward_last5"] is not None:
                print(f"  Reward (last 5): {summary['reward_last5']:.3f}")

        if orch_run:
            print(f"  Orchestrator: {orch_run.id} ({orch_run.state})")

    print(f"\n{'=' * 80}")
    print("Check W&B for full plots: https://wandb.ai/bdsaglam/aime-opd")


if __name__ == "__main__":
    main()
