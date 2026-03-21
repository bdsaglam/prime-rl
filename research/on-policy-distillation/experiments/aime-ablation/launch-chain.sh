#!/bin/bash
# Launch ablation runs sequentially (all on AIME 2025)
# Current order: E (running) → D → B → A
# Usage: started in tmux session 'chain'

set -e
cd /home/baris/repos/prime-rl

run_and_wait() {
    local config="$1"
    local name="$2"
    local logdir="$3"

    echo "$(date): Launching $name"
    mkdir -p "$logdir"

    python -m prime_rl.entrypoints.rl @ "$config" 2>&1 | tee "$logdir/run.log"

    echo "$(date): $name finished"

    # Kill zombie vLLM processes
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 10
}

# Wait for E to finish (currently running)
echo "$(date): Waiting for current run (E) to finish..."
while true; do
    if ! pgrep -f "prime_rl.entrypoints.rl" > /dev/null 2>&1; then
        echo "$(date): No training process found."
        break
    fi
    sleep 60
done

# Kill any zombie vLLM
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 10

# Run D: Self-Reflection OPD on AIME 2025 (core hypothesis)
run_and_wait \
    "configs/aime_mt/ablation-D-self-reflection-2025.toml" \
    "Ablation D: Self-Reflection OPD (AIME 2025)" \
    "outputs/aime-ablation-D-self-reflection-2025"

# Run B: Answer-Only OPD on AIME 2025
run_and_wait \
    "configs/aime/ablation-B-answer-opd-2025.toml" \
    "Ablation B: Answer OPD (AIME 2025)" \
    "outputs/aime-ablation-B-answer-opd-2025"

# Run A: GRPO Only on AIME 2025
run_and_wait \
    "configs/aime/ablation-A-grpo-only-2025.toml" \
    "Ablation A: GRPO Only (AIME 2025)" \
    "outputs/aime-ablation-A-grpo-only-2025"

echo "$(date): All ablation runs complete!"
