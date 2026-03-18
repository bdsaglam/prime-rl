#!/bin/bash
# Launch next ablation run after current one finishes
# Usage: bash launch-next.sh

set -e

echo "Waiting for current run to finish..."
echo "Checking every 60s..."

while true; do
    # Check if any prime_rl training processes are still running
    if ! pgrep -f "prime_rl.entrypoints.rl" > /dev/null 2>&1; then
        echo "No training process found. Launching next run..."
        break
    fi
    sleep 60
done

# Kill any zombie vLLM processes
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

# Launch ablation D: Self-Reflection OPD on AIME 2025
echo "Launching ablation D: Self-Reflection OPD on AIME 2025"
cd /home/baris/repos/prime-rl
python -m prime_rl.entrypoints.rl @ configs/aime_mt/ablation-D-self-reflection-2025.toml 2>&1 | tee outputs/aime-ablation-D-self-reflection-2025/run.log
