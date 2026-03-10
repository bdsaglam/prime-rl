# Agent Workflow: Managing Long-Running Tasks

The agent acts as an OS-level task manager — not just writing code, but launching, monitoring, and managing long-running processes.

## Principles

1. **Never stop until the goal is achieved.** There must always be at least one active background monitor task while work is in progress. If the agent has no background task and stops, it goes dormant until the user returns — which means crashes, instability, and GPU problems go undetected for hours. Always keep a heartbeat running.
2. **tmux for long-running processes.** Training runs, vLLM inference servers, evaluations — anything that runs for minutes/hours goes in a named tmux window.
3. **Background tasks for heartbeats only.** Background bash tasks are lightweight monitors that check a condition periodically (like a heartbeat) and notify the agent when something happens. They should NOT run actual workloads.
4. **Keep monitors short-lived (≤1 hour).** A single monitor should timeout within ~1 hour. If the process takes longer, the monitor triggers on timeout, the agent checks progress, and launches a new monitor. This gives the agent regular chances to detect and fix problems (crashes, OOM, stale processes) rather than sleeping through them.
5. **Minimize background tasks.** One monitor per active process is enough. Don't accumulate stale monitors. Kill them when no longer needed.
6. **Never let GPUs sit idle.** When a run finishes, the next experiment should launch immediately. Prepare configs in advance.

## What Goes Where

| Task Type | Where | Examples |
|-----------|-------|---------|
| Training runs | tmux | `python -m prime_rl.entrypoints.rl @ config.toml` |
| Inference servers | tmux | `python -m vllm.entrypoints.openai.api_server ...` |
| Evaluation scripts | tmux | `python eval.py ...` |
| Log tailing | tmux | `tail -F outputs/logs/orchestrator.stdout` |
| GPU monitoring | tmux | `watch -n 5 nvidia-smi` |
| Completion checks | bg task | `while ...; do grep "finished" log; sleep 60; done` |
| Error detection | bg task | `while ...; do grep "Traceback" log; sleep 30; done` |

## tmux Convention

Use a named tmux session (e.g., `opd-11`) with named windows:

```
opd-11:train        — main training process
opd-11:orchestrator — tail -F orchestrator.stdout
opd-11:gpu          — watch nvidia-smi
```

Launch processes via `tmux send-keys`:
```bash
tmux send-keys -t opd-11:train "python -m prime_rl.entrypoints.rl @ config.toml 2>&1 | tee /tmp/train.log" Enter
```

Read output via `tmux capture-pane`:
```bash
tmux capture-pane -t opd-11:train -p -S -50
```

## Background Tasks: The Heartbeat Pattern

Background tasks exist to **relieve the agent** so it can continue working or respond to the user while a long process runs. They are disposable monitors, not the workload itself.

**Good pattern — short-lived monitor with timeout:**
```bash
# Monitor for training completion — check every 60s, timeout after 45 min
TIMEOUT=2700  # 45 minutes
START=$(date +%s)
while true; do
  if grep -q "RL training finished" /tmp/train.log; then
    echo "DONE"; grep "Evaluated" logs/orchestrator.stdout; break
  fi
  if grep -q "Traceback" logs/orchestrator.stdout; then
    echo "ERROR"; tail -20 logs/orchestrator.stdout; break
  fi
  ELAPSED=$(( $(date +%s) - START ))
  if [ $ELAPSED -gt $TIMEOUT ]; then
    echo "HEARTBEAT: still running after ${TIMEOUT}s"
    tail -5 logs/orchestrator.stdout; break
  fi
  sleep 60
done
```

When the monitor returns (done, error, or timeout), the agent wakes up, checks status, fixes problems if needed, and launches a new monitor if work continues. This creates a supervision loop that catches problems within ~1 hour instead of sleeping indefinitely.

**Bad patterns:**
- Running actual training in a background task (use tmux)
- Starting vLLM servers as background tasks (use tmux)
- Having 5+ monitors for the same process
- Polling every 5 seconds (use 30-60s intervals)
- Forgetting to stop monitors after they're no longer needed
- **Letting the agent go idle with no background task while work is in progress**

## Lifecycle

1. **Before launching**: Kill zombie processes (`fuser /dev/nvidia*`), verify GPUs clean
2. **Launch**: Start process in tmux, start ONE heartbeat monitor as bg task
3. **While running**: Agent is free to do other work. Gets notified when monitor triggers
4. **On completion**: Stop the heartbeat monitor. Read results. Launch next experiment
5. **On error**: Stop monitor. Read logs from tmux. Fix. Restart in tmux

## Cleanup

At the start of each session or when the user requests:
- Stop all stale background tasks
- Kill zombie GPU processes
- Verify GPUs are clean
- Check tmux sessions for dead processes
