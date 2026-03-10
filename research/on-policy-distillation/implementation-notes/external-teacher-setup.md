# External Teacher Server Setup

Host the teacher model (Qwen3-32B) on a remote server instead of locally, freeing up a GPU.

## Remote Server

SSH into the server and start vLLM:

```bash
vllm serve willcb/Qwen3-32B \
  --dtype bfloat16 \
  --max-model-len 20000 \
  --gpu-memory-utilization 0.88 \
  --port 8907
```

Verify it's running:

```bash
curl http://localhost:8907/v1/models
```

## SSH Tunnel

From your local machine, forward port 8907:

```bash
ssh -N -L 8907:localhost:8907 144.122.52.26
```

## Config

Use `configs/arc_agi/opd-rl-qwen-8b-teacher-context-external.toml`. The key difference from the local teacher config:

```toml
[deployment]
num_train_gpus = 1
num_infer_gpus = 2
# no num_teacher_gpus — teacher is remote

# Instead of [teacher_inference], point orchestrator at the tunnel:
[orchestrator.teacher_model.client]
base_url = ["http://localhost:8907/v1"]

[orchestrator.teacher_model.model]
name = "willcb/Qwen3-32B"

[orchestrator.teacher_model]
max_model_len = 20000
```

## How It Works

- When `num_teacher_gpus` is absent and there's no `[teacher_inference]` section, prime-rl does **not** launch a local teacher process.
- The orchestrator connects directly to whatever URL is in `orchestrator.teacher_model.client.base_url`.
- Teacher logprobs are computed via vLLM's `/chat/completions/tokens` endpoint (prefill-only, no generation). This is vLLM-specific — standard OpenAI-compatible APIs (OpenRouter, Together.ai) won't work.
