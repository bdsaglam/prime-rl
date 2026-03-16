# Qwen3 8B 

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-8B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.4 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m Qwen/Qwen3-8B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3-8B -b http://0.0.0.0:8900/v1

# Qwen3 8B (willcb)

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-8B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m willcb/Qwen3-8B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m willcb/Qwen3-8B -b http://0.0.0.0:8900/v1

# Qwen3 14B

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-14B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.7 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m Qwen/Qwen3-14B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3-14B -b http://0.0.0.0:8900/v1


# Qwen3 32B

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-32B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m Qwen/Qwen3-32B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3-32B -b http://0.0.0.0:8900/v1

# Qwen3 32B (willcb)

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-32B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m willcb/Qwen3-32B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m willcb/Qwen3-32B -b http://0.0.0.0:8900/v1

# Qwen3-Coder-Next 80B (3B active)

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-Coder-Next \
    --port 8900 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder


# Devstral 2 Small

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve mistralai/Devstral-Small-2-24B-Instruct-2512 \
    --port 8900 \
    --data-parallel-size 4 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.75 \
    --enforce-eager \
    --max-model-len 65536 \
    --tool-call-parser mistral --enable-auto-tool-choice

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m mistralai/Devstral-Small-2-24B-Instruct-2512 -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m mistralai/Devstral-Small-2-24B-Instruct-2512 -b http://0.0.0.0:8900/v1

# GLM 4.7 Flash


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve zai-org/GLM-4.7-Flash \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 65536 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 1 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-4.7-flash

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m zai-org/GLM-4.7-Flash -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m zai-org/GLM-4.7-Flash -b http://0.0.0.0:8900/v1

# Nanbeige/Nanbeige4.1-3B

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Nanbeige/Nanbeige4.1-3B \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.7 \
    --dtype bfloat16 \
    --enforce-eager \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-model-len 65536

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -r 3 -m Nanbeige/Nanbeige4.1-3B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Nanbeige/Nanbeige4.1-3B -b http://0.0.0.0:8900/v1

uv run rl @ configs/prime-rl/arc-agi-nanbeige.toml

# Kill GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# OpenRouter 

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 \
    -m arcee-ai/trinity-large-preview:free \
    -b https://openrouter.ai/api/v1 \
    -k OPENROUTER_API_KEY

# gpt-oss-120b

docker run --rm --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8900:8900 \
  --ipc=host \
  vllm/vllm-openai:latest \
  openai/gpt-oss-120b \
  --port 8900 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 65536 \
  --tool-call-parser openai

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve openai/gpt-oss-120b \
    --port 8900 \
    --async-scheduling \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-auto-tool-choice --tool-call-parser openai \
    --enforce-eager

prime eval run arc-agi -x '{"data_dir":"data/arc-dummy"}' -n 1 -m openai/gpt-oss-120b -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"data_dir":"data/arc-prize-2024"}' -n 4 -r 3 -m openai/gpt-oss-120b -b http://0.0.0.0:8900/v1

# Nemotron Cascade 14B

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve nvidia/Nemotron-Cascade-14B-Thinking \
    --port 8900 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m nvidia/Nemotron-Cascade-14B-Thinking -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m nvidia/Nemotron-Cascade-14B-Thinking -b http://0.0.0.0:8900/v1

# Liquid AI

prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 \
    -m liquid/lfm-2.5-1.2b-thinking:free \
    -b https://openrouter.ai/api/v1 \
    -k OPENROUTER_API_KEY


prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 \
    -m liquid/lfm-2.5-1.2b-thinking:free \
    -b https://openrouter.ai/api/v1 \
    -k OPENROUTER_API_KEY

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 \
    -m qwen/qwen3.5-flash-02-23 \
    -b https://openrouter.ai/api/v1 \
    -k OPENROUTER_API_KEY

# Teacher


prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3.5-27B -b http://0.0.0.0:8907/v1


# Qwen3.5-27B

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8900:8900 \
    --ipc=host \
    vllm/vllm-openai:qwen3_5 \
    --port 8900 \
    --model Qwen/Qwen3.5-27B \
    --async-scheduling \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes


prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m Qwen/Qwen3.5-27B -b http://0.0.0.0:8900/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3.5-27B -b http://0.0.0.0:8900/v1

# Qwen3.5-9B

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8907:8907 \
    --ipc=host \
    vllm/vllm-openai:qwen3_5 \
    --port 8907 \
    --model Qwen/Qwen3.5-9B \
    --async-scheduling \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes


prime eval run arc-agi -x '{"dataset_name":"arc-dummy"}' -n 1 -r 1 -m Qwen/Qwen3.5-9B -b http://0.0.0.0:8907/v1

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3.5-9B -b http://0.0.0.0:8907/v1

# Fix flash-attn issue
uv sync --extra flash-attn

# Teacher

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-32B \
    --port 8932 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve willcb/Qwen3-32B \
    --port 8932 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser hermes

prime eval run arc-agi -x '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m willcb/Qwen3-32B -b http://0.0.0.0:8932/v1

# Qwen3.5-27B

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-27B \
    --port 8907 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --language-model-only \
    --enforce-eager \
    --enable-prefix-caching

prime eval run arc-agi -a '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3.5-27B -b http://0.0.0.0:8907/v1

# Qwen3.5-9B

docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8907:8907 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --port 8907 \
    --model Qwen/Qwen3.5-9B \
    --async-scheduling \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.80 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --reasoning-parser qwen3 \
    --enable-prefix-caching


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.5-9B \
    --port 8907 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager \
    --enable-prefix-caching 

prime eval run arc-agi -a '{"dataset_name":"arc-prize-2024"}' -n 4 -r 3 -m Qwen/Qwen3.5-9B -b http://0.0.0.0:8907/v1

# Qwen3.5-35B-A3B

docker run --runtime nvidia --gpus 1,2 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8935:8935 \
    --ipc=host \
    vllm/vllm-openai \
    --port 8935 \
    --model Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder


docker run --runtime nvidia --gpus 1,2 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8935:8935 \
    --ipc=host \
    vllm/vllm-openai \
    --port 8935 \
    --model Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --reasoning-parser qwen3 \
    --enable-prefix-caching


# Qwen3.5-27B


CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-27B \
    --port 8927 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --reasoning-parser qwen3 \
    --enforce-eager \
    --enable-prefix-caching

docker run --runtime nvidia --gpus 0 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8927:8927 \
    --ipc=host \
    vllm/vllm-openai:qwen3_5 \
    --port 8927 \
    --model Qwen/Qwen3.5-27B \
    --async-scheduling \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 