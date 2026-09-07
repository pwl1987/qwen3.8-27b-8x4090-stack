#!/bin/bash
# Qwen3.5-0.8B 纯文本小模型（条目主题校验/批量轻判别）：~3G | 文本横评10.9s/条
source /data/tools/serve/env.sh
exec $V/bin/vllm serve /data/models/Qwen3.5-0.8B --port ${PORT:-8012} --gpu-memory-utilization 0.85 \
  --max-model-len ${LEN:-8192} --enforce-eager
