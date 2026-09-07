#!/bin/bash
# GLM-4.6V-Flash-AWQ 轻量备选（20.2s/条）：⚠️需先改模型目录 video_preprocessor_config.json
#   longest_edge=31457280（默认1亿像素会超编码缓存 400）
source /data/tools/serve/env.sh
exec $V/bin/vllm serve /data/models/GLM-4.6V-Flash-AWQ-4bit --port ${PORT:-8010} --gpu-memory-utilization 0.85 \
  --max-model-len 98304 --max-num-batched-tokens 8192 --enforce-eager --limit-mm-per-prompt '{"video":1,"image":1}'
