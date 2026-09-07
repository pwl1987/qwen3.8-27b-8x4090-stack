#!/bin/bash
# Qwen3.5-2B 视觉/镜头分析轻量王（7.4s/条 视频基准）：~5G | M2分类器+e2e画面分析主力
source /data/tools/serve/env.sh
exec $V/bin/vllm serve /data/models/Qwen3.5-2B --port ${PORT:-8010} --gpu-memory-utilization 0.85 \
  --max-model-len ${LEN:-8192} --enforce-eager --limit-mm-per-prompt '{"video":1,"image":1}'
