#!/bin/bash
# Qwen3.6-27B-AWQ 视频画质档（VideoMME 87.7；71.1s/条）：双卡TP2 | 需配 reasoning parser 剥思考
source /data/tools/serve/env.sh
export CUDA_VISIBLE_DEVICES=${GPUS:-2,3}
exec $V/bin/vllm serve /data/models/Qwen3.6-27B-AWQ-INT4 --port ${PORT:-8010} \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 49152 \
  --max-num-batched-tokens 8192 --enforce-eager --limit-mm-per-prompt '{"video":1,"image":1}'
