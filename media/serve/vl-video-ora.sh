#!/bin/bash
# Video-ORA-9B 中坚档（VideoMME 76.7；23.4s/条；时序grounding强）：BF16 17.6G 单卡
source /data/tools/serve/env.sh
exec $V/bin/vllm serve /data/models/Video-ORA-9B --port ${PORT:-8010} --gpu-memory-utilization 0.92 \
  --max-model-len 32768 --max-num-batched-tokens 8192 --enforce-eager --limit-mm-per-prompt '{"video":1,"image":1}'
