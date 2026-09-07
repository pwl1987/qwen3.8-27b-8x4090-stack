#!/bin/bash
# MiniCPM-V4.5-AWQ 视频吞吐档（5.4s/条 速度王，字幕细节最强）：单卡 6.7G
source /data/tools/serve/env.sh
exec $V/bin/vllm serve /data/models/MiniCPM-V-4_5-AWQ --port ${PORT:-8010} --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --enforce-eager --trust-remote-code --limit-mm-per-prompt '{"video":1,"image":1}'
