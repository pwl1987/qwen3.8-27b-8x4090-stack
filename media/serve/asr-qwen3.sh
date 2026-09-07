#!/bin/bash
# Qwen3-ASR-1.7B 原生转写服务（vLLM 0.28）：GPU 3.9G权重/21.5G@0.85 | 271×实时@8并发 | 45s分片OK
source /data/tools/serve/env.sh
exec $V/bin/vllm serve /data/models/Qwen3-ASR-1.7B --port ${PORT:-8010} --gpu-memory-utilization 0.85
