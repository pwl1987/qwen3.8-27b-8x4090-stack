#!/bin/bash
# 媒体检索/视觉周边池三容器（复现自 docker inspect，宿主网络，权重挂 /data/models:ro）
# 布局: GPU5=embed(:19623)+rerank(:19624)  GPU7=Qwen3-VL-8B(:19625)
# 镜像: nvidia-4090-llm-inference:0.27.1-cu129（复用 vLLM 栈镜像）
set -e
IMG=nvidia-4090-llm-inference:0.27.1-cu129
COMMON="--restart unless-stopped --network host -v /data/models:/data/models:ro"

# ---- GPU5 · Qwen3-Embedding-0.6B :19623 ----
docker run -d --name zx-embed --gpus '"device=5"' $COMMON --entrypoint bash $IMG /app/venv/bin/vllm serve \
  /data/models/Qwen3-Embedding-0.6B --port 19623 --runner pooling \
  --gpu-memory-utilization 0.25 --max-model-len 8192

# ---- GPU5 · Qwen3-Reranker-0.6B :19624 ----
# 三坑（缺一即无区分度/报错）: ①--runner pooling ②--chat-template qwen3_reranker.jinja
# ③--hf-overrides 里 SeqCls 架构 + classifier_from_token ["no","yes"]
docker run -d --name zx-rerank --gpus '"device=5"' $COMMON --entrypoint bash $IMG /app/venv/bin/vllm serve \
  /data/models/Qwen3-Reranker-0.6B --port 19624 --runner pooling \
  --gpu-memory-utilization 0.25 --max-model-len 8192 \
  --chat-template /data/models/qwen3_reranker.jinja \
  --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"is_original_qwen3_reranker":true,"classifier_from_token":["no","yes"]}'

# ---- GPU7 · Qwen3-VL-8B-Instruct :19625 ----
docker run -d --name zx-vl8b --gpus '"device=7"' $COMMON --entrypoint bash $IMG /app/venv/bin/vllm serve \
  /data/models/Qwen3-VL-8B-Instruct --port 19625 \
  --gpu-memory-utilization 0.95 --max-model-len 16384

# 健康检查
for p in 19623 19624 19625; do
  sleep 3; curl -s -o /dev/null -w "port $p: %{http_code}\n" --max-time 5 http://127.0.0.1:$p/v1/models
done
# 检索+重排端到端时延参考: 中文查询命中 0.06-0.08s
