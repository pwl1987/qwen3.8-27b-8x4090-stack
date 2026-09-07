# services/ — 检索/视觉周边池

`start-services.sh`：三容器可复现启动（复现自 docker inspect）——GPU5=embed(:19623)+
rerank(:19624)、GPU7=Qwen3-VL-8B(:19625)，复用 vLLM 栈镜像，宿主网络。
reranker 三坑（P54）已写进脚本注释：--runner pooling / qwen3_reranker.jinja /
classifier_from_token ["no","yes"]。E2E 中文检索+重排 0.06-0.08s。
