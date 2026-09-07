# vLLM 生产镜像 · CUDA 12.9 / 驱动 550+ 时代（生产现役）

生产镜像 `nvidia-4090-llm-inference:0.27.1-cu129` 的配方即本目录上级的 `inference/vllm/docker/`：

- vLLM **0.27.1** 官方 wheel（`+cu129`，requirements-cu129.txt 钉死 URL 防 PyPI CUDA 漂移）
- 基镜像 `nvidia/cuda:12.9.1-base-ubuntu24.04`（README 注：Ulmus driver 550.163.02 不兼容 CUDA 13 预编译镜像——当年选 cu129 的原因）
- 构建时 28 补丁 + KVarN 自动应用（见 `docker/Dockerfile.cu129`）
- 驱动 580 下同样照常运行；cu13 原生环境见隔壁 `cu130-driver580/`
