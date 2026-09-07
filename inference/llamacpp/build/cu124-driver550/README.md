# llama.cpp 构建配方 · CUDA 12.4 / 驱动 550 时代（生产现役）

现役生产镜像 `llama-server:cuda12.4-b10715` 的构建资料（2026-08-31，驱动 550.163.01 时代）。

- `Dockerfile.runtime`：运行镜像（`nvidia/cuda:12.4.1-runtime-ubuntu22.04` + 清华源 + 复制预编译 bin）
- 源码：llama.cpp **b10715**（含本仓库 `training/llamacpp/` 的 LoRA 转换补丁两文件）
- 编译要点：必须在 devel 容器内加 `--gpus all`（链接需要 libcuda），产物 `build/bin` 打进 runtime 镜像
- 注：驱动升至 580 后此镜像照常运行（驱动向后兼容）；如需 cu13 原生重编译，参考
  `inference/vllm/build/cu130-driver580/` 的 wheel 自带工具链思路（llama.cpp 同理可用 cu13 devel 基镜像）
