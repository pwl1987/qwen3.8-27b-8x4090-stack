# ComfyUI + MiniMax-H3 (双4090) —— 本地基础镜像零拉取, 全部状态走 volume, 宿主零安装
FROM docker.m.daocloud.io/library/python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    HF_ENDPOINT=https://hf-mirror.com

# apt 换阿里云源(本机到 deb.debian.org 不稳)
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null; \
    sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends git gcc ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# torch 三件套 cu130 (驱动已升 580.173.02 / CUDA 13.0); 三者必须同版本钉死
# (torchaudio 新版是 cu13 构建, 与 cu12 torch 混装会在 libcudart 上崩——历史坑, 钉死对齐版)
# 轮子从 /data/comfyui/wheels 本地 COPY (buildkit 层内大文件下载会卡死, 宿主 wget -c 稳定)
COPY wheels/ /wheels/
RUN pip install --no-cache-dir --find-links=/wheels \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130

# 源码本地 COPY (0.34.0 快照, 从 cu128 回滚镜像提取; github 直连/codeload 当日均不稳)
COPY src/ComfyUI /app/ComfyUI
RUN set -eux; \
    test -f /app/ComfyUI/main.py; \
    pip install --no-cache-dir -r /app/ComfyUI/requirements.txt && \
    pip install --no-cache-dir --force-reinstall --no-deps \
      --find-links=/wheels --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      torchaudio==2.11.0+cu130; \
    rm -rf /wheels; \
    python -c "import torchaudio; print('torchaudio', torchaudio.__version__)"

# SageAttention 1.x (纯 triton, sm89 实测误差 0.001; 需 gcc 做 JIT) + 启动旗见 compose --use-sage-attention
RUN pip install --no-cache-dir sageattention==1.0.6 \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app/ComfyUI
# 容器以 uid 1000 运行(compose user:): 建 passwd 条目(torch dynamo getpass.getuser 需要) + 可写运行时目录
RUN useradd -u 1000 -d /home/comfy -m comfy && chown -R 1000:1000 /app/ComfyUI
ENV HOME=/home/comfy
EXPOSE 8188
ENTRYPOINT ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
