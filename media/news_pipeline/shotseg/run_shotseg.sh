#!/bin/bash
# 镜头分割编排：跨 venv 三信号 + 融合。用法: run_shotseg.sh <video> <workdir> [gpu_id]
set -e
VID=$1; W=$2; GPU=${3:-}
export CPATH=/data/tools/ffmpeg-deb/root/usr/include/python3.12:/data/tools/ffmpeg-deb/root/usr/include
mkdir -p "$W"
if [ -n "$GPU" ]; then export CUDA_VISIBLE_DEVICES=$GPU; fi
echo "[1/4] SigLIP 嵌入"; /data/tools/mage-env/bin/python /data/tools/shotseg/stage1_siglip.py "$VID" "$W"
echo "[2/4] 音频信号";   /data/tools/pyav-env/bin/python /data/tools/shotseg/stage2_audio.py "$VID" "$W"
echo "[3/4] 字幕条 OCR"; /data/tools/ocr-env/bin/python /data/tools/shotseg/stage3_ocr.py "$W"
echo "[4/4] 融合";       python3 /data/tools/shotseg/fuse_shots.py "$W"
