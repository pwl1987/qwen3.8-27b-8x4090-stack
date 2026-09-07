#!/bin/bash
# serve 公共环境（所有启动脚本 source 此文件）
export V=/data/tools/vllm28-env
export CUDA13=$V/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CUDA13
export PATH=$CUDA13/bin:$V/bin:$PATH
export CPATH=/data/tools/ffmpeg-deb/root/usr/include/python3.12:/data/tools/ffmpeg-deb/root/usr/include
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
