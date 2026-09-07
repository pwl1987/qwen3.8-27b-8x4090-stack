#!/usr/bin/env bash
# Comfy-Org/MiniMax-H3 双4090套餐下载 (hf-mirror, 断点续传+重试)
set -u
BASE=https://hf-mirror.com/Comfy-Org/MiniMax-H3/resolve/main
DEST=/data/comfyui/models
LOG=/data/comfyui/dl
mkdir -p "$DEST"/{diffusion_models,text_encoders,vae,loras} "$LOG"

dl() {  # dl <相对路径> <日志名>
  local rel=$1 logn=$2
  curl -L -C - --retry 20 --retry-delay 5 --retry-all-errors \
       -o "$DEST/$rel" "$BASE/$rel" >"$LOG/$logn.log" 2>&1
  echo "$rel exit=$?" >>"$LOG/status.txt"
}

dl diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors dit &
dl text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors      te  &
dl vae/minimax_h3_video_vae_fp16.safetensors                        vae &
dl vae/minimax_h3_audio_vae_fp32.safetensors                        ava &
dl loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors  lora &
wait
echo "ALL_DONE $(date '+%F %T')" >>"$LOG/status.txt"
