#!/usr/bin/env bash
# MiniMax-H3 ComfyUI 文件下载 (hf-mirror 直链, wget -c 断点续传, 字节数校验)
# 用法: bash download_models.sh   (可重复执行, 已完整的文件自动跳过)
set -u
BASE="https://hf-mirror.com/Comfy-Org/MiniMax-H3/resolve/main"
DEST="$(cd "$(dirname "$0")" && pwd)/models"
LOG="$(dirname "$0")/download.log"

# path|expected_bytes
FILES=(
  "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors|20958205608"
  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors|15687142551"
  "vae/minimax_h3_video_vae_fp16.safetensors|5207808496"
  "vae/minimax_h3_audio_vae_fp32.safetensors|605254808"
  "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors|1956193000"
  "embeddings/minimaxh3_art_is_explosion.safetensors|0"
  "embeddings/minimaxh3_blooming_flowers.safetensors|0"
  "embeddings/minimaxh3_bullet_time.safetensors|0"
  "embeddings/minimaxh3_dark_magic.safetensors|0"
  "embeddings/minimaxh3_fire_breath.safetensors|0"
  "embeddings/minimaxh3_four_seasons.safetensors|0"
  "embeddings/minimaxh3_kiss_camera.safetensors|0"
  "embeddings/minimaxh3_spiral_ascent.safetensors|0"
  "embeddings/minimaxh3_storm_magic.safetensors|0"
  "embeddings/minimaxh3_truman_show.safetensors|0"
)

ok=0; skip=0; fail=0
for entry in "${FILES[@]}"; do
  rel="${entry%%|*}"; want="${entry##*|}"
  out="$DEST/$rel"
  mkdir -p "$(dirname "$out")"
  have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  # want=0 表示小文件不校验, 存在即跳过
  if [[ "$have" -gt 0 && ( "$want" == "0" || "$have" == "$want" ) ]]; then
    echo "$(date '+%T') SKIP $rel ($have bytes)" >> "$LOG"; skip=$((skip+1)); continue
  fi
  echo "$(date '+%T') GET  $rel" >> "$LOG"
  if wget -c -q --show-progress --progress=dot:giga -O "$out" "$BASE/$rel" 2>> "$LOG"; then
    have=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [[ "$want" == "0" || "$have" == "$want" ]]; then
      echo "$(date '+%T') DONE $rel ($have bytes)" >> "$LOG"; ok=$((ok+1))
    else
      echo "$(date '+%T') SIZE-MISMATCH $rel want=$want got=$have (重跑续传)" >> "$LOG"; fail=$((fail+1))
    fi
  else
    echo "$(date '+%T') FAIL $rel (重跑续传)" >> "$LOG"; fail=$((fail+1))
  fi
done
echo "$(date '+%T') SUMMARY ok=$ok skip=$skip fail=$fail" >> "$LOG"
echo "ok=$ok skip=$skip fail=$fail (详见 $LOG)"
