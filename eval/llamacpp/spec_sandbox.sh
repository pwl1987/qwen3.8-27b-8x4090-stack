#!/usr/bin/env bash
# spec_sandbox.sh — GPU6:8085 上对 coding-v1 做 spec decoding A/B 矩阵
# 用法: ./spec_sandbox.sh matrix   # 4 配置: mtp / mtp+ngram / dflash / dflash+ngram
#       ./spec_sandbox.sh vram     # 256K 生产形态显存可行性 (dflash)
#       ./spec_sandbox.sh ubatch   # 最优配置上 -b/-ub 扫参
set -uo pipefail

GPU=6; PORT=8085; CT=qwen27b-sbx
IMAGE=llama-server:cuda12.4-b10715
MODEL=/models/Qwen3.8-27B-coding-v1-iq4xs.gguf
DFLASH=/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
OUT=/data/eval-rulers/spec-sandbox-20260904
mkdir -p "$OUT"
BASE_ARGS="-m $MODEL --host 0.0.0.0 --port 8080 -ngl 999 -np 1 \
 --cache-type-k q4_0 --cache-type-v q4_0 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
 -fa on --cache-reuse 2048 --jinja -a sbx --metrics"

boot() { # $1=label, $2=ctx, 其余=额外 args
  local label=$1 ctx=$2; shift 2
  docker rm -f $CT >/dev/null 2>&1 || true
  echo "=== boot [$label] ctx=$ctx extra: $*"
  docker run -d --name $CT --gpus "device=$GPU" -v /data/models:/models \
    -p 127.0.0.1:$PORT:8080 "$IMAGE" $BASE_ARGS -c $ctx "$@" >/dev/null || { echo "BOOT FAIL [$label]"; docker logs $CT 2>&1 | tail -30; return 1; }
  for i in $(seq 1 150); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && { sleep 2; return 0; }
    sleep 2
  done
  echo "HEALTH TIMEOUT [$label]"; docker logs $CT 2>&1 | tail -30; return 1
}

vram() { nvidia-smi --id=$GPU --query-gpu=memory.used,memory.total --format=csv,noheader; }

falog() { # 启动日志关键行: FA / KV / spec
  docker logs $CT 2>&1 | grep -iE 'flash|att.*cache|kv cache|spec|draft|ngram|dflash' | grep -vE '^\s*$' | head -25
}

matrix() {
  local -a CFGS=(
    "mtp|--spec-type draft-mtp"
    "mtp-ngram|--spec-type draft-mtp,ngram-cache"
    "dflash|--spec-type draft-dflash -md $DFLASH --spec-draft-n-max 7"
    "dflash-ngram|--spec-type draft-dflash,ngram-cache -md $DFLASH --spec-draft-n-max 7"
  )
  for cfg in "${CFGS[@]}"; do
    local label="${cfg%%|*}" args="${cfg#*|}"
    boot "$label" 65536 $args || continue
    { echo "--- [$label] vram: $(vram)"; falog; } | tee "$OUT/log-$label.txt"
    python3 /data/compose/qwen27b/eval/spec_bench.py "$label" $PORT "$OUT/bench-$label.json"
    echo "--- [$label] vram after: $(vram)"
    docker rm -f $CT >/dev/null 2>&1
  done
}

vram_check() {
  boot vram-256k-dflash 262144 --spec-type draft-dflash,ngram-cache -md $DFLASH --spec-draft-n-max 7 -np 2 || return 1
  echo "--- 256K+dflash+ngram 生产形态:"; vram; falog
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"回复OK"}],"max_tokens":16,"temperature":0}' | head -c 400; echo
  docker rm -f $CT >/dev/null 2>&1
}

ubatch() { # 在 dflash 最优猜测配置上扫 (b,ub)
  local -a TUNES=("ub512|-b 2048 -ub 512" "ub1024|-b 4096 -ub 1024" "ub2048|-b 8192 -ub 2048")
  for t in "${TUNES[@]}"; do
    local label="${t%%|*}" args="${t#*|}"
    boot "sweep-$label" 65536 --spec-type draft-dflash -md $DFLASH --spec-draft-n-max 7 $args || continue
    echo "--- [sweep-$label] vram: $(vram)"
    python3 /data/compose/qwen27b/eval/spec_bench.py "sweep-$label" $PORT "$OUT/bench-sweep-$label.json"
    docker rm -f $CT >/dev/null 2>&1
  done
}

case "${1:-matrix}" in
  matrix) matrix ;;
  vram) vram_check ;;
  ubatch) ubatch ;;
  *) echo "用法: $0 matrix|vram|ubatch"; exit 1 ;;
esac
