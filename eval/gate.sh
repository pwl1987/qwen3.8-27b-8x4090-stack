#!/usr/bin/env bash
# Phase 4 一键门禁: 双实例 A/B (控制=纯底座 / 实验=底座+LoRA热挂) → 全轴评测 → 判定
# 前置: 训练已完成(GPU4-7空闲), final-lora.gguf 已就绪
# 用法: bash gate.sh /data/compose/qwen27b/rft/final-lora.gguf
set -euo pipefail
ADAPTER_GGUF="${1:-/data/compose/qwen27b/rft/final-lora-q8.gguf}"
PY=/home/<USER>/miniforge3/envs/torch/bin/python
EVAL_DIR=/data/compose/qwen27b/eval
MODEL=/models/Qwen3.8-27B-Heretic-Ara-iq4_xs-3.0-mtp.gguf
IMG=llama-server:cuda12.4-b10715
SRV_ARGS='-m /models/Qwen3.8-27B-Heretic-Ara-iq4_xs-3.0-mtp.gguf --host 0.0.0.0 --port 8080 -c 131072 -ngl 999 --cache-type-k q4_0 --cache-type-v q4_0 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 -fa on --cache-reuse 2048 --jinja -a qwen3.8-27b --metrics --spec-type draft-mtp'
# rsLoRA 补偿: 训练用 alpha/sqrt(r)=64/sqrt(32)=11.31; llama.cpp 用 alpha/r=2 → scale=√32
SCALE=5.657

cleanup() { docker rm -f gate-ctl gate-lora >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

# 实例1: 控制 (GPU5, 8090)
docker run -d --name gate-ctl --gpus '"device=5"' -p 8090:8080 \
  -v /data/models:/models $IMG $SRV_ARGS > /dev/null
# 实例2: +LoRA 热挂 (GPU4, 8091) — host 网络下端口冲突, lora 实例用 8091
docker run -d --name gate-lora --gpus '"device=4"' -p 8091:8080 \
  -v /data/models:/models -v /data/compose:/adapter-ro $IMG \
  $SRV_ARGS --lora-scaled /adapter-ro/qwen27b/rft/$(basename $ADAPTER_GGUF):$SCALE > /dev/null

echo "waiting for servers..."
for i in $(seq 1 120); do
  ok1=$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8090/health 2>/dev/null || true)
  ok2=$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8091/health 2>/dev/null || true)
  [ "$ok1" = "200" ] && [ "$ok2" = "200" ] && break
  sleep 10
done
[ "$ok1" = "200" ] && [ "$ok2" = "200" ] || { echo "server health timeout: ctl=$ok1 lora=$ok2"; docker logs --tail 20 gate-lora; exit 1; }

# 快速冒烟: 两实例各答一题, 确认 lora 实例输出正常
$PY - <<'PYEOF'
import os, sys
sys.path.insert(0, '/data/compose/qwen27b/eval')
os.environ['EVAL_BASE'] = 'http://127.0.0.1:8091/v1'
from run_baseline import chat
r = chat([{"role":"user","content":"1+1=? Answer with just the number."}], max_tokens=256)
print("lora-instance smoke:", repr(r[:80]))
PYEOF

cd $EVAL_DIR
echo "== 控制组 (纯底座, 8090) =="
EVAL_BASE=http://127.0.0.1:8090/v1 MTP_METRICS_URL=http://127.0.0.1:8090/metrics NEEDLE_DEPTHS=64 \
  $PY run_baseline.py all --tag gate-control --workers 6
echo "== 实验组 (+LoRA, 8091) =="
EVAL_BASE=http://127.0.0.1:8091/v1 MTP_METRICS_URL=http://127.0.0.1:8091/metrics NEEDLE_DEPTHS=64 \
  $PY run_baseline.py all --tag gate-lora-r1 --workers 6

echo
echo "== 判定 (vs baseline-3.0 + backfill) =="
$PY compare_gate.py /data/eval-rulers/baseline-3.0.json /data/eval-rulers/baseline-3.0-backfill.json /data/eval-rulers/gate-lora-r1.json | tee /data/eval-rulers/gate-verdict-r1.md
echo "(对照控制组 gate-control.json 可查机器漂移)"
