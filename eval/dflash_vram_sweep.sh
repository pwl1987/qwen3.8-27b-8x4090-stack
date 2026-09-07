#!/usr/bin/env bash
# dflash 256K OOM 削减扫描: 按损失最小优先, 首个全测试通过且余量>=500MB 的配置获胜
set -uo pipefail
GPU=6; PORT=8085; CT=qwen27b-sbx
IMAGE=llama-server:cuda12.4-b10715
MODEL=/models/Qwen3.8-27B-coding-v1-iq4xs.gguf
DFLASH=/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
OUT=/data/eval-rulers/spec-sandbox-20260904
DFL_ARGS="--spec-type draft-dflash -md $DFLASH --spec-draft-n-max 7"

try_cfg() { # $1=label, $2=ctx, 其余 args
  local label=$1 ctx=$2; shift 2
  docker rm -f $CT >/dev/null 2>&1
  echo "=== [$label] ctx=$ctx extra: $*"
  docker run -d --name $CT --gpus "device=$GPU" -v /data/models:/models \
    -p 127.0.0.1:$PORT:8080 "$IMAGE" \
    -m $MODEL --host 0.0.0.0 --port 8080 -c $ctx -ngl 999 -np 2 \
    --cache-type-k q4_0 --cache-type-v q4_0 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
    -fa on --cache-reuse 2048 --jinja -a sbx --metrics $DFL_ARGS "$@" >/dev/null || { echo "BOOT FAIL"; return 2; }
  for i in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = 200 ] && break
    sleep 2
  done
  [ "$(docker inspect -f '{{.State.Running}}' $CT 2>/dev/null)" = true ] || { echo "DIED AT BOOT"; return 2; }
  python3 /data/compose/qwen27b/eval/spec_bench.py "$label" $PORT "$OUT/bench-$label.json" >/dev/null 2>&1
  if [ "$(docker inspect -f '{{.State.Running}}' $CT 2>/dev/null)" != true ]; then
    echo "RESULT [$label]: CRASHED during bench"; docker logs $CT 2>&1 | grep -m2 -iE 'CUDA error|out of memory'; return 1
  fi
  local used=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits)
  local head=$((24564 - used))
  python3 - "$OUT/bench-$label.json" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
s=sum(x["gen_tps"] for x in d["short"])/len(d["short"])
print(f"RESULT[{d['label']}]: short_avg={s:.1f} longgen={d['longgen']['gen_tps']:.1f} prefill={d['prefill']['prefill_tps']:.0f} acc={d['spec_metrics']['delta'].get('llamacpp:spec_decode_num_accepted_tokens_total',0)/max(d['spec_metrics']['delta'].get('llamacpp:spec_decode_num_drafts_total',1),1):.2f}/step")
EOF
  echo "VRAM: ${used} MiB (headroom ${head} MiB)"
  [ $head -ge 500 ] && { echo ">>> WINNER: $label"; return 0; }
  return 1
}

try_cfg dflash-256k-np1 262144 -np 1 && exit 0
try_cfg dflash-256k-ub256 262144 -ub 256 && exit 0
try_cfg dflash-240k 245760 && exit 0
try_cfg dflash-224k 229376 && exit 0
try_cfg dflash-208k 212992 && exit 0
echo ">>> NO CONFIG PASSED"
