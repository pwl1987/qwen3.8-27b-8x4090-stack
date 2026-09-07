#!/bin/bash
# P0 差距分解试验机：test_cfg <label> <env-file>
# 用法: test_cfg A1 /tmp/p0/a1.env  → 写 repo/.env → zx-p0(GPU2) recreate → 健康 →
#       步速分解(bench_step) + t3 夹具中位 + tok/step → /tmp/p0/results.tsv
set -u
REPO=/data/sandbox/ab-vllm/repo
OUT=/tmp/p0/results.tsv
PORT=19627

headline() {  # ulmus t3 夹具 decode 中位
  python3 $REPO/bench/ulmus_validate.py --benchmark --profile t3 --prefill-target 4096 \
    --api http://127.0.0.1:$PORT/v1 2>/dev/null | python3 -c "
import json,sys
try:
    txt=sys.stdin.read(); d=json.loads(txt[txt.index('{'):])
    print(d['benchmark']['decode_tok_s_median'])
except Exception: print('ERR')"
}

snap() {  # metrics 三计数器
  curl -s http://127.0.0.1:$PORT/metrics | awk '
    /vllm:spec_decode_num_drafts_total/ {d=$2}
    /vllm:spec_decode_num_accepted_tokens_total/ && !/per_pos/ {a=$2}
    /vllm:spec_decode_num_draft_tokens_total/ {t=$2}
    END{print d, a, t}'
}

test_cfg() {
  local label="$1" envf="$2"
  cp "$envf" $REPO/.env
  docker compose -p zx-p0 -f $REPO/compose.yaml -f /tmp/p0/compose.p0.yaml up -d --force-recreate >/dev/null 2>&1
  local ok=""
  for i in $(seq 1 50); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/v1/models 2>/dev/null)" = "200" ] && { ok=1; break; }
    sleep 12
  done
  if [ -z "$ok" ]; then
    echo -e "$label\tBOOT-FAIL\t$(docker logs --tail 5 zx-p0-qwen-1 2>&1 | tr '\n' ' ' | cut -c1-200)" | tee -a $OUT; return
  fi
  echo "== $label 步速分解 (3 次隔离 g512) =="
  python3 /tmp/p0/bench_step.py $PORT | tee /tmp/p0/${label}.steps
  read d0 a0 t0 <<< "$(snap)"
  local tps=$(headline)
  read d1 a1 t1 <<< "$(snap)"
  if [ "$tps" = "ERR" ] || [ -z "$tps" ]; then
    echo -e "$label\tBENCH-FAIL" | tee -a $OUT; return
  fi
  local tstep acc
  tstep=$(python3 -c "print(round(($a1-$a0)/max(1e-9,($d1-$d0))+1,2))")
  acc=$(python3 -c "print(round(100*($a1-$a0)/max(1,($t1-$t0)),1))")
  echo -e "$label\t${tps}\t${tstep} tok/step\t${acc}%/tok" | tee -a $OUT
}
