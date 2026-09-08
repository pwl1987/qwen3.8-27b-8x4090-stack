#!/bin/bash
# B1-B 串行四臂 + 每 ckpt replay 门（治理：GPU4 同一时刻仅一个训练进程）
set -e
cd /data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4
PY=/data/vllm/venv/bin/python
CK=/data/sandbox/ab-vllm/b1/ckpts
EVD=blevel-evidence
mkdir -p $CK

for ARM in zero b1-1 b1-2 b1-3; do
  echo "=== ARM $ARM $(date +%H:%M:%S) ==="
  $PY run_arms.py --arm $ARM --save-dir $CK/$ARM \
      > $EVD/arm-$ARM.log 2>&1
  tail -3 $EVD/arm-$ARM.log
  for CPT in $CK/$ARM/ckpt-*.pt; do
    N=$(basename $CPT .pt | sed 's/ckpt-//')
    $PY gate.py --ckpt $CPT --tag $ARM-$N --out $EVD/gate-$ARM-$N.json \
        > /dev/null 2>&1 || echo "gate $ARM-$N FAILED (rc=$?)"
    echo "gate $ARM-$N: $($PY -c "import json;r=json.load(open('$EVD/gate-$ARM-$N.json'));print(r['gates'], r['top16']['mean'], r['recall'], r['mean_margin'])")"
  done
done
echo "=== ALL ARMS DONE $(date +%H:%M:%S) ==="
