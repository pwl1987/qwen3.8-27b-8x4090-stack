#!/usr/bin/env bash
# DFlash2 soak 巡检 (cron */5): LB 健康 + 副本 spec 计数器 + VRAM, 旁路留痕不依赖 mon 容器
# 2026-09-05 缩容双卡: 只探测 8081/8082 + GPU0,1
LOG=/data/eval-rulers/soak-dflash-20260904.log
line="$(date '+%F %T')"
lb=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/health)
line="$line lb=$lb"
for p in 8081 8082; do
  m=$(curl -s -m 3 http://127.0.0.1:$p/metrics 2>/dev/null | awk '
    /^llamacpp:spec_decode_num_draft_tokens_total /{d=$2}
    /^llamacxx:spec_decode_num_accepted_tokens_total/{a=$2}
    /^llamacpp:spec_decode_num_accepted_tokens_total /{a=$2}
    END{printf "%.0f/%.0f", d+0, a+0}')
  line="$line p$p=${m:-DOWN}"
done
line="$line vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0,1 | tr '\n' ',' | sed 's/ MiB//g')"
# 副本异常检测: 任一 DOWN 或 lb!=200 时 docker ps 快照辅助定位
if [[ "$line" == *DOWN* || "$lb" != 200 ]]; then
  line="$line | $(docker ps --filter name=qwen27b --format '{{.Names}}:{{.Status}}' | tr '\n' ' ')"
fi
echo "$line" >> "$LOG"
