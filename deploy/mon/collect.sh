#!/bin/sh
# GPU + 副本状态采集: 每 2s 写 /www/data.json (busybox httpd 供前端轮询)
# GPU 数据: nvidia-smi 原生 JSON; 副本数据: 各副本 /slots 的 busy 计数
mkdir -p /www
while true; do
  GPUS=$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=json,noheader 2>/dev/null)
  [ -z "$GPUS" ] && GPUS='{"query_gpu":[]}'
  SLOTS=""
  for r in llama r1 r2 r3; do
    BUSY=$(wget -qO- -T 2 "http://$r:8080/slots" 2>/dev/null | grep -o '"is_processing":true' | wc -l)
    TOTAL=$(wget -qO- -T 2 "http://$r:8080/slots" 2>/dev/null | grep -o '"id"' | wc -l)
    [ "$TOTAL" = "0" ] && BUSY=-1
    SLOTS="$SLOTS\"$r\":{\"busy\":$BUSY,\"total\":$TOTAL},"
  done
  printf '{"ts":%s,"gpus":%s,"slots":{%s}}' "$(date +%s)" "$GPUS" "${SLOTS%,}" > /www/data.json.tmp 2>/dev/null
  mv /www/data.json.tmp /www/data.json
  sleep 2
done
