#!/bin/bash
# 噪音模式切换: 白天(8-18点)少卡低功耗, 晚上全速
#   day:   ① LB 软摘除 r2/r3(写 DISABLED 文件, 新请求立即漂到 r0/r1, 在途请求自然完成)
#          ② drain 45 秒等在途请求结束 → ③ docker stop(GPU2/3 断负载, 风扇降到底)
#   night: ① start r2/r3(加载 ~50s) → ② 等健康 → ③ 删 DISABLED 文件恢复接流
# 挂 crontab(<USER> 用户级, 无需 sudo):
#   0  8 * * * /data/compose/qwen27b/noise-mode.sh day
#   0 18 * * * /data/compose/qwen27b/noise-mode.sh night
set -u
DIR=/data/compose/qwen27b
MODE="${1:-}"
cd "$DIR"
case "$MODE" in
  day)
    printf 'r2:8080\nr3:8080\n' > lb/DISABLED
    echo "$(date '+%F %T') 软摘除 r2/r3, drain 45s..."
    sleep 45
    sg docker -c 'docker stop qwen27b-r2 qwen27b-r3'
    echo "$(date '+%F %T') 白天模式完成: 仅 GPU0/1 推理"
    ;;
  night)
    sg docker -c 'docker start qwen27b-r2 qwen27b-r3'
    for i in $(seq 1 20); do
      sleep 5
      ok=0
      for c in qwen27b-r2 qwen27b-r3; do
        sg docker -c "docker logs $c --since 2m 2>&1 | grep -q 'listening'" && ok=$((ok+1))
      done
      [ "$ok" = "2" ] && break
    done
    rm -f lb/DISABLED
    echo "$(date '+%F %T') 夜间模式完成: 4 副本全速(等待 ok=$ok/2)"
    ;;
  status)
    [ -f lb/DISABLED ] && echo "LB 软摘除: $(tr '\n' ' ' < lb/DISABLED)" || echo "LB 无软摘除"
    sg docker -c 'docker ps --filter name=qwen27b --format "{{.Names}} {{.Status}}"'
    nvidia-smi --query-gpu=index,utilization.gpu,power.draw,temperature.gpu,fan.speed --format=csv,noheader
    ;;
  *)
    echo "用法: $0 day|night|status"; exit 1
    ;;
esac
