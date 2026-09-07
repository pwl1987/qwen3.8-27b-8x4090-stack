#!/bin/bash
# 安全停服务：按监听端口取 PID（避免 pkill -f 匹配自家命令自杀）
PORT=${1:?用法: stop.sh <端口>}
for p in $(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | sort -u); do
  kill $p && echo "killed $p"
done
sleep 3
ss -tln 2>/dev/null | grep -q ":$PORT " && echo "警告: $PORT 仍被占用" || echo "$PORT 已释放"
