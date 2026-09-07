#!/usr/bin/env bash
# 驱动 580 升级后的开机自检 (@reboot cron, 延迟 90s 等容器起齐)
# 日志: /data/eval-rulers/driver580-bootcheck.log
LOG=/data/eval-rulers/driver580-bootcheck.log
sleep 90
{
  echo "===== bootcheck $(date '+%F %T') ====="
  echo "kernel: $(uname -r)"
  nvidia-smi | sed -n '3,4p'
  nvidia-smi --query-gpu=index,driver_version,memory.used,utilization.gpu --format=csv,noheader | head -4
  echo '--- containers ---'
  docker ps --format '{{.Names}}: {{.Status}}' | grep -E 'qwen27b|comfyui|zxb' | sort
  echo '--- llama.cpp LB :8000 ---'
  curl -s -o /dev/null -w 'health http=%{http_code}\n' -m 10 http://127.0.0.1:8000/health
  echo '--- comfyui :8188 ---'
  curl -s -m 10 http://127.0.0.1:8188/system_stats | head -c 200; echo
  echo '--- GPU 进程 ---'
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
} >> "$LOG" 2>&1
