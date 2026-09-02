#!/bin/bash
# GPU 功耗墙切换（降噪用，需 root 运行: sudo bash gpu-power.sh day）
#   day:   GPU0-3 功耗上限 250W（推理速度约降 10-20%，温度大降，风扇随温控曲线自动下降）
#   quiet: 200W 更安静档（速度约降 20-30%）
#   night: 恢复 450W 全速
#   status: 查看当前功耗墙
# 说明: 4090 消费卡不支持软件直接锁风扇转速(NVML 限制)，但风扇是温控曲线——
#       功耗墙压低 → 温度降 → 风扇自动降速，这是唯一可靠的软件降噪手段。
# 注意: 功耗墙重启后恢复默认 450W；GPU4(zxb 业务)未动，如需一并限制自行加 -i 4。
# 自动化(配到 root: sudo crontab -e):
#   0  8 * * * /data/compose/qwen27b/gpu-power.sh day
#   0 18 * * * /data/compose/qwen27b/gpu-power.sh night
set -u
MODE="${1:-}"
GPUS="0 1 2 3"
case "$MODE" in
  day)   W=250 ;;
  quiet) W=200 ;;
  night) W=450 ;;
  status)
    nvidia-smi --query-gpu=index,power.limit,power.draw,temperature.gpu,fan.speed --format=csv,noheader
    exit 0 ;;
  *) echo "用法: sudo bash $0 day|quiet|night|status"; exit 1 ;;
esac
for i in $GPUS; do
  nvidia-smi -pl "$W" -i "$i" || echo "GPU$i 设置失败"
done
echo "$(date '+%F %T') 已设置 GPU0-3 功耗墙 ${W}W ($MODE)"
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | head -4
