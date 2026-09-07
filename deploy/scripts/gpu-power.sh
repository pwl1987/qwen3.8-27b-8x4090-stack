#!/bin/bash
# GPU 功耗墙切换（降噪用，需 root 运行: sudo bash gpu-power.sh day|quiet|night|status）
#   2026-09-06 v2: 全部 8 卡覆盖 + 逐卡独立配置（gpu-power.conf, lytv 可编辑）
#   day:   工作日白天档（读 conf day 列）; 周末/节假日自动跳过（holidays.txt/workdays.txt）
#   quiet: 手动更静档（conf quiet 列, 不做日历判断）
#   night: 全卡恢复满血（conf night 列, 幂等——已是目标值自动跳过）
#   status: 当前功耗墙/温度/风扇 + 配置对照表
# 原理: GeForce 风扇不可软件直控(NVML 限制)，风扇是温控曲线——限功→降温→风扇自动降。
# 自动化(已配 root crontab, 每日触发, 日历判断在脚本内):
#   0  8 * * * /data/compose/qwen27b/gpu-power.sh day
#   0 18 * * * /data/compose/qwen27b/gpu-power.sh night
# 日志: /data/eval-rulers/gpu-power.log
set -u
MODE="${1:-}"
DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$DIR/gpu-power.conf"
LOG=/data/eval-rulers/gpu-power.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

case "$MODE" in
  day|quiet|night) ;;
  status)
    printf '%-4s %-10s %-8s %-8s %-6s %-6s %s\n' GPU 当前W day quiet night 温度C 风扇%
    while read -r idx wd wq wn _; do
      [[ "$idx" =~ ^#.*$ || -z "${idx:-}" ]] && continue
      cur=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i "$idx" 2>/dev/null)
      tf=$(nvidia-smi --query-gpu=temperature.gpu,fan.speed --format=csv,noheader -i "$idx" 2>/dev/null | tr ',' ' ')
      printf 'GPU%-2s %-10s %-8s %-8s %-6s %s\n' "$idx" "${cur:-?}" "$wd" "$wq" "$wn" "$tf"
    done < "$CONF"
    exit 0 ;;
  *) echo "用法: sudo bash $0 day|quiet|night|status"; exit 1 ;;
esac

# day 模式日历自检: 周末/节假日跳过 (quiet=手动档, night=恢复档, 均不做日历判断)
if [[ "$MODE" == "day" ]]; then
  TODAY=$(date +%F); DOW=$(date +%u)   # %u: 6=周六 7=周日
  if [[ "$DOW" -ge 6 ]]; then
    SKIP=1
    if [[ -f "$DIR/workdays.txt" ]] && grep -q "^${TODAY}$" "$DIR/workdays.txt"; then
      SKIP=0
    fi
    if [[ "$SKIP" == 1 ]]; then
      log "周末($TODAY)不限功, 跳过 day 模式"; exit 0
    fi
  fi
  if [[ -f "$DIR/holidays.txt" ]] && grep -q "^${TODAY}$" "$DIR/holidays.txt"; then
    log "节假日($TODAY)不限功, 跳过 day 模式"; exit 0
  fi
fi

[[ -f "$CONF" ]] || { log "错误: 配置 $CONF 不存在, 放弃($MODE)"; exit 1; }

# 逐卡应用: 读当前值→相同则跳过(幂等)→不同则 -pl 设置
APPLIED=0; SAME=0; FAILED=0
while read -r idx wd wq wn _; do
  [[ "$idx" =~ ^#.*$ || -z "${idx:-}" ]] && continue
  case "$MODE" in day) W=$wd;; quiet) W=$wq;; night) W=$wn;; esac
  if ! [[ "$W" =~ ^[0-9]+$ ]]; then log "GPU$idx 配置值非法('$W'), 跳过"; FAILED=$((FAILED+1)); continue; fi
  cur=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i "$idx" 2>/dev/null)
  if [[ -z "$cur" ]]; then log "GPU$idx 读取失败(不存在?)"; FAILED=$((FAILED+1)); continue; fi
  cur=${cur%.*}   # nounits 仍带小数(450.00), 取整比较
  if [[ "$cur" == "$W" ]]; then SAME=$((SAME+1)); continue; fi
  if nvidia-smi -pl "$W" -i "$idx" >/dev/null 2>&1; then
    log "GPU$idx 功耗墙 ${cur}→${W}W ($MODE)"; APPLIED=$((APPLIED+1))
  else
    log "GPU$idx 设置失败(目标 ${W}W)"; FAILED=$((FAILED+1))
  fi
done < "$CONF"
log "总结($MODE): 应用=$APPLIED 已是目标=$SAME 失败=$FAILED"
