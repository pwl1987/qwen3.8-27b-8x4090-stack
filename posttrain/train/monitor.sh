#!/usr/bin/env bash
# QLoRA stage_a 健康监控 — 打印单行状态; 进程死则给出最近错误
set -uo pipefail
LOG=/data/compose/qwen27b/train/stage_a.log
OUT=/data/compose/qwen27b/train/out/qlora-r1
PYBIN=/home/<USER>/miniforge3/envs/torch/bin

alive=$(ps aux | grep -E "sft-short|torch.distributed.run" | grep -v grep | wc -l)
# 最近一条 step 日志
last=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" 2>/dev/null | grep -oE "\{'loss':[^}]*\}" | tail -1)
step=$(echo "$last" | grep -oE "global_step/max_steps': '[0-9]+/[0-9]+'" | grep -oE "[0-9]+/[0-9]+" | head -1)
loss=$(echo "$last" | grep -oE "'loss': '[0-9.]+'" | grep -oE "[0-9.]+")
mem=$(echo "$last" | grep -oE "memory\(GiB\)': '[0-9.]+'" | grep -oE "[0-9.]+")
acc=$(echo "$last" | grep -oE "token_acc': '[0-9.]+'" | grep -oE "[0-9.]+")
# GPU 峰值
peak=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 4,5,6,7 2>/dev/null | awk -F', ' '{gsub(" MiB","",$2); if($2>m)m=$2} END{print m}')
# 最近 checkpoint (兼容 v0-x/checkpoint-N 与 x/v0-y/checkpoint-N 两种层级)
ckpt=$(ls -1d "$OUT"/*/checkpoint-* "$OUT"/*/*/checkpoint-* 2>/dev/null | sort -V | tail -1)
ckptstep=$( [ -n "$ckpt" ] && basename "$ckpt" | grep -oE "[0-9]+" | head -1 || echo 0 )

if [ "$alive" -gt 5 ]; then
  echo "STATUS=ALIVE procs=$alive step=${step:-?} loss=${loss:-?} acc=${acc:-?} mem=${mem:-?}GB gpu_peak=${peak}MiB last_ckpt=${ckptstep}"
else
  echo "STATUS=DEAD procs=$alive last_ckpt=${ckptstep}"
  echo "--- last error ---"
  sed 's/\x1b\[[0-9;]*m//g' "$LOG" 2>/dev/null | grep -iE "OutOfMemory|Traceback|Error:|ChildFailed|CUDA error|Killed" | grep -v "it/s" | tail -6
fi
