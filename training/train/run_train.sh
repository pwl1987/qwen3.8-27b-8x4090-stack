#!/usr/bin/env bash
# Qwen3.8-27B QLoRA 训练启动 — 三阶段 (短轨主 → 长轨 → FIM 轨)
# 显存账/卡/卡 (4bit 底座 14G + DoRA/LoRA 0.6G + 8bit 优化器分片 0.5G + 激活 2-3G ≈ 18G < 24G)
set -uo pipefail
PY=/home/<USER>/miniforge3/envs/torch/bin
SWIFT=$PY/swift
DATA=/data/compose/qwen27b/train/data
OUT=/data/compose/qwen27b/train/out
MODEL=/data/models/Qwen3.8-27B-BF16
GPUS="${GPUS:-4,5,6,7}"
# 4bit 底座 17.9G/卡(bf16); 冒烟实测 21.7G/卡, 峰值 < 23.6G
# 关键: --torch_dtype bfloat16 (swift 默认 float32 会让非量化层 emb/lm_head 翻倍 +5G 直接 OOM)
# PATH 需含 ninja (deepspeed cpu_adam/JIT); NPROC 走 torchrun 使 torch.distributed 预初始化(免 mpi4py)
export PATH=$PY:$PATH
export CUDA_VISIBLE_DEVICES=$GPUS
export NPROC_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_MAX_LOCAL_REG_SIZE=1048576
export NCCL_BUFFSIZE=4194304
DS=/data/compose/qwen27b/train/ds_zero3_nooffload.json

stage_a() {
  local RESUME_ARGS=""
  [ -n "${RESUME_CKPT:-}" ] && RESUME_ARGS="--resume_from_checkpoint $RESUME_CKPT"
  CUDA_VISIBLE_DEVICES=$GPUS $SWIFT sft \
    --model "$MODEL" --model_type qwen3_5 \
    --tuner_type lora \
    --template qwen3_8 \
    --dataset "$DATA/sft-short.jsonl" \
    --torch_dtype bfloat16 \
    --quant_bits 4 --quant_method bnb \
    --lora_rank 32 --lora_alpha 64 \
    --use_dora false --use_rslora true --lorap_lr_ratio 2.0 \
    --target_modules all-linear \
    --optim paged_adamw_8bit \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 2e-5}' \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
    --max_length 1536 \
    --gradient_checkpointing true \
    --max_grad_norm 1.0 --warmup_ratio 0.05 \
    --deepspeed "$DS" \
    --bf16 true --fp16 false \
    --save_steps 500 --save_total_limit 8 --logging_steps 10 \
    --dataloader_num_workers 8 --dataset_num_proc 64 \
    --output_dir "$OUT/qlora-r1" \
    --report_to none --seed 42 \
    --add_version true $RESUME_ARGS
}

stage_b() {
  # 长上下文轨: 8-16K 完整轨迹, 1 epoch
  local ADAPTER=$(ls -dt $OUT/qlora-r1/*/v0-*/checkpoint-* 2>/dev/null | head -1)
  [ -z "$ADAPTER" ] && ADAPTER=$(ls -dt $OUT/qlora-r1/* 2>/dev/null | head -1)
  CUDA_VISIBLE_DEVICES=$GPUS $SWIFT sft \
    --model "$MODEL" --model_type qwen3_5 \
    --tuner_type lora --adapters "$ADAPTER" \
    --template qwen3_8 \
    --dataset "$DATA/sft-long.jsonl" \
    --torch_dtype bfloat16 \
    --quant_bits 4 --quant_method bnb \
    --lora_rank 32 --lora_alpha 64 \
    --use_dora false --use_rslora true --lorap_lr_ratio 2.0 \
    --target_modules all-linear \
    --optim paged_adamw_8bit \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-5}' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --max_length 16384 \
    --gradient_checkpointing true \
    --max_grad_norm 1.0 --warmup_ratio 0.05 \
    --deepspeed "$DS" \
    --bf16 true --fp16 false \
    --save_steps 50 --save_total_limit 4 --logging_steps 10 \
    --dataloader_num_workers 4 --dataset_num_proc 32 \
    --output_dir "$OUT/qlora-r1-long" \
    --report_to none --seed 42 \
    --add_version true
}

stage_c() {
  # FIM 轨: 指令式前缀/后缀补全, 与 chat LoRA 分离训练
  local ADAPTER=$(ls -dt $OUT/qlora-r1-long/*/v0-*/checkpoint-* 2>/dev/null | head -1)
  [ -z "$ADAPTER" ] && ADAPTER=$(ls -dt $OUT/qlora-r1/* 2>/dev/null | head -1)
  CUDA_VISIBLE_DEVICES=$GPUS $SWIFT sft \
    --model "$MODEL" --model_type qwen3_5 \
    --tuner_type lora --adapters "$ADAPTER" \
    --template qwen3_8 \
    --dataset "$DATA/sft-fim.jsonl" \
    --torch_dtype bfloat16 \
    --quant_bits 4 --quant_method bnb \
    --lora_rank 32 --lora_alpha 64 \
    --use_dora false --use_rslora true --lorap_lr_ratio 2.0 \
    --target_modules all-linear \
    --optim paged_adamw_8bit \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-5}' \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
    --max_length 8192 \
    --gradient_checkpointing true \
    --max_grad_norm 1.0 --warmup_ratio 0.05 \
    --deepspeed "$DS" \
    --bf16 true --fp16 false \
    --save_steps 100 --save_total_limit 4 --logging_steps 10 \
    --dataloader_num_workers 8 --dataset_num_proc 32 \
    --output_dir "$OUT/qlora-r1-fim" \
    --report_to none --seed 42 \
    --add_version true
}

case "${1:-all}" in
  a) stage_a ;;
  b) stage_b ;;
  c) stage_c ;;
  all) stage_a && stage_b && stage_c ;;
  *) echo "usage: $0 [a|b|c|all]" ;;
esac
