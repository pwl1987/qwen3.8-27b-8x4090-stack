# training/ — 后训练（与推理分离）

```
train/     QLoRA 训练：run_train.sh（stage_a 短轨/断点续训/自动巡检）、convert_data.py（三轨分离
           + MinHash 去重）、ds_zero*.json、monitor.sh（cron 死亡自动续）、smoke_test.py、
           TRAIN-NOTES.md（★ 689 行执行记录：训练六坑 + 驱动升级 + vLLM 全程日志）
llamacpp/  LoRA→GGUF 转换补丁两文件：convert_lora_to_gguf.py（GDN out_proj 列重排死点）+
           qwen.py（_reorder_v_heads 置换路径）
```

要点：4×4090 ZeRO-3 + NF4 double-quant，显存账稳态 ~21.6G/卡；产物 checkpoint-4840 →
final-lora.gguf（f32 934M）/ final-lora-q8.gguf（248M），992 张量。
热挂须 rsLoRA 补偿 `--lora-scaled <adapter>:5.657`（√32）。
六坑全录 TRAIN-NOTES + PROBLEMS-AND-FIXES P6-P13。
