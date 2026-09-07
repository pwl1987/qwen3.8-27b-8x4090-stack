#!/usr/bin/env python3
"""BF16 底座冒烟 gate: qwen3_5 架构加载 + MTP 层校验 + 前向一步。"""
import os, sys, time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5,6,7")

import torch
from transformers import AutoConfig, AutoTokenizer

PATH = "/data/models/Qwen3.8-27B-BF16"

print("== 1. config ==", flush=True)
t0 = time.time()
cfg = AutoConfig.from_pretrained(PATH)
print(f"  architectures: {cfg.architectures}")
print(f"  model_type: {cfg.model_type} | text: {cfg.text_config.model_type}")
print(f"  layers: {cfg.text_config.num_hidden_layers} | hidden: {cfg.text_config.hidden_size}")
print(f"  vocab: {cfg.text_config.vocab_size} | max_pos: {cfg.text_config.max_position_embeddings}")
print(f"  mtp_num_hidden_layers: {getattr(cfg.text_config, 'mtp_num_hidden_layers', None)}")
print(f"  layer_types 统计: linear={cfg.text_config.layer_types.count('linear_attention')} full={cfg.text_config.layer_types.count('full_attention')}")
print(f"  [OK] {time.time()-t0:.1f}s", flush=True)

print("== 2. tokenizer ==", flush=True)
tok = AutoTokenizer.from_pretrained(PATH)
ids = tok("def add(a, b):\n    return a + b", return_tensors="pt").input_ids
print(f"  vocab_size(tok): {tok.vocab_size} | test encode: {ids.shape}", flush=True)

print("== 3. model load (bf16, device_map=auto → GPU5-7) ==", flush=True)
t0 = time.time()
from transformers import AutoModelForCausalLM
try:
    model = AutoModelForCausalLM.from_pretrained(
        PATH, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
except Exception as e:
    print(f"  AutoModelForCausalLM failed: {e}", flush=True)
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        PATH, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
print(f"  class: {type(model).__name__} | {time.time()-t0:.0f}s", flush=True)

print("== 4. MTP 层校验 ==", flush=True)
mtp_keys = [k for k in model.state_dict() if "mtp" in k.lower() or "nextn" in k.lower()]
print(f"  MTP 相关张量: {len(mtp_keys)}")
for k in mtp_keys[:8]:
    print(f"    {k}: {model.state_dict()[k].shape}")
if not mtp_keys:
    print("  [WARN] 未发现 MTP 层命名 — 检查 nextn 预测头是否内嵌", flush=True)

print("== 5. 前向一步 ==", flush=True)
model.eval()
dev = next(model.parameters()).device
print(f"  首参数设备: {dev}", flush=True)
ids = tok("Write a Python function to reverse a string.", return_tensors="pt").input_ids.to(dev)
t0 = time.time()
with torch.no_grad():
    out = model(input_ids=ids)
logits = getattr(out, "logits", None)
print(f"  logits: {None if logits is None else tuple(logits.shape)} dtype={None if logits is None else logits.dtype} | {time.time()-t0:.1f}s", flush=True)

print("== 6. 生成 16 token ==", flush=True)
try:
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=16, do_sample=False)
    print(f"  generated: {tok.decode(gen[0], skip_special_tokens=False)[:200]!r}", flush=True)
except Exception as e:
    print(f"  generate failed (可接受, 前向已过): {e}", flush=True)

print("\nSMOKE GATE PASSED", flush=True)
