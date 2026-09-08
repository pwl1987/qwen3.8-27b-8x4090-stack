# B0 结案：DFlash2 drafter 纯 torch 数值复刻——对齐门通过（2026-09-08）

**结论：B1 蒸馏训练器的前向地基成立。** bf16 双侧（引擎 bf16 drafter 沙箱 × 纯 torch 复刻）
在 260 步 / 1813 个 draft 位置上：

| 级别 | 内容 | 结果 | 门 | 判定 |
|---|---|---|---|---|
| A | fc：aux(5×5120 拼接) @ fc.weightᵀ vs 引擎 fc_out | 余弦 **2065/2065 = 1.0**；abs 差 p50=0（均值 0.73=bf16 ulp 级，大分量上 1-2 ulp） | — | ✅ |
| C | sample_hidden @ bf16 lm_headᵀ → top-16 vs 引擎候选 | **mean 15.83 / p50 16 / p10 15 / min 15**（n=1813 位置） | mean≥14/16 且 p10≥12/16 | ✅ |
| D-scores | 码本+hidden_projection 重算 selector scores | **max-abs-diff = 0.0（逐位一致，n=259 步）** | — | ✅ |
| D-path | 走链复刻 | greedy 链 **98.8%** 全 7 token 一致（p50/p10=1.0）；DP 链 13.9%（排除） | ≥95% | ✅ |
| B | fc_out→5 层 conv/attn→sample_hidden 逐层复刻 | **留给 B1**（本批 trace 已同时含 fc_out 与 sample_hidden 两侧张量，可离线续验） | — | ⏭ |

三个附带发现：
1. **bf16 引擎 eager 模式完全确定**：t3 夹具 3 跑 sha 逐字节一致（b6117c312d39），接受率
   2.56 tok/step（84 步/跑）。
2. **走链语义判明**：引擎 `_selector_walk_kernel` = 逐步贪心（给定前驱行取 argmax），非全局
   DP。3/259 步路径失配发生在 scores 完全一致（0 差）的步上 → 纯平票裁定顺序差（torch argmax
   取首指标 vs Triton 走链序），B1 训练器采样时按引擎裁定序对齐即可。
3. **top-16 级含目标 lm_head 量化噪声**：引擎侧 lm_head 是 W4A16(Marlin)，torch 侧用 bf16
   （量化前备份分片 `model-00007-of-00007.safetensors.bak`）；min=15/16 说明该噪声对候选集
   影响有限——这正是门设在 14/16 而非 16/16 的原因，实测余量充足。

## 复现链（约 40 分钟）

```bash
# 1. bf16-drafter 沙箱（GPU2:19627, zx-p0 overlay；trace 必须 --enforce-eager：
#    FULL cudagraph 下 _generate_draft 被捕获，Python dump 不在重放期执行）
cp eval/vllm/b0/b0.env /data/sandbox/ab-vllm/repo/.env
cd /data/sandbox/ab-vllm/repo && docker compose -p zx-p0 -f compose.yaml \
  -f /tmp/p0/compose.p0.yaml up -d --force-recreate   # 等健康(~2-8min)
# 2. 部署 ask 插桩（patch 见 speculator-ask-trace.patch，含 dflash_trace.on 恒开钩）
P=/app/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/dflash2
docker cp /tmp/trace_prep/speculator.py zx-p0-qwen-1:$P/speculator.py
docker cp /tmp/trace_prep/states.py      zx-p0-qwen-1:$P/states.py
docker exec zx-p0-qwen-1 touch /tmp/dflash_trace.on /tmp/dflash_ask.on
docker restart zx-p0-qwen-1              # 保 fs，重启即生效
# 3. 采集（t3 夹具 ×3 ≈ 260 步 in/out 对；每 40 步自动 flush /tmp/dflash_ask-NNNN.pt）
/data/vllm/venv/bin/python /tmp/p0/t3_probe.py 19627 3
# 4. 拷出 + GPU4 纯 torch replay（权重全 bf16：drafter safetensors + lm_head 备份分片）
mkdir -p /tmp/b0/ask && for f in $(docker exec zx-p0-qwen-1 ls /tmp/dflash_ask-\\*.pt); \
  do docker cp zx-p0-qwen-1:$f /tmp/b0/ask/; done
CUDA_VISIBLE_DEVICES=4 /data/vllm/venv/bin/python eval/vllm/b0/replay.py \
  --ask-dir /tmp/b0/ask --report replay-report.json
```

## 文件

- `b0.env` — 沙箱臂（认证 130 配置骨架；DRAFT→bf16 目录、--enforce-eager、MAX_LEN 32K/KV 2G
  ——bf16 权重比 W4-recal 大 2.6G，245K+KV4.86G 会 OOM；trace 采集不需要大 ctx）
- `replay.py` — 纯 torch 逐级对齐（A/C/D 级 + greedy/DP 双走链 + 分位数报告）
- `speculator-ask-trace.patch` — ask 级插桩（在 dflash2 backport 镜像源码之上，恒开 dflash_trace
  钩 + ask 钩：propose 入口录 aux/input_ids，_generate_draft 出口录 fc_out/sample_hidden/
  candidates/unary/scores/selector_tokens/anchor；600 步上限、40 步 flush）
- `replay-report.json` — 本批全量分位数

## B1 衔接

- 剩余一级：5 层 conv/attn 的 torch 复刻（grouped conv 数学照抄补丁源、滑窗注意力 window-mask
  SDPA、context-KV 预计算 rms_norm(fc_out)@W_kv——ROADMAP 方向 B 既定设计），完成后即可端到端
  可微，fc+selector 先训、深层冻结。
- 训练器验收沿用本 replay：top-16 overlap 与走链一致率作为每 epoch 的数值回归门。
- 本批 t3 上 bf16 草稿 2.56 tok/step（recal 曾 3.04）——重训目标 ≥3.5 的空间在分布适配而非精度。
