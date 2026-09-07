# vllm/ — coding-v1.1 单卡 240K 投机解码生产线（DFlash2 + KVarN）

基于 [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)（Apache-2.0，见本目录 LICENSE，
上游原版说明保留在 `README.upstream.md`）的 vLLM 0.27.1 单用户栈改造：28 补丁 + KVarN 4/2-bit KV +
DFlash2 块式投机解码 + V2 runner。在本仓的定位：**生产 llama.cpp 双副本（90 tok/s）之外的单卡高速通道**——
后训模型 Qwen3.8-27B-coding-v1.1 W4A16（AutoRound，lm_head/embed int8）跑 **decode 130 tok/s @240K**，
1.45× 生产副本、省 1 卡；prefill ~2550 tok/s @32K。

## 相对上游的改造

| 项 | 内容 |
|---|---|
| `compose.yaml` | MODEL 可覆盖（A/B 用）、挂载 `/data/models:ro`、tool-call-parser=qwen3_coder + reasoning-parser=qwen3 |
| `drafter/` | DFlash2 侧车 drafter 的 **GPTQ W4A16 重校准**（自分布 Hessian，+6.9% 接受率）：`capture_dflash2.py`（引擎内挂钩采集）→ `quant_dflash2.py`；`head_capture.py` 为 lm_head 校准的解码态隐状态采集 |
| `compose.int8ab.yaml` | 双引擎 A/B 模式样板（`!override` 换卡换端口，独立项目名 `-p`） |
| `.env.example` | **终配模板**（已认证） |
| `results/` | 实验 JSON |

## 终配（.env.example 要点）

```
MODEL=coding-v1.1-W4A16  SPEC=dflash2  DFLASH_TOKENS=7  CTX=huge
MAX_LEN=245760  DFLASH_MAX_LEN=245760
DRAFT=Qwen3.8-27B-DFlash2-W4A16-recal   # 重校准 drafter
LOOKUP=0                                 # lookup 起草对本模型微负
KV_MEM=4860000000  VLLM_V2_CUDAGRAPH_MEM_MIB=1000   # 240K 显存配平（省 0.8GB 过 FLA 碎片缘）
PREFIX_CACHE=1  VISION=1  VISION_OFFLOAD=1  GPU_UTIL=0.95
```

## 部署

```bash
docker build -t nvidia-4090-llm-inference:0.27.1-cu129 -f docker/Dockerfile.cu129 .   # 补丁构建时自动应用
# 模型准备（权重不入库）：目标模型 W4A16 + lm_head/embed int8（prepare/quant_lm_head.py + quant_embed.py），
# drafter 重校准见 drafter/README.md；prepare/ 首启幂等自检（verify.sh 为 FAIL 级门禁）
cp .env.example .env && docker compose up -d
```

## 已知的坑（都踩过，勿重复）

1. **262K 显存墙**：int8 头比上游 W4 头重 ~1GB，262,144 档在首请求/长 prefill 死于 42MB 级动态分配（FLA 内核碎片缘）；240K 靠 KV 5.26→4.86GB + CG 1400→1000MiB 配平。
2. **int4 lm_head 质量死刑**：4 轮校准（预填/解码 × g64/g128，cos 最高 0.9992）均留确定性退化——后训 lm_head 对 int4 敏感，速度上又无收益（步速本已持平），**不要再用**。
3. **adaptive×前缀缓存损坏**（上游栈已知 bug，已复测 4/12 残差中招）：`LOOKUP=1`+adaptive 长块+前缀缓存命中的第二轮输出会确定性错乱；`ADAPTIVE=0` 干净但增益归零。检测工具：`../../eval/vllm/p0/multi_residue_test.py`。
4. **GPU_UTIL 敏感性**：0.93（栈默认）与 0.95 的内存布局差异足以开关 lookup/adaptive 通道的可用性——A/B 时必须钉死。
5. **250W 功耗墙仅 -1.5% decode**（显存带宽型负载），无需为性能放宽日间限功。
6. `/metrics` 计数器带 `vllm:` 前缀——解析时锚定错了会得到全 NA 的 tok/step（本文档的很多"结论"曾被它坑过一轮）。

## 测量口径

- decode 夹具：`bench/ulmus_validate.py --benchmark --profile t3 --prefill-target 4096`（p565/g512，流式首末 token 计时，3 次中位）
- 步速/接受率分解：`../../eval/vllm/p0/bench_step.py <port>`（单请求隔离 + /metrics 差分 + 按位置接受剖面）
- 完整实验日志与技术结论：`../../docs/VLLM-OPTIMIZATION.md`
