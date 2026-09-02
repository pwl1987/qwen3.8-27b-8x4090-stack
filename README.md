# Qwen3.8-27B · 8×RTX 4090 推理与后训练全栈

> qwen3.8-27b（27.3B 参数，qwen35 混合线性注意力架构）在单机 8×RTX 4090 24GB 上的完整工程实践：
> **4 副本 256K 上下文推理 + OpenResty 动态会话粘滞负载均衡**、**QLoRA 后训练**、**LoRA→GGUF 转换与运行时热挂**、**A/B 门禁评测**。
> 所有数字均为本机实测，附完整踩坑记录与显存账。

- [仓库结构](#仓库结构)
- [硬件与模型](#硬件与模型)
- [推理部署（deploy/）](#推理部署deploy)
- [负载均衡策略（deploy/lb/route.lua）](#负载均衡策略deploylbroutelua)
- [后训练（posttrain/）](#后训练posttrain)
- [LoRA→GGUF 转换与热挂](#loragguf-转换与热挂)
- [门禁评测（eval/）](#门禁评测eval)
- [运维脚本](#运维脚本)
- [模型权重来源（不入库）](#模型权重来源不入库)
- [English TL;DR](#english-tldr)
- [License](#license)

---

## 仓库结构

```
├── deploy/                     # 推理服务栈
│   ├── docker-compose.yml      # 4×llama-server 副本 (GPU0-3) + OpenResty LB + 可选监控
│   ├── lb/
│   │   ├── qwen27b.conf        # OpenResty 配置（:8000 唯一入口 + /_lbstats + 监控路由）
│   │   └── route.lua           # 核心：会话粘滞 + 满载漂移 + 大小会话分流 (Lua)
│   ├── mon/                    # GPU/副本实时监控页（python:3.12-slim 容器, :9000）
│   │   ├── collect.py          # 采集器（nvidia-smi + 各副本 /slots）
│   │   ├── collect.sh
│   │   └── index.html          # 前端（2s 轮询, 24h 历史曲线）
│   └── scripts/
│       ├── gpu-power.sh        # GPU 功耗墙切换（软件降噪：450W→250W→200W）
│       └── noise-mode.sh       # 白天软摘除 r2/r3 副本停机降风扇, 晚间恢复
├── posttrain/
│   ├── train/                  # QLoRA 训练（ms-swift + DeepSpeed, 4×4090）
│   │   ├── run_train.sh        # 启动脚本（stage_a 短轨 / 断点续训 / 自动巡检）
│   │   ├── convert_data.py     # 多源数据 → swift messages 格式（thinking 三轨分离）
│   │   ├── ds_zero2.json / ds_zero3_*.json
│   │   ├── monitor.sh          # 单行训练状态 + cron 死亡自动续训
│   │   ├── smoke_test.py
│   │   └── TRAIN-NOTES.md      # ★ 执行记录 + 6 个踩坑（全部实测复现）
│   └── llamacpp/               # llama.cpp b10715 补丁后的两个文件
│       ├── convert_lora_to_gguf.py   # ★ GDN out_proj 列重排补丁（混合注意力 LoRA 转换死点）
│       └── qwen.py                       # ★ _reorder_v_heads LoRA 张量置换路径
├── eval/
│   ├── gate.sh                 # ★ Phase 4 一键门禁：双实例 A/B → 全轴评测 → 判定
│   ├── compare_gate.py         # 判定规则（任一轴回退>2pp FAIL；核心轴严格提升）
│   ├── run_baseline.py         # 评测执行器（humaneval/xfc/gsm8k/ifeval/needle/longgen/tps）
│   ├── rft/
│   │   ├── sandbox_bench.py    # RFT 验证沙箱（Docker 隔离, MBPP sanitized_test）
│   │   └── sbx_*.json          # 50/128/257 路并发压测结果
│   └── rulers/                 # 基线数据落盘（门禁的裁判依据）
│       ├── baseline-3.0.json / baseline-3.0-backfill.json / baseline-3.0-backfill2.json
│       └── xfc-sample.json     # xfc 工具轴 200 题固定卷（toolace 抽取, 已排除训练重叠）
├── model/
│   ├── config.json             # qwen3_5 架构定义（full_attention_interval=4 等）
│   ├── chat_template.jinja
│   └── generation_config.json
└── docs/
    └── QWEN27B-ANALYSIS.md     # ★ 服务深度分析：权重解剖/量化配方/MTP 实测/优化路线/灰度原则
```

## 硬件与模型

| 项 | 值 |
|---|---|
| GPU | 8×RTX 4090 24GB（GPU0-3 推理 / GPU4-7 训练+门禁 / 可全部分工） |
| 内存 | 503 GB（训练 CPU offload 池 / RFT 沙箱池） |
| 模型 | qwen3.8-27b，27.3B 参数，**qwen35 混合线性注意力**（64 层中仅 16 层全注意力持 KV，48 层 GDN），原生 VL，词表 248,320 |
| 现役权重 | `Qwen3.8-27B-Heretic-Ara-iq4_xs-3.0-mtp.gguf`（14.33 GB，混合量化 avg 4.20 bpw，内嵌 1 层 MTP 草稿） |
| 训练底座 | `Qwen3.8-27B-BF16/`（55.6 GB，18 分片） |
| 推理引擎 | llama.cpp **b10715**（CUDA 12.4，源码构建镜像 `llama-server:cuda12.4-b10715`） |

**256K 上下文塞进 24GB 的账**：16 层 × 4 KV 头 × 256 dim × 2(K+V) = 32,768 elem/token，
q4_0 KV ≈ 18.4 KB/token → 256K ≈ 4.6 GB + MTP 草稿层 q8_0 ≈ 0.55 GB。
GDN 循环态每序列固定 ~0.88 GiB（不随上下文增长）。单副本实测显存 23.94/24.56 GB（97%）。

## 推理部署（deploy/）

```bash
# 4 副本 + LB（GPU0-3），唯一入口 :8000（OpenAI 兼容 /v1 + Anthropic /v1/messages）
docker compose up -d
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":1024}'

# 可选：GPU/副本监控页 :9000
docker compose --profile mon up -d
```

- 每副本 1 卡、4 slot、单序列 256K ctx、`-fa on`、`--cache-reuse 2048`、`--spec-type draft-mtp`
- r2/r3 为**大会话池**（`-np 1` 单 slot 独享 256K，无驱逐，大上下文会话每轮增量 prefill）；llama/r1 为小会话池（`-np 2`）
- 思考型模型：`max_tokens` 给足（≥1024），否则思考过程耗尽额度导致正文为空

**性能实测**（b10715 双构建复核，2026-08-31）：

| 场景 | 单流 | 说明 |
|---|---|---|
| 短上下文 / 64K | 71.8–79.8 tok/s | 无衰减 |
| ~195K | 28.3–35.3 tok/s | 衰减 ~60% |
| prefill | 2230 @64K / 1454 @195K tok/s | |
| 16 slot 聚合 | ~420 tok/s | 48 路并发靠排队全部成功 |
| MTP 接受率 | 46.6% | 逐位置衰减平（p0=65% p1=62.8% p2=58.7%），草稿链有加长空间 |

## 负载均衡策略（deploy/lb/route.lua）

OpenResty + Lua 的动态路由（多副本 llama-server 的成熟方案很少，这是本仓库的核心贡献之一）：

1. **会话粘滞优先**：key = 请求体前 512B（同一会话每轮相同），TTL 30 分钟命中续期 → 前缀 KV 缓存命中率最大化
2. **粘滞不随池占用漂移**：大会话漂移到新副本 = 每轮全量冷 prefill，更糟；仅满载/不健康才漂
3. **新会话接最闲健康副本**；池占用 ≥ POOL_HI 的副本不接新会话（大上下文会话独占）
4. **大小会话分流**：请求体 >600KB（≈130K+ token）→ 大会话池（r2/r3 独享，每副本装 1 个 130-256K 会话）；其余 → 小会话池。避免大中小会话在共享 KV 池里互相驱逐（`Context size exceeded` / 反复冷 prefill 的根源）
5. 副本负载探测（busy slot 数 + 驻留 prompt token 总和）2 秒缓存；`/_lbstats` 路由计数（粘滞/漂移/新会话/每副本选中）；`DISABLED` 文件软摘除（缩容 drain 用，全禁用时忽略防自锁）

## 后训练（posttrain/）

ms-swift QLoRA，**4×4090（GPU4-7）ZeRO-3 + bnb NF4 double-quant**，2026-09-01 至 09-02 实测 4840 步 / 2 epochs 跑通：

```
DoRA→纯 LoRA (r=32, alpha=64, rsLoRA, lorap_lr_ratio=2.0), all-linear
paged_adamw_8bit, bf16, max_length=1536, bs1×ga2×4卡, lr=2e-4 cosine→2e-5
显存账: 4bit底座 12.40G + bf16 emb/lm_head 5.10G + LoRA/优化器 0.4G
        + CUDA ctx/NCCL 0.9G + 激活(≤1536, grad-ckpt) 2.2G = 稳态 ~21.6G/卡
```

- 数据：thinking 短轨（nemotron-code/swe-gym）+ 非 thinking 短轨（magicoder/commitpackft/xlam/toolace…）+ 长上下文轨（8-16K，对症 195K 衰减）三轨分离，MinHash 去重，eval-rulers 永不入训
- 断点续训：`RESUME_CKPT=<ckpt路径> bash run_train.sh a`；cron 每 30 分钟巡检，死亡自动从 checkpoint 续
- **踩坑 6 则全部记录在 `posttrain/train/TRAIN-NOTES.md`**（swift 默认 fp32 加载 OOM / Arrow schema 推断 / step2 反向重算 OOM 与序列长度无关 / pkill 自杀 / DoRA 不可转 GGUF / swift ckpt 两级目录 glob 恒空）

## LoRA→GGUF 转换与热挂

混合线性注意力（GDN）的 LoRA 转 GGUF 有公开资料未覆盖的死点，本仓库含 llama.cpp 补丁：

1. **GDN out_proj 列重排**（48 层，唯一转换死点，原生 `NotImplementedError`）：
   `convert_lora_to_gguf.py` 给 `LoraTorchTensor` 增加 `index_select`（行→B 列→A）；
   `conversion/qwen.py` `_reorder_v_heads` 让 LoRA 张量走 index_select 置换路径。
   合成 12 类模块适配器全量转换 exit=0，数值验证 7/7 PASS
2. **rsLoRA 缩放补偿**：llama.cpp 运行时 scale = `adapter_scale × alpha/rank`，不认 α/√r；
   热挂须 `--lora-scaled <adapter.gguf>:5.657`（√32）补偿，否则等效缩放差 5.66 倍
3. **免重启 A/B**：生产 llama-server 原生支持 `--lora-init-without-apply + POST /lora-adapters`
4. 12 类模块 GGUF 名映射：`in_proj_a→ssm_alpha, b→ssm_beta, qkv→attn_qkv(融合不拆), z→attn_gate, out_proj→ssm_out`；self_attn q/k/v/o→attn_*；mlp→ffn_*。A 存 (rank,in)，B 存 (out,rank)

实测产物：checkpoint-4840 → `final-lora.gguf`（f32, 934 MB）/ `final-lora-q8.gguf`（248 MB），992 张量 = 496 模块×2（48×5 GDN + 64×3 MLP + 16×4 self_attn，与架构全吻合）。

## 门禁评测（eval/）

**"比现在好"的可证伪定义**：每一版产物必须全轴 ≥ 基线、核心轴（代码/工具）严格 > 基线，否则不上线。

基线（`rulers/baseline-3.0-backfill2.json`，2026-09-02 生产 LB 实测）：

| 轴 | 基线 | 备注 |
|---|---|---|
| humaneval | 84.76% (139/164) | 代码 |
| xfc 工具调用 | 63.5% (127/200) | toolace 单轮抽取，已排除训练重叠 |
| gsm8k | 93% (93/100) | 数学 |
| ifeval | 56% (28/50) | 指令遵循（25 类官方约束） |
| needle | 64K 1/3, 195K 0/3 | 长上下文 |
| longgen | ~43.5 tok/s | ≥2048 token 长生成 |

**判定规则**（`compare_gate.py`）：任一轴绝对回退 >2pp → FAIL；humaneval/xfc 核心轴必须严格提升；tps 相对 2%。

**一键 A/B 门禁**（`gate.sh`）：GPU5 控制实例（8090, 纯底座）+ GPU4 热挂实例（8091, `--lora-scaled <adapter>:5.657`）→ 全轴 → 判定表。

RFT 验证沙箱（`rft/sandbox_bench.py`）：Docker `--network none --read-only --cap-drop ALL --memory 256m --pids-limit 64` 隔离执行 MBPP sanitized_test；50 路 0.8s / 128 路 10.8s / 257 路 16.1s 全部 100% pass，瓶颈在 docker daemon spawn 串行化（>100 路后），生产建议 128-256 一波分批。

## 运维脚本

- `gpu-power.sh`：功耗墙 450W→250W（day）/200W（quiet）——4090 不支持软件锁风扇，功耗墙压低 → 温度降 → 温控曲线自动降风扇，唯一可靠软件降噪手段
- `noise-mode.sh`：白天 LB 软摘除 r2/r3（`DISABLED` 文件，在途请求自然完成）→ docker stop 降风扇；晚间恢复接流。配 crontab 8 点/18 点自动切换
- `mon/`：2s 粒度 GPU + 副本 slot 监控，24h 历史落盘（容器重建不丢）

## 模型权重来源（不入库）

权重体积大，请自行获取（`/data/models/` 对应本仓库 `model/` 目录结构）：

- 官方：Qwen `Qwen3.8-27B`（BF16 18 分片，含 mmproj 可恢复视觉）
- 社区量化：`Qwen3.8-27B-Heretic-Ara-iq4_xs-3.0-mtp.gguf`（Heretic Ara 混合量化 + 内嵌 MTP 层）
- 草稿模型参考：`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`（1.14 GB）

## English TL;DR

Single-machine **8×RTX 4090** full stack for **qwen3.8-27b** (27.3B, hybrid linear-attention / GDN):

- **Inference**: 4× llama.cpp (b10715) replicas, 256K ctx each at 97% VRAM, MTP speculative decoding (~47% acceptance, 72 tok/s single-stream, ~420 tok/s aggregated over 16 slots), fronted by an **OpenResty + Lua dynamic load balancer** with session-sticky prefix-cache routing, saturation drift, and large/small session pool splitting.
- **Post-training**: 4-GPU QLoRA (NF4 + ZeRO-3, rsLoRA r=32) with full VRAM accounting and six documented OOM/schema pitfalls; ~27h for 4840 steps.
- **Deployment of the adapter**: llama.cpp patches to convert GDN/linear-attention LoRA adapters to GGUF (out_proj column-permute dead-end), rsLoRA `√r` scale compensation (`--lora-scaled …:5.657`), and no-restart hot-swapping via `POST /lora-adapters`.
- **Release gating**: a falsifiable A/B gate — baseline frozen on a fixed ruler (HumanEval / tool-call / GSM8K / IFEval / needle / long-gen), FAIL on any >2pp regression, strict improvement required on core axes.

## License

MIT — see [LICENSE](LICENSE). 模型权重版权归原作者所有，本仓库仅含工程代码、配置与评测数据。
