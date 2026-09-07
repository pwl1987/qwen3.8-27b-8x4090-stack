# Qwen3.8-27B · 8×RTX 4090 推理与后训练全栈

> qwen3.8-27b（27.3B 参数，qwen35 混合线性注意力架构）在单机 8×RTX 4090 24GB 上的完整工程实践：
> **推理双引擎**（llama.cpp 4 副本 256K + OpenResty 动态会话粘滞 LB / vLLM 单卡投机解码 130 tok/s @240K）、
> **QLoRA 后训练**、**LoRA→GGUF 转换与运行时热挂**、**A/B 门禁评测**。
> 所有数字均为本机实测，附完整踩坑记录与显存账。

- [仓库结构](#仓库结构)
- [硬件与模型](#硬件与模型)
- [推理 · llama.cpp 生产线（inference/llamacpp/）](#推理--llamacpp-生产线inferencellamacpp)
- [推理 · vLLM 单卡高速通道（inference/vllm/）](#推理--vllm-单卡高速通道inferencevllm)
- [构建配方：CUDA 12 / CUDA 13 双时代（各 build/）](#构建配方cuda-12--cuda-13-双时代各-build)
- [负载均衡策略（inference/llamacpp/lb/route.lua）](#负载均衡策略inferencellamacpplbroutelua)
- [后训练（training/）](#后训练training)
- [LoRA→GGUF 转换与热挂](#loragguf-转换与热挂)
- [门禁评测（eval/）](#门禁评测eval)
- [运维（ops/）](#运维ops)
- [后续优化路线图](#后续优化路线图)
- [模型权重来源（不入库）](#模型权重来源不入库)
- [English TL;DR](#english-tldr)
- [License](#license)

---

## 仓库结构

```
├── inference/                      # ★ 推理双引擎（分线）
│   ├── llamacpp/                   # —— llama.cpp 生产线（生产主力）
│   │   ├── docker-compose.yml      # 4×llama-server 副本 (GPU0-3) + OpenResty LB
│   │   ├── lb/                     # qwen27b.conf + route.lua（会话粘滞/满载漂移/大小会话分流）
│   │   └── build/
│   │       └── cu124-driver550/    # CUDA 12.4 / 驱动 550 时代构建配方（生产现役 b10715）
│   ├── vllm/                       # —— vLLM 单卡高速通道（基于 syv-ai/qwen38-27b-rtx3090 改造, Apache-2.0）
│       ├── compose.yaml            # 生产 compose（单卡 :19622, OpenAI 兼容）
│       ├── .env.example            # 终配模板（240K + DFlash2 k=7 + 重校准 drafter, 已认证）
│       ├── compose.int8ab.yaml     # 双引擎 A/B 样板（!override 换卡换端口）
│       ├── patches/ docker/ kvarn/ single-user/ profiles/ bench/ prepare/ verify.sh  # 栈本体（28 补丁）
│       ├── drafter/                # DFlash2 drafter GPTQ W4A16 重校准管线（+6.9%）
│       ├── results/                # 实验 JSON
│       └── build/
│           ├── cu129-driver550/    # CUDA 12.9 镜像配方说明（0.27.1-cu129, 生产现役）
│           └── cu130-driver580/    # vLLM 0.28 + CUDA13 原生环境五坑配方（驱动 580 时代）
│   └── sglang/                     # —— SGLang 探索存档（已弃用：Anthropic 无结构化 tool_use，
│                                   #    09-03 回归 llama.cpp；基准/巡检/FP8 诊断脚本保留）
├── media/                          # ★ 媒体生产线（M 线）
│   ├── comfyui-minimax-h3/         # ComfyUI+MiniMax-H3 视频生成（GPU2/3 按需）
│   ├── services/                   # 检索/视觉周边池三容器（embed/rerank/VL-8B 可复现启动）
│   ├── news_pipeline/              # 批次I 一键 11 阶段编目管线（P0-P4 全套+阈值真源）
│   ├── serve/                      # 8 个 vLLM 服务启停脚本（vllm28-env 底座）
│   └── docs/                       # RESULTS 滚动实测 / MODEL-PICKS 选型定案 / 素材清单
├── training/                       # ★ 后训练（与推理分离）
│   ├── train/                      # QLoRA 训练（ms-swift + DeepSpeed, 4×4090）
│   │   ├── run_train.sh / convert_data.py / ds_zero*.json / monitor.sh / smoke_test.py
│   │   └── TRAIN-NOTES.md          # ★ 执行记录（含 09-05 起 vLLM/驱动升级/P0 归因全程日志, 689 行）
│   └── llamacpp/                   # llama.cpp b10715 补丁（GDN LoRA→GGUF 转换死点）
├── eval/                           # ★ 评测（按引擎分线）
│   ├── llamacpp/                   # gate.sh 门禁 / compare_gate / run_baseline / rft 沙箱 /
│   │                               # rulers 基线 / spec_sandbox 四配置矩阵 / dflash_vram_sweep
│   └── vllm/                       # p0 差距归因工具箱 / spec_bench 接受率差分 /
│                                   # quality_ab 双引擎质量 A/B + int4 头判决书
├── ops/                            # 机器级运维（引擎无关）
│   ├── mon/                        # GPU/副本实时监控页 v3（collect.py + index.html, :9000）
│   └── scripts/                    # gpu-power.sh v2（逐卡三档功耗墙+节假日日历）/ noise-mode.sh
├── model/                          # qwen3_5 架构定义 / chat template / generation config
└── docs/
    ├── QWEN27B-ANALYSIS.md         # ★ 服务深度分析（权重解剖/量化配方/MTP 实测/§15 三引擎横评）
    ├── VLLM-OPTIMIZATION.md        # ★ vLLM 生产线实验日志（S1→S1.8: 量化/头部判死/drafter/T系列/P0 归因）
    ├── PROBLEMS-AND-FIXES.md      # ★ 问题→解决全记录（P1-P59 + 六条通用教训，08-28→09-07 全程）
    └── ROADMAP.md                  # ★ 后续优化设计构想（adaptive 修复→+32% / drafter 重训蓝图 / 0.28 迁移 / 262K）
```

## 硬件与模型

| 项 | 值 |
|---|---|
| GPU | 8×RTX 4090 24GB |
| 驱动/CUDA | 580.173.02 / CUDA 13.0（2026-09-05 升级；550+cu12.4 时代配方并存于各 build/） |
| 内存 | 503 GB（训练 CPU offload 池 / RFT 沙箱池） |
| 模型 | qwen3.8-27b，27.3B 参数，**qwen35 混合线性注意力**（64 层中仅 16 层全注意力持 KV，48 层 GDN），原生 VL，词表 248,320 |
| llama.cpp 现役权重 | `Qwen3.8-27B-Heretic-Ara-iq4_xs-3.0-mtp.gguf`（14.33 GB，混合量化 avg 4.20 bpw，内嵌 1 层 MTP 草稿） |
| vLLM 现役权重 | `Qwen3.8-27B-coding-v1.1-W4A16`（自后训模型，AutoRound W4A16 + int8 lm_head/embed，19 GB） |

**256K 上下文塞进 24GB 的账（llama.cpp 线）**：16 层 × 4 KV 头 × 256 dim × 2(K+V) = 32,768 elem/token，
q4_0 KV ≈ 18.4 KB/token → 256K ≈ 4.6 GB + MTP 草稿层 q8_0 ≈ 0.55 GB。
GDN 循环态每序列固定 ~0.88 GiB（不随上下文增长）。单副本实测显存 23.94/24.56 GB（97%）。

## 推理 · llama.cpp 生产线（inference/llamacpp/）

```bash
# 4 副本 + LB（GPU0-3），唯一入口 :8000（OpenAI 兼容 /v1 + Anthropic /v1/messages）
docker compose up -d
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":1024}'

# 可选：GPU/副本监控页 :9000（ops/mon/）
docker compose --profile mon up -d
```

- 每副本 1 卡、4 slot、单序列 256K ctx、`-fa on`、`--cache-reuse 2048`、`--spec-type draft-mtp`
- r2/r3 为**大会话池**（`-np 1` 单 slot 独享 256K，无驱逐）；llama/r1 为小会话池（`-np 2`）
- 思考型模型：`max_tokens` 给足（≥1024），否则思考过程耗尽额度导致正文为空

**性能实测**（b10715 双构建复核，2026-08-31）：

| 场景 | 单流 | 说明 |
|---|---|---|
| 短上下文 / 64K | 71.8–79.8 tok/s | 无衰减 |
| ~195K | 28.3–35.3 tok/s | 衰减 ~60% |
| prefill | 2230 @64K / 1454 @195K tok/s | |
| 16 slot 聚合 | ~420 tok/s | 48 路并发靠排队全部成功 |
| MTP 接受率 | 46.6% | 逐位置衰减平（p0=65% p1=62.8% p2=58.7%） |

## 推理 · vLLM 单卡高速通道（inference/vllm/）

后训模型 coding-v1.1 跑在改造版 [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
栈上（vLLM 0.27.1 + 28 补丁 + KVarN 4/2-bit KV + DFlash2 块式投机解码），单卡 24GB 内塞下
**240K 上下文 + 视觉 + 前缀缓存**：

| 指标 | 值 | 对比生产副本 |
|---|---|---|
| decode（p565/g512 中位） | **130 tok/s @240K** | 1.45×（90 tok/s），且省 1 卡 |
| prefill | ~2550 tok/s @32K | |
| 投机解码 | DFlash2 k=7，3.0-3.2 tok/step，24ms/步 | MTP 链实测 102 tok/s，弃 |

关键工程点（完整日志 `docs/VLLM-OPTIMIZATION.md`）：drafter GPTQ 重校准（+6.9%）、240K 显存配平
（KV 5.26→4.86GB + CUDA 图 1400→1000MiB）、int4 lm_head 四轮校准判死、P0 差距归因 20+ 臂
（步速与参照持平；lookup/adaptive 通道可到 171.6 但有前缀缓存正确性损坏，质量否决——修复路线见 `docs/ROADMAP.md`）。

## 构建配方：CUDA 12 / CUDA 13 双时代（各 build/）

机器经历 驱动 550.163.01+CUDA 12.4 → 驱动 580.173.02+CUDA 13.0（2026-09-05），两代配方并存：

| 目录 | 时代 | 内容 |
|---|---|---|
| `inference/llamacpp/build/cu124-driver550/` | cu12 / 550 | llama.cpp b10715 runtime Dockerfile + 编译要点（devel 容器 --gpus all） |
| `inference/vllm/build/cu129-driver550/` | cu12 / 550+ | vLLM 0.27.1-cu129 生产镜像（= 上级 docker/，28 补丁自动应用） |
| `inference/vllm/build/cu130-driver580/` | **cu13 / 580** | vLLM 0.28 + torch 2.13+cu130 原生环境**五坑配方**（CUDA_HOME=wheel 自带 cu13 / PATH+ninja / flashinfer 采样器关 / 视频像素预算 / pkill 自杀） |

驱动向后兼容：cu12 时代的两个生产镜像在 580 驱动下照常运行；cu13 原生重编译按上述配方。

## 负载均衡策略（inference/llamacpp/lb/route.lua）

OpenResty + Lua 的动态路由（多副本 llama-server 的成熟方案很少，这是本仓库的核心贡献之一）：

1. **会话粘滞优先**：key = 请求体前 512B（同一会话每轮相同），TTL 30 分钟命中续期 → 前缀 KV 缓存命中率最大化
2. **粘滞不随池占用漂移**：大会话漂移到新副本 = 每轮全量冷 prefill，更糟；仅满载/不健康才漂
3. **新会话接最闲健康副本**；池占用 ≥ POOL_HI 的副本不接新会话（大上下文会话独占）
4. **大小会话分流**：请求体 >600KB（≈130K+ token）→ 大会话池（r2/r3 独享）；其余 → 小会话池。避免大中小会话在共享 KV 池里互相驱逐
5. 副本负载探测 2 秒缓存；`/_lbstats` 路由计数；`DISABLED` 文件软摘除（缩容 drain 用，全禁用时忽略防自锁）

## 后训练（training/）

ms-swift QLoRA，**4×4090（GPU4-7）ZeRO-3 + bnb NF4 double-quant**，2026-09-01 至 09-02 实测 4840 步 / 2 epochs 跑通：

```
DoRA→纯 LoRA (r=32, alpha=64, rsLoRA, lorap_lr_ratio=2.0), all-linear
paged_adamw_8bit, bf16, max_length=1536, bs1×ga2×4卡, lr=2e-4 cosine→2e-5
显存账: 4bit底座 12.40G + bf16 emb/lm_head 5.10G + LoRA/优化器 0.4G
        + CUDA ctx/NCCL 0.9G + 激活(≤1536, grad-ckpt) 2.2G = 稳态 ~21.6G/卡
```

- 数据：thinking 短轨 + 非 thinking 短轨 + 长上下文轨（8-16K，对症 195K 衰减）三轨分离，MinHash 去重，eval-rulers 永不入训
- 断点续训：`RESUME_CKPT=<ckpt路径> bash run_train.sh a`；cron 每 30 分钟巡检，死亡自动从 checkpoint 续
- **踩坑 6 则全部记录在 `training/train/TRAIN-NOTES.md`**（swift 默认 fp32 加载 OOM / Arrow schema 推断 / step2 反向重算 OOM 与序列长度无关 / pkill 自杀 / DoRA 不可转 GGUF / swift ckpt 两级目录 glob 恒空）

## LoRA→GGUF 转换与热挂

混合线性注意力（GDN）的 LoRA 转 GGUF 有公开资料未覆盖的死点，本仓库含 llama.cpp 补丁（`training/llamacpp/`）：

1. **GDN out_proj 列重排**（48 层，唯一转换死点，原生 `NotImplementedError`）：
   `convert_lora_to_gguf.py` 给 `LoraTorchTensor` 增加 `index_select`（行→B 列→A）；
   `conversion/qwen.py` `_reorder_v_heads` 让 LoRA 张量走 index_select 置换路径。
   合成 12 类模块适配器全量转换 exit=0，数值验证 7/7 PASS
2. **rsLoRA 缩放补偿**：llama.cpp 运行时 scale = `adapter_scale × alpha/rank`，不认 α/√r；
   热挂须 `--lora-scaled <adapter.gguf>:5.657`（√32）补偿，否则等效缩放差 5.66 倍
3. **免重启 A/B**：生产 llama-server 原生支持 `--lora-init-without-apply + POST /lora-adapters`
4. 12 类模块 GGUF 名映射：`in_proj_a→ssm_alpha, b→ssm_beta, qkv→attn_qkv(融合不拆), z→attn_gate, out_proj→ssm_out`；self_attn q/k/v/o→attn_*；mlp→ffn_*。A 存 (rank,in)，B 存 (out,rank)

实测产物：checkpoint-4840 → `final-lora.gguf`（f32, 934 MB）/ `final-lora-q8.gguf`（248 MB），992 张量 = 496 模块×2。

## 门禁评测（eval/）

**"比现在好"的可证伪定义**：每一版产物必须全轴 ≥ 基线、核心轴（代码/工具）严格 > 基线，否则不上线。

**llama.cpp 线**（`eval/llamacpp/`）：基线 `rulers/baseline-3.0-backfill2.json`（2026-09-02 生产 LB 实测）

| 轴 | 基线 | 备注 |
|---|---|---|
| humaneval | 84.76% (139/164) | 代码 |
| xfc 工具调用 | 63.5% (127/200) | toolace 单轮抽取，已排除训练重叠 |
| gsm8k | 93% (93/100) | 数学 |
| ifeval | 56% (28/50) | 指令遵循（25 类官方约束） |
| needle | 64K 1/3, 195K 0/3 | 长上下文 |
| longgen | ~43.5 tok/s | ≥2048 token 长生成 |

判定规则（`compare_gate.py`）：任一轴绝对回退 >2pp → FAIL；humaneval/xfc 核心轴必须严格提升。
一键 A/B 门禁（`gate.sh`）：控制实例（纯底座）+ 热挂实例 → 全轴 → 判定表。
RFT 验证沙箱（`rft/sandbox_bench.py`）：Docker `--network none --read-only --cap-drop ALL` 隔离执行 MBPP
sanitized_test；50 路 0.8s / 128 路 10.8s / 257 路 16.1s 全部 100% pass。

**vLLM 线**（`eval/vllm/`）：`p0/` 差距归因工具箱（隔离步速分解 / **12 残差前缀命中正确性门** /
单变量 A/B 试验机）、`spec_bench.py` 接受率差分、`quality_ab_*.py` 双引擎 10 任务质量 A/B、
int4 头判死判决书。

## 运维（ops/）

- `gpu-power.sh`（v2）：功耗墙逐卡三档 `gpu-power.conf`（day/quiet/night）+ `holidays.txt`/`workdays.txt`
  节假日日历、周末自动跳过、幂等执行、status 对照表——4090 不支持软件锁风扇，功耗墙压低 → 温度降 →
  温控曲线自动降风扇，唯一可靠软件降噪手段。实测 250W 日间档对显存带宽型 decode 仅 -1.5%
- `noise-mode.sh`：白天 LB 软摘除 r2/r3 → docker stop 降风扇；晚间恢复。配 crontab 8/18 点自动切换
- `mon/`（v3）：2s 粒度 GPU + 副本 slot + vLLM 主力引擎监控，24h 历史落盘（容器重建不丢）

## 后续优化路线图

见 **`docs/ROADMAP.md`**——四方向设计构想：A) 修复 adaptive×前缀缓存损坏 → 171.6 档解禁（+32%，
配置与回归门已备）；B) DFlash2 drafter 深度重训蓝图（fc+selector 在线蒸馏，裸接受率 3.0→3.5+）；
C) vLLM 0.28 + cu13 栈迁移（28 补丁重验）；D) 262K 恢复（依附 A/C）。llama.cpp 线维持。

## 模型权重来源（不入库）

权重体积大，请自行获取（`/data/models/` 对应本仓库 `model/` 目录结构）：

- 官方：Qwen `Qwen3.8-27B`（BF16 18 分片，含 mmproj 可恢复视觉）
- 社区量化：`Qwen3.8-27B-Heretic-Ara-iq4_xs-3.0-mtp.gguf`（Heretic Ara 混合量化 + 内嵌 MTP 层）
- 自后训：`Qwen3.8-27B-coding-v1.1`（QLoRA 产物，见 training/；vLLM 线用其 W4A16+int8 头量化版）
- 草稿模型参考：`Qwen3.8-27B-DFlash2`（侧车 drafter，1.92B）

## English TL;DR

Single-machine **8×RTX 4090** full stack for **qwen3.8-27b** (27.3B, hybrid linear-attention / GDN):

- **Inference, two engines**: (1) 4× llama.cpp (b10715) replicas, 256K ctx each at 97% VRAM, MTP speculation (~47% acceptance, 72 tok/s single-stream, ~420 tok/s aggregated), fronted by an **OpenResty + Lua dynamic load balancer** (session-sticky prefix-cache routing, saturation drift, large/small session pools); (2) a single-card **vLLM fast lane** (adapted 0.27.1 stack: KVarN 4/2-bit KV + DFlash2 block speculation) running the post-trained model at **130 tok/s decode @240K** — 1.45× a production replica on one card, with a GPTQ-recalibrated drafter (+6.9%), a 20-arm gap-attribution campaign (step-time parity proven; the remaining headroom lives in a lookup/adaptive lane that is output-corrupting under prefix-cache hits and therefore vetoed), and a 12-residue prefix-cache correctness gate.
- **Build recipes for both CUDA eras**: cu12.4/550 (llama.cpp) and cu12.9/550+ (vLLM image) alongside a cu13/580 native vLLM 0.28 environment with its five documented pitfalls.
- **Post-training** (`training/`): 4-GPU QLoRA (NF4 + ZeRO-3, rsLoRA r=32) with full VRAM accounting and six documented pitfalls; llama.cpp patches converting GDN LoRA adapters to GGUF (out_proj column-permute dead-end), rsLoRA √r scale compensation, no-restart hot-swapping.
- **Release gating** (`eval/`, split per engine): falsifiable A/B gates against frozen rulers; sandboxed RFT execution; step-time/acceptance decomposition and correctness gates for the speculative-decoding lane.
- **Roadmap** (`docs/ROADMAP.md`): fix the adaptive×prefix-cache corruption to unlock a measured +32% config; distillation-retrain the DFlash2 drafter; migrate to vLLM 0.28/cu13; restore 262K.

## License

MIT — see [LICENSE](LICENSE). 模型权重版权归原作者所有，本仓库仅含工程代码、配置与评测数据。
`inference/vllm/` 目录包含来自 [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) 的 Apache-2.0 代码（附原许可证），其余为本仓库原创。
