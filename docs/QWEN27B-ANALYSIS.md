# Qwen3.8-27B 服务深度分析与优化路线

> 2026-08-31 由权重解析 + 三仓库对照整理。结论基于实测数据（GGUF 头解析、/metrics、
> syv-ai / r0b0tlab / alesha-pro 三仓库基准）。改动配置前先读第 5 节的灰度原则。

---

## 1. 服务现状快照

- 硬件：8×RTX 4090 24GB（GPU0-3 = Qwen 服务 4 副本；GPU4 = zxb-judge 裁判模型；
  GPU5-7 空闲但预留 ZxBench 评测）
- 引擎：llama.cpp **b10715**（镜像 `llama-server:cuda12.4-b10715`，源码
  `/data/build/llama.cpp-new`，2026-08-31 升级；旧镜像 `-local` 保留可回滚）
- 模型：`/data/models/Qwen3.8-27B-Heretic-Ara-iq4_xs-3.0-mtp.gguf`（14.33 GB）
- 每副本：1 卡、4 slot、单序列最高 256K ctx、KV q4_0（MTP 草稿层 KV q8_0）、
  fa on、`--cache-reuse 2048` 前缀复用、显存 23.94/24.56 GB（97%）
- 入口：OpenResty :8000 动态 LB（会话粘滞保前缀命中、满载漂移、TTL 30min）
- 性能（**2026-08-31 双构建实测**，此前记录的"~100-113 tok/s"无法复现，两镜像均为
  ~70，判定为历史口径问题，废弃）：短上下文 71.8-79.8 tok/s；64K 71.7-73.4（无衰减）；
  **~195K 仅 28.3-35.3（衰减 ~60%，旧"200K 针通过"只验了正确性未测速度）**；
  prefill 2230@64K / 1454@195K tok/s；MTP 接受率 46.6%（metrics 差分实测）；
  16 slot 聚合 ~420；48 路靠排队全成功

> 数据可信度分级：本档一手实测 > GGUF/源码解析 > 我方推导（显存账等）>
> 外部仓库 README 摘要（二手，引用时注明出处）。所有 tok/s 结论以本节实测为准。

## 2. 权重解剖（GGUF 头解析结果）

### 2.1 架构：qwen35 混合线性注意力，不是 dense

| 项 | 值 | 说明 |
|---|---|---|
| 层数 | 64 主干 + 1 MTP 草稿层 | `nextn_predict_layers=1`，MTP 内嵌 |
| full_attention_interval | **4** | 每 4 层才有 1 层全注意力（16/64 层），其余 48 层 GDN |
| SSM/GDN | state 128 / inner 6144 / conv 4 / dt_rank 48 / 16 组 | 循环状态每序列固定显存，不随上下文增长 |
| 注意力 | GQA 24 Q 头 / 4 KV 头，head_dim 256 | 仅 16 层持有 KV → 24GB 塞下 256K 的根本原因 |
| 词表 | 248,320 | token_embd + lm_head 各 1.27B 参数 |
| RoPE | 64 维/头，sections [11,11,10]，freq_base 10M | mRoPE，为 256K 调整 |
| 总参数 | 27.3B | 官方卡：BF16 55.6GB（18 分片），原生 VL 模型 |

**KV 成本账**：16 层 × 4 KV 头 × 256 dim × 2(K+V) = 32,768 elem/token；
q4_0 ≈ 18.4 KB/token → 256K ≈ 4.6 GB。MTP 草稿层 q8_0 再加 ~0.55 GB @256K。

### 2.2 量化配方（"iq4_xs-3.0"，平均 4.20 bpw）

| dtype | 张量数 | 参数量 | 用途 |
|---|---|---|---|
| IQ4_XS | 204 | 13.36B | 主体 |
| IQ3_S | 46 | 3.76B | 低敏感层压紧 |
| Q5_K | 63 | 3.43B | 含 lm_head |
| Q4_K / Q3_K | 51+9 | 2.12B + 1.87B | Q3_K 含 token_embd |
| IQ3_XXS / IQ2_S / IQ2_XS / Q2_K | 27 | 2.3B | ≤3bpw 压紧 |
| Q8_0 | 50 | 0.02B | GDN 小状态（dt/A/D/conv）特意保精度 |
| F32 | 408 | 0.01B | 全部 norm |

**弱点**：imatrix 校准只用 wikitext-2-raw 的 **93 chunks / 496 条**（量化者在
Windows 用 b9296 做的）——数据量小，且英文维基语料与实际用途（代码/Agent）不匹配。
约 28% 参数跑在 ≤3.4bpw。

### 2.3 视觉血统（重要）

词表保留全部视觉特殊 token：`<|vision_start|>`(248053)、`<|image_pad|>`(248056)、
`<|video_pad|>`(248057) + 定位 token（`<|box_start|>`/`<|quad_start|>` 等）。
官方 Qwen3.8-27B 是**原生 VL 模型**（pipeline_tag=image-text-to-text，图像+视频，
不支持音频——词表里的 audio token 是家族残留）。当前 GGUF 是"LM 抽取版"，恢复视觉
= 补 mmproj，见第 6 节。

### 2.4 MTP 实测（副本 0 /metrics，重启以来）

- 1666 轮草稿 × 每轮 3 token；接受 2162/4997 = **43.3% 总接受率**
- 逐位置条件接受率：p0=65%、p1=62.8%、p2=58.7%（**衰减平** → 草稿链有加长空间）
- 每轮净接受 1.30 token → 每次前向 ~2.4 token（2026-08-31 差分复核：接受率 46.6%，
  与累计值一致；对应实测单流 ~72 tok/s）

## 3. 三参考仓库对照

| 方案 | 硬件 | 单流 | 并发聚合 | 关键技术 |
|---|---|---|---|---|
| **本服** llama.cpp IQ4_XS+MTP3+q4_0 KV | 1×4090 | 72（短/64K）/ 28-35@195K | 420 @16 slot | 4 副本 + LB 粘滞 |
| syv-ai vLLM W4A16+int8 激活+MTP4/DFlash2 | 1×3090 | 114-124（4090 上 135.5） | **1000-1222** @64 路 | 自有 40k 草稿词表（接受率 69→74%）、k=4 拐点、前缀缓存 TTFT 22.4s→0.56s、n-gram 链 +7% |
| r0b0tlab NVFP4+FP8 KV+DFlash2 K=8 | GB10 | 67 vs MTP 27.8（2.4×） | 279 @c6 | SM121 FP4 GEMM 调优（不适用 4090）、质量几乎无损（GSM8K 87/HE 89）、262K NIAH 全过 |
| alesha-pro 基准：llama.cpp Q4_K_M | 1×3090 | 39.6（KV f16）→30.3@131K | — | 深度衰减数据；MTP 短窗口虚高（84 实为 75） |
| alesha-pro 基准：vLLM NVFP4 TP4+FP8 KV+MTP | 4×3090 | 84.26 @254K | — | **FP8 KV 是 vLLM 长上下文命门**（BF16 KV 时 MTP 崩到 8） |

**定位结论（2026-08-31 修正）**：性能三项 vLLM 全部领先或持平——短上下文单流
72 vs 114-135（syv-ai vLLM，3090/4090）；长上下文 28-35@195K vs vLLM TP2 64@254K
（每 GPU 归一 30 vs 32 大致持平）；聚合 420 vs 1000+/卡。llama.cpp 的真实优势在：
运维简单、KV 池显存效率、粘滞+前缀缓存原生、FIM/infill 生态、零迁移成本。

## 4. 本版 llama.cpp（b10715）可用杠杆

- `--spec-type` 支持逗号组合 9 种：`draft-mtp` / `draft-dflash` / `draft-eagle3` /
  `draft-dspark` / `ngram-simple` / `ngram-map-k` / `ngram-map-k4v` / `ngram-mod` /
  `ngram-cache`
- sidecar 草稿：`--spec-draft-model FILE` 或 `--spec-draft-hf`（类型可从草稿 GGUF
  元数据自动推断）
- 视觉：`--mmproj FILE`、`--mmproj-device DEVICE`（视觉塔可放另一卡）、
  `--mmproj-offload`；`-hf` 用法可自动下载 mmproj
- 多模态库 libmtmd.so 已在镜像内；llama-bench 可用（llama-quantize **不在**，
  重量化需从 `/data/build/llama.cpp-new` 构建）

## 5. 优化路线图（按路线三分）

### 5.1 路线一：当前 llama.cpp 深化（零迁移，默认主线）

优化项按优先级：
1. **DFlash2 草稿**（✅ 已验证支持）：b10715 含完整实现（PR #27342）；官方草稿
   `z-lab/Qwen3.8-27B-DFlash2-GGUF`（ModelScope 镜像可达）：Q4_K_M 1.1GB 最优档
   （接受长度实测 5.39 > BF16 的 5.28），vs 当前 MTP-1 每步 ~1.3。
   `-md /models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash
   --spec-draft-n-max 7`。显存 +1.1GB → 腾法：-c 262144→~180K 或 slot 4→3，
   先 GPU5 试点看 Heretic-Ara 目标下接受率漂移。vLLM 侧对应门槛 ≥0.28.0。
2. **ngram 组合草稿**：`--spec-type draft-mtp,ngram-cache`，代码场景超额加成
   （syv-ai 通用复现 +7%），1 小时灰度。
3. **KV 提质**：K→q5_1（V 保持 q4_0）；~1.2GB/256K。
4. **域校准重量化**：代码+中文+Agent 语料 ≥500 chunks 重做 imatrix，产 4.0 版
   混合配方（weights 14.3→~13GB，腾 1.3GB 给 mmproj/KV 提质）。需构建 llama-quantize。
5. **基座对照**：官方 27B vs Heretic-Ara 的 HumanEval/LiveCodeBench（ZxBench），
   决定重量化的起点。
6. 编程专项与视觉 mmproj：见第 6、7 节。

- 优点：零迁移（LB/监控/灰度全保留）；单流与长上下文每 GPU 效率三路线最高
  （1 卡 100-113@256K）；GDN 适配天然（KV 池 97%、前缀缓存原生、256K 进 24GB
  独有）；运维最轻；/infill、slot 持久化、量化粒度全可控
- 缺点：聚合 ~420 tok/s 结构性封顶（静态 slot）；无 CUDA Graphs/激活量化；
  单流极限 ~170；无 TP 降延迟；重量化工具链自建
- 预期：单流 72→110-160（DFlash2 理论 72×6.4/2.4≈190，扣草稿开销）；聚合不变；
  编程场景响应显著改善

### 5.2 路线二：vLLM 整体切换（吞吐换运维）

方案：vLLM ≥0.28.0（DFlash2 #52816 / DSpark #47808 已进主线）+ 自产 AutoRound
W4A16（域校准 + SpinQuant 旋转，llm-compressor 流水线）+ int8 激活 + FP8 KV +
DFlash2 + 移植 syv-ai GDN 前缀缓存补丁。

- 优点：聚合吞吐量级优势（syv-ai 单 3090 实测 1000-1222 tok/s，4 卡 2000+）；
  优化阶梯最深（CUDA Graphs/Marlin/TP/chunked prefill/自动前缀缓存）；
  单流 ~135；multi-LoRA 热挂载、视频帧采样；W4A16+旋转近无损（GSM8K 96.5）
- 缺点：GGUF 资产作废（格式不通用）；补丁债（GDN 前缀缓存移植+每版重打）；
  每 GPU 长上下文效率最低（4 卡 84 vs llama.cpp 1 卡 100+）；Python 生态运维；
  CUDA Graph 必须 PIECEWISE（FULL 在 MTP+256K 损坏输出）；FP8 KV 强制
- 预期：聚合 420→2000+；单流 ~135；批量承载质变
- 结论：仅当放弃 GGUF 资产且交互负载消失才成立；**永远以混合形态引入，不推倒重来**

### 5.3 路线三：混合部署（2+2 按负载分流）

架构：GPU0-1 两副本 llama.cpp（交互入口：长会话/补全/粘滞）+ GPU2-3 两副本
vLLM（批量入口：评测洪峰/高并发/LoRA 服务）；OpenResty 按请求特征分流或双端口。

- 优点：各取所长；渐进可逆（vLLM 先在 GPU5-7 验证再换入）；风险隔离（补丁
  问题只伤批量通道）；ZxBench 洪峰路由批量侧，交互零感知
- 缺点：双引擎运维 ×2；路由逻辑复杂化（判据错误=性能劣化）；llama.cpp 侧
  slot 减半；权重双份管理

### 5.4 对比与决策

| 维度 | 一 llama.cpp | 二 vLLM | 三 混合 |
|---|---|---|---|
| 单流（短/64K） | 72（DFlash2→110-160） | **~135** | 按路由最优 |
| 单流@~200K | 28-35 | **64-84**（TP2/TP4） | 按路由最优 |
| 聚合(4卡) | ~420 封顶 | **2000+** | 交互+批量各取 |
| 每 GPU 长上下文 | 30/卡（持平 TP2） | 21-32/卡 | 按负载匹配 |
| 前缀缓存 | 原生+粘滞 | 需补丁 | 双轨 |
| 迁移成本 | **零** | 高 | 中（渐进） |
| 运维 | 低 | 中高 | 中（×2 隔离） |
| 编程适配 | /infill 现成 | multi-LoRA | 全覆盖 |

决策：16 slot 够用 → 路线一（DFlash2 试点即刻做）；批量压力真实出现 →
路线三渐进混部；NCCL 调优（NCCL_P2P_DISABLE/ALGO）列入 vLLM 上线必测。

### 不要做

- NVFP4：4090 无 FP4 硬件（内核在但 GGUF 生态断链；vLLM Marlin 回退无优势，
  r0b0tlab +27% 是 SM121 调优专属）
- 2-bit / 极致量化：27B 无需塞 24GB，且有失控生成风险（alesha RUNAWAY）
- FP8 全量权重：单卡放不下（27.8GB），且比 NVFP4 慢 15-21%

### 验证方法论（alesha 的教训）

- MTP/草稿对比必须 ≥2048 token 长生成，不能只看 128-token 窗口（短窗虚高）
- 质量回归用 ZxBench 跑 GSM8K/长针/IFEval 对照

## 6. 视觉功能恢复

**结论：llama.cpp 现环境只差一个 mmproj 文件；vLLM 是另一套独立部署。**

- 现状：基座原生 VL（官方卡确认），当前 GGUF 词表/mRoPE/模板完整保留视觉接口，
  缺的只是视觉塔（ViT+投影器 ≈ 0.47B 参数）
- 现成产物：`unsloth/Qwen3.8-27B-GGUF` 仓的 **mmproj-BF16.gguf（931 MB）**
  （hf-mirror 可达）。b10715 的 mtmd 完整支持（`--mmproj`、`--mmproj-device`）
- **显存是唯一难点**（每副本仅剩 ~600MB）：

| 方案 | 代价 | 说明 |
|---|---|---|
| `--mmproj-device` 指到 GPU5-7 | 最小 | 视觉塔借空闲卡；每图回传几十 MB embedding；会在该卡起 ~400MB CUDA context，需与 ZxBench 错峰 |
| 砍 KV 池 | 中 | 931MB ≈ 5 万 token KV；`-c 262144→~210K` 或 slot 4→3 |
| mmproj 量化 q8_0/q4_0 | 中 | ~500/280MB；需构建 llama-quantize；视觉精度略降 |

- **风险**：LM 是 Heretic-Ara 社区微调，原厂视觉塔配微调后 LM 的对齐需验证
  （OCR/描述/box 定位三件套）；图像 token 会计入上下文（每图几百~几千 token）
- vLLM 路线：官方原生支持视觉 + 独有视频帧采样（fps）；但需 HF 格式检查点
  （BF16 55.6GB TP3-4 / 官方 FP8 TP2 / 自产 W4A16），与前缀缓存缺失问题同上
- 落地顺序：下载 mmproj → GPU5 测试容器验证质量 → 灰度 r3（--mmproj-device 或减
  ctx）→ 观察 MTP 接受率是否受图像请求影响 → 视频需求变重再评估 vLLM

## 7. 编程场景专项优化（2026-08-31 补充）

词表里存在完整编程 token：`<|fim_prefix|>/<|fim_middle|>/<|fim_suffix|>/<|fim_pad|>`
（248060-248063）+ `<|repo_name|>/<|file_sep|>`（248064/248065）——基座原生支持
FIM 行内补全与仓库级上下文，当前部署只用了 chat 端点。

已验证本构建（b10715）现成支持：
- **`/infill` 端点**（server.cpp 已注册）→ 直接接入 continue.dev/Tabby 做补全后端
- **`--slot-save-path PATH`**（KV 槽位落盘）→ 副本重启不丢上下文 + 常用仓库预热

编程专项旋钮：
| 项 | 改法 | 原理 |
|---|---|---|
| `--cache-reuse` | 2048 → 512 | 编辑器重发文件只改几行，小粒度复用=高 diff 命中 |
| LB 路由键 | 会话 ID → 仓库哈希 | 同仓库粘滞同副本，仓库 prefill 缓存常驻 |
| slot 几何 | 拿 1 副本改 8 slot × 32K | 补全负载=长 prompt 短输出，并发容量翻倍 |
| 采样路由 | 补全走非 thinking（0.7/0.8/penalty 1.5），难题走 thinking（1.0/20/0.95） | 官方卡两套参数，GGUF 嵌的是 thinking 套 |
| n-gram 草稿 | `draft-mtp,ngram-cache` | 代码标识符/样板重复度高于自然语言，加成超 syv-ai 的 +7% |
| 结构化输出 | GBNF/JSON schema 约束 | 工具调用 JSON 保证合法 |

**质量疑点**：Heretic-Ara（abliteration 系微调）可能伤编程能力；用 ZxBench 跑
官方 27B vs Heretic-Ara 的 HumanEval/LiveCodeBench 对照，若官方更强则换基座重量化
（域校准+混合配方，weights 14.3→~13GB 顺带腾 1.3GB 给 mmproj 或 KV 提质）。

## 8. 结构性微调与分层优化（按训练强度四层）

- **第 0 层 零训练（=路线 5.1.4，首选）**：逐张量 imatrix/Hessian 敏感度 → 分层位宽
  分配 → 重量化。硬件成本≈0。两种格式均可分层（2026-08-31 验证）：GGUF 逐张量
  自带 dtype（本文件即 10 种混装），工具 `llama-quantize --tensor-type-file
  配方`（本构建 tools/quantize 已具备）；safetensors 走 compressed-tensors /
  ModelOpt 元数据逐模块声明（r0b0tlab ckpt 即三精度混装），且可逐层混激活量化
  （W4A16/W4A8）。同一份敏感度分析两边的重量化流水线通用
- **第 1 层 QLoRA（可行，性价比最高）**：单卡 4090 即可（4-bit 底座 14GB + 适配器；
  全 FT 需 220GB+，8×24 不够且卡在服务）。玩法：量化损伤修复（QA-LoRA）、
  编程域适配、敏感层定向 LoRA。产物可 `--lora` 热挂现有副本（b10715 已验证支持）
  不停服试验，确认后合并进最终权重。蒸馏教师可用自家 4 副本（自蒸馏零额外显存）
- **第 2 层 剪层（64→48）**：短上下文速度 +25%，但 GDN"3:1 节律+循环状态动力学"
  无社区先例=自研实验，且收益与 DFlash2 重叠 → 低优先级
- **第 3 层 GDN 状态维度手术**：需重训练，纯研究 → 跳过

约束：GPU5-7 与 ZxBench 错峰；训练容器 torch 钉 ≤2.8（驱动约束）；
QLoRA 底座需官方 BF16 55.6GB（hf-mirror）。
推荐流水线：敏感度→分层量化→重量化 → ZxBench 门禁 → QLoRA 修复+域适配 →
门禁 → 合并 → 最终 GGUF 灰度。

## 9. 分层量化实施手册（两格式）+ 后训练增强（2026-08-31）

> 共同约定：训练/量化任务用 GPU5-7 且与 ZxBench 错峰；容器 torch 钉 ≤2.8；
> 所有产物过 ZxBench 门禁（GSM8K/长针/IFEval + ≥2048 token 长生成）。

### 9.1 A 线：GGUF 分步（工具已在本构建源码验证）

```
A0 构建工具链（源码 /data/build/llama.cpp-new）
   cmake -B build -DGGML_CUDA=ON && cmake --build build -j \
     --target llama-quantize llama-imatrix llama-bench
   # 产出 llama-quantize(--tensor-type-file)、llama-imatrix、llama-bench

A1 域校准语料（两线共用）
   代码+中文+Agent 对话混合 ≥500 chunks（每 chunk 512-2048 tok），
   拼接为 /data/quant/calib.txt；另存 JSONL 版给 B 线

A2 基座准备
   官方 BF16 55.6GB（hf-mirror 下载）→ convert_hf_to_gguf.py 转 F16 GGUF
   （同时 --mmproj 顺产视觉塔）；转换后必须验证含 blk.65 MTP 层
   （nextn_predict_layers=1），缺失则从现有 mtp 文件提取合并

A3 敏感度：在 F16 GGUF 上算 imatrix（勿在量化权重上算）
   llama-imatrix -m base-F16.gguf -f calib.txt -o domain.imatrix.dat \
     -c 2048 --chunks 500+

A4 敏感度 → 配方 tensor_types.txt（每行 张量名=ggml_type，支持正则）
   top 敏感（attn_v/ffn_gate/lm_head 类）→ Q6_K/Q5_K；主体 IQ4_XS；
   低敏感 → IQ3_XS；GDN 小状态保持 Q8_0；目标文件 ~13GB（腾 1.3GB）

A5 重量化
   llama-quantize --imatrix domain.imatrix.dat \
     --tensor-type-file tensor_types.txt base-F16.gguf \
     Qwen3.8-27B-iq4xs-4.0-mtp.gguf iq4_xs

A6 验证：llama-bench 三深度（短/64K/195K，对照 3.0 版：72/72/30 基线）
   + ZxBench 门禁 + MiP 接受率不受损

A7 灰度：r3 换文件，--metrics 观察 24h，回滚预案=保留 3.0 文件
```

### 9.2 B 线：safetensors/compressed-tensors 分步

```
B0 容器：python3.11 + torch≤2.8 + llm-compressor(≥0.7, 含 AutoRound
   集成与 SpinQuant transform) + vLLM≥0.28（验证用）；底座=官方 BF16
B1 校准集：复用 A1 的 JSONL
B2 旋转（推荐）：按 llm-compressor transform 示例跑 SpinQuant/Hadamard，
   产出旋转版底座（量化误差数学性压缩）
B3 分层配方：oneshot() modifiers 逐模块声明——
   敏感层 GPTQModifier(targets=[...], scheme="w8a16") +
   主体 scheme="w4a16" + lm_head/embedding 单独指定
B4 吞吐档（可选）：部分层改 int8 激活（W4A8），换 ~+40% 吞吐、
   PPL +0.9-3.7%（syv-ai 阶梯），做成可切换双产物
B5 验证：GPU5 单卡 vLLM serve，同题三深度 + ZxBench；
   质量参考线 syv-ai W4A16（GSM8K 96.5）
```

### 9.3 后训练增强：可以，分三档（回答"是否可提高能力"）

**原则：分层量化只止损不增益；后训练才加能力。**

| 档 | 做法 | 提升什么 | 成本 |
|---|---|---|---|
| 1 量化修复 | QA-LoRA：在量化模型上训 LoRA，BF16 输出做教师 | 恢复量化损失（PPL 回 ~BF16） | 单卡×1天 |
| 2 域适配（真提能力） | QLoRA SFT：代码/Agent 语料（5-10K 样本） | 目标域表现↑（代码风格/工具调用/格式）；通用能力不涨 | 单卡×1-2天/轮 |
| 3 QAT（高阶） | 旋转+量化感知训练 | 4bit 贴满 BF16 | 多卡多天，暂缓 |

两种次序（都合法，门禁不同）：
- **先量化后训练**：QLoRA 直接在 4-bit 底座训 → 产物即用，量化损伤被训练吸收
- **先训练后量化**：BF16 训 LoRA → 合并 → 再走 A5 量化 → 需二次门禁（量化会再损伤）

产物落地（已验证支持）：
- GGUF 侧：`convert_lora_to_gguf.py` 转 adapter → 副本 `--lora` **热挂不停服**试效
- vLLM 侧：`--enable-lora` 原生多适配器热挂
- 确认有效后合并进权重，纳入最终版量化

训练模板（容器内，GPU6 错峰）：
```
accelerate launch qlora_train.py --base Qwen/Qwen3.8-27B --load_in_4bit \
  --data /data/sft/code-agent.jsonl --lora_r 64 --lora_alpha 128 \
  --lr 2e-4 --per_device_bs 1 --ga 8 --epochs 2
```

### 9.4 推荐总流水线与时间预算

```
A1 校准语料 ──┬→ A2-A7 GGUF 分层重量化(1-2天) ──→ 门禁 ─→ 灰度上线
              └→ B0-B5 safetensors 分层量化(1-2天, 可并行) → vLLM 混部评估
门禁通过后 → 档1 量化修复 + 档2 编程域适配(各1-2天) → --lora 热挂验证
           → 合并 → 最终版
```
### 9.5 训练语料来源（编程/工具/Agent 域，2026-08-31）

RL 语料 ≠ SFT 语料：SFT 要示范轨迹，RL 要"题目+可验证奖励"。

- **SFT 层**：Magicoder OSS-Instruct / xlam-function-calling-60k / ToolBench /
  SWE-Gym / SWE-Master(RUC 全可复现管线) / DeepSWE(全开源含日志) /
  OpenThoughts·CodeI/O 推理链（**保留 thinking 格式**，模型是思考型）
- **RL 燃料层**：SWE-rebench（Nebius 开源 21K+ 真实任务）、LiveCodeBench/CodeContests
  训练拆分（单元测试=奖励）、terminal-bench/R2E-Gym（docker 内验证，本机设施对口）、
  τ-bench/BFCL（工具任务 ground truth）、SWE-RL 思路（自采 git PR/issue/diff）
- **自产层（本机独有）**：① 自家流量（OpenWebUI/ZxBench trace/agent 日志，
  隐私过滤）；② **拒绝采样 RFT**：自家 4 副本生成 N 解→单元测试过滤→留轨迹，
  ZxBench judge 复用为执行验证器——性价比之王；③ 蒸馏教师=自家副本
- **卫生红线**：ZxBench 门禁套件与 LiveCodeBench 评测拆分绝不入训练集；
  MinHash 去重；起步配比 代码50/工具25/agent轨迹25
- **FIM 基线实测（2026-09-01，生产副本 /infill）**：通道通、能力退化（add/mul
  混淆样例、quicksort 空输出、显式 token 探针近空）→ FIM 属"修复"而非从零教，
  成本低；**FIM LoRA 与 chat LoRA 分开训**（混训干扰），--lora 多适配器热挂管理；
  RFT 管线顺手产 FIM 数据（solution 挖空），一套管线两用
- **RFT 环境契合度**：生成器=现役副本空闲产能(API调用)、验证器=ZxBench judge+
  128核、题库=MBPP+SWE-Gym(带FAIL_TO_PASS)、训练段短可中断——**碎片化 GPU 窗口
  下唯一可行的 RL 形态**（GRPO 需连续长占卡，冲突）
- **版本审计（2026-09-01）**：Nemotron 系列/ToolACE/MEnvData/SWE-Gym=最新开放版；
  Magicoder(2023 弱模型合成)**降级为多样性补充**，①轴主数据=Nemotron code；
  CommitPackFT 年代旧但"真实人类 diff"性质不可替代；MBPP 角色使然无新旧。
  轴④下一代是产题管线：SWE-smith(任意仓库→题库)/SWE-rebench(21K 持续去污染)/
  R2E-Gym(DeepSWE 实证 59%)——RFT 扩容时取其一。v2/v3 门控库免费申请可解。
  跟踪索引：Post-Training-Data-Flywheel、mlabonne/llm-datasets
- **三阶段**：公开集 SFT(5-10K) → RFT 自产(1-2轮) → GRPO(SWE-Master/rLLM +
  SWE-rebench)；27B GRPO 在 4090 错峰窗口 rollout 是瓶颈，RFT≈80%收益，可停在 RFT
- **已落盘（2026-09-01，v4 终验完成，共 31GB）**：`/data/datasets/`（索引
  README.md，四能力轴组织）。核心增量：Nemotron code_v1.1 **496,206 条**（含
  reasoning 推理链字段，thinking 格式对口）+ interactive-agent **278,880 条** +
  SWE-Pivot 50,661 + Terminal-Pivot 31,111（均含 expected_action/answer，兼
  RFT 验证规范）+ CommitPackFT 286,790（12 语言）+ ToolACE 11,300 + FC-Pivot
  9,620。四轴**分开训分开门禁**。剩余缺口：FIM 自产、中文编程语料、SWE-rebench

### 9.6 缺口清单（2026-09-01 盘点）

**待下载权重**：① DFlash2 草稿 Q4_K_M 1.1GB ✅已落盘验证；② 官方 BF16 55.6GB
✅已落盘验证（52GiB/18 分片，/data/models/Qwen3.8-27B-BF16/，safetensors 头抽检
通过；基座对照推理需 TP3(GPU5-7)或 llama.cpp 部分卸载 503GB 主存跑评测）；
③ mmproj-BF16 931MB（视觉，待定）；④ Heretic-Ara HF 原始权重（源待找，对照后
决定）；⑤ DFlash2 HF 版+vLLM≥0.28 镜像（B线时）。训练数据已扩至 6 套
（见 9.5 落盘 v2）。

**待制备**：imatrix 校准语料 calib.txt（Magicoder 是 SFT 格式≠校准格式，
需转换+混自有代码）；SFT 数据对话化（chat+thinking+工具格式）。

**待构建**：llama-quantize/llama-imatrix（源码在，未构建）；
convert_lora_to_gguf 依赖（gguf 库）。

**待环境**：训练容器（torch≤2.8 + transformers/peft/trl/bitsandbytes，宿主无
torch）；B线 llm-compressor 容器；vLLM compose 骨架已有（/data/compose/vllm，
HF_ENDPOINT=hf-mirror + NCCL_P2P_DISABLE=1 已配）唯需钉 ≥0.28.0。

**待协调/决策**：GPU5-7 与 ZxBench 错峰窗口协议；基座选择（官方 vs
Heretic-Ara，对照后定，影响④）。

教师推理巧用：蒸馏教师=自家 4 副本服务，零额外显存。

### 9.7 4090 适配的量化技术矩阵与两线最终形态（2026-09-01）

母版唯一：官方 BF16 safetensors（GGUF 由 convert 生成，非下载项；sidecar 例外：
DFlash2 已下✅，mmproj 待定；现成量化版一律不自取）。

**4090 硬件事实**：无 FP4 TC（NVFP4/MXFP4 出局）；int8 TC 强（W4A8 独家甜点，
syv-ai +48%）；FP8 TC 半速（FP8 KV 有真硬件）；24GB→4-bit 家族唯一甜蜜点；
1008GB/s→decode 带宽受限（权重+KV 越小越快）；72MB L2→group 64-128 平衡点。

| 技术 | GGUF(A线) | safetensors(B线) |
|---|---|---|
| 域校准 | ✅ imatrix（激活感知简化版） | ✅ AutoRound 校准 |
| 学习旋转 SpinQuant | ❌ 工具链缺位 | ✅ **质量最大杠杆** |
| 学习取整 | imatrix 加权(RTN系) | ✅ AutoRound V/α/β |
| 逐模块混合精度 | ✅ --tensor-type-file | ✅ staging |
| 激活量化 | ❌ 无概念 | ✅ W4A8 双产物 |
| 非对称 KV | ✅ K q5_1+V q4_0 | FP8 KV |
| QAT | 缺位 | 天花板档(暂缓) |

最终形态：A线=域imatrix+敏感度分层+非对称KV+DFlash2（质量/显存优先）；
B线=SpinQuant+AutoRound W4A16+敏感层W8+W4A8双产物+FP8 KV+DFlash2 HF
（吞吐优先）。共享：同一校准集/敏感度分析/ZxBench 门禁；两线产物互为质量对照。
不做：NVFP4/MXFP4、FP8 全量权重、≤2bit、稀疏激活。

### 9.8 偏门优化（2026-09-01，均验证过本构建/主机支持）

主机资源：503GB RAM + 128 核（CPU 侧极富余）。核心判据：**张量每 token
读取字节数决定能否离开 GPU**（token_embd 行查找 ~3KB ✅；全量读权重/KV ❌，
PCIe 32GB/s 下权重整体卸载 ≈2 tok/s、热 KV 上 CPU ≈9 tok/s 封顶）。

- **① `-ot token_embd=CPU`**：嵌入挪主机内存，省 ~0.55GB/副本几乎无损，
  直接为 DFlash2/mmproj 腾位；变体：CPU 放 Q8_0 高精度嵌入（RAM 便宜）质量反升
- **② slot KV 跨副本迁移**（`POST /slots/:id_slot` save/restore）：LB 漂移时
  会话带 3.6GB KV 搬家（秒级），替代"丢缓存重 prefill 134s"
- **③ 副本异构舰队**：r0 长上下文 / r1 DFlash2 速度 / r2 视觉 / r3 均衡，
  LB 按特征路由
- ④ vLLM sleep mode：与 ZxBench 时间共享 GPU5-7（权重换出 503GB 主存）
- ⑤ CPU 夜间批量副本：128 核 CPU 推理 3-8 tok/s，消化低优先级队列
- ⑥ agent 会话自动压缩（200K→40K 摘要续跑，等效无限上下文）
- ⑦ `draft-dflash,ngram-cache` 组合草稿（复制粘贴+推理生成互补）
- ⑧ `nvidia-smi -pl 320` 限功率+锁频：批量 tok/J 与尾延迟
- 暂不值：P/D 分离、--rpc 借显存、-cmoe（dense）
- 坑：权重整体 CPU 卸载、热会话 KV 上 CPU（判据见上）

### 9.9 最新论文结合点（2026-09-01）

- **SWE-TRACE**（rubric 奖励塑形，arXiv 2604.14820）：给 RL 提供密集中间反馈
  替代稀疏 pass/fail——**最推荐**，纳入 9.5 RL 计划，教师=自家副本零成本
- **稀疏 KV 族**：Quest / RocketKV(NVlabs, 400×压缩 3.7×加速, 免训练) /
  DuoAttention——对症 195K 衰减；本模型仅 16/64 层注意力=只修"半边"；
  等 vLLM KVComp RFC 进主线，不自 fork
- **《Speculative Decoding for Batch Inference of LLM Agents》**（arXiv
  2608.24004）：agent 负载下草稿接受率行为——精读校准 DFlash2+ngram 调参
- Cassandra（同显存 1.81× vs EAGLE-3）：与 DFlash2 对照候选
- 多模态推测解码综述（preprints 202603.2344）：mmproj 上线后参考
- 架构观察线（选型用不回移植）：Gated DeltaNet-2(NVIDIA 2026, 解耦擦写)、
  InfiniteVL、Kimi Linear 3:1（验证本架构投资不贬值）；SINQ/GANQ 继续观察

### 9.10 剩余缺口审计（2026-09-01，按阻塞度）

1. 🔴 **评测尺子缺三根** → ✅ 材料已落盘（2026-09-01，/data/eval-rulers/：
   BFCL 33 文件 + SWE-bench-Lite 300 + HumanEval 164[FIM 底料]）；
   **污染对撞验证通过**（SWE-Gym×Lite 重叠=0）；剩执行项：ZxBench 接入三套
   评测并跑基线（GPU 窗口 1-2 天）
2. 🟡 基座决策未执行（BF16 已就位，阻塞 A 线与训练底座）
3. 🟡 数据格式统一管线（MEnvData→Qwen chat+工具格式）；thinking 格式决策
   （建议 FIM/工具轨非 thinking、代码推理轨 thinking，分开）
4. 🟡 训练容器未建（torch≤2.8 全套）
5. 🟢 RFT 沙箱并行容量未压测（SWE-Gym 每条带 docker_image）
6. 🟢 GPU 错峰协议、去重脚本等配套小件

## 10. 外部方案评审（2026-08-31，社区汇编资料逐项裁决）

对照本架构（qwen35 GDN、8×4090、4 副本 llama.cpp）的结论：

- 🚫 **照抄有害**：`--enable-prefix-caching`（GDN 不支持，需 syv-ai 补丁）；
  "vLLM≥0.27.1"（DFlash2 需 ≥0.28.0）；AWQ（AutoRound W4A16 已证更优）
- ✅ **真正新增可借鉴（3 项）**：
  1. NCCL 调优（vLLM TP 必测：`NCCL_P2P_DISABLE=1` vs 默认、`NCCL_ALGO=RING`，
     4090 无 NVLink、PCIe P2P 不稳）
  2. QLoRA 编程域特化 + vLLM multi-LoRA 热挂载（与推理线正交，超参起点
     lr 2e-4 / paged_adamw_8bit / bs1×ga8）
  3. 渐进压测方法论（128 路渐进，TTFT+吞吐双指标，ZxBench 即现成工具）
- ⚠️ **错位/存疑**：TriAttention 等 KV 压缩（qwen35 仅 16/64 层有 KV，另一半
  显存在 GDN 循环状态 0.88GiB/请求，token 淘汰不解决状态侧）；
  SINQ/GANQ/IQ1_S 极致量化（27B 无需塞、2-bit 有失控风险）；
  TurboQuant（无可复核来源）；PowerInfer（单用户场景不对口）
- 与已验证路线重合的项（FP8 KV、TP 策略、chunked prefill、CUDA Graph、
  MTP/推测解码）按本文档第 5 节执行即可，另注意 CUDA Graph 必须 PIECEWISE
  （syv-ai: FULL 捕获在 MTP+256K 特定 prompt 长度损坏输出）

## 11. 参考链接

- syv-ai/qwen38-27b-rtx3090 —— vLLM W4A16+int8 激活+DFlash2 单卡方案（18 条踩坑）
- r0b0tlab/qwen38-27b-nvfp4-sm121-vllm —— NVFP4+FP8 KV+四种草稿 profile
- alesha-pro/qwen38-27b-bench-4x3090 —— GGUF/引擎矩阵原始基准（引用单点数据前读 caveat）
- hf：Qwen/Qwen3.8-27B（官方卡）、unsloth/Qwen3.8-27B-GGUF（mmproj+全量化档）

### 9.11 自进化循环（参考 karpathy/autoresearch，2026-09-01）

autoresearch = 630 行/单 GPU/5 分钟迭代的 agent 自主研究循环。借鉴其编排模式，
不照搬其规模（27B 迭代为小时级）：

- **第一层（已有 80%）**：RFT 回路 = 自进化——自家副本生成 → ZxBench 过滤 →
  QLoRA 候选 → 四轴尺子门禁 → --lora 热挂灰度 → 监控/回滚 → 下一轮；
  autoresearch 补的是自动编排层（cron/CI 串起夜航循环）
- **第二层 代理模型经济学**：白天 0.5-1.5B 小模型跑 5-15 分钟级实验
  （配比/超参/FIM 策略/LoRA rank），赢家配置迁移到 27B 夜间训练窗口
- **第三层 agent 驱动规划**：自家模型读四轴评测结果→生成下一轮实验配置
  ——自托管模型改进自托管模型，闭环在机内
- **三安全阀**：①模型坍缩→每轮混入真实锚点数据（CommitPackFT/官方
  reasoning）+多样性监控；②尺子过拟合→四轴分轮换+永不入循环的人工验证集
  仲裁；③部署治理→自动循环只产候选，提升须人工晨审（夜航自进化/清晨晋升）


## 12. SGLang 可行性评估与 GPU 分工共存方案（2026-09-02）

> 背景：生产侧（llama.cpp 4 副本 + OpenResty LB）已稳定，RFT 回路的瓶颈从生成侧转移到
> **LoRA 候选评估链**（swift ckpt → GGUF 转换 → √r 补偿 → 双实例门禁）。vLLM 在 5.2 路线二
> 已评估（吞吐换运维，迁移成本高）；SGLang 的两个独有能力（原生 Multi-LoRA 热挂、
> RadixAttention 前缀 radix tree）是 vLLM 路线未覆盖的增量，单独立项评估。
> 数据可信度：本节显存账=一手推导（参数×字节数，无实测修正）；其余待 Step 1-4 实测回填。

### 12.1 显存账：形态判定（一手推导，27.3B 参数）

| 权重格式 | 体积 | 单卡 24.56G | TP2/卡 | 256K KV 余量 |
|---|---|---|---|---|
| BF16 | 54.6 GB | ❌ | **27.3G ❌ 连 TP2 都超 4090 上限** | — |
| FP8 (E4M3) | ~27.3 GB | ❌ | **13.65G ✅** | ✅ ~6G/卡（KV TP 分片后 256K 仅 ~2.3G/卡） |
| AWQ/GPTQ INT4 | ~16-18 GB | ⚠️ 勉强 | — | ⚠️ 256K 被挤到 ~128K，且需重量化 |
| **现役 llama.cpp iq4_xs** | **14.3 GB** | ✅ | — | ✅ 97% 满载实测（2.1/2.2 节） |

**判定**：SGLang 的量化生态（BF16/FP8/AWQ）没有 4.2bpw 级压缩，GGUF 支持不成熟 →
**SGLang 无法复刻单卡 256K 形态，不是生产替换项，是补位项**。

### 12.2 SGLang 独有价值（三块拼图）

1. **原生 Multi-LoRA 热挂**（`--enable-lora`，请求级 `lora_name` 路由）：
   RFT 候选评估跳过整条 GGUF 转换链（GDN out_proj 补丁 + `--lora-scaled :5.657` √r 补偿）。
   且**多候选可共存单进程**——一夜 N 个候选 LoRA 并行过门禁，自进化循环从串行变并行（见 12.4）
2. **RadixAttention**（进程内前缀 radix tree，自动增量 prefill + 跨请求共享 trunk）：
   注意边界——它是**进程内**机制，跨副本的会话粘滞路由仍然需要；OpenResty Lua LB 不白做，
   降级为通用多后端 LB（健康/漂移/池分流 + 粘滞），route.lua 增补 SGLang 后端探测
3. **FP8 TP2 + overlap scheduling + chunked prefill**：48 路并发聚合吞吐是现役唯一没摸到的
   指标（16 slot 排队兜底 ~420 tok/s，5.1 节）；MTP 层权重原生在 BF16 里
   （`mtp_num_hidden_layers=1`, `mtp_use_dedicated_embeddings=false`），SGLang MTP 路径若支持
   qwen3_5 混合架构可直接启用（接受率对照 llama.cpp 46.6% 实测基线）；
   结构化输出（xgrammar）对 xfc 工具轴有直接收益

### 12.3 GPU 分工共存方案（不是二选一）

| GPU | 用途 | 说明 |
|---|---|---|
| 0-3 | llama.cpp 4 副本（生产，不动） | iq4_xs + MTP + 256K，97% 显存，单流/长上下文/VRAM 天花板 |
| 4-5 | 门禁（2026-09-02 17:29 进行中，跑完释放） | A/B 双实例 |
| 6-7 | **SGLang FP8 TP2 单实例** | RFT 生成产能 + LoRA 候选评估 + 夜航自进化循环专用 |

FP8 TP2 每卡预算：权重 13.65G + 256K KV(TP 分片) 2.3G + GDN 态(分片) + 激活/CUDA ctx ~1G
≈ 18G/卡，余量 ~6G → 可跑 256K 但并发长上下文容量小于 llama.cpp 单副本（97% 满载），
定位是**隔离产能**不是替代产能。

### 12.4 RFT 回路重构（SGLang 引入后）

现行链（每候选）：生成(生产 LB) → 过滤 → QLoRA(GPU4-7) → **GGUF 转换(补丁)** →
**llama.cpp 双实例门禁(冷启动 ~50s×2 + 全轴 ruler)** → 晋级者热挂

SGLang 链（每候选）：生成(SGLang 专用实例) → 过滤 → QLoRA → **单进程多 LoRA A/B**
（base + N 个候选同进程，`lora_name` 请求级路由，一次 ruler 全候选全轴对比）→
**仅晋级者**转 GGUF 进 llama.cpp 生产（转换成本从每候选摊销到每夜一次）

收益：候选评估端到端从小时级（转换+双实例冷启动）降到分钟级；夜间窗口从 1 候选/夜
提升到 3-5 候选/夜（显存允许 3-5 个 r32 适配器共存，每 adapter ~0.4G）。

### 12.5 验证路线（按可信度排序，生死问题最便宜先做）

1. **Step 0**：等门禁 A/B 跑完（不占 GPU4/5，不碰生产）
2. **Step 1**：新容器（训练容器 cuda12.4 基底复用）装 SGLang，**首要验证 qwen3_5 混合 GDN
   架构是否在模型注册表**（`model_type=qwen3_5` / `Qwen3_5ForConditionalGeneration`）——
   这一步不过全案作废，成本最低先做
3. **Step 2**：BF16 原版 FP8 量化（compressed-tensors；**免 imatrix**——直接 16→8bit 无校准
   步骤，绕开现役 GGUF 量化 93 chunks imatrix 的已知弱点，2.2 节）
4. **Step 3**：GPU6/7 TP2 起服务，同套 ruler 跑基线（humaneval/xfc/gsm8k/ifeval/needle/tps）
   ——先证正确性（对照 12.7 基线表），再谈速度
5. **Step 4**：门禁完成后 A/B 吞吐：llama.cpp 4 副本 vs SGLang TP2×2（GPU4-5/6-7），
   16/48 路并发，三指标 TTFT + 聚合 tps + 前缀命中时延；MTP 接受率对照 46.6%
6. **产出**：仓库 `deploy/sglang/`（compose + 量化脚本 + route.lua v5 后端类型增补）+ 对照表
   （方法论同 3 节三仓库对照 + 7.9 渐进压测）

### 12.6 决策矩阵

| 场景 | 选择 | 理由 |
|---|---|---|
| 单流延迟 / 256K 长上下文 / VRAM 极限 | llama.cpp | iq4_xs 单卡 97%，SGLang 无对应形态 |
| RFT 候选评估 / 多 LoRA 并行门禁 | **SGLang** | 原生热挂，免转换链 |
| 高并发聚合吞吐 | 待 Step 4 实测 | 现役 420 tok/s 是唯一参照系 |
| 结构化输出（工具/agent 轴） | SGLang 优先 | xgrammar 硬约束，xfx 轴直接受益 |
| 前缀缓存（同进程内） | SGLang RadixAttention | 自动，免手工粘滞 |
| 前缀缓存（跨副本路由） | OpenResty LB 保留 | RadixAttention 不跨进程 |

### 12.7 待实测项（Step 1-4 完成后回填本节）

- [ ] qwen3_5 是否在 SGLang 模型注册表（生死项，Step 1）
- [ ] FP8 E4M3 量化后 ruler 全轴 vs 基线（Step 3，任何轴回退 >2pp 则该格式作废）
- [ ] SGLang MTP 路径对 qwen3_5 的支持与接受率（对照 46.6%）
- [ ] 48 路并发聚合 tps vs 420；TTFT vs 现役排队兜底
- [ ] RadixAttention 前缀命中 TTFT vs 粘滞 LB + `--cache-reuse 2048`
- [ ] Multi-LoRA 3-5 候选同进程显存账与 ruler 端到端时间
- [ ] 24h 稳定性（FP8 TP2 + 连续 RFT 负载）

### 12.8 不要做

- ❌ AWQ INT4 单卡挤 256K：重量化质量风险 + GDN 线性层支持不确定，双风险换无收益
  （FP8 TP2 严格更优且留 6G 余量）
- ❌ 为 SGLang 实验下生产副本（GPU0-3）：生产 SLA（48 路全成功）优先
- ❌ 跳过 Step 3 ruler 直接比吞吐：正确性未证前速度数据无意义（7.9 渐进压测原则）
- ❌ 用 SGLang 替代粘滞 LB 的跨副本路由：机制边界不同，见 12.2-2


## 13. 外部实证双参照：W4A16-vLLM 卡片 + 双 3080 SGLang 补丁（2026-09-02）

> 背景：§12 把 SGLang/vLLM 分支的生死项列为"qwen3_5 混合注意力是否在框架注册表"。本节两个
> 外部实证直接给出答案（vLLM 与 SGLang 双框架均可服务本架构），并各贡献一组可复用配方。
> 数据可信度：外部实测（他人硬件/量化），与我们硬件（4090 SM89）和现役（iq4_xs llama.cpp）
> 存在跨变量，仅作**形态判定与方向参考**，落地前须在我们 GPU6/7 复测（见 13.8）。

### 13.1 参照一：bowmanslayer/Qwen3.8-27B-Uncensored-W4A16-vision-mtp（vLLM 路线）

- **量化**：W4A16 GPTQ（AutoRound, group 128），**bf16 视觉塔完整保留**，MTP 头可选（独立 849MB `model-mtp.safetensors`）
- **部署**：2×RTX 3090 TP2 + **vLLM 0.20.2**，256K 上下文，权重 8.87G/卡，KV 余量 **12.93G/卡**（总池 415K token）
- **实测**：单流 thinking-on 66-68 tok/s；**16 并发聚合 ~700 tok/s**；prefill 峰值 3300 tok/s
- **量化配方（三点可偷）**：
  ① 量化前剥离视觉（`preprocessor_config.json` 量化期间须不在，否则 auto-round 静默切 MLLM 校准用错数据集）
  ② `linear_attn.in_proj_a/b` 排除量化（48 层 GDN 敏感层保精度）
  ③ MTP 头原样拷贝、绝不量化
- **评测方法论**：**ex-trunc（剔除截断）才是真实能力值**——thinking-ON 下 4096 token 采样预算在
  难题上常不够 `</think>` 闭合，8-17% 截断是采样预算伪影、非能力损失；"thinking ON vs OFF 不可跨表比较"
- **对我们的意义**：vLLM 分支生死项=已证可行；W4A16 TP2 比 §12 的 FP8 假设更优（KV 余量 12.9G vs 6G）；
  **700 vs 420 tok/s 是服务侧补位最强定量证据**（2×3090 干过我们 4×4090）；视觉恢复有现成验证路径

### 13.2 参照二：kk-pcl/sglang-qwen38-dual-3080-patch（SGLang 路线）

- **环境（锁定可复现）**：SGLang `0.5.19.dev228+g4cb5aebfe`（上游 commit `4cb5aebfe`）/ PyTorch 2.13.0+cu130 /
  FlashInfer 0.6.17 / Triton 3.7.1 / NCCL 2.29.7；2×RTX 3080 20GB（SM86）TP2 WSL2
- **权重**：主 `cyankiwi/Qwen3.8-27B-AWQ-INT4` + 草稿 `syvai/Qwen3.8-27B-DFlash2-W4A16`（~0.94G/卡）
- **三项优化**：
  ① DFlash2 `fc.weight`：每卡整份 `nn.Linear` → **TP 分片 `RowParallelLinear`**（改 `sglang/srt/models/dflash.py`）
  ② 主模型 KV：默认 FP16 → **逐头 Per-KV-Head FP8 E4M3 静态校准**（16 全注意力层，per-head scale
     比整层共用单 scale 更控长上下文误差）；DFlash2 草稿用更稳的逐层标量 FP8
  ③ SM<90 兼容：禁 Hopper 专用 symmetric-memory logits gather → 回退普通 TP NCCL all-gather
     （避 target verify CUDA Graph 阶段 `SIGFPE`）
- **实测**：KV 池 218,278 token（mem_fraction 0.90）；repeat 请求 `cached_tokens=64`；DFlash `accept rate`
  可观测；无 OOM / illegal-memory-access
- **作者机实测（含 CPU HiCache）**：GPU 上下文池 ~255K token；prefill <10K ≈ **1300-1400 tok/s**；
  decode 日常 **90+ tok/s**；**DFlash2 高接受时峰值 >200 tok/s**
- **关键坑**：**Mamba/GDN State 槽位独立于 attention KV 池**，`max-mamba-cache-size` 提到 20，
  避免"attention KV 还在但 State 被淘汰"触发整段 re-prefill
- **CPU HiCache**：被淘汰的前缀 KV 落 CPU 内存，减少重复 prefill（对 503GB 内存级主机是数量级收益）
- **工程**：`apply_patch.py --verify / --rollback` 自检回滚；离线测试覆盖 scale schema / TP 头切分 /
  KV write 形状；语义锚点非行号（源码漂移会停而非盲套）
- **风险**：静态校准非动态；scale 绑定 TP=2 与模型结构；**3080 无原生 FP8 张量核**（FP8 在此为
  容量/带宽优化，而 **4090 SM89 有原生 FP8 张量核 → 容量+计算双收益，比 3080 报告更优**）；
  SGLang 升级须先在副本环境 `apply_patch.py --verify`

### 13.3 三方对照（形态判定，跨变量仅作方向）

| 维度 | llama.cpp（现役） | vLLM W4A16（参照一） | SGLang AWQ+DFlash2（参照二） |
|---|---|---|---|
| 量化 | iq4_xs GGUF | W4A16 GPTQ g128 | AWQ-INT4 + DFlash2-W4A16 |
| 硬件 | 4×4090 四副本 | 2×3090 TP2 | 2×3080 20G TP2 |
| 投机解码 | MTP 46.6% | MTP 可选 | **DFlash2 峰值 200+** |
| KV | q4_0 | FP16 默认 | **逐头 FP8（静态校准）** |
| 256K | ✅ 97% 显存 | ✅ 12.9G KV 余量 | ✅ 255K 池 |
| 单流 | 72-80 tok/s | 66-68 tok/s | 90+ / 峰值 200+（3080） |
| 16 并发聚合 | ~420 tok/s | **~700 tok/s** | 未报 |
| Multi-LoRA | 手工 GGUF 链 | ✅ | **✅ 最强（RFT 主战场）** |
| CPU 卸载 | ❌ | ❌ | **✅ HiCache（503GB 内存）** |
| 视觉 | ❌ LM only | ✅ bf16 塔 | 未报 |
| GDN State | 固定 0.88G/seq | — | ✅ 可调池（坑③） |
| 外部依赖 | 无 | 无（下载即用） | 社区补丁（钉版可复现） |

### 13.4 对 §12 的修订

1. **生死项已闭合**：qwen3_5 混合注意力在 vLLM 与 SGLang **双框架**均实证可服务（不再待测）
2. **KV 压缩配方落地**：§12 假设的"FP8"细化为**逐头 Per-KV-Head FP8 E4M3 + 静态校准**
   （per-head 比整层 scale 更控长上下文误差），有现成 scale 文件与校准流程可参考
3. **投机解码双路径**：MTP（llama.cpp/vLLM 原生，免补丁）vs DFlash2（SGLang，需 fc.weight
   RowParallel 补丁，峰值更高，~0.94G/卡）
4. **GDN State 池独立 sizing**：`max-mamba-cache-size` 是一等公民资源（印证 §2 GDN 态固定成本
   分析，但 SGLang 侧可/须显式调，调太小触发整段 re-prefill）
5. **CPU HiCache × 503GB 内存**：大池问题的第三条解法（继"独占 slot"、"前缀 radix"之后），
   可能直接消解 r2/r3 独占 slot 设计（被淘汰前缀落 CPU 而非全量冷 prefill）

### 13.5 决策分叉（服务侧补位选哪条）

- **路线 A：vLLM + W4A16**——最简（下载即用免补丁）、聚合最高（700）、带视觉、MTP 免补丁。
  适合"直接补强生产 / 要视觉 / 要最高聚合"
- **路线 B：SGLang + AWQ + DFlash2 + HiCache**——单流潜力最高（200+）、CPU 卸载（503GB 内存独有）、
  Multi-LoRA 最强（RFT 主战场）、GDN State 可调。需社区补丁（钉版可复现）。
  适合"RFT 回路 + 长上下文 + 高并发"三合一
- **推荐：B 为主补位，A 为视觉/聚合备选**。理由：
  ① RFT multi-LoRA 是我们最重要的用例，SGLang 最强；
  ② 503GB 内存是只有 SGLang HiCache 能吃到的独有资产；
  ③ DFlash2 单流 200+（3080 实测）有望反超 llama.cpp MTP（4090 上 72-80），单流延迟不再是 llama.cpp 独占优势。
  A 留作"要视觉或要免补丁聚合"时的备胎。

### 13.6 修订后的快路径（GPU6/7，不碰生产/门禁，零自研量化）

1. clone 补丁测试环境（`scripts/create_tested_env.sh`：SGLang 0.5.19.dev228 钉版 + 锁定依赖）
2. 下载主 `cyankiwi/Qwen3.8-27B-AWQ-INT4` + 草稿 `syvai/Qwen3.8-27B-DFlash2-W4A16`（safetensors，非 GGUF）
3. `apply_patch.py`（逐头 FP8 KV + DFlash2 fc.weight）→ `--verify` → 离线测试
4. GPU6/7 TP2 启动：`max-mamba-cache-size=20` + CPU HiCache（吃 503GB）+ `mem_fraction 0.90`
5. 跑 ruler（humaneval/xfc/gsm8k/ifeval/needle/tps，thinking ON + **ex-trunc**），对照 llama.cpp 生产
6. 判定：质量 ≥ 生产 且（单流或聚合反超）→ 服务侧补位定 B，回填 §12.7 + 本节 13.8 待实测项

### 13.7 新增风险

- **社区补丁依赖**：非 SGLang 官方功能，钉 `4cb5aebfe` commit 复现；升级前副本环境 `apply_patch.py --verify`
- **静态校准非动态**：scale 绑定 TP=2 与模型结构，换拓扑/架构须重校
- **跨变量**：3080(SM86 无 FP8 核) vs 4090(SM89 有 FP8 核)、AWQ vs iq4_xs、3090 vs 4090——
  所有外部数字仅作方向，落地以我们 GPU6/7 复测为准
- **DFlash2 需 W4A16 safetensors**（syvai），我们手头是 GGUF Q4_K_M，须另下

### 13.8 待实测（在我们 4090 上，回填本节）

- [ ] SGLang 0.5.19.dev228 + 补丁在 4090 SM89 起服务（原生 FP8 张量核是否进一步加速）
- [ ] DFlash2 在 4090 的 accept rate 与峰值 tok/s（对照 llama.cpp MTP 46.6% / 72-80 tok/s）
- [ ] 逐头 FP8 KV 256K 下的 ruler 质量（长上下文误差是否可控）
- [ ] CPU HiCache 在 503GB 内存下的 re-prefill 消解率（大池问题的第三条解法）
- [ ] `max-mamba-cache-size` 的 GDN State 驱逐阈值实测（调太小是否触发整段 re-prefill）
- [ ] Multi-LoRA 3-5 候选同进程（RFT 回路重构，§12.4）
- [ ] 16/48 并发聚合 vs 现役 420 tok/s
- [ ] 对照 vLLM W4A16（路线 A）同条件聚合，裁决 A/B 主次
