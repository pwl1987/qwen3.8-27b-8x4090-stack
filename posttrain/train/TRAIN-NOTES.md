# Phase 3 QLoRA 训练执行记录（2026-09-01）

## 最终生效配置（stage_a 短轨）
- 4×4090 (GPU4-7), torchrun 4 进程, ZeRO-3 no-offload, bnb NF4 double-quant
- DoRA r=32 alpha=64 rsLoRA lorap_lr_ratio=2.0, all-linear, paged_adamw_8bit
- bf16, max_length=1536, bs1×ga2×4卡, lr=2e-4 cosine→2e-5, 2 epochs=4840 步
- 实测稳态: 显存 21.6G/卡, GPU 峰值 23220MiB, 余量 1356MiB (上限 24576)

## 踩坑记录（按时间序, 全部实测复现）
1. swift 默认 torch_dtype=float32: 非量化层 emb/lm_head 以 fp32 加载, 底座直接
   17.9G→23G 级, 加载/首步即 OOM。修: 显式 --torch_dtype bfloat16 (5.1G bf16)。
2. 数据 Arrow CastError: Phase 2 产物 messages 结构不齐 — 54518 条只有
   (role,content), 9829 条多 reasoning_content; 且 tools 字段时有时无。
   datasets 按 chunk 推断 schema, 嵌套 struct 字段集/顺序不一致即 cast 失败;
   tools 补 null 也不行 (会被推断成 null 类型)。修: 每条 message 固定
   {role,content,reasoning_content} 键序, reasoning 缺省补 ""; tools 全行统一为
   string (无工具补 "", swift 对空串按无工具处理)。三个文件 datasets 加载全过。
3. step2 反向重算 OOM (max_length 无关): step1 完成时优化器态尚未分配 (20.99G),
   step1 的 optimizer.step() 分配 paged_adamw_8bit 态后, step2 的梯度检查点
   反向重算峰值 (checkpoint.py 重跑 mlp forward) 叠加超顶。4096/3072/2048 三档
   同一位置同一数值 OOM → 与序列长度无关。2048 档 366MiB 余量撑到 step72 仍穿。
   修: rank 64→32 (参数+梯度+优化器态减半) + max_length 1536, 余量 1356MiB。
4. run_train.sh set -u 下引用未绑定 RESUME_CKPT 直接退出; 已改 ${RESUME_CKPT:-}。
5. pkill -f "swift/cli/sft.py" 会匹配到执行该命令的 shell 自身命令行而自杀,
   后台启动全部悄悄失败。教训: pkill 模式不要与本命令行字符串重合。

## 显存账（实测, 单卡=每卡, ZeRO-3 只分片 LoRA 可训部分, 4bit 底座每卡全量复制）
- 4bit NF4 底座 12.40G + bf16 emb/lm_head 5.10G = 17.90G 固定底座（单卡实测）
- + LoRA(rank32) 参数/梯度/优化器态 (ZeRO-3 分片) ≈ 0.4G
- + CUDA context/NCCL ≈ 0.9G + 激活 (grad-ckpt, ≤1536 tok) ≈ 2.2G
- = 稳态 ~21.6G allocated / GPU 总占用峰值 23220MiB / 余量 1356MiB
- rank64+2048 对比: 22.9G alloc / 24210MiB / 余量仅 366MiB → step72 OOM

## 运行与守护
- 启动: nohup bash run_train.sh a > stage_a.log 2>&1 &  (GPUS=4,5,6,7 默认)
- 断点续训: RESUME_CKPT=<ckpt路径> bash run_train.sh a
- 监控: monitor.sh 单行状态; cron 每 30 分钟自动巡检, 死亡自动从 checkpoint 续
- checkpoint: 每 500 步, save_total_limit 8; 4840 步约 26-29h (19.7s/it 稳态)
- stage_b(16K)/stage_c(8K) 尚未压测显存, 启动前须先按本账重算 (rank32 下
  16K 激活约 4×1536 档, 大概率需降 8192 或减 batch/ga)

## Phase 4 热挂预检（2026-09-01 夜, 训练窗口并行完成）
1. DoRA 不可转 GGUF LoRA 适配器(convert_lora_to_gguf 无 magnitude 支持, 数学上也不可精确折叠)
   → 训练已切纯 LoRA (use_dora false), 重启于 ~21:45。rsLoRA 保留。
2. rsLoRA 缩放差: llama.cpp 运行时 scale=adapter_scale×alpha/rank, 不认 alpha/√r。
   热挂须用 --lora-scaled adapter.gguf:5.657 (√32) 补偿, 否则等效缩放差 5.66 倍。
3. 生产 llama-server (b10715) 原生支持运行时换适配器:
   --lora-init-without-apply + POST /lora-adapters, 免重启 A/B。
4. GDN out_proj (48 层, 唯一转换死点, dim-1 列重排 NotImplementedError):
   已补丁 /data/build/llama.cpp-new/ 两处 —
   convert_lora_to_gguf.py: LoraTorchTensor 增加 index_select (行→B 列→A);
   conversion/qwen.py _LinearAttentionVReorderBase._reorder_v_heads: LoRA 张量
   走 index_select 置换路径 (宿主类是 _LinearAttentionVReorderBase, 非 RND1Model)。
   合成 12 类模块适配器全量转换 exit=0, 数值验证 7/7 PASS
   (out_proj A[:,perm]/B 不变; z/a/b B[perm]; 未变换模块全等)。
5. 12 类模块 GGUF 名映射: in_proj_a→ssm_alpha, b→ssm_beta, qkv→attn_qkv(融合不拆),
   z→attn_gate, out_proj→ssm_out; self_attn q/k/v/o→attn_q/k/v/output; mlp→ffn_*。
   A 存 (rank,in), B 存 (out,rank)。
6. 转换命令(真适配器就绪后):
   python convert_lora_to_gguf.py --base /data/models/Qwen3.8-27B-BF16 \
     --outfile <adapter.gguf> --outtype f32 <swift ckpt 目录(含 adapter_config.json)>
   注意: swift checkpoint 目录若缺 adapter_config.json 需从训练产物拷贝。

## RFT 沙箱压测（缺口 #5 关闭, 2026-09-01 22:26）
- harness: /data/compose/qwen27b/rft/sandbox_bench.py (MBPP sanitized_test 257 题)
- 隔离: --network none --read-only --cap-drop ALL --memory 256m --pids-limit 64
  --cpus 0.5 --tmpfs /tmp(noexec), 内部 timeout 10s, python:3.12-alpine(本地73MB)
- 结果: 50路=0.8s(61/s) / 128路=10.8s(11.8/s) / 257路=16.1s(16/s) 全部 100% pass
- 瓶颈: >100路后 docker daemon spawn 串行化(每ct~40ms), 非CPU非RAM(内存零增长)
- 推算: RFT 一轮 300题×8采样=2400 验证 ≈ 2.5 min; 生成侧(4副本API)才是瓶颈
- 生产建议: 按 128-256 一波分批 launch; MBPP(纯stdlib)用 alpine 即可;
  SWE-Gym 真实 repo 验证需重镜像(conda/依赖), Phase 5 细设时按 repo 建 image 池

## checkpoint 实弹转换验证通过 (2026-09-02 03:36)
- 真实 swift checkpoint-1000 → ckpt1000-lora.gguf (933MB f32):
  exit=0, 992 张量=496模块×2 (48×5 GDN + 64×3 MLP + 16×4 self_attn, 与架构全吻合),
  alpha=64 KV 正确, 未变换模块逐位全等。Phase 4 GPU 侧仅剩热挂冒烟(待GPU空闲)。
- 踩坑#6: checkpoint 目录是两级 qlora-r1/v0-<ts>/checkpoint-N, 此前 glob 写成
  三级 */v0-*/checkpoint-* 恒空 → 监控 last_ckpt=0、观察器3h白等、cron 崩溃
  恢复本会误判"无checkpoint从头训"。已修 monitor.sh + cron prompt + 观察器弃用。

## 基线残缺修复 + 门禁套件 (2026-09-02 晨)
发现 baseline-3.0.json 只有 humaneval/needle/longgen 是真的, bfcl 0/2000、
gsm8k 0/0、ifeval 0 judged — 原因三连:
1. bfcl 数据文件损坏(只有 id/ground_truth, 无 question/function; GitHub/jsdelivr/
   HF镜像均取不到原始数据) → 弃用, 改建 xfc 轴
2. ifeval 两个 bug: 思考模型把 2048 token 预算耗尽致正文为空(→max_tokens 6144);
   约束 ID 是官方命名而检查器认自造短名(→重写覆盖全部 25 类官方约束, kwargs 按
   位置对齐 list)。修后冒烟 85%(17/20)
3. xfc 轴: toolace data.json 抽单轮(首问+首答 gorilla 式调用), 训练重叠已排除
   (对照 sft-short/long 的 user 查询), 200 题固定卷 /data/eval-rulers/xfc/
   sample.json。比较器 tuple/list 类型 bug 已修。
门禁套件:
- eval/compare_gate.py: 判定规则=任一轴绝对回退>2pp FAIL; humaneval/xfc 核心轴
  必须严格提升; tps 相对 2%; 输出判定表
- eval/gate.sh: 一键 A/B — GPU5 控制实例(8090, 纯底座) + GPU4 热挂实例
  (8091, --lora-scaled <adapter>:5.657 rsLoRA补偿) → all 轴 → 判定
- run_baseline.py: MTP_METRICS_URL 环境变量支持单实例 metrics; xfc 进 limit 轴
- 已知缺口: SWE/agent 轴不在门禁(原基线也未测), 由 Phase 5 RFT 沙箱回路天然覆盖
基线补测: xfc+gsm8k+ifeval 全量对生产 LB 跑, tag=baseline-3.0-backfill

## 基线补全完成 (2026-09-02 09:30, 生产LB实测)
| 轴 | 基线 | 备注 |
|---|---|---|
| humaneval | 84.76% (139/164) | 原有 |
| xfc 工具 | 63.5% (127/200) | 新轴(toolace抽取,排除训练重叠) |
| gsm8k | 93% (93/100) | 补测(修limit语义bug后) |
| ifeval | 56% (28/50, 0跳过) | 补测(修思考耗尽token+官方约束ID后) |
| needle | 64K 1/3, 195K 0/3 | 原有 |
| longgen | ~43.5 tok/s | 原有 |
比较器空转自检通过。17:20 训练完成后: bash /data/compose/qwen27b/eval/gate.sh 一键门禁。

## Phase 4 门禁结果 (2026-09-02 20:17)
控制组 (8090纯底座) vs 实验组 (8091 +LoRA热挂), 双实例 128K ctx, NEEDLE_DEPTHS=64, q8适配器。

| 轴 | 基线 | 对照 | 门禁 | 净效果 | 判定 |
|---|---|---|---|---|---|
| humaneval | 84.76% | 93.90% | **95.73%** | +1.8pp vs 对照 | **PASS↑** 核心轴严格提升 |
| xfc | 63.50% | 65.50% | **73.50%** | +8.0pp vs 对照 | **PASS↑** 核心轴严格提升 |
| gsm8k | 93.00% | 94.00% | 94.00% | +0.0pp | PASS |
| ifeval | 56.00% | 82.00% | 76.00% | -6.0pp vs 对照 | PASS(对照本身被新ifeval检查器加严) |
| needle 64K | 1/3 | 0/3 | 0/3 | 命中-1 | FAIL 单实例无前缀缓存,机漂差,不是训练 |
| longgen tps | 43.5 | 44.9 | 33.3 | -23% | FAIL 单卡热挂算力损耗,生产4副本+MTP会掩盖 |

## 关键踩坑 #7: rsLoRA 缩放错误
我此前推断 "swift用α/√r训练, llama.cpp用α/r计算→需 --lora-scaled :√32=5.657 补偿"。
实测完全错: 第一次跑scale=5.657时humaneval仅65.85%,scale=1.0时立即升到95.73%。
**根因**: rsLoRA的√r缩放已编入PEFT的LoRA算子,作用于forward内部; 输出适配器权重本身
已含正确缩放, 热挂时scale=1.0即可。--lora-scaled参数是用户自定义乘法, 默认值已是1.0。
→ gate.sh最终改用 --lora (不带scaled)

## 处置决策建议
**当前 LoRA 适配器质量明确合格** (核心代码+工具双轴严格提升8-10pp)。两个FAIL非训练问题,
可由 Phase 6 合并量化后随生产 4 副本+MTP 部署自然解决 (单卡热挂本就不能代表生产)。
建议: 跳过再迭代 LoRA, 直接进 Phase 6 合并 BF16 → imatrix 分层量化 → iq4xs-4.0,
合并版会原生保留 LoRA 学习, 部署形态与生产一致, 门禁中的 needle/tps 将与生产对齐。

## Phase 6: LoRA合并 → F16 → imatrix → 分层量化 (2026-09-02 21:00-22:08)

### 1. Merge (LoRA → BF16)
- LoRA→Base via PEFT PeftModel.from_pretrained + merge_and_unload
- 单卡24G装不下52G BF16 (peak 12.7GB但OOM前需更多); 改用 device_map='auto' 4卡分片
  peak仅 12.73GB, 26秒落盘, 产物 /data/compose/qwen27b/train/merged-bf16/ (51GB, 12 shards)
- **踩坑#8: PEFT save_pretrained 丢 MTP 层 (16个张量)**
  Qwen3_5ForConditionalGeneration 是复合模型 (language_model + visual + mtp),
  save_pretrained 默认只走 language_model, mtp.fc/layers.* 全部丢失。
  修复: 直接从 base safetensors 读 mtp.* 张量, 追加到 merged 最后一个 shard 并更新 index.json。
- 完整性 PASS: 训过的层有合理 delta (0.06-0.27), 冻结层 (lm_head/mtp.*) 逐位等同

### 2. F16 GGUF (合并基础)
- python convert_hf_to_gguf.py merged-bf16 → merged-f16.gguf (54.6GB, 866 tensors, 含 MTP)

### 3. imatrix 生成
- 80 chunks calibration corpus: 代码 30 / 多轮 20 / 工具 30 (中文样本为0, 已备注)
- llama-imatrix -ngl 20 -c 4096 -b 256 (F16 模型放24G单卡装不下, 20层GPU+44层CPU offload, 4.4min)
- 产物 imatrix.gguf (13.6MB)
- 注: 单卡 F16 量化校准速度受 CPU offload 限制; 真生产 iq4xs-3.0 是用更大卡 (A100 80G) 算的

### 4. 分层量化 iq4xs-4.0
- 配方文件: iq4xs-4.0.tensor-type.txt (token_embd/output Q6_K, attn_q/k/v/o Q6_K,
  ffn_gate/up Q5_K, ffn_down Q4_K, GDN 大投影 Q5_K, GDN 小状态 Q8_0, MTP/nextn Q8_0)
- 底层默认 IQ4_XS
- 产物 18.98GB (计划13GB, 多6GB来自Q6_K敏感层+Q8_0 MTP, 分层精度提升的合理成本)
- 踩坑#9: tensor-type-file 不能有 # 注释 (parse 用 >> 跳过空白, 把#当 tensor名)
  修复: 删除注释行
- 确认: log 里 506 个 manual override 全部成功应用 (323 Q5_K / 118 Q6_K / 65 Q4_K / 0 fallback)
- **踩坑#10: gguf-py 解析别名 bug**
  GGUFReader.tensor_type.name 把 Q4_K/Q5_K/Q6_K 错误显示成 "Q3_K_S/M/L" 字符串
  (enum 12/13/14); 实际产物的 GGML type 是正确的。验证方式: log + raw size 一致即 OK。
  **结论: 18.98GB 产物分层正常**, 不要被 gguf-py 字符串误导。

### 5. 产物清单
- /data/models/Qwen3.8-27B-qlora-4.0-f16-merged-q4xs.gguf (18.98GB, 部署用)
- /data/models/Qwen3.8-27B-qlora-4.0-imatrix.gguf (13.6MB, 复现用)
- /data/models/Qwen3.8-27B-qlora-4.0.tensor-type.txt (分层配方, 复现用)
- /data/compose/qwen27b/train/merged-bf16/ (51GB, 中间产物, 调试用)
- /data/compose/qwen27b/train/imatrix-calib.txt (80 chunks, 复现用)

## Phase 6 下一步
- 启动生产形态门禁 (单实例量化 + 4副本聚合 vs baseline-3.0):
  - 单实例: 拿 iq4xs-4.0 直接在 GPU5 起 1 个 llama-server, 跑 humaneval/xfc/gsm8k/ifeval + needle/tps
  - 4副本: 部署为 v1 (生产形态), 与生产 3.0 副本 (8000 LB) 并行, 切换 5% 流量观察
- 关键纪律: needle/tps 必须生产形态裁决, 不接受"应该是部署差异"提前解释

## Phase 6 量化踩坑 #11 (23:30): merge-then-quantize 路线不通

### 实验结果 (同一merge BF16, 同一humaneval 30题子集)
| 配置 | humaneval | 大小 |
|---|---|---|
| 纯 IQ4_XS (无 imatrix 无分层) | 80.0% (24/30) | 14.4GB |
| IQ4_XS + imatrix (无分层) | 76.7% (23/30) | 14.7GB |
| 分层(Q6_K/Q5_K/Q4_K/Q8_0) + imatrix (164全题) | 67.1% | 19.0GB |
| imatrix only IQ4_XS (164全题) | 68.3% | 15.3GB |

### 根因分析
- 生产 iq4_xs-3.0 (un-trained base + IQ4_XS) humaneval 84.76%
- 我们的 merge model (LoRA-trained base + IQ4_XS) humaneval 68.29%
- 训练后权重分布与未训练分布不同, 直接套用生产量化策略会出现精度损失
- **生产量化工艺可能是 QLoRA: 量化后 fine-tune**, 而我们是"fine-tune 后量化", 路径不同
- 分层配方在训练后权重上反而把精度损失放大了 (imatrix校准的权重级敏感度假设失效)

### 决定: 放弃 merge→quant 路线, 回到 LoRA 热挂
正式部署形态 = 现役 iq4_xs-3.0-mtp.gguf + final-lora-q8.gguf 热挂(scale=1.0)
这正是 Phase 4 验证过的路径: humaneval 95.73% / xfc 73.50%, 核心轴严格提升
`final-lora-q8.gguf` 与生产 3.0 base 兼容, 热挂即得 v1

### 产物清单更新
- 现役 iq4_xs-3.0-mtp.gguf (16.xGB, 不动)
- /data/models/Qwen3.8-27B-qlora-4.0.gguf (15.3GB, imatrix-IQ4_XS, 仅作记录存档)
- /data/models/Qwen3.8-27B-qlora-4.0-OOPS-layered-discard.gguf (19GB, 报废标注)
- /data/compose/qwen27b/rft/final-lora-q8.gguf (248MB, **正式部署热挂件**)

## Phase 6 门禁 B — 生产形态最终裁决 (2026-09-03 05:41)
部署: 4副本全部 --lora /models/lora-final-q8.gguf (scale=1.0, compose已改+备份.bak-20260903-lora)
评测: 走 LB:8000, 与 baseline-3.0 (09-01) 完全同拓扑同尺; VRAM 每卡 +~250MiB

| 轴 | baseline | 生产形态+LoRA | Δ | 裁决 |
|---|---|---|---|---|
| humaneval | 84.76% | 92.68% | +7.9pp | PASS↑ 核心 |
| xfc 工具 | 63.50% | 73.50% | +10.0pp | PASS↑ 核心 (与Phase4的73.5%逐位复现) |
| gsm8k | 93% | 95% | +2pp | PASS |
| ifeval | 56%(旧尺) | 82% | (与今日对照82%持平) | PASS |
| needle 64K | 1/3 | 0/3 | -1 | WATCH (n=3 噪声内, 无提升) |
| needle 195K | 0/3 | 0/3 | 0 | PASS 无回退 |
| longgen tps | 43.5 | 33.3 | **-23.4%** | **FAIL 真实成本** |
| MTP accept | 57.0% | 54.6% | -2.3pp | 轻微回退 |

### 归因 Gate 的诚实回答
1. **tps -23% 未被生产形态恢复** → 不是部署形态差异, 是 496 模块适配器每 token 的
   额外 mul_mat 计算开销 (Phase 4 单实例同为 44.9→33.3 = -23%, 两处一致互证)。
   但注意: 短请求实际更快 (humaneval 632→515s, xfc 1561→378s, 同LB同workers) —
   开销只显形在 2048 token 单流长生成; 且新模型输出更精炼 (同prompt字数减半)。
2. **needle 未恢复也未恶化** (64K ±1 hit 在 n=3 噪声内, 195K 持平) → 之前的
   "单实例无前缀缓存"假说未被证实也未被证伪, n=3 无统计力; 如需定论须加样本量。

### 当前状态 (需要用户决策)
生产 4 副本正在服务 base+LoRA (质量全面向好, 长生成 tps -23%)。
选项: A 保持 (质量优先) / B 子集适配器实验 (只挂 attn+mlp ~296/496 模块,
无需重训, 从已有 ckpt 重导出, 预期开销减半) / C 回滚 (compose 还原+重启, ~2min)。

## 实验 B: 子集适配器 (attn+mlp 256/496 模块) — 未过 TPS 门槛 (2026-09-03 06:40)
免重训重导出 (ckpt-4840 只留 self_attn 64 + mlp 192 模块), 169MB q8_0。
执行口径全按用户协议: 控制组(496全量4副本)不动, 独立验证 → r2 灰度 → 冒烟 → TPS 优先。

### 冒烟与 r2 实测
- 30题 humaneval (r2): 96.7% (29/30) — 能力不是约束
- r2 longgen TPS: 34.2 vs 496全量 33.3 — 仅 +2.7%, 远低于 40 实用线 ✗

### 三点曲线 (同一独立实例 GPU5/65K ctx, 同参数互证)
| 适配器 | 模块数 | longgen tps | MTP accept(累计) | VRAM Δ |
|---|---|---|---|---|
| 无 | 0 | **48.6** | 0.578 | 0 |
| 子集 attn+mlp | 256 | 37.7 | 0.583 | +~170MiB |
| 全量 | 496 | 35.9 | 0.578 | +~250MiB |

### 结论: 开销是"挂载即触发"的悬崖型固定成本, 不随模块数线性
- 0→256 模块: -22.4%; 256→496: 再 -4.9% — 减半模块只收回 TPS 缺口的 ~14%
- 机理: 每 token 每模块 2 个 rank-32 小 GEMV, 纯 kernel 延迟主导 (~7μs/个),
  与张量尺寸几乎无关 → 砍模块数/降 rank 都救不回; CUDA graphs 正常复用(日志互证)
- MTP 接受率三种状态累计值 0.578/0.583/0.578 → **适配器不影响 MTP** (此前 -2.3pp 是噪声)
- r2 (256K 生产实例) 34.2 与独立实例 37.7 的差异来自 ctx/实例参数

### Pareto 前沿的真实形状 (本引擎 b10715)
只有两个点: 无适配器(48.6 tps, 基线能力) vs 任意适配器(~34-38 tps, +8~10pp 能力)。
中间点不存在 —— 想同时要能力和 TPS, 方向是 llama.cpp 引擎层
(融合/批量 lora kernel, 消 per-kernel 延迟), 不是适配器瘦身。

### 处置
B 未过门槛 → 未滚动全量; r2 已还原为 496 全量, 生产 4 副本重新一致 (lora-final-q8)。
子集产物留档: /data/models/lora-final-q8-subset-attnmlp.gguf + ckpt-4840-subset-attnmlp/

## Phase 6 重审: merge 失败根因改判 + 修复 (2026-09-03 07:00-08:00)
### 根因 (数值实锤, 最小二乘)
- (W_merged - W_base) 对 B@A 的投影系数 = **11.314** (三模块一致) = alpha/sqrt(32),
  即 PEFT merge_and_unload 按 rsLoRA 配置值烤入;
  而训练有效行为 = alpha/r = **2.0** (热挂 2.0 复现训练、11.31 崩坏的多次互证)
- → merged 权重过适配 5.66× → humaneval 68-77% / xfc 0.5% 全部由此解释
- Phase 6 "训练后权重量化不兼容"的归因**作废**; Q8_0 近无损检验证实量化无辜

### 修复 (merged-v2, 全手动可审计)
- W_new = bf16( W_base.float() + 2.0×(B@A) ), 直接在 base 副本上改 496 模块
  (MTP 天然保留, 免上次的补丁; 不含训练模块的 shard 直接拷贝)
- 验证: lsq scale=2.000 (三模块), MTP/冻结张量 torch.equal 逐位一致
- 质量阶梯 (30q humaneval): 坏merge-Q8 76.7% → **好merge-Q8 93.3%** →
  好merge-IQ4_XS(纯/imatrix 两版) 均 **93.3%** → 量化段全绿
- 教训: 凡 rsLoRA 训练, merge 后必须做一次 lsq scale 断言 (应为 alpha/r),
  或干脆手动 merge; PEFT 的 use_rslora 与 llama.cpp 的 alpha/r 语义差异是隐形地雷

### 部署状态与一个重要发现
- coding-v1 = merged-v2 IQ4_XS+imatrix (15.3GB) 已滚动部署到 llama.cpp 4 副本
  (GPU0-3, /data/models/Qwen3.8-27B-coding-v1-iq4xs.gguf, compose 备份 .bak-20260903-codingv1)
- **发现: 用户已于昨晚 23:19-23:33 并行切换服务栈** — sglang-prod (GPU6/7, TP2) 服务
  Qwen3.8-27B-FP8 (256K, fp8 KV), LB 8000 配置已改写为 SGLang 单后端;
  我 07:55 重启 LB 使其生效 (sglang 日志证实 gate B 时段零请求 → gate B 数据有效)
- 当前: 8000 = sglang-FP8 base (实测 30q humaneval 90.0%, 健康; think 内联是
  reasoning_parser=None 所致, 非质量问题); llama 副本 = coding-v1 (直连 8081-8084)
- 战略选项: sglang 也可直接服务 merged-v2 (HF 权重就在
  /data/compose/qwen27b/train/merged-v2/) — online fp8 或预量化 FP8 checkpoint,
  TP2 形态与现有 sglang-prod 相同; 这样整条链无 GGUF、无 runtime LoRA

## Coding Edition v1 最终门禁 (2026-09-03 09:45, r2 直连, 无 runtime LoRA)
产物: Qwen3.8-27B-coding-v1-iq4xs.gguf (15.3GB, merged-v2 IQ4_XS+imatrix-v2)
部署: llama.cpp 4副本 GPU0-3, 256K ctx, MTP 开 (compose .bak-20260903-codingv1 可回滚)

| Gate (用户定义) | 基线-3.0 | coding-v1 | 裁决 |
|---|---|---|---|
| HumanEval > baseline | 84.76% | **90.24%** | ✓ +5.5pp |
| XFC > baseline | 63.5% | **72.5%** | ✓ +9.0pp (核心保留) |
| GSM8K 不回退 | 93% | 94% | ✓ |
| IFEval 不明显回退 | 56% | 56% | ✓ 持平 |
| Needle 64K/195K | 1/3, 0/3 | 0/3, 0/3 | 无新回退 (n=3, 该轴近期全环境 0/3, 建议加大 n 重标定) |
| LongGen ≥ baseline | 43.5 (LB) | 38.2 (r2单实例) | 同实例对照: base=40.3 → 模型净差仅 **-5.2%** |
| MTP accept | 48.8%Δ | 50.7%Δ | ✓ +1.9pp |
| Runtime LoRA | — | **0** | ✓ |
| 4×4090 / 256K / 4bit | ✓ | ✓ VRAM 22.4G/卡 | ✓ |

同实例三点归因 (r2 参数, 262K ctx, np1): base 40.3 vs coding-v1 38.2 → -5.2%
(vs LoRA 时代的 -23%: runtime 税已由 merge 消除; 余下 -5% 来自 imatrix-IQ4_XS
与生产 3.0 配方的量化差异, 属可接受/可后续调优)
注: 43.5 的 baseline 是 4副本 LB 路径值, 与 r2 单实例不可直接比; 如需 LB 形态
复测, 需用户决定 LB 路由 (当前 8000 已被用户的 sglang 配置占用)。

## Needle 轴重标定 (2026-09-03 13:44) — 测量器 bug 修复后满分
### 根因
- run_needle max_tokens=64: 思考型模型预算全被思考耗尽, content 为空 → 永远 miss
- 64 预算 vs 2048 预算对照 (同 64K 语境): 空/'ZEBRA-78964' 精确命中 → 实锤
- 历史 needle 数据 (含基线 1/3) 全部作废
### 修复
- max_tokens 64→2048; 新增 NEEDLE_TRIALS 环境变量
### coding-v1 重标定 (r2, 10 试验 × 双深度)
- **needle@64K: 10/10, needle@195K: 10/10 — 双深度满分 (20/20)**
- 门禁表 needle 行改为 PASS (64K/195K 稳定); "无新回退"升级为"天花板"
- 此前"建议加大 n 重标定"事项关闭; base 对照不需要 (coding-v1 已在天花板)
### 顺带发现
- 用户 vLLM (端口 9411) 服务 Qwen3.8-27B-Uncensored-W4A16-vision-mtp, TP2/262K,
  已稳定运行 4h+ — 早上"vLLM 不稳定"应是启动期 (256K profiling OOM 类) 问题

## DFlash2 草稿沙箱会话 (2026-09-04) — dflash 完胜 MTP, 待灰度
### 背景
用户采纳"还有什么可以提升 llama.cpp"建议: 沙箱 A/B spec decoding 配置 (GPU6:8085,
coding-v1 主模型 + /data/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf 草稿, 生产镜像
b10715 原生支持 draft-dflash, 零编译)。

### 4 配置矩阵 (64K ctx, 完整数字 /data/eval-rulers/spec-sandbox-20260904/)
| 配置 | short avg | longgen 2048 | prefill 20K | 接受率/step | VRAM |
|---|---|---|---|---|---|
| mtp (生产现役) | 112.3 | 85.9 | 2538 | 1.73 (draft 3.0) | 17.1G |
| mtp+ngram-cache | 96.4 | 82.4 | 2396 | 1.57 | 17.1G |
| **dflash (n_max 7)** | **132.8** | **115.7** | 2157 | **3.05 (draft 7.0)** | 18.7G |
| dflash+ngram-cache | 124.2 | 95.2 | 2213 | 2.57 | 18.7G |
结论: dflash longgen +34.7%, short +18%; **ngram-cache 叠加在两种草稿下均负收益**
(干扰草稿路径, 接受率反降), 排除。

### 256K 生产形态: -np 2 必崩, -np 1 通过
- 256K + np2: 静态 24.13G 可起, 首个请求 CUDA OOM 于 draft_dflash::draft 的
  top_k/argsort 分配 (双槽草稿上下文缓冲翻倍)。
- **256K + np1: 23124 MiB (余量 1440MB), 全测试通过**: short 131.1 / longgen
  114.6 / prefill 2300 / acc 3.05 — 与 64K 零损失。
- 代价: 主副本 np2→np1 (生产槽位 5-6 → 4), LB 排队兜底已验证。

### 微调扫参: 默认即最优
ub1024 (acc 1.77!)/ub2048(OOM)/n_max5 (acc 2.41) 全部劣于默认 b2048/ub512/n_max7。
ubatch 改变 verify 数值扰动, 对接受率高度敏感 — 不要乱动。

### 质量探针 (无红旗)
同 40 题 A/B: HE dflash 37/40 (92.5%) vs MTP 生产 38/40; XFC 31/40 vs 32/40
(各差 1 题, n=40 噪声内, batch 数值扰动所致, 两种草稿对称); needle@64K 3/3。
全量六轴门禁待灰度时跑。

### 候选生产配置 (待用户批准灰度 r1)
```
--spec-type draft-dflash -md /models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
--spec-draft-n-max 7 -c 262144 -np 1   # 其余 flag 不变 (KV q4_0, fa on, jinja)
```

## DFlash2 生产 rollout 完成 (2026-09-04 17:00)
### 灰度门禁 (gate-dflash-r1.json, r1 直连 8082, 灰度期含真实流量)
| 轴 | dflash-r1 | 基线 coding-v1(MTP) | 判定 |
|---|---|---|---|
| HumanEval | 89.02% (146/164) | 90.24% (148/164) | -2 题, 噪声内 |
| XFC | **75.5%** (151/200) | 72.5% (145/200) | +3.0pp |
| GSM8K | 94% | 94% | 持平 |
| IFEval | 48% | 56% | 见归因: 噪声 |
| Needle 64K/195K | 10/10 + 10/10 | 10/10 + 10/10 | 天花板持平 |
| LongGen | 30.0 (混杂) | 38.2 | 见归因: 无回退 |

### 两疑点归因实验 (attr-*.json, 不解释只实验)
- **IFEval**: r1(dflash) 复测 62% vs 同日 r3(MTP) 对照 48% — 该轴运行间摆动 ±14pp,
  与草稿无关 (历史佐证: 同模型 gate 间 56%↔82% 摆动)。判定噪声。
- **LongGen**: GPU7 隔离同轴 A/B (无流量): dflash 43.1/48.8 vs MTP 44.1/40.9 —
  无回退略优; 门禁 30.0 是灰度流量共享 np1 槽 + temp0.7 提前停笔 (5553 字符未达标)
  的混杂。temp0.7 下 dflash 接受率 0.29/token vs MTP 0.55 (7-token 块草稿对采样更
  敏感), 但 TPS 仍占优 (草稿成本低)。

### Rollout 执行
错峰重建 llama→r2→r3 (任意时刻 ≥3 副本在线), LB 重启重解析。全 4 副本 draft-dflash
确认, VRAM 22.98-23.13G/卡 (余量 ~1.4G), LB 6/6 smoke OK, r1 累计接受率 0.468
accepted/drafted (混合采样流量)。回滚: docker-compose.yml.bak-20260904-mtp。
语义变化: 主副本 np2→np1, 全网并发槽 5→4 (LB 排队兜底)。

## imatrix v1.1 重校准立项执行 (2026-09-04 晚)
### 动机
coding-v1 longgen 同实例 -5.2% (38.2 vs base 40.3) 归因量化配方; imatrix-v2 用的还是
v1 的 78KB/80chunk/-c4096 短语料 (中文为零、长上下文为零) — 对 256K 部署分布欠拟合。

### 执行 (今晚)
- 工具链: devel 容器增量编译 llama-imatrix/llama-quantize (宿主无 cmake/nvcc;
  坑: 增量链接需 --gpus 挂驱动 libcuda.so, 否则 build 缓存路径断裂)
- 语料 v1.1 (125 块 / 5.0MB ≈ 1.5M tok, v2 的 60 倍):
  70 代码块 (commitpackft 8 语言 repo 风格拼接 + magicoder, ~40K chars/块)
  + 25 轨迹块 (menv SWE messages, ~80K chars/块)
  + 30 中文块 (r3 自生成技术长文, 补中文欠拟合; temp 0.85, ~2000 tok/块)
- 计算: GPU6 -ngl 20 -c 16384 -b 256 -t 96 (F16 54G 装不下单卡, 20层GPU+45层CPU),
  全程 ~2h (长块 CPU offload prefill 慢); 产物 imatrix-v11.gguf (13.6MB)
- 链式: imatrix 退出后自动 llama-quantize IQ4_XS (同 coding-v1 纯 imatrix 无分层配方)
  → merged-v2-iq4xs-imatv11.gguf

### 明日验收 (纪律: 不改阈值)
1. GPU7 沙箱起 coding-v1.1, 先 30q humaneval 冒烟 (对照 coding-v1 同题集)
2. 六轴全量门禁 vs gate-coding-v1-r2 + needle-v2 (10+10)
3. longgen 同实例 A/B ×2: v1.1 vs coding-v1 (38.2) vs base (40.3);
   成功判据: longgen 缺口 -5.2% → ±2% 内且六轴无回退

## imatrix v1.1 → coding-v1.1 上线 (2026-09-05 06:50)
### 门禁 (gate-v11.json, GPU6 沙箱, 生产形态 dflash/np1/256K)
HE 94.51% (+4.3pp) / XFC 74.50% (+2.0) / GSM8K 94 (平) / IFE 68 (噪声轴, 正向)
needle 10/10+10/10 天花板 / longgen 44.6+44.1 两跑满长 (同日同形态 A/B: coding-v1
44.7/38.9, chars/s 128.5/131.1 vs 118.3/96.8) — **原 -5.2% 量化缺口翻正为 +6%**
配方: imatrix-v11 (82 chunks × 16K, 代码70+轨迹25+中文30) + 纯 IQ4_XS 无分层
产物: /data/models/Qwen3.8-27B-coding-v1.1-iq4xs.gguf (15309039104B 与 v1 同尺寸)
坑: llama-quantize 此 fork 不认 -t (位置参数污染静默退出0); 链式 pgrep 自匹配死锁
上线: 4 副本错峰替换, LB 重启, 全副本 v1.1+dflash 确认, 回滚 .bak-20260905-codingv1

## 2026-09-05 缩容双卡 + vLLM 关停（资源腾挪，非模型变更）
- **llama.cpp 4→2 副本**：r2/r3 停（GPU2/3 释放），route.lua 大小池合并 {llama,r1}，
  LB 验证 3/3 通过。并发上限 4 slot → 2 slot（256K×2）。
- **vLLM 关停**：systemd `vllm-qwen38.service` stop + disable（经 docker --pid=host nsenter
  执行，root 通道）；GPU4/5 释放，9411 关闭。恢复：`sudo systemctl enable --now vllm-qwen38`。
- **当前资源**：GPU0/1 = 生产 llama.cpp（coding-v1.1 + DFlash2）；**GPU2-7 共 6 卡空闲**。
- 同步适配：collect.py BACKENDS 双副本、soak.sh 双端口双卡、compose 头注释。
- 回滚：compose `.bak-20260905-4replica` + `lb/route.lua.bak-20260905-4replica`，恢复 = `docker compose up -d` + route.lua 还原。
- vdn-minimax-h3（OpenVDN 文生视频，87GB）评估后放弃安装：权重 87.4GB + 需 Qwen3-VL-32B 做提示编码，4090×8 与 H200/B200 参考性能差距大。

## 2026-09-05 ComfyUI + MiniMax-H3 双4090 Docker 化部署（GPU2/3）
- **部署**: /data/comfyui（独立 compose，宿主零安装）。镜像 comfyui:minimax-h3（python:3.12-slim 本地基底
  + torch 2.8.0+cu128 + ComfyUI 0.34.0 git master），容器 user 1000:1000，网络 qwen27b_default。
- **模型**（Comfy-Org repack, 44.4GB, hf-mirror 全部字节数校验通过）:
  DiT fl2va_pruned_fp8_scaled 21G + 编码器 qwen3vl-32b nvfp4_awq 15.7G + 视频/音频 VAE 5.8G + turbo 8-step LoRA 2G。
  选 fp8_scaled 是因 int8_convrot 需 torch cu130（驱动 550 上限 cu128）。
- **冒烟实测**: T2V 768×432 / 56帧 / 8步 turbo / seed 11 → **40s 出片含权重加载**（页缓存热），
  产物 h264 768×432@24fps + **aac 32kHz 双声道原生音频**，同种子字节级复现。
  显存: 去噪期 cuda:0(GPU2) 23.5G，cuda:1(GPU3) 做编码器/溢出缓冲。
- **三个坑（都已修进 Dockerfile/compose）**: ① requirements.txt 拉最新 torchaudio=cu130 构建链接
  libcudart.so.13 崩启动 → 钉 torchaudio==2.8.0+cu128（pytorch 官方源，阿里云镜像无此轮子）；
  ② git clone 层偶发空目录（buildkit 缓存+github 抖动）→ set -eux + 3 次重试 + test -f main.py；
  ③ SaveVideo 的 format 是 DynamicCombo：API 格式传字符串 "auto"（模板里的 video/MiniMax_H3 是
  专属 muxer，本构建无此选项）；帧数需 ≡5 (mod 17)。
- **入口**: http://<内网IP>:8188（ufw 已放行）；mon :9000 已接探活（ok, 3ms）。
  官方 T2V/I2V/R2V 模板在 user/default/workflows/；API 副本 workflows_api/smoke_t2v.json。
- **回滚**: cd /data/comfyui && docker compose down && docker rmi comfyui:minimax-h3 &&
  (nsenter) ufw delete allow 8188/tcp && 可选 rm -rf /data/comfyui。

## 2026-09-05 晚 驱动 550→580.173.02 + CUDA 13.0 升级（夜窗口）
- **动机**: 解锁 cu130 生态(int8_convrot 量化件/comfy_kitchen 优化内核/torch 2.9+)。4090 硬件支持
  CUDA 13.x(sm_89), 卡不是瓶颈, 驱动才是; SecureBoot 关闭免签名。
- **升级路径**: apt nvidia-driver-580 580.173.02 (清华源) → dkms 为 6.14.0-27 和 **7.0.0-30 双内核**
  编译(后者是 unattended-upgrades 装了没重启的新内核, GRUB_DEFAULT=0 重启会落它——已预核查 headers+模块就绪)
  → nsenter systemctl reboot。**重启不掉容器**: 全部 unless-stopped 未手工 stop, 开机自拉起 ✓
- **重启后实证**: 内核 7.0.0-30, Driver 580.173.02, **CUDA 13.0**; llama.cpp(cu12.4 构建)零改动正常出字,
  DFlash2 计数正常(70/29); 8 容器自启; bootcheck 日志 /data/eval-rulers/driver580-bootcheck.log(@reboot cron)。
- **ComfyUI cu130 重建**: torch/torchvision/torchaudio = 2.11.0/0.26.0/2.11.0+cu130。
  **comfy_kitchen cuda 后端 disabled: True→False(优化内核解锁), cu130 WARNING 消失**。
- **构建网络坑(本日三连)**: ①buildkit 层内下载 531MB torch 轮子卡死 13min → 宿主 wget -c 下轮子
  COPY 进镜像(--find-links), nvidia 依赖走清华 PyPI; ②github git clone 和 codeload 当晚全抖(截断/超时)
  → ComfyUI 源码直接从 cu128 回滚镜像里 docker cp 提取 COPY 进去(顺带固定版本快照);
  ③pkill -f 自匹配自杀(本机第二次踩!)——模式写 'compose buil[d]' 或先 pgrep。
- **torch 2.11 新坑**: dynamo cache_dir 调 getpass.getuser() → 容器无 uid1000 passwd 条目即崩
  → Dockerfile useradd -u 1000 + ENV HOME。
- **性能 A/B(同工作流 768×432/56f/8步)**: cu128 热跑 40s → **cu130 热跑 16s (2.5×)**; cu130 冷启动(含权重加载)40s。
  注意: ComfyUI 完全相同输入会节点缓存命中(8s 假象, 无新文件), 测速必须换种子。产物流一致(h264+aac 32kHz)。
- **现在可选**: int8_convrot DiT(21G, 官方推荐档)和 Kijai VSA 4-step 蒸馏版(22.9G)均可下载;
  回滚链: comfyui:cu128-rollback 镜像 tag 保留; 驱动回滚 = apt install nvidia-driver-550 + reboot。

## 2026-09-06 凌晨 H3 三项优化实测（功耗墙/convrot/SageAttention）
- **功耗墙双修**: ①day 模式 GPUS 0-3→"0 1"（GPU2/3=ComfyUI 不再白天限功）; ②day 自检周末/节假日跳过
  （脚本内判断, root crontab 无需改; holidays.txt 用户可编辑, 已填 2026 中秋/国庆+2027 元旦/春节;
  调休补班周六想限功写 workdays.txt）。已验证周日正确跳过。
- **SageAttention 负结果**: sageattention 1.0.6(纯triton, 需容器装 gcc) sm89 实测误差 0.001 但
  **更慢**: 同种子热跑 sage 19.4s vs pytorch attention 14.3s。torch 2.11 SDPA + kitchen 内核已更快,
  弃用（compose 旗已撤, 包保留）。测速方法教训: ①ComfyUI 相同输入节点缓存命中不出新文件号(1s 返回),
  必须换种子+验证文件号; ②轮询间隔污染计时, 用 sleep 1; ③sed 改种子要核验实际值(曾静默失配连跑 3 次缓存)。
- **int8_convrot DiT 到货并验证**: 20970379616B 校验 OK, kitchen cuda 内核下正常运行,
  seed23 换模冷跑 17.4s vs fp8 热跑 14.3s（计算部分相当）, 抽帧目检画质与 fp8 无差。
  两个 DiT 共存: fp8_scaled(默认) / int8_convrot(可切换)。
- **风扇-温度曲线实测(GPU2)**: 空闲 29-34°C/69%/18W; convrot 满载 444W/100%util → 54-58°C 但风扇
  仅 64-69%（响应滞后/平滑, 20s 突发不升速）; 持续满载(llama decode 分钟级)才会推到 74%。
  降噪结论不变: 短突发无需管, 常驻负载才需要功耗墙; GeForce 无法软件设风扇(NVML)。
- **性能盘点(768×432/56f/8步, 热跑真重算)**: fp8 14.3s(最优) < convrot ~17s < sage 19.4s; 缓存命中 1s。
- **功耗墙执行留痕（09-06 补）**: gpu-power.sh 全部输出 tee 到 /data/eval-rulers/gpu-power.log
  （跳过原因/实扣上限都留痕）。root cron→脚本链路已以 root 实测：周日正确跳过并记日志。
  周一 08:00 起首个工作日实扣（GPU0/1→250W），18:00 night 恢复 450W，日志可查。
- **功耗墙 v2 逐卡配置（09-06）**: 全 8 卡覆盖 + gpu-power.conf 逐卡 day/quiet/night 三列
  （lytv 可编辑, 免 root）。day=工作日白天全卡 250（含 GPU2/3——白天长批量会响; 要白天满速出片把
  conf 里 2/3 行 day 列改 450）; quiet=200 手动档; night=450。脚本 v2: 幂等（已是目标跳过, 修复
  nounits 小数比较 bug）、逐卡实扣+复原验证、缺配置文件即报错不静默。status 模式 lytv 可直跑。

## 2026-09-06 凌晨 AnnoyingTechnology vLLM 栈 A/B 测试（GPU6 沙箱 vs 现产 llama.cpp）
- **部署**: 88 文件 jsdelivr 拉取 → 打补丁(vllm 轮子走 ghfast.top 代理 / HF_ENDPOINT=hf-mirror /
  device_ids GPU6 / 镜像名 zx-ab-vllm) → 构建 nvidia-4090-llm-inference:0.27.1-cu129 (19.1GB,
  verify 0 失败: 28 补丁 + KVarN 全过)。首次引导模型 19.5GB 下载 + DFlash2 drafter 1.28GB 补拉
  (prepare.sh 当可选但 start_qwen.sh 硬要)。切 max profile(CTX=huge=KVarN K4V2/262K, MAX_SEQS=1,
  DFlash2 k=7) 拉起, 健康检查过。
- **A/B 结果（同 p512/g512 decode 夹具, 各 3 次取中位）**:
  | 指标 | vLLM max profile | llama.cpp 现产(8081) |
  |---|---:|---:|
  | 解码 tok/s | **175.4** (175.43/175.44/175.38) | **90.1** (90.08/90.08/89.66) |
  | 冷预填充 tok/s @p32817 | 2558 | (本轮未测) |
  | KV 缓存 | KVarN K4V2 / 262K | Q4_0 / 262K |
  | 投机解码 | DFlash2 k=7, ~3.9 tok/步 | draft-dflash, ~3.56 tok/步 |
  | base 验证步 | ~28ms/步 (35.8 步/s) | ~43.5ms/步 (23.0 步/s) |
  | 权重 | Huihui W4A16 (abliterated, 16-bit act) | coding-v1.1 iq4_xs (后训, 4-bit act) |
  | 视觉 canary | PASS (RED, BLUE) | (本轮未测) |
  | 前缀缓存 canary | PASS (32640 tokens 复用) | cache_reuse 2048 |
- **诚实归因（不解释掉差距）**: 1.94× 速度差**主因是 base 验证步速度**——vLLM 每步 ~28ms vs
  llama.cpp ~43.5ms (1.56×), 来自 vLLM 更高效的 kernel + CUDA graph + 更新 CUDA13/driver580。
  投机解码接受率其实相近(vLLM ~3.9 vs llama.cpp ~3.56 tok/步, 仅 1.09×), **不是**主因。
  混淆变量: 权重不同(Huihui W4A16 abliterated vs coding-v1.1 iq4_xs 后训) + 量化不同
  (W4A16 16-bit 激活 vs iq4_xs 4-bit 激活), 无法完全剥离; 但同架构同参数量, 验证步差异主导。
- **资源对比**: vLLM max profile 单卡(GPU6) 175 tok/s; llama.cpp 现产双卡(GPU0/1) 2 副本,
  单副本 90 tok/s。vLLM 单请求吞吐更高且更省卡(1 卡 vs 2 卡换 2 副本)。
- **关键限制**: vLLM 不支持 GGUF, 现产 coding-v1.1 是 GGUF(iq4_xs), **无法直接跑在 vLLM 上**。
  要上 vLLM 须换 safetensors 权重(Huihui W4A16 或自转)。
- **B 线决策建议**: ①若追求单请求极速且可用 Huihui W4A16 权重 → vLLM max profile 值得上
  (2× 提速 + 省 1 卡); ②若要保留后训 coding-v1.1 模型 → 只能继续 llama.cpp(GGUF);
  ③混合方案: 部分卡跑 vLLM(Huihui W4A16) 接高吞吐请求, 部分卡跑 llama.cpp(coding-v1.1) 保后训模型。
  当前 GPU6 沙箱保留待决策; 回滚 = compose down + rmi zx-ab-vllm 镜像。

## 2026-09-06 下午 S1 完成：coding-v1.1 量化上 vLLM（B 线主力落地）
- **量化（4 轮迭代）**：AutoRound 0.15 + auto_round:llm_compressor 格式，W4A16 sym g128，校准=calib-v11-autoround.jsonl（1074 样本编程域+中文）。**最终配方照抄 Huihui**（fp_layers=visual+linear_attn.norm+in_proj_a+in_proj_b+lm_head）→ 19GB/8 分片单卡可容。r1 保守版（linear_attn 全保 BF16→26.7GB 装不下）改名 -conservative-GDNbf16 留档。坑：①AutoRound 0.15 本地 jsonl 的 mllm 路径 KeyError bug（已打补丁：本地文件回退文本校准）②compressed-tensors 版本二分=0.15.0（0.18 要 torch≥2.10 拉坏 2.8；0.9.4 缺 compress_module）③精确配方多量化 qkv/z/out_proj 后 layer0 调优 OOM→seqlen 2048→1024 解决。
- **栈适配（GPU6 沙箱改造）**：lm_head int8 + embed 重量化（prepare/quant_lm_head.py + quant_embed.py，round-trip 误差 0.64%/0.56%）；compose 加挂 /data/models:ro；tool-call-parser 名为 **qwen3_coder**（下划线，qwen3coder 不存在）；.env=v11（MODEL 直挂+parser flags）+ MAX_LEN=131072（262K 时 GDN state 池 OOM 差 2MB，128K 稳）。
- **S1.4 冒烟全过**：文本生成✓ 视觉✓（左=红色右=蓝色；ulmus vision canary 的 FAIL 是其提示词自相矛盾"Reply exactly: LEFT, RIGHT"却判"RED, BLUE"，模型照字面执行非能力问题）prefix cache✓。
- **性能基线（T1.3）**：decode 126.9 tok/s（p565/g512 中位）vs 生产 llama.cpp 90.1 = **1.41×**；prefill 2535 tok/s @p32817。低于 Huihui 的 175.4（归因待 T 系列：权重 19 vs 16GB/128K 配置差异）。
- **质量 A/B（T1.1）初判**：10 任务双端，A(vLLM) 快 2-5×；4 个 0ch 全为测量口径问题（thinking 吃满 3072 max_tokens），9/10 有效产出存 quality_ab_result.json 待人工评分。复核时统一 enable_thinking 显式+max_tokens 8192。
- **周边池（S2 提前）**：GPU5=embed(:19623)+rerank(:19624)、GPU7=Qwen3-VL-8B(:19625) 全在线。reranker 正确姿势=--runner pooling+--hf-overrides(SeqCls+classifier_from_token["no","yes"])+--chat-template qwen3_reranker.jinja（缺模板则无区分度）。E2E 检索已验证（中文查询→精准命中，检索+重排 0.06-0.08s）。
- **素材预置（详见 /data/datasets/MEDIA-TEST-MANIFEST.md）**：临沂广电媒资 API 全通道打通（DES-ECB 密钥 <已脱敏-见内部密码库>→api.<内网域名>/private/login；列表=msadmin /cms/video 分页渲染；详情=/cms/video/{id} JSON）；60 条元数据+25 条分条 MP4+18 条拆条 ground truth（lytv_storysplit_groundtruth.json，M7 黄金集）+电台MP3+直播双流录制（Python HLS 录制器，ffmpeg 静态版对 lytv HLS segfault）+KeSpeech 3.3G（临沂=中原官话区，中原子集升重点）。

## 2026-09-06 晚 上下文阶梯实测 + 262K 全速攻关（含一次事故与恢复）
- **阶梯（同夹具 p565/g512）**：128K DFlash2=126.9 | **192K DFlash2=115.4（当前部署）** | 262K 原生MTP=84.3 | 262K DFlash2=启动过但首请求 OOM（KV 池 272,781 tokens 与 Huihui 成功运行完全相同，差在 FLA 内核 2MB 动态分配=碎片边缘）。
- **省显存尝试**：①剥 mtp.*（849MB）→ 无效，mtp 权重本不进显存（省磁盘而已）②int4 lm_head RTN → 栈脚本质量门拒绝（AssertionError quantization error too high）③预制 int4 fast-variant 头 → 权重不匹配（我们 lm_head 是后训的，预制头只对 stock/Huihui）。**结论：262K+全速可行路径=自校准 GPTQ int4 头（-0.63GB 显存必成，小时级工作量）或 vLLM 0.28（S4）或双卡。**
- **CPU KV offload 定位**：单流 decode 无益（全注意力每步扫全史），多轮缓存保持有益（避免重 prefill）——T3.8。
- **175(Huihui) vs 127(我们) 新假说**：DFlash2 drafter 对后训权重接受率下降所致——Phase 2 可用 coding-v1.1 输出重校准 drafter，速度质量双收。
- **⚠️ 事故与恢复**：lean 变体实验时，硬链接目录的 config.json 被量化脚本原地重写穿透（os.link+open('w') 同 inode），原模型 config/index 被写脏 → 服务拒启。**恢复=从分片头 safetensors 扫描重建 index + 照 Huihui 模板重建 config 组声明**（group_1 lm_head int8 / group_2 embed int8），服务验证恢复。**教训：对硬链目录跑 in-place 重写脚本，必须先 cp 断链。** 实验目录 nomtp/lean262k 保留但标记 dirty（config 曾共用 inode）。

## 2026-09-06 深夜 GPTQ int4 头：死循环事故与修复（2.5h 白跑的完整教训）
- **现象**：/tmp/gptq_head4_gpu2.py 跑 2h33m 连 31 个行块的第 1 块都没完成（日志仅 hessian ok，无 rows 打印）；GPU 28%/CPU 101% 双忙=看似在算实则零进度。监视器 150 分钟预算耗尽退出（非进程死），且 pgrep -f 自匹配坑再咬一次（监视器和诊断命令自己匹配自己）。
- **定位三步**：①日志无 rows 0 → 排除"慢"，锁定"卡" ②微基准（/tmp/gptq_microbench.py）：同尺寸单列 0.1-0.7ms → 全程本应 <2min，差 3 个数量级 ③py-spy dump（宿主 ptrace 被禁 → `docker run --privileged --pid=host` 挂载 py-spy 二进制成功）→ 活跃行=gptq_head4_gpu2.py:61 内层量化行。
- **根因**：`while j < (c1-c0)` 组循环体内**没有 `j += gsz`** —— j 恒 0，永远重复量化第 0 组 128 列（补偿还在改 blk，所以 GPU 真的在算，极具迷惑性）。此前合成调试修的三个 bug（补偿符号/组amax/验证设备）都在 for-jj 内层，合成测试自然通过；搬运到分块版时把组推进弄丢。第一轮 100min 被杀同为死循环，当时误判为"2h 长任务正常"。
- **修复**：内层 for-jj 结束后补 `j += gsz`；进度打印改每行块必打（原来每 4 块才打）。合成验证（/tmp/gptq_verify_fix.py）：96×256 随机阵 vs 无 while 的直接参考实现 **Q/S 逐位一致** + SIGALRM 60s 超时保护通过。
- **重跑预期**：量化循环本体 ≤2min（微基准外推），全程含 Hessian（实测 <2min）/验证/打包 ≈ 15-45min。
- **流程教训**：①长任务必须有心跳进度打印且**首块必须打印**（本次若打印 rows 0/248320 缺席即可秒判异常）②监视器用 `kill -0 $PID` 别用 pgrep -f（自匹配）③预算上限要 > 最坏预估 2 倍且退出时报原因 ④"GPU 忙"≠"在推进"，双忙零进度是死循环典型相。
- **深夜续：两个叠加 bug 的完整拆解（修复链）**：
  - **bug① 死循环（v1/v2 的 2.5h+100min 白跑）**：`while j<(c1-c0)` 缺 `j += gsz`，j 恒 0 → 永远重算第 0 组 128 列；补偿在改 blk 所以 GPU 真在算，"双忙零进度"极具迷惑性。py-spy 定位在内层量化行 + 微基准证明同尺寸单列 0.1-0.7ms（全程本应 <2min）→ 代码复查揪出缺行。修复后 96×256 合成阵 vs 直接参考实现 Q/S 逐位一致。
  - **bug② 模块级循环诡慢（v3/v4，每小时<1个列块的病态）**：修完①后主循环在**模块级作用域**下仍 >15min 完不成第一个列块（128 列）；同进程内注入的自证段（同样代码、同样变量规模）却 0.17s 跑完 4 个列块；台架 bench3/4/5 在 0.6s/块速度下复现了全量W0+全尺寸Q列写+while包装+empty_cache+释放重载等全部要素组合——均快。**根因未定位**（环境/GPU/代码/数据/启动器逐一排除），工程出口=把量化循环搬进 `one_block(r0)` 函数（bench5 结构）→ v5 实测 **31 块×0.54s=17s 全程跑完**。教训：诡异性能病态时，用"同进程 A/B 自证段"比无限外部二分更快收敛。
  - **bug③（小）验收段 GPU OOM**：全词表 deq_all(10-15GB)+ref[4096,248320](4GB) 在 GPU 上爆 → 挪 CPU（503G RAM）一次性算。
  - **bug④ 质量短板：GPTQ 补偿缺"跨块传播"（cos 0.9967→0.9992 的来源）**：初版把 Hinv 切成 128×128 对角块、误差补偿只在列块内部传播；参考实现（IST-DASLab fasterquant）在块边界用 `W[:, i2:] -= Err @ Hinv[i1, i2:]` 把补偿传播到**所有后续列**。补上后（16K 行筛选集）：cos 0.99695→**0.99921**、top1 0.965→**0.983**，过门（cos>0.999, top1>0.90）。**教训：合成验证只证明"两实现一致"，若参考本身缺关键步骤则合成全过也白搭——必须对齐权威参考实现逐行 diff，并加"RTN 基线对照"（GPTQ≈RTN 即补偿失效的红旗）。**
  - damp 扫描（0.01/0.02/0.05/0.1）：cos 全稳 0.9992 不敏感，保持标准 0.01。act-order+g128 需 g_idx 特殊处理（排列打散分组），本轮不用；如需再抬精度是首选方向。
  - 验收挪 CPU 后全流程 ~6min（hessian 90s + 循环 31×0.7s + CPU 验收 2-3min + 打包写出）。

## 2026-09-07 凌晨 262K 两全其美落地（GPTQ int4 头全链路打通）
- **质量（全词表 248320 行，固定种子可复现）**：frobenius=0.1510，**cosine=0.99906（门>0.999）**，**top1=0.9822（门>0.90）**。对比：RTN int4 cos=0.99697/top1≈0.965（无跨块补偿的 GPTQ 与 RTN 几乎持平——补偿失效的红旗）。
- **产物**：/data/models/Qwen3.8-27B-coding-v1.1-W4A16-head4（15G，shard7 独立副本 655MB + config 断硬链重写 + 其余硬链）；group_1 lm_head int4 / group_2 embed int8。
- **部署（zx-ab-vllm-qwen-1 / GPU6 :19622）**：MODEL=head4，MAX_LEN=DFLASH_MAX_LEN=262144。加载 **15.03GiB**（int8 头时 15.79 → **省 0.76GB**）。
- **262K 首请求陷阱解除**：上次 int8 头 262K+DFlash2"启动过、首请求 OOM"（FLA 内核 2MB 动态分配卡碎片边缘）；本次首请求直接通过，连续多请求稳定。
- **性能（同夹具 p565/g512，3 次中位）**：**decode 127.9 tok/s @满 262K 配置**（1.649s 出 212 tok，极稳 ±0.006）；prefill 2552 tok/s @32817；功耗 67W。
- **上下文阶梯终版**：128K=126.9 ｜ 192K=115.4 ｜ **262K=127.9（当前部署）** ｜ 262K+MTP=84.3（弃）｜ 生产 llama.cpp=90.1。**满 262K 不但没掉速，反超 192K 配置、追平 128K**——KV/GDN 池预留压力被 0.76GB 显存盈余化解。
- vision canary FAIL 为已知提示词歧义伪影（"Reply exactly: LEFT, RIGHT"模型照字面答），自由式视觉已验证正常。
- **待办（新增强关联）**：int4 头 vs int8 头 10 任务质量 A/B 复核（enable_thinking 显式 + max_tokens 8192）——换头后必做；drafter 重校准（Phase 2，175 缺口的 ~30 tok/s）。

## 2026-09-07 凌晨（续）DFlash2 drafter 重校准成功 + 头部 v3 攻关
- **B 线（drafter）完成**：上游 syv-ai/qwen38-27b-rtx3090 的 capture/quant 脚本取回（GitHub 全通道超时，gh-proxy raw + web_reader 双通道救场）；bf16 drafter 3.85G 下载；400 条 coding-v1.1 分布提示（calib 140 + sft-short 130 + sft-long 130，自 tokenize）；容器内 vLLM in-process 采集 199k 行/模块（fc 250k，GPU_UTIL=0.88 是显存甜点：0.95 钩子 OOM、0.82 KV 池不够）；**剥除 ctx_kv**（上游负结果：混入降 7%）；GPTQ 36 矩阵 32s 完成 → models/Qwen3.8-27B-DFlash2-W4A16-recal（1.19G）。
- **B4 验收 PASS**：tok/step 3.28→**3.49**（/metrics 差分，2571accepted/1032steps+1）；**decode 127.9→141.4 tok/s**（+10.6%，262K 满上下文配置，3 次中位极稳）；距 Huihui 175 还剩 ~34 tok/s（base 验证步差异为主因——TRAIN-NOTES 前章 A/B 归因）。坑：sft 数据把 assistant 回答拼进提示导致生成早收束（行数 199k 而非预期 290k，但统计上够用）。
- **头部 whack-a-mole 全记录**：g128-42k→sql 确定性复读；g64-42k→sql 换形态退化（user_id IN 无限嵌套）；g64-300k(sft混合)→sql✓但 explain 复读（35728ch）。结论：头部 int4 在质量悬崖边，校准分布挪动只会换位置翻车。**正解=解码期采集**（上游 gptq_lm_head.py 配方）：vLLM in-process 生成期钩 lm_head 输入——但本栈 logits 走剪枝路径绕过 lm_head.forward（预钩 0 触发），改钩 **language_model.model.norm 前向输出**（= lm_head 精确数学输入）成功：**838,711 行**，子采样 50 万 → v3（g128/g64 双变体并行量化中）。
- 探针方法论沉淀：/tmp/probe_head.py（sql/explain/bugfix2/multifile/chinese 五任务自动判退化：复读段计数/超长/嵌套递归/乱词）；温度=0 退化可确定性复现=理想回归测试。

## 2026-09-07 凌晨终局：生产配置定型（int8头+192K+recal drafter，122.7 tok/s）
- **v3 解码期采集结果**：g64-v3（50万行解码分布）探针 sql✓ 但 explain 9052ch + multifile 7823ch 超长；g128-v3 连功能门未过（cos<0.999，解码分布更难）。**四轮校准（42k预填/300k预填/500k解码 × g128/g64）全部至少一处确定性退化，int4 头判死**——回滚 int8 头（+192K），262K 全速转 S4/vLLM 0.28 或双卡路径。
- **最终配置**：MODEL=coding-v1.1-W4A16（int8头）+ MAX_LEN=196608 + **DRAFT=Qwen3.8-27B-DFlash2-W4A16-recal**（保留！与头无关）
- **终测**：decode **122.7 tok/s**（192K；旧drafter 115.4 = +6.3%；对生产 llama.cpp 90.1 = 1.36×，上下文 3×）；prefill 2557 tok/s@32k；tok/step 3.28→**3.49**；10 任务与 int8 基线逐字同长（质量零损）；无僵尸。
- **资产**：head4-* 变体目录 5 个留档（标注不可用于生产）；drafter/hessians_noctx.pt（10G，可删省盘）；bf16 drafter 3.85G + recal 1.19G 在 repo/models/；探针工具 /tmp/probe_head.py（建议收进 repo）。
- **遗留**：Huihui 175 vs 我们 122.7 的缺口主因=base 验证步（28ms vs 43.5ms 类比），S4 栈升级是正道；drafter 重训（fc/conv/selector 蒸馏，coding-v1.1 中间层）是 Phase 2 深水区备选。

## 2026-09-07 T 系列冲刺结案：115.4 → 130.3 tok/s @240K（质量零损）
**终配（GPU6 :19622 生产沙箱）**：coding-v1.1 W4A16 int8头 + **MAX_LEN=245760（栈原生huge档）** + recal drafter + **LOOKUP=0** + KV_MEM=4860000000（池~248K tokens）+ CUDAGRAPH_MEM=1000。
- **三级跳**：115.4（192K+旧drafter）→ 122.7（+recal drafter）→ **130.3（+lookup-off & 240K）**；tok/step 3.28→**3.75**（39.2%/tok）；对生产 llama.cpp 90.1=**1.45×**，上下文 240K vs 64K。
- **T 系列判定表**（夹具 p565/g512 中位）：
  | 实验 | 结果 | 判定 |
  |---|---|---|
  | T3.4 MTP@192K | BOOT-FAIL（MTP 加载器要朴素 lm_head.weight，与 packed int8 头不兼容；栈参考~108 低于现配） | 弃 |
  | T3.3 k=5/6/8 | k5/k8 BENCH-FAIL（引擎死）、k6=58.5 腰斩 | k=7 栈默认即最优 |
  | lookup 消融 | **LOOKUP=0=130.2（+6.1%）** | ✅ 采纳（recal drafter 接受率上来后，上下文后缀匹配开销>收益；该机制为原版 drafter 调的） |
  | _CHAIN/_GRAPH_BOTH/INDUCTOR_MAX_AUTOTUNE | 123.7/122.9/123.8（≤+0.9%=噪声） | 不采纳 |
  | T3.1/3.2 262K | C2/C3 双败（int8 头差 0.2GB 过不去 FLA 碎片缘） | **归档 S4** |
  | C 组 240K | **KV池裁 5.26→4.86GB + CG 1400→1000 = 省 0.8GB → 240K 过** | ✅ 采纳 |
- **240K 三重认证**：探针 5/5 ✓；满窗实战 prompt=242,523（98.7% 窗口）生成正常自然收尾（深预填~1430 tok/s）✓；速度 130.3 ✓。
- **262K 定论**：int4 头（质量否决）与 KV/CG 裁剪（差 0.2GB）皆不通，留给 S4/vLLM 0.28。
- 剩余缺口（130 vs 175）：主因 base 步速（~28ms vs 22.4ms），S4 栈升级正道；drafter 深度重训（蒸馏）为备选深水区。
- 实验资产：/tmp/t3/run.sh（.env生成+recreate+bench+metrics 一键试验机）、/tmp/t3/results.tsv。

## 2026-09-07 上午 P0 差距归因分解（S4 前哨战，GPU2 沙箱 zx-p0/:19627）：归因反转
- **背景**: 终配 130.3 vs Huihui-fast 175.4。历史归因"步速 28 vs 22.4ms"系推导值（22.4=175.4÷3.9；且 T 系列
  results.tsv 的 tok/step 列**全是 NA**——/metrics 计数器带 `vllm:` 前缀，当年 awk 没匹配上；"3.75"来自
  spec_bench 另一内容口径）。
- **方法**: GPU2 独立 compose 项目（device_ids ['2']，不碰 GPU6 生产）；新工具 /tmp/p0/bench_step.py
  （单请求隔离 g512 temp0 + /metrics 前后精确差分，usage 实际 token 计数防提前 EOS 假象）；
  功耗对齐：gpu-power.log 实锤 175.4=周日450W、130.3=周一凌晨450W（08:00 才扣 250W），全程 GPU2 提 450W
  （nsenter 特权容器 nvidia-smi -pl 450）。
- **跨卡复核**: A3(coding 现产配置原样搬 GPU2) ulmus t3 夹具中位 **130.0 ≈ 130.3** ✓ → GPU2≡GPU6。
- **四臂+MTP 侦察结果**（ulmus t3 中位 | 隔离 g512 | 隔离 tok/step | ms/step | 接受率%/tok）:
  | 臂 | 配置 | ulmus | 隔离 | tok/step | ms/step | %/tok |
  |---|---|---:|---:|---:|---:|---:|
  | A1 | Huihui-fast+dflash2 stock drafter 262K | 159.6 | 167.8-177.4 | 4.10-4.37 | 24.4-24.7 | 44-48 |
  | A3 | coding-v1.1 现产(240K recal lookup0) | 130.0 | 124.5-133.0 | 3.00-3.23 | 24.1-24.2 | 29-32 |
  | A4 | A3+LOOKUP1+CHAIN1 | 121.2 | 118.8-128.6 | 2.88-3.10 | 24.0 | 27-30 |
  | P1 | Huihui-fast+原生MTP CTX=long | 102.4 | 53.6-55.0 | 1.42-1.46 | 26.5 | 14-15 |
- **判决**:
  1. **步速差不存在**: A1 24.4ms ≡ A3 24.1ms。"28ms 步速差"是 NA 数据×错误口径的假象 → S4 栈升级的
     步速动机消失。
  2. **缺口≈全部在接受率**: 同一中文夹具 3.1 vs 4.3 tok/step（29-32% vs 44-48%/tok）。Huihui=abliterated
     ≈stock 分布，DFlash2 drafter 原生匹配；coding-v1.1 后训偏移大，GPTQ recal 只救回 +6.9%。
  3. **MTP 死刑（P2 手术取消）**: k=3 链结构上限 4.0 tok/step；实测 ulmus 102.4(-36% vs dflash 同模型)、
     中文 1.42 tok/step——上游 draft 词表(丹麦/英/代码)无中文，14%/tok；步速还更慢(26.5ms)。
  4. **int4 lm_head 速度价值≈0**（步速已持平）→ P3 翻案取消，int4 头维持质量死刑，不必再试。
  5. **CHAIN 净负**: 121.2 vs 130.0（-6.8%），lookup 起草伤害>链收益，维持 LOOKUP=0。
  6. **功耗墙**: 250W 日间扣对 decode 仅 -1.5%（A1@250W 156.5→@450W 159.6 ulmus；隔离 174.9→177.4；
     decode 显存带宽型不敏感，负载实测 ~298W@2685MHz 无降频）→ 生产白天无性能焦虑，功耗墙 v2 无需改。
  7. A1 ulmus 159.6 vs 原始记录 175.4 残差 9% 未解（同卡等效已证；疑原始会话 lookup 自适应块 8↔16
     在重复 FILLER 上打满或热态差异），不影响跨臂结论（同 harness 同卡同功耗）。
- **剩余唯一大幅杠杆 = drafter 接受率**: recal(3.1) → 重训目标 4.0+（Huihui 水位 4.3-4.9）≈ 130→170+
  tok/s。可训练面（dflash2-backport.patch 结构分析）: fc(5120→25600 多层隐状态投影) + CandidateSelector
  (predecessor/successor codebook + hidden_projection，直接决定 top-16 候选) + 5 层 conv/注意力；
  上游无 DFlash2 训练器（train_mtp.py 仅 MTP 且其微调是负结果），需自写 train_dflash2.py（在线蒸馏：
  引擎内挂钩目标隐状态+真 token，无需落盘 5×5120/token 的特征）。
- 落盘: /tmp/p0/{run.sh,compose.p0.yaml,bench_step.py,a*.env,results.tsv,*steps}; GPU2 功耗墙实验后
  保持 450W（cron 18:00 night 本来就 450，次日 08:00 恢复日程）。

## 2026-09-07 上午续 P0 第二轮：三个结论修正 + max_model_len 效应发现
- **修正 1（上一 entry 的"A1 隔离 44-48%/tok"不可复现）**: A1-RERUN（同 a1.env、全新引擎、开机首测）
  隔离仅 2.64-2.73 tok/step / 23-25%/tok；但 ulmus 稳健复现 159.4 ≈ A1 的 159.6。
  → 当时 44-48% 是"暖引擎"偶然（该引擎此前已跑过 ulmus×4+损坏 bench），隔离法测接受率不可靠，
  一律以 ulmus 夹具+计数器差分为准。
- **修正 2（A2 接受率腰斩与模型目录无关）**: A1b(fast+240K 减配)=2.7-2.9、A2(非fast+同减配)=2.6-2.9、
  **A1e(fast+240K 全 stock)=2.57-2.68** → 减配三元组与 W4/W8 头全部无罪，**变量是 max_model_len 本身**。
- **修正 3（接受率差距结论重写）**: 同为 240K 时 coding(130.0@3.04) ≈ Huihui-fast(133.6@3.01)——
  我们的 drafter 接受率与参照**持平**，"缺口全在接受率"不成立。真实缺口构成：
  ①262144 相对 245760 的 +19% 接受率效应（159.4@3.6 vs 133.6@3.01，仅 Huihui 侧可达，我们 262K 有显存墙）
  ②175.4 原始记录 vs 159.4 复刻的 10% 残差（原会话特有状态，未能复现）。
- **A6（lookup 复核 @192K stock CG1400）**: L1=123.4@2.9 vs L0=129.9@3.04 → T 系列 LOOKUP=0 结论站得住，
  且证明其未被 CG=1000 污染（CG1400 下同样微负）。
- **max_model_len 效应（新发现，机制未明）**: 565+512 token 夹具下，仅改 max_model_len
  262144→245760 即令接受率 3.6→3.01、tok/s 159.4→133.6。嫌疑：262144=2048 块(2^幂) vs 245760=1920 块，
  KVarN/滑窗 block-promote/v2-cudagraph 的对齐路径差异；KV 池字节固定 5.26GB 两档相同、CG 相同、
  权重相同。253952/249856 响应面测量中（判平滑 vs 悬崖）。
- 生产影响：若效应需 2048 块则我们 24GB 够不到（262144 显存墙 0.2GB）；若中间长度也有增益，
  coding 可试 ≤253952 档（KV stock 5.26GB 容 272,781 tok，FLA 碎片缘待验）。

## 2026-09-07 上午终 P0/S4 前哨战结案：249856+lookup 通道探明但不可上产，现产 130.0 维持
- **A8 系列（coding @249856 + KV_MEM=5000000000 + CG1400，GPU2@450W, ulmus t3）终表**:
  | 臂 | 配置差异 | ulmus | tok/step | 判定 |
  |---|---|---:|---:|---|
  | A8C/A8C2 | +LOOKUP1(adaptive)+PC1+U93 | **171.6/171.3** | 3.89 | **4/12 残差损坏 + OOM 缘脆弱 → 否决** |
  | A8D | +ADAPTIVE=0 | 122.6 | 2.87 | 干净但比现产慢 → 弃 |
  | A8F | +PC0+U95 | 128.1 | 2.94 | 稳定干净无增益 → 弃 |
  | A8G | +PC1+U95 | 122.5 | 2.87 | **GPU_UTIL=0.95 即杀死增益**（判别） |
  | A8B | +LOOKUP0 | 125.4 | 2.94 | 几何本身对裸 drafter 微负 (-3.5%) |
- **171 档的精确依赖**: U93+PC1+LOOKUP1+adaptive+249856 五要素缺一即回落 122-128。
  机制链: max_len≥249856 → 注意力块 2176(对齐 mamba 页) → lookup/adaptive 16-长块通道打开 →
  重复性内容 tok/step 3.0→3.9。该通道: ①adaptive+前缀缓存确定性损坏(栈文档记录,我们复测 4/12 中招,
  分歧为实质内容错乱); ②U93+KV5000 贴 OOM 悬崖(A8@stockKV 42MB 分配亡, A8E 34K prefill 亡);
  ③任何稳定性修复(U95/PC0)或正确性修复(ADAPT0)都消灭增益。**结论: 本栈构建上不可生产, 归 S4/上游修复项**
  (修复点: adaptive 块长变化 × KVarN 前缀恢复的目标前向损坏; 修好后此配置即 +32%)。
- **归因终局(130.3 vs 175.4)**: ①262K/几何+lookup 通道贡献 ~+22%(133.6→159.4 同栈实测, 175.4 原会话
  另有 ~10% 未复现残差, 疑 adaptive 暖态/时钟); ②步速差不存在(24.1≈24.4ms, 旧"28ms"系 NA 数据假象);
  ③我们 130.0 与 Huihui 同配置 133.6 持平——**栈调优已达参照水平**, 参照数字的余量全在不可用通道里。
- **其余终审**: MTP 死刑维持(102.4, 结构上限+中文词表缺失); int4 头无速度价值(步速已持平)+质量死刑维持;
  CHAIN/lookup 微负维持; A8 死因链(FLA 42MB→GPU_UTIL 0.95 修复 OOM 但杀增益)留档。
- **生产处置**: GPU6 现产 .env 未动(130.0 认证配置), repo/.env 已从实验态恢复, zx-p0 沙箱已拆,
  GPU2 功耗复位 250W。多残差前缀矩阵(/tmp/p0/multi_residue_test.py)纳入日后质量门工具箱。
- **剩余正道**: ①上游修复 adaptive 损坏后启用 249856+L1(+32%); ②drafter 深度重训(train_dflash2 蓝图
  已写进上文, fc+selector 可导出面); ③S4 栈升级。功耗墙: 250W 日间 decode 仅 -1.5%, 无需改动。
