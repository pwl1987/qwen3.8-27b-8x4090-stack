# 后续优化路线图（设计构想）

基于 2026-09-07 P0 归因定案（`VLLM-OPTIMIZATION.md` §4）：现产 130 tok/s 与同配置参照持平，
步速差不存在；所有已知余量都压在三条可工程化的路线上。按 投入/产出/风险 排序：

## 方向 A：修复 adaptive×前缀缓存损坏 → +32%（**已关闭**，2026-09-08 根因闭环）

09-08 GPU2 沙箱全日战役推翻了本方向的两条前提，结案文档 `eval/vllm/p0/S4-ROOTCAUSE-20260908.md`：

1. **没有 bug 可修**：残差矩阵分歧与 lookup/adaptive 无关（LOOKUP=0 下 7/12 比 L1 的 3~4/12
   更重）。根因=引擎级计算布局非确定性：暖轮尾段重算的 chunk 边界（恢复点）≠冷轮分块
   边界 → 混合 GDN 状态扫描比特差 → 近平局 greedy 翻转。全部分歧为连贯备选续写（无垃圾）。
   上游同源：#55524 exact-replay RFC 明确**排除** spec-decode+prefix-caching 组合的位一致性；
   #53436 记录了同类 temp=0 逐次不确定。token 级定位：b2 恢复 nct=8704/10514，首分歧在生成区
   第 41 token，两轮 verify 形状恒等（k=7 下 adaptive 结构性关闭，8↔16 切换从未发生——
   start_qwen.sh 自测矩阵是 k>7 场景，A8D-ADAPT0 env 为 no-op，其 122.6 是低档 boot）。
2. **171.6 不可稳定交付**：t3 在相同配置与流程下呈 per-boot 双峰 ~169（41.3%/tok）/
   ~120（27.1%/tok），boot 内粘滞、跨 boot 随机（合并 ~44% 高），低档**低于现产 130**；
   g512 先行/完整 canary 均不能翻高。lookup 融合的价值只在高档 boot 的引用型内容上成立。

**连带影响**：现产 130.0 认证配置（L0+PC1）本身带 7/12 暖/冷分歧——vLLM 通道无用户流量
（主产=llama.cpp），上产前须知：单轮零影响，多轮暖命中轮续写可能非逐字节同（良性），
复现敏感路径用 cache_salt。`multi_residue_test.py` 语义改为「暖/冷分歧率监测」：
连贯备选=良性，垃圾/复读=真损坏（历史上未出现过）。

**优先级重排**：B（drafter 蒸馏重训，130→150）升为首选；C（0.28 迁移）次之，首任务=
本矩阵在 0.28 复测；D 依附 C。

## P0 前置：llama.cpp 生产语义 Gate（**2026-09-08 完成，通过**）

B 开工前对生产主力做的同源体检（工具 `eval/llamacpp/semantic_gate.py`，证据
`eval/llamacpp/semantic-gate-20260908/report.md`）：

- **FAIL=0**：38 份输出 0 垃圾/0 复读环/0 JSON 破坏/0 多轮约束丢失；多轮 ZEBRA 标记 6/6 保留。
- **确定性极好**：冷-冷 24/24、跨重启 16/16、跨副本（生产 :8081 vs :8082）4/4 逐字节 EXACT
  ——llama.cpp 无 vLLM 那类 per-boot 双峰。
- **唯一分歧轴=冷-暖**（前缀缓存恢复 vs 全量重算）：12 格中 2 EXACT/3 BENIGN/7 UNRESOLVED，
  逐条人工复核 7/7 为近平局措辞/引文翻转（如多一个"了"字），且**跨 boot 逐字复现**——
  与 vLLM S4 良性类同构，非随机抖动。生产影响：多轮会话缓存命中轮措辞可能与冷复算略有出入，
  语义等价，对指令型生产 prompt 预期罕见且无害。
- **处置：现产配置冻结**（2 副本 + 现行 flags 不动），B 线解冻开工。
- 工具沉淀：`cache_prompt:false`=冷路旋钮（计时+cache_n 双实证）、`/tokenize` token 级残差
  实录、四层判据（零 LLM Judge，阈值冻结 0.5-2.0/前缀≥20 字）、canary/cmp 复用子命令。

## 方向 B：DFlash2 drafter 深度重训 → 裸接受率 3.0→3.5+（根治）

**B0 已过门（2026-09-08，`eval/vllm/b0/REPORT.md`）**：bf16 双侧纯 torch 复刻对齐——fc 余弦
2065/2065=1.0；top-16 overlap mean 15.83/p10 15/min 15（门 14/12）；selector scores 逐位 0 差；
走链语义判明=逐步贪心（98.8% 一致，3 例平票裁定序差）。附带发现：bf16 引擎 eager 完全确定
（t3 3 跑 sha 一致）；t3 上 bf16 草稿 2.56 tok/step（recal 3.04）→ 重训空间在分布适配。
剩余一级：5 层 conv/attn 的 torch 复刻（in/out 两侧张量已备）→ 端到端可微 → fc+selector 先训。

GPTQ 重校准只救回 +6.9%（115.4→122.7）；后训模型分布偏移的根治是蒸馏重训。
接受率 3.04→3.5 ≈ 130→150 tok/s，且不依赖任何 bug 修复。

**架构可训练面**（dflash2-backport.patch 逆向）：

| 模块 | 参数量 | 导出 |
|---|---|---|
| fc（25600→5120 多层隐状态投影） | 131M | 直接替换（quant_dflash2 已支持） |
| CandidateSelector 码本×2 + hidden_projection | ~254MB（rank=256） | 直接替换 |
| 5 层 conv（kernel_projection）+ 注意力/MLP | ~1.4B | 可冻结 |

**B1 已执行完毕（2026-09-08 晚，`eval/vllm/b1/REPORT.md`，四臂+沙箱 A/B+W4 shadow）**：
- B1-A：五层复刻 B 级门全过（sh_cos 0.9999、top16 15.81、teacher 流 anchor 243/243、
  k 2.51≈引擎 2.56）+ 平票契约 + teacher 监督 + 三组分 loss 训练闭环（zero 零更新断言）。
- B1-B 四臂：**候选侧收益全部来自 fc**（b1-2 单独追平 recall 0.9908）；selector 贡献
  margin（7.03 vs 6.47）；ΔL2/增益分解入档。引擎重叠漂移（15.76→13.9）= 训练重塑
  候选分布的预期效应，adjudicated 非复刻损坏。
- B1-C：**+0.088 tok/step（+2.6%），21/10/0，bootstrap CI [0.011,0.172] 过，Wilcoxon
  p=0.092 未过 → 冻结双门判 FAIL（科学成功未达成，方向一致但 n=31 统计力不足）**；
  工程目标 3.5 差 0.006。t3 sha 门前提被引擎布局非确定性否定（S4 机制同义改写分叉，
  P0 裁良性类）。
- **W4 shadow：收益未穿过量化**（8 probe 对基线 ≈ −0.12）——收益载体 fc 恰是 int4
  量化对象且 Hessian 未重校准。**B2 优先级：量化敏感度（Hessian 对训练后流量重校准 /
  fc 提精度 8-16bit），仍不解冻 5 层**。

**B2-A 已执行完毕（2026-09-08/09，`eval/vllm/b2a/REPORT.md`）——上行的两个假设均被证伪**：
fc 保 bf16（R=−0.39）/ fc int8（R=−0.05）/ trained-flow Hessian 重校准（R=−1.51，最差臂）
全部零恢复；**破坏者 = int4 层量化本身**——训练收益仅存在于 trained-fc × bf16 层协同
（机理：训练信号 rel 0.54% vs int4 噪声 13.2% = 24×，全零行 SNR>1），且 int4 层对原版
权重也净亏（现产 recal 在 FINAL 语料 −0.084 vs bf16——语料敏感，独立发现）。
**B2-B 候选待裁决**：b1 = QAT 式重训（复刻栈换载部署等效 int4 层权重重训 fc+selector，
工具全就绪 ~1.5h）｜ b2 = 直接部署全 bf16 drafter（3.85GB，245K 需显存重预算）。
Hessian 线关闭；不解冻 5 层维持。
**已裁决（2026-09-09）→ B2-B = deployment-aware QAT**（`eval/vllm/b2b/`，CONTRACT-B2B.md
冻结）：B2-A 结论按裁定修订收紧为「fc 精度与 Hessian 均排除，剩余 = 训练 fc×量化层函数
协同/放大 + 语料敏感性；训练函数≠部署函数」，R 降辅助指标、五字段报告强制、语料角色
冻结（DEV 优化/FINAL-60 科学/t3 fixture/production recal 仅参照）；QAT = 冻结栈换
`…-b1-fc16` 35 矩阵反量化重训 fc+selector（与 B1-B b1-1 唯一变量=冻结栈），smoke 前置门
G0/G1/G2 → 正式 2000 步 → 导出（层 packed 逐位不变）→ 16 FINAL 筛查 → ≥60 FINAL
双统计门 → ≥3.5 = 工程成功；四层任一失败即停该层如实入 REPORT。
**B2-B 已执行完毕（2026-09-09，`eval/vllm/b2b/REPORT.md`）——tier-1 FAIL，QAT 线证伪**：
G0 fake-int4 忠实性 PASS（修订口径：walk 相对降 −3.77% ≈ 引擎 −4.4%；recall 条款为
规格错误，训练前治理修订）；smoke G1/G2 PASS；正式 2000 步训练代理正常（与 B1-b1-1
轨迹几乎重合）但**部署栈重放被朴素迁移全面击败**——int4 训练 masters walk 0.709 <
bf16 训练迁移 0.7612，逐 ckpt 被支配，栈间 gap 恶化（−9.5% vs −3.8%）。机理：固定
int4 噪声实现上的补偿学习过拟合 TRAIN 上下文；干净函数训练对零均值扰动更稳健。
**三条恢复路（fc 精度/Hessian/QAT）全部证伪 → 训练增益需要 bf16 层函数本身。**
剩余选项：部署全 bf16 drafter（S1=3.4943，245K 需 +2.6GB 重预算）vs 维持现产 recal
（语料敏感 3.2951）。层权重 QAT（=解冻域）未试，需用户明示解禁才可开题。

**在线蒸馏设计**（免落盘 5×5120/token 特征）：
1. 引擎内进程起目标模型（capture_dflash2.py 同款挂钩），批量生成采
   （aux 层隐状态，真 next-token）对；
2. 纯 torch 复刻 drafter 前向（grouped conv 数学可照抄补丁；滑窗注意力用
   window-mask SDPA；context-KV 预计算 = rms_norm(fc_out)@W_kv，梯度穿过 fc）；
3. 损失 = 7 个预测深度上真 token 的 CE（深度权重 1, 0.5, ... 参照上游 train_mtp），
   可选 selector 边分正则；
4. 先冻结 5 层只训 fc+selector（显存 ~6GB，可与目标同卡），复现 vLLM 数值
   （重放捕获的调用对比 top-16 重叠 ≥14/16）后再解冻深层；
5. 验收：spec_bench 接受率、ulmus 中位、10 任务质量套件三重。

**上游反面教材**：他们对 MTP 头做 KL 蒸馏是负结果（top-1 一致率 0.685 不动）——
但那是把分布蒸馏给一个已收敛的头；我们训的是分布偏移后的适配，性质不同。

## 方向 C：vLLM 0.28 + cu13 栈迁移（基建性）

28 补丁逐个重验（`patches/_check_applied.py` 内容级校验）+ DFlash2/KVarN 在 0.28 主线的
上游化程度盘点（部分补丁本就是 main 分支 PR 的回移，0.28 可能已原生）。原生 cu13 环境
配方已验证（`inference/vllm/build/cu130-driver580/`）。收益预期：更新的 kernel、
上游 spec-decode 修复（含方向 A 同源问题的 main 侧修复）、262K 显存余量。

## 方向 D：262K 恢复（依附 A 或 C）

262,144 档的接受率几何红利（133.6→159.4 同栈实测）+ 240K 显存配平经验都在手上；
int8 头模型差 0.2GB 过不了 FLA 碎片缘——方向 A 的显存治理或 C 的 0.28 内存布局
改动任一落地后重试 C2/C3 配置（KV trim + CG 700 + GPU_UTIL 0.96）。

## llama.cpp 通道：维持

生产双副本稳定（90 tok/s、会话粘滞 LB、功耗墙 v2、mon v3），无主动优化计划；
跟踪上游 dflash/MTP 改进即可。补充纪律：对照任何历史数字先核对功耗档（250W/450W）与
引擎暖态（首测 vs 热机差 2×）。

## 评测基建演进

- `multi_residue_test.py` 升级为常规门禁项（任何动投机解码/前缀缓存的改动必跑）
- 单变量 A/B 纪律：全新 recreate、钉死 GPU_UTIL、同 harness 同功耗
- 历史教训制度化：/metrics 计数器 `vllm:` 前缀、tok/step 口径单一来源（ulmus 夹具）
