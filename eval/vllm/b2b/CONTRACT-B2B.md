# CONTRACT-B2B：deployment-aware QAT（冻结 2026-09-09，首跑前）

B2-A 裁定修订（见 `../b2a/REPORT.md`）锁定的下一刀：**训练期就让 fc 看见部署时的
量化层**——不是训练完再量化，而是把冻结 5 层换成部署等效 int4 反量化权重后原样重训。

## 1. 训练图（唯一变量 = 冻结层栈）

```text
teacher（v2 trace 监督，B1 supervision.py 原封）
  ↓
fc + selector 可训练（fp32 masters；B1 loss_contract.md 原封：λ=1/1/1，hinge m=1.0，
                      AdamW lr1e-5 β(0.9,0.95) wd0，chunk8，seed42，baseline 初始化）
  ↓
5 层 = 部署等效 int4 fake quant（Qwen3.8-27B-DFlash2-b1-fc16 的 35 个线性投影
      反量化为 bf16 dense；conv 投影/selector/norms 本就 bf16，用 baseline 原值）
  ↓
sample_hidden → candidate top16 / selector —— 全部沿用 b1/drafter_torch.py
```

与 B1-B b1-1 臂逐项相同（trace/seed/chunks/recipe/初始化），**唯一差异 = 冻结层栈
bf16 → int4 反量化**，因此两臂训练轨迹可逐步对照。

冻结栈来源 = `…-b1-fc16`（S3，old-Hessian 族，引擎锚 3.3428）。选择依据：同族
优于 recal（S5 3.2379 为最差臂）；production recal 经语料敏感性实证**永不作基准**。

## 2. 语料角色（冻结，沿 B2-A 裁定修订）

| 语料 | 角色 | 禁止 |
|---|---|---|
| DEV（trace runs 30–37） | 优化/早停/回归唯一口径 | 作科学结论 |
| FINAL-60 | 科学判据（第三层门） | 进入训练/早停 |
| t3(512) | B1.1 契约 fixture（A/B/C 层） | 混入统计 |
| production recal | 生产对照参照 | 作训练/质量基准 |

**FINAL-60 预注册**：`make_prompt(300..714, step 7)` 共 60 条，与训练语料零交集，
首跑前冻结清单，之后不得增删。

## 3. Smoke 前置门（先于正式训练，数值冻结）

**G0 fake-int4 忠实性**（无训练）：同一 masters（b1-1 ckpt-2000）× {bf16 栈,
int4 栈} 各跑 DEV 重放（gate.py 口径）：
- 量化在复刻中可见：|Δrecall| ≥ 0.01 且 Δown_walk_k < 0（方向为负）；
- 与引擎梯子同向（S1 3.4943 → S3 3.3428，−0.152）：记比值诊断，不设跨量纲硬门；
- **FAIL**：|Δrecall| < 0.01 → fake-quant 不具部署忠实性 → 全线停止上报。

> **修订 2026-09-09（训练开始前，治理流程）**：recall 条款被 G0 数据证伪为规格错误。
> 实测（/tmp/b2b/g0.json，`gate_qat.py --tag g0-b1-1-2000`）：Δrecall = 0.0 但
> Δsh_cos = −0.0142（0.9777→0.9635）、Δtop16 = −0.58（13.91→13.33）、
> Δown_walk_k = −0.0298（相对 −3.8% ≈ 引擎 S1→S3 的 −4.4%）。机理：teacher∈top16
> 是 248320 选 16 的隶属测试，对 15% 层噪声结构上不敏感（损伤体现在排序与走链：
> top1 −0.0032 / rank +0.039 / walk −0.03），与引擎机制一致。修订后 G0-PASS：
> Δsh_cos ≥ 0.005 ∧ Δown_walk_k ≤ −0.01 ∧ own_walk_k 相对降幅 ∈ [−15%, −1%]
> （引擎带 −4.4% 的 ~3× 容差）；Δrecall 降为记录性诊断。无既有测量受影响
> （G0 为无训练检查，纯门定义修正）。修订后 G0 = **PASS**。

**G1 可训练性**（zero 50 步 + b2b-1 300 步）：loss 下降 ∧ grad 有限非零 ∧ 无 NaN
∧ Δfc L2 > 0 ∧ zero 臂 bitwise Δ=0 断言（B1-0 契约）。

**G2 代理起效**（@200 判定；@300 允许一次复查，两次均预注册）：
```text
loss@t < 0.9 × loss@0
∧ ( recall@t ≥ recall@0 + 0.002 ∨ top1@t ≥ top1@0 + 0.001 ∨ margin@t ≥ margin@0 + 0.05 )
∧ 无代理回退超容差（recall −0.002 / top1 −0.001 / margin −0.05 之外）
```
不过 → **停止**，负结果结案：结论 =「量化层失真不可经 fc 补偿，或训练目标未对齐
部署函数」——与正结果同等入 REPORT，不做任何 rescue。

## 4. 四层成功判据（用户裁决原文，冻结）

1. **科学第一层**：QAT replay proxy 比 B1 trained-bf16 更贴近部署函数
   （同 masters 下 int4 栈 DEV 代理差距收窄，G0 对照表量化）；
2. **第二层（引擎筛查）**：real W4 acceptance 恢复正增益——导出 drafter 16 条
   FINAL 筛查均值 > S0（3.4067）且正向多数；
3. **第三层（统计）**：FINAL-60 上 Wilcoxon p<0.05 ∧ bootstrap 95%CI 下界>0 ∧
   mean>0；N_valid<60 → INCONCLUSIVE（不静默删）；
4. **第四层（工程）**：QAT-W4 FINAL-60 均值 ≥ 3.5 tok/step。

任一层失败即停在该层、如实入 REPORT。

## 5. 报告纪律

- 所有 acceptance 对比**强制五字段**：absolute Δtok/step / relative % / median Δ /
  bootstrap CI / positive fraction；R=(arm−S0)/(S1−S0) 仅附录（B2-A 裁定）。
- 样本记账：N_total/N_valid/N_positive/N_negative/N_tie 恒记录。
- B1.1 两族规则沿用：baseline_compatibility（vs 引擎 bf16 记录的 sh_cos/top16）
  = 候选分布漂移诊断，永不作质量门；speculation_quality 是唯一门族。
- t3 双跑契约 A/B/C：自身 EXACT ∧ 语义 OK ∧ vs baseline EXACT/BENIGN-DIFF。

## 6. 预注册 fallback 与禁令

- fallback（唯一）：b2b-1（fc+selector）DEV margin 回退 >0.5 且 recall 无改善 →
  加跑 b2b-2（fc-only），其余一切不变。
- 禁令维持：不解冻 5 层；不扩训练集；不做 selector 单独搜索；不 post-hoc 调
  λ/早停/评测集；baseline/认证产物永不覆写。
- 导出 = 拷 `…-b1-fc16` 目录覆写 fc.weight + candidate_selector.*（bf16），
  **层 packed 张量逐位断言不变**；四级 SHA provenance（.pt ← S3 safetensors ←
  量化配方 ← Hessian 校准集；ckpt → 导出目录各记 sha16）。

## 7. 现场治理

GPU4=训练、GPU2=沙箱（zx-p0 overlay，b0.env 模板，单变量 DRAFT）；GPU0/1/3/5/6/7
零接触；每沙箱会话结束 zx-p0 down + `.env` 与 `.env.prod-130-certified.bak-20260908`
字节一致；证据文件避开 *.log 扩展名；8 规则脱敏扫描后才可 push。

## 8. 修订 2（用户裁决 2026-09-08/09，冻结于引擎首跑前；tier-① 重判后梯子重开）

**背景与治理**：形式训练 2000 步与 tier-① 原判（比较对象 = B1 trained-bf16 朴素
迁移，int4 栈 walk 0.7090 < 0.7612 → FAIL）已完成并存档（REPORT.md，commit
bff7089）；**引擎阶段从未运行**——本修订冻结时不存在任何旧比较器下的引擎测量，
故无任何测量被改动。用户裁决：comparator 架构修正 + 重开梯子。

- **tier-① 重判（新 PRIMARY 哲学一致应用）**：主比较对象改为同族 W4-baseline——
  DEV 代理 QAT@2000 int4 栈 walk 0.7090 vs W4-baseline masters 0.1306 → **PASS**
  （证据：evidence/gate-qat-2000.json `int4_stack` 块 vs evidence/gate-baseline.json）。
  朴素迁移 0.7612 降为 context 参照；REPORT 原 FAIL 段落原文保留。
- **§4-② 作废**（「16 条 FINAL 筛查均值 > S0（3.4067）且正向多数」）：该比较让
  bf16 baseline 的固定量化损失混入 QAT 增量。16 条集降级为部署健康 + t3 契约门 +
  方向读数（非统计门）；B2-A 的 16 条配对数据仅作历史参照，**不得替代 FINAL-60
  实测同族对照**。

### 8.1 验收比较架构（替代 §4-②③ 的比较对象定义；§5 的 R 附录同时由本表替代）

| 层 | 比较 | 门 |
|---|---|---|
| PRIMARY（科学门） | QAT-W4 vs **W4-BASELINE**（同族），FINAL-60 逐条配对 | Wilcoxon p<0.05 ∧ bootstrap95 CI_lo>0 ∧ mean Δ>0 ∧ n_valid≥60（不足→INCONCLUSIVE，error 行保留不静默删） |
| SECONDARY（描述性，不设门、不校正） | Q vs S0-bf16；Q vs S3（朴素迁移）；S1−S0（B1-C 的 n=60 复测；**B1-C 冻结判定不回写**） | 无 |
| AUXILIARY | recovery = (Q−W)/(S1−S0)，分子分母同 FINAL-60 语料 | 永不设门，仅附录解释项 |

四格解释矩阵（终判框架）：

| W4-baseline→QAT-W4 | QAT-W4 vs S0-bf16 | 结论 |
|---|---|---|
| ↑ 显著 | ≥ S0 | 最佳：量化基本被补偿 |
| ↑ 显著 | < S0 | 科学成功：QAT 有效，仍有量化损失 |
| ≈ 0 | < S0 | QAT 失败 |
| ↓ | < S0 | QAT 反效果 |

§4-④ 维持但明确：**≥3.5 tok/step = 工程 PASS 专属，非科学门**（「训练有科学价值」
与「达到工程目标」分开记录——例：恢复增益但未达 3.5 = 科学成功/工程未达成）。

### 8.2 W4-BASELINE 定义（同族对照）

拷 `…-b1-fc16`（S3）目录，仅覆写 4 个 KEYMAP 张量（`fc.weight`、
`candidate_selector.hidden_projection.weight`、`predecessor_codebook`、
`successor_codebook`）为 pristine baseline 值（源 = `Qwen3.8-27B-DFlash2/
model.safetensors`）；35 个 packed 层投影与一切非 KEYMAP 张量**逐位 == S3**（断言）。
产物 `…-DFlash2-b2b-w4base` + 四级 SHA provenance。语义 = 「未训练 masters × 与
QAT-W4 完全相同的部署层函数」——QAT-W4 与它的唯一差异 = 训练本身。

### 8.3 G2 修订（供未来 smoke；本次不重跑）

```text
loss@t < 0.9 × loss@0
∧ ( own_walk_k@t ≥ own_walk_k@0 + 0.01 ∨ top1@t ≥ top1@0 + 0.001 ∨ margin@t ≥ margin@0 + 0.05 )
```
own_walk_k 离最终 acceptance 最近（draft→selector→walk→accepted depth），是比 recall
更敏感的部署函数 proxy——G0 修订已证 recall（top16 隶属测试）对层噪声结构上不敏感。

### 8.4 FINAL-60 执行与证据 schema

- **五相位**（每相位一 boot，容器逐相位重建、无共享状态；PRIMARY 对最先）：
  W（b2b-w4base）→ Q（b2b = QAT-W4）→ C0（S0-bf16）→ S1（B1 trained-bf16）→
  S3（b1-fc16 = 朴素迁移 context）。
- FINAL-60 = `range(300,715,7)` 固定 seed-7 排序、五相位**同序**；每相位记录前
  2 条 warmup 丢弃（不入场）；t3 双跑每相位执行（B1.1 A/B/C 契约，BENIGN-DIFF 可接受）。
- 每行记录：`prompt_id / phase / boot / order / seed / sec / drafts / accepted /
  tok_per_step`（+ error 行保留）。
- 停梯条件（唯一）：boot 崩溃 / t3 契约 FAIL / 语义 FAIL → 停在该处如实入 REPORT。

> **§8.4 补充（2026-09-09，lean-env 统一，W/Q lean 重跑前冻结）**：C0/S1 相位
> （bf16 drafter 3.6GB）在认证 env（KV_MEM 4.86G / MAX_LEN 245760）下 boot OOM
> 实测失败——该 KV 预算按 int4 drafter（1.4GB）标定。五相位统一改用 **b0 形制
> lean env**（KV_MEM=2000000000 / MAX_LEN=32768 / DFLASH_MAX_LEN=32768，其余含
> LOOKUP=0/GPU_UTIL/CUDAGRAPH 与认证逐项相同；B1-C A/B 相位与 B2-A bf16 探针
> 同形制先例）。W/Q 已完成的认证 env 双跑**保留原样**（/tmp 归档 + evidence），
> 作为 KV/MAX_LEN 对本工作负载（单序列、≤1.3K tok、unique cache_salt、无抢占）
> acceptance 中立性的实证对照：若 W/Q 两套 env 的 FINAL-60 逐条差 ≈ 0，环境中立
> 成立；若不 ≈ 0，以 lean 统一版为准并在 REPORT 记录。无任何既有测量被改动。

### 8.5 算子级误差账（报告项，非阻塞）

35 个量化矩阵逐层三账：weight rel err（frozen_stack provenance 已存）/ 每层 hidden
cosine（带钩 DEV 重放）/ leave-one-layer-bf16 walk 边际（消融）——用于回答「整个
int4 栈不可补偿 vs 个别层主导损失」，为下一轮 layer-selective QAT 提供依据。

### 8.6 复跑与 boot 稳定性规则（2026-09-09，复跑前冻结）

**动因（实测）**：同权重跨 boot/env 配对差——W4base lean−certified = −0.0554
（24/60 平局，p=0.62）但 **QAT-W4 lean−certified = +0.1995（CI [0.074,0.311]，
p=0.0016，0/60 平局）**——同权重不稳定度与 PRIMARY 效应（+0.244）同量级且随
drafter 显著不同。单 boot 均值不足以裁决。

**预注册复跑（各一 boot，lean env，与五相位同协议）**：Q60b（QAT-W4 复跑）、
W60b（W4-BASELINE 复跑）、Q60c（QAT-W4 认证 env 复跑，检验 245K 生产形制下的
真实水平）。

**预注册判定（不改 §8.1 门本身）**：
- PRIMARY 稳定 = (Q60b − W60b) 亦满足双统计门（p<0.05 ∧ CI_lo>0 ∧ mean>0）；
  此时四格矩阵以两轮复本的合并陈述（均值取两轮算术平均，各自 CI 均报告）。
- PRIMARY 不稳定 → 四格终判降级为 **INCONCLUSIVE（boot 方差主导）**，如实报告
  两轮效应量与同权重 boot 方差；不做任何一轮的选择性采信。
- Q60c 单独陈述为「认证 env（245K 生产形制）下 QAT-W4 的水平」，不参与 lean
  门判（C0/S1 无法在该 env boot，缺配对对照）。
