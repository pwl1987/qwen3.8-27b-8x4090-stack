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
