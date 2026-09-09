# B2-B：deployment-aware QAT（2026-09-09 执行完毕，tier-1 FAIL 结案）

**问题**（B2-A 裁定修订锁定的路线）：训练函数（bf16 层）≠ 部署函数（int4 层）——
让 fc+selector 在训练期直接看见部署等效 int4 层函数（QAT），能否恢复 B1 的
acceptance 增益？

**结论（tier-1 判定，按 CONTRACT-B2B.md §4 冻结规则停止）：**

> **QAT 被朴素迁移全面击败。** 在部署等效 int4 栈上，int4 训练的 masters
> （walk 0.709）在**每一个 checkpoint** 都低于 bf16 训练 masters 的直接迁移
> （0.7612）；且 QAT masters 的栈间敏感度反而恶化（bf16→int4 walk gap：
> B1 masters −3.8% vs QAT masters −9.5%）。tier-1 FAIL → 按契约不导出、
> 不 boot、不跑 FINAL-60。

## 执行记录与门

| 门 | 判定 | 数据 |
|---|---|---|
| G0 fake-int4 忠实性 | **PASS（修订口径）** | 同一 b1-1@2000 masters：sh_cos 0.9777→0.9635、top16 13.91→13.33、walk 0.791→0.7612（相对 −3.77% ≈ 引擎 S1→S3 的 −4.4%）。原 recall 条款被证伪为规格错误（teacher∈top16 是 248320 选 16 隶属测试，结构上不敏感），训练开始前按治理流程修订（见契约修订节），无既有测量受影响 |
| G1 可训练性 | **PASS** | zero 臂 int4 栈下 50 步 bitwise Δ=0 断言 ✓；b2b-1 无 NaN、grad 有限非零、Δfc L2>0 |
| G2 smoke 代理起效 | **PASS** | @200：loss 4.818→4.175（−13.4%）∧ top1 +0.0076 ∧ 无回退（@0 基线取自 zero 臂 EVAL50=baseline 初始化，见 evidence/g2-adjudication.json——脚本 @0 评估 bug 已在正式跑前修复） |
| **tier-1 科学第一层** | **FAIL** | 见下表。QAT replay proxy **未**比 B1 trained-bf16 更贴近部署函数，两种读法（绝对值/gap 收窄）均不成立，且逐 ckpt 被支配 |

## 核心 2×3 矩阵（DEV 全量重放，int4 栈 = 部署函数）

| masters | int4 recall | **int4 walk** | bf16 walk | 栈间 gap |
|---|---|---|---|---|
| baseline（未训练） | 0.9790 | 0.1306 | 0.1194 | +0.011 |
| qat@500 | 0.9871 | 0.5187 | 0.5410 | −0.022 |
| qat@1000 | 0.9887 | 0.6679 | 0.6604 | +0.008 |
| qat@1500 | 0.9879 | 0.7052 | 0.7164 | −0.011 |
| qat@2000（预注册导出点） | 0.9887 | 0.7090 | 0.7836 | **−0.075** |
| **b1-1@2000（bf16 训练迁移）** | 0.9895 | **0.7612** | 0.7910 | −0.030 |

（训练侧 DEV 代理全程正常：qat@2000 recall 0.9885 / top1 0.8849 / margin 7.052，
与 B1-b1-1 的 0.9908/0.8882/7.028 几乎重合——**训练在优化自己的代理上完全成功**，
失败发生在"对部署函数的泛化"这一步。）

## 机理判读

1. **QAT 确实在适应部署栈**（int4 walk 0.13→0.52→0.67→0.71 单调上升）——
   fake-int4 训练信号存在且有效，G0 的忠实性成立。
2. **但适应被 38 个 TRAIN run 上的固定噪声实现过拟合**：int4 栈的权重扰动是一次
   具体实现（rel err 15.1%），fc/selector 学到的"补偿"绑定在 TRAIN 上下文里该实现
   的误差签名上，DEV 上失效；而 bf16 训练的 masters 学到的是**干净函数**，对零均值
   扰动天然更稳健——迁移反而更好。
3. **与 B2-A 梯子闭合**：S3（b1-1 masters × int4 层，引擎 3.3428）< S0（baseline ×
   bf16 层，3.4067）说明 int4 层函数本身的损伤 ~0.15 tok/step；本轮证明这损伤**不
   能从 fc/selector 侧恢复**——fc 精度（S3/S4）、Hessian 校准（S5）、QAT（本轮）
   三条路全部证伪。**训练增益需要 bf16 层函数本身。**

## 工程结论（下一步仅存选项）

- **捕获训练增益 = 部署全 bf16 drafter**（B2-A 的 b2 选项）：S1=3.4943 已实测；
  b0 配置可跑；245K 生产配置需显存重预算（drafter +2.6GB）。
- 否则维持现产 recal（FINAL 语料 3.2951，语料敏感已实证）。
- 已全部关闭：Hessian 线（S5 证伪）、QAT 线（本轮）、解冻 5 层（禁令，无任何证据
  支持）。若仍想在量化族内救收益，唯一未试的是**层权重本身进 QAT**（=解冻禁令
  域，需用户明确解除才可开题）。

## 现场与工件

- 训练 GPU4（已完成释放）；GPU2 沙箱**未 boot**（tier-1 FAIL 停在导出前）；
  生产/GPU6 零接触；`.env` 全程未动。
- 冻结栈 `/data/sandbox/ab-vllm/b2b/frozen-int4-stack.pt`（sha16 `b58c75ed9d9276ba`，
  35 矩阵，rel err mean 15.1%，四级 SHA 见 evidence/frozen-int4-stack.pt.provenance.json）。
- QAT ckpts `/data/sandbox/ab-vllm/b2b/qat-b2b-1/`（best-by-rule=ckpt-2000）。
- 工具：`frozen_stack.py`（构建）/`overlay.py`（换栈）/`gate_qat.py`（双栈配对重放）/
  `train_qat.py`（QAT 训练器，smoke+formal）/`export_qat.py`+`probe.py`+`final60.py`
  （tier-2/3 工具，本轮按契约未动用，供后续任何复跑）。

## 复现

```bash
# 冻结栈 + G0
python frozen_stack.py && CUDA_VISIBLE_DEVICES=4 python gate_qat.py \
  --ckpt /data/sandbox/ab-vllm/b1/ckpts/b1-1/ckpt-2000.pt --tag g0 --out /tmp/b2b/g0.json
# QAT smoke + formal（与 B1-B b1-1 同 trace/seed/chunks，唯一变量=冻结栈）
python train_qat.py --arm b2b-1 --max-steps 300 --eval-at 100,200,300 --save-dir .../smoke-b2b-1
python train_qat.py --arm b2b-1 --max-steps 2000 --eval-at 500,1000,1500,2000 --save-dir .../qat-b2b-1
# 逐 ckpt 双栈矩阵
for ck in 0000 0500 1000 1500 2000; do python gate_qat.py --ckpt .../qat-b2b-1/ckpt-$ck.pt ...
```
