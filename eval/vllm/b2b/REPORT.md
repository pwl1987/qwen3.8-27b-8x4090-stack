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

---

## 修订 2 注记（2026-09-08/09，用户裁决；引擎阶段重开）

本报告上述结论截至 tier-1（DEV 代理层），按当时冻结的比较对象（朴素迁移 0.7612）
判 FAIL 并停止——**引擎阶段（导出/boot/FINAL-60）从未运行**。用户裁决 comparator
架构修正（见 CONTRACT-B2B.md §8）后梯子重开：

1. **tier-① 重判 PASS**（同一批测量、比较对象按新 PRIMARY 哲学换为同族
   W4-baseline：QAT@2000 int4 栈 walk 0.7090 vs W4-baseline 0.1306）；上文
   "QAT 被朴素迁移全面击败"的记录原文保留——它回答的是另一个（部署选择）问题，
   并继续作为 context 参照（SECONDARY Q vs S3 相位）。
2. 引擎阶段以 §8 修订后的比较架构执行：五相位 FINAL-60（W/Q/C0/S1/S3）+
   四格矩阵终判。结果见下文「引擎五相位终判」节。

---

# 引擎五相位终判（修订2 执行，2026-09-09）

## 结论速览

| 判据 | 结果 |
|---|---|
| PRIMARY（lean 统一，Q−W 配对 FINAL-60） | **PASS 两轮**：+0.2441（CI [0.125,0.358]，p≈1e-6，48/12/0）→ 复跑 +0.3257（CI [0.158,0.506]，p=1e-5，46/14/0） |
| PRIMARY（**认证生产形制**，确定性） | **FAIL**：−0.0109（CI [−0.085,0.056]，p=0.055）——增益不存在于部署配置 |
| 四格矩阵（按部署形制定判） | **≈0 → QAT 失败**（lean 下为正，但配置脆弱，不能宣称科学成功） |
| 工程 ≥3.5 | 未达（Q 最佳均值 3.4555@lean / 3.2560@certified） |
| t3 / target_correctness | PASS（五相位全自确定、语义 OK、成对 BENIGN-DIFF 良性类） |
| B1-C 复测（S1−C0，描述性） | +0.0346（CI 含 0，p=0.023）——方向为正不显著；**B1-C 冻结判定不回写** |

**一句话**：QAT 在引擎中确实产生了同族增益（W4-baseline → QAT-W4，lean 下 +0.24/+0.33
两轮复现），但该增益**只在 32K/KV2G lean 配置下出现**；在认证生产形制（245K/KV4.86G，
boot 确定性、两次逐位一致）下增益完全消失（−0.011）。B2-B 判定：**QAT 线在生产配置
下无可用增益**——deployment-aware QAT 假设被引擎证据否定（DEV 代理 tier-① 的悲观
预言在生产形制下成立）。

## 五相位 FINAL-60（lean 统一形制：KV 2G / MAX_LEN 32K / LOOKUP=0）

| 相位 | drafter | FINAL-60 均值 | 16 条筛查 |
|---|---|---|---|
| W（W4-BASELINE） | pristine masters × int4 层（`…-b2b-w4base`，sha 1eb2240c） | 3.2114 / 复跑 3.0505 | 3.2410 |
| Q（QAT-W4） | QAT@2000 masters × 同 int4 层（`…-b2b`，masters 6063fdf2） | **3.4555** / 复跑 3.3762 | 3.4053 |
| C0（S0-bf16） | 原版全 bf16 | 3.2627 | 3.2540 |
| S1（B1 trained-bf16） | b1-1@1500 导出全 bf16 | 3.2973 | 3.3385 |
| S3（朴素迁移） | b1-1 masters × 同 int4 层 | 3.2684 | 3.3517 |

lean 内部排序：**Q ≫ S1 > S3 ≈ C0 > W**；Q 对 C0 +0.1928（p=3e-5）、对 S3 +0.1871
（p=0.004）——lean 形制下 QAT 同时击败 bf16 基线与朴素迁移。

## 决定性发现：acceptance 的配置脆弱性（同权重跨 env/boot）

| 对比（同权重、同 60 prompts） | Δ tok/step | 备注 |
|---|---|---|
| W：certified → lean | −0.0554（CI 含 0，24/60 平局） | 稳定 |
| Q：certified → lean | **+0.1995**（CI [0.074,0.311]，p=0.0016，0/60 平局） | 全面分歧 |
| W：lean boot#1 → #2 | −0.161 | lean boot 间混沌 |
| Q：lean boot#1 → #2 | −0.079 | lean boot 间混沌 |
| Q：certified boot#1 vs #2 | **60/60 逐位一致** | 认证形制确定性 |

- **认证形制是确定性的**（两次 boot 逐位复现 3.2560）；lean 形制（KV 2G/32K）存在
  boot 间布局混沌（良性改写类，B1.1 P0 语义），相位均值漂移 ±0.1–0.2。
- 在确定性生产形制下：W 3.2668 > Q 3.2560（Q−W = −0.0109，FAIL）。
- **方法论警告（波及既往）**：drafter A/B 的相位均值在 ±0.1–0.2 量级上配置/布局
  脆弱——任何 ±0.1 级 acceptance 结论（含 B1-C 的 +0.088@31 条回看）都需要多
  boot/多配置稳健性佐证；单一 boot 的均值不足为凭。

## 算子级误差账（§8.5，b1-1@2000 masters 口径，参考 walk 0.791/0.7612 复现 G0）

- 权重 rel err 各层均匀 ~15%（GPTQ 构造使然）。
- 每层 hidden cosine：L0 0.9998 / L1 0.9996 / L2 0.9962 / **L3 0.9873（最深畸变）** / L4 0.9987。
- leave-one-layer-bf16 walk 边际：**L4 +0.0448（主导）** > L2 +0.0149 > L0 +0.0075 >
  L3 +0.0037 > L1 −0.0149（噪声/代偿）；边际和 0.056 > 总损伤 0.0298 → 非可加（恢复
  任一层恢复的是重叠损伤）。
- 判读：损伤不在权重噪声最大层，而在**输出侧 L4**（离 selector 最近）；hidden 畸变
  最深的 L3 大部分被下游代偿。若未来重启量化族，layer-selective（先 L4）是唯一有
  依据的切法——但本轮结论下量化族已全线关闭。

## 现场与证据

- 8 次 boot（5 相位 + W/Q certified 首跑 + Q60b/W60b/Q60c 复跑），全部 zx-p0 down 后
  `.env` 字节复原校验通过；GPU2 终态 2MiB；生产/GPU6 零接触。
- 证据：`evidence/accept60/`（每相位 60 行 + warmup + t3 contract）、
  `evidence/stats-{lean-five-phase,replicate,certifiedenv}.json`、
  `evidence/layer-account.json`、`evidence/probe-*.json`。
- 构件：`models/Qwen3.8-27B-DFlash2-b2b-w4base`（四级 SHA provenance）与
  `models/Qwen3.8-27B-DFlash2-b2b`；35 层 packed 与 S3 逐位一致（断言过）。

## 终局裁定

1. **B2-B 关闭，QAT 线证伪（生产形制）**：四条恢复路（fc 精度 / Hessian 重校 / QAT /
   —— 加上本轮的配置依赖证据）全部走完，训练增益不可在认证部署函数上捕获。
2. 剩余选项不变，待用户裁决：① 部署全 bf16 drafter（S1；245K 需 KV 重预算 +2.6GB，
   本轮 b0 形制实测 S1 lean 3.2973/C0 3.2627）；② 维持现产 recal（零动作）；
   ③ 解冻层权重 QAT（禁令域，需明示解禁）。
3. 新增待裁决：**acceptance 配置脆弱性**是否要求对所有未来 drafter A/B 引入
   「认证形制 + 多 boot」最低标准（本轮实证：lean 单 boot 均值不可采信）。
