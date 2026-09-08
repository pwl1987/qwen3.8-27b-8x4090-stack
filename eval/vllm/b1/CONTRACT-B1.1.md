# B1.1 —— 事后基线/经验校正版（2026-09-08 冻结）

本文件把 B1 实测对两处冻结门前提的否定固化为正式契约。v3.1 基线的其余部分
（治理锁、数据三级隔离、早停只凭 proxy、双统计门、三项禁令）**不变**。

## 1. target correctness 五层契约（替代 t3 sha 逐字节硬门）

**被否定前提的实测证据**：B1-C 中 trained drafter 的 t3 sha `d1de8d63ec92` ≠ 基线
`b6117c312d39`，但 trained 实例自身两跑逐字节一致；首分歧在字符 719 =
同义改写近平局分叉。机制：trained drafter 改变接受长度分布 → verify 批组合变化 →
GDN 状态扫描归约序差 → 近平局 greedy 翻转——与 S4 归因的引擎布局非确定性同源
（连基线自身的暖/冷都能翻，P0 Gate 已裁该类良性）。**sha 变化不能证明 target/drafter
损坏**；反之 sha 相等也不能替代质量检查。

### 判定层（依次）

```
A. self-determinism    baseline 两跑 EXACT ∧ trained 两跑 EXACT
                       ——任一不满足 → FAIL（引擎实例损坏/非确定）
B. semantic correctness 无垃圾（可读字符率）/ 无紧邻周期复读环 / 结构完整（JSON 格式
                       任务可解析）/ 非空 —— 全部确定性判据（沿用 P0 语义 Gate 规则）
C. comparative behavior trained vs baseline 逐层输出 = EXACT / BENIGN-DIFF / UNRESOLVED / FAIL
                       —— sha 不同但 A∧B 成立且分歧为已知良性类（同义近平局/连贯备选）
                          = ACCEPTABLE BENIGN-DIFF
D. acceptance statistics Wilcoxon p<0.05 ∧ bootstrap 95%CI 下界>0 ∧ mean Δ>0（原双门）
E. performance quality  recall / path / ulmus 中位不退化
```

**结论级**：A-E 全过 = target correctness PASS；A 或 B 或 C=FAIL → FAIL；
C=UNRESOLVED → 人工复核；D 单独不构成 correctness 判定（属增益判定）。

## 2. replay 门两族拆分（gate.py 落实）

B1-B 实测：trained top16 与旧基线的 overlap 下降（15.76→13.9）**不是 regression**——
fc 正在改变 candidate distribution，本就是训练目标的一部分（recall↑ top1↑ margin↑
与 overlap↓ 同时发生 = 模型在学习，不是模型坏了）。

```
BASELINE COMPATIBILITY（候选分布漂移；永不参与质量门）
    ├── top16 overlap vs engine 记录
    ├── sample_hidden cosine vs engine 记录
    └── 绝对底线（mean≥14 / p10≥12）降级为本族诊断

SPECULATION QUALITY（唯一参与质量 Gate）
    ├── teacher_in_top16 recall（teacher ∈ student 自身 top-16）
    ├── target_rank / margin（teacher 在候选内的排位与裕度）
    ├── own_walk_k（自身走链对 teacher 流的前缀命中，噪声底基线 0.12-0.14）
    └── DEV loss（三组分）
```

训练期回归门 = 本族指标对**同一 replay 代码、同一 DEV 集**的自比（baseline 先跑一次
作参照）；compatibility 族仅记录漂移供归因。

## 3. 增益归因冻结（B1-B 四臂实测）

- **fc 是候选质量主载体**：b1-2（仅 fc）recall 0.9908 追平主臂 b1-1；
- **selector 是辅助路径优化**：贡献 margin（7.03 vs 6.47），不驱动候选召回；
- **b1-3 候选代理结构性盲区**：recall/top1/margin 只经 lm_head 路径，selector 不经过
  ——selector 单独收益只能由 L_sel 或引擎 acceptance 评判；
- B1 正式定义：**「训练方向获得强方向性证据（+2.6%，21/10/0，CI 下界为正）但未获
  统计学通过（p=0.092, n=31）；成功定位下一瓶颈在 fc 的量化迁移，而非 5 层表达能力」**。

## 4. B2 优先级（冻结，用户裁决 2026-09-08）

1. fc quantization sensitivity（B2-A，本契约后执行）
2. fc 更高精度 / calibration
3. trained-flow Hessian recalibration
4. acceptance 再验证（≥60 FINAL 预注册统计门 + 本契约 A-E 全套）
5. selector / 深层解冻（仅当 fc 被证明到天花板）

三项禁令维持：不解冻 5 层；不扩大训练集；不重做无意义 selector 搜索。
