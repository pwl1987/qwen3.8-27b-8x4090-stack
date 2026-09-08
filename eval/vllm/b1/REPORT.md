# B1：DFlash2 drafter 蒸馏训练闭环 —— B1-A 阶段报告（2026-09-08）

**使命句**：证明 fc/selector 的分布适配能真实转化为 acceptance 提升，同时 target 输出零回归。
**上游**：B0（`../b0/`，fc/top16/scores 逐位复刻已证）；B1-A 补齐最后一级（5 层 conv/attn）
并建成可微训练闭环。B1-B/C（四臂训练与沙箱验收）后续追加于本文件。

## 结论速览

| 项 | 结果 | 门 |
|---|---|---|
| A1 五层复刻 B 级对齐 | **PASS（全门）** | 见下 |
| A2 可微训练 smoke（zero + b1-1） | **PASS** | 零更新断言 true；loss 5.52→4.19；无 NaN |
| A3 平票契约 | **HOLDS** | 非并列步 244/244 = 100% |
| A4 监督构建器 | **PASS** | anchor 内检 243/243 |
| 监督门三率（诊断基线） | 深度 3.39/7；teacher∈cand 96.4% | 无 pass/fail（按基线冻结） |

## A1 五层复刻（`drafter_torch.py`）

逐算子复刻引擎 drafter 前向（源：容器 vLLM 0.27.1+backport 的
`qwen3_dflash2.py`/`qwen3_dflash.py`/`dflash/speculator.py`，与 vllm28-env 基类逐行核对）：

- **context KV** = `RoPE(k_norm(k_proj(hidden_norm(fc_out))))`，V 不 norm 不 rope
- **query** = `embed([anchor, mask×7])`，positions C..C+7，块内非因果、context 侧 2048 滑窗
- **5 层** = RMSNorm → conv.prepare → GQA(32/8, per-head q/k RMSNorm, θ=1e7 neox) → conv.finish
  → add+RMSNorm → conv.prepare → MLP(SiLU,17408) → conv.finish；
  conv = taps=2、320 组×16、`pos&7==0` 块边界重置、base+delta 加性系数、非残差
- `sample_hidden` = final RMSNorm 的 mask 行；gate_up 融合权重预计算（修显存泄漏：每层每步
  cat 物化 0.357GB×5）

**官方门结果**（`blevel-evidence/blevel-official-dequant.json`，dequant 权重源，全量重跑）：

```
B_sample_hidden_cos  mean 0.9999 / p10 0.9999 / min 0.9998   （门 0.999/0.995）
C_e2e_top16_overlap  mean 15.81 / p50 16 / p10 15 / min 14   （门 15.4/14）
anchor 链            243/243 = 100%                          （门 100%）
engine_path_k_mean   2.514 vs 引擎 /metrics 2.56             （|Δ|≤0.15）
own_walk_k           0.136 = 复刻噪声底（诊断，永不作门）
```

### 门候选修改（治理流程记录）

原冻结门 `e2e_path_nontie≥0.98` **FAIL=0.0，根因实测**：引擎走链行裕度中位 **0.07**（p10
0.016），而任何非逐位实现的 selector scores 扰动 ≥5.2（embed/lm_head 用引擎等效 W4 反量化
后仍 9.9，证明主导残差在注意力/GEMM 归约序），argmax 链必翻盘。B0 的 98.8% 是逐位 scores
条件下的结果，该条件在 B 级不成立。**替代门** = 重建一致性（anchor 链 100% + k 均值对齐
引擎独立计数器），修改后全量重跑为官方结果。B1-B 训练回归门同样只用"同代码自比"指标，
不依赖对引擎 path 的逐位复现。

### 复刻工程发现（5 层之外）

1. **chunked prefill**：t3 的 565-token prompt 分 384+128+53 三块，每块后都有一次
   propose+draft（draft 与下一块 prompt token 比对）；run 切分规则=大 nt 仅在 decode 步之后开新 run。
2. **teacher 流重建**：temp=0 下被接受 draft=匹配 target 贪心，k=前缀匹配长度；anchor 链
   243/243 验证；k 均值 2.514 对上引擎 /metrics 2.56（独立计数器互证）。
3. **终态步排除**：anchor 落在流末尾（EOS 区）的步 v 推导部分盲（文本无 EOS），且对训练
   无价值，按规则排除（s≥len(stream)）。
4. **embed/lm_head 权重源**：引擎用 W4A16 packed 量化值；`dequant_packed` 反量化后复刻
   保真略优（top16 15.81 vs 15.76 bf16 源），差距小说明量化非主导残差。

## A2 训练闭环（`train.py` + `loss_contract.md`）

- 三组分 speculation-aware loss（λ=1/1/1、hinge m=1.0 冻结，见 `loss_contract.md`）
- **fp32 master**（bf16 参数 + lr 1e-5 更新会被 ulp 吞）；frozen 栈 bf16，梯度穿过 cast
- smoke（GPU4，50 步×chunk 8，seed 42）：
  - **B1-0 zero**：完整 loop 含 optimizer.step 后 **Δfc=0、Δselector=0 逐位断言 true**
  - **B1-1**：loss 5.52→4.19，梯度全组非零（fc~6 / hp~14-20 / pred~0.19 / succ~3），
    无 NaN，峰值 19.06GB，0.37s/步
  - proxy：top1 0.57-0.78，recall 0.83-1.0，margin 尾部 ~1.8-3.7（L_rank 在起作用）

## A3 平票契约（`tiebreak_test.py`）

- 契约：scores 行内精确并列（fp32 相等）→ 并列步剔除出一切 path 门；非并列步 torch
  首指标 argmax 必须 100% 复现引擎路径
- **HOLDS：244/244 非并列步一致**；12 个并列步按契约剔除（其中 torch 与引擎实际全一致，
  但 Triton 裁定序不构成契约）
- **B0 归因修正**：B0 报告的"3/259 失配=平票裁定序"不准确——那 3 步是 **warmup 请求的
  温度采样 walk**（temp>0 Gumbel），与贪心不可比；真正的并列步反而全部一致

## A4 监督构建器（`supervision.py`）与插桩 v2（`ask-trace-v2.patch`）

- **契约**：TEACHER=target 判定流（被接受 drafts + 纠错 anchor）；label 永不取自 drafter
  未被接受的输出；每条监督带 `{depth, teacher_token, candidate_index, valid, source}` 审计
- 索引契约：in-record j 的 num_rejected = D_{j-1} 的拒绝数 → v_j = nt_j − rejects(D_{j-1})；
  labels 用 D_j 自己的拒绝数（在 j+1 步可观测）
- **v2 插桩**：`num_rejected/num_sampled` 直录（propose 函数参数，版本无关）+
  `DFLASH_ASK_MAX` 可配上限——大采数不再需要输出文本重建
- 监督门三率（t3 260 对基线）：**valid_depth_rate 0.484**（3.39/7 深度可观测）、
  **teacher_in_candidate_rate 0.964**、**valid_selector_rate 0.964**
  → 解读：候选生成已很强（96% 可观测 teacher 在 top-16 内），瓶颈在近平局走链选择——
  这是四臂解读的关键先验（B1-3 类臂的主战场）

## 复现

```bash
cd eval/vllm/b1
CUDA_VISIBLE_DEVICES=4 /data/vllm/venv/bin/python drafter_torch.py   # B 级官方门
/data/vllm/venv/bin/python tiebreak_test.py                          # 平票契约
/data/vllm/venv/bin/python supervision.py                            # 监督门三率
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=4 /data/vllm/venv/bin/python train.py --arm zero --steps 50
CUDA_VISIBLE_DEVICES=4 /data/vllm/venv/bin/python train.py --arm b1-1 --steps 50
```

trace 归档：`/data/sandbox/ab-vllm/b1/trace-b0-260/`（222MB，.pt 不入 git）。

## B1-B 四臂训练（2026-09-08 晚，GPU4 串行）

**数据**：v2 trace 2080 对（`/data/sandbox/ab-vllm/b1/trace-v2`，插桩 v2 num_rejected
直录）；三级隔离 TRAIN=calib 行 0-29+38-45（34 runs）/ DEV=行 30-37（8 runs）/
FINAL=ulmus 新提示词（B1-C 专用，采数全程不可见）。DEV 基线 gate 全过
（sh_cos 0.9999、top16 15.76）；DEV 代理基线 recall 0.9823 / top1 0.8347 / margin 5.35；
引擎 k 3.99（calib 语料对 drafter 友好，接受已高——天花板效应的先兆）。

**四臂结果**（recipe 冻结：AdamW 1e-5 / β(0.9,0.95) / wd 0 / chunk 8 / seed 42 / ≤2000 步 /
ckpt@500 / 早停 patience 2 仅凭 proxy）：

| 臂 | 可训练 | best DEV（步） | ΔL2 相对（fc / hp / succ） | 引擎 top16 终值 |
|---|---|---|---|---|
| B1-0 | 无 | =基线（对照） | **0（逐位）** | 15.76（=基线逐位） |
| B1-1 | fc+selector | recall 0.9908 / top1 0.8882 / margin 7.03（2000） | 0.65% / 10.7% / 0.34% | 13.91 |
| B1-2 | fc | recall 0.9908 / top1 0.8880 / margin 6.47（2000） | 0.68% / 0 / 0 | 13.82 |
| B1-3 | selector | 候选代理不动（1500 早停）；L_sel 4.44→4.31 | 0 / 8.6% / 0.26% | 15.76（=基线） |

**增益分解（acceptance attribution）**：
- **候选侧收益全部来自 fc**（b1-2 单独追平 b1-1 的 recall 0.9908）；
- selector 增益体现在 margin 深度（b1-1 margin 7.03 vs b1-2 的 6.47）；
- b1-3 的候选代理结构性盲区：recall/top1/margin 只经 lm_head 路径，selector 不经过——
  权重确实移动（ΔL2 与 b1-1 的 selector 部分同量级）但代理不可见；selector 单独收益
  只能由 L_sel 或引擎 acceptance 评判。

**门的 adjudication（治理记录，非改门）**：fc 臂对引擎记录的 top16 重叠随训练单调漂移
（15.76→14.97→…→13.9，500 步起破相对门 −0.4，2000 步破绝对底线 14）。根因：训练
本 reshapes 候选分布（teacher 进前排=候选集改变），重叠度量的是"与旧 drafter 一致性"
而非质量——非复刻损坏（无 NaN、recall↑、DEV loss↓、绝对底线在 500-1500 保持）。
**选择规则按冻结执行**：绝对底线过滤 + DEV 最佳 → **b1-1@1500**（margin 6.69，top16 14.26）。

## B1-C 沙箱验收（GPU2，2 boot + shadow 1 boot）

**A/B 设计**：baseline-bf16 vs trained-bf16（b1-1@1500 导出，唯一变量=权重）；
31 条 FINAL 提示词（ulmus 300..690 步进 13，全新）；per-prompt tok/step（含 bonus）；
双统计门 + N_valid≥30 契约。**A 相 t3 sha = `b6117c312d39` 与 B0 基线逐位一致**（锚定）。

**结果**（`blevel-evidence/accept-stats.json`）：

```
N_total=31  N_valid=31  N_positive=21  N_negative=10  N_tie=0
mean Δ tok/step = +0.0876（3.407 → 3.494，+2.6%）   median Δ = +0.0546
Gate B（bootstrap 95% CI）= [0.0113, 0.1719]  下界>0  ✓ PASS
Gate A（Wilcoxon 正态近似）p = 0.0919 > 0.05           ✗ FAIL
verdict = FAIL（冻结双门要求 A∧B；禁令③禁止事后扩验收集）
```

**两层成功判定**：科学成功 = **未达成**（Gate A 未过；方向一致、幅度真实但 n=31 对
+0.09 效应统计力不足）；工程成功（bf16 ≥3.5）= 未达成（3.494，差 0.006）。

**t3 sha 门的前提否定（候选修改，治理记录）**：trained 引擎 t3 sha=`d1de8d63ec92`
≠A。取证：B 相自身两跑逐位一致（引擎仍确定）；首分歧在字符 719 =
**同义改写近平局分叉**（"cannot efficiently read and write data" vs "cannot process
the data fast enough"）。机制=S4 已归因的布局非确定性：接受长度分布变化 → verify 批
组合变化 → GDN 状态扫描归约序差 → 近平局翻转——**贪心输出与 drafter 无关这一前提在
本引擎不成立**（连基线自身暖/冷都翻，P0 Gate 已裁该类良性）。门按 P0 语义标准重述：
输出须连贯无垃圾/复读/结构破坏（B 文本符合）；逐字节等式不可达。

## W4 shadow 迁移预筛（范围严控，单 boot 极小 probe）

**provenance 四级 SHA**（`export.py` + quant 输出）：

```
bf16 ckpt 导出  e782419de5ce2a0c   (models/Qwen3.8-27B-DFlash2-b1)
量化 recipe     8ac39a85a82929ec   (repo/drafter/quant_dflash2.py, GPTQ g128 对称)
校准集          51eedabac2d51877   (repo/drafter/hessians_noctx.pt, 基线流量捕获)
shadow drafter  b5460283b9d7527a   (models/Qwen3.8-27B-DFlash2-b1-w4, 1.19GiB)
```

8 提示词方向性 probe（非统计门）：**W4 对 bf16 基线平均 ≈ −0.12 tok/step**
（+0.11×2 / −0.12×2 / −0.03 / −0.09 / −0.45 / −0.40）——训练收益未能清晰穿过量化。
机制自洽：**收益载体 fc 恰是量化对象**（quant_dflash2 对 layers+fc 做 int4，selector
恒 bf16 无损穿过），且 GPTQ Hessian 来自基线 drafter 流量、未对训练后分布重校准。

## B1 终局结论

1. **训练价值方向成立但未达显著门槛**：+0.088 tok/step（+2.6%），21/10/0 正负比，
   CI 下界为正；Wilcoxon p=0.092。按冻结双门 = FAIL，如实记录。
2. **增益归因**：全部候选侧收益来自 fc 的分布适配（监督门早已预告：基线 recall
   96-98%，天花板效应）；selector 贡献 margin 深度。
3. **量化是下一刀的约束**（shadow 结论）：B2 应先做量化敏感度（Hessian 对训练后
   drafter 流量重校准；或 fc 提精度 --fc-bits 8/16——fc 仅 131M，bf16 只 +180MB），
   **而非解冻 5 层**（三项禁令维持）。
4. 复盘要点：DEV 语料引擎 k 已 3.99（接受天花板近）；若未来重测，n≥60 预注册。

## 复现（B1-B/C）

```bash
cd eval/vllm/b1
CUDA_VISIBLE_DEVICES=4 /data/vllm/venv/bin/python gate.py --tag base   # DEV 基线门
./run_arms.sh                                                          # 串行四臂+门
/data/vllm/venv/bin/python export.py --ckpt .../b1-1/ckpt-1500.pt --out ...
# 沙箱 A/B：b0.env 模板，DRAFT 分别指向 baseline 与 export；accept_ab.py A/B/stats
CUDA_VISIBLE_DEVICES=2 /data/vllm/venv/bin/python <repo>/drafter/quant_dflash2.py \
    <export> <shadow> <repo>/drafter/hessians_noctx.pt                 # W4 shadow
```

训练产物：`/data/sandbox/ab-vllm/b1/ckpts/{zero,b1-1,b1-2,b1-3}/`（.pt 不入 git）；
导出与 shadow：`/data/sandbox/ab-vllm/repo/models/Qwen3.8-27B-DFlash2-b1{,-w4}/`。

