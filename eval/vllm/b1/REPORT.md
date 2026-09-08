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

## B1-B / B1-C

（待后续阶段追加：四臂训练 + 每 ckpt replay 门 + 沙箱双统计验收 + W4 shadow。）
