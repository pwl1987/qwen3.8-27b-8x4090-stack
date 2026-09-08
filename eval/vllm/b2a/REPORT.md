# B2-A：fc 量化敏感度实验（2026-09-08/09 执行完毕）

**问题**（用户冻结的下一刀）：B1-C 训练收益（bf16 +0.088 tok/step）未穿过 W4 量化
（shadow −0.12）——瓶颈是 fc 精度、Hessian 校准、还是别的？

**臂矩阵**（同一 trained 导出 `…-b1`（b1-1@1500）构建；探针=16 条 FINAL 提示词
（全在 B1-C A/B 缓存内）配对；S0/S1 参照=31 提示词全量缓存）：

| 臂 | 构建 | tok/step | Δvs S0 | R(总体) |
|---|---|---|---|---|
| S0 baseline-bf16 | 原版 bf16（B1-C A 相） | 3.4067 | — | — |
| S1 trained-bf16 | b1-1@1500（B1-C B 相） | 3.4943 | **+0.0876** | 1.0 |
| S4 fc-int8+int4 层(旧 H) | `--fc-bits 8` | 3.3744 | −0.005 | −0.05 |
| S3 fc-bf16+int4 层(旧 H) | `--fc-bits 16` | 3.3428 | −0.036 | −0.39 |
| S6 现产 recal | 既有 `…-W4A16-recal` | 3.2951 | **−0.084** | −0.90 |
| S2 fc-int4+int4 层(旧 H) | B1 的 shadow | 3.2631 | −0.123 | −1.55 |
| S5 fc-int4+int4 层(trained-flow H) | 重捕 Hessian 重校准 | 3.2379 | −0.141 | −1.51 |

所有臂 t3 双跑自确定性 ✓（S2 早于契约未测记 None）；各臂 sha 各异=各接受分布下的
S4 类同义近平局分叉（B1.1 契约 ACCEPTABLE BENIGN-DIFF 类）。

## 结论（冻结判定 + 数据精确化）

冻结规则触发 **S3 R<0.3 ∧ S5 R<0.3 → "收益脆弱"分支**，但臂矩阵把根因精确化：

1. **fc 精度不是症结**：fc 完整保 bf16（S3）与 int8（S4）恢复率均为 0——把 fc 一比特
   不动地放进 int4 层栈，收益照样消失。
2. **Hessian 重校准无效**：S5（trained-flow Hessian，最费工时）是**最差臂**；旧/新
   Hessian 之间（S2 vs S5）在配对中位上甚至反号——层输入分布校准不是瓶颈。
3. **破坏者 = int4 层量化本身**：训练收益活在 trained-fc × **bf16 层函数**的协同里；
   int4 层 15% rel err 的函数畸变使其不可迁移（与 fc 的 24× 噪声/信号比同构——机理
   分析见 `fc-signal-vs-noise.json`：训练信号 rel 0.54% vs int4 噪声 rel 13.2%，
   全零行 SNR>1）。且 int4 层对**原版权重**也是净亏（S6 −0.084 vs S0 bf16）。
4. **独立发现（生产相关）**：现产 recal drafter 在本 FINAL 语料上低于 bf16 基线
   0.084——drafter 家族的优劣是语料敏感的（t3 上 recal 曾 3.04 vs bf16 2.56）。
   生产 llama.cpp 线不受影响；vLLM 认证沙箱（130.0）的 drafter 选择可再议。

## 两条出路（B2-B 候选，待裁决）

- **B2-b1 训练目标修正（QAT 式，推荐）**：把复刻训练栈的 5 层冻结权重换成**部署等效
  int4 反量化权重**（DrafterTorch 换载 `…-b1-fc16` 的层权重即可），同 trace 同 recipe
  重训 fc+selector——训练直接适配部署函数。全部工具就绪（b1/ 套件复用），成本
  ≈B1-B 一轮（~1.5h GPU4）。若过 ≥60 FINAL 预注册门 → 收益可生产化。
- **B2-b2 部署侧让步**：直接部署 S1（全 bf16 drafter，3.85GB）——b0 配置实测可跑
  （MAX_LEN 32K/KV 2G）；245K 生产配置需显存重预算（drafter +2.6GB）。
- 不采纳：继续调 Hessian（S5 证伪）；解冻 5 层（禁令；且 S6 显示 int4 层本身有亏）。

## S5 捕获方法记录（供复现）

- bf16 trained drafter + W4 target 单卡捕获 OOM 四次后，以 `kv_cache_memory_bytes`
  上限（1.5G，与现产 start_qwen 的 KV_MEM 同机制）+ MAX_LEN 2048/MAX_SEQS 2/提示词
  截断 1500 成功（39 分钟，fc 250k 行、down_proj 各 191k 行）。
- capture 必须在打了 dflash2 backport 的镜像内跑（宿主 vllm 0.20.2 无该架构）；
  容器挂载须含 `/data/models`（`coding-v1.1-W4A16` 是指向它的符号链接——本坑已踩）。
- reduce 相在宿主 GPU4 复刻执行（`reduce_host.py`）；`ctx_kv` 已删（noctx 教训：
  混入实测 acceptance −7%）。Hessian 归档
  `/data/sandbox/ab-vllm/b1/hessians-capture/hessians_trained_noctx.pt`（10GB，
  sha16 `04ce5c5d4473e7c0`）。
- 逐臂 provenance：S5 = ckpt b1-1@1500（e782419d…）→ recipe quant_dflash2.py
  （8ac39a85…）→ calib hessians_trained_noctx（04ce5c5d…）→ drafter
  `…-b1-w4recal`（1569632b…）。

## 复现

```bash
cd eval/vllm/b2a && /data/vllm/venv/bin/python analyze.py    # 恢复表
# 构建臂:  cd /data/sandbox/ab-vllm/repo && CUDA_VISIBLE_DEVICES=4 python drafter/quant_dflash2.py \
#   models/Qwen3.8-27B-DFlash2-b1 models/…-b1-fc16 drafter/hessians_noctx.pt --fc-bits 16  # S3
# 探针:    沙箱 boot（b0.env 模板换 DRAFT）→ python probe.py <label>
```
