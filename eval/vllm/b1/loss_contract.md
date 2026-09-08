# B1 loss contract — FROZEN 2026-09-08, before the first smoke run

任何改动 = 候选修改 + 全量重跑（治理锁 2）。本文件是 L_total 实现的唯一权威定义；
不同实现声称"同一个 loss"时以本文件为准。

## 总形

```
L_total = λ1·L_tok + λ2·L_rank + λ3·L_sel      λ = 1.0 / 1.0 / 1.0（冻结）
```

## 冻结的实现细节

- **数值域**：三个损失全部在 fp32 计算。可训练参数以 **fp32 master** 持有
  （bf16 参数 + lr 1e-5 的更新小于 bf16 ulp 会被完全吞掉）；导出/验收时降 bf16。
- **L_tok（target-token CE，训 fc）**：`logits = fp32(sample_hidden) @ fp32(lm_head_frozen).T`
  （lm_head 恒 frozen、no-grad 使用 = 对 lm_head 等效 stop-grad，梯度只回 student）；
  full-vocab CE at `teacher_token`；仅 valid 深度参与；深度权重 uniform，在观测槽上归一。
- **L_rank（候选排序，训 fc）**：候选 = **student 自己的 top-16**（取自 L_tok 的
  logits，不用 trace 候选）；label = teacher 在其中的下标；teacher 不在 top-16 时
  该槽 mask（计入 teacher_in_candidate_rate 诊断，不静默丢弃）。
  损失 = `CE(cand_logits[16]) + relu(m − (t_logit − max(其余15)))`，
  **m = 1.0（冻结；logit 域、不除温度、不除 softmax）**。teacher 是常数标签，
  无需 detach；max(其余15) 一侧不 detach（对 student 反传）。
- **L_sel（selector 走链 CE，训 selector+fc）**：selector scores 以 fp32 master 权重
  重算（hidden_projection / 两个 codebook）；teacher-forced 链：深度 d 的行取
  `pred_idx = teacher_{d-1} 在 cand_{d-1} 的下标`（d=0 用 anchor 行，即 p=0 行）；
  `CE(scores[d, pred_idx], teacher_idx_d)`；链断（前驱 teacher 不在候选）则该深度起
  全部 mask。
- **监督契约（治理锁 8）**：TEACHER = target 判定流（被接受 drafts + 纠错 anchor）；
  label 永不取自 drafter 未被 target 接受的输出。每条监督带审计字段
  `{"depth","teacher_token","candidate_index","valid","source"}`。
- **优化 recipe（冻结）**：AdamW lr=1e-5、β(0.9,0.95)、wd=0、chunk=8 步/更新、
  seed=42。

## 训练期监控（每 step）

`loss(3 分量) / grad_norm(fc) / grad_norm(selector) / NaN·Inf / 显存 / 步时 /
top1 agreement / top16 recall(teacher∈own top16) / teacher rank / margin`
——早停只许用这些 proxy，绝不用 acceptance/FINAL（治理锁 6）。
