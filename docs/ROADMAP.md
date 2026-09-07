# 后续优化路线图（设计构想）

基于 2026-09-07 P0 归因定案（`VLLM-OPTIMIZATION.md` §4）：现产 130 tok/s 与同配置参照持平，
步速差不存在；所有已知余量都压在三条可工程化的路线上。按 投入/产出/风险 排序：

## 方向 A：修复 adaptive×前缀缓存损坏 → +32%（首选）

**收益**：`MAX_LEN=249856 + LOOKUP=1 + adaptive + GPU_UTIL=0.93 + 前缀缓存` 五要素齐活时实测
**171.6 tok/s**（重复性内容 3.89 tok/step）。配置、判别链与回归门（`eval/vllm/p0/multi_residue_test.py`）
已全部备好，差的只是那个 bug 的修复。

**修复点定位**（上游栈已知问题，复测 4/12 残差中招）：
- 现象：verify 块长在 8↔16 间交替的请求，从前缀缓存恢复后，目标模型前向输出确定性错乱
  （暖命中轮 vs 冷基准轮内容分歧，如 "is not supporting" vs "is being removed"）
- 已排除：lookup 起草内容本身（错误草稿不可能污染 greedy 输出——rejection_sampler 恒输出
  target_argmax）、KVarN 内核本体（KVARN_FUSED_VERIFY=0 仍错）
- 嫌疑面：块长变化 × 前缀恢复路径上的调度器/注意力元数据一致性（resumed 请求的
  num_computed_tokens 与变长 verify 的 query 布局）
- 修好后验收：12 残差矩阵全 CLEAN + 五要素配置 ulmus ≥165 + 5 探针零损

**显存连带**：该通道贴 OOM 悬崖（U93+KV5000 才活），修复时应一并查 42MB 级动态分配的
碎片缘（FLA 内核 expandable_segments 映射失败），给通道留 ≥300MB 余量。

## 方向 B：DFlash2 drafter 深度重训 → 裸接受率 3.0→3.5+（根治）

GPTQ 重校准只救回 +6.9%（115.4→122.7）；后训模型分布偏移的根治是蒸馏重训。
接受率 3.04→3.5 ≈ 130→150 tok/s，且不依赖任何 bug 修复。

**架构可训练面**（dflash2-backport.patch 逆向）：

| 模块 | 参数量 | 导出 |
|---|---|---|
| fc（25600→5120 多层隐状态投影） | 131M | 直接替换（quant_dflash2 已支持） |
| CandidateSelector 码本×2 + hidden_projection | ~254MB（rank=256） | 直接替换 |
| 5 层 conv（kernel_projection）+ 注意力/MLP | ~1.4B | 可冻结 |

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
