# vLLM 投机解码生产线实验日志（coding-v1.1 → 130 tok/s @240K）

时间线 2026-09-06 ~ 09-07。对象：`vllm/` 目录的 syv-ai 栈改造版（vLLM 0.27.1 + 28 补丁 + KVarN + DFlash2），
模型 = 自后训 Qwen3.8-27B-coding-v1.1（W4A16 AutoRound + int8 lm_head/embed），单卡 RTX 4090 24GB。
所有速度为 p565/g512 decode 夹具 3 次中位（`vllm/bench/ulmus_validate.py`），功耗对齐 450W。

## §1 基线与参照（先拆穿参照系）

| 配置 | decode tok/s | tok/step | ms/step | 接受率%/tok |
|---|---:|---:|---:|---:|
| 我们终配（245760+recal drafter+L0） | **130.0** | 3.04 | 24.1 | 29-32 |
| 参照 Huihui-fast 同长度（245760, stock） | 133.6 | 3.01 | 22.7 | 28.7 |
| 参照 Huihui-fast @262144（原 175.4 记录条件） | 159.4（复刻） | 3.6 | — | 37.1 |
| 我们 + lookup/adaptive 通道（§4） | 171.6 | 3.89 | 24.2 | 41.3 |

三条硬结论：
1. **步速差不存在**（24.1 ≡ 24.4ms）。历史"28ms vs 22.4ms"归因是 /metrics 前缀解析事故（`vllm:` 前缀漏匹配
   → tok/step 全 NA → 用错误口径反推步速）。
2. 同长度下我们与参照**持平**——后训模型的 drafter 接受率并未落后（更早的"3.75 vs 3.9"是内容口径混算）。
3. 参照 175.4 的余量全部来自 **lookup/adaptive 通道 + 262K 档几何**，另有 ~10% 原会话残差未复现。

## §2 模型侧：量化与头部

- W4A16 g128 AutoRound（照抄上游配方，fp_layers=visual+linear_attn.norm+in_proj_a/b+lm_head）→ 19GB/8 分片。
- lm_head/embed int8 packed（round-trip 0.64%/0.56%）。**int4 lm_head 判死**：4 轮 GPTQ 校准
  （42k/300k 预填 + 500k 解码态隐状态 × g64/g128，cos 至 0.9992）在 10 任务套件上均留确定性退化
  （SQL 复读/嵌套递归/中文乱词等，每轮换位）；且步速已持平，int4 头**无速度收益**。
  GPTQ 工程教训：跨块补偿 `W[:,c1:] -= Err @ Hinv[c0:c1,c1:]` 缺失时 cos 卡 0.9967（≈RTN），
  必须以官方参考实现+RTN 对照组验收。

## §3 drafter 重校准（+6.9%）

DFlash2 侧车（1.92B，5 层滑窗 conv + fc 25600→5120 + top-16 selector）GPTQ W4A16 重校准：
`capture_dflash2.py` 引擎内挂钩采自分布 Hessian（qkv/gate_up 共享；宽矩阵 memmap 落盘后归约）→
`quant_dflash2.py`。115.4 → 122.7 tok/s（@192K）。要点：k/v 的 context-KV 分布**不并入**（实测 -7%）；
解码态与预填态 Hessian 等效（上游结论复认）。

## §4 T 系列 + P0：速度腿全部封顶后的配置定型

k 扫描（k=7 唯一可行）、lookup-off（-6.1%→采纳关闭）、MTP（102.4 tok/s，k=3 链结构上限 4.0 tok/step
且上游 draft 词表无中文 → 死刑）、240K 显存配平（KV 5.26→4.86GB + CG 1400→1000MiB = 省 0.8GB 过
FLA 碎片缘）→ **130.3 定型**。三级认证：5 探针质量 / 满窗 242K 实战 / 夹具中位。

### P0 归因冲刺（20+ 对照臂，GPU2 沙箱）

- **max_model_len 几何效应**：仅改 max_len，245760→133.6、249856→157.1、253952→166.4、262144→159.4。
  机制：drafter max_len 被压到目标 max_len，注意力块大小自适应对齐 mamba 页（249856→2176-token 块），
  ≥249856 档改善 lookup 融合的 drafter 几何；245760 档（栈默认！）落在饿死 drafter 的几何上。
- **171.6 档"五要素"（09-08 根因闭环后改写，全量见 `eval/vllm/p0/S4-ROOTCAUSE-20260908.md`）**：
  249856 + LOOKUP=1 + GPU_UTIL=0.93 + 前缀缓存的实测高点，但——①adaptive 要素在 k=7 结构性
  关闭（draft_block==num_speculative_steps==7），8↔16 切换从未生效，"五要素"实为四要素，
  A8D-ADAPT0 之谜=低档 boot；②171.6 是 **per-boot 双峰**（~169/~120）的高档采样，boot 内粘滞、
  跨 boot 随机（合并 ~44% 高），低档低于现产 130，**不可程序化交付**；③残差矩阵分歧
  （4/12）**非 lookup 损坏**——LOOKUP=0 下更重（7/12），属引擎级计算布局非确定性
  （暖轮尾段重算 chunk 边界≠冷轮 → GDN 状态比特差 → 近平局 greedy 翻转，全部为连贯备选
  续写，无垃圾输出；上游 #55524 exact-replay RFC 明确排除 spec+prefix 组合的位一致性）。
- **功耗墙**：250W 日间限功对 decode 仅 -1.5%（显存带宽型负载不敏感，负载 ~298W 无降频）。

## §5 剩余路线图（09-08 方向 A 关闭后改写）

1. ~~上游修复 adaptive×前缀损坏后启用 249856+L1（+32%）~~ **已关闭**：无 bug 可修
   （布局非确定性是架构属性，上游 exact-replay 排除本组合），171 档双峰不可稳定交付
   （低档 120 < 现产 130）。见 `eval/vllm/p0/S4-ROOTCAUSE-20260908.md`。
2. **drafter 深度重训（新首选）**：fc + CandidateSelector（码本 rank=256，254MB）为可导出面，
   在线蒸馏（引擎挂钩目标隐状态+真 token，无需落盘特征）；3.04→3.5 ≈ 130→150，
   不依赖任何确定性修复。DFlash2 无现成训练器，需自写。
3. vLLM 0.28 栈升级（28 补丁重验）；首任务=残差矩阵在 0.28 复测（上游 batch-invariant/
   exact-replay 进展，#55524/#54993）。

## 附：工具

| 工具 | 用途 |
|---|---|
| `eval/vllm/p0/bench_step.py <port>` | 单请求隔离步速分解（tok/s / tok/step / ms/step / 按位置接受剖面） |
| `eval/vllm/p0/multi_residue_test.py <port>` | 12 残差前缀命中正确性门（adaptive 损坏检测） |
| `eval/vllm/p0/run.sh` + `arms/` | 单变量 A/B 试验机（.env 重生成 + 独立 compose 项目 + 夹具 + 计数器差分） |
| `eval/vllm/p0/compose.p0.yaml` | 沙箱双引擎样板（`!override` 换卡） |
