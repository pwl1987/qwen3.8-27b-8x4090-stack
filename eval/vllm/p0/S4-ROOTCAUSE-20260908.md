# S4/方向A 根因闭环 —— 171 通道与残差损坏的真相（2026-09-08）

接 `docs/VLLM-OPTIMIZATION.md` §4/§5 与 `docs/ROADMAP.md` 方向 A。全部实验在 GPU2
zx-p0 沙箱（:19627），生产 GPU6/llama.cpp 全程未动。证据文件在本目录 `evidence-20260908/`。

## 摘要：三条颠覆性结论

1. **残差损坏与 lookup/adaptive 无关**。LOOKUP=0 下 12 格矩阵损坏 **7/12**（比 LOOKUP=1 的
   3~4/12 更重）。这是引擎级**计算布局非确定性**：同一 token 序列冷算 vs 前缀恢复重算，
   比特不一致 → 近平局 greedy 翻转。现产 130.0 认证配置（L0+PC1）今天就带这个属性。
2. **171.6 是 per-boot 双峰分布的高档采样，不可程序化交付**。t3 夹具在完全相同的配置与
   流程下，每个容器实例锁定 ~169（41.3%/tok）或 ~120（27.1%/tok）之一，boot 内粘滞、
   跨 boot 随机（昨日 3高/1低，当日复测 1高/5低，合并 ~44%）。低档**低于现产 130**。
3. **adaptive 要素在 k=7 从未生效**。`draft_block == num_speculative_steps == 7` →
   `VLLM_DFLASH2_LOOKUP_ADAPTIVE` 分支结构性关闭，8↔16 块长切换（start_qwen.sh 自测矩阵
   的 k>7 场景）在生产候选配置上不存在。历史「五要素」实为四要素；A8D-ADAPT0=122.6 之谜
   解除：该 env 是 no-op，那只是一个低档 boot。

## 实验记录

| # | 实验 | 结果 |
|---|---|---|
| 1 | A8C-R2 立臂（249856+L1，run.sh 全流程） | 168.7 tok/s（3.89 tok/step, 41.3%/tok）；矩阵 3/12（D1@40200/D2@39000/D2@39600） |
| 2 | trace 插桩（容器内 speculator.py+states.py，`/tmp/dflash_trace.on` 文件开关，JSONL 落盘 add/rm/toks/ask 事件） | init: `k=7, draft_block=7, adaptive=false` |
| 3 | D2@39000 token 级定位 | b2 恢复 nct=8704/10514（尾 1810 重算），首分歧在**生成区第 41 token**；两轮 verify 形状恒等（k=7 固定）；turn1 冷-冷逐字节一致 |
| 4 | 跨 boot 文本确定性 | 同格 a1/a2/b2 跨 boot 逐字节一致（含分歧位置）——分歧**确定性**复现 |
| 5 | LOOKUP=0 全矩阵 | **7/12 损坏**（D1@39600/40200/40800、D2 全部 4 格） |
| 6 | t3 探针（复刻 decode 行：FILLER×12 提问，temp0 seed4242 唯一 salt，metrics 差分） | boot 内 3 次 sha/tok/步数完全一致；5 个全新容器全低档（27.1%/tok, 1.90/步, 89 步） |
| 7 | 低档 boot 补救尝试 | 中途跑 g512 / 先 g512 再探 / 完整 ulmus（含 vision+cache canary）→ 均不翻高 → 非预热顺序、非捕获顺序 |
| 8 | 步速归因 | 低档 ms/step 正常（≈23.9），纯接受率塌缩 → lookup 融合贡献归零，非热降频 |

## 根因模型（统一解释）

冷/暖分歧、t3 高低双峰、跨 boot 翻转，全部同源：**混合 GDN 架构的状态扫描与注意力 kernel
不是「计算布局不变」（bitwise）的**。

- 暖轮恢复：尾段以恢复边界（8704）为新 chunk 起点，冷轮以 8192 边界分 chunk → GDN 状态
  与下游 KV 比特差 → 近平局 greedy 翻转。全部观察到的分歧均为**语义连贯的备选续写**
  （如 "the HTTP POST trans..." vs "the `Accept` header..."），无垃圾/复读/乱码 → 良性类。
- t3 双峰：FILLER 重复内容刀刃密集，boot 级数值状态（源头未定位：驱动级设备状态/triton
  JIT/时钟组合）使文本翻到非引用变体 → lookup 引用链断裂 → 接受率 41%→27% 塌缩。
  残差格内容钝感，故跨 boot 位同。
- 上游同源佐证：vLLM #55524（RFC Mamba2 exact-replay：让 prefill/chunked-prefill/decode
  产出位一致，**且该模式明确排除 spec-decode+prefix-caching 组合**）、#53436（temp=0 下
  spec-decode 逐次不确定，文本稳吞吐不稳）、#54993（batch-invariant Mamba2 prefill）。
  即：上游也认为这套组合下位一致性不可得。

## 生产影响与决策

- GPU6 vLLM 沙箱（130.0，L0+PC1）**无用户流量**（主产=llama.cpp :8000 四容器），无实际暴露。
- 若 vLLM 通道将来上产：单轮请求零影响；多轮会话中暖命中轮的续写可能与冷跑不同（连贯但
  非逐字节同）；复现敏感路径用 `cache_salt` 隔离（既有纪律）。
- **方向 A 关闭**：没有可修的 bug；171 不可稳定交付且低档<现产；残差门禁的对象（暖/冷
  字节一致）在此架构上不可达。生产维持 130.0。
- 优先级重排：**B（drafter 蒸馏重训，3.04→3.5 ≈ 130→150）** > **C（vLLM 0.28 迁移）**，
  C 的首个任务=本矩阵在 0.28 复测（上游 batch-invariant 进展）。
- `multi_residue_test.py` 语义重定义：从「损坏门禁」改为「暖/冷分歧率监测」。判据：连贯
  备选续写=良性分歧；垃圾/复读/乱码=真损坏（后者历史上从未观察到）。

## 开放遗留

1. boot 双峰的数值状态源头（非阻断；将来驱动维护窗口可用 GPU 复位做控制变量）。
2. llama.cpp 通道同类风险未测（spec+prefix/session 复用）——建议跑一次同款矩阵。
3. A8E-PC0 BENCH-FAIL 未补（低价值）。

## 本役工具（入仓）

`t3_probe.py`（t3 decode 探针：文本 sha+接受率差分）、`residue_cell.py`（单格残差+四输出
落盘）、`trace_analyze.py`（dflash_trace.jsonl 分析器）；容器内插桩 diff 见
`evidence-20260908/`（states.py/speculator.py 的加法式改动，`/tmp/dflash_trace.on` 开关）。
