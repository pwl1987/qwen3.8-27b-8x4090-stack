# SGLang 探索存档（状态：已弃用，2026-09-03 回归 llama.cpp）

2026-09-02 ~ 09-03 的第三引擎探索完整周期：可行性评估 → FP8 乱码修复 → 三引擎横评 →
2 卡生产决策 → 生产迁移执行 → **决策反转回归**。本目录保留当时的基准/巡检/诊断脚本与结果；
完整叙事在 `docs/QWEN27B-ANALYSIS.md` §12（可行性）§13（外部参考）§15（横评与决策）。

## 时间线与结论

| 阶段 | 内容 | 结论 |
|---|---|---|
| §12 可行性（09-02） | GPU 共存方案设计：GPU0-3 llama.cpp 生产 + GPU6-7 SGLang FP8 TP2（RFT 产能/多 LoRA 热挂专用） | SGLang 量化生态（BF16/FP8/AWQ）**没有 4.2bpw 级压缩、GGUF 支持不成熟** → 无法复刻单卡 256K 形态，定位为补位项而非替换项 |
| §15.4 关键修复 | **`is_layer_skipped` 子串匹配 bug 致 FP8 输出乱码**（qwen3_5 混合层名误判跳过量化） | 修复后验证干净——上游 qwen3_5 混合架构适配的第一坑 |
| §15.5/15.6 A/B 实测 | 同脚本同提示集横评（`ab_bench.py`/`agg_bench.py`） | 聚合吞吐占优、单流延迟与长上下文不占优；TP2 单流 34.8/67.8 未达外推预期 |
| 256K 虚标更正 | KV 池实测 **216,938 token**（非标称 262,144），单请求上限 ~217K | 文档全部更正（§15.5/15.6/15.7）——"标称上下文"必须以池实测为准 |
| §15.7 生产决策（09-02） | 2 卡 SGLang 替代 4 副本 llama.cpp | 吞吐/卡比划算，RFT 多 LoRA 门禁原生支持 |
| 迁移执行（09-03） | SGLang 容器化 + LB 切换 + 四轴复测 + 浸泡（`soak-sglang.py`） | 上线一天 |
| **§15.12 决策反转（09-03）** | **SGLang 0.5.10 Anthropic 协议致命缺口：无结构化 tool_use** | 生产工具调用链路断裂 → **回归 llama.cpp**，停 SGLang 释放 GPU6/7；另记录 thinking 泄漏进 content（`--reasoning-parser qwen3` 修复，Anthropic/OpenAI 双端点验证干净） |
| §15.13 替代评估 | vLLM 0.20.2（AutoRound W4A16 uncensored-vision，GPU4/5 实测） | 聚合 573 / 单流 69 / 视觉 OK / **Anthropic thinking+tool_use 结构化完整** → 成为 vLLM 线前身；其部署坑与 reasoning-parser 吞输出教训沉淀入 PROBLEMS-AND-FIXES |

## 文件清单

| 文件 | 用途 |
|---|---|
| `ab_bench.py` | 三引擎 A/B 基准（同脚本同提示集，单流对比） |
| `agg_bench.py` | 聚合吞吐基准 |
| `soak-sglang.py` | SGLang 浸泡巡检（与 `ops/mon/soak.sh` 同族，日志在 /data/eval-rulers/） |
| `results/diag-sglang-fp8-base.json` | FP8 乱码问题的 base 配置诊断输出（§15.4 修复前取证） |
| `results/127.0.0.1_agg_*.json` | 聚合基准两引擎结果（stock / uncensored） |

## 教训沉淀（进 PROBLEMS-AND-FIXES #SGLang 组）

1. **协议完整性先于吞吐**：Anthropic 端点"存在"≠结构化 tool_use 可用——上线前必须用真实
   工具调用报文冒烟，不能只 curl /health；
2. 混合注意力（GDN）新架构上量化前先查 `is_layer_skipped` 这类层名匹配逻辑（子串误匹配
   会静默跳过量化层输出乱码）；
3. 上下文能力以 **KV 池实测 token 数**为准，不信配置标称；
4. 思考型模型跨引擎迁移，reasoning 泄漏是必查项（qwen3_5 需显式 parser）。
