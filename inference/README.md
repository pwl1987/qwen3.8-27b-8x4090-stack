# inference/ — 推理双引擎（+一个已弃用存档）

| 引擎 | 目录 | 状态 | 定位 |
|---|---|---|---|
| llama.cpp | `llamacpp/` | **生产主力** | 4 副本 256K + OpenResty 会话粘滞 LB，唯一入口 :8000（OpenAI+Anthropic 双协议）；72 tok/s 单流 / 420 聚合 |
| vLLM | `vllm/` | **单卡高速通道** | coding-v1.1 后训模型，DFlash2+KVarN，130 tok/s @240K（1.45× 单副本且省一卡） |
| SGLang | `sglang/` | 已弃用存档 | 2026-09-03 因 Anthropic 协议无结构化 tool_use 回归 llama.cpp；探索全记录+基准脚本保留 |

选型速查：单流延迟/256K 极限显存形态 → llama.cpp；后训模型单卡吞吐 → vLLM；
多 LoRA 并行门禁（SGLang 独有）→ 等协议补齐再议（P18 教训）。

两引擎共用：`/data/models` 权重池、ops/ 监控与功耗墙、eval/ 各自门禁。
