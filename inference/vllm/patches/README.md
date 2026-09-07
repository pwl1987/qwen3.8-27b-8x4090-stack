# patches/ — vLLM 0.27.1 的 27 个栈补丁

构建时逐个 `patch -p1` 应用（Dockerfile.cu129 内置），`_check_applied.py` 做内容级校验。
补丁头注明 "Written against vLLM 0.27.1. Reapply after upgrades"——0.28 迁移时逐个重验
（ROADMAP 方向 C）。按功能分组：

## DFlash2 投机解码（核心）
| 补丁 | 作用 |
|---|---|
| `dflash2-backport` | DFlash2 块式 drafter 整体回移（上游 PR #52816）：5 层滑窗 conv + fc + CandidateSelector + V2 runner；含 packed 权重反量化路径 |
| `dflash2-lookup-drafting` | LABD 上下文查找式起草：最长后缀匹配 + 置换信度融合；adaptive 8↔16 长块（**与 KVarN 前缀缓存有已知损坏，P48**） |
| `dflash2-ngram-chains` | drafter-free n-gram 链（_CHAIN=1，需 LOOKUP=1，greedy-only） |
| `dflash2-prewarm` | speculator 预热 |

## KV 与显存
| 补丁 | 作用 |
|---|---|
| `int4-kv-per-token-head` / `spec-decode-int4-kv-mq3d` / `spec-decode-int8-kv` | KV 量化族：K4V2/int4 多查询验证/int8 per-token-head |
| `hybrid-kv-groups-v2-cudagraph` | 混合 KV 组与 V2 runner 的 CUDA 图协同 |
| `hybrid-sw-block-promote` | drafter 滑窗层块提升（免 385 个近空块） |
| `vllm-pr50021-gdn-spec-bounds` | GDN spec 解码边界（上游 PR 回移） |
| `offload-dflash-eagle-groups` / `offload-wsl2-devptr` | offload 族 |
| `vision-tower-cpu-offload` | 视觉塔 CPU offload（KV_MEM 配平的搭档） |

## 内核与速度
| 补丁 | 作用 |
|---|---|
| `marlin-int8-layer-select` / `marlin-int8-negative-scales` / `marlin-repack-staged-sm80` / `marlin-tune-table` | Marlin W4/W8 内核族（分层选择/负 scale/分级 repack/调优表） |
| `spec-decode-attn` | split-KV 多查询验证注意力（bf16 KV 档） |
| `triton-prefill-attn-int8` | int8-QK 预填注意力 |
| `sampler-small-topk-fast-softmax` / `spec-sampler-prewarm` / `spec-decode-scratch-token-units` | 采样与 scratch 族 |
| `speed-knobs-envs` | 性能旗标 env 化 |

## 模型结构
| 补丁 | 作用 |
|---|---|
| `qwen3_5-embed-quant` | quant_config 穿透到 embed/MTP 的 VocabParallelEmbedding（否则 packed 崩） |
| `qwen3_5-mtp-draft-vocab` | MTP 截断 draft 头（40960 词表 + mtp_draft_vocab_ids.pt） |
| `mamba-align-checkpoint-order` | GDN state 检查点顺序对齐 |
| `xgrammar-spec-terminated` | 投机解码下的结构化输出终止符处理 |
