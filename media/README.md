# media/ — 媒体生产线（M 线）

自动新闻生产系统的执行层：入库编目（镜头分割→ASR→词对齐→人脸→OCR→语义拆条）→ 混合检索 →
自动成片（coding-v1.1 脚本 + EDL 装配 + ComfyUI/MiniMax-H3 补空镜）。推理底座与
`inference/` 共用（27B 主力 + GPU5/7 周边池），文档在 `docs/` 子目录。

```
comfyui-minimax-h3/   ComfyUI 0.34 + MiniMax-H3 视频生成（GPU2/3 双卡，按需启停）
services/             检索/视觉周边池三容器（embed:19623 / rerank:19624 / VL-8B:19625）
news_pipeline/        批次I 一键 11 阶段编目管线（P0-P4 全套脚本 + 阈值唯一真源 config）
serve/                8 个 vLLM 服务启停脚本（ASR/2B/8B/27B-TP2/GLM/MiniCPM/Ora）
docs/                 RESULTS（滚动实测）/ MODEL-PICKS（五域选型定案）/ PERCEPTION-MATRIX
                      （能力→模型映射矩阵）/ MEDIA-TEST-MANIFEST（素材清单）/ 茶余饭后预标校对
```

## 当前形态速览

- **编目管线**：`news_pipeline.py` 一键 11 阶段，原子缓存（config 哈希指纹，改阈值自动失效）、
  Run Lock、GPU 守卫（仅 2/3 卡、空闲显存 <10G fail-fast）；F1 边界质量 0.94；
  LLM 仅作证人非主权者（2B 语义层 + 27B 终审通道）
- **方言 ASR**：FireRedASR2-AED 主力（CER 5.07% / RTF 0.009，KeSpeech 决赛第一），
  Qwen3-ASR-1.7B 亚军（服务化优势），走 vllm28-env 新底座
- **视频生成**：MiniMax-H3 双卡 fp8（t2v 768×432 8 步 40s 出片），turbo LoRA 加速
- **检索**：Qwen3-Embedding+Reranker（E2E 中文查询命中 0.06-0.08s）

详细实测与决策记录：`docs/RESULTS.md`（滚动）与 `docs/MODEL-PICKS-2026-09.md`（定案）。
