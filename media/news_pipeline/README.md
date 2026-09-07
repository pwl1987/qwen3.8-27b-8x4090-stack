# news_pipeline — 批次 I 一键编目管线（P0-P4 全套）

`news_pipeline.py`：电视节目 → 结构化编目（节目→故事条目→镜头→词级时间轴）的一键编排器。

## 11 阶段 DAG

shotseg（三信号级联：SigLIP 渐变+音频突变+OCR 字幕条）→ asr → words（词级对齐）→ … →
故事拆条（20s 语义单元 + LLM 窗口）→ validation。LLM 是证人非主权者：
Qwen3.5-2B 语义层（:8010，自起自停按 PID）+ 27B 终审通道（可升级 url 在 config）。

## 工程机制（都为"改阈值不用重跑全链"设计）

- **原子缓存**：指纹 = batch-I-config 哈希 + 输入 path/size/mtime_ns；上游变下游自动失效
- **状态机**：RUNNING→SUCCESS，Run Lock（O_EXCL+PID 双实例 fail-fast）
- **GPU 守卫**：仅允许 GPU2/3，空闲显存 <10G fail-fast
- **每阶段独立 venv**：pyav-env / funasr-env / align-env（互不污染）
- **阈值唯一真源**：`batch-I-config.json`（schema_version 1）——帧域容差、siglip 渐变参数、
  候选窗先验、OCR 人名后缀表 chyron_suffixes + 电视台正则、镜头评分权重、
  validation 不变量（words_coverage ≥0.97、ssim ≥0.8）

## 脚本分层

| 层 | 脚本 |
|---|---|
| P0 换算 | `framedomain.py`（帧域换算唯一真源） |
| P1/P2 检测 | `shotseg/`（run_shotseg.sh + stage1_siglip/stage2_audio/stage3_ocr + fuse_shots）、`frame_refine.py` |
| P2a/P3 | `story_selfcheck.py`（语义候选 v5）、`news_pipeline.py`（编排） |
| P4 验收 | `boundary_review.py`（边界回放合成图）、`evaluate.py`（边界误差+混淆矩阵）、`validate.py`（不变量）、`storysplit_validate.py`(shotseg/) |
| 产出层 | `catalog_reindex.py`、`edl_export.py`（stories→EDL 帧域+ms）、`render_story.py`（EDL→MP4）、`packed_md.py`（catalog→agent 可读）、`timeline_qa.py`（决策点合成图） |
| 词对齐 | `align_words.py`（Qwen3-ForcedAligner 词级时间轴） |

质量基线：F1 边界 0.94；测试集与黄金集见 `../docs/MEDIA-TEST-MANIFEST.md`（M7 拆条黄金集）。
