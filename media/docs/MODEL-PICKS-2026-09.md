# M 系列模型选型（2026-09-06 v2 换代版）

> 原则：不存在"效率+性能+资源"三全的单一模型，每任务选 2-3 个甜点位，用自有素材实测定夺。
> **硬约束（用户 2026-09-06 定）：商用许可 Apache-2.0/MIT 一票过滤。**
> 硬件：4090（M 线用 GPU2/3）｜推理底座：vllm28-env（vLLM 0.28.0）｜模型源：hf-mirror + ModelScope
> 实测原始数字与部署坑：/data/datasets/bench_outputs/RESULTS.md

## ✅ 五域换代定案（2026-09-07 凌晨终版，端到端已贯通）

### 视频三档+轻量王（9 模型真视频终榜，全部 26/26）
| 档 | 模型 | 速度 | 用途 |
|---|---|---|---|
| 画质档 | Qwen3.6-27B-AWQ（TP2） | 71.1s/条 | 精编目/封面帧精析 |
| **中坚档★** | **Video-ORA-9B**（Apache，单卡 17.6G） | 23.4s/条 | VideoMME76.7、时序事件定位 |
| 吞吐档 | MiniCPM-V4.5-AWQ | 5.4s/条 | 批量字幕/初筛 |
| **轻量王★** | **Qwen3.5-2B**（~5G） | 7.4s/条 | **M2 镜头分类+e2e 画面分析主力**（衣着/地标/标识 JSON 全对） |
- 出局：Qwen3.5-4B(27.5s 反慢)/0.8B(质量)；MiniCPM-4.6-AWQ(vLLM 不兼容挂账)；GLM-Flash(备选)

### 端到端验证（用户蓝图五层全通）
镜头分割(SigLIP2 三信号,金标召回100%) → 条目拆分(演播室画面×主播声双确认,11vs10条) → 说话人角色(CAM++ 3主播库) → 语义编目(5W1H+文字稿,11/11完整) → 标识识别(旗帜/牌匾/横幅/带字 JSON) + 人脸(AuraFace 127帧命中) + 跨模态检索(WeMM hit@1 50%)

### 人脸终局（四件套实测，42 人库）
AuraFace(MIT) 0.185/74% 🏆 > buffalo_s 0.092/57%(非商用) > **dlib(CC0) -0.027/40% 崩盘** > **SFace(Apache) -0.056/40% 崩盘**——商用干净池无一敌手；DeepFace 判定=聚合器仅取后端（8/11 后端许可污染）；SeetaFace6 待模型可得补测



| 域 | 模型 | License | 实测 | 结论 |
|---|---|---|---|---|
| ASR 🏆 | **FireRedASR2-AED** | Apache-2.0 | KeSpeech 冀鲁+胶辽 200 条 **CER 5.07%**、RTF 0.0090 | **精度+速度双冠，方言主力换代**（代码=FireRedASR2S 仓库；2S 一体化=VAD+LID+标点） |
| ASR | Qwen3-ASR-1.7B | Apache-2.0 | CER 8.62%、RTF 0.0125；45s 分片已修 | 服务化/流式/30 语种档，与 FireRed2 混部 |
| ASR | GLM-ASR-Nano-2512 | MIT | CER 18.16%（幻觉前缀） | ✗ 方言出局 |
| ASR | SenseVoice-Small / SeACo | ⚠️ FunASR 自定义协议 / Apache | RTF 0.003 / 0.013 | SenseVoice 待法务（降参照）；SeACo 热词实测无显著收益 |
| 对齐 | Qwen3-ForcedAligner-0.6B | Apache-2.0 | 字级，RTF 0.033，零显存 | ✓ 不变 |
| 视频画质档★ | **Qwen3.6-27B-AWQ-INT4** | Apache-2.0 | **决赛 26/26，71.1s/条（双卡TP2）；内容质量最强**（角标日期地名/记者署名/人名条全抠出）；官方 VideoMME 87.7 | 画质档定案；需 reasoning parser 剥思考 |
| 视频吞吐档★ | **MiniCPM-V4.5-AWQ** | Apache-2.0 | 26/26，**5.4s/条 = 27B 的 13 倍速**（96× token 压缩），字幕细节最强（编辑 staff 字幕都抠出） | 吞吐档定案；think 需剥 |
| 视频轻量 | GLM-4.6V-Flash-AWQ | MIT | 26/26，20.2s/条 | think 泄漏+时序尺度崩（60s 片标 11:00）；备选 |
| 视频暂缓 | Qwen3.6-35B-A3B-AWQ | Apache-2.0 | 26/26，67.1s/条——**单请求不比 27B 快**（MoE 优势需并发批量兑现） | 挂账 M6 并发实测后再议 |
| 视频实证 | Qwen3.8-27B-FP8（生产同源） | Apache-2.0 | 26/26，90.1s/条（双卡TP2）；抽取细节与 3.6 同级 | **疑案了断：视频能用**；但画质档仍推 3.6 |
| 视频基线 | Qwen3-VL-4B（上代） | Apache-2.0 | 26/26，7.2s/条 | 开箱结构最干净、片内时间尺度正确；Qwen3-VL 全线已被 Qwen3.5/3.6 同规模超越 |
| 流式 | Mage-VL 4.7B | Apache-2.0 | 图片 10.2G/视频 20.7G | 仅 M11 直播线（vLLM 无支持） |
| 人脸 🏆 | **AuraFace-v1 (glintr100)** | **MIT** | 42 人库判别分离度 **0.189 / 71%** 正确分离，28ms/张 CPU | **商用切换净赚**（buffalo_s 仅 0.076/55%）→ 生产主力 |
| 人脸 | buffalo_l/s | ❌ 非商用 | 检出/速度基线 | 降级内部测试 |
| 声纹 | CAM++ | Apache-2.0 | 茶余饭后 120 段内外差 0.385、56ms/段 | ✓ 主力；生产建库须 VAD 段+凝聚聚类 |
| 声纹 | ERes2NetV2 | Apache-2.0 | 0.334、168ms/段；funasr 1.4.14 已带注册修复 | 精度档可用（旧"未注册"结论作废） |
| OCR | **PP-OCRv6-medium** | Apache-2.0 | 5487ms/帧 CPU（同帧 v5-mobile 5508ms），精度官方+5% | ✓ v5-server △ 关闭；v6 只有 tiny/small/medium 三档 |
| 文档解析 | PaddleOCR-VL-1.6B | Apache-2.0 | 已下载 /data/models/PaddleOCR-VL-1.6B | M3 文档线落地（vLLM serving） |
| 淘汰 | TeleSpeech（停滞）/ MiMo-VL（vLLM 无支持）/ Kimi-VL bf16（超卡）/ LongVA（非商用） | — | — | — |
| 待测 | Fun-ASR-Nano-2512 | Apache(留档) | funasr 1.4.14 内置其 vLLM 管线 | 复活待测；FireRedASR2-LLM（8.3B 精度天花板档） |

## 部署速查（vllm28-env，全坑见 RESULTS）
```bash
# ASR（原生，无需 qwen-asr wrapper）
CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$V/site-packages/nvidia/cu13 PATH=$CUDA13/bin:$V/bin:$PATH \
VLLM_USE_FLASHINFER_SAMPLER=0 CPATH=.../python3.12:.../include \
$V/bin/vllm serve /data/models/Qwen3-ASR-1.7B --port 8010 --gpu-memory-utilization 0.85
# FireRedASR2：PYTHON FireRedASR2S 代码仓 + FireRedAsr2Config(use_gpu=1,use_half=1)
# 27B 视频画质档：--tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 49152 \
#   --max-num-batched-tokens 8192 --enforce-eager --limit-mm-per-prompt '{"video":1,"image":1}'
# MiniCPM：加 --trust-remote-code；GLM：先改 video_preprocessor_config.json longest_edge=31457280
```

## M3 人脸（换代后）
| 档 | 模型 | License | 备注 |
|---|---|---|---|
| **生产主力★** | **AuraFace-v1**（scrfd_10g 检测 + glintr100 识别） | MIT | 判别力反而强于 buffalo 2.5 倍；42 人库已验证 |
| 内部测试 | buffalo_l / buffalo_s | ❌ 非商用 | 商用须 InsightFace 企业授权 |
| 备选 | SeetaFace2 | BSD | 精度老 |

## M1 方言 ASR（决赛已出）
| 档 | 模型 | License | KeSpeech 200 条 |
|---|---|---|---|
| **精度+速度双冠★** | **FireRedASR2-AED**（1.15B 参数） | Apache-2.0 | **CER 5.07% / RTF 0.0090** |
| 服务化档★ | Qwen3-ASR-1.7B | Apache-2.0 | CER 8.62% / RTF 0.0125；流式/30 语种/混部 |
| 对齐★ | Qwen3-ForcedAligner-0.6B | Apache-2.0 | 字级时间戳 AAS 42.9ms |
| 流式（M11） | Dolphin-CN-Dialect 0.4B | Apache-2.0 | 22 方言流式 |
| 旗舰待测 | FireRedASR2-LLM（8.3B）/ Fun-ASR-Nano | Apache | CER 天花板档视需要补测 |

金标：KeSpeech GT（现成）✅ 已用；茶余饭后 GT v0 = 双模型预标 + 人工校对（REVIEW.md 已生成，小分歧3/大分歧16/音乐1）。

## VL/视频（M8 真视频横评，23 条自有素材）
| 档 | 模型 | License | VideoMME | 部署 |
|---|---|---|---|---|
| **画质档★** | Qwen3.6-27B-AWQ | Apache-2.0 | **87.7**（千问系王者，超 397B 旗舰 87.5） | 双卡 TP2（单卡 KV 0.85G 不够 45s 视频，实测） |
| 吞吐档★ | MiniCPM-V4.5-AWQ（8.7B） | Apache-2.0 | 73.5（效率王） | 单卡 6.7G 权重 |
| 轻量备选 | GLM-4.6V-Flash-AWQ（9B） | MIT | ~69 | 单卡，需改视频像素预算 |
| 流式（M11） | Mage-VL 4.7B | Apache-2.0 | 64.0 | codec 原生，transformers |
| 上代基线 | Qwen3-VL-4B/8B | Apache-2.0 | 71.4(8B w/o sub) | 已被 Qwen3.5/3.6 同规模超越，换代 |
| 出局 | Kimi-VL-A3B（bf16 超卡）/ MiMo-VL（无 vLLM 支持） | MIT | — | — |

注：Qwen3.8-27B（本机生产模型）官方零视频基准，不入选视频线；如需用须先过自有视频评测。

## 声纹（换代无变化，修一处）
CAM++ 主力（Apache）+ ERes2NetV2 精度档（funasr 1.4.14 已带注册修复）。建库法：方言节目音频 → **VAD 段切分** → 嵌入 → 凝聚聚类（固定 5s 窗 2-means 会退化，实测 [3,117]）→ 说话人簇。

## OCR / 场景
- **PP-OCRv6-medium**（Apache）：CPU 5.5s/帧与 v5-mobile 持平、精度+5% → v5-server △ 关闭；paddleocr 3.7 包自带（模型名 PP-OCRv6_medium_det/rec）。
- **PaddleOCR-VL-1.6B**（Apache，OmniDocBench 96.3 SOTA）：已下载，文档解析线 M3 落地（vLLM serving + paddlepaddle-gpu 跑 DocLayout）。
- 场景检索轻量档：Chinese-CLIP（后续）。

## 附录：设计缺口清单（v2 新增，未闭环项挂账）
1. **无模型能力项**（PERCEPTION-MATRIX 待补）：logo/台标检测（转播来源识别）、地标与旗帜细分类头、昼夜/天气分类、音频事件模型二选一（SenseVoice 标签 vs PANNs）、L1 镜头切割器落地实现（三信号级联定案但未施工）。
2. **商用遗留**：SenseVoice-Small 协议法务复核；InsightFace 若坚持精度档需企业授权。
3. **工程债**：CMS 图片通道 /private/media-pictures 网关 404（台里部署）；打斗类风险样本仍缺（备选公安演练关键词再采）；网页截图集 10-20 张（E1 用）；RTMP 模拟流（M11 时建）；27B 双卡批处理需配 reasoning parser + enable_thinking:false（英文 think 泄漏已实证）。
4. **待人工**：人脸库 42 人真名指名；茶余饭后金标校对（REVIEW.md）。
5. **挂起实验**：Fun-ASR-Nano（funasr 内置管线待跑）；FireRedASR2-LLM 旗舰档；35B-A3B 并发批量吞吐实测（M6）；Mage-VL M11 codec/流式门控。
6. **video-use 借鉴落地（2026-09-07 批次H，详见 RESULTS）**：browser-use/video-use 为对话式剪辑技能（MIT），与 M 线互补；已固化四件——timeline_qa 合成图（QA 界面层）、packed.md（catalog 75k token→26.8KB agent 可读层）、story_selfcheck 自评环（边界 F1 0.67→0.88；**0.8B 判别不可靠实证，整条目级判断 2B 起步**）、edl_export+render（词边界吸附/30ms fade/字幕最后挂——剪辑工程规则清单引用其 SKILL.md 12 条硬规则）。ForcedAligner 现役于 align-env（qwen-asr 包，20min/4964 字实测）。
