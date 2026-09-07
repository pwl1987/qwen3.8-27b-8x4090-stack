# 智能媒资 + 全线测试素材预置清单（MANIFEST）

> 建立：2026-09-06 ｜ 目的：测试时零现场收集 ｜ **山东方言为主战场**（用户确认）
> 位置：素材统一在 /data/datasets/，模型在 /data/models/

## 一、素材就位状态

### ✅ 已就位

| 素材 | 位置 | 规格 | 服务于 |
|---|---|---|---|
| **临沂广电媒资库直采（API 抓取）** | /data/datasets/lytv_media/（25 条 MP4，287MB）+ lytv_cms_video_list.json（60 条元数据+直链） | 新闻分条 0:16-3:50 为主，1280×720，**真实生产素材**：省运会/时政快讯/简明新闻/沂蒙精神/我为群众办实事等全品类 | **全部 M 系列主力**（短视频单元=编目理想粒度） |
| **直播流录制** | /data/datasets/live_streams/（tv_live_10min.ts 电视 / radio_live_raw.ts 广播） | 电视 HLS 录 10min + 广播录 15min（Python HLS 录制器，ffmpeg 静态版对此流 segfault） | **M11/M12 直播线真实流**；广播音轨=临沂口音 ASR 素材 |
| **临沂方言节目（茶余饭后话临沂）** | /data/datasets/dialect_programs/（m4a 83MB + audio_chayu_20250814_16k.wav 171MB + **chayu_episode_index.json 全量 1178 期索引**） | 2025-08-14 期整档 **89.5 分钟纯音频**（CMS 详情确认 m3u8 即源版本，无更佳 rendition）；AAC-LC 48k 129kbps 无损流复制；16kHz 单声道 s16 WAV 已抽；**同款节目 CMS 存量 1178 期（2021-05~2025-08）可随时扩样** | **M1 方言 ASR 长音频主力**（唯一整档真实方言节目，临沂口音对白+闲谈，含热词素材）；亦可用于 M1 分段/长音频稳定性测试 |
| **整期长节目（files.<内网域名> 直采）** | /data/datasets/full_programs/（8.6G） | 省运开幕式 ceremony_opening.ts **114min（实测6836s）1080p H264 完整**；闭幕式 ceremony_closing.ts **64min（实测3863s，初记42min有误）**1080p（**2132s 处 3.6s 缺口**，段593 504）；沂河夜景演出 yihe_night_show.ts **110min（实测6610s）**720p（**6001s 处 3.6s 缺口**，段1668 504）；临昕新闻广播 radio_linxinews_20251020.mp3 19.9min；缺口已实测确认、其余时间轴连续 | **M5/M6 长视频吞吐压力位**（1080p×114min）；M2/M7 非新闻场景对照（活动/演出）；M1 口音+热词（广播整期）；RTMP 回放源 |
| **人脸库（自建）** | /data/datasets/face_library/（42 人 × ≤8 张裁剪 + index.json 含来源帧） | 由临沂新闻 2 期加密抽帧（4s 间隔 ×600）+ lytv_media 25 条分条抽帧（2s ×~1500）→ SCRFD 检测（≥80px、score≥0.6，共 842 张合格脸）→ ArcFace 嵌入贪心聚类（cos≥0.45、簇≥4 张）产出 **42 人**；簇纯度已复验（簇内平均相似度 0.767、无一簇均值<0.45）；人物01=主播(70帧) 人物02=主播(55帧) 其余为领导/嘉宾 | **M3 全流程可跑**（检测+聚类+识别）；指名识别仅差人工命名（人物XX→真名） |
| **危险动作/风险样本（自采）** | /data/datasets/risk_samples/（16 条 MP4，292MB） | 烟花实拍×4（古城烟花铁花/蒙童烟花狂欢/夜色临沂/盒子灯非遗）；火灾×4（酒店火灾科普/干粉灭火器/绿化带真实火情/航空救援纪录片）；事故演练×7（防汛/森林火情/交通事故/工贸事故/危化品×2/高层建筑，含模拟伤亡跌倒画面）；真实救援×1（沂水落水救援）；另有 ceremony_opening.ts 开幕式焰火（本地已存） | **M4 风险识别主力**；打斗类仍缺（自有库无此内容，备选 XD-Violence 或后续搜"公安演练"）；检索关键词：消防/演练/火灾/救援/事故/烟花 共约 800+ 条可扩 |
| **琅琊网自办节目全量索引** | /data/datasets/langya_programs/langya_episode_index.json | 入口 langya.cn/aly/index.html，17 个自办节目（临沂新闻/首发聚焦/首发直通车/行风热线/沂蒙先锋/健康临沂/**百姓调解**/临沂教育/畅游临沂/沂蒙金盾/临沂金融/平安临沂/临沂政务报道/成长起跑线/老爸老妈的美好生活/沂蒙军号/乡村振兴沂蒙行），**5320 集 / 5265 条 files.<内网域名> m3u8 直链**（公开页面无需 token，存档至 2022 年底；部分节目 403 集为爬虫页数上限可加深）。实测百姓调解=720p H264+AAC 约 19min/期 | **全 M 系列按需取材**：百姓调解=方言对话视频（M1 方言 ASR 扩样+M3 路人脸+M4 纠纷场景）；临沂新闻=整期补充；行风热线=访谈语音；爬虫脚本可增量重跑 |
| **临沂新闻整期（用户直供）** | /data/datasets/news_videos/linyi_news_2026090{4,5}.mp4 | 2×20min 1080p 广播级；帧 120×2 + 16kHz 音轨已抽 | M 系列整期场景 |
| **B 站新闻对照素材** | /data/datasets/news_videos/news_BV*.mp4（1.2GB，4 期） | 720p；帧+音轨已抽 | 跨台包装风格对照 |
| **KeSpeech 方言测试集** | /data/datasets/KeSpeech（3.3GB） | 8 方言含冀鲁/胶辽/**中原（临沂话属中原官话区，同为重点）** | M1 方言 ASR |
| **GigaSpeechBench** | /data/datasets/GigaSpeechBench（**79GB 已完整**，2026-09-06 核验） | CH-EN-Dialects 12 子集（6方言+6口音，各0.8-1.4G）+ Vertical-Domain 24 子集（12域×中英）+ Low-Resource 14 子集；.cache 内 155 个 *.lock 为断点残留锁，无害可清 | M1 通用方言 + 热词 |
| **OmniParser** | /data/models/OmniParser（8.1GB） | UI 元素检测 + 图标模型 | S3 视觉级联 / E1 |
| ASR-Testset 汇总 | /data/datasets/ASR-Testset（**仅 README 索引 28K**，未拉实际音频） | 仓库本体=开源测试集跳转索引；ks-Ji-Lu / ks-Jiao-Liao 实际数据在 KeSpeech（已就位 3.3GB），此仓库仅备用 | M1 备用 |

### ✅ 无进行中下载（2026-09-06 全量核验：素材侧全部落盘，ffprobe 可解码）

### ⏳ 待用户提供（唯一外部依赖）
| 素材 | 用途 | 说明 |
|---|---|---|
| **人脸库指名**（人物01~42 真实姓名标注） | M3 指名识别 | ✅ 库已自建（见下），仅需人工过一遍 42 个文件夹改名为真实姓名（主播/领导为主，几分钟工作量）；不标也不影响 M3 跑"检测+聚类+识别" |
| ~~现役素材库样本~~ | — | ✅ 已提供（临沂新闻 2 期，files.<内网域名>） |

### ⏳ 自产待做（不依赖外部）
| 素材 | 方案 | 服务于 |
|---|---|---|
| 网页截图集（10-20 张） | Playwright 对真实站点截图（浏览器内核需过镜像下载，待做） | E1 网页复刻 |
| ~~新闻关键帧集~~ | — | ✅ 已产出（临沂 240 张 + B 站 2600 张） |
| RTMP 模拟流 | ffmpeg -re 推流回放新闻视频（测试时一条命令） | M11/M12 直播线 |
| 热词表（临沂人名地名机构名） | 从临沂新闻 ASR 转写中抽取 + 人工补 | M1 热词命中 |

## 二、实验→素材映射（测试当天即取即用）

| 实验 | 直接取用 |
|---|---|
| M1 ASR（山东方言） | **dialect_programs（茶余饭后话临沂 89.5min 整档）** + **langya 百姓调解（方言对话视频，按需下）** + KeSpeech/冀鲁+胶辽子集 + GigaSpeechBench 热词表 + news_videos 抽音频 + full_programs 临昕广播（口音/热词） |
| M2/M7 镜头/条目 | news_videos 4 期 + full_programs（活动/演出非新闻场景对照） |
| M3 人脸 | **face_library（自建 42 人）** + news_videos/lytv_media 全部抽帧 |
| M4 场景/危险 | **risk_samples（16 条）+ full_programs 开幕式焰火** + news_videos 抽帧（正常场景对照） |
| M5/M6 端到端/吞吐 | news_videos 全量 + **full_programs ceremony_opening（1080p×114min 压力位）** |
| M8 统一模型横评 | 关键帧集（自产） |
| M9 检索质量 | M5 产出的编目库 |
| M11/M12 流式/直播 | RTMP 模拟流（自产，回放源用 news_videos/full_programs）+ live_streams 真实流 |
| E1 网页复刻 | 截图集（自产） |
| T1.1 质量 A/B | quality_ab_v11.py（已备）+ :8000/:19622 |

## 三、素材生成工具链备注
- **目录约定（新素材一律照此落位）**：`lytv_media/`=CMS API 新闻分条（短 MP4）；`news_videos/`=整期新闻（本台+B站对照，附 audio/ 16k 抽轨 + frames_*/ 抽帧）；`dialect_programs/`=方言节目整档（M1）；`full_programs/`=活动/演出/广播整期长节目（files.<内网域名> 直采）；`live_streams/`=直播流录制。yt-dlp 下载的 `*.f数字.*` 流中间文件合并后即删。
- **媒资 API 通道（已打通，2026-09-06）**：登录 `POST https://api.<内网域名>/private/login`，密码 **DES-ECB-PKCS7 加密**（密钥 `<已脱敏，见内部密码库>`，Base64 输出），载荷 {username, password, phone_number:"", remember_login:false, verification_code:""}；token 存 cookie `token=` 或头 `X-CSRF-Token:`；视频列表 = `GET https://cms.<内网域名>/cms/video?page=N&size=20`（服务端渲染 HTML 表格，带 cookie）；单条详情+MP4 直链 = `GET https://cms.<内网域名>/cms/video/{id}`（需 X-Requested-With: XMLHttpRequest）。token 缓存 /tmp/ms_token.txt。
- **音频通道 /cms/audio（已打通，2026-09-06）**：列表 = `GET https://cms.<内网域名>/cms/audio?page=N&size=5&category=0&status=处理成功&keyword=X`（同 HTML 表格模式，行 ID 在 `<tr data-info='{"id":...}'>`）；详情 = `GET /cms/audio/{id}` + X-Requested-With → JSON `versions[]` 直链（短素材=MP3 直链；整期节目=m3u8，配 hls2m4a.py 合成）。**坑：keyword 中文必须 URL 编码**（curl 用 `-G --data-urlencode`，裸 UTF-8 进 URL 会静默返回 0 行）。库规模 ~6.9 万条：分类「媒体」= 每日整期广播节目存档（2021 至今：茶余饭后话临沂/蒙山夜话/今晚有约/101汽车帮/976大家帮/母爱好时光等，1-2h/期），「音频素材」= 短音频（1-6min）。临昕新闻不在库中。
- **图片通道 /cms/picture（已验证，2026-09-06）**：列表页结构同上，但**库当前为空**（0 条、无翻页）；页面 JS 显示另有 `/private/media-pictures` JSON API 但网关 404（未部署）。待有素材入库后即可用列表+详情模式取图。
- **方言节目全量索引**：`/data/datasets/dialect_programs/chayu_episode_index.json` = 茶余饭后话临沂 **1178 期**（2021-05-01～2025-08-14，id+标题+时长），取任意一期 = 查 id → `GET /cms/audio/{id}` 拿直链 → 下载。
- **自产工具脚本**（/data/tools/，均 pyav-env 运行）：`extract_frames.py`（视频抽帧）；`build_face_library.py`（帧目录→SCRFD+ArcFace→聚类人脸库，模型在 /data/models/insightface/models/buffalo_l）；`fetch_cms_videos.py`（CMS id 批量取直链下载 MP4）；`crawl_langya_programs.py`（琅琊网自办节目索引爬虫，公开页无需 token，可增量重跑）；`hls2m4a.py`/`any2wav16k.py`/`hls_probe.py`（见 ffmpeg 条目）。**批次H 新增（video-use 借鉴，2026-09-07）**：`timeline_qa.py`（决策点合成图：filmstrip+RMS+ASR 标签，pyav-env）、`packed_md.py`（catalog→26.8KB agent 可读层）、`story_selfcheck.py`（条目自评环：主播声纹跨度+LLM 整条目合并，边界 F1 0.88，funasr-env+2B:8013）、`align_words.py`（词级对齐 4964字/20min，**align-env**=qwen-asr transformers 后端）、`edl_export.py`+`render_story.py`（EDL 拆条导出：词边界吸附/30ms fade/字幕最后挂/分段单编码无损 concat）。
- **批次I 帧域闭环工具（2026-09-07，详见 RESULTS.md 批次I）**：`framedomain.py`（帧域唯一换算库+stories_final schema gate）、`frame_refine.py`（帧信号/候选/镜头落帧/refine，pyav-env）、`story_selfcheck.py v5`（语义候选+27B 终审 `--llm27b`，funasr-env）、`news_pipeline.py`（11 阶段 DAG 一键编排：原子缓存/Run Lock/GPU 守卫{2,3}/serve PID 归属）、`edl_export.py v2`（双域 EDL）、`render_story.py`（帧断言）、`validate.py`（不变量+帧身份判别式 SSIM）、`evaluate.py`（边界误差方向性报告）、`boundary_review.py`（±3s 回放+解释卡）、`catalog_reindex.py`（final 口径重索引）、`batch-I-config.json`（阈值全外置，参与缓存指纹）；单测 `tests/test_framedomain.py`（python3 直跑）。
- **琅琊网节目通道（2026-09-06 打通）**：langya.cn/aly/<节目拼音>/index.html 分页 index_N.html（~13集/页），文章页内嵌 files.<内网域名> m3u8+封面 jpg 直链（公开）。视频节目 m3u8 含 H264+AAC，下载后 hls2m4a.py 合成；音频节目（如茶余饭后话临沂）为 audio-only rendition。
- ffmpeg：~~宿主无系统 ffmpeg~~ → **已修复（2026-09-06）**：`/data/tools/ffmpeg-deb/ffmpeg`（Ubuntu noble 官方 deb 6.1.1 + 全依赖解包 root/，含 libjack/lapack 等 100+ 包），本地解码/HLS 网络/ffprobe 全验证通过；另有软链 /data/tools/ffmpeg。imageio-ffmpeg 静态二进制仍不可用（任何 demux 输入 SIGSEGV）。音视频 Python 处理走 `/data/tools/pyav-env`（PyAV）。**torch venv 首跑 triton 编译需 CPATH=/data/tools/ffmpeg-deb/root/usr/include/python3.12:/data/tools/ffmpeg-deb/root/usr/include**（python3.12-dev 头文件已解包于此）。venv 清单（2026-09-07 更新）：pyav-env(音视频+insightface+PIL)/**vllm28-env(M 线新底座：vLLM 0.28.0+tf5.16.1+torch2.13，ASR/VL serving 全走这里，部署要点见 RESULTS)**/**align-env(qwen-asr transformers 后端，ForcedAligner 专用)**/qwenasr-env(遗留，勿再用)/funasr-env(FunASR 1.4.14+FireRedASR2+SeACo+CAM++)/funasr-main-env(funasr git main，ERes2NetV2 备用)/mage-env(Mage-VL)/ocr-env(PaddleOCR 3.7)。4090 实测数据：/data/datasets/bench_outputs/RESULTS.md + MODEL-PICKS 实测表
- **新素材/产物（2026-09-06 晚）**：`bench_outputs/video_set/`=真视频横评测试集 23 条（A风险10×45s/B新闻10×45s/C长段3×60s，统一 720p H.264，manifest.json 含 prompt 与 GT 提示）；`bench_outputs/m1_kespeech/`=M1 决赛子集（冀鲁100+胶辽100，gt.jsonl+hyp_*.jsonl）；`bench_outputs/chayu_prelabel/`=茶余饭后双模型预标 20 片 + REVIEW.md 人工校对清单；`bench_outputs/video_results/`=各模型横评 jsonl。新模型落位：Qwen3.6-27B-AWQ-INT4(20G)/Qwen3.6-35B-A3B-AWQ(19G)/GLM-4.6V-Flash-AWQ(8.3G)/MiniCPM-V-4_5-AWQ(6.7G)/FireRedASR2-AED(4.5G)/GLM-ASR-Nano-2512(4.3G)/Fun-ASR-Nano-2512-hf/AuraFace-v1/PaddleOCR-VL-1.6B；代码仓 FireRedASR2S-code（注意 FireRedASR 老仓不含 v2）。
- B 站下载：yt-dlp 2026.08.19，**必须用 av 号链接 + 浏览器 UA**
- HF 下载：`hf download` + `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`

- **新工具（2026-09-07）**：shotseg/（M2 镜头分割四件套+编排）、m7_stories*.py（条目拆分 v1/v2）、anchor_voice_build.py（主播声纹库）、pipeline_e2e.py（端到端五段 asr/faces/vision/semantic/catalog）、video_bench.py、bench_queue.sh（横评自动换装）、face_candidates_bench.py、wemm_retrieval_poc.py、make_registry.py、serve/*.sh+stop.sh、/data/models/registry.json+restore.sh、/data/backup/models/mline-keyset-20260907.tar(18G)；voice_library/（主播声纹）；新模型 Qwen3.5-0.8B/2B/4B、Video-ORA-9B、WeMM-2B、SigLIP2、MiniCPM-V-4.6-AWQ、SFace/dlib 模型已入册 registry.json

## 四、来源链接（备查）
- **模型选型预设**：/data/datasets/MODEL-PICKS-2026-09.md（人脸/方言ASR/声纹/OCR/VL 五域甜点位选型 + 测法）
- **感知能力全景**：/data/datasets/PERCEPTION-MATRIX-2026-09.md（四层级联架构 + 能力→模型映射：镜头结构/语义/文字/人物/音频/技术QC/审校辅助/镜头语言）
- KeSpeech: huggingface.co/datasets/TwinkStart/KeSpeech ｜ 论文 openreview.net/pdf?id=b3Zoeq2sCLq
- ASR-Testset: huggingface.co/datasets/xuyaya/ASR-Testset（ks-Ji-Lu / ks-Jiao-Liao 划分）
- GigaSpeechBench: huggingface.co/datasets/speechcolab/GigaSpeechBench
- 讯飞方言库（山东 432h，商业申请）: xfyun.cn/solutions/datarequest
- 国家基础学科中心方言音库（含山东话）: nbsdc.cn/general/dataDetail?id=666066f3195d266d328f1d69
