# 4090 实测结果（滚动记录）

## 🔥 新闻结构化全流程落地（2026-09-07 凌晨，用户蓝图版端到端）

### 批次A｜视频横评终榜（9 模型 × 26 真视频查询，全部 26/26）
| 模型 | avg/p50 s | 定位 |
|---|---|---|
| MiniCPM-V4.5-AWQ | **5.4**/6.7 | 吞吐档速度王（字幕细节最强） |
| Qwen3-VL-4B | 7.2/7.3 | 上代基线 |
| **Qwen3.5-2B** | 7.4/6.9 | **轻量王**（~5G，中文最强，a07 连"声音来源…刘树朋"角标全抠出；**M2 镜头分类+e2e 画面分析主力**） |
| Qwen3.5-0.8B | 10.9/6.8 | 仅路由级（VideoMME 57.7） |
| GLM-4.6V-Flash | 20.2/21.2 | 备选（时序尺度崩） |
| **Video-ORA-9B** | 23.4/24.2 | **中坚档**（VideoMME 76.7、时序 grounding 强、BF16 单卡） |
| Qwen3.5-4B | 27.5/28.4 | 反而慢于 2B 出局 |
| Qwen3.6-35B-A3B | 67.1/69.2 | 暂缓（单请求无优势） |
| Qwen3.6-27B-AWQ-TP2 | 71.1/74.8 | 画质档（细节最深需 parser） |
- MiniCPM-V4.6-AWQ：vLLM0.28 量形不兼容（挂账 transformers 裸跑）；MLM 换装队列脚本 bench_queue.sh 已固化

### 批次B｜WeMM-2B 跨模态检索 POC ✓
60 帧×20 中文查询：**hit@1 50% / hit@3 75%**（"老人落水救援"精确命中）→ 以文搜帧成立，入 M10 检索层（CPU 嵌入 51s/60帧；mage-env+SentenceTransformer，需 torchvision+qwen_vl_utils）

### 批次C｜M2 镜头分割器落地 ✓（/data/tools/shotseg/）
- 三信号：SigLIP2-base 帧间余弦(z=2.5 峰值) + RMS/VAD + 底部字幕条 OCR Jaccard
- 20min 临沂新闻 → **347 镜头（均值 3.5s）**；**条目金标边界召回 10/10=100%@±2s**（storysplit_validate.py：GT clip 首尾帧 SigLIP 最近邻定位时间轴）
- CPU 全链 ~15min（SigLIP 2400 帧 12min + OCR stride8 12min）

### 批次G｜M7 条目拆分 + 说话人角色 ✓
- 主播声纹库：VAD→CAM++→聚类 → 3 簇入档（voice_library/anchor_voices.json）
- **M7v2 关键洞察：主播也念旁白，纯声纹不可分——"演播室画面(SigLIP 质心)×主播声"双确认才是导语**
- 结果：**11 条 vs 金标 10 条**；边界 6/10 精确命中±10s、1 大条漏拆（挂账：条目级 LLM 主题校验）；纯演播室<45s 并入下一条规则有效
- **批次H 已补：自评环 v4 后 F1 0.67→0.88**（见下）

### 批次H｜video-use 借鉴落地（2026-09-07，QA 合成图/packed 层/自评环/词级对齐/EDL 拆条导出）✓✓
对 browser-use/video-use（MIT，"LLM 读视频而非看视频"）的评估结论：**互补不竞争**——它做素材→成片（选择性、交互式），我们做素材→结构化编目（穷尽、离线）。搬来四个工程思想，全部落地：
1. **timeline_qa.py**（决策点合成图，filmstrip 8帧+RMS 波形+镜头边界+ASR 标签）→ `shotseg/news_20260904/qa/` 21 张（10 边界+11 条目卡片）；人工抽查从"拖视频"变"看图"
2. **packed_md.py**（catalog 75k token → **26.8KB** agent 可读层，~7×压缩）：条目头（时间/标题/画面构成/标识/人脸）+ 词级真值时间戳转录行；说话人标签由镜头类型推导（口播=主播/采访=同期）
3. **story_selfcheck.py v4 自评环**（补 G 批挂账）：**边界 F1 0.67→0.88**（P/R 均 0.88，金标 7/8）——1172s 守艺人正确拆出、71/174 提要误拆与 870 彩排一分为二正确合并；剩 239s 亲清会客厅疑难例（提要同名+无主播声跨度起点，留痕 selfcheck.json）
   - **三次失败教训**：0.8B 自由找切点=发明话题；merge 上下文共享 15s 分段=看到相同文本胡并；二元窗口判别在"时政对时政"不可靠
   - **正解结构**：主播声纹跨度（确定性，CAM++×3质心，但画外音贯穿全程→53个起点仅是候选）∪ v2 边界 → **LLM 只做整条目级同事件判断**（标题+摘要+开头转录，2B 够用；0.8B 判别不可靠已实证）
4. **align_words.py**（Qwen3-ForcedAligner-0.6B，新建 align-env/qwen-asr）：整期 20min **4964 字毫秒级**（280s/块×5，GPU ~1.5G）；packed.md 时间戳随之升级词级真值
5. **edl_export.py + render_story.py**（EDL 拆条导出，固化 video-use 硬规则：词边界吸附±60ms pad/切点 30ms 音频淡入淡出/字幕最后挂/分段单编码+无损 concat/--preview 720p）→ `edl/story_001.mp4`(大湾区 108.4s)+`story_008.mp4`(守艺人 22.9s，v3 新拆条目的直接产物)；**验收全绿**：ffprobe 时长=EDL 预期、切点 RMS 无爆音、首尾帧目检=主播导语/现场画面
- 服务：text-qwen35-08b.sh(8012)/vl-qwen35-2b.sh(8013, GPU3) 用后已停，GPU2/3 已释放

### 批次I｜帧域确定性闭环（2026-09-07，五轮评审 60+ 条约束 → P0-P4 施工）✓✓✓
**0904 边界 F1 0.88 → 0.94（P 0.89 / R 1.00，金标 v1/v2 双口径一致）；239s 亲清会客厅跨批次疑难例关闭**（OCR 题花事件@240ms 与金标 239.0 精准对应）。唯一 FP=760.9s 彩排起点（金标自身不可判区间，gt_bounds_v2 已注记）。误差方向性：early 1 / late 7，P50=340ms。
- **P0 framedomain.py**（全项目唯一换算库）：`[start,end)` 半开区间；fps 有理数；仅 floor/ceil 两入口（整数时间基，`frame_to_ms` 用 ceil——floor 在分数 fps 丢帧有反例实证）；VFR/时长不一致显式 fail；Canonical Boundary ID（story/shot 同物理切点同 id）；stories_final schema gate 25 项单测全绿（含 30000/1001 与真实源 29975 帧@25/1，容器 nb_frames 与解码差 1 帧以解码为 canonical）
- **P1 检测层 frame_refine.py**：单次全片解码（197s CPU）双通道帧差（直方图相关+像素 MAD）+局部 MAD 归一化+10ms RMS；候选=帧差峰(去抖聚类)+渐变三模式(亮度坡/像素平台/siglip 抬升复用)+静音谷+OCR 三级；**镜头轨 346 段 318 边界吸附到确切帧**（原 0.5s 栅格）；**双跑 SHA256 一致**（确定性 gate）
- **P2 语义层+裁决**：v4 架构（v2∪声纹跨度∪题花事件候选 → LLM 顺序合并）+ 四个实证补丁（题花仅 highconf+首次出现+非提要区；B 侧题花提示且弃 catalog what；k=3 投票；置信加成仅 highconf）。**边界帧落位**=±1.5s 窗内最优切镜峰（z+静音谷加成）→ stories_final.json 双置信度（sem/frame 分离）
  - **⚠ 五版失败留痕（数据驱动纪律的代价与证据）**：纯双向双问 62/97 AMBIGUOUS；证据分级免 LLM（Anchor 级）0.20/0.21——人名字条分类泄漏让"证据主权"放大噪声；span 级 desc 巨 span 污染；tail/head 窗 A 侧最近题花毒性；temp=0 下 k=3 三票全同（seed 无效）。**教训：未经验证的新规则不得取代已验证架构；LLM 是一名证人而非主权者**
  - **2B 判定跨 run 翻转实证**（404/568.5 清晰案例）→ 触发计划内 27B 终审通道（本机 qwen3.8-27b@127.0.0.1:8000，思考型 max_tokens 4096+reasoning 兜底）→ 一步到位 0.94
- **P2c 帧级验收**：EDL v2 双域（frame canonical+ms 派生，render 断言互推一致）；**帧身份判别式验证**——渲染输出帧↔源帧 SSIM 判别（S(e) 须严格优于 S(e±1)），story_001 切点处 0.9995 vs 0.1619；静态段 ±1 帧不可分分类为 MARGINAL_STATIC（非错帧）；ffprobe 时长/帧数精确
- **P3 news_pipeline.py**：11 阶段 DAG（指纹=config 哈希+输入(size,mtime_ns)+依赖指纹；改 config 全失效/`--force` 子树失效实证）；原子标记（RUNNING/SUCCESS/FAILED，废除"文件存在=done"）；Run Lock（活锁 fail-fast/死锁接管留痕）；GPU 守卫（仅{2,3}，CUDA_VISIBLE_DEVICES 子进程注入——堵住 asr 跑 cuda:0 生产卡的泄漏 bug）；serve PID 归属制；run_log 耗时表
- **P4 评测/可解释性**：boundary_error_report.json（层1 内部帧保真 exact/层2 vs 金标 ms+方向性 early/late）；boundary_review.py ±3s 回放图+boundary_cards.md 解释卡（候选/证据/LLM 表决/落帧方式全留痕）；catalog_reindex（final 口径重索引，3 处合并留痕）→ packed.md 25.6KB 10 条目；确定性子网双跑 SHA256 一致
- 金标外置 gt_bounds.json/v2（OCR 证据修正 403.5→404；彩排 733 缺口注记）；F1 评测移入 evaluate.py（selfcheck 不再内嵌金标）
- 复跑一条命令：`python3 /data/tools/news_pipeline.py <run_dir> --llm27b`（0905 零改码回归见 README）

- **0905 零改码回归 ✓**（唯一一次人工干预=GPU2 被 27B LB 副本动态占用时按守卫指引 `NEWS_PIPELINE_GPU=3` 续跑）：全链 4820s（shotseg 2649 + vision 1420 + selfcheck 384 + semantic 90 + asr 36 + 词对齐/落帧/EDL/校验 <2min）；18 条目/333 镜头，无碎条（最短 23.5s），不变量全绿，抽 2 条目渲染帧身份判别 PASS（切点 0.999 vs 邻帧 0.09-0.23）
  - **回归期四个工程修复**：①GPU 守卫实战拦截（GPU2 空闲 7.8G<10G fail-fast，27B LB 副本漂移是真实威胁）②27B 服务端 `peg-native format` 500=输出被截断/畸形时服务端拒绝——重试翻倍 max_tokens+升温扰动解决 ③管线补 vision/semantic 两阶段（0905 无 catalog 实测合并 pass 20/20 全留崩盘——**语义层标题是合并判定的架构依赖，非可选项**）④align_words device_map 幂等/B5 补丁落位
  - 0905 无金标：2B 语义标题粗糙（"最务会第一十二"乱码/人名字条当标题）与省运会簇/办实事二连拆等疑点全部进入 review/ 17 张回放图待人工裁定——按规则冻结纪律，无错误分布不调参
  - 耗时结构结论：shotseg(OCR 段 CPU 28min) 与 vision(2B 1420s) 是两大瓶颈 → 批次J 门控加速的实证依据

### 批次F｜端到端 catalog（M5/M6 预演）✓✓
临沂新闻 20260904 整期 20min → `e2e/catalog.jsonl`：**11 条目/347 镜头/语义字段完整 11/11**
- ASR=FireRedASR2 15s 分片；人脸=AuraFace 每镜头中帧（127 帧命中 42 人库）；画面分析=Qwen3.5-2B（类型/衣着/地标/天气/标识 JSON）；条目语义=5W1H+文字稿（**"张宝亮主持""白色西装+胸针""LYTV 台标"全对**）
- 蓝图五层全通：镜头→条目→角色→语义→标识。复跑见 bench_outputs/README.md

### 批次D｜人脸终局 ✓（face_candidates_bench.py）
| 模型 | margin | 分离率 | 判定 |
|---|---|---|---|
| **AuraFace(MIT)** | **0.185** | **74%** | 🏆 卫冕 |
| buffalo_s(非商用) | 0.092 | 57% | 内部测试 |
| SFace(Apache) | **-0.056** | 40% | 崩盘（最差异类 0.949，1:N 不可用） |
| dlib(CC0) | -0.027 | 40% | 崩盘+慢(108ms) |
- 三大陷阱实证：代码MIT≠权重可商用（CompreFace/Faceplugin/DeepFace 三连）；"Open Source"名≠有LICENSE；LFW 饱和须业务实测。SeetaFace6 模型仅百度网盘→待补测

### 批次E｜治理 ✓
registry.json（41 模型含 env/serve/结论/状态）｜serve/*.sh 六件+stop.sh（ss 取 PID 防自杀）｜清理量化：模型 .cache 残留仅 3.5M、uv 缓存 46G 留作加速、hub 重复 597M 可清｜备份 /data/backup/models/mline-keyset-20260907.tar（18G+sha256）+ restore.sh（20 repo 一键重拉）



## 🔥 推理栈现代化 v2（2026-09-06 晚，vllm28-env：vLLM 0.28.0 + transformers 5.16.1 + torch 2.13.0+cu130）

### 新底座部署要点（替代 qwenasr-env 0.14 老栈）
- `uv venv + uv pip install "vllm[audio]==0.28.0"`；Qwen3-ASR 原生 `vllm serve` 即可（qwen-asr wrapper 废弃，仅 ForcedAligner 保留旧环境）
- 启动必备：`CUDA_HOME=<venv>/lib/python3.12/site-packages/nvidia/cu13`（vLLM 0.28 wheel 自带完整 CUDA13 工具链含 nvcc）+ `PATH=$CUDA_HOME/bin:<venv>/bin:$PATH`（flashinfer JIT 需要 nvcc+ninja）+ `VLLM_USE_FLASHINFER_SAMPLER=0`（否则 flashinfer 采样核 JIT 因 CCCL 头版本冲突崩：flashinfer 0.6.16.post3 自带 cccl 与 cu13 nvcc 不兼容）+ CPATH python3.12 头（老坑）
- **>40s 分片 bug 在 0.28 已修复**：45s 单片 0.61-0.74s 干净返回、零 <asr_text> 循环（老栈 ~60s 多段循环）→ 生产分片上限可放宽（建议仍 ≤60s + 限输出长度）
- **新坑①**：茶余饭后 540s 处音乐/串场段触发无上限生成（300s 超时；transcriptions 端点无 max_tokens 帽）→ 生产需 VAD 门限或 chat 接口限长
- **新坑②**：GLM-4.6V-Flash 视频默认预算 1 亿像素（video_preprocessor_config.json longest_edge=100352000）→ 45s 片 61824 token 超编码缓存全 400。解法：改模型目录该文件 longest_edge=31457280（32M 像素 ≈ 4 万 token 上限）+ --max-model-len 98304 --enforce-eager --max-num-batched-tokens 8192（0.85util）后 KV 余 10.3G 正常
- 复测 Qwen3-ASR：顺序 19/20 片 0.45-0.67s/片 → RTF≈0.020 与老栈持平；短音频 M1 200 条 wall 13.9s → RTF 0.0125

### M1 方言 ASR 决赛（KeSpeech 冀鲁100+胶辽100=200条，2841字，GT现成）
| 模型 | CER | RTF | 结论 |
|---|---|---|---|
| **FireRedASR2-AED**（Apache-2.0，走 FireRedASR2S 官方代码仓库） | **5.07%** | **0.0090** | 🏆 方言主力换代：精度速度双第一 |
| Qwen3-ASR-1.7B（Apache-2.0，vLLM 0.28 原生） | 8.62% | 0.0125 | 亚军；服务化/流式/多语种优势仍在 |
| GLM-ASR-Nano-2512（MIT，vLLM 0.28 原生） | 18.16% | 0.0151 | ✗ 方言出局（有幻觉前缀"我打点干啥呀？"） |
- FireRedASR2 部署：代码在 github.com/FireRedTeam/FireRedASR2S（不是 FireRedASR 老仓！老仓代码无 CTC 头会 state_dict 报错）；权重 FireRedTeam/FireRedASR2-AED 4.5G；FireRedAsr2Config(use_gpu=1,use_half=1) + batch transcribe
- 茶余饭后双模型预标：两模型相似度仅 0.52-0.70（方言闲聊分歧实在）→ 无自动共识，全部分级进人工校对清单 /data/datasets/bench_outputs/chayu_prelabel/REVIEW.md（小分歧3/大分歧16/音乐1，约10-15分钟人工）
- SeACo 热词注入细测：6 片对照基本无差异（"临沂交通旅游广播"等专名基线已对）→ 热词表留作生产配置项，非关键路径

### OCR 换代（同帧集对照：临沂新闻30帧，CPU，mkldnn=False）
| 模型 | 速度 | 文本行 | 备注 |
|---|---|---|---|
| PP-OCRv5-mobile（基线） | 5508ms/帧 | 197 | 同帧复测（旧记录2965ms/279行系另一帧集，作废） |
| **PP-OCRv6-medium**（3.7 包自带，Apache） | **5487ms/帧** | 197 | 与 v5-mobile 同速同行数，识别精度官方+5%；v5-server CPU 24.8s 的坑由 v6-medium 填平 → △ 关闭 |

### 人脸商用切换对照（42人库 240 张预对齐裁剪，CPU）
| 模型 | 许可 | 判别分离度 | 正确分离率 | embed速度 |
|---|---|---|---|---|
| buffalo_s(w600k_mbf) | ❌非商用 | 0.076 | 55% | 8ms/张 |
| **AuraFace-v1(glintr100)** | **MIT** | **0.189** | **71%** | 28ms/张 |
- 结论：商用切换零代价净赚（判别力反而强 2.5 倍）；buffalo 全系降级为内部测试。脚本 /data/tools/auraface_vs_buffalo.py

### 真视频横评（23条：A风险10×45s / B新闻10×45s / C长段3×60s，统一720p H.264，26查询/模型）【决赛完毕】
| 模型 | 部署 | 成功 | 速度avg/p50 | 开箱质量初判 |
|---|---|---|---|---|
| MiniCPM-V4.5-AWQ（Apache，6.7G） | 单卡0.85，需 --trust-remote-code | 26/26 | **5.4s**/6.7s | 字幕细节最强（编辑staff字幕都抠出）；think 泄漏需剥 |
| Qwen3-VL-4B（Apache，上代基线） | 单卡0.85 | 26/26 | 7.2s/7.3s | 结构最干净、60s片内时间尺度正确；偶有日期自我纠正幻觉 |
| GLM-4.6V-Flash-AWQ（MIT，8.3G） | 单卡0.85+32M像素预算改档 | 26/26 | 20.2s/21.2s | think 泄漏+"同上×N"退化；**时序时间尺度崩**（60s片标11:00） |
| **Qwen3.6-27B-AWQ-INT4**（Apache，画质档） | **双卡TP2** 0.92 | 26/26 | 71.1s/74.8s | 内容质量最强：a07 抠出"120化身119"标题+副题+"7月31日 临沂市沂河新区"角标日期地名；英文 think 泄漏需 reasoning parser 修 |
| Qwen3.6-35B-A3B-AWQ（Apache，吞吐档候选） | 双卡TP2 0.92 | 26/26 | 67.1s/69.2s | 质量同 27B 级；**单请求不比27B稠密快**——MoE 优势须并发批量才兑现 |
| Qwen3.8-27B-FP8（本机生产同源，疑案实证） | 双卡TP2 0.92 | 26/26 | 90.1s/92.3s | **视频实证通过**：抽取细节与 3.6 同级（记者署名"临沂台 马彪"、人名条全对）；官方零视频分→用户疑问的实证答案=能用，但画质档仍推 3.6（87.7 官方背书+AWQ 更快） |

**三档定型决议（M8 收口）**：
- 画质档 = **Qwen3.6-27B-AWQ 双卡TP2**（细节最深，时序/字幕/角标全覆盖；生产需配 reasoning parser 剥思考）
- 吞吐档 = **MiniCPM-V4.5-AWQ 单卡**（5.4s/条=27B 的 13 倍速，字幕 OCR 向最强；think 剥离后可作批量字幕线主力）
- 轻量/备选 = GLM-4.6V-Flash（MIT 备胎）；Qwen3-VL-4B 维持现役兼容基线
- 35B-A3B 暂缓：单请求无优势，若批处理并发实测兑现 MoE 优势再入列（挂账 M6）
- 共性工程要求：① think 系模型须 reasoning-parser/关思考；② 视频像素预算逐模型配置；③ 27B 级视频=双卡 TP2 起步

### 真视频横评工程坑（滚动）
- vLLM 杀服务后 8012 端口常被 API 子进程残留占用 → `ss -tlnp | grep 8012` 取 pid 再 kill（pkill -f 会匹配自家命令行自杀，已踩 3 次）
- 27B-AWQ 单卡视频不可行实证：0.95util 下 KV 仅剩 0.85G，45s 片需 3.3G → ValueError 拒启，TP2 后 KV 10.3G 从容
- Qwen3.6/3.8 双 27B 均有英文 meta-think 泄漏（"The user wants..."）→ 生产必配 `--reasoning-parser` + enable_thinking:false

### 声纹（茶余饭后 20片×5s窗=120段，CPU，2-means 粗聚类）
| 模型 | 内外差 | 速度 | 备注 |
|---|---|---|---|
| CAM++（Apache） | **0.385** | **56ms/段** | 建库主力维持不变 |
| ERes2NetV2（Apache） | 0.334 | 168ms/段 | funasr 1.4.14 已带注册修复（models/eres2net/model.py）——下午的"未注册"结论过时；精度档可用 |
- 注：两模型 2-means 簇均退化 [3,117]（5s 固定窗含叠话/音乐）→ 生产建库须用 VAD 段+凝聚聚类，非固定窗
- funasr 1.4.14 wheel 还内置 fun_asr_nano vLLM 管线与 Qwen3ASR/GLM-ASR 注册（models/fun_asr_nano/），Fun-ASR-Nano 复活待测



## Qwen3-ASR-1.7B（GPU2, vLLM 0.14.0, qwen-asr 0.0.6）
- 部署: qwen-asr-serve /data/models/Qwen3-ASR-1.7B --gpu-memory-utilization 0.85（可调低混部）
- 显存: 权重3.87G；总占用21.5G@0.85预分配（含KV）
- 单片30s方言音频: 0.57s（冷缓存）→ RTF 0.019
- 顺序20×30s: 12.2s → RTF 0.020（50×实时）
- 8并发20×30s: 2.21s → 聚合 271× 实时
- ⚠️ 分片>~40s 触发模型多段循环生成（响应含多个<asr_text>，单片耗时暴涨到~60s）→ 生产分片≤30s
- 质量样例: 方言片头曲转写正确（"临沂话拉呱忙，不说方言急得慌"）；广告段（明德实验学校）也转出
- 结论: ✓ 单卡4090轻松跑，1h音频约72s顺序/13s并发处理完

## Qwen3-ForcedAligner-0.6B（/data/models/Qwen3-ForcedAligner-0.6B, qwen_asr 包）
- 字级时间戳；30s方言音频对齐 0.99s → RTF 0.033（30×实时）；显存≈0（轻量，CPU/混部友好）
- API: Qwen3ForcedAligner.from_pretrained(path) → .align(wav, text, 'zh') → items[{text,start_time,end_time}]
- 结论: ✓ M2/M7 分条对轴/字幕生成直接可用

## 批1 ASR 对照（GPU3, 5×30s 方言片）
| 模型 | RTF | 显存 | 备注 |
|---|---|---|---|
| SenseVoiceSmall | **0.003** | 1.1G | 带情感/BGM标签(<|HAPPY|><|BGM|>)；funasr 1.4.14 直接用 |
| Qwen3-ASR-1.7B | 0.020(顺序)/0.0037(8并发聚合) | 3.9G权重 | vLLM服务化，分片须≤30s |
| SeACo-Paraformer | 0.013 | 2.0G | 含VAD；热词注入待M1细测 |
| FireRedASR-AED-L | 0.070 | 6.8G | beam=3质量档；batch接口 |
| Qwen3-ForcedAligner-0.6B | 0.033 | ≈0 | 字级时间戳，M2/M7直接可用 |

FireRedASR 部署坑（funasr-env 复现要点）：①PYTHONPATH=/data/models/FireRedASR-code（ModelScope快照无model.py）；②pip install kaldi-native-fbank；③torch≥2.6 需 torch.serialization.add_safe_globals([argparse.Namespace])；④triton 首编译需 CPATH=/data/tools/ffmpeg-deb/root/usr/include/python3.12:/data/tools/ffmpeg-deb/root/usr/include（python3.12-dev 头文件已解包在 ffmpeg-deb/root）

## 批2 VL（GPU3）
### Qwen3-VL-4B-Instruct（vLLM 0.14, /data/models/Qwen3-VL-4B-Instruct, 8.3G bf16）
- 显存: 21.1G@0.85预alloc（权重~9G，混部可调低）；启动需 CPATH=python3.12头（同triton坑）
- 24帧统一prompt(场景+有无文字+OCR): 顺序 1.08s/帧(P50 0.99/P95 1.81)，4并发 4.49帧/s，69 tok/帧
- 质量: 新闻片头图形台标"LIN YI NEWS"识别准确；输出结构规整可解析
- 换算: 1h视频@1帧/2s=1800帧 → 顺序32min / 4并发6.7min
- 结论: ✓ 4B档单卡4090编目主力；8B档现役zx-vl8b基准已有

## 批2 VL 续
### Microsoft Mage-VL 4.7B（/data/models/Mage-VL, transformers 5.x + trust_remote_code）
- 加载: 2s（bf16 权重9G）；显存 图片模式 10.2GB / **视频模式(16帧) 20.7GB**——单卡4090放得下，帧数↑需控制显存
- 图片推理 3.52s/帧（transformers 裸generate，无vLLM加速）；视频(10s/16帧) 7.36s
- 质量: 新闻片头3D地图帧描述准确并正确判"无文字"；视频描述可用（精度横评归M11）
- 待M11: codec后端(H.264/DCVC，需PATH有ffmpeg——/data/tools/ffmpeg-deb/ffmpeg已就绪)、streammind_gate流式门控、vLLM/SGLang加速
- 结论: ✓ 4090 可跑（图片档从容；视频档紧）
- 注: mage-env 启动同样需要 CPATH=python3.12头文件（见批1备注④）

### VL 其余候选（InternVL3.5-8B/MiMo-VL-7B/Phi-4-MM）
- 未下载实测（各15GB+，GPU2/3已被本测试占用）；显存按参数量预估 ~17/16/12GB bf16 均可单卡
- **Kimi-VL-A3B（16B MoE）bf16 ~33GB ✗ 单卡超限**——如需入选必须AWQ（~17GB，M8时验证），否则以 MiMo-VL-7B 替补
- Qwen3-VL-32B 仅AWQ/INT4档可试（~20GB，M8）

## 批3 轻件线
| 模型 | 实测 | 结论 |
|---|---|---|
| buffalo_s 人脸包 | CPU 366ms/帧（检出68/78脸）vs buffalo_l 830ms/帧 | ✓ 速度2.3×、检出87%，轻量档可用； buffalo_l 精度档 |
| CAM++ 声纹 | RTF 0.003，192维，方言音频嵌入正常（跨段相似度0.38~0.94随说话人变化） | ✓ 建库主力 |
| ERes2NetV2 声纹 | funasr 1.4.14 wheel 未注册该模型（speaker模块注册缺失） | △ 需 3D-Speaker 仓库运行，精度档后补 |
| PP-OCRv5 | 见下（跑完后补录） | |

## 环境/坑速查
- 所有 torch venv（qwenasr/funasr/mage-env）首跑 triton 编译需：`CPATH=/data/tools/ffmpeg-deb/root/usr/include/python3.12:/data/tools/ffmpeg-deb/root/usr/include`（python3.12-dev 头文件解包在 ffmpeg-deb/root）
- ffmpeg/ffprobe: /data/tools/ffmpeg-deb/ffmpeg(6.1.1 deb动态版，全部依赖已解包root/)，另有软链 /data/tools/ffmpeg
- Qwen3-ASR 服务: GPU2:8010（启动命令见MODEL-PICKS）；Qwen3-VL-4B: GPU3:8011
- PaddleOCR CPU 需 enable_mkldnn=False（oneDNN PIR bug: ConvertPirAttribute2RuntimeAttribute）
| PP-OCRv5-mobile | CPU 2965ms/帧，文本行279/30帧（样例"张宝亮"人名条识别正确）| ✓ CPU 可用；GPU paddle 更佳 |
| PP-OCRv5-server | CPU 24829ms/帧（无mkldnn），文本行330 | △ CPU不可用，须 paddlepaddle-gpu 或修复 oneDNN（M8 补测） |
注: PaddleOCR 3.7 CPU 需 enable_mkldnn=False 规避 oneDNN PIR bug
