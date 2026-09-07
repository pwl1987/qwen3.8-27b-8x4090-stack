# coding-v1.1 vLLM 迁移 + 核心专家架构总方案（v2.1，2026-09-06 晚修订）

> 硬件：lytv-H3C-R5300-G6，8×RTX 4090（24GB），503GB RAM，/data 4.7TB NVMe（现余 5.9T）
> 红线：不修改 Gate 阈值，不解释掉 FAIL，用受控实验证明归因
> v2.1 变更摘要：吸收 2026-09-06 晚"推理栈现代化+五域换代实测"全部结论（详见 /data/datasets/bench_outputs/RESULTS.md）；GPU 布局勘误；M10 补定义；商用许可列为硬约束。

---

## 〇、一页总览

**终局**：以 coding-v1.1 为核心专家的三线系统——
① **编程线**：ZCode + vLLM 主力 + MCP 工具群（检索/视觉/渲染/校验）
② **媒资线**：视频 → 镜头级编目（谁/场景/地点/事件/台词/时间轴）→ 混合检索
③ **直播线**：实时流 → 确定性切点 + 流式 VLM 标注 → 切片直接入编目库
④ 终点：**自动新闻生产**（检索素材 → coding-v1.1 写脚本 → ffmpeg 装配 → ComfyUI 补空镜）

**六条全局原则**（所有选型/实验通用）：
1. **中文优先**：95%+ 场景中文，中文质量一票否决
2. **四维标准**：速度/准确率/吞吐/资源，直播线速度加权、批处理线准确加权（5.4 节）
3. **时间轴强制**：每条轨道输出 (t_start, t_end, payload)，毫秒内部统一（5.4 节）
4. **绑定光谱**：运行时松（MCP 总线）→ 按需加密（网关）→ 训练时紧（微调内化）
5. **外围化**：确定性的活交工具/小模型，核心模型只做决策+推理；逃逸通道必须保留
6. **商用许可硬过滤**（v2.1 新增）：入选模型必须 Apache-2.0/MIT（用户 2026-09-06 定）。已切换：人脸 buffalo→AuraFace-v1(MIT)；SenseVoice-Small 自定义协议待法务（降为参照）；淘汰 TeleSpeech/MiMo-VL/LongVA

**当前状态**（2026-09-06 21:45 勘误）：GPU0/1 llama.cpp 生产（不动）｜GPU2/3 = M 线横评池（本晚真视频横评用）｜GPU4 = S1 量化 r4 进行中（另一 session，关键路径，勿动）｜GPU5 = zx-embed+rerank 12G｜GPU6 = zx-ab-vllm 27B W4A16 23.9G｜GPU7 = zx-vl8b（Qwen3-VL-8B）22.2G 现役。

**待用户提供**：人脸库 42 人真名指名（几分钟，不阻塞 M3）；茶余饭后金标人工校对（REVIEW.md，10-15 分钟）。~~新闻素材/人脸库/风险样本~~ ✅ 已自采齐（见 MANIFEST）。

---

## 一、基线与已定决策

| 项目 | 值 |
|---|---|
| 模型 | Qwen3.8-27B coding-v1.1（BF16 merged，12 shards，52GB，`/data/compose/qwen27b/train/merged-bf16/`） |
| 架构 | Qwen3_5ForConditionalGeneration：64 层 = 48 线性注意力（GDN，无 KV）+ 16 全注意力（仅此 16 层有 KV）+ 原生 MTP 头 1 层 + 视觉塔 878MB（27 层 ViT，图片上限 4096×4096=16384 token，视频 25M 像素≈12288 token） |
| 现产 | llama.cpp GPU0/1 双副本，iq4_xs GGUF，~90 tok/s（thinking ON） |
| A/B 实测 | AnnoyingTechnology vLLM 栈 175.4 tok/s = 1.94×；主因 base 验证步 28ms vs 43.5ms（1.56×），非投机解码（1.09×）；混淆变量=权重不同 |
| **已定** | B 线：量化 coding-v1.1 上 vLLM（保留后训权重），S1 进行中（r4） |
| ⚠️ v2.1 注记 | Qwen3.8-27B 官方**零视频基准**（编程定位），视频能力未验证——M 线视频主力**不选它**，选 Qwen3.6-27B（VideoMME 87.7 千问系王者，见 5.2 M8）。若要 3.8 上视频须先过自有视频评测 |

## 二、总体架构：一核三线

```
                 ┌────────────────────────────────────┐
                 │   coding-v1.1 27B（核心专家，GPU6） │
                 │   决策 + 推理，其余全部外围化        │
                 └────────────┬───────────────────────┘
        ┌────────────┬────────┴─────────┬──────────────┐
   【编程线】    【媒资批处理线】      【直播线】      【校验/专家】
   ZCode(MCP)   镜头编目→向量库      双路径切+标     Tier0 工具环
   repo_search  检索引擎(共用GPU4)   快路径确定性    Tier1 角色专家
   read_symbol  ASR/人脸/音乐       慢路径VLM标注   渲染对比闭环
   parse_shot   统一小模型caption    切片→编目库     (同一vLLM实例)
   render_cmp   (GPU5 专家池)
```

三线在两处汇合：**编目层**（直播切片与批处理入库同构）与**检索层**（素材库与代码库共用 embedding+reranker 基建，GPU4）。

## 三、推理层（T 系列）

### 3.1 技术清单（v2.1：生产沙箱锁 0.27.1-cu129；M 线新底座 vllm28-env = vLLM 0.28.0 + tf 5.16.1 + torch 2.13.0+cu130，已实测）

**0.28 新底座部署要点**（/data/tools/vllm28-env，已踩坑验证）：
`CUDA_HOME=<venv>/site-packages/nvidia/cu13`（wheel 自带 CUDA13 工具链）+ PATH 加 cu13/bin 与 venv/bin（flashinfer JIT 需 nvcc+ninja）+ `VLLM_USE_FLASHINFER_SAMPLER=0`（规避 flashinfer CCCL 头冲突崩溃）+ CPATH python3.12 头。Qwen3-ASR 0.16+ 原生 serving（qwen-asr wrapper 废弃）；**>40s 分片 bug 已修复**（45s 片 0.7s 干净返回）；新坑：音乐段无上限生成（需 VAD 门限/限长）。

**注意力后端**：FLASH_ATTN / FLASHINFER / TRITON / FLEX / KVARN（配 KVarN）/ TURBOQUANT / TORCH_SDPA(仅ViT)

**KV 量化**：kvarn_k4v2_g128（当前）/ k4v4 / turboquant_k8v4 / k4v4_nc / k3v4_nc / fp8 / bf16；`kv_cache_dtype_skip_layers` 支持分层 KV

**投机解码**：DFlash2（当前 k=7）/ DSpark（0.28 改进）/ **原生 MTP（模型自带，栈内 mtp-long profile）** / ngram（编辑场景甜区）/ eagle / medusa

**权重量化**：**awq_marlin（Ada 最快，必选）** / awq / gptq / compressed-tensors；NVFP4 不可用（Blackwell 专属）

**编译**：optimization_level O2 + performance_mode interactivity（交互）/ throughput（批量）；cudagraph_mm_encoder（ViT CUDA 图）；VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE（稳态 +3-8%，启动慢）

**调度/内存**：async_scheduling 默认开；max_num_batched_tokens 16384（0.28 默认，**大视觉模型建议显式降 8192**，见 RESULTS GLM 案例）；kv_offloading_size（CPU L2）+ offload_backend；VLLM_VISION_CPU_OFFLOAD_GB=1（+36ms/图省 878MB）；VLLM_ENABLE_STARTUP_PLAN=1

**多模态**：mm_encoder_attn_dtype=fp8；mm_processor_cache_gb；video_pruning_rate+evs（视频 token 剪枝）；**视频总像素预算=显存第一杀手**（GLM 默认 1 亿像素须改 32M；27B-AWQ 单卡 KV 不够 45s 视频 → 27B 档 = 双卡 TP2，均已实测）

### 3.2 三级缓存（HiCache）

L1 GPU KV（KVarN）→ L2 CPU KV（`--kv-offloading-size 150 --offload-backend native`，503GB RAM 充裕）→ L3 磁盘（0.28 新增）。多轮长对话不重算前文；配置进 Phase 3。

### 3.3 T 系列实验（注：原编号 T2 组并入 T1 视觉/基线验证，v2.1 清理悬空编号）

| 编号 | 变量 | 对照 | 指标 |
|---|---|---|---|
| T1.1 | 量化引擎 | llama.cpp iq4_xs | coding 质量（10 任务人工+自动） |
| T1.2 | 视觉（GPU 驻留 vs CPU offload） | 互为对照 | vision canary + 显存 |
| T1.3 | 基线性能 | llama.cpp 90 tok/s | decode/prefill tok/s、TTFT |
| T3.1 | KV 量化 | KVarN K4V2 | TurboQuant k8v4/k4v4/k3v4 |
| T3.2 | 分层 KV（前2+后2 FP8，中间 KVarN） | 全层 KVarN | 质量+显存 |
| T3.3 | DFlash2 k=5/6/7/8/10 | k=7 | tok/s+接受率 |
| T3.4 | DFlash2 vs **原生 MTP** vs DSpark(0.28) | DFlash2 | tok/s+TTFT |
| T3.5 | performance_mode 三档 | balanced | 延迟/吞吐 |
| T3.6 | ViT CUDA 图 | 默认 | 图片延迟 |
| T3.7 | INDUCTOR_MAX_AUTOTUNE | 默认 | 稳态 tok/s |
| T3.8 | CPU KV offload 100GB | 无 | 长对话延迟 |
| T3.9 | ViT FP8 attention | FP16 | 编码延迟+显存 |
| T3.10 | Marlin vs Triton AWQ | Marlin | tok/s |

（分层 KV 显存账：64 层仅 16 层有 KV——全 KVarN ~262K 单卡可容；加 4 层 FP8 增 ~1.6GB，T3.2 实测定取舍。）

## 四、编程外围层（E1-E10）

### 4.1 视觉级联（截图复刻/扒站）
截图 → OmniParser V2（元素）+ PaddleOCR ch（文字）+ PIL（hex 调色板）→ JSON + 原图 → coding-v1.1 生成 → Playwright 渲染 → 双图对比评审 → 修订 2-3 轮。副产品=复刻三元组（Phase 2 教材）。

### 4.2 专家流水线（Tier 0 > Tier 1，铁律不可颠倒）
- **Tier 0 确定性**：编译/类型检查报错回喂环（最强单模式）、linter、测试运行、Playwright
- **Tier 1 角色**（同一 vLLM 实例换 system prompt）：Reviewer 挑刺 / Quality / Test（写测试交 Tier0 判红绿）/ Architect（长任务拆解+跨文件接口一致性）/ Judge（best-of-N）
- 局限：自评偏差，self-critique 修复 20-40%；专家轨迹落盘 → Phase 2 微调（专家当老师）

### 4.3 上下文工程（长期维护生命线）
Repo Map（tree-sitter 签名视图）+ 代码 RAG（Qwen3-Embedding-0.6B @GPU4）+ **Reranker 门卫**（Qwen3-Reranker-0.6B，分层过滤：规则 100% 准 → BM25 → embedding → reranker 精筛；偏向召回，永不丢弃清单）+ 依赖签名注入。压缩三档 20/50/80%，甜点 E8 扫描。

### 4.4 输出经济学
search-replace 编辑块（大文件输出 -60~90%）+ xgrammar 结构化输出（编辑语法 100% 合法 + JSON/YAML/SQL 约束）；编辑场景 ngram 投机解码 A/B（E6）。

### 4.5 外围化清单（模型只做决策+推理）
已列：视觉预处理/上下文过滤/仓库检索/**工具输出消化（日志→错误行 ~50:1）**/**符号级读取（整文件→单函数 ~100:1）**/**任务状态外置**。命门：判断力不外包；过滤器偏向召回；逃逸通道必须保留。

### 4.6 输出格式（IDE 集成）
`--reasoning-parser qwen3`（思考走 reasoning_content，content 干净）+ 按需 `enable_thinking:false` + 兜底剥离 + `--tool-call-parser qwen3coder`（工具调用结构化）。

### 4.7 MCP 集成与调度
三层打通：vLLM 双 flag；ZCode 注册 MCP servers（repo-tools/vision-tools/render-tools）；模型零配置（Qwen3.5 原生工具格式）。模式 A 模型主动 / 模式 B harness 前置。"保证用我的小模型"三层（引导→网关拦截→诚实边界：精确匹配 rg 本来就该赢）；IDE 可移植（MCP 开放标准，换 IDE=改配置）。

### 4.8 E 系列实验
E1 级联+渲染闭环 vs 直出（10 张网页截图，像素 diff）｜E2 挑刺修订 vs 直出（30 任务通过率）｜E3 Bo-1 vs Bo-3+Judge｜E4 工具反馈环（首编通过率）｜E5 编辑块 vs 整文件（token+失败率）｜E6 编辑场景 DFlash2 vs ngram｜E7 裸上下文 vs RepoMap+RAG｜E8 压缩档位扫描（25/50/75/100%）｜E9 过滤器配置（BM25/+emb/+rerank）｜E10 工具输出消化

## 五、媒资批处理线（M1-M10）

### 5.1 管线（与时间轴规范 5.4 绑定；v2.1 换代后阵容）
```
视频 → 镜头分割（三信号级联）→ 每镜头并行：
  统一视觉 caption = Qwen3.6-27B 双卡TP2 画质档 / MiniCPM-V4.5 吞吐档（场景/地点/事件/OCR，中文）
  + AuraFace 人脸比对(商用✓) + 声纹 CAM++/ERes2NetV2(说话人)
  + ASR 台词 = FireRedASR2-AED 精度档 / Qwen3-ASR-1.7B 服务化档
  + Qwen3-ForcedAligner(字级) + 音乐(chromaprint/musicnn) + embedding
→ 编目记录（毫秒时间轴，M10 落库）→ 向量库(Qdrant/Milvus)+结构化字段
```

### 5.2 关键选型（v2.1 换代定案，全部商用许可已核；实测数据见 RESULTS.md）
**方言主战场 = 山东方言**。**素材全部就位**（/data/datasets/MEDIA-TEST-MANIFEST.md）：茶余饭后话临沂整档 89.5min+1178 期索引、临沂新闻整期 2 期+关键帧、整期长节目 8.6G、人脸库 42 人✓、风险样本 16 条✓、langya 5320 集索引、直播流录制 2 条、KeSpeech 冀鲁+胶辽✓、GigaSpeechBench 53G✓。
- **ASR（M1 决赛已出，KeSpeech 200 条）**：**FireRedASR2-AED 🏆 CER 5.07% / RTF 0.0090**（Apache，精度+速度双第一，20 万小时/20+ 方言，代码=FireRedASR2S 仓库）＞ Qwen3-ASR-1.7B 8.62% / 0.0125（Apache，服务化/流式/30 语种优势，保留混部）＞ GLM-ASR-Nano 18.16%（✗ 方言出局）。淘汰 TeleSpeech（停滞）。SenseVoice/SeACo 参照档（SenseVoice 协议待法务）。SeACo 热词实测无显著收益（非关键路径）
- **镜头分割（M2）**：三信号级联（SigLIP 逐帧距离 + 音频不连续 + 字幕条 OCR 变化）→ 小 VLM 精判 + 镜头类型标注。传统算法（PySceneDetect/TransNetV2）已否
- **条目分割（M7）**：ASR 转写 → LLM 主题转折分割；**金标已落盘** /data/datasets/lytv_storysplit_groundtruth.json
- **统一视觉/视频模型（M8，真视频横评进行中）**：**Qwen3.6-27B-AWQ 画质档**（VideoMME 87.7 千问系王者，Apache，双卡 TP2 实测 KV 10.3G 从容）vs **MiniCPM-V4.5-AWQ 吞吐档**（Apache，96× token 压缩，5.4s/条最快，字幕细节最强）vs GLM-4.6V-Flash-AWQ（MIT，20.2s/条，think 泄漏+时序弱，轻量备选）；Qwen3-VL-4B = 上代基线（7.2s/条，开箱结构最干净）。**Qwen3-VL 全线被 Qwen3.5/3.6 同规模超越（官方数据），编目主力换代**。注："MegaLV"查无此名（2026-09 查证）；Kimi-VL bf16 超单卡已出局
- **人脸（M3）**：**AuraFace-v1（MIT）已实测商用切换净赚**（42 人库判别分离度 0.189/71% vs buffalo_s 0.076/55%）→ 生产主力；buffalo 降级内部测试。待用户真名指名
- **声纹**：CAM++ 主力（内外差 0.385、56ms/段）+ ERes2NetV2 精度档（funasr 1.4.14 已带注册修复，实测可用）；生产建库须 VAD 段+凝聚聚类（固定 5s 窗会退化）
- **OCR**：PP-OCRv6-medium（Apache）已实测关掉 v5-server △（5487ms/帧 与 v5-mobile 持平、精度+5%）；PaddleOCR-VL-1.6B 已下载，文档解析线 M3 期落地
- 编目聚合与元数据生成：coding-v1.1（总编角色）

### 5.3 M 系列批处理实验
M3 人脸/OCR 落地（媒体画面多角度/低清/遮挡 + PaddleOCR-VL 文档线）｜M4 VQA 场景/地点/危险动作准确性｜M5 端到端元数据质量（字段完整率+人工）｜M6 吞吐（1h 视频耗时+GPU 峰值）｜M9 编目检索质量（recall@k/nDCG）｜**M10 = 编目库落地（v2.1 补定义）**：编目记录 schema 定稿 + 写入管线（批处理与直播切片同构入库）+ Qdrant/Milvus 选型 + 检索 API——M5/M9 的载体，S3 期施工

### 5.4 时间轴规范（所有轨道强制）
每轨道 (t_start, t_end, payload)：镜头帧级 / ASR 句级±50ms+字级 / 说话人±200ms / 人物出场秒级 / 音乐字幕条危险动作秒级。铁规：①内部毫秒，导出 SRT/VTT + **EDL/FCPXML（Premiere/达芬奇直接导入）**；②直播双时间戳（相对 ms+UTC 墙钟）；③字级对齐 Qwen3-ForcedAligner。

## 六、直播线（M11-M12）

**双路径铁律（切与标分离）**：
```
直播流 ─ 快路径【切，<500ms 确定性】：SigLIP 逐帧距离 + 音频 VAD/能量突变 → ffmpeg segment
      └ 慢路径【标，容忍 1-3s】：流式 VLM 段落打标 → 直接写编目库（与批处理汇合）
```
流式横评（M11）：**Mage-VL**（codec 原生免解码；4090 预实测已过：图片 10.2G/视频 20.7G、ffmpeg CLI 已修复，codec 后端+streammind_gate+实时性留 M11）vs **滑窗小 VLM**（SimpleStream 证明 4 帧滑窗即第一梯队，用 MiniCPM-V4.5/Qwen 小档）vs StreamingVLM vs StreamMind（参考）；**Dolphin-CN-Dialect**（Apache，方言流式 ASR）纳入 M11 候选。M12：真流 pilot（切点帧精度/端到端延迟/24h 稳定；直播流已录 live_streams/）。GPU 争抢预案：生产期流式独占或绝对优先。

## 七、选型与部署拓扑

### 7.1 GPU 布局（v2.1 勘误：2026-09-06 晚实测快照）
| GPU | 用途 | 占用 |
|---|---|---|
| 0/1 | llama.cpp 生产双副本（不动） | 23GB×2 |
| 2/3 | **按需横评池**（真视频横评现役：27B TP2 双卡；ComfyUI 停用按需复启） | 动态 |
| 4 | S1 量化 r4（他 session，关键路径）→ 完成后还给编程外围池（embed+rerank+OmniParser+PaddleOCR） | 7GB |
| 5 | zx-embed + zx-rerank（vLLM 0.27.1 容器） | 12GB |
| 6 | **vLLM 主力**（zx-ab-vllm 27B W4A16 23.9G，S1 完成后切量化新权重） | 23.9GB |
| 7 | zx-vl8b（Qwen3-VL-8B 现役）→ M8 换代决议后升级 Qwen3.6 档 | 22.2GB |

### 7.2 模型阵容（v2.1 换代后）
- 核心：coding-v1.1 W4A16-AutoRound（S1 产物）+ DFlash2 k=7 drafter（GPU6，生产沙箱 0.27.1 锁定）
- M 线新底座：**vllm28-env**（vLLM 0.28.0，部署要点见 3.1）
- 媒资线：Qwen3.6-27B-AWQ（画质档，TP2）/ MiniCPM-V4.5-AWQ（吞吐档）/ FireRedASR2-AED（ASR 精度档）/ Qwen3-ASR-1.7B（ASR 服务化档）/ Qwen3-ForcedAligner / AuraFace-v1 / CAM+++ERes2NetV2 / PP-OCRv6-medium / PaddleOCR-VL-1.6B（M3 期）
- 轻量杂务（日志消化/摘要）复用 :8000 现役端点不新增。OmniParser 内置 OCR 英文倾向——中文 UI 文字一律 PaddleOCR ch

### 7.3 集成
ZCode = harness（不自研）；MCP 工具 5 件：repo_search / read_symbol / parse_screenshot / render_compare / run_checks；网关（B 模式拦截）为触发式升级项，不提前建设。

## 八、落实施工表（S1 进行中）

### S1 量化 + 主力上线 ✅ 进行中（r4 23/64，GPU4，另一 session）
| # | 任务 | 产出/DoD |
|---|---|---|
| 1.1 | 量化环境检查：transformers 5.16.1 + AutoRound/GPTQModel；GDN 线性层白名单 | 环境结论 + 量化脚本 |
| 1.2 | AutoRound W4A16 量化：语言 INT4 / 视觉塔 BF16；校准 imatrix-calib-v11-final.txt；多卡加载 | 模型 + 量化报告 |
| 1.3 | vLLM 启动（GPU6 沙箱切指针，awq_marlin+KVarN+DFlash2+VISION+PREFIX_CACHE+双 parser） | /health 通过 |
| 1.4 | 冒烟：vision canary + cache canary + ulmus_validate | T1.2/T1.3 出数 |
| 1.5 | 质量 A/B：vLLM vs llama.cpp 同权重 10 编码任务 | T1.1 出数；回退<5% 放行 S2 |

### S2 GPU4 周边池 + 首批 MCP（依赖 S1.3）
部署 embed+rerank + OmniParser 容器；MCP repo_tools；跑 E7/E8/E9。

### S3 视觉级联 + 媒资横评落地 ✅ 前置已解除（素材齐；M1 已出决赛、M8 真视频横评本晚进行）
E1 级联闭环；M8 收口（35B-A3B 吞吐档补测 + 质量评审）；M3 人脸/OCR 落地（AuraFace 建库+PaddleOCR-VL 文档线）；**M10 编目库落地**；GPU2/3 按需池征用。

### S4 专家流水线 + 栈调优
T3.1-T3.10 全量；E2/E3/E4/E10；专家轨迹开始落盘；vLLM 0.28 升级评估（M 线已验证 0.28，生产沙箱待 S4 后统一步伐）。

### S5 直播线 + 生产化
M11/M12；GPU7 用途定夺（vl8b 换代后作流式独占或副本）；三线四维复测；网关按触发条件评估；STARTUP_PLAN 开启；TRAIN-NOTES/记忆收尾。

## 九、风险与红线
① INT4 量化质量损失——T1.1 严判，回退>5% 转 GPTQ/AutoRound 调参或混合精度；② GDN 自定义架构量化工具兼容性——S1.1 先验证；③ vLLM 版本双轨——生产沙箱锁 0.27.1（28 补丁需重验），M 线已用 0.28.0（vllm28-env 部署要点固化 RESULTS），S4 后统一；④ 直播与批处理 GPU 争抢——S5 定独占/优先级；⑤ 专家流水线延迟——交互场景慎用，批量适用；⑥ 中间产物一律四维标准留痕；⑦ 商用许可——SenseVoice 条款法务复核后再上生产（v2.1 新增）。

## 十、参考文件
- TRAIN-NOTES.md（训练+A/B 全记录）｜/data/sandbox/ab-vllm/repo/（A/B 栈）｜merged-bf16/｜imatrix-calib-v11-final.txt
- **选型与实测文档体系（v2.1 纳入）**：/data/datasets/MODEL-PICKS-2026-09.md（五域选型+license）｜/data/datasets/PERCEPTION-MATRIX-2026-09.md（四层级联）｜/data/datasets/bench_outputs/RESULTS.md（全部实测数字+部署坑）｜/data/datasets/MEDIA-TEST-MANIFEST.md（素材+工具链）
- 模型：Qwen3-ASR: huggingface.co/Qwen/Qwen3-ASR-1.7B｜FireRedASR2: github.com/FireRedTeam/FireRedASR2S（注意非老仓）｜Qwen3.6-27B / MiniCPM-V-4_5 / AuraFace-v1（均已本地化 /data/models/）
- gpu-server 记忆（GPU/容器状态）

---
*附：v2 定稿的历史版本与 14.6 时间轴示例见 git/备份。v2.1 修订依据=2026-09-06 晚推理栈现代化实测（vllm28-env + 五域换代），原始数字一律以 RESULTS.md 为准。*
