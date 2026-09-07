# 问题与解决全记录（2026-08-28 → 09-07）

从裸机到自动新闻生产线，全程踩坑与解法。每条：**现象 → 根因 → 解法 → 预防**。
编号 P# 为本索引键；详细上下文在 TRAIN-NOTES / QWEN27B-ANALYSIS / VLLM-OPTIMIZATION 对应章节。

## 一、初始化与 llama.cpp 上线期（08-28 ~ 08-31）

- **P1 ghcr.io 官方镜像全被墙**：llama.cpp server 镜像直连/daocloud/ghcr.m.daocloud.io 全卡死 →
  改**本地源码构建**（github codeload 直连可下）；编译必须 devel 容器 + `--gpus all`（链接要 libcuda）。
  配方：`inference/llamacpp/build/cu124-driver550/`。
- **P2 containerd snapshotter 路径**：Docker 29 镜像实际存 /data/containerd（config.toml root 改指数据盘），
  /data/docker 是空壳——磁盘满时别看错地方。
- **P3 no-new-privileges 与 nvidia BPF 冲突**：加了就容器起不来 GPU → 永不加。
- **P4 GDN 循环态显存账**：每序列固定 ~0.88GiB 不随上下文涨，256K 单卡 97% 显存的账要算上它。
- **P5 思考型模型正文为空**：思考过程耗尽 max_tokens → 调用方 max_tokens ≥1024 起步（后统一 8192+显式 enable_thinking）。

## 二、训练期（09-01 ~ 09-02，六坑全录 TRAIN-NOTES）

- **P6 swift 默认 fp32 加载 OOM** → 显式 bf16。
- **P7 Arrow schema 推断错乱** → 数据转换先物化 schema。
- **P8 step2 反向重算 OOM**：与序列长度无关 → grad-ckpt + paged 优化器组合解。
- **P9 pkill 自杀**：`pkill -f` 模式串匹配自家复合命令 → pgrep 取 PID 再 kill，**分两次执行**（此坑在 vLLM 时代再次出现，见 P44）。
- **P10 DoRA 不可转 GGUF** → 训练定纯 LoRA。
- **P11 swift checkpoint 两级目录 glob 恒空**：断点续训取不到 → 修 glob 路径。
- **P12 GDN out_proj 列重排 = LoRA→GGUF 唯一死点**：原生 `NotImplementedError` →
  给 LoraTorchTensor 加 index_select + qwen.py 走置换路径（补丁在 `training/llamacpp/`，数值验证 7/7）。
- **P13 rsLoRA 缩放差 5.66 倍**：llama.cpp 只认 α/r 不认 α/√r → `--lora-scaled adapter.gguf:5.657`。

## 三、SGLang 探索期（09-02 ~ 09-03，详见 inference/sglang/README.md）

- **P14 FP8 输出乱码**：`is_layer_skipped` **子串匹配**把 qwen3_5 混合层名误判跳过量化层 → 修复后双端验证干净。
  预防：新架构上量化前先审层名匹配逻辑。
- **P15 256K 虚标**：KV 池实测 216,938 token（非标称 262,144）→ 全文档更正。预防：上下文能力以池实测为准。
- **P16 TP2 单流未达外推**（34.8/67.8）→ 多卡收益以实测为准，不线性外推。
- **P17 thinking 泄漏进 content**（qwen3_5 跨引擎）→ `--reasoning-parser qwen3`，Anthropic/OpenAI 双端验证。
- **P18 SGLang Anthropic 协议无结构化 tool_use（致命）**：生产工具链路断 → **上线一天即回归 llama.cpp**。
  预防：引擎选型冒烟必须用真实工具调用报文，不能只 curl /health。
- **P19 vLLM 0.20.2 reasoning-parser 吞输出** → 教训并入 P17。

## 四、H3 / 驱动升级期（09-05 ~ 09-06）

- **P20 ComfyUI buildkit 层内下载卡死** → torch wheels 宿主 wget 后本地 COPY 进镜像。
- **P21 ComfyUI 容器 dynamo getuser 失败** → uid 1000 + passwd 条目显式建。
- **P22 ComfyUI 版本不匹配** → 0.34.0 源码快照 COPY（从回滚镜像提取），钉版本。
- **P23 驱动 550→580 + CUDA 13 夜窗升级**：白天满负荷窗口不动驱动；升级后 H3 以 torch cu130 重建。
- **P24 sageattention 1.0.6 实测更慢** → 弃用（保留在 Dockerfile 备查）。优化必须 A/B 实测定去留。
- **P25 convrot int8 重建版**：与 fp8 版并存按需选用（H3 三项优化实测见 TRAIN-NOTES 09-06）。

## 五、vLLM 栈移植与量化期（09-06，S1）

- **P26 GitHub 直连超时**：clone/WebFetch 全挂 → jsdelivr 拉文件 + gh-proxy 拉 raw；后来 ssh.github.com:443 成为最稳通道。
- **P27 AutoRound 0.15 本地 jsonl mllm 路径 KeyError** → 打补丁回退文本校准。
- **P28 compressed-tensors 版本二分**：0.18 要 torch≥2.10（拉坏 2.8）、0.9.4 缺 compress_module → **0.15.0**。
- **P29 精确配方多量化 qkv/z/out_proj 后 layer0 调优 OOM** → 校准 seqlen 2048→1024。
- **P30 lm_head/embed 重量化**：round-trip 0.64%/0.56% 通过；262K GDN state 池差 2MB OOM → 首版限 128K。
- **P31 容器 /app 挂载覆盖镜像 venv**：docker run -v repo:/app 后 exit 127 → 挂 /repo 用镜像内 venv。
- **P32 compose 端口/设备合并叠加**：base+override 双发布双卡 → `!override` YAML 标签（compose ≥2.40）。

## 六、GPTQ int4 头攻关（09-06 深夜 ~ 09-07 凌晨，四 bug 全录）

- **P33 GPTQ 死循环（2.5h 白跑）**：`while j < (c1-c0)` 缺 `j += gsz` → 诊断法：watcher 到期 + 无输出行 +
  微基准证单列 0.1-0.7ms + py-spy（宿主 ptrace 被阻 → **特权容器 --pid=host** 跑）。
- **P34 模块级循环病态慢**：同代码函数内 0.6s / 模块级 >15min，根因未明（环境/GPU/数据/启动器全排除）→
  工程解：包成 `one_block()` 函数结构（17s 完成）。教训：**同进程 A/B 自证段** 胜过无尽外部二分。
- **P35 验证 OOM**：全词表反量化 10-15GB 放不下 → CPU 验证（503G RAM）。
- **P36 缺跨块补偿，cos 卡 0.9967（≈RTN）**：块边界 `W[:,c1:] -= Err @ Hinv[c0:c1,c1:]` 缺失 → 加上即 0.9992。
  教训：合成一致性只证明两实现互同——**必须对照权威参考 + RTN 对照组**。
- **P37 int4 头四轮校准全留确定性退化**（SQL 复读/嵌套递归/中文乱词，每轮换位）→ **判死**。
  且后续 P45 证明 int4 头无速度收益——质量速度双输，永不复活。

## 七、drafter 重校准与 T 系列（09-07 凌晨）

- **P38 lm_head hook 0 行**：剪枝 logits 路径绕过 lm_head.forward → 改挂 `language_model.model.norm` 输出（=lm_head 输入）。
- **P39 capture 显存两难**：0.95 没地方放 Hessian、0.82 KV 池不够 → **0.88 甜点**。
- **P40 k/v 的 ctx_kv 分布并入 GPTQ → 接受率 -7%** → 永不混合。
- **P41 DFLASH_TOKENS 甜点唯一**：k5/k8 EngineDead、k6 灾难（58.5）→ k=7。
- **P42 240K FLA 碎片缘**：int8 头比上游重 ~1GB，首请求 42MB 级动态分配即亡 →
  KV 5.26→4.86GB + CG 1400→1000MiB 配平（省 0.8GB）。262K 差 0.2GB 归档 S4。

## 八、P0 差距归因（09-07 上午，方法论坑最多的一天）

- **P43 /metrics 计数器带 `vllm:` 前缀**：awk/python 锚定错 → 全 NA 不报错 →
  历史上"28ms 步速差"由此诞生（错误口径反推）。预防：解析后断言非空。
- **P44 隔离测量暖态假象**：同配置开机首测 vs 热机接受率差 2× → 判别实验一律全新 recreate。
  （同日再次踩 pkill 自杀变体。）
- **P45 步速差不存在**：24.1 ≡ 24.4ms——"接受率+步速双落后"被推翻；我方与同配置参照持平。
- **P46 功耗墙混杂**：历史数字须先对齐功耗档（250W/450W decode 仅 -1.5%，但 burst 明显）。
- **P47 max_model_len 几何效应**：245760（栈默认档）饿死 drafter 滑窗覆盖；≥249856（注意力块 2176 对齐
  mamba 页）打开 lookup/adaptive 16-长块通道 → 171.6 tok/s。但：
- **P48 adaptive×前缀缓存确定性输出损坏**：12 残差矩阵 4 组中招（暖命中轮内容实质分歧）→ 质量一票否决。
  已排除 lookup 内容与 KVarN 本体；嫌疑=变长 verify × resumed 请求元数据一致性。检测工具已固化
  （`eval/vllm/p0/multi_residue_test.py`）。修复路线见 ROADMAP 方向 A。
- **P49 五要素通道 OOM 缘**：171 档需 U93+KV5000 精确组合，任何稳定性修复（U95/PC0）消灭增益——
  本栈构建上不可生产。
- **P50 MTP 中文词表缺失**：上游 draft 词表丹麦/英/代码语料 → 中文 1.42 tok/step（14%/tok）→ MTP 死刑维持。
- **P51 隔离 prompt 提前 EOS 假象**：填充式 prompt 让模型早停 → tok/s=512/wall 虚高 →
  改真实 completion_tokens 计数 + 指令式长文 prompt。
- **P52 生产端并发污染测量**：cron soak 流量混入计数器差分 → 测量一律独立沙箱引擎。

## 九、媒资线（M 线）

- **P53 PyAV 四坑**：bsd 语法/模板拼接/消耗包语义/HLS 时间戳 → `hls2m4a.py` 固化解法；
  imageio-ffmpeg 静态二进制本机段错误勿用。
- **P54 reranker 三坑**：缺 `--runner pooling` / 缺 qwen3_reranker.jinja 模板 / 缺 classifier_from_token
  ["no","yes"] —— 缺一即无区分度或不报错地输出垃圾。
- **P55 vLLM 0.28+cu13 五坑**：CUDA_HOME 指系统无 toolkit / flashinfer JIT 缺 nvcc+ninja /
  采样核 CCCL 版本冲突 / 大视觉模型像素预算 OOM / pkill 自杀 →
  配方固化 `inference/vllm/build/cu130-driver580/README.md`。
- **P56 ZxBench 三坑**：nginx inode 耗尽 / prisma 生成物 / 评测 GPU 防护 → DEPLOY-OPS（独立仓库线）。
- **P57 ASR 长音乐段无上限生成**：transcriptions 端点无 max_tokens 帽 → 生产加 VAD 门限或限长。
- **P58 GLM 视频默认 1 亿像素预算**：45s 片 61824 token 超缓存全 400 → 改 longest_edge=31457280
  + --max-num-batched-tokens 8192 + --enforce-eager。
- **P59 预标双模型分歧实在**（相似度仅 0.52-0.70）→ 无自动共识，全部分级人工校对，不硬上自动。

## 十、通用教训（跨期提炼）

1. **口径纪律**：任何性能数字必须钉死 harness / 功耗档 / 引擎暖态 / 计数器语义——四个都坑过。
2. **单变量 + 全新 recreate**：配置差一个字母的结论不合并。
3. **质量一票否决**：速度增益无论多大，确定性输出损坏即否决（P48/P37）。
4. **对照组设计**：GPTQ 必带 RTN 对照；引擎对照必须同夹具同脚本。
5. **标称不信**：上下文看池实测、接受率看计数器、速度看流式首末 token 中位。
6. **坑即资产**：每个 P# 都沉淀为脚本/配置/文档/门禁，不靠记忆。
