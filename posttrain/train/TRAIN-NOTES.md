# Phase 3 QLoRA 训练执行记录（2026-09-01）

## 最终生效配置（stage_a 短轨）
- 4×4090 (GPU4-7), torchrun 4 进程, ZeRO-3 no-offload, bnb NF4 double-quant
- DoRA r=32 alpha=64 rsLoRA lorap_lr_ratio=2.0, all-linear, paged_adamw_8bit
- bf16, max_length=1536, bs1×ga2×4卡, lr=2e-4 cosine→2e-5, 2 epochs=4840 步
- 实测稳态: 显存 21.6G/卡, GPU 峰值 23220MiB, 余量 1356MiB (上限 24576)

## 踩坑记录（按时间序, 全部实测复现）
1. swift 默认 torch_dtype=float32: 非量化层 emb/lm_head 以 fp32 加载, 底座直接
   17.9G→23G 级, 加载/首步即 OOM。修: 显式 --torch_dtype bfloat16 (5.1G bf16)。
2. 数据 Arrow CastError: Phase 2 产物 messages 结构不齐 — 54518 条只有
   (role,content), 9829 条多 reasoning_content; 且 tools 字段时有时无。
   datasets 按 chunk 推断 schema, 嵌套 struct 字段集/顺序不一致即 cast 失败;
   tools 补 null 也不行 (会被推断成 null 类型)。修: 每条 message 固定
   {role,content,reasoning_content} 键序, reasoning 缺省补 ""; tools 全行统一为
   string (无工具补 "", swift 对空串按无工具处理)。三个文件 datasets 加载全过。
3. step2 反向重算 OOM (max_length 无关): step1 完成时优化器态尚未分配 (20.99G),
   step1 的 optimizer.step() 分配 paged_adamw_8bit 态后, step2 的梯度检查点
   反向重算峰值 (checkpoint.py 重跑 mlp forward) 叠加超顶。4096/3072/2048 三档
   同一位置同一数值 OOM → 与序列长度无关。2048 档 366MiB 余量撑到 step72 仍穿。
   修: rank 64→32 (参数+梯度+优化器态减半) + max_length 1536, 余量 1356MiB。
4. run_train.sh set -u 下引用未绑定 RESUME_CKPT 直接退出; 已改 ${RESUME_CKPT:-}。
5. pkill -f "swift/cli/sft.py" 会匹配到执行该命令的 shell 自身命令行而自杀,
   后台启动全部悄悄失败。教训: pkill 模式不要与本命令行字符串重合。

## 显存账（实测, 单卡=每卡, ZeRO-3 只分片 LoRA 可训部分, 4bit 底座每卡全量复制）
- 4bit NF4 底座 12.40G + bf16 emb/lm_head 5.10G = 17.90G 固定底座（单卡实测）
- + LoRA(rank32) 参数/梯度/优化器态 (ZeRO-3 分片) ≈ 0.4G
- + CUDA context/NCCL ≈ 0.9G + 激活 (grad-ckpt, ≤1536 tok) ≈ 2.2G
- = 稳态 ~21.6G allocated / GPU 总占用峰值 23220MiB / 余量 1356MiB
- rank64+2048 对比: 22.9G alloc / 24210MiB / 余量仅 366MiB → step72 OOM

## 运行与守护
- 启动: nohup bash run_train.sh a > stage_a.log 2>&1 &  (GPUS=4,5,6,7 默认)
- 断点续训: RESUME_CKPT=<ckpt路径> bash run_train.sh a
- 监控: monitor.sh 单行状态; cron 每 30 分钟自动巡检, 死亡自动从 checkpoint 续
- checkpoint: 每 500 步, save_total_limit 8; 4840 步约 26-29h (19.7s/it 稳态)
- stage_b(16K)/stage_c(8K) 尚未压测显存, 启动前须先按本账重算 (rank32 下
  16K 激活约 4×1536 档, 大概率需降 8192 或减 batch/ga)

## Phase 4 热挂预检（2026-09-01 夜, 训练窗口并行完成）
1. DoRA 不可转 GGUF LoRA 适配器(convert_lora_to_gguf 无 magnitude 支持, 数学上也不可精确折叠)
   → 训练已切纯 LoRA (use_dora false), 重启于 ~21:45。rsLoRA 保留。
2. rsLoRA 缩放差: llama.cpp 运行时 scale=adapter_scale×alpha/rank, 不认 alpha/√r。
   热挂须用 --lora-scaled adapter.gguf:5.657 (√32) 补偿, 否则等效缩放差 5.66 倍。
3. 生产 llama-server (b10715) 原生支持运行时换适配器:
   --lora-init-without-apply + POST /lora-adapters, 免重启 A/B。
4. GDN out_proj (48 层, 唯一转换死点, dim-1 列重排 NotImplementedError):
   已补丁 /data/build/llama.cpp-new/ 两处 —
   convert_lora_to_gguf.py: LoraTorchTensor 增加 index_select (行→B 列→A);
   conversion/qwen.py _LinearAttentionVReorderBase._reorder_v_heads: LoRA 张量
   走 index_select 置换路径 (宿主类是 _LinearAttentionVReorderBase, 非 RND1Model)。
   合成 12 类模块适配器全量转换 exit=0, 数值验证 7/7 PASS
   (out_proj A[:,perm]/B 不变; z/a/b B[perm]; 未变换模块全等)。
5. 12 类模块 GGUF 名映射: in_proj_a→ssm_alpha, b→ssm_beta, qkv→attn_qkv(融合不拆),
   z→attn_gate, out_proj→ssm_out; self_attn q/k/v/o→attn_q/k/v/output; mlp→ffn_*。
   A 存 (rank,in), B 存 (out,rank)。
6. 转换命令(真适配器就绪后):
   python convert_lora_to_gguf.py --base /data/models/Qwen3.8-27B-BF16 \
     --outfile <adapter.gguf> --outtype f32 <swift ckpt 目录(含 adapter_config.json)>
   注意: swift checkpoint 目录若缺 adapter_config.json 需从训练产物拷贝。

## RFT 沙箱压测（缺口 #5 关闭, 2026-09-01 22:26）
- harness: /data/compose/qwen27b/rft/sandbox_bench.py (MBPP sanitized_test 257 题)
- 隔离: --network none --read-only --cap-drop ALL --memory 256m --pids-limit 64
  --cpus 0.5 --tmpfs /tmp(noexec), 内部 timeout 10s, python:3.12-alpine(本地73MB)
- 结果: 50路=0.8s(61/s) / 128路=10.8s(11.8/s) / 257路=16.1s(16/s) 全部 100% pass
- 瓶颈: >100路后 docker daemon spawn 串行化(每ct~40ms), 非CPU非RAM(内存零增长)
- 推算: RFT 一轮 300题×8采样=2400 验证 ≈ 2.5 min; 生成侧(4副本API)才是瓶颈
- 生产建议: 按 128-256 一波分批 launch; MBPP(纯stdlib)用 alpine 即可;
  SWE-Gym 真实 repo 验证需重镜像(conda/依赖), Phase 5 细设时按 repo 建 image 池

## checkpoint 实弹转换验证通过 (2026-09-02 03:36)
- 真实 swift checkpoint-1000 → ckpt1000-lora.gguf (933MB f32):
  exit=0, 992 张量=496模块×2 (48×5 GDN + 64×3 MLP + 16×4 self_attn, 与架构全吻合),
  alpha=64 KV 正确, 未变换模块逐位全等。Phase 4 GPU 侧仅剩热挂冒烟(待GPU空闲)。
- 踩坑#6: checkpoint 目录是两级 qlora-r1/v0-<ts>/checkpoint-N, 此前 glob 写成
  三级 */v0-*/checkpoint-* 恒空 → 监控 last_ckpt=0、观察器3h白等、cron 崩溃
  恢复本会误判"无checkpoint从头训"。已修 monitor.sh + cron prompt + 观察器弃用。

## 基线残缺修复 + 门禁套件 (2026-09-02 晨)
发现 baseline-3.0.json 只有 humaneval/needle/longgen 是真的, bfcl 0/2000、
gsm8k 0/0、ifeval 0 judged — 原因三连:
1. bfcl 数据文件损坏(只有 id/ground_truth, 无 question/function; GitHub/jsdelivr/
   HF镜像均取不到原始数据) → 弃用, 改建 xfc 轴
2. ifeval 两个 bug: 思考模型把 2048 token 预算耗尽致正文为空(→max_tokens 6144);
   约束 ID 是官方命名而检查器认自造短名(→重写覆盖全部 25 类官方约束, kwargs 按
   位置对齐 list)。修后冒烟 85%(17/20)
3. xfc 轴: toolace data.json 抽单轮(首问+首答 gorilla 式调用), 训练重叠已排除
   (对照 sft-short/long 的 user 查询), 200 题固定卷 /data/eval-rulers/xfc/
   sample.json。比较器 tuple/list 类型 bug 已修。
门禁套件:
- eval/compare_gate.py: 判定规则=任一轴绝对回退>2pp FAIL; humaneval/xfc 核心轴
  必须严格提升; tps 相对 2%; 输出判定表
- eval/gate.sh: 一键 A/B — GPU5 控制实例(8090, 纯底座) + GPU4 热挂实例
  (8091, --lora-scaled <adapter>:5.657 rsLoRA补偿) → all 轴 → 判定
- run_baseline.py: MTP_METRICS_URL 环境变量支持单实例 metrics; xfc 进 limit 轴
- 已知缺口: SWE/agent 轴不在门禁(原基线也未测), 由 Phase 5 RFT 沙箱回路天然覆盖
基线补测: xfc+gsm8k+ifeval 全量对生产 LB 跑, tag=baseline-3.0-backfill

## 基线补全完成 (2026-09-02 09:30, 生产LB实测)
| 轴 | 基线 | 备注 |
|---|---|---|
| humaneval | 84.76% (139/164) | 原有 |
| xfc 工具 | 63.5% (127/200) | 新轴(toolace抽取,排除训练重叠) |
| gsm8k | 93% (93/100) | 补测(修limit语义bug后) |
| ifeval | 56% (28/50, 0跳过) | 补测(修思考耗尽token+官方约束ID后) |
| needle | 64K 1/3, 195K 0/3 | 原有 |
| longgen | ~43.5 tok/s | 原有 |
比较器空转自检通过。17:20 训练完成后: bash /data/compose/qwen27b/eval/gate.sh 一键门禁。
