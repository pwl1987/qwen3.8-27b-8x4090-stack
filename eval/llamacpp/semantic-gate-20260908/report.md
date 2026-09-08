# Llama.cpp 生产语义 Gate 结案报告（2026-09-08）

**被测对象**：llama.cpp 生产主力栈（OpenResty :8000 → 2×llama-server 副本 GPU0/1，直连 :8081/:8082）
**目的**：回答「生产在 冷/暖 × 单/多轮 × cache-reuse × DFlash 草稿 下是否语义稳定」，区分
良性 token 分歧与真语义损坏（垃圾/复读/结构破坏/多轮约束丢失）。判据口径承接
`eval/vllm/p0/S4-ROOTCAUSE-20260908.md`（vLLM 方向 A 根因闭环的结论）。
**工具**：`../semantic_gate.py`（probe/matrix/canary/cmp；零 LLM Judge，四层判据全确定性）。

## 冻结判据（G2 首跑前定死，全程未调）

- BENIGN-DIFF 窗口：`0.5 ≤ len_ratio ≤ 2.0` 且公共前缀 ≥20 字符
- FAIL：乱码（常用字符占比<0.85）/ 复读环（10-120 字周期**紧邻**重复≥3 次）/ JSON 结构破坏 /
  多轮约束标记丢失（冷暖**不对称**）/ 空或过短（<10 字）
- 过门：FAIL=0 且 UNRESOLVED=0（UNRESOLVED=人工复核通道，不自动归 BENIGN）

## 测量形态

- 沙箱 = GPU2 生产 flags 逐字复刻（coding-v1.1 + draft-dflash + `-md DFlash2-Q4_K_M` +
  `--spec-draft-n-max 7` + `-c 262144 -np 1` + KV q4_0 + `--cache-reuse 2048`，镜像 b10715）
- 12 格矩阵：D1/D2 共享 token 目标档 12361..15361（步进 600，residue mod128 ∈
  {73,33,121,81,41,1}，`/tokenize` 截断实录，见 matrix.jsonl 每格 target_tokens/prompt_tokens/cache_n）
- 冷 = `cache_prompt:false`（探针实证：冷 1.6-2.0s vs 缓存 0.10-0.16s，cache_n=0 全量重算）；
  暖 = 默认缓存重放（b2 实测 cache_n/prompt_tokens ≈ 99.7%，暖命中 12/12 实证）
- 温度 0 + seed 4242 + enable_thinking=false；max_tokens 350

## 结果总表

| 轴 | 结果 | 判定 |
|---|---|---|
| 冷-冷（同格重发，12 格×2 轮） | **24/24 逐字节 EXACT** | 引擎确定性 ✓ |
| 跨重启（bootA vs bootB，4 格×4 请求） | **16/16 逐字节 EXACT**（含暖轮分歧点逐字复现） | 跨 boot 确定性 ✓ |
| 跨副本（生产 :8081 vs :8082，D2@14161×4） | **4/4 EXACT**（canary.jsonl） | 副本等价 ✓ |
| 冷-暖（12 格） | 2 EXACT / 3 BENIGN-DIFF / **7 UNRESOLVED** | 全部近平局翻转（见复核表） |
| 多轮约束（D2@13561，ZEBRA 标记×3 轮×冷暖） | 6/6 回复标记全保留 | **OK** |
| JSON 结构（2 用例×冷/暖） | 4/4 合法 JSON（JSON2 冷暖逐字节一致） | **OK** |
| LB 粘滞多轮（生产 :8000） | 3 轮标记全保留；`/_lbstats` 差分 sticky_hit=2 | **OK** |
| 垃圾/复读环 | **0 例**（38 份输出全检） | ✓ |

自动门：`FAIL=0, UNRESOLVED=7 → PASS=False`（冻结规则不允许把复核并入自动判定）。

## UNRESOLVED 逐条人工复核（7/7 → 全部裁为良性）

| 格 | 分歧位 | 现象 | 裁定 |
|---|---|---|---|
| D2@12361 | @8 | `提到：“…` vs `提到了：“…`（多一个"了"字，len_ratio 0.989，其余一致） | 良性：近平局 greedy 翻转 |
| D2@13561 | @8 | 引文选择差（"You are a progr…" vs "I have uploaded…"，语料为 jsonl 混排，"第一段"指认本就近平局） | 良性：近义引文备选 |
| D2@14161 | @8 | 同类引文/措辞翻转 | 良性 |
| D2@14761 | @8 | `提到了…原文如下：` vs `提到：“I have uploaded…` | 良性 |
| D2@15361 | @8 | 同上（bootB 复现同一分歧） | 良性 |
| D1@12361 | @64 | grep 命令变体（`grep -r "OperationType"` vs `grep -r "ast.OperationType"`），连贯备选但长度比出窗 | 良性（窗口裁 UNRESOLVED） |
| D1@15361 | @0 | 冷路延续工具调用格式 vs 暖路正常中文作答（语料头部=agent 工具转录，模型在模式边界） | 良性：病态语料上的模式级近平局 |

复核依据：①两侧输出各自通过全部质量检查（无垃圾/无复读/结构完整）；②分歧形态与 vLLM S4
定性的「近平局 greedy 翻转→连贯备选续写」完全同源；③同一分歧**跨 boot 逐字复现**（非随机
抖动，是暖轮恢复重算与冷轮分块重算的系统性数值差）；④生产金丝雀同格（D2@14161）冷-暖
EXACT——翻转出现与否取决于逐比特运行态，无论出现与否均为良性形态。

## 结论

1. **生产无事故风险**：38 份输出 0 垃圾、0 复读环、0 结构破坏、0 多轮约束丢失；
   副本间/重启间逐字节一致；LB 粘滞多轮完好。
2. **唯一分歧轴 = 冷-暖**（前缀缓存恢复 vs 全量重算），形态全部为可复现的近平局措辞/引文
   翻转，与 vLLM S4 结论同构。生产影响：多轮会话命中缓存的轮次，措辞可能与冷复算略有出入
   （一个字/一处引文选择），语义等价。对新闻生产（指令型 prompt）预期为罕见且无害。
3. **llama.cpp 确定性显著优于 vLLM**：无 per-boot 双峰、无跨副本差；分歧只在冷-暖轴且可复现。
4. **处置：现产配置冻结**（flags/拓扑不变），按路线图进入方向 B（drafter 蒸馏重训）。

## 过程记录（工具误报两类，均为机制修复+全矩阵重跑，冻结阈值未动）

- pilot1：复读检测用「30 字窗口出现≥3 次」把 D1 语料的工具调用**结构性重复**误判为复读环
  → 改为紧邻周期检测（10-120 字周期连续≥3 次），单测：结构重复 False / 真环 True。
- pilot2：`json_ok` 兜底截取先试花括号对，把合法 JSON **数组**两侧的 `[]` 剃掉误判 broken
  → 改为先整体 parse、围栏次之、括号截取数组优先。JSON2 实际输出冷暖逐字节一致。
- pilot1 另发现 D1 头部语料对多轮指令格 fixture 敌意（模型无视指令延续工具格式，冷暖对称
  失败）→ MT fixture 固定用 D2；对称失败=FIXTURE-WEAK、不对称才 FAIL（本轮 D2 版对称全过）。
- 教训：kill 后台评测进程要 kill python 实际 pid（nohup wrapper pid 无效）；两个 writer 同写
  一个 jsonl 会乱序——工具已加 .matrix.lock（含陈旧锁接管）。

## 复现

```bash
# 沙箱（生产 flags 逐字）
docker run -d --name qwen27b-gate --gpus "device=2" -v /data/models:/models \
  -p 127.0.0.1:8085:8080 llama-server:cuda12.4-b10715 \
  -m /models/Qwen3.8-27B-coding-v1.1-iq4xs.gguf --host 0.0.0.0 --port 8080 \
  -c 262144 -np 1 -ngl 999 --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 -fa on --cache-reuse 2048 \
  --jinja -a qwen3.8-27b --metrics \
  --spec-type draft-dflash -md /models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-draft-n-max 7
python3 semantic_gate.py probe  --base http://127.0.0.1:8085 --ev <dir>
python3 semantic_gate.py matrix --base http://127.0.0.1:8085 --ev <dir> --tag bootA
docker restart qwen27b-gate   # 健康后:
python3 semantic_gate.py matrix --base http://127.0.0.1:8085 --ev <dir> --tag bootB \
  --only "D1@0,D1@5,D2@0,D2@5"
python3 semantic_gate.py cmp    --ev <dir> --a bootA --b bootB
python3 semantic_gate.py canary --ev <dir>          # 生产只读金丝雀
```
