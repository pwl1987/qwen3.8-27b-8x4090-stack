# p0/ — vLLM 差距归因工具箱

在单变量 A/B 中拆解投机解码引擎的 decode 吞吐构成（tok/s = steps/s × tok/step），并守护前缀缓存正确性。

| 文件 | 用法 | 说明 |
|---|---|---|
| `bench_step.py` | `python3 bench_step.py <port>` | 3 次隔离 g512 请求（temp0，中文长文指令防提前 EOS）+ `/metrics` 差分：tok/s / tok/step / ms/step / %/tok / 按位置接受剖面。计数器须匹配 `vllm:` 前缀 |
| `multi_residue_test.py` | `python3 multi_residue_test.py <port>` | **12 残差前缀命中分歧率监测**：两文档 × 6 个截断长度，暖命中轮 vs 冷盐轮逐字节对比。09-08 起语义变更：分歧≠损坏——本混合 GDN 栈的布局非确定性必然产生少量连贯备选续写（L0 7/12 / L1 3~4/12），垃圾/复读/乱码才是真损坏（未观察到）。根因见 `S4-ROOTCAUSE-20260908.md` |
| `prefix_hit_test.py` | `python3 prefix_hit_test.py <port>` | 单残差快速版（冒烟） |
| `run.sh` | `source run.sh; test_cfg <label> <env文件>` | 单变量试验机：写 `.env` → 独立 compose 项目 recreate → 等健康 → 步速分解 + `ulmus_validate.py` t3 夹具中位 + 计数器差分 → 追加 results.tsv |
| `compose.p0.yaml` | 与主 compose 叠加（`-p <项目名> -f inference/vllm/compose.yaml -f compose.p0.yaml`） | 沙箱引擎样板：`!override` 换卡换端口，不碰生产 |
| `arms/*.env` | — | 归因实验的各臂环境文件（A1 参照复刻 / A3 现产复刻 / A8c 系列 249856+lookup 判别链），留作复现模板 |
| `results.tsv` | — | 当日 20+ 臂原始结果 |
| `S4-ROOTCAUSE-20260908.md` | — | **09-08 方向 A 根因闭环结案**（残差损坏定性 + 171 双峰 + adaptive k=7 失效） |
| `evidence-20260908/` | — | 结案证据：两轮矩阵 log、trace jsonl（token 级定位）、单格输出、探针/分析脚本 |
| `evidence-20260908/t3_probe.py` | `python3 t3_probe.py <port> [N]` | t3 decode 探针：复刻 t3 请求（唯一 salt），文本 sha + /metrics 接受率差分——判别 boot 模式与文本漂移 |
| `evidence-20260908/residue_cell.py` | `python3 residue_cell.py <port> D1\|D2 <cut>` | 单格残差复测：a1/b1（冷-冷）与 a2/b2（暖命中）分开判定，四份输出落盘 |
| `evidence-20260908/trace_analyze.py` | `python3 trace_analyze.py <jsonl> <起始行>` | dflash_trace.jsonl 分析：请求分段、恢复 nct、(nct→块长) 轨迹、token 序列首分歧定位（配套容器内插桩：speculator.py/states.py 加法式改动，`/tmp/dflash_trace.on` 文件开关，见结案文档） |

## 经验教训（为什么这些工具长这样）

- **夹具与计数器口径必须二选一并贯穿始终**：流式夹具（ulmus）与单发隔离（bench_step）的 tok/step
  不可混比——内容域（重复英文 vs 中文散文）对投机接受率影响巨大。
- **暖引擎状态会骗人，且比想的更骗人**：同一配置开机首测与热机后的接受率可差 2×；09-08 进一步
  定性——t3 类刀刃内容的接受率呈 **per-boot 双峰**（~41% vs ~27%，boot 内粘滞、跨 boot 随机，
  g512/canary 补救无效），单次 boot 的数字不可作为配置结论，重要 A/B 至少 3 个独立容器实例。
- **`/metrics` 前缀**：vLLM prometheus 计数器带 `vllm:` 前缀，awk/python 锚定错误会得到全 NA 而非报错。
- **功耗对齐**：对照历史数字前先核对功耗墙（250W vs 450W 对 decode 约 -1.5%，但对 burst 类测量更明显）。
