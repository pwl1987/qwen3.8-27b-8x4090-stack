# p0/ — vLLM 差距归因工具箱

在单变量 A/B 中拆解投机解码引擎的 decode 吞吐构成（tok/s = steps/s × tok/step），并守护前缀缓存正确性。

| 文件 | 用法 | 说明 |
|---|---|---|
| `bench_step.py` | `python3 bench_step.py <port>` | 3 次隔离 g512 请求（temp0，中文长文指令防提前 EOS）+ `/metrics` 差分：tok/s / tok/step / ms/step / %/tok / 按位置接受剖面。计数器须匹配 `vllm:` 前缀 |
| `multi_residue_test.py` | `python3 multi_residue_test.py <port>` | **12 残差前缀命中正确性门**：两文档 × 6 个截断长度，暖命中轮 vs 冷盐轮逐字节对比。用于检测 adaptive 块长 × 前缀缓存的确定性输出损坏（实测可抓到 4/12 中招） |
| `prefix_hit_test.py` | `python3 prefix_hit_test.py <port>` | 单残差快速版（冒烟） |
| `run.sh` | `source run.sh; test_cfg <label> <env文件>` | 单变量试验机：写 `.env` → 独立 compose 项目 recreate → 等健康 → 步速分解 + `ulmus_validate.py` t3 夹具中位 + 计数器差分 → 追加 results.tsv |
| `compose.p0.yaml` | 与主 compose 叠加（`-p <项目名> -f compose.yaml -f compose.p0.yaml`） | 沙箱引擎样板：`!override` 换卡换端口，不碰生产 |
| `arms/*.env` | — | 归因实验的各臂环境文件（A1 参照复刻 / A3 现产复刻 / A8c 系列 249856+lookup 判别链），留作复现模板 |
| `results.tsv` | — | 当日 20+ 臂原始结果 |

## 经验教训（为什么这些工具长这样）

- **夹具与计数器口径必须二选一并贯穿始终**：流式夹具（ulmus）与单发隔离（bench_step）的 tok/step
  不可混比——内容域（重复英文 vs 中文散文）对投机接受率影响巨大。
- **暖引擎状态会骗人**：同一配置开机首测与热机后的接受率可差 2×，判别实验一律全新 recreate。
- **`/metrics` 前缀**：vLLM prometheus 计数器带 `vllm:` 前缀，awk/python 锚定错误会得到全 NA 而非报错。
- **功耗对齐**：对照历史数字前先核对功耗墙（250W vs 450W 对 decode 约 -1.5%，但对 burst 类测量更明显）。
