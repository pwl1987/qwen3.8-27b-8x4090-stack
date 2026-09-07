# eval/ — 门禁评测（按引擎分线）

**"比现在好"的可证伪定义**：全轴 ≥ 基线、核心轴（代码/工具）严格 >，任一轴回退 >2pp 即 FAIL。
- `llamacpp/`：gate.sh 一键 A/B 门禁 + rulers 冻结基线（humaneval 84.76 / xfc 63.5 / gsm8k 93 /
  ifeval 56 / needle / longgen）+ RFT 沙箱 + spec 四配置矩阵 + gate-verdict 判决存档
- `vllm/`：p0 差距归因工具箱（步速分解 / **12 残差前缀正确性门** / A-B 试验机）+
  spec_bench 接受率差分 + quality_ab 双引擎 10 任务质量 A/B + int4 头判决书
