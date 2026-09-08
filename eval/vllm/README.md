# eval/vllm — vLLM 线评测

- `p0/`：差距归因工具箱（bench_step 隔离步速+按位置接受剖面 / multi_residue 12 残差前缀
  正确性门 / prefix_hit 冒烟 / run.sh 单变量试验机 / compose.p0 沙箱 / arms 配置模板 / results.tsv）
- `b0/`：**B 线复刻地基（2026-09-08 过门）**——bf16 双侧纯 torch 数值复刻：ask 级引擎插桩
  patch + replay.py 逐级对齐（fc/top-16/selector scores/走链）+ 分位数门
  （mean≥14/16、p10≥12/16、path≥95%），实测 15.83/15/98.8%。见 `b0/REPORT.md`
- `b1/`：**drafter 蒸馏训练闭环（2026-09-08 B1-A 过门）**——五层 conv/attn 纯 torch 复刻
  （B 级门全过：sh_cos 0.9999/top16 15.81/anchor 243/243/k 2.51≈引擎 2.56）+ 平票契约 +
  teacher 监督构建器（三率监督门）+ 三组分 speculation-aware loss 训练器（zero 臂零更新
  断言、b1-1 smoke 无 NaN）。见 `b1/REPORT.md` 与 `b1/loss_contract.md`
- `spec_bench.py`：接受率测量（short×3 / longgen / prefill + /metrics 差分——注意 `vllm:` 前缀，P43）
- `quality_ab_v11.py` / `quality_ab_analyze.py`：双引擎 10 任务质量 A/B（temp0 / enable_thinking
  显式 / max_tokens 8192；ast 语法检查 + 单测执行 + 并排报告）
- `quality_ab_head4_vs_head8.md`：int4 头判死判决书（四轮校准退化表）
