# eval/llamacpp — llama.cpp 线门禁

- `gate.sh`：GPU5 控制实例（纯底座）+ GPU4 热挂实例（--lora-scaled :5.657）→ 全轴 → 判定表
- `compare_gate.py`：判定规则（任一轴 >2pp 回退 FAIL；humaneval/xfc 必须严格提升；tps 2%）
- `run_baseline.py`：评测执行器（humaneval/xfc/gsm8k/ifeval/needle/longgen/tps）
- `rulers/`：冻结基线（baseline-3.0-backfill2.json 等）+ gate-verdict-r1.md（r1 判决 FAIL 存档：
  needle_64k 回退 + longgen -23.4% → 正确拒绝上线）
- `rft/`：RFT 验证沙箱（Docker --network none --read-only --cap-drop ALL，MBPP sanitized_test；
  257 路 16.1s 全 pass，瓶颈在 docker spawn）
- `spec_sandbox.sh` / `dflash_vram_sweep.sh`：spec 四配置矩阵 / DFlash2 VRAM 扫描
- `semantic_gate.py`：**生产语义 Gate**（2026-09-08，probe/matrix/canary/cmp）——冷/暖×单/多轮×
  cache-reuse×DFlash 草稿的语义稳定性四层判定（EXACT/BENIGN-DIFF/UNRESOLVED/FAIL，零 LLM
  Judge，阈值冻结）。结案：FAIL=0、唯一分歧轴=冷-暖近平局翻转（全部裁良性）、跨副本/跨重启
  逐字节一致 → 现产配置冻结。证据与逐条复核见 `semantic-gate-20260908/report.md`
