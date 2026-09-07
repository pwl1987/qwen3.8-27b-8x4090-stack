# ops/ — 机器级运维（引擎无关）

- `mon/`：监控 v3 —— collect.py/index.html（:9000，2s 粒度 GPU+副本 slot+vLLM 主力，24h 历史）+
  collect.sh + soak.sh（*/5 cron：LB 健康 + 8081/8082 spec 计数器 + VRAM 旁路留痕）
- `scripts/`：gpu-power.sh v2（逐卡 day/quiet/night 功耗墙 + gpu-power.conf + holidays/workdays
  日历，root cron 08:00/18:00；250W 对 decode 仅 -1.5%）+ noise-mode.sh（白天软摘除 r2/r3）+
  bootcheck.sh（@reboot 自检：驱动/容器/LB/comfyui 探活 → 日志）
