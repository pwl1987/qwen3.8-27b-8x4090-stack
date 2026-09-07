# llama.cpp 生产线（生产主力）

4×llama-server（b10715, cu12.4 构建）+ OpenResty LB，GPU0-3，唯一入口 :8000。

## 部署

```bash
docker compose up -d          # 4 副本 + LB；--profile mon 加监控页 :9000
```

- 每副本 1 卡 4 slot、256K ctx、`-fa on`、`--cache-reuse 2048`、`--spec-type draft-mtp`（内嵌 MTP 层，接受率 46.6%）
- r2/r3 大会话池（单 slot 独享 256K）；llama/r1 小会话池
- 权重：`Qwen3.8-27B-Heretic-Ara-iq4_xs-3.0-mtp.gguf`（14.33G，显存账见根 README）
- 构建配方：`build/cu124-driver550/`（ghcr 被墙的本地构建解，P1）

## 性能（2026-08-31 双构建复核）

64K 单流 71.8-79.8 tok/s；195K 衰减 ~60%（28.3-35.3）；prefill 2230@64K；
16 slot 聚合 ~420 tok/s（48 路并发排队全成功）。

## 相关

- 负载均衡机制：`lb/README.md`
- 门禁与基线：`../../eval/llamacpp/`
- LoRA 热挂（A/B 免重启）：`../../training/llamacpp/` 的转换补丁 + `--lora-scaled :5.657`
