# serve/ — vLLM 服务启停脚本（媒资线小模型池）

8 个模型的 vLLM serve 启停（配 `env.sh` 公共环境 + `stop.sh` 统一停）。
运行底座 = `/data/tools/vllm28-env`（vLLM 0.28 + cu13，**五坑配方**见
`inference/vllm/build/cu130-driver580/README.md`：CUDA_HOME=venv 内 cu13 / PATH+ninja /
VLLM_USE_FLASHINFER_SAMPLER=0 / 大视觉模型像素预算 / pkill 自杀）。

| 脚本 | 模型 | 卡/端口 |
|---|---|---|
| `asr-qwen3.sh` | Qwen3-ASR-1.7B（原生 /v1/audio/transcriptions） | 801X |
| `text-qwen35-08b.sh` | Qwen3.5-0.8B 文本 | 801X |
| `vl-qwen35-2b.sh` | Qwen3.5-VL-2B（编目语义层主力） | GPU2 :8010 |
| `vl-glm-flash.sh` | GLM-4.6V-Flash（视频） | 801X |
| `vl-minicpm45.sh` | MiniCPM4.5-VL（需 --trust-remote-code） | 801X |
| `vl-qwen36-27b-tp2.sh` | Qwen3.6-VL-27B 双卡 TP2（单卡 KV 不够 45s 视频） | 801X |
| `vl-video-ora.sh` | ORA 视频 | 801X |

横评换装队列：`bench_queue.sh`（GPU2:8010 自动换装跑 video_bench）。
