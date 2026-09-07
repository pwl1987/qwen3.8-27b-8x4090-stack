# ComfyUI + MiniMax-H3 视频生成服务（GPU2/3 双 4090，按需启停）

2026-09-05 部署，`docker start comfyui` 即恢复（当前停机状态是常态——GPU2/3 白天另作他用）。
入口 http://<内网IP>:8188（ufw 已放行），已接入 mon :9000 探活。

## 文件

| 文件 | 说明 |
|---|---|
| `docker-compose.yml` | 双卡、user 1000、挂 models/output/user、加入 qwen27b_default 网络供 mon 探活；注释含回滚与单卡回退（503G RAM 自动 offload） |
| `Dockerfile` | python:3.12-slim + 阿里源；torch 2.11.0+cu130 三件套**钉死同版本**，wheels 宿主下载后本地 COPY |
| `bootcheck.sh` | @reboot 开机自检（cron 延迟 60s+90s）：驱动/容器/LB/comfyui 探活 → 日志 |
| `download_models.sh` | 模型下载器 v2：wget -c 断点续传 + 字节数硬校验，16 文件幂等清单 |
| `dl.sh` | hf-mirror 并行下载器（curl --retry 20，5 并行） |
| `workflows_api/smoke_t2v.json` | API 冒烟工作流（768×432 56 帧 8 步，40s 出片） |

模型放置（不入库，~70G）：`diffusion_models/`（fp8 pruned + int8 convrot 两版 21G）、
`text_encoders/`（qwen3vl-32b-nvfp4-awq 15.7G）、`vae/`（video fp16 5.2G + audio fp32 0.6G）、
`loras/`（turbo 8-step 2G）、`embeddings/`（10 个官方触发词）。

## 部署史三坑（详见 TRAIN-NOTES 09-05 节 + PROBLEMS-AND-FIXES）

1. **buildkit 层内下载卡死** → torch wheels 改宿主 wget 后本地 COPY 进镜像；
2. **uid 1000 + passwd 条目**必须显式建（dynamo getuser 依赖）；
3. ComfyUI 版本回滚（0.34.0 源码快照 COPY，从 cu128 回滚镜像提取）。

驱动 550→580 升级后以 torch cu130 重建（TRAIN-NOTES 09-05 晚节）；H3 三项优化实测
（功耗墙/convrot int8/SageAttention）见 TRAIN-NOTES 09-06 节——**sageattention 1.0.6 实测更慢已弃用**。
