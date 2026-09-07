#!/usr/bin/env python3
"""镜头分割·信号①：SigLIP2 逐帧嵌入 + 帧间余弦距离。mage-env 运行（GPU 或 CPU）。
用法: stage1_siglip.py <video> <workdir> [fps=2]"""
import sys, os, subprocess, glob
import numpy as np

video, workdir = sys.argv[1], sys.argv[2]
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
os.makedirs(f"{workdir}/frames", exist_ok=True)

if not glob.glob(f"{workdir}/frames/*.jpg"):
    subprocess.run(["/data/tools/ffmpeg", "-y", "-loglevel", "error", "-i", video,
                    "-vf", f"fps={fps},scale=-2:720", "-q:v", "3",
                    f"{workdir}/frames/f%06d.jpg"], check=True)
frames = sorted(glob.glob(f"{workdir}/frames/*.jpg"))
print(f"frames={len(frames)}")

import torch
from transformers import AutoModel, AutoProcessor
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("/data/models/siglip2-base-patch16-384")
model = AutoModel.from_pretrained("/data/models/siglip2-base-patch16-384").to(dev).eval()

embs = []
B = 32
for i in range(0, len(frames), B):
    from PIL import Image
    imgs = [Image.open(f).convert("RGB") for f in frames[i:i+B]]
    with torch.no_grad():
        inp = proc(images=imgs, return_tensors="pt").to(dev)
        out = model.get_image_features(**inp)
        if hasattr(out, "pooler_output"): out = out.pooler_output  # transformers 5.x 可能返回 ModelOutput
        out = torch.nn.functional.normalize(out, dim=-1)
    embs.append(out.cpu().numpy())
E = np.concatenate(embs)
np.save(f"{workdir}/siglip_emb.npy", E)

D = 1.0 - np.sum(E[:-1] * E[1:], axis=1)  # 相邻帧余弦距离
# 帧时间 = 帧序号/fps（首帧 t=0）
import json
json.dump({"fps": fps, "n_frames": len(frames), "dist": [round(float(x), 5) for x in D]},
          open(f"{workdir}/siglip_dist.json", "w"))
print(f"siglip done: {len(E)} embs, {len(D)} dists, mean={D.mean():.4f} std={D.std():.4f} max={D.max():.4f}")
