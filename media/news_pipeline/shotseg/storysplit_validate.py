#!/usr/bin/env python3
"""M7/C 验收：storysplit 金标 clip 首尾帧 → SigLIP 最近邻定位整期时间轴 → 与 shots.json 对齐。
mage-env 运行。用法: storysplit_validate.py <episode_key 如20260904> <shots_workdir>"""
import sys, json, os, subprocess, time
import numpy as np

EP, W = sys.argv[1], sys.argv[2]
GT = json.load(open("/data/datasets/lytv_storysplit_groundtruth.json"))[EP]
CDIR = f"/data/datasets/lytv_storysplit/{EP}"
os.makedirs(CDIR, exist_ok=True)

# 1) 下载 clips（公开直链）
paths = []
for c in GT["clips"]:
    p = f"{CDIR}/{c['id']}.mp4"
    if not os.path.exists(p):
        subprocess.run(["curl", "-sL", "--max-time", "120", "-o", p, c["url"]], check=False)
    if os.path.exists(p) and os.path.getsize(p) > 100000:
        paths.append((p, c))
print(f"clips downloaded: {len(paths)}/{len(GT['clips'])}")

# 2) 整期嵌入与 clip 首尾帧嵌入
E = np.load(f"{W}/siglip_emb.npy"); fps = json.load(open(f"{W}/siglip_dist.json"))["fps"]
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
proc = AutoProcessor.from_pretrained("/data/models/siglip2-base-patch16-384")
model = AutoModel.from_pretrained("/data/models/siglip2-base-patch16-384").eval()

def embed_img(pil):
    with torch.no_grad():
        inp = proc(images=[pil], return_tensors="pt")
        out = model.get_image_features(**inp)
        if hasattr(out, "pooler_output"): out = out.pooler_output
        return torch.nn.functional.normalize(out, dim=-1).numpy()[0]

def grab(path, t):
    f = f"/tmp/gt_{os.path.basename(path)}_{t}.jpg"
    subprocess.run(["/data/tools/ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", path,
                    "-frames:v", "1", "-q:v", "3", f], check=True)
    return embed_img(Image.open(f).convert("RGB"))

gt_bounds = []
for p, c in paths:
    dur = subprocess.run(["/data/tools/ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip()
    dur = float(dur or 0)
    v0 = grab(p, 0.1); v1 = grab(p, max(dur - 0.3, 0.5))
    i0 = int(np.argmax(E @ v0)); i1 = int(np.argmax(E @ v1))
    gt_bounds.append({"id": c["id"], "title": c["title"], "t_start": round(i0/fps, 2), "t_end": round(i1/fps, 2)})
    print(f"  {c['title'][:22]}  {i0/fps:6.1f} → {i1/fps:6.1f}s")

# 3) 与 shots.json 对齐（±2s 容差召回）
shots = json.load(open(f"{W}/shots.json"))
det = [s["t_start_ms"]/1000 for s in shots["shots"]]
hits = miss = 0
for g in gt_bounds:
    ok = any(abs(g["t_start"] - d) <= 2.0 for d in det)
    hits += ok; miss += (not ok)
json.dump({"episode": EP, "gt_bounds": gt_bounds, "detected_shots": len(det),
           "boundary_recall@2s": round(hits/max(len(gt_bounds),1), 3)},
          open(f"{W}/gt_eval.json", "w"), ensure_ascii=False, indent=1)
print(f"\n== 条目边界召回@±2s: {hits}/{len(gt_bounds)} = {hits/max(len(gt_bounds),1)*100:.0f}%  (检出镜头数 {len(det)}) ==")
