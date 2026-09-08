#!/usr/bin/env python3
"""Host-side replica of capture_dflash2.reduce_wide(): X^T X over the dumped
wide rows (fc 25600-dim, down_proj 17408-dim) on a free host GPU."""
import json, os, sys
import numpy as np, torch

OUT = "/data/sandbox/ab-vllm/b1/hessians-capture"
sys.path.insert(0, "/data/sandbox/ab-vllm/repo/drafter")
from gptq_utils import accumulate_hessian

res = torch.load(f"{OUT}/hessians_small.pt", map_location="cpu")
meta = json.load(open(f"{OUT}/rows_meta.json"))
print("small keys:", len(res), "wide:", meta)
dev = "cuda"
for k, m in meta.items():
    n, K = m["n"], m["K"]
    mm = np.load(f"{OUT}/rows_{k}.npy", mmap_mode="r")
    H = torch.zeros(K, K, device=dev, dtype=torch.float32); seen = 0
    for a in range(0, n, 8192):
        X = torch.from_numpy(np.array(mm[a:min(n, a + 8192)])).view(torch.bfloat16).to(dev)
        H, seen = accumulate_hessian(H, X, seen)
    res[k] = {"H": H.cpu(), "n": seen}
    del H; torch.cuda.empty_cache()
    print(f"reduced {k}: {seen} rows", flush=True)
torch.save(res, f"{OUT}/hessians.pt")
print("saved hessians.pt:", {k: (v['H'].shape, v['n']) for k, v in res.items() if k=='fc'})
