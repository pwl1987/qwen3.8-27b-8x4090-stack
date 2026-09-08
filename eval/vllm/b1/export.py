#!/usr/bin/env python3
"""B1-C export: trained masters -> a new bf16 drafter model dir.

Copies the pristine drafter dir (config/README) and writes model.safetensors
with ONLY fc.weight and candidate_selector.* replaced by the trained tensors
(downcast to bf16).  The pristine checkpoint is never touched.

Usage: python export.py --ckpt /data/sandbox/ab-vllm/b1/ckpts/b1-1/ckpt-1000.pt \
         --out /data/sandbox/ab-vllm/b1/export/b1-1-1000
"""
import argparse
import hashlib
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

SRC = "/data/sandbox/ab-vllm/repo/models/Qwen3.8-27B-DFlash2"
KEYMAP = {"fc": "fc.weight",
          "hp": "candidate_selector.hidden_projection.weight",
          "pred": "candidate_selector.predecessor_codebook",
          "succ": "candidate_selector.successor_codebook"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    masters = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(SRC):
        if f == "model.safetensors" or f == ".cache":
            continue
        shutil.copy2(os.path.join(SRC, f), os.path.join(args.out, f))

    sd = load_file(os.path.join(SRC, "model.safetensors"))
    replaced = {}
    for k, wkey in KEYMAP.items():
        if k in masters:
            new = masters[k].to(torch.bfloat16)
            assert new.shape == sd[wkey].shape, (k, new.shape, sd[wkey].shape)
            delta = (new.float() - sd[wkey].float()).norm().item()
            sd[wkey] = new
            replaced[k] = {"dL2_abs": round(delta, 4),
                           "rel": round(delta / sd[wkey].float().norm().item(), 6)}
    save_file(sd, os.path.join(args.out, "model.safetensors"))
    prov = {
        "ckpt": args.ckpt,
        "ckpt_sha16": sha256(args.ckpt),
        "out_sha16": sha256(os.path.join(args.out, "model.safetensors")),
        "src_sha16": sha256(os.path.join(SRC, "model.safetensors")),
        "replaced": replaced,
    }
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
