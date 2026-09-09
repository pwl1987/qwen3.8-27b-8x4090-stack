#!/usr/bin/env python3
"""B2-B export: QAT masters -> mixed-precision drafter dir.

Base = Qwen3.8-27B-DFlash2-b1-fc16 (S3: fc bf16 + int4 layers, old-Hessian —
the exact layer function QAT trained against).  Only fc.weight and
candidate_selector.* are overwritten with the trained bf16 tensors; every
other tensor — including all 35 packed int4 layer projections — is asserted
bitwise-identical to the base dir after the rewrite.

Usage: python export_qat.py --ckpt /data/sandbox/ab-vllm/b2b/qat-b2b-1/ckpt-2000.pt \
        --out /data/sandbox/ab-vllm/repo/models/Qwen3.8-27B-DFlash2-b2b
"""
import argparse
import hashlib
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

BASE = "/data/sandbox/ab-vllm/repo/models/Qwen3.8-27B-DFlash2-b1-fc16"
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
    for f in os.listdir(BASE):
        if f in ("model.safetensors", ".cache"):
            continue
        if os.path.isfile(os.path.join(BASE, f)):
            shutil.copy2(os.path.join(BASE, f), os.path.join(args.out, f))

    sd = load_file(os.path.join(BASE, "model.safetensors"))
    replaced = {}
    for k, wkey in KEYMAP.items():
        if k not in masters:
            continue
        new = masters[k].to(torch.bfloat16)
        assert new.shape == sd[wkey].shape, (k, new.shape, sd[wkey].shape)
        replaced[k] = {"dL2_abs": round((new.float() - sd[wkey].float()).norm().item(), 4),
                       "rel": round(((new.float() - sd[wkey].float()).norm()
                                     / sd[wkey].float().norm()).item(), 6)}
        sd[wkey] = new.contiguous()
    save_file(sd, os.path.join(args.out, "model.safetensors"))

    # the deployment function must be untouched: everything outside KEYMAP
    # stays bitwise-identical (incl. all 35 packed int4 layer projections)
    after = load_file(os.path.join(args.out, "model.safetensors"))
    assert set(after) == set(sd)
    for k in after:
        if k not in KEYMAP.values():
            assert torch.equal(after[k], sd[k]), k

    stack_prov = json.load(open(
        "/data/sandbox/ab-vllm/b2b/frozen-int4-stack.pt.provenance.json"))
    prov = {
        "ckpt": args.ckpt, "ckpt_sha16": sha256(args.ckpt),
        "out_sha16": sha256(os.path.join(args.out, "model.safetensors")),
        "base_dir": BASE, "base_sha16": sha256(os.path.join(BASE, "model.safetensors")),
        "layers_bitwise_unchanged": True, "replaced": replaced,
        "chain": {
            "L0_qat_masters": sha256(args.ckpt),
            "L1_frozen_stack": stack_prov["frozen_stack_sha16"],
            "L2_s3_source": stack_prov["level1_s3_model_sha16"],
            "L3_recipe": stack_prov["level2_recipe_sha16"],
            "L4_hessian_calib": stack_prov["level3_hessian_sha16"],
        },
    }
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
