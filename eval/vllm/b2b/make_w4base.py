#!/usr/bin/env python3
"""B2-B 修订2: build the W4-BASELINE drafter (same-family PRIMARY control).

W4-BASELINE = pristine baseline masters (fc + candidate_selector, bf16) x the
exact same 35-matrix int4 old-Hessian layer function as S3 / QAT-W4.  Built by
copying S3 (Qwen3.8-27B-DFlash2-b1-fc16) and overwriting ONLY the 4 KEYMAP
tensors with pristine values from the baseline drafter; every other tensor —
including all 35 packed int4 layer projections — is asserted bitwise-identical
to S3.  QAT-W4 vs this dir isolates the training effect on the deployment
function (CONTRACT-B2B.md §8.2).

Usage: python make_w4base.py \
        --out /data/sandbox/ab-vllm/repo/models/Qwen3.8-27B-DFlash2-b2b-w4base
"""
import argparse
import hashlib
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

BASE = "/data/sandbox/ab-vllm/repo/models/Qwen3.8-27B-DFlash2-b1-fc16"  # S3
PRISTINE = "/data/sandbox/ab-vllm/repo/models/Qwen3.8-27B-DFlash2/model.safetensors"
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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(BASE):
        if f in ("model.safetensors", ".cache", "provenance.json"):
            continue
        if os.path.isfile(os.path.join(BASE, f)):
            shutil.copy2(os.path.join(BASE, f), os.path.join(args.out, f))

    sd = load_file(os.path.join(BASE, "model.safetensors"))
    pristine = load_file(PRISTINE)
    replaced = {}
    for k, wkey in KEYMAP.items():
        new = pristine[wkey]
        assert new.shape == sd[wkey].shape, (k, new.shape, sd[wkey].shape)
        assert new.dtype == sd[wkey].dtype, (k, new.dtype, sd[wkey].dtype)
        rel = ((new.float() - sd[wkey].float()).norm() / sd[wkey].float().norm()).item()
        # sanity: S3's KEYMAP are trained values, must differ from pristine (pred
        # codebook barely moves in training: rel ~5e-5 at ckpt-1500 — see b1-1 history)
        assert rel > 1e-6, (k, rel)
        replaced[k] = {"rel_vs_s3": round(rel, 6)}
        sd[wkey] = new.contiguous()
    save_file(sd, os.path.join(args.out, "model.safetensors"))

    # the deployment function must be untouched: everything outside KEYMAP
    # stays bitwise-identical to S3 (incl. all 35 packed int4 layer projections)
    after = load_file(os.path.join(args.out, "model.safetensors"))
    assert set(after) == set(sd)
    for k in after:
        if k not in KEYMAP.values():
            assert torch.equal(after[k], sd[k]), k

    stack_prov = json.load(open(
        "/data/sandbox/ab-vllm/b2b/frozen-int4-stack.pt.provenance.json"))
    prov = {
        "artifact": "W4-BASELINE (CONTRACT-B2B.md 8.2)",
        "pristine_source": PRISTINE, "pristine_sha16": sha256(PRISTINE),
        "out_sha16": sha256(os.path.join(args.out, "model.safetensors")),
        "base_dir": BASE, "base_sha16": sha256(os.path.join(BASE, "model.safetensors")),
        "layers_bitwise_unchanged": True, "replaced": replaced,
        "chain": {
            "L0_pristine_baseline": sha256(PRISTINE),
            "L1_s3_source": stack_prov["level1_s3_model_sha16"],
            "L2_recipe": stack_prov["level2_recipe_sha16"],
            "L3_hessian_calib": stack_prov["level3_hessian_sha16"],
        },
    }
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
