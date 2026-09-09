#!/usr/bin/env python3
"""B2-B frozen-stack build: dequantize the 35 int4 layer projections of the
S3 drafter (Qwen3.8-27B-DFlash2-b1-fc16: fc bf16 + int4 layers, old-Hessian)
into dense bf16 — the deployment-equivalent layer function QAT trains against.

What is taken from S3 (packed int4 -> dense, engine-effective values):
  layers.{i}.self_attn.{q,k,v,o}_proj, layers.{i}.mlp.{gate,up}_proj,
  layers.{i}.mlp.down_proj                                    (7 x 5 = 35)
What stays from the baseline bf16 drafter (never quantized by recipe):
  conv kernel projections, base kernels, norms, embed/lm_head/fc/selector.

Sanity asserted at build time:
  - every dequantized matrix rel-err vs baseline bf16 is in [1%, 30%]
    (B2-A measured int4 layer noise ~13% rel);
  - S3's non-quantized tensors are bitwise-equal to the baseline drafter's;
  - per-matrix rel err recorded to provenance (noise fingerprint).

Usage (CPU ok):
  /data/vllm/venv/bin/python frozen_stack.py \
      [--out /data/sandbox/ab-vllm/b2b/frozen-int4-stack.pt]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, "/data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b1")
from drafter_torch import NL, dequant_packed, load_layer  # noqa: E402
from safetensors import safe_open  # noqa: E402

S3 = "/data/sandbox/ab-vllm/repo/models/Qwen3.8-27B-DFlash2-b1-fc16"
BASE = "/data/sandbox/ab-vllm/repo/models/Qwen3.8-27B-DFlash2"
RECIPE = "/data/sandbox/ab-vllm/repo/drafter/quant_dflash2.py"
HESSIAN = "/data/sandbox/ab-vllm/repo/drafter/hessians_noctx.pt"

# DrafterTorch layer-dict key -> checkpoint subkey (below)
LAY = {
    "qw": "self_attn.q_proj", "kw": "self_attn.k_proj",
    "vw": "self_attn.v_proj", "ow": "self_attn.o_proj",
    "gw": "mlp.gate_proj", "uw": "mlp.up_proj", "dw": "mlp.down_proj",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/sandbox/ab-vllm/b2b/frozen-int4-stack.pt")
    args = ap.parse_args()

    with safe_open(os.path.join(BASE, "model.safetensors"), framework="pt",
                   device="cpu") as f:
        W = {k: f.get_tensor(k) for k in f.keys()}

    stack, rel_errs = {}, {}
    for i in range(NL):
        for dk, sub in LAY.items():
            ck = f"layers.{i}.{sub}"
            dense = dequant_packed(os.path.join(S3, "model.safetensors"),
                                   ck).to(torch.bfloat16)
            ref = W[ck + ".weight"]
            assert dense.shape == ref.shape, (ck, dense.shape, ref.shape)
            rel = ((dense.float() - ref.float()).norm() / ref.float().norm()).item()
            assert 0.01 < rel < 0.30, f"{ck}: rel err {rel:.4f} outside [1%,30%]"
            stack[f"layers.{i}.{dk}"] = dense
            rel_errs[ck] = round(rel, 4)
        if i == 0:
            print(f"  layer0 rel errs: {list(rel_errs.values())[-7:]}")

    # the recipe leaves these bf16 in S3: they must be bitwise baseline values
    with safe_open(os.path.join(S3, "model.safetensors"), framework="pt",
                   device="cpu") as f:
        s3keys = set(f.keys())
        checked = 0
        for i in range(NL):
            lp = load_layer(W, i)
            for k, ckpt_key in [
                ("ac_k", f"layers.{i}.attention_conv.kernel_projection.weight"),
                ("mc_k", f"layers.{i}.mlp_conv.kernel_projection.weight"),
                ("ac_b", f"layers.{i}.attention_conv.base_kernel"),
                ("mc_b", f"layers.{i}.mlp_conv.base_kernel"),
                ("iln", f"layers.{i}.input_layernorm.weight"),
                ("pln", f"layers.{i}.post_attention_layernorm.weight"),
            ]:
                assert ckpt_key in s3keys, f"S3 missing bf16 tensor {ckpt_key}"
                assert torch.equal(f.get_tensor(ckpt_key), lp[k]), ckpt_key
                checked += 1
        # fc / candidate_selector in S3 are the TRAINED b1-1 tensors (S3 was
        # quantized from the b1 export) — deliberately NOT taken from S3;
        # QAT masters start from baseline. Only the never-quantized side
        # tensors must be bitwise baseline values.
        for k in ("hidden_norm.weight", "norm.weight"):
            assert torch.equal(f.get_tensor(k), W[k]), k
            checked += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(stack, args.out)
    prov = {
        "frozen_stack_sha16": sha256(args.out),
        "level1_source": "S3 = fc-bf16 + int4 layers (old-Hessian), engine 3.3428",
        "level1_s3_model_sha16": sha256(os.path.join(S3, "model.safetensors")),
        "level2_recipe_sha16": sha256(RECIPE),
        "level3_hessian_sha16": sha256(HESSIAN),
        "n_matrices": len(stack),
        "bitwise_bf16_sidechecks": checked,
        "rel_err": {"mean": round(sum(rel_errs.values()) / len(rel_errs), 4),
                    "min": min(rel_errs.values()), "max": max(rel_errs.values())},
        "per_matrix": rel_errs,
    }
    with open(args.out + ".provenance.json", "w") as fp:
        json.dump(prov, fp, indent=1)
    print(json.dumps({k: v for k, v in prov.items() if k != "per_matrix"},
                     indent=1))


if __name__ == "__main__":
    main()
