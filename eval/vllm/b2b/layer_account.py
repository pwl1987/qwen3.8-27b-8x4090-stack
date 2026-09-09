#!/usr/bin/env python3
"""B2-B 修订2 §8.5: per-layer operator-level error account (report-only, no gate).

Three ledgers for the 35 quantized matrices (aggregated to 5 layers):
  1. weight rel err per layer      — from frozen-stack provenance (already stored)
  2. per-layer hidden cosine       — query-path layer outputs, bf16 stack vs int4
                                     stack, same masters (b1-1 ckpt-2000), DEV runs
  3. leave-one-layer-bf16 walk     — restore layer i to baseline, keep other four
                                     int4; own_walk_k marginal vs full-int4 stack

Answers: is the int4 damage layer-uniform or dominated by a few layers — the
evidence base for any future layer-selective QAT.

Usage (GPU4):
  CUDA_VISIBLE_DEVICES=4 python layer_account.py \
      --ckpt /data/sandbox/ab-vllm/b1/ckpts/b1-1/ckpt-2000.pt \
      --out /tmp/b2b/layer-account.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "/data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b1")
sys.path.insert(0, "/data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b2b")
from drafter_torch import DrafterTorch, load_pairs, segment_runs  # noqa: E402
from gate import load_ckpt, replay_dev  # noqa: E402
from gate_qat import replay  # noqa: E402
from overlay import KEYS  # noqa: E402

NL = 5


def build(ckpt: str, stack: dict, overlay: bool, skip: int | None):
    """Fresh model + masters; if overlay, int4 stack applied to all layers
    except `skip` (skip=None -> all layers int4; overlay=False -> pure bf16)."""
    model = DrafterTorch(torch.device("cuda"), embed_source="dequant")
    load_ckpt(model, ckpt)
    if overlay:
        for i, lp in enumerate(model.layers):
            if i == skip:
                continue
            for dk in KEYS:
                lp[dk] = stack[f"layers.{i}.{dk}"].to(model.dev, torch.bfloat16)
            lp["guw"] = torch.cat([lp["gw"], lp["uw"]], 0)
    return model


def capture_walk(ckpt: str, stack: dict, overlay: bool, skip: int | None,
                 dev, cap=None):
    model = build(ckpt, stack, overlay, skip)
    if cap is not None:
        orig, idx = model._layer, {"i": 0}
        def wrapped(lp, x, residual, qpos, Kc, Vc, kpos):
            out = orig(lp, x, residual, qpos, Kc, Vc, kpos)
            cap.setdefault(idx["i"], []).append(out[0].detach().float().flatten().cpu())
            idx["i"] = (idx["i"] + 1) % NL
            return out
        model._layer = wrapped
    stats = {"fc_cos": [], "sh_cos": [], "top16": [], "recall": [], "top1": [],
             "rank": [], "margin": [], "own_walk_k": [], "engine_k": []}
    with torch.no_grad():     # metrics-only replay: masters carry requires_grad,
        replay_dev(model, dev, stats)   # graphs would pin ~8GB activations per replay
    if cap is not None:
        model._layer = None   # break wrapper<->bound-method cycle (model is ~10GB)
        import gc
        gc.collect()
    del model
    torch.cuda.empty_cache()
    return round(sum(stats["own_walk_k"]) / max(1, len(stats["own_walk_k"])), 4), \
        round(sum(stats["recall"]) / max(1, len(stats["recall"])), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/sandbox/ab-vllm/b1/ckpts/b1-1/ckpt-2000.pt")
    ap.add_argument("--frozen-stack",
                    default="/data/sandbox/ab-vllm/b2b/frozen-int4-stack.pt")
    ap.add_argument("--provenance",
                    default="/data/sandbox/ab-vllm/b2b/frozen-int4-stack.pt.provenance.json")
    ap.add_argument("--ask-dir", default="/data/sandbox/ab-vllm/b1/trace-v2")
    ap.add_argument("--out", default="/tmp/b2b/layer-account.json")
    args = ap.parse_args()

    stack = torch.load(args.frozen_stack, map_location="cpu", weights_only=False)
    pairs = load_pairs(args.ask_dir)
    dev = [r for i, r in enumerate(segment_runs(pairs)) if 30 <= i <= 37]

    # ledger 2+3 references
    cap_bf, cap_i4 = {}, {}
    walk_bf16, _ = capture_walk(args.ckpt, stack, overlay=False, skip=None, dev=dev,
                                cap=cap_bf)
    walk_i4, _ = capture_walk(args.ckpt, stack, overlay=True, skip=None, dev=dev,
                              cap=cap_i4)

    cos = {}
    for i in range(NL):
        cs = [torch.nn.functional.cosine_similarity(a, b, dim=0).item()
              for a, b in zip(cap_bf[i], cap_i4[i])]
        cos[f"layer{i}"] = {"n": len(cs),
                            "mean": round(sum(cs) / len(cs), 5),
                            "p10": round(sorted(cs)[max(0, len(cs) // 10)], 5)}

    abl = {}
    for i in range(NL):
        w, _ = capture_walk(args.ckpt, stack, overlay=True, skip=i, dev=dev)
        abl[f"layer{i}"] = {"walk_loo_bf16": w,
                            "marginal_vs_full_int4": round(w - walk_i4, 4)}

    prov = json.load(open(args.provenance))
    werr = {}
    for i in range(NL):
        vals = [v for k, v in prov["per_matrix"].items()
                if k.startswith(f"layers.{i}.")]
        werr[f"layer{i}"] = round(sum(vals) / len(vals), 4)

    report = {
        "masters": args.ckpt,
        "references": {"walk_bf16_stack": walk_bf16, "walk_full_int4": walk_i4,
                       "total_walk_damage": round(walk_i4 - walk_bf16, 4)},
        "ledger1_weight_rel_err_per_layer": werr,
        "ledger2_hidden_cosine_per_layer": cos,
        "ledger3_leave_one_layer_bf16": abl,
        "sum_of_marginals": round(sum(v["marginal_vs_full_int4"]
                                      for v in abl.values()), 4),
        "note": "report-only (CONTRACT-B2B.md 8.5); marginals measured with "
                "b1-1@2000 masters where the walk signal has range",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
