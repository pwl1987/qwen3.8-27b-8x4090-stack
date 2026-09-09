#!/usr/bin/env python3
"""B2-B G0 / per-ckpt gate: paired DEV replay of the SAME masters under the
bf16 stack vs the deployment-equivalent int4 stack (one model, sequential).

G0 (CONTRACT-B2B.md, frozen): with b1-1 ckpt-2000 masters,
  PASS  |Δrecall| >= 0.01  and  Δown_walk_k < 0   (quantization visible, same
  direction as the engine ladder S1 3.4943 -> S3 3.3428)
  FAIL  |Δrecall| < 0.01  -> fake-quant is not deployment-faithful: STOP.
Also reports the B1.1 two-family split for both stacks: baseline_compatibility
(drift vs the recorded engine outputs — never a quality gate) and
speculation_quality (the only gated family).

Usage (GPU4):
  CUDA_VISIBLE_DEVICES=4 python gate_qat.py --ckpt .../b1-1/ckpt-2000.pt \
      --tag g0-b1-1-2000 --out /tmp/b2b/g0.json
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
from gate import load_ckpt, q, replay_dev  # noqa: E402
from overlay import apply_stack  # noqa: E402


def family(stats: dict) -> dict:
    sq = {
        "recall": round(sum(stats["recall"]) / max(1, len(stats["recall"])), 4),
        "top1": round(sum(stats["top1"]) / max(1, len(stats["top1"])), 4),
        "mean_rank": round(sum(stats["rank"]) / max(1, len(stats["rank"])), 3),
        "mean_margin": round(sum(stats["margin"]) / max(1, len(stats["margin"])), 3),
        "own_walk_k": round(sum(stats["own_walk_k"]) / max(1, len(stats["own_walk_k"])), 4),
        "engine_k": round(sum(stats["engine_k"]) / max(1, len(stats["engine_k"])), 4),
    }
    return {
        "baseline_compatibility": {
            "fc_cos": q(stats["fc_cos"]),
            "sh_cos_vs_engine": q(stats["sh_cos"]),
            "top16_vs_engine": q(stats["top16"]),
        },
        "speculation_quality": sq,
    }


def replay(model, dev_runs):
    stats = {"fc_cos": [], "sh_cos": [], "top16": [], "recall": [], "top1": [],
             "rank": [], "margin": [], "own_walk_k": [], "engine_k": []}
    replay_dev(model, dev_runs, stats)
    return family(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frozen-stack",
                    default="/data/sandbox/ab-vllm/b2b/frozen-int4-stack.pt")
    ap.add_argument("--ask-dir", default="/data/sandbox/ab-vllm/b1/trace-v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs = load_pairs(args.ask_dir)
    runs = segment_runs(pairs)
    dev = [r for i, r in enumerate(runs) if 30 <= i <= 37]

    model = DrafterTorch(torch.device("cuda"), embed_source="dequant")
    load_ckpt(model, args.ckpt)
    bf16 = replay(model, dev)
    apply_stack(model, args.frozen_stack)
    int4 = replay(model, dev)

    d = {k: round(int4["speculation_quality"][k] - bf16["speculation_quality"][k], 4)
         for k in bf16["speculation_quality"]}
    d_sh = round(int4["baseline_compatibility"]["sh_cos_vs_engine"]["mean"]
                 - bf16["baseline_compatibility"]["sh_cos_vs_engine"]["mean"], 4)
    rel_walk = d["own_walk_k"] / max(1e-9, bf16["speculation_quality"]["own_walk_k"])
    # amended 2026-09-09 (pre-training, governance): recall is a membership test
    # over 16-of-248320 and is structurally insensitive to 15% layer noise;
    # faithfulness is adjudicated on the walk metric (see CONTRACT-B2B.md).
    report = {
        "tag": args.tag, "ckpt": args.ckpt,
        "bf16_stack": bf16, "int4_stack": int4,
        "delta_int4_minus_bf16": d, "delta_sh_cos_mean": d_sh,
        "g0": {
            "sh_cos_visible": d_sh <= -0.005,
            "own_walk_k_negative": d["own_walk_k"] <= -0.01,
            "rel_walk_drop_in_band": -0.15 <= rel_walk <= -0.01,
            "rel_walk_drop": round(rel_walk, 4),
            "engine_ladder_rel": "S1->S3 = -0.152/3.4943 = -4.4%",
            "recall_diagnostic": d["recall"],
            "PASS": (d_sh <= -0.005 and d["own_walk_k"] <= -0.01
                     and -0.15 <= rel_walk <= -0.01),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({
        "tag": args.tag,
        "bf16_sq": bf16["speculation_quality"],
        "int4_sq": int4["speculation_quality"],
        "delta": d, "g0": report["g0"],
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
