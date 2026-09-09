#!/usr/bin/env python3
"""B2-B deployment-aware QAT trainer (GPU4): the b1-B trainer with the frozen
5-layer stack swapped for the deployment-equivalent int4 dequantized weights
(frozen_stack.py / overlay.py).  Trace, seed, chunk stream, recipe, loss and
baseline initialization are byte-identical to B1-B — the ONLY variable vs the
b1-1 arm is the frozen stack (CONTRACT-B2B.md §1).

Arms:
  zero   B1-0 zero-update control under the int4 stack (bitwise Delta=0 assert)
  b2b-1  fc + selector       (mirrors b1-1)
  b2b-2  fc only             (pre-registered fallback arm)

Smoke mode: --eval-at 0,100,200,300 --max-steps 300 (G1/G2 of the contract);
formal: --eval-at 500,1000,1500,2000 (MAX/PATIENCE/CKPT_EVERY as in B1).

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 \
    python train_qat.py --arm b2b-1 --max-steps 300 --eval-at 0,100,200,300 \
      --save-dir /data/sandbox/ab-vllm/b2b/smoke-b2b-1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b1")
sys.path.insert(0, "/data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b2b")
from drafter_torch import DrafterTorch  # noqa: E402
from run_arms import (  # noqa: E402
    dev_proxies, load_split, make_chunks, train_chunk)
from train import BETAS, LR, SEED, WD  # noqa: E402
from overlay import apply_stack  # noqa: E402


def g2_verdict(hist: list[dict]) -> dict:
    """CONTRACT-B2B.md §3 G2: adjudicated at 200, one re-check at 300."""
    by = {h["step"]: h["dev"] for h in hist}
    out = {}
    for t in (200, 300):
        if t not in by or 0 not in by:
            continue
        d0, dt = by[0], by[t]
        loss_ok = dt["loss"] < 0.9 * d0["loss"]
        move = (dt["recall"] >= d0["recall"] + 0.002
                or dt["top1"] >= d0["top1"] + 0.001
                or dt["margin"] >= d0["margin"] + 0.05)
        no_reg = (dt["recall"] >= d0["recall"] - 0.002
                  and dt["top1"] >= d0["top1"] - 0.001
                  and dt["margin"] >= d0["margin"] - 0.05)
        out[f"@{t}"] = {"loss<0.9x": loss_ok, "proxy_move": move,
                        "no_regression": no_reg, "PASS": loss_ok and move and no_reg}
    out["SMOKE_PASS"] = any(v["PASS"] for v in out.values() if isinstance(v, dict))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["zero", "b2b-1", "b2b-2"])
    ap.add_argument("--ask-dir", default="/data/sandbox/ab-vllm/b1/trace-v2")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--frozen-stack",
                    default="/data/sandbox/ab-vllm/b2b/frozen-int4-stack.pt")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--eval-at", default="500,1000,1500,2000",
                    help="optimizer steps for DEV proxy evals (0 always added)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    dev_gpu = torch.device("cuda")
    model = DrafterTorch(dev_gpu, embed_source="dequant")
    apply_stack(model, args.frozen_stack)          # the deployment function
    lm32 = model.lm_head.float()
    model.lm_head = None

    train_fc = args.arm in ("b2b-1", "b2b-2")
    train_sel = args.arm == "b2b-1"
    masters = {
        "fc": model.fc.detach().float().clone().requires_grad_(train_fc),
        "hp": model.hid_proj.detach().float().clone().requires_grad_(train_sel),
        "pred": model.pred_cb.detach().float().clone().requires_grad_(train_sel),
        "succ": model.succ_cb.detach().float().clone().requires_grad_(train_sel),
    }
    if args.arm == "zero":
        for v in masters.values():
            v.requires_grad_(False)
    base = {k: v.detach().clone() for k, v in masters.items()}
    dummy = torch.nn.Parameter(torch.zeros(1, device=dev_gpu))
    params = [v for v in masters.values() if v.requires_grad] + [dummy]
    opt = torch.optim.AdamW(params, lr=LR, betas=BETAS, weight_decay=WD)

    tr, dv = load_split(args.ask_dir)
    print(f"arm={args.arm} train_runs={len(tr)} dev_runs={len(dv)} "
          f"train_sup_steps={sum(len(s) for _, s in tr)}", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    chunks = make_chunks(tr, SEED)                 # same stream as B1-B
    eval_at = sorted({0, *(int(x) for x in args.eval_at.split(",") if x)})

    hist, t0, nan_any = [], time.time(), False

    def eval_dev(it):
        dev = dev_proxies(model, masters, lm32, dv, SEED + 1)
        dl2 = {k: {"abs": round((masters[k] - base[k]).norm().item(), 5),
                   "rel": round(((masters[k] - base[k]).norm()
                                 / base[k].norm()).item(), 6)}
               for k in ("fc", "hp", "pred", "succ")}
        hist.append({"step": it, "dev": dev, "dL2": dl2,
                     "sec": round(time.time() - t0, 1)})
        print(f"EVAL {it}: dev={dev}", flush=True)
        torch.save({k: v.detach().cpu() for k, v in masters.items()},
                   f"{args.save_dir}/ckpt-{it:04d}.pt")

    if 0 in eval_at:
        eval_dev(0)                                # the G2 @0 reference point
    for it, (run, chunk) in enumerate(chunks[:args.max_steps], start=1):
        m = train_chunk(model, masters, lm32, run, chunk, opt=opt, backward=True)
        nan_any = nan_any or not (m["loss"] == m["loss"]) or abs(m["loss"]) == float("inf")
        if it % 50 == 0:
            print(f"  step {it} loss={m['loss']} top1={m['top1']} "
                  f"recall={m['recall']} {time.time()-t0:.0f}s", flush=True)
        if it in eval_at or it == args.max_steps:
            eval_dev(it)

    g1 = {"no_nan": not nan_any,
          "fc_moved": (masters["fc"] - base["fc"]).norm().item() > 0,
          "zero_assert": None}
    if args.arm == "zero":
        g1["zero_assert"] = all(torch.equal(base[k], masters[k]) for k in masters)
        assert g1["zero_assert"], "zero arm mutated frozen weights!"

    report = {"arm": args.arm, "frozen_stack": args.frozen_stack,
              "max_steps": args.max_steps, "g1": g1,
              "g2": g2_verdict(hist) if args.arm != "zero" else None,
              "history": hist}
    with open(f"{args.save_dir}/history.json", "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "history"},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
