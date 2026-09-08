#!/usr/bin/env python3
"""B1-B sequential four-arm trainer on the v2 trace (GPU4).

Arms (frozen): zero -> b1-1 -> b1-2 -> b1-3, at most 2000 optimizer steps
each, checkpoint every 500, early stop after 2 consecutive checkpoints with
no DEV-proxy improvement.  Early-stop proxy = TRAINING-TIME signals only
(governance lock 6): improved := recall up OR top1 up OR margin up; never
acceptance/FINAL.

Data: TRAIN = v2 runs 0-29 + 38-45, DEV = runs 30-37 (three-tier isolation).
Context rebuild per chunk with CURRENT fc (fp32 masters -> bf16 at the frozen
stack boundary).  Valid rows per step read directly from the in-record's
num_rejected (v2), no text/oracle reconstruction.

Per checkpoint: save masters, record dL2 (fc / selector), DEV proxies; the
engine-metric replay gate runs separately via gate.py (see run_arms.sh).

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=4 python run_arms.py --arm b1-1 \
      --save-dir /data/sandbox/ab-vllm/b1/ckpts/b1-1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from drafter_torch import DrafterTorch, load_pairs, segment_runs  # noqa: E402
from supervision import build_run  # noqa: E402
from train import (  # noqa: E402
    CHUNK, LR, BETAS, WD, SEED, fc_forward, selector_scores_fp32, step_losses)

MAX_STEPS = 2000
CKPT_EVERY = 500
PATIENCE = 2
TRAIN_RUNS = list(range(0, 30)) + list(range(38, 46))
DEV_RUNS = list(range(30, 38))


def load_split(ask_dir):
    pairs = load_pairs(ask_dir)
    runs = segment_runs(pairs)
    assert len(runs) == 46, f"expected 46 runs, got {len(runs)}"
    tr, dv = [], []
    for i, run in enumerate(runs):
        steps, _m, _v2, ok, tot = build_run(run, None, 0, 0)
        (tr if i in TRAIN_RUNS else dv).append((run, steps))
    return tr, dv


def run_rows(run):
    """All valid aux rows of a run in order (v2: v = nt - num_rejected)."""
    rows = []
    for rin, _rout in run:
        nt = rin["num_tokens"]
        nr = int(rin["num_rejected"][0])
        rows.append(rin["aux"][:nt - nr])
    return torch.cat(rows, 0)


def make_chunks(run_steps, seed):
    g = torch.Generator().manual_seed(seed)
    chunks = []
    for _epoch in range(100):                     # far more than needed
        idx = torch.randperm(len(run_steps), generator=g).tolist()
        for i in idx:
            run, steps = run_steps[i]
            if len(steps) < 2:
                continue
            m = len(steps)
            s0 = int(torch.randint(0, m, (1,), generator=g))
            chunks.append((run, [steps[(s0 + t) % m] for t in range(CHUNK)]))
        if len(chunks) >= MAX_STEPS:
            break
    return chunks[:MAX_STEPS]


def train_chunk(model, masters, lm32, run, chunk, opt=None, backward=True):
    by_j = {j: pr for j, pr in enumerate(run)}
    all_aux = run_rows(run).to(model.dev)
    first = chunk[0]
    target_C = first["ctx_len_queries"] - first["num_valid_rows"]
    ctx_fc = fc_forward(all_aux[:target_C].float(), masters["fc"]).to(torch.bfloat16)
    Kc, Vc = model.context_kv(ctx_fc, 0)
    Kc, Vc = list(Kc), list(Vc)
    C = target_C
    total = 0.0
    agg = {"top1": [], "recall": [], "rank": [], "margin": []}
    for st in chunk:
        rin = by_j[st["j"]][0]
        v = st["num_valid_rows"]
        rows32 = fc_forward(rin["aux"][:v].to(model.dev).float(),
                            masters["fc"]).to(torch.bfloat16)
        Ks, Vs = model.context_kv(rows32, C)
        Kc = [torch.cat([a, b], 0) for a, b in zip(Kc, Ks)]
        Vc = [torch.cat([a, b], 0) for a, b in zip(Vc, Vs)]
        C += v
        sh = model.forward_queries(st["anchor"], C, Kc, Vc)
        loss, stats = step_losses(lm32, sh, st["labels"], st["anchor"], masters)
        total = total + loss / len(chunk)
        agg["top1"].extend(stats["top1"])
        agg["recall"].extend(stats["recall"])
        agg["rank"].extend(stats["rank"])
        agg["margin"].extend(stats["margin"])
    if backward and opt is not None:
        opt.zero_grad(set_to_none=True)
        (total + next(iter(opt.param_groups))["params"][-1] * 0.0).backward()
        opt.step()
    m = {
        "loss": round(float(total.detach()), 4),
        "top1": round(sum(agg["top1"]) / max(1, len(agg["top1"])), 4),
        "recall": round(sum(agg["recall"]) / max(1, len(agg["recall"])), 4),
        "margin": round(sum(agg["margin"]) / max(1, len(agg["margin"])), 3),
    }
    return m


def dev_proxies(model, masters, lm32, dev_steps, seed):
    """Fixed DEV chunk set — the early-stop / regression proxy (no engine)."""
    with torch.no_grad():
        chunks = make_chunks(dev_steps, seed)
        chunks = chunks[:24]                      # fixed, deterministic
        sums = {"top1": [], "recall": [], "margin": [], "loss": []}
        for run, chunk in chunks:
            m = train_chunk(model, masters, lm32, run, chunk, opt=None, backward=False)
            sums["loss"].append(m["loss"])
            sums["top1"].append(m["top1"])
            sums["recall"].append(m["recall"])
            sums["margin"].append(m["margin"])
    return {k: round(sum(v) / len(v), 4) for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["zero", "b1-1", "b1-2", "b1-3"])
    ap.add_argument("--ask-dir", default="/data/sandbox/ab-vllm/b1/trace-v2")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    dev_gpu = torch.device("cuda")
    model = DrafterTorch(dev_gpu, embed_source="dequant")
    lm32 = model.lm_head.float()
    model.lm_head = None

    train_fc = args.arm in ("b1-1", "b1-2")
    train_sel = args.arm in ("b1-1", "b1-3")
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
    # monkey-patch: train_chunk references the dummy via param_groups for zero
    # arm graph anchoring — simpler: rebind a module-level anchor
    global _DUMMY
    _DUMMY = dummy

    chunks = make_chunks(tr, SEED)
    dev0 = dev_proxies(model, masters, lm32, dv, SEED + 1)
    print("DEV baseline proxies:", dev0, flush=True)
    best = {"recall": -1, "top1": -1, "margin": -1, "ckpt": 0}
    patience = 0
    hist = []
    t0 = time.time()
    for it, (run, chunk) in enumerate(chunks[:args.max_steps], start=1):
        train_chunk(model, masters, lm32, run, chunk, opt=opt, backward=True)
        if it % 50 == 0:
            print(f"  step {it} {time.time()-t0:.0f}s", flush=True)
        if it % CKPT_EVERY == 0 or it == len(chunks[:args.max_steps]):
            dev = dev_proxies(model, masters, lm32, dv, SEED + 1)
            dl2 = {}
            for k in ("fc", "hp", "pred", "succ"):
                d = (masters[k] - base[k]).norm().item()
                dl2[k] = {"abs": round(d, 5),
                          "rel": round(d / base[k].norm().item(), 6)}
            improved = (dev["recall"] > best["recall"] + 1e-4
                        or dev["top1"] > best["top1"] + 1e-3
                        or dev["margin"] > best["margin"] + 0.1)
            ck = {
                "step": it, "dev": dev, "dL2": dl2, "improved": improved,
                "sec": round(time.time() - t0, 1),
            }
            hist.append(ck)
            print(f"CKPT {it}: dev={dev} improved={improved}", flush=True)
            torch.save({k: v.detach().cpu() for k, v in masters.items()},
                       f"{args.save_dir}/ckpt-{it:04d}.pt")
            if improved:
                best = {"recall": dev["recall"], "top1": dev["top1"],
                        "margin": dev["margin"], "ckpt": it}
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE and it >= 1000:
                    print(f"EARLY STOP at {it} (patience {PATIENCE})", flush=True)
                    break
    if args.arm == "zero":
        assert all(torch.equal(base[k], masters[k]) for k in masters), \
            "zero arm mutated frozen weights!"
        print("zero-update assert: OK", flush=True)
    with open(f"{args.save_dir}/history.json", "w") as f:
        json.dump({"arm": args.arm, "dev_baseline": dev0, "best": best,
                   "history": hist}, f, indent=1)
    print("BEST:", best, flush=True)


if __name__ == "__main__":
    main()
