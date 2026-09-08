#!/usr/bin/env python3
"""B1-A2 differentiable trainer (speculation-aware three-component loss).

Implements eval/vllm/b1/loss_contract.md exactly (frozen 2026-09-08):
  L_total = 1.0*L_tok + 1.0*L_rank + 1.0*L_sel
Trainable params are held as fp32 masters (bf16 + lr 1e-5 loses updates to
ulp); the frozen 5-layer stack / embed / lm_head stay bf16 (engine-faithful,
gradients flow through them).

Arms (single variable = trainable set; 5 layers always frozen):
  zero  B1-0  nothing trainable — full loop incl. optimizer.step, then assert
              bitwise Delta(fc)=0 / Delta(selector)=0 (zero-update control)
  b1-1  main  fc + selector(hidden_projection + both codebooks)
  b1-2        fc only
  b1-3        selector only

Smoke (>=50 optimizer steps) records loss components, grad norms per group,
NaN/Inf, GPU memory, step time, and the proxy metrics top1 agreement /
top16 recall / teacher rank / margin.

Usage (GPU4):
  CUDA_VISIBLE_DEVICES=4 /data/vllm/venv/bin/python train.py --arm zero --steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from drafter_torch import (  # noqa: E402
    DRAFT_BLOCK, DrafterTorch, TOPK, build_prompt_ids, load_pairs, segment_runs,
    token_stream)
from supervision import build_run  # noqa: E402

MARGIN = 1.0            # frozen (loss_contract.md)
LR, BETAS, WD = 1e-5, (0.9, 0.95), 0.0
CHUNK = 8               # steps per optimizer update
SEED = 42


def build_training_steps(ask_dir, text):
    pairs = load_pairs(ask_dir)
    runs = segment_runs(pairs)
    stream = token_stream(text).tolist()
    prompt_ids = build_prompt_ids()
    out = []
    for ridx, run in enumerate(runs):
        steps, mode, _v2, _ok, _tot = build_run(run, stream, len(prompt_ids), ridx)
        out.append((run, steps))
    return out


def chunk_iter(run_steps, n_chunks, seed):
    """Chunks of CHUNK contiguous supervision steps within one run."""
    g = torch.Generator().manual_seed(seed)
    run_list = [i for i, (_run, steps) in enumerate(run_steps) if len(steps) >= 2]
    order = []
    while len(order) < n_chunks:
        for i in run_list:
            order.append(i)
    for run_idx in order[:n_chunks]:
        run, steps = run_steps[run_idx]
        m = len(steps)
        # sample a start, keep contiguous CHUNK (may wrap to next epoch start)
        s0 = int(torch.randint(0, max(1, m), (1,), generator=g))
        idx = [(s0 + t) % m for t in range(CHUNK)]
        yield run, [steps[i] for i in idx]


def fc_forward(aux32, fc32):
    return aux32 @ fc32.T


def selector_scores_fp32(cand, unary, sh32, anchor_id, hp32, pred32, succ32):
    """fp32 re-implementation of _score_edges for training (masters fp32)."""
    cand = cand.long()
    hp = sh32 @ hp32.T                                        # [7,256]
    succ = succ32[cand]                                       # [7,K,256]
    pred_ids = torch.cat([
        torch.full((1, TOPK), anchor_id, device=cand.device, dtype=cand.dtype),
        cand[:-1]], 0)
    pred = pred32[pred_ids]
    sc = unary[:, :, None] + torch.einsum("lpr,lcr->lpc",
                                          pred * hp[:, None, :], succ)
    return torch.nan_to_num(sc, nan=-1e30, posinf=1e30, neginf=-1e30)


def step_losses(lm32, sh, labels, anchor, masters):
    """Three-component loss per supervision step. Returns (loss, stats)."""
    logits = sh.float() @ lm32.T                              # [7,V]
    logp = F.log_softmax(logits, dim=-1)

    tok_terms, rank_terms, sel_terms = [], [], []
    stats = {"top1": [], "recall": [], "rank": [], "margin": [], "miss": 0}
    cand_logits, cand = torch.topk(logits, TOPK, dim=-1)      # student top-16 (ids)

    sel_scores = selector_scores_fp32(
        cand, cand_logits.float(), sh.float(), anchor,
        masters["hp"], masters["pred"], masters["succ"])

    prev_chain_idx = 0            # teacher-forced predecessor row index
    for lab in labels:
        if not lab["valid"]:
            stats["miss"] += 0    # unobserved: not a miss, just masked
            prev_chain_idx = None
            continue
        t = lab["teacher_token"]
        tok_terms.append(-logp[lab["depth"], t])
        # ranking on student's own candidates
        hit = (cand[lab["depth"]] == t).nonzero().flatten().tolist()
        row_logits = cand_logits[lab["depth"]].float()
        top1 = int(cand[lab["depth"], int(row_logits.argmax())])
        stats["top1"].append(top1 == t)
        if hit:
            i = hit[0]
            stats["recall"].append(True)
            stats["rank"].append(i + 1)
            t_logit = row_logits[i]
            others = row_logits.clone()
            others[i] = -float("inf")
            omax = others.max()
            stats["margin"].append(float((t_logit - omax).detach()))
            ce = F.cross_entropy(row_logits.unsqueeze(0),
                                 torch.tensor([i], device=row_logits.device))
            rank_terms.append(ce + F.relu(MARGIN - (t_logit - omax)))
        else:
            stats["recall"].append(False)
            stats["miss"] += 1
        # selector chain CE
        if sel_scores is not None and hit and prev_chain_idx is not None:
            row = sel_scores[lab["depth"], prev_chain_idx]
            sel_terms.append(F.cross_entropy(
                row.unsqueeze(0),
                torch.tensor([i], device=row.device)))
            prev_chain_idx = i
        else:
            prev_chain_idx = None      # chain broken at this depth
    n = max(1, len(tok_terms))
    loss = (sum(tok_terms) / n + sum(rank_terms) / n + sum(sel_terms) / n)
    return loss, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="zero", choices=["zero", "b1-1", "b1-2", "b1-3"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--ask-dir", default="/data/sandbox/ab-vllm/b1/trace-b0-260")
    ap.add_argument("--text", default="/tmp/p0/t3probe-run0.txt")
    ap.add_argument("--out", default="blevel-evidence/smoke-zero.json")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")   # fp32 master matmuls via TF32
    dev = torch.device("cuda")
    model = DrafterTorch(dev, embed_source="dequant")

    # fp32 masters for the trainable sets (from the bf16 engine-faithful model)
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
    base = {k: v.detach().clone() for k, v in masters.items()}   # for Δassert
    params = [v for v in masters.values() if v.requires_grad]
    # dummy: keeps the autograd graph alive for the zero arm (grad is always
    # exactly 0, and AdamW makes no update on a zero gradient with wd=0)
    dummy = torch.nn.Parameter(torch.zeros(1, device=dev))
    opt = torch.optim.AdamW(params + [dummy], lr=LR, betas=BETAS, weight_decay=WD)

    run_steps = build_training_steps(args.ask_dir, args.text)
    print(f"arm={args.arm} runs={len(run_steps)} "
          f"sup_steps={sum(len(s) for _, s in run_steps)}")

    lm32 = model.lm_head.float()          # hoisted once (4.7 GiB; saved once)
    model.lm_head = None                  # training never touches the bf16 copy
    log = {"arm": args.arm, "steps": [], "torch": torch.__version__}
    t_last = time.time()
    from supervision import nr_from_records
    stream = token_stream(args.text).tolist()
    prompt_ids_len = len(build_prompt_ids())
    for it, (run, chunk) in enumerate(chunk_iter(run_steps, args.steps, SEED)):
        by_j = {j: pr for j, pr in enumerate(run)}
        first = chunk[0]
        # collect ALL run aux valid rows sequentially (prefill full; decode via
        # supervision nr chain from the run's own steps, recomputed here)
        nr, _mode = nr_from_records(run, stream, prompt_ids_len)
        rows = []
        prev_nr = None
        for j, (rin, rout) in enumerate(run):
            nt = rin["num_tokens"]
            if nt > 8:
                rows.append(rin["aux"])
                prev_nr = nr[j] if _mode == "v2" else (
                    nr[j] if (j + 1 < len(run)
                              and run[j + 1][0]["num_tokens"] == DRAFT_BLOCK + 1
                              and nr[j] is not None) else 0)
                continue
            v = nt if prev_nr is None else nt - prev_nr
            rows.append(rin["aux"][:v])
            prev_nr = nr[j]
        all_aux = torch.cat(rows, 0).to(dev)
        # context slice for the chunk's first step
        target_C = first["ctx_len_queries"] - first["num_valid_rows"]
        # forward: fp32 fc over the context, cast bf16 into the frozen stack
        opt.zero_grad(set_to_none=True)
        ctx_fc32 = fc_forward(all_aux[:target_C].float(), masters["fc"])
        ctx_fc = ctx_fc32.to(torch.bfloat16)
        Kc, Vc = model.context_kv(ctx_fc, 0)
        Kc, Vc = list(Kc), list(Vc)
        C = target_C
        chunk_loss = 0.0
        agg = {"top1": [], "recall": [], "rank": [], "margin": [], "miss": 0}
        for st in chunk:
            rin = by_j[st["j"]][0]
            v = st["num_valid_rows"]
            aux32 = rin["aux"][:v].to(dev).float()
            rows32 = fc_forward(aux32, masters["fc"]).to(torch.bfloat16)
            Ks, Vs = model.context_kv(rows32, C)
            Kc = [torch.cat([a, b], 0) for a, b in zip(Kc, Ks)]
            Vc = [torch.cat([a, b], 0) for a, b in zip(Vc, Vs)]
            C += v
            sh = model.forward_queries(st["anchor"], C, Kc, Vc)
            loss, stats = step_losses(lm32, sh, st["labels"], st["anchor"], masters)
            chunk_loss = chunk_loss + loss / len(chunk)
            for k in ("top1", "recall"):
                agg[k].extend(stats[k])
            agg["rank"].extend(stats["rank"])
            agg["margin"].extend(stats["margin"])
            agg["miss"] += stats["miss"]
        chunk_loss = chunk_loss + dummy * 0.0
        chunk_loss.backward()
        gn = {}
        for k, v in masters.items():
            if v.grad is not None:
                gn[f"grad_{k}"] = round(float(v.grad.norm()), 4)
            elif v.requires_grad:
                gn[f"grad_{k}"] = 0.0
        opt.step()
        log["steps"].append({
            "it": it, "loss": round(float(chunk_loss.detach()), 4),
            "grad": gn,
            "nan": bool(not torch.isfinite(chunk_loss.detach())),
            "top1": round(sum(agg["top1"]) / max(1, len(agg["top1"])), 4),
            "recall": round(sum(agg["recall"]) / max(1, len(agg["recall"])), 4),
            "mean_rank": round(sum(agg["rank"]) / max(1, len(agg["rank"])), 2),
            "mean_margin": round(sum(agg["margin"]) / max(1, len(agg["margin"])), 3),
            "miss": agg["miss"],
            "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "sec": round(time.time() - t_last, 2),
        })
        t_last = time.time()
        if it % 10 == 0:
            s = log["steps"][-1]
            print(f"it={it} loss={s['loss']} top1={s['top1']} "
                  f"recall={s['recall']} mem={s['mem_gb']}G {s['sec']}s")

    # arm-zero assertion: bitwise zero update on every master
    zero_ok = all(torch.equal(base[k], masters[k]) for k in masters)
    report = {
        "arm": args.arm, "n_steps": len(log["steps"]),
        "final_loss": log["steps"][-1]["loss"] if log["steps"] else None,
        "first_loss": log["steps"][0]["loss"] if log["steps"] else None,
        "zero_update_assert": zero_ok if args.arm == "zero" else None,
        "peak_mem_gb": max((s["mem_gb"] for s in log["steps"]), default=0),
        "mean_step_sec": round(sum(s["sec"] for s in log["steps"])
                               / max(1, len(log["steps"])), 2),
        "any_nan": any(s["nan"] for s in log["steps"]),
        "steps": log["steps"],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "steps"},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
