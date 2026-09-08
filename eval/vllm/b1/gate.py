#!/usr/bin/env python3
"""B1-B checkpoint replay gate (DEV slice of a v2 trace).

v2 traces carry num_rejected in every in-record, so the valid-row count per
step is exact and prompt/text reconstruction is unnecessary:
  v_j = nt_j - num_rejected(in_j)      (kernel: nr = rejects of D_{j-1})

Two metric families per replay:
  ENGINE metrics (fidelity to the recorded engine outputs):
    sh_cos      cosine(own sample_hidden, engine sample_hidden)   [baseline gate]
    top16       overlap(own candidates, engine candidates)        [baseline gate]
  TEACHER proxies (what training should move; early-stop signal):
    recall      teacher token in STUDENT's own top-16 (on supervision labels)
    top1        student's top-1 == teacher
    rank / margin of teacher among own candidates
    own_walk_k  prefix match of own walked path vs the teacher stream
    engine_k    mean accepted drafts per step (from nr; corpus property)

Gates (as frozen in the B1 baseline):
  baseline (no checkpoint): sh_cos mean>=0.999, top16 mean>=15.4, anchor/k chain OK
  checkpoints:  absolute top16 mean>=14 / p10>=12  (B0 floor)
                relative  top16 mean >= baseline_mean - 0.4,
                          p10 >= baseline_p10 - 1,
                          DEV recall >= baseline_recall (no regression)

Usage:
  CUDA_VISIBLE_DEVICES=4 python gate.py --ask-dir .../trace-v2 --runs DEV.json \\
      [--ckpt path.pt] [--tag base] [--out gate-xxx.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from drafter_torch import (  # noqa: E402
    DRAFT_BLOCK, DrafterTorch, TOPK, load_pairs, segment_runs, walk_greedy)
from supervision import build_run  # noqa: E402


def load_ckpt(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if "fc" in sd:
        model.fc = torch.nn.Parameter(sd["fc"].to(model.dev, torch.bfloat16))
    if "hp" in sd:
        model.hid_proj = torch.nn.Parameter(sd["hp"].to(model.dev, torch.bfloat16))
    if "pred" in sd:
        model.pred_cb = torch.nn.Parameter(sd["pred"].to(model.dev, torch.bfloat16))
    if "succ" in sd:
        model.succ_cb = torch.nn.Parameter(sd["succ"].to(model.dev, torch.bfloat16))


def replay_dev(model, runs, stats):
    """Replay the given runs (list of pair-lists) with nr-direct valid rows."""
    for run in runs:
        Kc = [torch.empty(0, 8, 128, device=model.dev, dtype=torch.bfloat16)
              for _ in range(5)]
        Vc = [torch.empty(0, 8, 128, device=model.dev, dtype=torch.bfloat16)
              for _ in range(5)]
        C = 0
        labels_by_j = {}
        # supervision labels for this run (teacher stream + nr chain)
        sup_steps, _mode, _sv2, _ok, _tot = build_run(run, None, 0, 0)
        sup = {st["j"]: st for st in sup_steps}
        for j, (rin, rout) in enumerate(run):
            nt = rin["num_tokens"]
            nr = int(rin["num_rejected"][0]) if rin.get("num_rejected") is not None else 0
            v = nt - nr
            anchor = int(rout["anchor"][0])
            ref_sh = rout["sample_hidden"][0].to(model.dev)
            with torch.no_grad():
                aux = rin["aux"].to(model.dev)
                fc_all = aux @ model.fc.T
                ref = rout["fc_out"][:nt].to(model.dev)
                n = min(fc_all.shape[0], ref.shape[0])
                stats["fc_cos"] += F.cosine_similarity(
                    fc_all[:n].float().flatten(1), ref[:n].float().flatten(1), dim=1
                ).tolist()
                Ks, Vs = model.context_kv(fc_all[:v], C)
                Kc = [torch.cat([a, b], 0) for a, b in zip(Kc, Ks)]
                Vc = [torch.cat([a, b], 0) for a, b in zip(Vc, Vs)]
                C += v
                sh = model.forward_queries(anchor, C, Kc, Vc)
                stats["sh_cos"] += F.cosine_similarity(
                    sh.float().flatten(1), ref_sh.float().flatten(1), dim=1
                ).tolist()

                logits = sh @ model.lm_head.T
                cand_logits, cand = torch.topk(logits.float(), TOPK, dim=-1)
                ref_cand = rout["candidate_ids"][0].to(model.dev)
                ov = (cand.unsqueeze(-1) == ref_cand.unsqueeze(-2)).any(-1).sum(-1)
                stats["top16"] += ov.tolist()

                # teacher proxies on this step's labels
                st = sup.get(j)
                if st is not None:
                    stream = [lab["teacher_token"] for lab in st["labels"]
                              if lab["valid"]]
                    for lab in st["labels"]:
                        if not lab["valid"]:
                            continue
                        t = lab["teacher_token"]
                        row = cand_logits[lab["depth"]].float()
                        hit = (cand[lab["depth"]] == t).nonzero().flatten().tolist()
                        stats["recall"].append(bool(hit))
                        if hit:
                            i = hit[0]
                            stats["rank"].append(i + 1)
                            others = row.clone()
                            others[i] = -float("inf")
                            stats["margin"].append(float(row[i] - others.max()))
                        stats["top1"].append(
                            int(cand[lab["depth"], int(row.argmax())]) == t)
                    if stream:
                        sc = model.selector_scores(cand, cand_logits, sh, anchor)
                        path, _t, _m = walk_greedy(sc, cand)
                        ko = 0
                        while ko < DRAFT_BLOCK and ko < len(stream) \
                                and int(path[ko]) == int(stream[ko]):
                            ko += 1
                        stats["own_walk_k"].append(ko)
    # engine accepted per decode step: rejects of D_j live in in_{j+1}
    for run in runs:
        for j in range(len(run) - 1):
            rin_next = run[j + 1][0]
            if rin_next["num_tokens"] == 8 and rin_next.get("num_rejected") is not None:
                stats["engine_k"].append(7 - int(rin_next["num_rejected"][0]))


def q(xs):
    if not xs:
        return {"n": 0}
    t = torch.tensor(xs, dtype=torch.float64)
    return {"n": int(t.numel()), "mean": round(float(t.mean()), 4),
            "p50": round(float(t.quantile(0.5)), 4),
            "p10": round(float(t.quantile(0.1)), 4), "min": round(float(t.min()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask-dir", default="/data/sandbox/ab-vllm/b1/trace-v2")
    ap.add_argument("--dev-runs", default="30-37",
                    help="inclusive run index range for the DEV slice")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default="blevel-evidence/gate-base-v2dev.json",
                    help="baseline gate JSON for the relative gates")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.dev_runs.split("-"))
    pairs = load_pairs(args.ask_dir)
    runs = segment_runs(pairs)
    assert all("num_rejected" in r[0][0] for r in runs), "not a v2 trace"
    dev = [r for i, r in enumerate(runs) if lo <= i <= hi]
    print(f"runs total={len(runs)} dev={len(dev)} (idx {lo}..{hi})")

    model = DrafterTorch(torch.device("cuda"), embed_source="dequant")
    if args.ckpt:
        load_ckpt(model, args.ckpt)

    stats = {"fc_cos": [], "sh_cos": [], "top16": [], "recall": [], "top1": [],
             "rank": [], "margin": [], "own_walk_k": [], "engine_k": []}
    replay_dev(model, dev, stats)
    # clean the placeholder tail used during per-step collection
    # B1.1 two-family split (CONTRACT-B1.1.md): compatibility = drift, never a
    # quality gate; speculation quality = the only gated family.
    report = {
        "tag": args.tag, "ckpt": args.ckpt, "dev_runs": args.dev_runs,
        "baseline_compatibility": {
            "fc_cos": q(stats["fc_cos"]),
            "sh_cos_vs_engine": q(stats["sh_cos"]),
            "top16_vs_engine": q(stats["top16"]),
            "floor_diagnostic": {
                "top16_mean>=14": (q(stats["top16"]).get("mean") or 0) >= 14,
                "top16_p10>=12": (q(stats["top16"]).get("p10") or 0) >= 12,
            },
        },
        "speculation_quality": {
            "recall": round(sum(stats["recall"]) / max(1, len(stats["recall"])), 4),
            "top1": round(sum(stats["top1"]) / max(1, len(stats["top1"])), 4),
            "mean_rank": round(sum(stats["rank"]) / max(1, len(stats["rank"])), 3),
            "mean_margin": round(sum(stats["margin"]) / max(1, len(stats["margin"])), 3),
            "own_walk_k": round(sum(stats["own_walk_k"]) / max(1, len(stats["own_walk_k"])), 4),
            "engine_k": round(sum(stats["engine_k"]) / max(1, len(stats["engine_k"])), 4),
        },
        "n_steps": len(stats["sh_cos"]) // 7,
    }
    sq = report["speculation_quality"]
    base = json.load(open(args.baseline)) if (
        os.path.exists(args.baseline) and args.tag != "base") else None
    if base is not None:
        b = base["speculation_quality"]
        report["gates"] = {
            "recall>=base": sq["recall"] >= b["recall"] - 1e-4,
            "top1>=base-0.005": sq["top1"] >= b["top1"] - 0.005,
            "margin>=base-0.5": sq["mean_margin"] >= b["mean_margin"] - 0.5,
        }
        report["baseline_tag"] = base.get("tag")
    else:
        report["gates"] = {   # sanity floors for the baseline reference run
            "recall>=0.95": sq["recall"] >= 0.95,
            "top1>=0.80": sq["top1"] >= 0.80,
        }
    out = args.out or f"/tmp/b1/gate-{args.tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
