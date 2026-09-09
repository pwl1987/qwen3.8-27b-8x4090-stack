#!/usr/bin/env python3
"""B2-B tier-3 formal statistics: FINAL-60 preregistered paired A/B.

FINAL-60 (CONTRACT-B2B.md §2, frozen before any run):
  make_prompt(300..714, step 7)  = 60 prompts, disjoint from every training
  corpus.  Phase A60 = S0 baseline-bf16 drafter boot, phase B60 = QAT-W4
  boot; per-prompt tok/step deltas paired.  Flags identical (b0 template),
  single variable = drafter weights.

Report fields mandated by the B2-A adjudication (R demoted to auxiliary):
  absolute mean Δ / relative % / median Δ / bootstrap 95% CI / positive
  fraction; plus N bookkeeping and Wilcoxon (normal approx).

Usage:
  python final60.py --phase A60|B60 [--label X]   # against a live boot
  python final60.py --phase stats                 # paired test + report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, "/data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b1")
sys.path.insert(0, "/data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b2b")
from accept_ab import run_prompt, semantic_ok, t3_full  # noqa: E402

CACHE = "/tmp/b2b/accept60"
os.makedirs(CACHE, exist_ok=True)
FINAL60 = list(range(300, 715, 7))          # 60 prompts, preregistered
assert len(FINAL60) == 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["A60", "B60", "stats"])
    ap.add_argument("--port", type=int, default=19627)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    label = args.label or args.phase

    if args.phase in ("A60", "B60"):
        t3a = t3_full(base, label)
        t3b = t3_full(base, label)
        contract = {"self_deterministic": t3a["sha12"] == t3b["sha12"],
                    "sha12": t3a["sha12"], "chars": len(t3a["text"]),
                    "semantic": t3a["semantic"],
                    "t3_tok_per_step": t3a["tok_per_step"],
                    "text": t3a["text"]}
        with open(f"{CACHE}/{label}-contract.json", "w") as f:
            json.dump(contract, f, indent=1, ensure_ascii=False)
        print(f"[{label}] t3 sha={contract['sha12']} "
              f"det={contract['self_deterministic']} sem={t3a['semantic']}",
              flush=True)
        results = [{"target_tokens": 512, "tag": label,
                    "tok_per_step": t3a["tok_per_step"], "sha12": t3a["sha12"]}]
        for tt in FINAL60:
            try:
                r = run_prompt(base, tt, label)
                results.append(r)
                print(f"[{label}] {tt} tok/step={r['tok_per_step']:.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                results.append({"target_tokens": tt, "tag": label,
                                "error": repr(e)[:200], "tok_per_step": None})
                print(f"[{label}] {tt} ERROR {e!r}", flush=True)
        with open(f"{CACHE}/{label}.json", "w") as f:
            json.dump(results, f, indent=1)

    if args.phase == "stats":
        import torch
        A = {r["target_tokens"]: r for r in json.load(open(f"{CACHE}/A60.json"))}
        B = {r["target_tokens"]: r for r in json.load(open(f"{CACHE}/B60.json"))}
        ca = json.load(open(f"{CACHE}/A60-contract.json"))
        cb = json.load(open(f"{CACHE}/B60-contract.json"))
        with open(f"{CACHE}/A60-contract.json") as f:
            ta = json.load(f)["text"]
        with open(f"{CACHE}/B60-contract.json") as f:
            tb = json.load(f)["text"]
        la, lb = len(ta), len(tb)
        cp = next((i for i in range(min(la, lb)) if ta[i] != tb[i]), min(la, lb))
        comparative = ("EXACT" if ta == tb else
                       "BENIGN-DIFF" if (cp >= 20 and 0.5 <= la / max(1, lb) <= 2.0)
                       else "UNRESOLVED")
        contract = {
            "A_self_deterministic": ca["self_deterministic"],
            "B_self_deterministic": cb["self_deterministic"],
            "A_semantic": ca["semantic"], "B_semantic": cb["semantic"],
            "C_comparative": comparative,
            "common_prefix": cp,
            "A_sha12": ca["sha12"], "B_sha12": cb["sha12"],
            "target_correctness": ("PASS" if ca["self_deterministic"]
                                   and cb["self_deterministic"]
                                   and ca["semantic"]["ok"] and cb["semantic"]["ok"]
                                   else "FAIL"),
        }
        pairs, n_total = [], 0
        n_valid = n_pos = n_neg = n_tie = 0
        for tt in FINAL60:
            n_total += 1
            a, b = A[tt].get("tok_per_step"), B[tt].get("tok_per_step")
            if a is None or b is None:
                continue
            n_valid += 1
            d = b - a
            pairs.append(d)
            n_pos += d > 0
            n_neg += d < 0
            n_tie += d == 0
        t = torch.tensor(pairs, dtype=torch.float64)
        mean = float(t.mean()) if n_valid else 0.0
        mean_a = float(torch.tensor(
            [A[tt]["tok_per_step"] for tt in FINAL60
             if A[tt].get("tok_per_step")]).mean())
        nz = t[t != 0]
        if n_valid >= 10 and len(nz) >= 5:
            ranks = nz.abs().argsort().argsort().to(torch.float64) + 1
            w_plus = float(ranks[nz > 0].sum())
            n_nz = len(nz)
            mu = n_nz * (n_nz + 1) / 4
            sigma = (n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24) ** 0.5
            z = (w_plus - mu) / sigma
            p = math.erfc(abs(z) / math.sqrt(2))
        else:
            z, p = 0.0, 1.0
        g = torch.Generator().manual_seed(7)
        boots = sorted(float(t[torch.randint(0, n_valid, (n_valid,),
                                              generator=g)].mean())
                       for _ in range(10000))
        ci_lo, ci_hi = boots[249], boots[9749]
        verdict = ("PASS" if (n_valid >= 60 and p < 0.05 and ci_lo > 0 and mean > 0)
                   else "INCONCLUSIVE" if n_valid < 60 else "FAIL")
        s1_gain = 3.4943 - 3.4067          # engine S1-S0 (B2-A ladder, 16-FINAL)
        report = {
            "b11_contract": contract,
            "N_total": n_total, "N_valid": n_valid, "N_positive": n_pos,
            "N_negative": n_neg, "N_tie": n_tie,
            "five_field": {
                "abs_mean_delta_tok_per_step": round(mean, 4),
                "rel_percent": round(100 * mean / mean_a, 2),
                "median_delta": round(float(t.median()) if n_valid else 0, 4),
                "bootstrap95_CI": [round(ci_lo, 4), round(ci_hi, 4)],
                "positive_fraction": round(n_pos / max(1, n_valid), 4),
            },
            "mean_A60": round(mean_a, 4),
            "mean_B60": round(mean_a + mean, 4),
            "wilcoxon_p_normalapprox": round(p, 5),
            "z": round(float(z), 3),
            "R_auxiliary_vs_B1_gain": round(mean / s1_gain, 2),
            "engineering_ge_3p5": (mean_a + mean) >= 3.5,
            "verdict": verdict,
        }
        with open(f"{CACHE}/stats.json", "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
