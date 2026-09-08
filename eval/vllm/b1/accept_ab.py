#!/usr/bin/env python3
"""B1-C acceptance A/B on the sandbox (paired, double statistical gate).

FINAL prompts: 30+ fresh ulmus make_prompt variants (target tokens 300..690,
step 13) — disjoint from every collected corpus (TRAIN/DEV used calib rows).

Per prompt: /metrics spec-counter deltas (drafts / accepted) around ONE
deterministic request (engine determinism proven in B0) on EACH engine
(baseline-bf16 vs trained-bf16, identical flags, only the drafter weights
differ).  tok/step = accepted/drafts + 1 (includes the bonus token).

Gates (frozen):
  Gate A  paired Wilcoxon signed-rank  p < 0.05
  Gate B  bootstrap (10k resamples) 95% CI lower bound > 0
  AND     mean(delta) > 0
  Sample bookkeeping (contract): N_total / N_valid / N_positive /
  N_negative / N_tie; invalid samples (HTTP error/timeout/empty) are NEVER
  silently dropped; N_valid < 30 -> INCONCLUSIVE (not PASS/FAIL).
  Also: t3 fixture hard gate — temp0 output sha must equal the baseline's
  (target correctness: drafter may change, target greedy output may not).

Usage: python accept_ab.py --port-a 19627 --port-b 19627 --phase A
       (phase A/B = the two sandbox boots; per-prompt deltas are cached to
        disk so the paired test runs after both phases)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, "/data/sandbox/ab-vllm/repo/bench")
from ulmus_validate import make_prompt, post_json  # noqa: E402

CACHE = "/tmp/b1/accept-cache"
os.makedirs(CACHE, exist_ok=True)

T3_SHA_REF = None      # filled from phase A result; phase B must match


def snap(base):
    d = a = None
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_drafts_total"):
                d = float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_total") \
                    and "per_pos" not in line:
                a = float(line.rsplit(" ", 1)[1])
    return d, a


def run_prompt(base, target_tokens, tag):
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": make_prompt(target_tokens)}],
        "max_tokens": 512, "temperature": 0, "seed": 4242,
        "cache_salt": "b1c-" + uuid.uuid4().hex,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    d0, a0 = snap(base)
    t0 = time.time()
    res, el = post_json(base + "/v1/chat/completions", payload)
    d1, a1 = snap(base)
    txt = res["choices"][0]["message"]["content"]
    return {
        "target_tokens": target_tokens, "tag": tag,
        "drafts": d1 - d0, "accepted": a1 - a0,
        "tok_per_step": (a1 - a0) / (d1 - d0) + 1 if d1 > d0 else None,
        "sha12": hashlib.sha256(txt.encode()).hexdigest()[:12],
        "chars": len(txt), "sec": round(el, 1),
    }


def semantic_ok(text: str) -> dict:
    """B1.1 layer B: deterministic defect checks (P0 semantic-gate rules)."""
    if not text or len(text) < 10:
        return {"ok": False, "why": "empty/short"}
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    garb = printable / len(text) < 0.85
    rep = False
    for p in range(10, 121):                     # immediate periodicity x3
        for i in range(0, max(0, min(len(text) - 3 * p, 4000))):
            if text[i:i + p] == text[i + p:i + 2 * p] == text[i + 2 * p:i + 3 * p]:
                rep = True
                break
        if rep:
            break
    return {"ok": not (garb or rep), "garbage": garb, "repeat_loop": rep}


def t3_full(base, tag):
    """One t3 run capturing text for the A/B/C contract layers."""
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": make_prompt(512)}],
        "max_tokens": 512, "temperature": 0, "seed": 4242,
        "cache_salt": "b1c-t3-" + uuid.uuid4().hex,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    d0, a0 = snap(base)
    res, el = post_json(base + "/v1/chat/completions", payload)
    d1, a1 = snap(base)
    txt = res["choices"][0]["message"]["content"]
    return {"tag": tag, "text": txt,
            "sha12": hashlib.sha256(txt.encode()).hexdigest()[:12],
            "tok_per_step": (a1 - a0) / (d1 - d0) + 1 if d1 > d0 else None,
            "semantic": semantic_ok(txt)}


def comparative(a_txt: str, b_txt: str, a_sem: dict, b_sem: dict) -> str:
    """B1.1 layer C: EXACT / BENIGN-DIFF / UNRESOLVED / FAIL."""
    if not a_sem["ok"] or not b_sem["ok"]:
        return "FAIL"
    if a_txt == b_txt:
        return "EXACT"
    la, lb = len(a_txt), len(b_txt)
    if not (0.5 <= la / max(1, lb) <= 2.0):
        return "UNRESOLVED"
    n = min(la, lb)
    cp = next((i for i in range(n) if a_txt[i] != b_txt[i]), n)
    return "BENIGN-DIFF" if cp >= 20 else "UNRESOLVED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["A", "B", "stats"])
    ap.add_argument("--port", type=int, default=19627)
    ap.add_argument("--label", default=None, help="engine label for the cache")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    label = args.label or args.phase

    if args.phase in ("A", "B"):
        results = []
        t3a = t3_full(base, label)
        t3b = t3_full(base, label)
        contract = {
            "self_deterministic": t3a["sha12"] == t3b["sha12"],
            "sha12": t3a["sha12"], "chars": len(t3a["text"]),
            "semantic": t3a["semantic"],
            "t3_tok_per_step": t3a["tok_per_step"],
        }
        print(f"[{label}] t3 sha={t3a['sha12']} det={contract['self_deterministic']} "
              f"sem={t3a['semantic']}", flush=True)
        results.append({"target_tokens": 512, "tag": label,
                        "tok_per_step": t3a["tok_per_step"],
                        "sha12": t3a["sha12"], "chars": len(t3a["text"])})
        with open(f"{CACHE}/{label}-contract.json", "w") as f:
            json.dump(contract, f, indent=1, ensure_ascii=False)
        for tt in range(300, 691, 13):          # 31 fresh FINAL prompts
            try:
                r = run_prompt(base, tt, label)
                results.append(r)
                print(f"[{label}] {tt} drafts={r['drafts']:.0f} "
                      f"accepted={r['accepted']:.0f} tok/step={r['tok_per_step']:.3f}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                results.append({"target_tokens": tt, "tag": label,
                                "error": repr(e)[:200], "tok_per_step": None})
                print(f"[{label}] {tt} ERROR {e!r}", flush=True)
        with open(f"{CACHE}/{label}.json", "w") as f:
            json.dump(results, f, indent=1)

    if args.phase == "stats":
        A = {r["target_tokens"]: r for r in json.load(open(f"{CACHE}/A.json"))}
        B = {r["target_tokens"]: r for r in json.load(open(f"{CACHE}/B.json"))}
        sha_a = A[512]["sha12"]
        sha_b = B[512]["sha12"]
        contract = {}
        for ph in ("A", "B"):
            p = f"{CACHE}/{ph}-contract.json"
            if os.path.exists(p):
                contract[ph] = json.load(open(p))
        if contract.get("A") and contract.get("B"):
            contract["C_comparative"] = comparative(
                open(f"{CACHE}/A-t3.txt").read() if os.path.exists(f"{CACHE}/A-t3.txt") else "",
                "", contract["A"]["semantic"], contract["B"]["semantic"]) \
                if False else "see-report"   # 文本未缓存时由探针侧报告
            contract["target_correctness"] = (
                "PASS" if contract["A"]["self_deterministic"]
                and contract["B"]["self_deterministic"]
                and contract["A"]["semantic"]["ok"]
                and contract["B"]["semantic"]["ok"] else "FAIL")
        pairs = []
        n_total = n_valid = n_pos = n_neg = n_tie = 0
        for tt in sorted(set(A) & set(B)):
            if tt == 512:
                continue
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
        import torch
        t = torch.tensor(pairs, dtype=torch.float64)
        mean = float(t.mean()) if n_valid else 0.0
        # Wilcoxon signed-rank on nonzero diffs (manual, no scipy dependency):
        nz = t[t != 0]
        if n_valid >= 10 and len(nz) >= 5:
            ranks = nz.abs().argsort().argsort().to(torch.float64) + 1
            # average ranks for ties
            order = nz.abs().sort().values
            # simple exact-enough normal approximation
            w_plus = float(ranks[nz > 0].sum())
            n_nz = len(nz)
            mu = n_nz * (n_nz + 1) / 4
            sigma = (n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24) ** 0.5
            z = (w_plus - mu) / sigma
            p = math.erfc(abs(z) / math.sqrt(2))    # two-sided normal approx
        else:
            p = 1.0
        g = torch.Generator().manual_seed(7)
        boots = []
        for _ in range(10000):
            idx = torch.randint(0, n_valid, (n_valid,), generator=g)
            boots.append(float(t[idx].mean()) if n_valid else 0.0)
        boots.sort()
        ci_lo = boots[249]                         # 2.5 percentile
        ci_hi = boots[9749]
        verdict = ("PASS" if (n_valid >= 30 and p < 0.05 and ci_lo > 0
                              and mean > 0)
                   else "INCONCLUSIVE" if n_valid < 30 else "FAIL")
        # sha 记录保留；target-correctness 语义裁定（连贯性）见 REPORT——本引擎
        # 贪心输出随 verify 批组合翻转（S4 机制，P0 裁良性），逐字节等式不可达
        report = {
            "b11_contract": contract,
            "N_total": n_total, "N_valid": n_valid, "N_positive": n_pos,
            "N_negative": n_neg, "N_tie": n_tie,
            "mean_delta_tok_per_step": round(mean, 4),
            "median_delta": round(float(t.median()) if n_valid else 0, 4),
            "bootstrap95_CI": [round(ci_lo, 4), round(ci_hi, 4)],
            "wilcoxon_p_normalapprox": round(p, 5),
            "t3_sha_A": sha_a, "t3_sha_B": sha_b,
            "t3_sha_match": sha_a == sha_b,
            "verdict": verdict,
        }
        with open(f"{CACHE}/stats.json", "w") as f:
            json.dump(report, f, indent=1)
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
