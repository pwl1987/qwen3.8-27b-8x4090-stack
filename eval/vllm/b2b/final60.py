#!/usr/bin/env python3
"""B2-B 修订2（CONTRACT-B2B.md §8）: FINAL-60 五相位配对 acceptance + 四格终判。

Phases（每相位 = 一次全新 zx-p0 boot，单变量 DRAFT，flags 全同）:
  W60   = W4-BASELINE  (...-b2b-w4base)   PRIMARY 同族对照（未训练 masters × 同 int4 层）
  Q60   = QAT-W4       (...-b2b)          处理臂
  C60   = S0 bf16      (...-DFlash2)      SECONDARY 参照
  S1_60 = B1 trained bf16 (...-b1)        B1-C 的 n=60 复测（描述性，不回写 B1-C 判定）
  S3_60 = naive transfer (...-b1-fc16)    context 臂

FINAL-60 = make_prompt(300..714 step 7) 共 60 条，固定 seed-7 排序、五相位同序；
每相位记录前 2 条 warmup（260/760，FINAL 集外）丢弃不入场。证据 schema 每行：
prompt_id / phase / boot / order / seed / sec / drafts / accepted / tok_per_step
（error 行保留不静默删）。

Usage:
  python final60.py --phase W60 --boot zx-p0-W   # against a live boot (:19627)
  python final60.py --phase stats                # paired tests + four-quadrant verdict
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, "/data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b1")
sys.path.insert(0, "/data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b2b")
from accept_ab import run_prompt, t3_full  # noqa: E402

CACHE = "/tmp/b2b/accept60"
os.makedirs(CACHE, exist_ok=True)
FINAL60 = list(range(300, 715, 7))          # 60 prompts, preregistered
assert len(FINAL60) == 60
WARMUP = [260, 760]                          # outside FINAL60, discarded
PHASES = ["W60", "Q60", "C60", "S1_60", "S3_60"]

_ORDER = FINAL60[:]
random.Random(7).shuffle(_ORDER)             # fixed order, identical across phases
ORDER = {tt: i for i, tt in enumerate(_ORDER)}


def run_engine_phase(base: str, plabel: str, boot: str) -> None:
    t3a = t3_full(base, plabel)
    t3b = t3_full(base, plabel)
    contract = {"self_deterministic": t3a["sha12"] == t3b["sha12"],
                "sha12": t3a["sha12"], "chars": len(t3a["text"]),
                "semantic": t3a["semantic"],
                "t3_tok_per_step": t3a["tok_per_step"],
                "text": t3a["text"]}
    with open(f"{CACHE}/{plabel}-contract.json", "w") as f:
        json.dump(contract, f, indent=1, ensure_ascii=False)
    print(f"[{plabel}] t3 sha={contract['sha12']} "
          f"det={contract['self_deterministic']} sem={t3a['semantic']}", flush=True)

    warm = []
    for tt in WARMUP:
        try:
            r = run_prompt(base, tt, plabel)
            warm.append(r)
            print(f"[{plabel}] warmup {tt} tok/step={r['tok_per_step']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            warm.append({"target_tokens": tt, "tag": plabel, "error": repr(e)[:200],
                         "tok_per_step": None})
    with open(f"{CACHE}/{plabel}-warmup.json", "w") as f:
        json.dump(warm, f, indent=1)

    records = []
    for tt in _ORDER:
        try:
            r = run_prompt(base, tt, plabel)
            row = dict(r)
            row.update(prompt_id=tt, phase=plabel, boot=boot, order=ORDER[tt], seed=4242)
            records.append(row)
            print(f"[{plabel}] #{ORDER[tt]:02d} {tt} tok/step={r['tok_per_step']:.3f}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            records.append({"prompt_id": tt, "phase": plabel, "boot": boot,
                            "order": ORDER[tt], "seed": 4242, "error": repr(e)[:200],
                            "tok_per_step": None})
            print(f"[{plabel}] #{ORDER[tt]:02d} {tt} ERROR {e!r}", flush=True)
    with open(f"{CACHE}/{plabel}.json", "w") as f:
        json.dump(records, f, indent=1)
    n_ok = sum(r["tok_per_step"] is not None for r in records)
    print(f"[{plabel}] done: {n_ok}/60 valid rows -> {CACHE}/{plabel}.json", flush=True)


def paired_stats(diffs, mean_a):
    """five-field + Wilcoxon(normal approx) + bootstrap(seed 7, 10k)."""
    import torch
    t = torch.tensor(diffs, dtype=torch.float64)
    n_valid = len(diffs)
    n_pos = int((t > 0).sum())
    n_neg = int((t < 0).sum())
    n_tie = int((t == 0).sum())
    mean = float(t.mean()) if n_valid else 0.0
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
    return {
        "N_total": 60, "N_valid": n_valid, "N_positive": n_pos,
        "N_negative": n_neg, "N_tie": n_tie,
        "five_field": {
            "abs_mean_delta_tok_per_step": round(mean, 4),
            "rel_percent": round(100 * mean / mean_a, 2) if mean_a else None,
            "median_delta": round(float(t.median()) if n_valid else 0, 4),
            "bootstrap95_CI": [round(ci_lo, 4), round(ci_hi, 4)],
            "positive_fraction": round(n_pos / max(1, n_valid), 4),
        },
        "wilcoxon_p_normalapprox": round(p, 5),
        "z": round(float(z), 3),
    }


def compare(ph_b: str, ph_a: str, phase_rows, gated=False):
    """Δ = B − A over FINAL-60 prompt_id intersection; verdict only if gated."""
    a = {r["prompt_id"]: r.get("tok_per_step") for r in phase_rows[ph_a]}
    b = {r["prompt_id"]: r.get("tok_per_step") for r in phase_rows[ph_b]}
    ids = [tt for tt in FINAL60
           if a.get(tt) is not None and b.get(tt) is not None]
    diffs = [b[tt] - a[tt] for tt in ids]
    mean_a = (sum(a[tt] for tt in FINAL60 if a.get(tt) is not None)
              / max(1, sum(1 for tt in FINAL60 if a.get(tt) is not None)))
    out = {"delta": f"{ph_b} minus {ph_a}", "n_valid_pairs": len(ids)}
    out.update(paired_stats(diffs, mean_a))
    if gated:
        ff = out["five_field"]
        out["verdict"] = (
            "PASS" if (out["N_valid"] >= 60 and out["wilcoxon_p_normalapprox"] < 0.05
                       and ff["bootstrap95_CI"][0] > 0
                       and ff["abs_mean_delta_tok_per_step"] > 0)
            else "INCONCLUSIVE" if out["N_valid"] < 60 else "FAIL")
    return out


def comparative(a_text: str, b_text: str):
    la, lb = len(a_text), len(b_text)
    cp = next((i for i in range(min(la, lb)) if a_text[i] != b_text[i]),
              min(la, lb))
    return ("EXACT" if a_text == b_text else
            "BENIGN-DIFF" if (cp >= 20 and 0.5 <= la / max(1, lb) <= 2.0)
            else "UNRESOLVED")


def stats():
    phase_rows, contracts = {}, {}
    for ph in PHASES:
        p = f"{CACHE}/{ph}.json"
        if not os.path.exists(p):
            sys.exit(f"missing phase file: {p} (run engine phases first)")
        phase_rows[ph] = json.load(open(p))
        contracts[ph] = json.load(open(f"{CACHE}/{ph}-contract.json"))

    means = {}
    for ph in PHASES:
        vals = [r["tok_per_step"] for r in phase_rows[ph]
                if r.get("tok_per_step") is not None]
        means[ph] = round(sum(vals) / len(vals), 4)

    primary = compare("Q60", "W60", phase_rows, gated=True)      # PRIMARY gate
    secondary = {
        "Q_minus_C0": compare("Q60", "C60", phase_rows),
        "Q_minus_S3_naive": compare("Q60", "S3_60", phase_rows),
        "S1_minus_C0_B1C_retest": compare("S1_60", "C60", phase_rows),
    }
    qw = primary["five_field"]["abs_mean_delta_tok_per_step"]
    s1c0 = secondary["S1_minus_C0_B1C_retest"]["five_field"][
        "abs_mean_delta_tok_per_step"]
    recovery = round(qw / s1c0, 3) if s1c0 else None

    # B1.1 t3 contract: every phase self-det + semantic; pairwise comparatives
    b11 = {ph: {"self_deterministic": c["self_deterministic"],
                "semantic_ok": c["semantic"]["ok"],
                "sha12": c["sha12"]} for ph, c in contracts.items()}
    for name, (x, y) in {"Q_vs_W": ("Q60", "W60"), "Q_vs_C0": ("Q60", "C60"),
                         "Q_vs_S3": ("Q60", "S3_60"),
                         "S1_vs_C0": ("S1_60", "C60")}.items():
        b11[name] = comparative(contracts[x]["text"], contracts[y]["text"])
    target_correctness = "PASS" if all(
        v["self_deterministic"] and v["semantic_ok"] for v in b11.values()
        if isinstance(v, dict) and "self_deterministic" in v
    ) and not any(v == "UNRESOLVED" for v in b11.values()) else "FAIL"

    # four-quadrant (CONTRACT-B2B.md §8.1)
    if primary["verdict"] == "PASS":
        quadrant = ("最佳：量化基本被补偿" if means["Q60"] >= means["C60"]
                    else "科学成功：QAT 有效，仍有量化损失")
    elif qw > 0:
        quadrant = "QAT 失败（方向为正但不显著，≈0）"
    else:
        quadrant = "QAT 反效果"

    report = {
        "comparator": "CONTRACT-B2B.md 修订2 §8.1 (PRIMARY Q vs W; SECONDARY "
                      "Q-C0/Q-S3/S1-C0; AUX recovery same-corpus)",
        "phase_means_FINAL60": means,
        "PRIMARY_Q_minus_W": primary,
        "SECONDARY": secondary,
        "AUXILIARY_recovery_same_corpus": recovery,
        "engineering_Q_ge_3p5": means["Q60"] >= 3.5,
        "four_quadrant": quadrant,
        "b11_contract": b11,
        "target_correctness": target_correctness,
    }
    with open(f"{CACHE}/stats.json", "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=PHASES + ["stats"])
    ap.add_argument("--port", type=int, default=19627)
    ap.add_argument("--label", default=None)
    ap.add_argument("--boot", default="zx-p0", help="boot id recorded per row")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    if args.phase == "stats":
        stats()
    else:
        run_engine_phase(base, args.label or args.phase, args.boot)


if __name__ == "__main__":
    main()
