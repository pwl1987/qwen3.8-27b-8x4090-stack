#!/usr/bin/env python3
"""B2-A recovery table + frozen decision rules."""
import glob
import json

A = {r["target_tokens"]: r["tok_per_step"] for r in json.load(open("/tmp/b1/accept-cache/A.json")) if r.get("tok_per_step")}
B = {r["target_tokens"]: r["tok_per_step"] for r in json.load(open("/tmp/b1/accept-cache/B.json")) if r.get("tok_per_step")}
tt = sorted(set(A) & set(B))
tt = [t for t in tt if t != 512]

ARMS = {
    "S2 fc-int4+oldH": "/tmp/b1/w4probe.json",
    "S3 fc-bf16+oldH": "/tmp/b2a/probe-b1-fc16.json",
    "S4 fc-int8+oldH": "/tmp/b2a/probe-b1-fc8.json",
    "S6 prod-recal": "/tmp/b2a/probe-prod-recal.json",
}
for f in sorted(glob.glob("/tmp/b2a/probe-*.json")):
    lab = json.load(open(f))["label"]
    if lab not in ("b1-fc16", "b1-fc8", "prod-recal"):
        ARMS[f"S5 {lab}"] = f

def m(v, keys):
    keys = [k for k in keys if k in v]
    return sum(v[k] for k in keys) / len(keys) if keys else None

s0, s1 = m(A, tt), m(B, tt)
print(f"配对 n={len(tt)}  S0 baseline-bf16={s0:.4f}  S1 trained-bf16={s1:.4f}  Δ={s1-s0:+.4f}")
print(f"{'臂':<22s} {'均值':>7s} {'ΔvsS0':>8s} {'R总体':>7s} {'R逐对中位':>9s} {'t3':>6s} det sha")
table = {}
for lab, f in ARMS.items():
    d = json.load(open(f))
    rows = d["rows"] if isinstance(d, dict) else d
    v = {r["target_tokens"]: r["tok_per_step"] for r in rows if r.get("tok_per_step")}
    keys = [k for k in tt if k in v]
    sub0, sub1, subv = m(A, keys), m(B, keys), m(v, keys)
    R = (subv - sub0) / (sub1 - sub0) if sub1 != sub0 else None
    pers = sorted((v[k] - A[k]) / (B[k] - A[k]) for k in keys if abs(B[k] - A[k]) > 1e-6)
    Rmed = pers[len(pers) // 2] if pers else None
    c = d.get("contract", {}) if isinstance(d, dict) else {}
    t3 = v.get(512)
    print(f"{lab:<22s} {subv:7.4f} {subv-sub0:+8.4f} "
          f"{('%+.2f' % R) if R is not None else '  n/a':>7s} "
          f"{('%+.2f' % Rmed) if Rmed is not None else 'n/a':>9s} "
          f"{t3:6.3f} {c.get('self_deterministic')} {c.get('sha12', '')[:8]}")
    table[lab] = {"mean": subv, "delta_vs_S0": subv - sub0, "R": R,
                  "R_median": Rmed, "n": len(keys)}
json.dump({"S0": s0, "S1": s1, "arms": table},
          open("/data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b2a/recovery-table.json", "w"),
          indent=1)
print("\n冻结判定: S3 R>=0.7 → 量化精度是症结 | S3<0.3 且 S5<0.3 → 收益脆弱 | 其间按数据裁定")
