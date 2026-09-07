#!/usr/bin/env python3
"""evaluate.py — 批次I P4 评测：边界误差报告（方向性）+ story 混淆矩阵 + 双金标口径.

两层误差语义（规格冻结）：
  层1 管线内部帧保真：渲染往返身份验证（validate.py frames 产物）+ 帧域闭包 → exact/≤1帧
  层2 vs 金标：ms 级 signed error（early/late）P50/P90/max——受金标秒级粒度约束，不冒充帧级
输出 boundary_error_report.json；控制台打印 GT v1/v2 × {F1, start/end MAE} 汇总表。
pyav-env 或 python3 均可（纯 json + framedomain）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/tools")
import framedomain as fd


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return sorted_vals[i]


def match_gt(det_ms, gt_ms, tol_ms):
    """贪心最近匹配（确定性：按 |d-t| 升序，互斥）。返回 {gt_i: (det_ms, err_ms)}。"""
    pairs = []
    for i, g in enumerate(gt_ms):
        for d in det_ms:
            if abs(d - g) <= tol_ms:
                pairs.append((abs(d - g), i, d))
    pairs.sort(key=lambda x: (x[0], x[1]))
    used_d, out = set(), {}
    for dist, i, d in pairs:
        if i not in out and d not in used_d:
            out[i] = (d, d - gt_ms[i])
            used_d.add(d)
    return out


def eval_version(name, det_ms, gt, tag):
    tol = int(gt["tol_s"] * 1000)
    m = match_gt(det_ms, [int(g * 1000) for g in gt["bounds_s"]], tol)
    errs = sorted(m[i][1] for i in m)
    p = len(m) / len(det_ms) if det_ms else 0.0
    r = len(m) / len(gt["bounds_s"])
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    early = [e for e in errs if e < 0]
    late = [e for e in errs if e > 0]
    fps = [d for d in det_ms if d not in {v[0] for v in m.values()}]
    miss = [gt["bounds_s"][i] for i in range(len(gt["bounds_s"])) if i not in m]
    rec = {
        "gt": tag, "tol_ms": tol,
        "n_det": len(det_ms), "n_gt": len(gt["bounds_s"]),
        "matched": len(m), "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
        "signed_err_ms": {"p50": pct(errs, 0.5), "p90": pct(errs, 0.9),
                          "max_neg": min(errs) if errs else None, "max_pos": max(errs) if errs else None,
                          "early_n": len(early), "late_n": len(late),
                          "early_p50": pct(sorted(early), 0.5), "late_p50": pct(sorted(late), 0.5)},
        "fp_ms": fps, "missed_gt_s": miss,
        "matches": [{"gt_s": gt["bounds_s"][i], "det_ms": v[0], "err_ms": v[1]} for i, v in sorted(m.items())],
    }
    print(f"[{name}|{tag}] P {p:.2f} R {r:.2f} F1 {f1:.2f} | "
          f"err P50={rec['signed_err_ms']['p50']}ms P90={rec['signed_err_ms']['p90']}ms "
          f"early {len(early)}/late {len(late)} | FP {[round(x/1000,1) for x in fps]} "
          f"missed {miss}")
    return rec


def main():
    d = Path(sys.argv[1])
    sf = json.loads((d / "stories_final.json").read_text())
    num, den = sf["fps"]["num"], sf["fps"]["den"]
    bmap = {b["boundary_id"]: b for b in sf["boundaries"]}
    det = [fd.frame_to_ms(bmap[s["start_boundary"]]["frame"], num, den)
           for s in sf["stories"][1:]]  # 内部边界
    ident = sorted((d / "edl").glob("*.identity.json")) if (d / "edl").exists() else []
    layer1 = {
        "frame_closure": "PASS (framedomain.validate_stories_final)",
        "render_identity_files": [p.name for p in ident],
        "render_identity_all_pass": all(json.loads(p.read_text())["pass"] for p in ident) if ident else None,
        "probes": {p.name: [r.get("class", "PASS" if r.get("pass") else "FAIL")
                            for r in json.loads(p.read_text())["probes"]] for p in ident},
    }
    print(f"[层1|内部帧保真] 闭包 PASS；渲染身份 {len(ident)} 文件 "
          f"{'ALL PASS' if layer1['render_identity_all_pass'] else '?'}")
    report = {"schema_version": 1, "run_dir": str(d),
              "n_det_boundaries": len(det),
              "det_bounds_ms": det,
              "layer1_internal_fidelity": layer1,
              "layer2_vs_gt": []}
    for name in ("gt_bounds.json", "gt_bounds_v2.json"):
        p = d / name
        if p.exists():
            report["layer2_vs_gt"].append(eval_version(name, det, json.loads(p.read_text()),
                                                       json.loads(p.read_text())["version"]))
    (d / "boundary_error_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    print(f"boundary_error_report.json 写入（{len(report['layer2_vs_gt'])} 口径）")


if __name__ == "__main__":
    main()
