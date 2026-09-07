#!/usr/bin/env python3
"""门禁比较器: baseline(3.0+backfill 合并) vs gate 运行, 输出判定表。

规则 (计划 v3): 任一轴绝对回退>2pp → FAIL; 核心轴(code/tool)必须严格提升。
吞吐轴(approx_tps)按相对2%判定; needle 按命中率。
用法: python compare_gate.py baseline.json [baseline_backfill.json] gate.json
"""
import json
import sys

# 轴 -> [(指标键, 方向, 是否核心, 判定方式)]  方向 +1=越高越好
SPEC = {
    "humaneval": [("pass_at_1", +1, True, "pp")],
    "xfc":       [("accuracy", +1, True, "pp")],
    "gsm8k":     [("accuracy", +1, False, "pp")],
    "ifeval":    [("accuracy", +1, False, "pp")],
    "needle":    [("needle_64k.hit", +1, False, "count"),
                  ("needle_195k.hit", +1, False, "count")],
    "longgen":   [("approx_tps", +1, False, "rel")],
}


def dig(d, path):
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def main():
    files = sys.argv[1:]
    gate_path = files[-1]
    base_files = files[:-1]
    gate = json.load(open(gate_path))
    base = {"axes": {}}
    for f in base_files:
        d = json.load(open(f))
        base["axes"].update(d.get("axes", {}))

    rows, fail, core_need = [], [], []
    for ax, metrics in SPEC.items():
        for key, dirn, core, mode in metrics:
            b, g = dig(base.get("axes", {}), f"{ax}.{key}"), dig(gate.get("axes", {}), f"{ax}.{key}")
            if b is None or g is None:
                rows.append((ax, key, b, g, "MISSING", "—"))
                fail.append(f"{ax}.{key} 数据缺失")
                continue
            delta = (g - b) * dirn
            if mode == "pp":
                ok = delta >= (-2.0 if not core else 0.0)
                verdict = "PASS" if delta > 0 else ("PASS" if ok else "FAIL")
                dstr = f"{(g-b)*100:+.1f}pp"
            elif mode == "rel":
                ok = delta >= -0.02 * b
                verdict = "PASS" if ok else "FAIL"
                dstr = f"{(g-b)/b*100:+.1f}%"
            else:  # count
                ok = g >= b
                verdict = "PASS" if ok else "FAIL"
                dstr = f"{g}/{b}"
            if not ok:
                fail.append(f"{ax}.{key} 回退 {dstr}")
            if core and delta <= 0:
                core_need.append(f"{ax}.{key} 未严格提升")
            rows.append((ax, key, b, g, dstr, verdict))

    print(f"| 轴 | 指标 | 基线 | 门禁 | Δ | 判定 |")
    print("|---|---|---|---|---|---|")
    for ax, key, b, g, d, v in rows:
        fb = f"{b:.4f}" if isinstance(b, float) else str(b)
        fg = f"{g:.4f}" if isinstance(g, float) else str(g)
        print(f"| {ax} | {key} | {fb} | {fg} | {d} | {v} |")

    print()
    if fail:
        print("结论: FAIL")
        for f_ in fail:
            print(f"  - {f_}")
    elif core_need:
        print("结论: 条件通过(无回退, 但核心轴未严格提升 — 按计划不可上线, 可进下一轮)")
        for c in core_need:
            print(f"  - {c}")
    else:
        print("结论: PASS — 全轴无回退且核心轴严格提升, 可进入 Phase 6 合并量化")


if __name__ == "__main__":
    main()
