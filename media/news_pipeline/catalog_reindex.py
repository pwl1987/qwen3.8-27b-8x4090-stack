#!/usr/bin/env python3
"""catalog_reindex.py — 批次I 导出层对齐：v2 catalog → stories_final 口径（确定性重索引）.

不重跑 LLM 语义：每条 final 条目取重叠最大的 v2 catalog 条目为语义主体（合并条目标
merged_from），标题优先用题花事件；shot/audio 统计按 final 边界重算。
输出 e2e/catalog_final.jsonl（schema 同 catalog.jsonl + merged_from/title_event）。
"""
import json
import sys
from pathlib import Path


def main():
    d = Path(sys.argv[1])
    sf = json.loads((d / "stories_final.json").read_text())
    prov = json.loads((d / "provenance.json").read_text())
    sig = json.loads((d / "signal_candidates.json").read_text())
    if (d / "e2e" / "catalog.jsonl").exists():
        cat = [json.loads(l) for l in open(d / "e2e" / "catalog.jsonl")]
    else:  # 新一期无 catalog：semantic.json(v2 口径) 提供语义，shots/audio 原料直取
        cat = []
        vis = {}
        if (d / "e2e" / "vision.json").exists():
            for k, v in json.loads((d / "e2e" / "vision.json").read_text()).items():
                try:
                    vis[int(k)] = json.loads(v) if isinstance(v, str) else v
                except (json.JSONDecodeError, ValueError):
                    pass
        sh_list = json.loads((d / "shots.json").read_text())["shots"]
        segs = json.loads((d / "e2e" / "asr.json").read_text())["segments"]
        for s in json.loads((d / "e2e" / "semantic.json").read_text()):
            try:
                sem = json.loads(s["result"])
            except (json.JSONDecodeError, TypeError):
                sem = {}
            inner_shots = [dict(sh, vision=vis.get(sh["index"], {})) for sh in sh_list
                           if sh["t_start_ms"] < s["t_end_ms"] and sh["t_end_ms"] > s["t_start_ms"]]
            inner_segs = [sg for sg in segs
                          if sg["t_start_ms"] < s["t_end_ms"] and sg["t_end_ms"] > s["t_start_ms"]]
            cat.append({"story_id": len(cat), "t_start_ms": s["t_start_ms"], "t_end_ms": s["t_end_ms"],
                        "semantic": sem, "shots": inner_shots,
                        "audio": {"transcript_segments": inner_segs}})
    shots = json.loads((d / "shots.json").read_text())["shots"]
    bmap = {b["boundary_id"]: b for b in sf["boundaries"]}
    edges = {e["t_ms"]: e for e in prov["story_edges"]}
    out = []
    for i, st in enumerate(sf["stories"]):
        f0 = bmap[st["start_boundary"]]["frame"]
        f1 = bmap[st["end_boundary"]]["frame"]
        t0, t1 = bmap[st["start_boundary"]]["time_ms"], bmap[st["end_boundary"]]["time_ms"]
        ov = [(min(c["t_end_ms"], t1) - max(c["t_start_ms"], t0), c) for c in cat]
        ov = [(o, c) for o, c in ov if o > 0]
        ov.sort(key=lambda x: (-x[0], x[1]["story_id"]))
        dom = ov[0][1] if ov else None
        merged = [c["story_id"] for o, c in ov[1:] if o > 3000]
        e = next((e for t, e in edges.items() if abs(t - t0) <= 2500), None)
        title_ev = None
        if e and e.get("title_t_ms"):
            title_ev = next((x for x in sig["ocr_events"] if x["t_ms"] == e["title_t_ms"]), None)
        sem = dict(dom["semantic"]) if dom else {}
        if title_ev and title_ev.get("new_titles"):
            sem["title"] = title_ev["new_titles"][0]
            sem["title_source"] = "title_event_ocr"
        n_shots = sum(1 for s in shots if s["t_start_ms"] < t1 and s["t_end_ms"] > t0)
        # 携带 v2 catalog 的 shots/audio（按 final 边界裁剪）——packed_md 无需改结构
        cat_shots = []
        for o, c in ov:
            for sh in c.get("shots", []):
                if sh["t_start_ms"] < t1 and sh["t_end_ms"] > t0:
                    cat_shots.append(sh)
        segs = []
        for o, c in ov:
            for sg in c.get("audio", {}).get("transcript_segments", []):
                if sg["t_start_ms"] < t1 and sg["t_end_ms"] > t0:
                    segs.append(sg)
        rec = {"story_id": i, "t_start_ms": t0, "t_end_ms": t1,
               "start_frame": f0, "end_frame": f1,
               "semantic": sem, "n_shots": n_shots or len(cat_shots),
               "shots": sorted(cat_shots, key=lambda x: x["t_start_ms"]),
               "audio": {"transcript_segments": sorted(segs, key=lambda x: x["t_start_ms"])},
               "merged_from": merged,
               "source": "catalog_reindex(v2 dominant-overlap)" if dom else "new_story"}
        out.append(rec)
    with open(d / "e2e" / "catalog_final.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"catalog_final.jsonl: {len(out)} 条（新故事 {sum(1 for r in out if r['source']!='catalog_reindex(v2 dominant-overlap)')}，"
          f"合并 {sum(1 for r in out if r['merged_from'])}）")
    for r in out:
        print(f"  S{r['story_id']:02d} [{r['t_start_ms']/1000:7.1f}-{r['t_end_ms']/1000:7.1f}s] "
              f"《{r['semantic'].get('title','')[:20]}》 shots={r['n_shots']} merged={r['merged_from']}")


if __name__ == "__main__":
    main()
