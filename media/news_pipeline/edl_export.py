#!/usr/bin/env python3
"""edl_export.py v2 — stories_final → EDL JSON 双域（帧 canonical + ms 派生）.

批次I 边界主权：canonical frame 由 refine 唯一产生，EDL 不再吸附词边界（旧 v1 行为）；
ms 一律由 framedomain.frame_to_ms 派生存储，validate 校验两域一致。标题取题花事件→catalog 兜底。

用法:
  python edl_export.py <run_dir> --story 3            # stories_final.json 的条目
  python edl_export.py <run_dir> --all                # 全条目导出
  python edl_export.py <run_dir> --from-frame 10000 --to-frame 15000 --name custom
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/tools")
import framedomain as fd


def story_title(d: Path, t0_ms, t1_ms):
    """标题：该条目起点的题花事件标题 → v2 catalog（按中点）兜底。"""
    try:
        prov = json.loads((d / "provenance.json").read_text())
        sig = json.loads((d / "signal_candidates.json").read_text())
        for e in prov.get("story_edges", []):
            if abs(e.get("t_ms", -1) - t0_ms) <= 2500 and e.get("title_t_ms"):
                ev = next((x for x in sig["ocr_events"] if x["t_ms"] == e["title_t_ms"]), None)
                if ev and ev.get("new_titles"):
                    return ev["new_titles"][0]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    try:
        cat = [json.loads(l) for l in open(d / "e2e" / "catalog.jsonl")]
        mid = (t0_ms + t1_ms) / 2
        c = next((c for c in cat if c["t_start_ms"] <= mid < c["t_end_ms"]), None)
        return c["semantic"]["title"] if c else ""
    except FileNotFoundError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--story", type=int, help="stories_final 条目 id")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--from-frame", type=int)
    ap.add_argument("--to-frame", type=int)
    ap.add_argument("--name")
    a = ap.parse_args()
    d = Path(a.run_dir)
    sf = json.loads((d / "stories_final.json").read_text())
    num, den = sf["fps"]["num"], sf["fps"]["den"]
    n = sf["n_frames"]
    bmap = {b["boundary_id"]: b for b in sf["boundaries"]}

    def export(st, idx):
        f0, f1 = bmap[st["start_boundary"]]["frame"], bmap[st["end_boundary"]]["frame"]
        s_ms, e_ms = fd.frame_to_ms(f0, num, den), fd.frame_to_ms(f1, num, den)
        name = a.name or f"story_{idx:03d}"
        title = story_title(d, s_ms, e_ms)
        edl = {
            "version": 2, "source": str(
                d.parent.parent.parent / "news_videos" / f"linyi_news_{d.name.split('_')[-1]}.mp4"),
            "title": title, "fps": {"num": num, "den": den},
            "ranges": [{"source": name, "start_frame": f0, "end_frame": f1,
                        "start_ms": s_ms, "end_ms": e_ms, "reason": "story export"}],
            "fades": {"in_ms": 30, "out_ms": 30}, "subtitles": None,
            "total_duration_ms": e_ms - s_ms, "duration_frames": f1 - f0,
        }
        out = d / "edl" / f"{name}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(edl, ensure_ascii=False, separators=(",", ":")))
        print(f"{out.name}: 帧[{f0},{f1}) {f1-f0}帧 {s_ms}-{e_ms}ms 《{title[:24]}》")

    if a.all:
        for i, st in enumerate(sf["stories"]):
            export(st, i)
    elif a.story is not None:
        export(sf["stories"][a.story], a.story)
    elif a.from_frame is not None and a.to_frame is not None:
        f0, f1 = max(0, a.from_frame), min(n, a.to_frame)
        fake = {"start_boundary": None, "end_boundary": None}
        name = a.name or f"clip_{f0}_{f1}"
        edl = {
            "version": 2, "source": str(
                d.parent.parent.parent / "news_videos" / f"linyi_news_{d.name.split('_')[-1]}.mp4"),
            "title": "", "fps": {"num": num, "den": den},
            "ranges": [{"source": name, "start_frame": f0, "end_frame": f1,
                        "start_ms": fd.frame_to_ms(f0, num, den),
                        "end_ms": fd.frame_to_ms(f1, num, den), "reason": "custom"}],
            "fades": {"in_ms": 30, "out_ms": 30}, "subtitles": None,
            "total_duration_ms": fd.frame_to_ms(f1, num, den) - fd.frame_to_ms(f0, num, den),
            "duration_frames": f1 - f0,
        }
        out = d / "edl" / f"{name}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(edl, ensure_ascii=False, separators=(",", ":")))
        print(f"{out.name}: 帧[{f0},{f1})")
    else:
        ap.error("需 --story / --all / --from-frame+--to-frame")


if __name__ == "__main__":
    main()
