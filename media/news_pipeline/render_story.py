#!/usr/bin/env python3
"""render_story.py — EDL JSON → MP4（借鉴 video-use 渲染硬规则）.

固化规则：每段单次编码 + concat 无损拼接（绝不二次编码）；每切点 30ms 音频淡入淡出防爆音；
--srt 字幕最后挂（filter chain 末位）；--preview 720p 快出。单段则直接一次编码。

用法: python render_story.py <edl.json> [--preview] [--srt xx.srt]
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

FF = "/data/tools/ffmpeg-deb/ffmpeg"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"ffmpeg failed: {r.stderr[-800:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edl")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--srt")
    a = ap.parse_args()
    edl = json.loads(Path(a.edl).read_text())
    src = edl["source"]
    fades = edl.get("fades", {})
    fi, fo = fades.get("in_ms", 30) / 1000, fades.get("out_ms", 30) / 1000
    out = Path(a.edl).with_suffix(".mp4")
    vf = "scale=1280:-2" if a.preview else None
    if edl.get("version", 0) >= 2:  # 帧域断言：ms 必须与 frame 严格互推（framedomain 契约）
        import sys
        sys.path.insert(0, "/data/tools")
        import framedomain as fd
        num, den = edl["fps"]["num"], edl["fps"]["den"]
        for seg in edl["ranges"]:
            for k, fk in (("start_ms", "start_frame"), ("end_ms", "end_frame")):
                want = fd.frame_to_ms(seg[fk], num, den)
                if seg[k] != want:
                    raise SystemExit(f"EDL 双域不一致: {k}={seg[k]} != frame {seg[fk]}→{want}ms")

    def encode(seg, dest):
        dur = (seg["end_ms"] - seg["start_ms"]) / 1000
        af = f"afade=t=in:d={fi},afade=t=out:st={max(dur - fo, 0):.3f}:d={fo}"
        if a.srt:  # 规则1：字幕最后挂（段内偏移已由 srt 时间轴适配，这里直接 subtitles 滤镜置尾）
            vf_chain = (vf + "," if vf else "") + f"subtitles={a.srt}"
            cmd = [FF, "-y", "-loglevel", "error", "-ss", f"{seg['start_ms']/1000}", "-i", src,
                   "-t", f"{dur:.3f}", "-vf", vf_chain, "-af", af,
                   "-c:v", "libx264", "-crf", "18" if not a.preview else "23",
                   "-preset", "veryfast", "-c:a", "aac", str(dest)]
        else:
            cmd = [FF, "-y", "-loglevel", "error", "-ss", f"{seg['start_ms']/1000}", "-i", src,
                   "-t", f"{dur:.3f}"]
            cmd += (["-vf", vf] if vf else []) + ["-af", af,
                   "-c:v", "libx264", "-crf", "18" if not a.preview else "23",
                   "-preset", "veryfast", "-c:a", "aac", str(dest)]
        run(cmd)

    if len(edl["ranges"]) == 1:
        encode(edl["ranges"][0], out)
    else:  # 规则2：分段单次编码 → concat 无损拼
        with tempfile.TemporaryDirectory() as td:
            parts = []
            for i, seg in enumerate(edl["ranges"]):
                p = Path(td) / f"part{i:02d}.mp4"
                encode(seg, p)
                parts.append(p)
            lst = Path(td) / "list.txt"
            lst.write_text("".join(f"file '{p}'\n" for p in parts))
            run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                 "-c", "copy", str(out)])
    d = subprocess.check_output(["/data/tools/ffmpeg-deb/ffprobe", "-v", "error",
                                 "-show_entries", "format=duration", "-of", "csv=p=0", str(out)], text=True).strip()
    print(f"{out}  渲染时长 {float(d):.2f}s (EDL 预期 {edl['total_duration_ms']/1000:.2f}s)")


if __name__ == "__main__":
    main()
