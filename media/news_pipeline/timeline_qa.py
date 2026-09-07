#!/usr/bin/env python
"""timeline_qa.py — 决策点合成图生成器（借鉴 browser-use/video-use 的 timeline_view 思想）.

一张 PNG = filmstrip 帧条 + RMS 波形 + 镜头边界 + ASR 文本标签，
用于条目边界/任意时间点的人工抽查与 LLM 自评，免拖视频。

用法（pyav-env 运行）:
  python timeline_qa.py <run_dir> boundaries [--window 1.5]   # 全部条目内部边界
  python timeline_qa.py <run_dir> stories                     # 每条目首尾帧卡片
  python timeline_qa.py <run_dir> --at <ms> [--window 1.5]    # 任意时间点
可选: --video <mp4 路径>（默认 run_dir/../../news_videos/linyi_news_<ep>.mp4）

run_dir 约定（bench_outputs/shotseg/news_YYYYMMDD/）:
  audio_sig.json(jump=0.5s 网格 RMS) shots.json stories_v2.json e2e/asr.json e2e/catalog.jsonl
输出: run_dir/qa/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W = 1600
N_FRAMES = 8


def mmss(ms):
    s = ms / 1000.0
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


class VideoReader:
    def __init__(self, path):
        import av
        self.ctx = av.open(str(path))
        self.vs = self.ctx.streams.video[0]
        self.tb = self.vs.time_base

    def frame_at(self, t_s):
        import av
        self.ctx.seek(int(t_s / self.tb), stream=self.vs, backward=True, any_frame=False)
        for pkt in self.ctx.demux(self.vs):
            for fr in pkt.decode():
                if abs(fr.time - t_s) < 0.6 or fr.time >= t_s:
                    if fr.time - t_s > 1.5:  # seek 后解出的首帧离目标太远，重试精确 seek
                        self.ctx.seek(int(t_s / self.tb) + 1, stream=self.vs, backward=False)
                        continue
                    return fr.to_ndarray(format="rgb24")
        raise RuntimeError(f"no frame at {t_s}")


def load(run_dir, video):
    d = Path(run_dir)
    ep = d.name.split("_")[-1]
    video = Path(video) if video else (d.parent.parent.parent / "news_videos" / f"linyi_news_{ep}.mp4")
    sig = json.load(open(d / "audio_sig.json"))
    shots = json.load(open(d / "shots.json"))
    shots = shots["shots"] if isinstance(shots, dict) else shots
    try:
        stories = json.load(open(d / "stories_v2.json"))
        stories = stories["stories"] if isinstance(stories, dict) else stories
    except FileNotFoundError:
        stories = []
    try:
        asr = json.load(open(d / "e2e" / "asr.json"))["segments"]
    except FileNotFoundError:
        asr = []
    catalog = {}
    cf = d / "e2e" / "catalog.jsonl"
    if cf.exists():
        for line in open(cf):
            rec = json.loads(line)
            catalog[rec["story_id"]] = rec
    return d, video, sig, shots, stories, asr, catalog


def draw_boundary(out_png, at_ms, window, vr, sig, shots, stories, asr, catalog):
    lo, hi = at_ms - window * 1000, at_ms + window * 1000
    si = next((i for i, s in enumerate(stories) if s["t_start_ms"] <= at_ms < s["t_end_ms"]), None)
    img = Image.new("RGB", (W, 560), "white")
    dr = ImageDraw.Draw(img)
    f_t = ImageFont.truetype(FONT_B, 22)
    f_s = ImageFont.truetype(FONT, 15)
    f_x = ImageFont.truetype(FONT, 13)

    # ── 头部 ──
    nxt = si + 1 if si is not None and si + 1 < len(stories) else None
    t1 = catalog.get(si, {}).get("semantic", {}).get("title", f"story{si}")
    t2 = catalog.get(nxt, {}).get("semantic", {}).get("title", f"story{nxt}") if nxt is not None else "—END—"
    dr.rectangle([0, 0, W, 46], fill="#1a355e")
    dr.text((14, 10), f"S{si} 「{t1[:28]}」 → S{nxt} 「{t2[:28]}」 @ {mmss(at_ms)}  ±{window}s", font=f_t, fill="white")

    # ── filmstrip ──
    ts = np.linspace(lo, hi, N_FRAMES + 2)[1:-1]
    fw = (W - 40) // N_FRAMES
    fh = int(fw * 9 / 16)
    for k, t in enumerate(ts):
        arr = vr.frame_at(t / 1000)
        im = Image.fromarray(arr).resize((fw - 6, fh))
        img.paste(im, (20 + k * fw, 60))
        c = "#b00020" if t > at_ms else "#333"
        dr.text((20 + k * fw, 62 + fh), f"{mmss(t)}", font=f_s, fill=c)
    dr.line([(20 + (W - 40) * 0.5, 58), (20 + (W - 40) * 0.5, 64 + fh + 20)], fill="#b00020", width=3)

    # ── 波形 ──
    y0 = 64 + fh + 34
    wh = 130
    dr.rectangle([18, y0, W - 20, y0 + wh], fill="#f6f8fb")
    grid = sig.get("rms_grid_s", 0.5)
    rms = np.asarray(sig.get("jump", []), dtype=float)
    ctx_lo, ctx_hi = lo - 2000, hi + 2000
    ii = np.arange(int(ctx_lo / 1000 / grid), int(ctx_hi / 1000 / grid) + 1)
    ii = ii[(ii >= 0) & (ii < len(rms))]
    if len(ii):
        v = rms[ii]
        mx = max(v.max(), 1e-6)
        xs = 20 + (W - 40) * (ii * grid * 1000 - ctx_lo) / (ctx_hi - ctx_lo)
        ys = y0 + wh - 8 - (v / mx) * (wh - 20)
        dr.line(list(zip(xs.tolist(), ys.tolist())), fill="#2b6cb0", width=2)
    def x_of(ms_):
        return 20 + (W - 40) * (ms_ - ctx_lo) / (ctx_hi - ctx_lo)
    for s in sig.get("silence_mids_s", []):
        if ctx_lo <= s * 1000 <= ctx_hi:
            dr.line([(x_of(s * 1000), y0), (x_of(s * 1000), y0 + wh)], fill="#8fd14f", width=1)
    for sh in shots:
        if lo <= sh["t_start_ms"] <= hi:
            x_s = x_of(sh["t_start_ms"])
            dr.line([(x_s, y0), (x_s, y0 + wh)], fill="#666", width=1)
    xb = x_of(at_ms)
    dr.line([(xb, y0 - 4), (xb, y0 + wh + 4)], fill="#b00020", width=3)
    dr.text((24, y0 + 4), f"RMS 0.5s 网格 | 绿=静音中点 灰=镜头边界 红=查询点", font=f_x, fill="#555")

    # ── ASR 文本 ──
    y = y0 + wh + 16
    dr.text((20, y), "ASR:", font=f_s, fill="#333"); y += 24
    for seg in asr:
        if seg["t_start_ms"] < hi and seg["t_end_ms"] > lo:
            side = "▶" if seg["t_start_ms"] >= at_ms else "◀"
            txt = seg["text"][:64]
            dr.text((28, y), f"[{mmss(seg['t_start_ms'])}-{mmss(seg['t_end_ms'])}] {side} {txt}", font=f_s,
                    fill="#b00020" if seg["t_start_ms"] >= at_ms else "#222")
            y += 22
            if y > 540: break

    # ── 底部: 邻近镜头的边界信号 ──
    near = [s for s in shots if lo - 3000 <= s["t_start_ms"] <= hi + 3000]
    sig_txt = " | ".join(f"shot{s['index']}@{mmss(s['t_start_ms'])}:{','.join(s['boundary_signals'].keys()) or '无'}"
                         for s in near[:3])
    dr.text((20, 536), sig_txt[:110], font=f_x, fill="#777")
    img.save(out_png)


def draw_story_card(out_png, story, idx, vr, catalog):
    img = Image.new("RGB", (980, 420), "white")
    dr = ImageDraw.Draw(img)
    f_t = ImageFont.truetype(FONT_B, 21)
    f_s = ImageFont.truetype(FONT, 16)
    sem = catalog.get(idx, {}).get("semantic", {})
    dr.rectangle([0, 0, 980, 44], fill="#1a355e")
    dr.text((12, 9), f"S{idx} 「{sem.get('title','?')[:40]}」", font=f_t, fill="white")
    arrs = (vr.frame_at(story["t_start_ms"] / 1000 + 0.2), vr.frame_at(max(story["t_end_ms"] / 1000 - 0.2, 0)))
    for k, arr in enumerate(arrs):
        im = Image.fromarray(arr).resize((468, 264))
        img.paste(im, (12 + k * 488, 56))
        dr.text((12 + k * 488, 324), ("首帧 " if k == 0 else "末帧 ") + mmss(story["t_start_ms"] if k == 0 else story["t_end_ms"]),
                font=f_s, fill="#b00020" if k else "#333")
    meta = f"时长 {story['dur_s']:.0f}s | 镜头 {story['n_shots']} | 导语 {'是' if sem.get('is_lead_in') else '否'} | 人物: {(sem.get('who') or '?')[:24]}"
    dr.text((12, 352), meta, font=f_s, fill="#333")
    dr.text((12, 378), f"何事: {(sem.get('what') or '?')[:52]}", font=f_s, fill="#333")
    img.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("mode", nargs="?", choices=["boundaries", "stories"], default="boundaries")
    ap.add_argument("--at", type=int)
    ap.add_argument("--window", type=float, default=1.5)
    ap.add_argument("--video")
    a = ap.parse_args()
    d, video, sig, shots, stories, asr, catalog = load(a.run_dir, a.video)
    if not video.exists():
        sys.exit(f"video not found: {video}")
    qa = d / "qa"; qa.mkdir(exist_ok=True)
    vr = VideoReader(video)
    if a.at is not None:
        draw_boundary(qa / f"at_{a.at}.png", a.at, a.window, vr, sig, shots, stories, asr, catalog)
        print(qa / f"at_{a.at}.png")
    elif a.mode == "boundaries":
        outs = []
        for i in range(1, len(stories)):
            at = stories[i]["t_start_ms"]
            p = qa / f"bound_S{i - 1:02d}_S{i:02d}_{at}.png"
            draw_boundary(p, at, a.window, vr, sig, shots, stories, asr, catalog)
            outs.append(p)
        print(f"{len(outs)} boundary PNGs -> {qa}")
    else:
        for i, st in enumerate(stories):
            draw_story_card(qa / f"story_{i:02d}.png", st, i, vr, catalog)
        print(f"{len(stories)} story cards -> {qa}")


if __name__ == "__main__":
    main()
