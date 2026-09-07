#!/usr/bin/env python3
"""镜头分割·融合器：三信号 → shots.json（毫秒时间轴）。
信号①SigLIP 距离（主）+②能量/停顿（辅）+③字幕条变化（新闻强信号）。
用法: fuse_shots.py <workdir> [--z 2.5] [--min-shot 1.5] [--merge 0.5]"""
import sys, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("workdir")
ap.add_argument("--z", type=float, default=2.5)
ap.add_argument("--min-shot", type=float, default=1.5)
ap.add_argument("--merge", type=float, default=0.5)
a = ap.parse_args()
W = a.workdir

sig = json.load(open(f"{W}/siglip_dist.json"))
aud = json.load(open(f"{W}/audio_sig.json"))
ocr = json.load(open(f"{W}/ocr_sig.json"))
fps = sig["fps"]
D = np.array(sig["dist"])

# ① 主信号：z-score 自适应阈值 + 局部峰值（±0.5s 窗）
thr = D.mean() + a.z * D.std()
win = max(1, int(0.5 * fps))
vis_peaks = []
for i in range(1, len(D) - 1):
    if D[i] > thr and D[i] == max(D[max(0,i-win):i+win+1]):
        vis_peaks.append(round((i + 1) / fps, 2))  # 边界发生在 i 与 i+1 帧之间

# ②③ 辅信号：仅当落在无视觉峰值的空档时补充
cands = sorted(set(vis_peaks))
for t in aud["energy_peaks_s"] + aud["silence_mids_s"]:
    if not any(abs(t - c) < a.merge for c in cands):
        cands.append(round(t, 2))
for t in ocr["change_points_s"]:
    if not any(abs(t - c) < a.merge for c in cands):
        cands.append(round(t, 2))
cands.sort()

# 合并窗口内候选 + 最小镜头长度约束
merged = []
for t in cands:
    if merged and t - merged[-1] < a.merge:
        continue
    merged.append(t)
shots, prev = [], 0.0
for t in merged:
    if t - prev >= a.min_shot:
        shots.append(t)
        prev = t
# 起终：0 → 每边界，最后到视频时长（用末帧时间近似）
dur = sig["n_frames"] / fps
bounds = [0.0] + shots + [round(dur, 2)]

out = {"config": {"z": a.z, "min_shot_s": a.min_shot, "merge_s": a.merge, "fps": fps,
                  "vis_thr": round(float(thr), 5)},
       "n_shots": len(bounds) - 1,
       "shots": [{"index": i, "t_start_ms": int(bounds[i]*1000), "t_end_ms": int(bounds[i+1]*1000),
                  "dur_s": round(bounds[i+1]-bounds[i], 2),
                  "boundary_signals": {"vis": bounds[i+1] in vis_peaks if i < len(bounds)-2 else None}}
                 for i in range(len(bounds)-1)]}
json.dump(out, open(f"{W}/shots.json", "w"), ensure_ascii=False, indent=1)
print(f"fused: {out['n_shots']} shots (vis_peaks={len(vis_peaks)}, aud={len(aud['energy_peaks_s'])+len(aud['silence_mids_s'])}, ocr_changes={len(ocr['change_points_s'])})")
for s in out["shots"][:12]:
    print(f"  #{s['index']:03d} {s['t_start_ms']/1000:8.2f}s → {s['t_end_ms']/1000:8.2f}s ({s['dur_s']}s)")
