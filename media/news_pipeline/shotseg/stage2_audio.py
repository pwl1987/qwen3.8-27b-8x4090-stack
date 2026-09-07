#!/usr/bin/env python3
"""镜头分割·信号②：音频 RMS 能量差分 + 停顿点。stdlib+numpy（pyav-env）。
用法: stage2_audio.py <video> <workdir>"""
import sys, os, subprocess, wave, json
import numpy as np

video, workdir = sys.argv[1], sys.argv[2]
os.makedirs(workdir, exist_ok=True)
wav = f"{workdir}/audio16k.wav"
if not os.path.exists(wav):
    subprocess.run(["/data/tools/ffmpeg", "-y", "-loglevel", "error", "-i", video,
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)

w = wave.open(wav)
sr, n = w.getframerate(), w.getnframes()
x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0

win = int(sr * 0.02)  # 20ms RMS
m = len(x) // win
rms = np.sqrt((x[:m*win].reshape(m, win) ** 2).mean(axis=1))
t_rms = np.arange(m) * 0.02

# 1s 滑窗均值（对齐 0.5s 网格），相邻差分 → 能量骤变
step = int(0.5 / 0.02)
smooth = np.array([rms[max(0, i-step):i+step].mean() for i in range(0, m, step)])
t_s = np.arange(len(smooth)) * 0.5
jump = np.abs(np.diff(smooth))
thr = jump.mean() + 2.0 * jump.std()
peaks = [i for i in range(1, len(jump)-1)
         if jump[i] > thr and jump[i] >= jump[i-1] and jump[i] >= jump[i+1]]

# 静音停顿段（新闻换条特征：能量 < 0.1*全局均值 持续>0.3s）
sil_thr = rms.mean() * 0.1
silent = rms < sil_thr
sil_points = []
i = 0
while i < m:
    if silent[i]:
        j = i
        while j < m and silent[j]: j += 1
        if (j - i) * 0.02 >= 0.3:
            sil_points.append(round((i + j) / 2 * 0.02, 2))
        i = j
    else:
        i += 1

json.dump({"rms_grid_s": 0.5, "jump": [round(float(v), 5) for v in jump],
           "energy_peaks_s": [round(float(t_s[p+1]), 2) for p in peaks],
           "silence_mids_s": sil_points},
          open(f"{workdir}/audio_sig.json", "w"))
print(f"audio done: energy_peaks={len(peaks)} silence={len(sil_points)}")
