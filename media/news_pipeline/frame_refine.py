#!/usr/bin/env python3
"""frame_refine.py — 批次I P1 检测层（确定性，零 LLM；P2 判别/裁决见 refine 子命令）.

子命令：
  sig        全片单次原生解码（PyAV, CPU）→ frame_sig.json：双通道帧差(直方图相关+像素MAD)
             + 每帧亮度 + 10ms hop 细粒度 RMS。VFR/时长不一致 → fail（framedomain 契约）。
  candidates 信号候选：MAD 局部归一化异常度 → 硬切峰(去抖聚类)；渐变三模式(亮度坡 fade /
             像素平台 dissolve / siglip 嵌入抬升复用)；OCR 底部条三级(provisional/highconf)；
             静音谷(细 RMS 局部极小) → signal_candidates.json
  shottrack  既有 shots.json 边界(0.5s 栅格) → 帧域吸附(±shot_prior 窗内最高 z 峰，
             单调+refractory 约束) → shots_frame.json（帧域闭包）

全程确定性：同输入同代码双跑 → 产物 SHA256 一致（P1 gate）。
pyav-env 运行：/data/tools/pyav-env/bin/python frame_refine.py <run_dir> <cmd>
"""
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/tools")
import framedomain as fd

CONFIG = "/data/tools/batch-I-config.json"


def load_cfg():
    return json.loads(Path(CONFIG).read_text())


def resolve_video(d: Path) -> Path:
    v = d.parent.parent.parent / "news_videos" / f"linyi_news_{d.name.split('_')[-1]}.mp4"
    if not v.exists():
        raise SystemExit(f"源视频不存在: {v}")
    return v


def stream_domain(d: Path, cfg):
    v = resolve_video(d)
    meta = fd.probe_stream(v)
    dom = fd.check_stream_meta(meta, cfg["frame_domain"]["dur_tol_s"], cfg["frame_domain"]["av_tol_s"])
    return v, dom


# ════════════════════════ sig：全片单次解码 ════════════════════════
def cmd_sig(d: Path, cfg):
    import av
    t0 = time.time()
    video, dom = stream_domain(d, cfg)
    n, num, den = dom["n_frames"], dom["fps_num"], dom["fps_den"]
    sc = cfg["signal"]["frame_sig"]
    W = sc["gray_downscale"]
    H = W * 9 // 16

    container = av.open(str(video))
    vs = container.streams.video[0]
    hist_corr, pixel_mad, luma = [], [], []
    prev_gray = prev_hist = None
    count = 0
    for frame in container.decode(vs):
        gray = frame.reformat(width=W, height=H, format="gray").to_ndarray()
        hist = np.histogram(gray, bins=sc["hist_bins"], range=(0, 256))[0].astype(np.float64)
        luma.append(round(float(gray.mean()), 2))
        if prev_gray is not None:
            a, b = prev_hist - prev_hist.mean(), hist - hist.mean()
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            hist_corr.append(round(float(a @ b / (na * nb + 1e-12)), 5))
            pixel_mad.append(round(float(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).mean()), 3))
        prev_gray, prev_hist = gray, hist
        count += 1
    container.close()
    if count != n:
        if abs(count - n) > 2:
            raise fd.FrameDomainError(f"解码帧数 {count} != 流声明 {n}（VFR 嫌疑，拒绝）")
        # 广播 MP4 末帧容器元数据常见 ±1 帧漂移：帧域以实际可解码数为 canonical
        print(f"[warn] 容器 nb_frames={n} 与实际解码 {count} 差 {n-count} 帧，以 {count} 为 canonical")
        n = count
        dom["n_frames"] = n
        dom["duration_ms"] = fd.frame_to_ms(n, num, den)

    # 音频：run_dir/audio16k.wav（stage2 产物）10ms hop RMS
    hop_ms = sc["audio_hop_ms"]
    w = wave.open(str(d / "audio16k.wav"))
    sr = w.getframerate()
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    w.close()
    hop = int(sr * hop_ms / 1000)
    usable = (len(pcm) // hop) * hop
    rms = np.sqrt((pcm[:usable].reshape(-1, hop) ** 2).mean(axis=1)) / 32768.0

    out = {
        "schema_version": 1,
        "video": {"path": str(video), "fps_num": num, "fps_den": den,
                  "n_frames": n, "duration_ms": dom["duration_ms"]},
        "audio": {"sample_rate": sr, "hop_ms": hop_ms, "n_hops": int(len(rms))},
        "signals": {
            "hist_corr": hist_corr,          # 长度 n-1；1=同 0=异
            "pixel_mad": pixel_mad,          # 长度 n-1；灰度均值绝对差(0-255)
            "luma": luma,                    # 长度 n；8bit 灰度均值
            "rms": [round(float(x), 5) for x in rms],
        },
        "granularity": {"frame_diff_frames": 1, "rms_ms": hop_ms},
        "source_fingerprint": fd.fingerprint(video),
        "decoder": {"library": "pyav", "version": av.__version__},
    }
    (d / "frame_sig.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    z = hardcut_z(out, cfg)
    print(f"frame_sig.json: {n}帧 {len(rms)}rms-hop 解码 {time.time()-t0:.1f}s | "
          f"硬切峰(z>={cfg['candidate']['peak_min_z']}) {len(find_peaks(z, cfg))} 个")


# ════════════════════════ 派生量（确定性） ════════════════════════
def local_z(x: np.ndarray, win_frames: int, eps=1e-6) -> np.ndarray:
    """局部中位数/MAD 归一异常度：x 相对局部正常状态有多异常（演播室/运动现场统一可比）。"""
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.asarray(x, dtype=np.float64)
    half = win_frames // 2
    pad = np.pad(x, (half, win_frames - 1 - half), mode="edge")
    w = sliding_window_view(pad, win_frames)
    med = np.median(w, axis=1)
    mad = np.median(np.abs(w - med[:, None]), axis=1)
    return (x - med) / (mad + eps)


def hardcut_z(fsig: dict, cfg) -> np.ndarray:
    """双通道融合硬切异常度：hist 异常与像素异常各取局部 z，逐点取 max。"""
    sc = cfg["signal"]["frame_sig"]
    fps = fsig["video"]["fps_num"] / fsig["video"]["fps_den"]
    win = int(sc["local_window_s"] * fps)
    z_h = local_z(1.0 - np.array(fsig["signals"]["hist_corr"]), win, sc["mad_epsilon"])
    z_p = local_z(np.array(fsig["signals"]["pixel_mad"]), win, sc["mad_epsilon"])
    return np.maximum(z_h, z_p)


def find_peaks(z: np.ndarray, cfg, min_z=None, guard=2):
    """确定性峰：z≥阈值 ∧ 局部最大(±guard，平顶取最小序号) → cluster_frames 内去抖合并
    (取 z 最大，平手取小)。返回 (帧号, z)，帧号=z序号+1=新内容首帧（[start,end) 语义）。"""
    min_z = cfg["candidate"]["peak_min_z"] if min_z is None else min_z
    cluster = cfg["candidate"]["cluster_frames"]
    zz = np.concatenate([[z[0]] * guard, z, [z[-1]] * guard])  # 边界补齐
    idx = [i for i in range(len(z))
           if z[i] >= min_z and z[i] >= max(zz[i:i + guard]) and z[i] >= max(zz[i + guard + 1:i + 2 * guard + 1])]
    groups, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > cluster:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return [(max(g, key=lambda i: (z[i], -i)) + 1, float(z[max(g, key=lambda i: (z[i], -i))]))
            for g in groups]


def fade_events(luma, fps, gcfg):
    """亮度单调坡（fade）：同号连续段，时长∈[min_s, max_ms]，累计变化≥30 灰阶。"""
    d = np.diff(np.asarray(luma, dtype=np.float64))
    ev, i, n = [], 0, len(d)
    while i < n:
        sgn = d[i] > 0
        j = i
        while j + 1 < n and (d[j + 1] > 0) == sgn and abs(d[j + 1]) > 0.3:
            j += 1
        run, total = j - i + 1, float(abs(d[i:j + 1].sum()))
        dur_s = run / fps
        if gcfg["luma_ramp_min_s"] <= dur_s <= gcfg["luma_ramp_max_ms"] / 1000 and total >= 30:
            ev.append({"frame": (i + j + 1) // 2, "type": "fade",
                       "dur_frames": run, "luma_delta": round(total, 1),
                       "direction": "in" if sgn else "out"})
        i = j + 1
    return ev


def dissolve_events(pixel_mad, fps, cfg, hardcut_frames):
    """像素异常平台（dissolve）：rolling 均值抬升持续≥ramp 帧，且无硬切峰在段内（防双记）。"""
    gcfg = cfg["signal"]["gradual"]
    k = gcfg["dissolve_ramp_frames"]
    x = np.asarray(pixel_mad, dtype=np.float64)
    rm = np.convolve(x, np.ones(k) / k, mode="same")
    z = local_z(rm, int(cfg["signal"]["frame_sig"]["local_window_s"] * fps))
    hot = z >= gcfg["siglip_elev_z"]
    ev, i, n = [], 0, len(hot)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hot[j + 1]:
            j += 1
        run = j - i + 1
        if run >= k:
            seg = range(max(0, i - 1), min(n, j + 1))  # 平台帧区间（帧号=下标+1 的切变沿）
            if not any(f in hardcut_frames for f in seg):
                best = max(seg, key=lambda f_: (z[f_], -f_))
                ev.append({"frame": best + 1, "type": "dissolve", "dur_frames": run,
                           "z": round(float(z[best]), 3)})
        i = j + 1
    return ev


def siglip_events(d: Path, cfg, hardcut_ms_set):
    """复用 shotseg 的 siglip_dist（0.5s 嵌入距离）：局部 z 抬升平台≥plateau_s → 渐变候选。"""
    p = d / "siglip_dist.json"
    if not p.exists():
        return []
    j = json.loads(p.read_text())
    step_s = 1.0 / j["fps"]
    x = np.asarray(j["dist"], dtype=np.float64)
    gcfg = cfg["signal"]["gradual"]
    win = int(cfg["signal"]["frame_sig"]["local_window_s"] / step_s) | 1
    z = local_z(x, win)
    need = max(2, int(gcfg["siglip_plateau_s"] / step_s))
    hot = z >= gcfg["siglip_elev_z"]
    ev, i, n = [], 0, len(hot)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hot[j + 1]:
            j += 1
        if j - i + 1 >= need:
            t_ms = int((i + j + 1) / 2 * step_s * 1000)
            # 段内任一 0.5s 采样已有硬切峰则跳过（防双记）
            if not any(abs(t_ms - h) < 750 for h in hardcut_ms_set):
                ev.append({"t_ms": t_ms, "type": "siglip_plateau",
                           "dur_s": round((j - i + 1) * step_s, 1),
                           "z": round(float(z[i:j + 1].max()), 3)})
        i = j + 1
    return ev


def silence_valleys(rms, hop_ms, cfg):
    """细 RMS 静音谷：3-hop 平滑后局部极小 且 < ratio×局部(±2s)中位数；300ms 内合并取最小。"""
    r = np.asarray(rms, dtype=np.float64)
    sm = np.convolve(r, np.ones(3) / 3, mode="same")
    win = max(3, int(4000 / hop_ms) | 1)
    from numpy.lib.stride_tricks import sliding_window_view
    half = win // 2
    pad = np.pad(sm, (half, win - 1 - half), mode="edge")
    med = np.median(sliding_window_view(pad, win), axis=1)
    thr = cfg["candidate"]["silence_rms_ratio"] * med
    idx = [i for i in range(1, len(sm) - 1)
           if sm[i] < thr[i] and sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1]]
    out, cur = [], []
    merge_hops = int(300 / hop_ms)
    for i in idx:
        if cur and i - cur[-1] > merge_hops:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return [{"t_ms": int(min(g, key=lambda i: (sm[i], -i)) * hop_ms),
             "rms": round(float(sm[min(g, key=lambda i: (sm[i], -i))]), 5)} for g in out]


def has_cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def ocr_events(d: Path, cfg):
    """底部条 OCR 三级事件：常驻 UI(文档频率≥static_doc_freq)滤除；内容文本新增即事件；
    provisional=1 采样，highconf=≥2 连续采样；记 persist/change_type。"""
    j = json.loads((d / "ocr_sig.json").read_text())
    frame_fps = json.loads((d / "siglip_dist.json").read_text())["fps"]  # frames 抽取帧率(2.0)
    step_s = j["stride"] / frame_fps
    ocfg = cfg["candidate"]["ocr"]
    texts = [set(t) for t in j["texts"]]
    n = len(texts)
    doc = {}
    for s in texts:
        for t in s:
            doc[t] = doc.get(t, 0) + 1
    static = {t for t, c in doc.items() if c >= ocfg["static_doc_freq"] * n}
    content = [{t for t in s if t not in static and len(t) >= ocfg["min_title_len"] and has_cjk(t)}
               for s in texts]
    events = []
    for i in range(1, n):
        new = content[i] - content[i - 1]
        if new:
            k = 1
            while i + k < n and new & content[i + k]:
                k += 1
            dropped = content[i - 1] - content[i]
            events.append({
                "t_ms": int(i * step_s * 1000),
                "tier": "highconf" if k >= ocfg["highconf_samples"] else "provisional",
                "new_titles": sorted(new)[:4],
                "change_type": "replacement" if dropped else "addition",
                "persist_s": round(k * step_s, 1),
                "static_ui_n": len(static),
            })
        else:
            dropped = content[i - 1] - content[i]
            if dropped:  # 消失型：条清除（如口播起、条目收尾），后续未复现才成立
                k = 1
                while i + k < n and not (dropped & content[i + k]):
                    k += 1
                events.append({
                    "t_ms": int(i * step_s * 1000),
                    "tier": "highconf" if k >= ocfg["highconf_samples"] else "provisional",
                    "dropped_titles": sorted(dropped)[:4],
                    "change_type": "drop",
                    "gone_s": round(k * step_s, 1),
                    "static_ui_n": len(static),
                })
    return events, len(static)


# ════════════════════════ candidates ════════════════════════
def cmd_candidates(d: Path, cfg):
    fsig = json.loads((d / "frame_sig.json").read_text())
    num, den, n = fsig["video"]["fps_num"], fsig["video"]["fps_den"], fsig["video"]["n_frames"]
    fps = num / den
    hop_ms = fsig["audio"]["hop_ms"]

    z = hardcut_z(fsig, cfg)
    hardcut = find_peaks(z, cfg)
    hardcut_frames = {f for f, _ in hardcut}
    hardcut_ms = [fd.frame_to_ms(f, num, den) for f, _ in hardcut]

    sc = cfg["signal"]["frame_sig"]
    fades = fade_events(fsig["signals"]["luma"], fps, cfg["signal"]["gradual"])
    disses = dissolve_events(fsig["signals"]["pixel_mad"], fps, cfg, hardcut_frames)
    siglips = siglip_events(d, cfg, hardcut_ms)
    valleys = silence_valleys(fsig["signals"]["rms"], hop_ms, cfg)
    ocr_ev, static_n = ocr_events(d, cfg)

    out = {
        "schema_version": 1,
        "fps": {"num": num, "den": den}, "n_frames": n,
        "hardcut_peaks": [{"frame": f, "z": round(zz, 3)} for f, zz in hardcut],
        "gradual": {"fade": fades, "dissolve": disses, "siglip_plateau": siglips},
        "silence_valleys": valleys,
        "ocr_events": ocr_ev,
        "counts": {"hardcut": len(hardcut), "fade": len(fades), "dissolve": len(disses),
                   "siglip": len(siglips), "valleys": len(valleys),
                   "ocr_total": len(ocr_ev),
                   "ocr_highconf": sum(1 for e in ocr_ev if e["tier"] == "highconf"),
                   "static_ui": static_n},
        "config_fingerprint": None,  # P3 由 pipeline 填（config 哈希参与失效）
    }
    (d / "signal_candidates.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"signal_candidates.json: {out['counts']}")


# ════════════════════════ shottrack：镜头边界落帧 ════════════════════════
def cmd_shottrack(d: Path, cfg):
    fsig = json.loads((d / "frame_sig.json").read_text())
    num, den, n = fsig["video"]["fps_num"], fsig["video"]["fps_den"], fsig["video"]["n_frames"]
    z = hardcut_z(fsig, cfg)
    peaks = find_peaks(z, cfg)
    shots = json.loads((d / "shots.json").read_text())["shots"]
    min_gap = int(round(cfg["shot"]["min_shot_s"] * num / den))
    w_ms = cfg["candidate"]["windows_ms"]["shot_prior"]

    snapped, used = [], set()
    last = 0
    for i, sh in enumerate(shots):
        if i == 0:
            snapped.append({"index": 0, "start_frame": 0, "snap": None})
            continue
        prior = fd.ms_to_frame_floor(sh["t_start_ms"], num, den)
        lo = fd.ms_to_frame_floor(max(0, sh["t_start_ms"] - w_ms), num, den)
        hi = fd.ms_to_frame_ceil(min(fsig["video"]["duration_ms"], sh["t_start_ms"] + w_ms), num, den)
        best = None
        for f, zz in peaks:  # 升序遍历，确定性：取 z 最大，平手取小帧号
            if f in used or f < lo or f > hi:
                continue
            if f - last < min_gap:  # refractory：不得紧贴上一镜头边界
                continue
            if best is None or (zz, -f) > (best[1], -best[0]):
                best = (f, zz)
        if best is None:
            frame = prior if prior - last >= min_gap else last + min_gap
            snapped.append({"index": i, "start_frame": frame, "snap": None})
        else:
            used.add(best[0])
            snapped.append({"index": i, "start_frame": best[0],
                            "snap": {"z": round(best[1], 3), "delta_frames": best[0] - prior}})
        last = snapped[-1]["start_frame"]
    frames = [s["start_frame"] for s in snapped] + [n]
    # 闭包整理：短段吸收——移除造成 <min_gap 段的共享边界（含原 shots.json 尾部 0.18s 段这类
    # 上游数据瑕疵）；帧 0 与 n 永不移除；每次移出即把两段合一，循环至全部满足 refractory。
    dropped = 0
    changed = True
    while changed:
        changed = False
        for i in range(1, len(frames) - 1):
            if frames[i + 1] - frames[i] < min_gap or frames[i] - frames[i - 1] < min_gap:
                frames.pop(i)
                dropped += 1
                changed = True
                break
    # 闭包校验（帧域单调 + refractory + 末尾=n）
    for a, b in zip(frames, frames[1:]):
        if b - a < min_gap:
            raise fd.FrameDomainError(f"shot 闭包违例: {a}→{b} < min_gap {min_gap}")
    n_snap = sum(1 for s in snapped if s["snap"])
    out = {"schema_version": 1, "fps": {"num": num, "den": den}, "n_frames": n,
           "min_gap_frames": min_gap, "n_shots": len(frames) - 1,
           "n_prior_shots": len(snapped), "merged_short_segments": dropped,
           "n_snapped": n_snap, "start_frames": frames}
    (d / "shots_frame.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"shots_frame.json: {len(frames)-1} 镜头（原 {len(snapped)}，吸收短段 {dropped}），"
          f"{n_snap} 边界吸附到帧差峰，闭包校验通过")


# ════════════════════════ refine：判别 + 全局裁决 + 落帧（P2b） ════════════════════════
def is_person_chyron(s: str, ocfg) -> bool:
    """人名字条分类（DYNAMIC_UI）：职务后缀 / 电视台记者署名模式。非字条 = CONTENT 题花。"""
    import re
    s = s.strip()  # OCR 串常带首尾空格，先剥再做后缀匹配
    if not s:
        return True
    if any(s.endswith(x) for x in ocfg["chyron_suffixes"]):
        return True
    return re.search(ocfg["chyron_station_re"], s) is not None


def _title_class(ev, ocfg):
    """OCR 事件是否题花级（新/消失标题里存在非人名字条文本）。"""
    keys = ev.get("new_titles") or ev.get("dropped_titles") or []
    return any(not is_person_chyron(t, ocfg) for t in keys)


def cmd_refine(d: Path, cfg):
    fsig = json.loads((d / "frame_sig.json").read_text())
    sig = json.loads((d / "signal_candidates.json").read_text())
    semp = d / "semantic_candidates.json"
    if not semp.exists():
        raise SystemExit("先跑 story_selfcheck.py（semantic_candidates.json 缺失）")
    sem = json.loads(semp.read_text())
    num, den, n = fsig["video"]["fps_num"], fsig["video"]["fps_den"], fsig["video"]["n_frames"]
    ocfg, scfg = cfg["candidate"]["ocr"], cfg["story"]
    ocr_ev = sig["ocr_events"]
    valleys = [v["t_ms"] for v in sig["silence_valleys"]]
    peaks = sig["hardcut_peaks"]  # [{frame, z}]，frame=新内容首帧
    shots_f = json.loads((d / "shots_frame.json").read_text())
    words = json.loads((d / "e2e" / "asr_words.json").read_text())["words"] \
        if (d / "e2e" / "asr_words.json").exists() else []

    title_add = [e for e in ocr_ev if e["change_type"] == "addition" and _title_class(e, ocfg)]
    title_drop = [e for e in ocr_ev if e["change_type"] == "drop" and _title_class(e, ocfg)]
    cands = sem["candidates"]
    refr = int(scfg["refractory_s"] * 1000)
    lead_ms = int(scfg["lead_in_title_window_s"] * 1000)
    recap_ms = int(scfg["recap_zone_s"] * 1000)

    # ── 1) 边界集：v4 架构（selfcheck 合并 pass）的输出即权威 story 边界 ──
    if sem.get("schema_version", 0) < 2:
        raise SystemExit("semantic_candidates.json 需 schema_version>=2（先跑 v5 selfcheck 合并 pass）")
    bounds = sorted(sem["bounds_out"])
    cluster_w = int(scfg["hysteresis_confirm_s"] * 1000)
    cand_by_t = {c["t_ms"]: c for c in sem["candidates"]}
    merges = {m["t_ms"]: m for m in sem.get("merges", [])}

    def near_cand(t):
        best = None
        for ct, c in cand_by_t.items():
            if abs(ct - t) <= cluster_w and (best is None or abs(ct - t) < abs(best[0] - t)):
                best = (ct, c)
        return best[1] if best else None

    # ── 2) 落帧：±1.5s 窗内最优切镜峰；J-cut 延展（声先画后：题花 ≤45s 后到）──
    shot_frames = shots_f["start_frames"]
    story_edges = []
    for t in bounds:
        win_lo, win_hi = t - cfg["candidate"]["windows_ms"]["shot_prior"], t + cfg["candidate"]["windows_ms"]["shot_prior"]
        best, best_s = None, None
        for p in peaks:
            f_ms = fd.frame_to_ms(p["frame"], num, den)
            if not (win_lo <= f_ms <= win_hi):
                continue
            s = (p["z"] + (1.5 if any(abs(f_ms - v) <= cfg["candidate"]["windows_ms"]["silence"]
                                      for v in valleys) else 0))
            if best is None or (s, -p["frame"]) > (best_s, -best):
                best, best_s = p["frame"], s
        title_t = next((e["t_ms"] for e in title_add if t - 2000 < e["t_ms"] <= t + lead_ms), None)
        if best is None and title_t:  # J-cut：导语声已起、切镜在题花上墙处
            win_hi = title_t + 1500
            for p in peaks:
                f_ms = fd.frame_to_ms(p["frame"], num, den)
                if not (win_lo <= f_ms <= win_hi):
                    continue
                s = (p["z"] + (1.5 if any(abs(f_ms - v) <= cfg["candidate"]["windows_ms"]["silence"]
                                          for v in valleys) else 0))
                if best is None or (s, -p["frame"]) > (best_s, -best):
                    best, best_s = p["frame"], s
        c = near_cand(t)
        if best is not None:
            frame, frame_conf, how = best, 0.9, f"peak"
        elif any(win_lo <= fd.frame_to_ms(f, num, den) <= win_hi for f in shot_frames):
            cf_ = [f for f in shot_frames if win_lo <= fd.frame_to_ms(f, num, den) <= win_hi]
            frame, frame_conf, how = min(cf_, key=lambda f: (abs(fd.frame_to_ms(f, num, den) - t), f)), 0.7, "shot_fallback"
        else:
            frame, frame_conf, how = fd.ms_to_frame_floor(t, num, den), 0.4, "floor"
        sem_conf = 0.75
        if c:
            if c["features"].get("title_change") == "highconf":  # provisional 题花=字幕闪烁，不给加成
                sem_conf += 0.10
            if "v2_prior" in c["sources"]:
                sem_conf += 0.05
        sem_conf = round(min(0.95, sem_conf), 3)
        story_edges.append({
            "frame": frame, "t_ms": t, "semantic_onset_ms": t, "title_t_ms": title_t,
            "semantic_confidence": sem_conf, "frame_confidence": frame_conf,
            "nearest_cand": c["cand_id"] if c else None,
            "how": how,
            "boundary_type": "story_title" if (title_t and abs(title_t - fd.frame_to_ms(frame, num, den)) <= 1500)
                             else "story_semantic",
        })

    story_edges.sort(key=lambda e: e["frame"])
    dropped_refr = []
    kept = []
    for e in story_edges:  # refractory 兜底
        if kept and fd.frame_to_ms(e["frame"], num, den) - fd.frame_to_ms(kept[-1]["frame"], num, den) < refr:
            if (e["semantic_confidence"] + e["frame_confidence"]
                    > kept[-1]["semantic_confidence"] + kept[-1]["frame_confidence"]):
                dropped_refr.append(kept[-1])
                kept[-1] = e
            else:
                dropped_refr.append(e)
            continue
        kept.append(e)
    story_edges = kept

    # 首尾最小语义单元：首/尾条目 < min_semantic_unit → 丢弃造成碎段的边（如片尾字幕卡）
    min_unit_ms = scfg["min_semantic_unit_s"] * 1000
    if story_edges and fd.frame_to_ms(story_edges[0]["frame"], num, den) < min_unit_ms:
        dropped_refr.append(story_edges.pop(0))
    if story_edges and (fd.frame_to_ms(n, num, den)
                        - fd.frame_to_ms(story_edges[-1]["frame"], num, den)) < min_unit_ms:
        dropped_refr.append(story_edges.pop())
    story_edges.sort(key=lambda e: e["frame"])
    keep_edges = []
    for e in story_edges:
        if keep_edges and fd.frame_to_ms(e["frame"], num, den) - fd.frame_to_ms(keep_edges[-1]["frame"], num, den) < refr:
            if e["semantic_confidence"] + e["frame_confidence"] > \
               keep_edges[-1]["semantic_confidence"] + keep_edges[-1]["frame_confidence"]:
                keep_edges[-1] = e
            continue
        keep_edges.append(e)
    story_edges = keep_edges

    # ── 4) 组装 canonical：boundaries / shots / stories（framedomain 契约） ──
    shot_set = {}
    for f in shot_frames:
        if 0 < f < n:
            shot_set[f] = fd.make_boundary(
                f"B{f:06d}", f, num, den, roles=["shot_start", "shot_end"],
                boundary_type="shot_hard_cut", transition_type="hard_cut",
                reason="shot track (P1 snap)",
                semantic_confidence=0.5, frame_confidence=0.95)
    edges = []
    for e in story_edges:
        f = e["frame"]
        if f in shot_set:  # 同一物理切点 → 同一 boundary_id，roles 合并（Canonical Boundary ID）
            shot_set[f]["roles"] = sorted(set(shot_set[f]["roles"]) | {"story_start", "story_end"})
            shot_set[f]["boundary_type"] = e["boundary_type"]
            shot_set[f]["semantic_confidence"] = e["semantic_confidence"]
            shot_set[f]["frame_confidence"] = max(shot_set[f]["frame_confidence"], e["frame_confidence"])
            shot_set[f]["semantic_onset_ms"] = e["semantic_onset_ms"]
            edges.append(shot_set[f])
        else:
            b = fd.make_boundary(
                f"BS{f:06d}", f, num, den, roles=["story_start", "story_end"],
                boundary_type=e["boundary_type"],
                transition_type="semantic" if e["how"] != "peak" else "hard_cut",
                reason=f"story-only edge (J-cut?) via {e['how']}",
                semantic_confidence=e["semantic_confidence"], frame_confidence=e["frame_confidence"],
                semantic_onset_ms=e["semantic_onset_ms"])
            edges.append(b)
    b_start = fd.make_boundary("B000000", 0, num, den, roles=["program_start", "story_start", "shot_start"],
                               boundary_type="shot_hard_cut", transition_type="none", reason="program start",
                               semantic_confidence=1.0, frame_confidence=1.0)
    b_end = fd.make_boundary(f"B{n:06d}", n, num, den, roles=["program_end", "story_end", "shot_end"],
                             boundary_type="shot_hard_cut", transition_type="none", reason="program end",
                             semantic_confidence=1.0, frame_confidence=1.0)
    boundaries = sorted([b_start, b_end] + list(shot_set.values()) + [b for b in edges if b not in shot_set.values()],
                        key=lambda b: b["frame"])
    bmap = {b["boundary_id"]: b for b in boundaries}

    shots, sf_list = [], sorted(shot_frames + [0, n])
    for i in range(len(sf_list) - 1):
        a, b = sf_list[i], sf_list[i + 1]
        if a == b:
            continue
        sid = f"sh{i:04d}"
        shots.append({"shot_id": sid,
                      "start_boundary": "B000000" if a == 0 else f"B{a:06d}",
                      "end_boundary": f"B{n:06d}" if b == n else f"B{b:06d}"})
    edge_frames = [0] + [e["frame"] for e in story_edges] + [n]
    edge_frames = sorted(set(edge_frames))
    stories, counts = [], {k: 0 for k in range(len(edge_frames) - 1)}
    for s in shots:
        a = bmap[s["start_boundary"]]["frame"]
        b = bmap[s["end_boundary"]]["frame"]
        dur = b - a
        for ti in range(len(edge_frames) - 1):
            s0, e0 = edge_frames[ti], edge_frames[ti + 1]
            inside = min(b, e0) - max(a, s0)
            if inside > 0 and inside * 2 >= dur:
                counts[ti] += 1
                break
    for ti in range(len(edge_frames) - 1):
        s0, e0 = edge_frames[ti], edge_frames[ti + 1]
        stories.append({"story_id": f"S{ti:02d}",
                        "start_boundary": "B000000" if s0 == 0 else (f"B{s0:06d}" if s0 in shot_set else f"BS{s0:06d}"),
                        "end_boundary": f"B{n:06d}" if e0 == n else (f"B{e0:06d}" if e0 in shot_set else f"BS{e0:06d}"),
                        "n_shots": counts[ti], "semantic": None})

    sf_final = {"schema_version": 1, "fps": {"num": num, "den": den}, "n_frames": n,
                "duration_ms": fd.frame_to_ms(n, num, den),
                "source_fingerprint": fsig["source_fingerprint"],
                "boundaries": boundaries, "shots": shots, "stories": stories}
    fd.validate_stories_final(sf_final, n, num, den)

    prov = {"schema_version": 1,
            "decision_architecture": "v4 merge-pass (F1 0.88) + word/title context + frame placement; "
                                      "speculative anchor-grading path rejected empirically (F1 0.20/0.21)",
            "config": {"refractory_s": scfg["refractory_s"],
                       "lead_in_title_window_s": scfg["lead_in_title_window_s"]},
            "story_edges": [{**{k: v for k, v in e.items()}} for e in story_edges],
            "dropped_refractory": dropped_refr,
            "merges": sem.get("merges", []),
            "n_bounds_in": len(bounds), "n_bounds_out": len(story_edges)}
    (d / "stories_final.json").write_text(json.dumps(sf_final, ensure_ascii=False, separators=(",", ":")))
    (d / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, separators=(",", ":")))
    print(f"stories_final.json: {len(stories)} stories / {len(shots)} shots / {len(boundaries)} boundaries | "
          f"bounds {len(bounds)}→{len(story_edges)}（refractory 弃 {len(dropped_refr)}）")
    for st in stories:
        b0, b1 = bmap[st["start_boundary"]], bmap[st["end_boundary"]]
        print(f"  {st['story_id']} [{b0['frame']:>6}-{b1['frame']:>6}] {round((b1['frame']-b0['frame'])*den/num,1):>6}s "
              f"shots={st['n_shots']} type={b1['boundary_type']} conf={b1['final_confidence']}")

    # 控制台速览 F1（正式数字归 P4 评测层）
    gt = d / "gt_bounds.json"
    if gt.exists():
        g = json.loads(gt.read_text())
        det = [fd.frame_to_ms(e["frame"], num, den) / 1000 for e in story_edges]
        hit = sum(1 for x in g["bounds_s"] if any(abs(x - y) <= g["tol_s"] for y in det))
        fp = [round(y, 1) for y in det if not any(abs(x - y) <= g["tol_s"] for x in g["bounds_s"])]
        p = hit / len(det) if det else 0
        r = hit / len(g["bounds_s"])
        print(f"[速览] P {p:.2f} R {r:.2f} F1 {2*p*r/(p+r) if p+r else 0:.2f} | FP={fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("cmd", choices=["sig", "candidates", "shottrack", "refine", "p1", "p2"])
    a = ap.parse_args()
    d = Path(a.run_dir)
    cfg = load_cfg()
    {"sig": lambda: cmd_sig(d, cfg),
     "candidates": lambda: cmd_candidates(d, cfg),
     "shottrack": lambda: cmd_shottrack(d, cfg),
     "refine": lambda: cmd_refine(d, cfg),
     "p1": lambda: (cmd_sig(d, cfg), cmd_candidates(d, cfg), cmd_shottrack(d, cfg)),
     "p2": lambda: cmd_refine(d, cfg)}[a.cmd]()


if __name__ == "__main__":
    main()
