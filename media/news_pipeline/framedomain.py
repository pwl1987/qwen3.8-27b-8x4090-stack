#!/usr/bin/env python3
"""framedomain.py — 批次I P0：帧域换算与 canonical 边界数据结构（全项目唯一真源）.

冻结约定（批次I v5 规格，开工后不得边做边改）：
  1) 区间一律半开 [start_frame, end_frame)：duration_frames = end_frame - start_frame；
     相邻区间共享边界帧号 → 无重叠、无缝隙。
  2) fps 一律有理数 (fps_num, fps_den)；换算只有 ms_to_frame_floor / ms_to_frame_ceil
     两个入口（整数时间基），任何模块禁止自行 round / ×40ms 特例。
  3) frame 是 canonical，ms 是派生展示值（frame_to_ms 用 ceil，在 fps<1000
     守卫下可证明可逆，单测覆盖 25/1 与 30000/1001）；EDL 双域一致性以 frame 为准。
  4) VFR / r_frame_rate≠avg_frame_rate / 帧数×帧长≠时长 / 音视频时长不一致 →
     FrameDomainError（显式拒绝，绝不猜测）。
  5) 边界主权：canonical boundary 只能由 refine 阶段产生（make_boundary 仅供
     refine 与测试调用）；ASR/OCR/shotseg/LLM 只产 evidence/candidate。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

FFPROBE = "/data/tools/ffmpeg-deb/ffprobe"

BOUNDARY_TYPES = (
    "story_semantic",   # 语义切换（ASR/LLM 证据主导）
    "story_title",      # 题花/标题条切换（OCR 证据主导）
    "shot_hard_cut",    # 硬切
    "shot_dissolve",    # 溶解/渐变
    "shot_graphic",     # 图形/字幕动画包装
    "shot_camera",      # 机位/景别变化
)
TRANSITION_TYPES = ("hard_cut", "dissolve", "fade", "graphic", "semantic", "none")
BOUNDARY_ROLES = ("story_start", "story_end", "shot_start", "shot_end", "program_start", "program_end")


class FrameDomainError(Exception):
    """帧域契约违反——必须 fail-fast，不允许静默降级。"""


# ── 换算（仅此两入口） ────────────────────────────────────────────
def ms_to_frame_floor(ms: int, fps_num: int, fps_den: int) -> int:
    if ms < 0:
        raise FrameDomainError(f"负时间 {ms}ms 无帧")
    _check_fps(fps_num, fps_den)
    return (ms * fps_num) // (1000 * fps_den)


def ms_to_frame_ceil(ms: int, fps_num: int, fps_den: int) -> int:
    if ms < 0:
        raise FrameDomainError(f"负时间 {ms}ms 无帧")
    _check_fps(fps_num, fps_den)
    q, r = divmod(ms * fps_num, 1000 * fps_den)
    return q + (1 if r else 0)


def frame_to_ms(frame: int, fps_num: int, fps_den: int) -> int:
    """canonical→派生毫秒（ceil）。

    ceil 可证明可逆：fps<1000（_check_fps 守卫）时 ms/帧 step>1，
    ceil(true) ∈ (true, true+1] → ceil(true)/step ∈ (f, f+1) → floor 回帧恒为 f。
    floor 则在 true 贴近整数上方时丢帧（单测 30000/1001 帧3 反例实证）。
    25/1 下整除，ceil 与 floor 等价。
    """
    if frame < 0:
        raise FrameDomainError(f"负帧号 {frame}")
    _check_fps(fps_num, fps_den)
    q, r = divmod(frame * fps_den * 1000, fps_num)
    return q + (1 if r else 0)


def _check_fps(num: int, den: int):
    if num <= 0 or den <= 0:
        raise FrameDomainError(f"非法 fps {num}/{den}")
    # ms/帧 = 1000*den/num 必须 >1 才能保证 frame→ms→frame 可逆（fps<1000 的现实约束）
    if 1000 * den <= num:
        raise FrameDomainError(f"fps {num}/{den} ≥1000，frame↔ms 不可逆，不支持")


# ── 源流校验与指纹 ───────────────────────────────────────────────
def probe_stream(video) -> dict:
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-show_entries",
         "stream=r_frame_rate,avg_frame_rate,nb_frames,duration,width,height,sample_rate",
         "-of", "json", str(video)], text=True)
    streams = json.loads(out)["streams"]
    v = next(s for s in streams if s.get("nb_frames"))
    a = next((s for s in streams if s.get("sample_rate")), None)

    def frac(s):
        n, d = s.split("/")
        return int(n), int(d)

    r_num, r_den = frac(v["r_frame_rate"])
    a_num, a_den = frac(v["avg_frame_rate"])
    return {
        "video": str(video),
        "r_frame_rate": [r_num, r_den],
        "avg_frame_rate": [a_num, a_den],
        "nb_frames": int(v["nb_frames"]),
        "duration_s": float(v["duration"]),
        "width": v.get("width"), "height": v.get("height"),
        "audio_sample_rate": int(a["sample_rate"]) if a else None,
        "audio_duration_s": float(a["duration"]) if a and "duration" in a else None,
    }


def check_stream_meta(meta: dict, dur_tol_s: float = 0.05, av_tol_s: float = 0.5) -> dict:
    """CFR/时长契约校验，返回 {fps_num, fps_den, n_frames, duration_ms}；违规即抛。"""
    (rn, rd), (an, ad) = meta["r_frame_rate"], meta["avg_frame_rate"]
    if (rn, rd) != (an, ad):
        raise FrameDomainError(f"VFR 嫌疑：r_frame_rate {rn}/{rd} ≠ avg_frame_rate {an}/{ad}")
    _check_fps(rn, rd)
    expected = meta["nb_frames"] * rd / rn
    if abs(expected - meta["duration_s"]) > dur_tol_s:
        raise FrameDomainError(
            f"帧数×帧长 {expected:.3f}s ≠ 容器时长 {meta['duration_s']:.3f}s")
    adur = meta.get("audio_duration_s")
    if adur is not None and abs(adur - meta["duration_s"]) > av_tol_s:
        raise FrameDomainError(
            f"音视频时长不一致：video {meta['duration_s']:.3f}s vs audio {adur:.3f}s")
    return {"fps_num": rn, "fps_den": rd, "n_frames": meta["nb_frames"],
            "duration_ms": frame_to_ms(meta["nb_frames"], rn, rd)}


def fingerprint(path, strong: bool = False) -> dict:
    """源指纹分级：quick=size+mtime_ns；strong 追加 SHA256（发版/终验）。"""
    st = os.stat(path)
    fp = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if strong:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        fp["sha256"] = h.hexdigest()
    return fp


# ── canonical boundary（仅 refine 与测试可调用 make_boundary） ────
def make_boundary(bid: str, frame: int, fps_num: int, fps_den: int, *, roles,
                  boundary_type, transition_type, reason,
                  semantic_confidence: float, frame_confidence: float,
                  evidence: dict | None = None, semantic_onset_ms: int | None = None) -> dict:
    if frame < 0:
        raise FrameDomainError(f"{bid}: 负帧号")
    if not 0.0 <= semantic_confidence <= 1.0 or not 0.0 <= frame_confidence <= 1.0:
        raise FrameDomainError(f"{bid}: 置信度越界")
    if boundary_type not in BOUNDARY_TYPES:
        raise FrameDomainError(f"{bid}: 未知 boundary_type {boundary_type}")
    if transition_type not in TRANSITION_TYPES:
        raise FrameDomainError(f"{bid}: 未知 transition_type {transition_type}")
    for r in roles:
        if r not in BOUNDARY_ROLES:
            raise FrameDomainError(f"{bid}: 未知 role {r}")
    return {
        "boundary_id": bid,
        "frame": frame,
        "time_ms": frame_to_ms(frame, fps_num, fps_den),
        "roles": sorted(set(roles)),
        "boundary_type": boundary_type,
        "transition_type": transition_type,
        "reason": reason,
        "semantic_confidence": round(float(semantic_confidence), 4),
        "frame_confidence": round(float(frame_confidence), 4),
        "final_confidence": round((semantic_confidence + frame_confidence) / 2, 4),
        "semantic_onset_ms": semantic_onset_ms,
        "evidence": evidence or {},
    }


def validate_boundaries(boundaries: list, n_frames: int):
    """帧严格递增、id 唯一、首尾必须显式含 program_start/program_end（帧 0 与 n_frames）。"""
    ids = [b["boundary_id"] for b in boundaries]
    if len(ids) != len(set(ids)):
        raise FrameDomainError("boundary_id 重复")
    frames = [b["frame"] for b in boundaries]
    if frames != sorted(frames) or len(set(frames)) != len(frames):
        raise FrameDomainError("boundary 帧序非严格递增")
    if boundaries[0]["frame"] != 0 or "program_start" not in boundaries[0]["roles"]:
        raise FrameDomainError("缺少 frame-0 program_start 边界")
    if boundaries[-1]["frame"] != n_frames or "program_end" not in boundaries[-1]["roles"]:
        raise FrameDomainError(f"缺少 frame-{n_frames} program_end 边界")
    for b in boundaries:
        if b["boundary_type"] not in BOUNDARY_TYPES or b["transition_type"] not in TRANSITION_TYPES:
            raise FrameDomainError(f"{b['boundary_id']}: 枚举字段非法")


# ── stories_final schema gate ────────────────────────────────────
def validate_stories_final(sf: dict, n_frames: int, fps_num: int, fps_den: int):
    """全下游唯一边界真源的完整契约。任何违反 = validate 阶段红。

    结构：{schema_version, fps:{num,den}, n_frames, duration_ms, source_fingerprint,
           boundaries:[make_boundary...], shots:[{shot_id,start_boundary,end_boundary}],
           stories:[{story_id,start_boundary,end_boundary,n_shots,semantic}]}
    不变量：
      A shots 经边界引用无缝铺满全片（shot[i].end == shot[i+1].start，帧域闭包）
      B stories 相邻共享边边界（story 无缝覆盖），start/end 引用存在
      C story-only 边（J-cut，roles 不含 shot_*）：跨骑 shot 按时长多数派归一侧，
        Σn_shots == 总镜头数
      D n_shots 与多数派归属一致；story 首/末 shot 与 story 边重合（非 J-cut 时）
    """
    if sf.get("schema_version") != 1:
        raise FrameDomainError("schema_version != 1")
    if (sf["fps"]["num"], sf["fps"]["den"]) != (fps_num, fps_den) or sf["n_frames"] != n_frames:
        raise FrameDomainError("stories_final 帧域参数与源不符")
    bmap = {b["boundary_id"]: b for b in sf["boundaries"]}
    validate_boundaries(sf["boundaries"], n_frames)

    # A: shot 闭包
    shots = sf["shots"]
    for s in shots:
        if s["start_boundary"] not in bmap or s["end_boundary"] not in bmap:
            raise FrameDomainError(f"{s['shot_id']}: 引用不存在的边界")
        if bmap[s["start_boundary"]]["frame"] >= bmap[s["end_boundary"]]["frame"]:
            raise FrameDomainError(f"{s['shot_id']}: 起止帧倒挂")
    for s, t in zip(shots, shots[1:]):
        if s["end_boundary"] != t["start_boundary"]:
            raise FrameDomainError(f"shot 闭包断裂：{s['shot_id']}→{t['shot_id']}")
    if shots and (bmap[shots[0]["start_boundary"]]["frame"] != 0
                  or bmap[shots[-1]["end_boundary"]]["frame"] != n_frames):
        raise FrameDomainError("shot 轨未铺满 [0, n_frames)")

    # B: story 无缝覆盖
    stories = sf["stories"]
    for st in stories:
        if st["start_boundary"] not in bmap or st["end_boundary"] not in bmap:
            raise FrameDomainError(f"{st['story_id']}: 引用不存在的边界")
    for a, b in zip(stories, stories[1:]):
        if a["end_boundary"] != b["start_boundary"]:
            raise FrameDomainError(f"story 缝隙/重叠：{a['story_id']}→{b['story_id']}")
    if stories and (bmap[stories[0]["start_boundary"]]["frame"] != 0
                    or bmap[stories[-1]["end_boundary"]]["frame"] != n_frames):
        raise FrameDomainError("story 轨未铺满 [0, n_frames)")

    # C/D: story-only 边（J-cut，roles 不含 shot_*）时，跨骑 shot 按时长多数派归属
    #      （inside*2 >= dur；平局归左=先遍历到的条目）；核对 n_shots 记账
    counts = [0] * len(stories)
    for s in shots:
        a, b = bmap[s["start_boundary"]]["frame"], bmap[s["end_boundary"]]["frame"]
        dur = b - a
        for ti, st in enumerate(stories):
            s0, e0 = bmap[st["start_boundary"]]["frame"], bmap[st["end_boundary"]]["frame"]
            inside = min(b, e0) - max(a, s0)
            if inside > 0 and inside * 2 >= dur:
                counts[ti] += 1
                break
        else:
            raise FrameDomainError(f"{s['shot_id']}: 无法多数派归属任何条目")
    for ti, st in enumerate(stories):
        if st.get("n_shots") != counts[ti]:
            raise FrameDomainError(f"{st['story_id']}: n_shots 记账 {st.get('n_shots')} != 归属 {counts[ti]}")


def load_stories_final(path) -> dict:
    return json.loads(Path(path).read_text())
