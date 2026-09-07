#!/usr/bin/env python
"""packed_md.py — catalog.jsonl → packed.md 紧凑 agent 可读层（借鉴 browser-use/video-use takes_packed.md 思想）.

机器 schema（catalog.jsonl，一期 ~75k token）压成人/LLM 可整读的 ~15-20KB 文本：
条目头（时间/标题/人物/画面构成/标识/人脸）+ `[mm:ss-mm:ss] 角色 句子` 转录行。
说话人标签由镜头类型推导（口播=主播 采访=同期 其余=旁白），词级对齐+声纹细化后可再生成 v2。

用法: python packed_md.py <run_dir>   # 读 e2e/catalog.jsonl → e2e/packed.md
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

SENT_SPLIT = re.compile(r"(?<=[。！？；])")


def mmss(ms):
    s = ms / 1000.0
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def word_lines(d, t0, t1, role_fn, shots):
    """asr_words.json 可用时：按 ~40 字组行，时间戳取词级真值。"""
    wf = d / "e2e" / "asr_words.json"
    if not wf.exists():
        return None
    words = [w for w in json.load(open(wf))["words"] if t0 <= w["s_ms"] < t1]
    lines, buf = [], []
    for w in words:
        buf.append(w)
        if len("".join(x["w"] for x in buf)) >= 40:
            lines.append((buf[0]["s_ms"], buf[-1]["e_ms"], "".join(x["w"] for x in buf)))
            buf = []
    if buf:
        lines.append((buf[0]["s_ms"], buf[-1]["e_ms"], "".join(x["w"] for x in buf)))
    return [f"[{mmss(a)}-{mmss(b)}] {role_fn(shots, a, b)} {txt}" for a, b, txt in lines]


def split_phrases(text, max_len=48):
    parts = [p for p in SENT_SPLIT.split(text) if p]
    out = []
    for p in parts:
        while len(p) > max_len:
            cut = p.rfind("，", 0, max_len)
            cut = cut if cut > max_len // 2 else max_len
            out.append(p[:cut])
            p = p[cut:].lstrip("，")
        out.append(p)
    return out


def dominant_role(shots, t0, t1):
    """该时间窗内时长占比最大的镜头类型 → 说话人标签。"""
    dur = Counter()
    for sh in shots:
        ov = min(sh["t_end_ms"], t1) - max(sh["t_start_ms"], t0)
        if ov > 0:
            dur[sh["vision"].get("type", "?")] += ov
    if not dur:
        return "旁白"
    tp = dur.most_common(1)[0][0]
    return {"口播": "主播", "采访": "同期"}.get(tp, "旁白")


def main():
    d = Path(sys.argv[1])
    # 批次I：优先 stories_final 口径（catalog_reindex 产物），回退旧 v2 catalog
    src = d / "e2e" / "catalog_final.jsonl"
    stories = ([json.loads(l) for l in open(src)] if src.exists()
               else [json.loads(l) for l in open(d / "e2e" / "catalog.jsonl")])
    ep = d.name.split("_")[-1]
    lines = [
        f"# 临沂新闻 {ep} · packed 视图（agent 可读层）",
        "",
        f"总时长 {mmss(stories[-1]['t_end_ms'])} | {len(stories)} 条目 | "
        f"{sum(len(s['shots']) for s in stories)} 镜头 | 转录行时间戳：词级真值（asr_words.json，缺失时退化为线性插值）",
        "说话人标签由镜头类型推导：口播=主播 采访=同期 其余=旁白",
        "",
        "## 条目目录",
    ]
    for i, s in enumerate(stories):
        sem = s["semantic"]
        lines.append(f"- S{i} [{mmss(s['t_start_ms'])}-{mmss(s['t_end_ms'])}] {sem['title']}")
    for i, s in enumerate(stories):
        sem = s["semantic"]
        tc = Counter(sh["vision"].get("type", "?") for sh in s["shots"])
        comp = "/".join(f"{k}{v}" for k, v in tc.most_common())
        lines += ["", f"## S{i} [{mmss(s['t_start_ms'])}-{mmss(s['t_end_ms'])}] {sem['title']}",
                  f"- 时长 {(s['t_end_ms'] - s['t_start_ms']) / 1000:.0f}s | {len(s['shots'])}镜头 | 画面: {comp}"
                  f" | 导语: {'含' if sem.get('is_lead_in') else '无'}",
                  f"- 谁: {sem.get('who', '?')} | 哪: {sem.get('where', '?')}"]
        signs, seen = [], set()
        for sh in s["shots"]:
            for sg in sh["vision"].get("signs", []):
                key = (sg["kind"], sg["text"])
                if key not in seen and sg["text"]:
                    seen.add(key)
                    signs.append(f'{sg["kind"]}「{sg["text"]}」')
        if signs:
            lines.append(f"- 标识: {' '.join(signs[:10])}")
        faces = sorted({f["person"] for sh in s["shots"] for f in sh.get("faces", []) if f["person"] != "未识别"})
        if faces:
            lines.append(f"- 人脸库命中: {', '.join(faces)}")
        wl = word_lines(d, s["t_start_ms"], s["t_end_ms"], dominant_role, s["shots"])
        if wl:
            lines += wl
        else:
            for seg in s["audio"]["transcript_segments"]:
                t0, t1 = seg["t_start_ms"], seg["t_end_ms"]
                role = dominant_role(s["shots"], t0, t1)
                phrases = split_phrases(seg["text"])
                span = max(t1 - t0, 1)
                cum = 0
                total = sum(len(p) for p in phrases)
                for p in phrases:
                    a = t0 + span * cum / total
                    cum += len(p)
                    b = t0 + span * cum / total
                    lines.append(f"[{mmss(a)}-{mmss(b)}] {role} {p}")
    out = d / "e2e" / "packed.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{out}  {out.stat().st_size / 1024:.1f}KB  {len(stories)} stories")


if __name__ == "__main__":
    main()
