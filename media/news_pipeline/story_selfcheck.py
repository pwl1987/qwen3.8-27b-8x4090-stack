#!/usr/bin/env python3
"""story_selfcheck.py v5 — 语义候选层（批次I P2a）.

架构位置：Evidence Fusion 的语义臂。**LLM 只回答"边界是否存在"并引用候选 ID，
绝不输出帧号**（边界主权：canonical frame 唯一由 frame_refine.refine 产生）。

输入：e2e/asr.json, e2e/asr_words.json(优先), stories_v2.json, audio16k.wav,
      signal_candidates.json(OCR事件/静音谷), shots_frame.json(镜头帧轨), anchor 声纹库
输出：semantic_candidates.json — 每候选 {cand_id, t_ms, sources, features{正/负证据},
      llm{forward,backward,decision,reason}, story_state}

v4→v5：
  - 金标外置 gt_bounds.json；F1 移至 P4 评测层，本层不算分
  - 候选 = v2 边界 ∪ 主播声跨度起点 ∪ OCR 题花事件 ∪ 长条目(>long_story_scan_s)内部扫描
  - 上下文 = 词级 ±llm_window_s（弃 15s 分片 → 无跨界污染，v2 胡并事故根因）
  - 2B 单调用双向判定：forward_new_event 与 backward_same_story 不一致 = AMBIGUOUS
  - 负证据显式记账（same_title/same_speaker/mid_word/no_acoustic_break）
funasr-env 运行；2B 服务默认 :8010（batch-I-config semantic 段）。
"""
import argparse
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, "/data/tools")
from frame_refine import has_cjk, is_person_chyron, _title_class  # 纯函数，无重依赖

CONFIG = "/data/tools/batch-I-config.json"
CAMPPLUS = "/data/models/modelscope/models/damo--speech_campplus_sv_zh-cn_16k-common/snapshots/master"
MODEL_27B = "qwen3.8-27b"
VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
VOICE_LIB = "/data/datasets/voice_library/anchor_centroids.npz"
GAP_S = 3.0        # 跨度前非主播间隔 ≥ 此值才算新开场
SPAN_MIN_S = 3.0

BIDIR_PROMPT = """你在质检电视新闻的条目切分。判断候选切点前后两段是否属于同一个新闻条目（同一次报道的对象）。
判定：同一事件/同一会议/同一次活动的不同阶段 = 同一条；两个不同地点、不同参与人、不同事件 = 两条；主播提要串词与其引出的条目 = 同一条；演播室口播导语与其引出的现场报道 = 同一条。注意：两段标题若是同一事件的不同措辞（全称/简称/语序/来源措辞差异）仍算同一条；你判断的是新闻事件本身，不是标题字面。
只输出 JSON：{{"same_event": true/false, "reason": "一句话"}}

A段（切点前）：{ta}
B段（切点后）：{tb}"""


def chat(port, model, prompt, max_tokens, seed=20260906, base=None):
    """base=None → 本地 :port vLLM；'27b' → 本机 127.0.0.1:8000 qwen3.8-27b（思考型，
    max_tokens 须 ≥1024 否则思考耗尽额度正文为空；content 字段即正文，忽略 reasoning_content）。"""
    url = f"http://127.0.0.1:8000/v1/chat/completions" if base == "27b" \
        else f"http://127.0.0.1:{port}/v1/chat/completions"
    last = None
    for attempt in range(5):  # transport ∧ schema 双重重试；27B LB 副本漂移期有间歇 500
        time.sleep((0, 5, 15, 30, 60)[attempt])
        try:
            # peg-native 500 = 服务端拒绝畸形/截断输出：重试时翻倍 token 预算并小幅升温扰动
            r = requests.post(url, json={
                "model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0 if attempt < 2 else 0.3,
                "max_tokens": max_tokens * (2 ** min(attempt, 2)), "seed": seed,
            }, timeout=300)
            if r.status_code != 200:
                last = f"HTTP{r.status_code}: {r.text[:200]}"
                time.sleep((0, 5, 15, 30, 60)[attempt])
                continue
            m = r.json()["choices"][0]["message"]
            out = parse_json(m.get("content") or "")
            if out is None and base == "27b":  # 思考型：思考耗尽额度时正文空，兜底从思考里提
                out = parse_json(m.get("reasoning_content") or "")
            if out is not None and "same_event" in out:
                return out
            last = f"schema 不符: {str(out)[:80]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(2)
    raise SystemExit(f"LLM 判定失败（transport∧schema）: {last}")


def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def anchor_spans(wav16k, sim_th, log):
    """VAD→CAM++→主播验证 → 主播声跨度 [[start_s,end_s],...]（同 v4，确定性）。"""
    from funasr import AutoModel
    vad = AutoModel(model=VAD_MODEL, disable_update=True, device="cpu", disable_pbar=True)
    segs = vad.generate(input=wav16k)[0]["value"]
    segs = [(s / 1000, e / 1000) for s, e in segs if e - s >= 3000]
    sv = AutoModel(model=CAMPPLUS, disable_update=True, device="cpu", disable_pbar=True)
    cents = {k: np.asarray(v) for k, v in np.load(VOICE_LIB).items()}
    w = wave.open(wav16k)
    sr = w.getframerate()
    import os
    import tempfile
    tmp = tempfile.mkdtemp()
    hits = []
    for i, (s, e) in enumerate(segs):
        p = f"{tmp}/s{i:04d}.wav"
        w.setpos(int(s * sr)); fr = w.readframes(int((e - s) * sr))
        with wave.open(p, "wb") as ow:
            ow.setnchannels(1); ow.setsampwidth(2); ow.setframerate(sr); ow.writeframes(fr)
        emb = np.asarray(sv.generate(input=p)[0]["spk_embedding"]).ravel()
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        best_k, best_v = max(((k, float(emb @ c)) for k, c in cents.items()), key=lambda x: x[1])
        if best_v >= sim_th:
            hits.append((s, e, best_k, round(best_v, 3)))
    w.close()
    spans = []
    for s, e, k, v in hits:
        if spans and s - spans[-1][1] < 1.5:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    spans = [sp for sp in spans if sp[1] - sp[0] >= SPAN_MIN_S]
    log["anchor_seg_hits"] = len(hits)
    log["anchor_spans"] = [[round(a, 1), round(b, 1)] for a, b in spans]
    return spans


# ── 上下文材料 ──────────────────────────────────────────────────
def load_words(d: Path):
    wf = d / "e2e" / "asr_words.json"
    if wf.exists():
        return [(w["s_ms"], w["e_ms"], w["w"]) for w in json.loads(wf.read_text())["words"]], True
    segs = json.loads((d / "e2e" / "asr.json").read_text())["segments"]
    words = []
    for s in segs:  # 退化：段内线性插值（15s 栅格，仅兜底）
        txt = s["text"]
        dur = s["e_ms"] - s["s_ms"]
        per = dur / max(len(txt), 1)
        for i, ch in enumerate(txt):
            words.append((s["s_ms"] + int(i * per), s["s_ms"] + int((i + 1) * per), ch))
    return words, False


def text_in(words, t0, t1, cap=160):
    return "".join(w for a, b, w in words if t0 <= a < t1)[:cap]


def title_timeline(d: Path, cfg):
    """每秒内容题花集合（静态 UI 已滤）→ (title_at, dominant_title, content 列表, step_s)。"""
    j = json.loads((d / "ocr_sig.json").read_text())
    step_s = j["stride"] / json.loads((d / "siglip_dist.json").read_text())["fps"]
    ocfg = cfg["candidate"]["ocr"]
    texts = [set(t) for t in j["texts"]]
    doc = {}
    for s in texts:
        for t in s:
            doc[t] = doc.get(t, 0) + 1
    static = {t for t, c in doc.items() if c >= ocfg["static_doc_freq"] * len(texts)}
    content = [{t for t in s if t not in static and len(t) >= ocfg["min_title_len"] and has_cjk(t)}
               for s in texts]
    n = len(content)

    def title_at(t_ms, off_s):
        i = max(0, min(n - 1, int((t_ms / 1000 + off_s) / step_s)))
        c = content[i]
        return max(c, key=len) if c else ""

    def dominant_title(t0_ms, t1_ms):
        """区间主标题：人名字条滤除后出现最多的'真标题形态'文本（含 ：/“/、 或 ≥12 字），
        否则空串（交由 v2 catalog 标题兜底——v4 已验证主场）。"""
        from collections import Counter
        cnt = Counter()
        for i in range(max(0, int(t0_ms / 1000 / step_s)), min(n, int(t1_ms / 1000 / step_s) + 1)):
            for t in content[i]:
                if not is_person_chyron(t, ocfg):
                    cnt[t] += 1
        for t, _ in cnt.most_common(5):
            if len(t) >= 12 or any(ch in t for ch in (chr(65306), chr(8220), chr(8221), chr(12289))):
                return t
        return ""
    return title_at, dominant_title


def is_anchor(spans, t_ms):
    return any(s * 1000 <= t_ms < e * 1000 for s, e in spans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--model")
    ap.add_argument("--sim", type=float, default=0.55)
    ap.add_argument("--no-llm", action="store_true", help="只出确定性候选与特征，跳过 LLM（调试）")
    ap.add_argument("--llm27b", action="store_true", help="合并 pass 用本机 27B 终审（计划既定升级通道）")
    a = ap.parse_args()
    cfg = json.loads(Path(CONFIG).read_text())
    scfg = cfg["semantic"]
    port = a.port or scfg["port"]
    model = a.model or scfg["model"]
    d = Path(a.run_dir)

    words, word_level = load_words(d)
    v2 = json.loads((d / "stories_v2.json").read_text())["stories"]
    v2_bounds = [s["t_start_ms"] for s in v2[1:]]
    end_ms = v2[-1]["t_end_ms"]
    sig = json.loads((d / "signal_candidates.json").read_text())
    ocr_ev = sig["ocr_events"]
    valleys = [v["t_ms"] for v in sig["silence_valleys"]]
    shots_f = json.loads((d / "shots_frame.json").read_text())
    fps = shots_f["fps"]["num"] / shots_f["fps"]["den"]
    shot_bounds_ms = [round(f * 1000 / fps) for f in shots_f["start_frames"]]
    title_at, dominant_title = title_timeline(d, cfg)
    W = cfg["story"]["llm_window_s"][1] * 1000
    lead_ms_cfg = int(cfg["story"]["lead_in_title_window_s"] * 1000)

    log = {"model": model, "port": port, "word_level": word_level}

    # ── 1) 确定性候选 ──
    spans = anchor_spans(str(d / "audio16k.wav"), a.sim, log)
    span_starts = []
    for s, e in spans:
        prev_end = max((b for b, _ in spans if b < s), default=0)
        if s - prev_end >= GAP_S:
            span_starts.append(s)

    cands = {}  # t_ms -> set(sources)
    for b in v2_bounds:
        cands.setdefault(b, set()).add("v2_prior")
    for s in span_starts:
        t = int(s * 1000)
        cands.setdefault(t, set()).add("anchor_voice")
    # 题花级候选：仅 highconf + 首次出现 + 非提要区（0904 实证：provisional=字幕闪烁噪声、
    # 提要区题花=预告蒙太奇、同名重加=题花闪烁；亲清@296 重加曾致 FP）
    recap_ms = int(cfg["story"]["recap_zone_s"] * 1000)
    seen_titles = set()
    for ev in ocr_ev:
        if (ev["change_type"] != "addition" or ev["t_ms"] < recap_ms
                or ev["tier"] != "highconf" or not _title_class(ev, cfg["candidate"]["ocr"])):
            continue
        if not any(t not in seen_titles for t in ev["new_titles"]):
            continue
        seen_titles.update(ev["new_titles"])
        cands.setdefault(ev["t_ms"], set()).add("ocr_highconf_addition")
    # 长条目内部扫描：>long_story_scan_s 的 v2 条目内，取"静音谷∧镜头边"复合候选（上限/条目）
    scan_s = cfg["story"]["long_story_scan_s"]
    n_internal = 0
    for i, st in enumerate(v2):
        if (st["t_end_ms"] - st["t_start_ms"]) / 1000 <= scan_s:
            continue
        inner = []
        for t in valleys:
            if not (st["t_start_ms"] + 5000 < t < st["t_end_ms"] - 5000):
                continue
            near_shot = any(abs(t - sb) <= cfg["candidate"]["windows_ms"]["silence"] for sb in shot_bounds_ms)
            if near_shot:
                inner.append(t)
        merged = []
        for t in inner:  # 5s 内去重
            if merged and t - merged[-1] < 5000:
                continue
            merged.append(t)
        for t in merged[:8]:
            if all(abs(t - c) > 5000 for c in cands):
                cands.setdefault(t, set()).add("internal_scan")
                n_internal += 1

    log["span_starts_s"] = [round(s, 1) for s in span_starts]
    log["n_internal_scan"] = n_internal

    # ── 1.5) LLM 前去重：≤2s 内多源候选合并（代表 t=最早，sources 并集）──
    merged = {}
    for t in sorted(cands):
        keys = sorted(merged)
        if keys and t - keys[-1] <= 2000:
            merged[keys[-1]] |= cands[t]
        else:
            merged[t] = set(cands[t])
    cands = merged
    log["n_candidates"] = len(cands)

    # ── 2) 特征（正/负证据，确定性，无 LLM）──
    out = []
    for t in sorted(cands):
        src = sorted(cands[t])
        near_ocr = [e for e in ocr_ev if abs(e["t_ms"] - t) <= cfg["candidate"]["windows_ms"]["ocr_prior"]]
        add_ev = [e for e in near_ocr if e["change_type"] in ("addition", "replacement")]
        drop_ev = [e for e in near_ocr if e["change_type"] == "drop"]
        tl, tr = title_at(t, -5), title_at(t, 5)
        anchor_start_near = any(abs(t - s * 1000) <= 1500 for s in span_starts)
        valley_near = any(abs(t - v) <= cfg["candidate"]["windows_ms"]["silence"] for v in valleys)
        mid_word = any(s2 < t - cfg["semantic"]["veto_midword_ms"] and e2 > t + cfg["semantic"]["veto_midword_ms"]
                       for s2, e2, _ in words)
        features = {
            "title_change": (add_ev[0]["tier"] if add_ev else None),
            "title_dropped": bool(drop_ev),
            "same_title": bool(tl and tl == tr),
            "speaker_change": anchor_start_near,
            "same_speaker": is_anchor(spans, t - 2000) and is_anchor(spans, t + 2000),
            "acoustic_break": valley_near,
            "no_acoustic_break": not valley_near,
            "mid_word": mid_word,
            "titles": [tl[:30], tr[:30]],
        }
        out.append({"cand_id": f"C{len(out):03d}", "t_ms": t, "sources": src, "features": features})

    # ── 3) LLM 顺序合并 pass（v4 已验证架构 F1 0.88；升级点=题花+词级干净上下文）──
    #    v5 教训留痕：纯双向双问 62/97 AMBIGUOUS、证据分级免 LLM 的 speculative 路线
    #    实测 F1 0.20/0.21 崩盘——规则未经验证不得取代已验证架构（数据驱动纪律）。
    cat = []
    if (d / "e2e" / "catalog.jsonl").exists():
        cat = [json.loads(l) for l in open(d / "e2e" / "catalog.jsonl")]
    elif (d / "e2e" / "semantic.json").exists():
        # 无 catalog 的新一期：semantic.json(v2 口径) 提供标题/what（合并判定的关键上下文——
        # 0905 零 catalog 实测 20/20 全留崩盘，语义层是架构依赖非可选项）
        for s in json.loads((d / "e2e" / "semantic.json").read_text()):
            try:
                sem = json.loads(s["result"])
            except (json.JSONDecodeError, TypeError):
                sem = {}
            cat.append({"t_start_ms": s["t_start_ms"], "t_end_ms": s["t_end_ms"],
                        "semantic": {"title": sem.get("title", ""), "what": sem.get("what", "")}})

    # 题花级 addition 事件（字条滤除，供 B 侧提示）
    title_add_events = [e for e in ocr_ev
                        if e["change_type"] == "addition" and _title_class(e, cfg["candidate"]["ocr"])]
    ocr_bounds = {t for t, s in cands.items() if any(x.startswith("ocr_") for x in s)}

    def side_desc(t0, t1):
        """v4 实证 span 级格式（F1 0.88）：catalog 标题+what + span 开头（词级）。
        唯一增强：B 侧 span 起点==题花候选边界时以事件标题提示并弃 what（240 修复；
        其他 span 一律 v4 原样——A 侧最近题花/tail-head 均实证有毒性）。"""
        mid = (t0 + t1) // 2
        c = next((c for c in cat if c["t_start_ms"] <= mid < c["t_end_ms"]), None)
        title = c["semantic"]["title"] if c else ""
        what = c["semantic"].get("what", "")[:60] if c else ""
        if t0 in ocr_bounds:
            ev = next((e for e in title_add_events if abs(e["t_ms"] - t0) <= 8000), None)
            if ev:
                title, what = f"{ev['new_titles'][0]}（画面题花）", ""
        head = text_in(words, t0, min(t0 + W, t1), cap=110)
        return f"「{title}」{what}｜开头: {head}"

    def merge_vote(t0, t1, t2):
        """27B 终审=单调用（计划既定升级通道）；2B 时 k=3 投票（temp=0 下 seed 无效已实证，
        投票仅是防御；2B 对清晰案例跨 run 翻转的根治=升级 27B）。平票保守=合并。"""
        ta, tb = side_desc(t0, t1), side_desc(t1, t2)
        prompt = BIDIR_PROMPT.format(ta=ta, tb=tb)
        if a.llm27b:
            same = bool(chat(port, MODEL_27B, prompt, 4096, base="27b")["same_event"])
            return same, ta, tb, [same]
        votes = [bool(chat(port, model, prompt, scfg["max_tokens"], seed=s)["same_event"])
                 for s in (20260906, 20260907, 20260908)]
        return votes.count(True) >= 2, ta, tb, votes

    bounds = sorted(cands)
    merges = []
    if not a.no_llm:
        i = 0
        spans_list = [0] + bounds
        while i < len(spans_list) - 1:
            t0, t1 = spans_list[i], spans_list[i + 1]
            t2 = spans_list[i + 2] if i + 2 < len(spans_list) else end_ms
            same, ta, tb, votes = merge_vote(t0, t1, t2)
            merges.append({"t_ms": t1, "same_event": same, "votes": votes,
                           "ta": ta[:70], "tb": tb[:70]})
            print(f"{'合' if same else '留'} @{t1/1000:7.1f}s votes={votes} A[{ta[:34]}] B[{tb[:34]}]")
            if same:  # 合并 → 移除边界并重判 (A vs C)，同 v4
                bounds.remove(t1)
                spans_list = [0] + bounds
                continue
            i += 1
    kept_bounds = bounds

    stats = {"n_candidates": len(out), "n_bounds_in": len(bounds),
             "n_bounds_out": len(kept_bounds), "n_merges": sum(1 for m in merges if m["same_event"])}
    log["stats"] = stats
    result = {"schema_version": 2, "model": model, "port": port,
              "candidates": out, "merges": merges,
              "bounds_in": bounds, "bounds_out": kept_bounds, "log": log}
    (d / "semantic_candidates.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    print(f"semantic_candidates.json: {stats} | word_level={word_level} internal_scan={n_internal}")


if __name__ == "__main__":
    main()
