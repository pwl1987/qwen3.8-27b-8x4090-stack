#!/usr/bin/env python3
"""align_words.py — e2e ASR 段级转录 → 词(字)级时间轴（Qwen3-ForcedAligner-0.6B，≤5min/块）。

读 run_dir/audio16k.wav + e2e/asr.json（15s 段级），按 ≤BLOCK_S 组块对齐，
输出 e2e/asr_words.json: {"words": [{"w","s_ms","e_ms"}...], "block_ms": [...]}
align-env 运行；--device cuda:0（须配 CUDA_VISIBLE_DEVICES）或 cpu。

用法: python align_words.py <run_dir> [--block-s 280] [--device cpu]
"""
import argparse
import json
import wave
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--block-s", type=int, default=280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-blocks", type=int, default=0, help="0=全部（冒烟用）")
    a = ap.parse_args()
    d = Path(a.run_dir)
    from qwen_asr import Qwen3ForcedAligner
    dev = a.device if a.device != "cuda" else "cuda:0"
    kw = {"device_map": dev} if dev.startswith("cuda") else {}
    al = Qwen3ForcedAligner.from_pretrained("/data/models/Qwen3-ForcedAligner-0.6B", **kw)

    asr = json.load(open(d / "e2e" / "asr.json"))["segments"]
    w = wave.open(str(d / "audio16k.wav"))
    sr = w.getframerate()
    total_ms = int(w.getnframes() / sr * 1000)

    blocks, cur, cur_t0 = [], [], None
    for s in asr:
        if cur_t0 is None or (s["t_end_ms"] - cur_t0) > a.block_s * 1000:
            if cur:
                blocks.append((cur_t0, cur))
            cur_t0, cur = s["t_start_ms"], []
        cur.append(s)
    if cur:
        blocks.append((cur_t0, cur))

    words, block_meta = [], []
    for bi, (t0, segs) in enumerate(blocks):
        if a.max_blocks and bi >= a.max_blocks:
            break
        t1 = min(segs[-1]["t_end_ms"], t0 + a.block_s * 1000, total_ms)
        w.setpos(int(t0 / 1000 * sr))
        audio = np.frombuffer(w.readframes(int((t1 - t0) / 1000 * sr)), dtype=np.int16).astype(np.float32) / 32768
        text = "".join(s["text"] for s in segs)
        if not text.strip():
            continue
        res = al.align((audio, sr), text, "Chinese")[0]
        n = 0
        for it in res.items:
            ws = getattr(it, "text", None) or getattr(it, "word", "")
            s_ms = t0 + int(getattr(it, "start_time", 0) * 1000)
            e_ms = t0 + int(getattr(it, "end_time", 0) * 1000)
            words.append({"w": ws, "s_ms": s_ms, "e_ms": e_ms})
            n += 1
        block_meta.append({"t0_ms": t0, "t1_ms": t1, "chars": len(text), "items": n})
        print(f"block {bi}: [{t0}-{t1}ms] {len(text)}字 → {n} items")
    w.close()
    # B5 后处理：零时长词拉平 ≥1 帧（25fps=40ms；帧域最短单元，w.start < w.end 保证）
    for _w in words:
        if _w["e_ms"] <= _w["s_ms"]:
            _w["e_ms"] = _w["s_ms"] + 40
    out = d / "e2e" / "asr_words.json"
    out.write_text(json.dumps({"words": words, "blocks": block_meta}, ensure_ascii=False))
    print(f"{out}: {len(words)} items")


if __name__ == "__main__":
    main()
