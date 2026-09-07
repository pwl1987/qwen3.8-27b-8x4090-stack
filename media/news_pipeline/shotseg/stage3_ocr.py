#!/usr/bin/env python3
"""镜头分割·信号③：底部字幕条区域 OCR 文本变化（新闻条目切换强信号）。ocr-env 运行。
用法: stage3_ocr.py <workdir> [stride=2 即每2帧(默认fps2→每秒1帧)]"""
import sys, glob, json, time

workdir = sys.argv[1]
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 2
frames = sorted(glob.glob(f"{workdir}/frames/*.jpg"))
sel = frames[::stride]

from paddleocr import PaddleOCR
ocr = PaddleOCR(text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="PP-OCRv6_medium_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, enable_mkldnn=False)
from PIL import Image

texts = []
for i, f in enumerate(sel):
    img = Image.open(f)
    w, h = img.size
    strip = img.crop((0, int(h*0.75), w, h))  # 底部 25%
    import numpy as np
    r = ocr.predict(np.array(strip))
    words = sorted({t for p in r for t in p.get("rec_texts", []) if t.strip()})
    texts.append(words)
    if (i+1) % 30 == 0: print(f"[{i+1}/{len(sel)}]", flush=True)

def jac(a, b):
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    return len(A & B) / max(len(A | B), 1)

sims = [jac(texts[i], texts[i+1]) for i in range(len(texts)-1)]
fps = json.load(open(f"{workdir}/siglip_dist.json"))["fps"]
frame_t = [i * stride / fps for i in range(len(sel))]
changes = [round(frame_t[i+1], 2) for i in range(len(sims)) if sims[i] < 0.5]
json.dump({"stride": stride, "texts": texts, "sims": [round(s,3) for s in sims],
           "change_points_s": changes}, open(f"{workdir}/ocr_sig.json", "w"))
print(f"ocr done: {len(sel)} sampled, {len(changes)} change points")
