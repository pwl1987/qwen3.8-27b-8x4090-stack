#!/usr/bin/env python3
"""boundary_review.py — 批次I P4：±3s 边界回放合成图 + boundary_cards.md 解释卡.

每条 story 边界：±3s 七帧胶片条（红线=切点帧）+ 证据行（类型/双置信/来源/LLM 合并记录/
题花标题/声纹锚点）；解释卡由 provenance+merges 渲染，零额外计算。
pyav-env 运行。输出 review/boundary_*.png + boundary_cards.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/tools")
import framedomain as fd
from validate import FG  # 帧精确抓取

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def grab_str(g, t_s, fps):
    import av
    tol = 0.5 / fps
    g.ctx.seek(int(max(0.0, t_s - 1.5) / float(g.vs.time_base)), stream=g.vs, backward=True)
    for pkt in g.ctx.demux(g.vs):
        for f in pkt.decode():
            if f.time is None:
                continue
            if abs(f.time - t_s) <= tol:
                return f.reformat(width=240, height=135, format="rgb24").to_ndarray()
            if f.time > t_s + tol + 1.0:
                break
    raise RuntimeError(f"no frame {t_s}")


def main():
    d = Path(sys.argv[1])
    sf = json.loads((d / "stories_final.json").read_text())
    prov = json.loads((d / "provenance.json").read_text())
    sem = json.loads((d / "semantic_candidates.json").read_text())
    bmap = {b["boundary_id"]: b for b in sf["boundaries"]}
    num, den = sf["fps"]["num"], sf["fps"]["den"]
    fps = num / den
    video = d.parent.parent.parent / "news_videos" / f"linyi_news_{d.name.split('_')[-1]}.mp4"
    g = FG(video)
    out_dir = d / "review"
    out_dir.mkdir(exist_ok=True)
    merges_by_t = {m["t_ms"]: m for m in sem.get("merges", [])}
    cands_by_t = {c["t_ms"]: c for c in sem["candidates"]}
    edge_by_frame = {e["frame"]: e for e in prov["story_edges"]}

    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT, 15)
    font_s = ImageFont.truetype(FONT, 12)
    cards = ["# boundary_cards.md — 每条目边界解释卡（provenance 渲染）\n"]

    for i, st in enumerate(sf["stories"][1:], 1):
        b = bmap[st["start_boundary"]]
        e = edge_by_frame.get(b["frame"], {})
        t_s = b["frame"] / fps
        offs = [-3, -2, -1, 0, 1, 2, 3]
        frames = [grab_str(g, t_s + o, fps) for o in offs]
        W = 240 * len(offs)
        img = Image.new("RGB", (W, 135 + 78), (18, 18, 18))
        dr = ImageDraw.Draw(img)
        for k, (o, fr) in enumerate(zip(offs, frames)):
            im = Image.fromarray(fr)
            img.paste(im, (k * 240, 24))
            dr.rectangle([k * 240, 24, k * 240 + 239, 158], outline=(200, 60, 60) if o == 0 else (70, 70, 70))
            dr.text((k * 240 + 6, 6), f"{t_s + o:8.2f}s{'  ◀CUT' if o == 0 else ''}",
                    fill=(255, 90, 90) if o == 0 else (150, 150, 150), font=font_s)
        title = e.get("title_t_ms")
        src = ",".join(e.get("cluster", {}).get("members", [])[:3]) if e.get("cluster") else \
              (e.get("nearest_cand") or "?")
        how = e.get("how", "?")
        line2 = (f"B{b['frame']} @{t_s:.2f}s {b['boundary_type']}/{b['transition_type']} "
                 f"sem={b['semantic_confidence']} frame={b['frame_confidence']} 落帧={how} "
                 f"题花@{title and round(title/1000,1)}s")
        dr.text((6, 162), line2, fill=(220, 220, 220), font=font_s)
        dr.text((6, 180), f"cand={src}  onset={e.get('semantic_onset_ms') and round(e['semantic_onset_ms']/1000,1)}s",
                fill=(160, 160, 160), font=font_s)
        img.save(out_dir / f"boundary_{i:02d}_f{b['frame']}.png")
        card = [f"## S{i:02d} 起点 @{t_s:.2f}s (帧 {b['frame']})",
                f"- 类型 `{b['boundary_type']}` / `{b['transition_type']}`，roles={b['roles']}",
                f"- 置信 sem={b['semantic_confidence']} frame={b['frame_confidence']} final={b['final_confidence']}",
                f"- 语义起点 @{e.get('semantic_onset_ms') and round(e['semantic_onset_ms']/1000,2)}s"
                f"（导语声先起），题花确认 @{title and round(title/1000,2)}s",
                f"- 落帧方式: {how}"]
        cl = e.get("cluster") or {}
        if cl:
            card.append(f"- 证据: {cl.get('det')} grade={cl.get('grade')} "
                        f"llm 新/同={cl.get('llm', {}).get('new')}/{cl.get('llm', {}).get('keep')}")
        c = cands_by_t.get(e.get("t_ms", -1))
        if c:
            card.append(f"- 候选来源: {c['sources']} 题花[{c['features']['titles'][0][:16]}→"
                        f"{c['features']['titles'][1][:16]}]")
        cards.append("\n".join(card) + "\n")
    (out_dir / "boundary_cards.md").write_text("\n".join(cards), encoding="utf-8")
    print(f"review/: {len(sf['stories'])-1} 张回放图 + boundary_cards.md")


if __name__ == "__main__":
    main()
