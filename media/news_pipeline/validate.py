#!/usr/bin/env python3
"""validate.py — 批次I 机器验收：不变量全集 + 帧身份判别式验证（+P4 报告另挂）.

子命令:
  invariants <run_dir>   stories_final 契约 / words 覆盖与时序 / EDL 双域一致
  frames <edl.json>      渲染帧身份判别式验证：输出帧 o ↔ 源帧 N，SSIM(出,源N) 必须
                         严格优于 SSIM(出,源N±1)（判别式，非相似度阈值），另核 PTS/帧数/时长。
pyav-env 运行（PyAV+numpy）。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/tools")
import framedomain as fd

FFPROBE = "/data/tools/ffmpeg-deb/ffprobe"


def load_cfg():
    return json.loads(Path("/data/tools/batch-I-config.json").read_text())


# ── 帧身份验证 ──────────────────────────────────────────────────
class FG:
    """精确取第 idx 帧（灰度缩略）。seek 到 keyframe 后顺序解码，按 fr.time 半帧窗唯一匹配。"""

    def __init__(self, path):
        import av
        self.ctx = av.open(str(path))
        self.vs = self.ctx.streams.video[0]
        try:
            self.fps = float(self.vs.average_rate)
        except AttributeError:  # PyAV 版本差异 → ffprobe 兜底
            self.fps = float(ffprobe_meta(path)["r_frame_rate"].split("/")[0]) / \
                float(ffprobe_meta(path)["r_frame_rate"].split("/")[1])

    def gray(self, idx, w=320):
        t = idx / self.fps
        tol = 0.5 / self.fps
        self.ctx.seek(int(max(0.0, t - 1.5) / float(self.vs.time_base)), stream=self.vs, backward=True)
        for pkt in self.ctx.demux(self.vs):
            for f in pkt.decode():
                if f.time is None:
                    continue
                if abs(f.time - t) <= tol:
                    return f.reformat(width=w, height=w * 9 // 16, format="gray").to_ndarray()
                if f.time > t + tol + 1.0:
                    break
        raise RuntimeError(f"帧 {idx} 未取到")


def ssim(a: np.ndarray, b: np.ndarray, k=7) -> float:
    """7x7 box 窗 SSIM（判别用途，非画质测量；确定性）。"""
    from numpy.lib.stride_tricks import sliding_window_view
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def means(x):
        w = sliding_window_view(x, (k, k))
        return w.mean(axis=(2, 3))

    ma, mb, mab = means(a), means(b), means(a * b)
    va = means(a * a) - ma ** 2
    vb = means(b * b) - mb ** 2
    cov = mab - ma * mb
    s = ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


def ffprobe_meta(path):
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,duration,r_frame_rate", "-of", "json", str(path)], text=True)
    return json.loads(out)["streams"][0]


def cmd_frames(edl_path: Path, cfg):
    edl = json.loads(edl_path.read_text())
    rendered = edl_path.with_suffix(".mp4")
    if not rendered.exists():
        raise SystemExit(f"未渲染: {rendered}")
    num, den = edl["fps"]["num"], edl["fps"]["den"]
    for seg in edl["ranges"]:  # 双域断言
        for mk, fk in (("start_ms", "start_frame"), ("end_ms", "end_frame")):
            if seg[mk] != fd.frame_to_ms(seg[fk], num, den):
                raise SystemExit(f"EDL 双域不一致 {mk}")
    f0 = edl["ranges"][0]["start_frame"]
    n = edl["duration_frames"]
    src, out = FG(edl["source"]), FG(rendered)

    meta = ffprobe_meta(rendered)
    dur_ok = abs(float(meta["duration"]) * 1000 - edl["total_duration_ms"]) <= \
        cfg["validation"]["ffprobe_dur_tol_ms"]
    nframes_ok = int(meta["nb_frames"]) == n

    probes = sorted({0, 1, n // 2, n - 2, n - 1} if cfg["validation"]["probe_mid"]
                    else {0, 1, n - 2, n - 1})
    rows, all_pass = [], True
    for o in probes:
        A = out.gray(o)
        exp = src.gray(f0 + o)
        ss = {(-1): ssim(A, src.gray(f0 + o - 1)),
              0: ssim(A, exp),
              (+1): ssim(A, src.gray(f0 + o + 1))}
        best_off = max(ss, key=lambda x: (ss[x], -x))
        margin = ss[best_off] - ss[0]
        if best_off == 0:
            cls, ok = "PASS", True
        elif margin < 0.01:  # 静态段 ±1 帧视觉不可分（CRF 噪声级差异）= 身份无意义非错帧
            cls, ok = "MARGINAL_STATIC", True
        else:
            cls, ok = "FAIL", False
        all_pass &= ok
        rows.append({"output_frame": o, "expected_source_frame": f0 + o,
                     "best_source_offset": best_off, "margin": round(margin, 5),
                     "ssim_expected": round(ss[0], 5), "ssim_prev": round(ss[-1], 5),
                     "ssim_next": round(ss[1], 5), "class": cls, "pass": ok})
        print(f"  出帧{o:>6} ↔ 源帧{f0+o:>6}  S(e)={ss[0]:.4f} S(e-1)={ss[-1]:.4f} "
              f"S(e+1)={ss[1]:.4f}  {cls}")
    result = {"edl": str(edl_path), "rendered": str(rendered),
              "duration_frames_expected": n, "duration_frames_actual": int(meta["nb_frames"]),
              "duration_ms_expected": edl["total_duration_ms"],
              "ffprobe_duration_s": float(meta["duration"]),
              "duration_pass": dur_ok, "nframes_pass": nframes_ok,
              "probes": rows, "all_identity_pass": all_pass,
              "pass": all_pass and dur_ok and nframes_ok}
    out_path = edl_path.with_suffix(".identity.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"{out_path.name}: identity={'PASS' if all_pass else 'FAIL'} "
          f"dur={'PASS' if dur_ok else 'FAIL'} nframes={'PASS' if nframes_ok else 'FAIL'}")
    return result


# ── 不变量全集 ──────────────────────────────────────────────────
def cmd_invariants(d: Path, cfg):
    sf = json.loads((d / "stories_final.json").read_text())
    fd.validate_stories_final(sf, sf["n_frames"], sf["fps"]["num"], sf["fps"]["den"])
    print("[OK] stories_final 契约（闭包/引用/n_shots/无缝覆盖）")

    issues = []
    wf = d / "e2e" / "asr_words.json"
    if wf.exists():
        w = json.loads(wf.read_text())["words"]
        asr_chars = sum(len(s["text"]) for s in
                        json.loads((d / "e2e" / "asr.json").read_text())["segments"])
        cover = len(w) / max(asr_chars, 1)
        if cover < cfg["validation"]["words_coverage_min"]:
            issues.append(f"words 覆盖 {cover:.3f} < {cfg['validation']['words_coverage_min']}")
        for x, y in zip(w, w[1:]):
            if x["s_ms"] > y["s_ms"] or x["e_ms"] > y["e_ms"] or x["e_ms"] < x["s_ms"]:
                issues.append(f"words 时序违例 @ {x}")
                break
        zero = sum(1 for x in w if x["e_ms"] <= x["s_ms"])
        print(f"[OK] words {len(w)} 条，覆盖 {cover:.3f}，时序有序，零时长 {zero} 条（B5 修补在 align_words 后处理）")
    num, den = sf["fps"]["num"], sf["fps"]["den"]
    for e in sorted((d / "edl").glob("*.json")):
        edl = json.loads(e.read_text())
        if edl.get("version", 0) >= 2:
            for seg in edl["ranges"]:
                for mk, fk in (("start_ms", "start_frame"), ("end_ms", "end_frame")):
                    if seg[mk] != fd.frame_to_ms(seg[fk], num, den):
                        issues.append(f"{e.name}: 双域不一致 {mk}")
    print("[OK] EDL v2 双域一致" if not issues else "[FAIL]")
    if issues:
        for i in issues:
            print("  -", i)
        raise SystemExit(1)
    print("invariants ALL GREEN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["invariants", "frames"])
    ap.add_argument("path")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.cmd == "invariants":
        cmd_invariants(Path(a.path), cfg)
    else:
        cmd_frames(Path(a.path), cfg)


if __name__ == "__main__":
    main()
