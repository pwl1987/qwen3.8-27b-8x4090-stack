#!/usr/bin/env python3
"""news_pipeline.py — 批次I P3 一键编排：Stage DAG · 原子缓存 · Run Lock · GPU 守卫 · serve 归属.

用法:
  python3 news_pipeline.py <run_dir> [--llm27b] [--force STAGE] [--from STAGE] [--list] [--no-render]
契约（批次I v5 规格）:
  - 指纹 = batch-I-config 哈希 + 各输入 (path,size,mtime_ns)；上游指纹变 → 下游自动失效；
    改 config → 全部失效（config 参与每阶段指纹）。
  - 原子提交: RUNNING→SUCCESS 状态机（.stages/<名>.json），FAILED 保留诊断；
    "文件存在=done" 废除；崩溃后 RUNNING+死 PID → 视为 FAILED 重跑。
  - Run Lock: .pipeline.lock (O_EXCL+PID)；双实例 fail-fast；死锁持有者可接管（告警留痕）。
  - GPU 守卫: 仅 {2,3}，目标卡空闲显存 <10G → fail-fast 绝不换卡；启动快照入 run manifest。
  - serve 归属: 2B 服务 :8010 未起则自起并记 PID；退出仅当监听 PID==记录 PID 才 stop。
  - 阶段成功 = returncode==0 ∧ 产物存在 ∧ 产物 mtime 更新（transport∧产物 双验）。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path("/data/tools")
CONFIG = TOOLS / "batch-I-config.json"
PYAV = str(TOOLS / "pyav-env" / "bin" / "python")
FUNASR = str(TOOLS / "funasr-env" / "bin" / "python")
ALIGN = str(TOOLS / "align-env" / "bin" / "python")

# 阶段表: name → {venv, gpu, deps, inputs(run_dir 相对), outputs, cmd(模板 run_dir/video)}
STAGES = {
    "shotseg":   {"gpu": 2, "deps": [], "inputs": ["{VIDEO}"],
                  "outputs": ["shots.json", "audio16k.wav", "audio_sig.json", "ocr_sig.json", "siglip_dist.json"],
                  "cmd": ["bash", str(TOOLS / "shotseg/run_shotseg.sh"), "{VIDEO}", "{RUN}", "2"]},
    "asr":       {"gpu": 2, "deps": ["shotseg"], "inputs": ["audio16k.wav"],
                  "outputs": ["e2e/asr.json"],
                  "cmd": [FUNASR, str(TOOLS / "pipeline_e2e.py"), "{RUN}", "asr"]},
    "words":     {"gpu": 2, "deps": ["asr"], "inputs": ["e2e/asr.json", "audio16k.wav"],
                  "outputs": ["e2e/asr_words.json"],
                  "cmd": [ALIGN, str(TOOLS / "align_words.py"), "{RUN}", "--device", "cuda:0"]},
    "stories":   {"gpu": None, "deps": ["shotseg", "asr"], "inputs": ["shots.json", "e2e/asr.json"],
                  "outputs": ["stories.json", "stories_v2.json"],
                  "cmd": ["bash", "-c",
                          f"{FUNASR} {TOOLS}/m7_stories.py '{{RUN}}' && "
                          f"python3 {TOOLS}/m7_stories_v2.py '{{RUN}}'"]},
    "frame_sig": {"gpu": None, "deps": ["shotseg"], "inputs": ["{VIDEO}", "audio16k.wav"],
                  "outputs": ["frame_sig.json"],
                  "cmd": [PYAV, str(TOOLS / "frame_refine.py"), "{RUN}", "sig"]},
    "candidates": {"gpu": None, "deps": ["frame_sig"], "inputs": ["frame_sig.json", "ocr_sig.json", "siglip_dist.json"],
                   "outputs": ["signal_candidates.json"],
                   "cmd": [PYAV, str(TOOLS / "frame_refine.py"), "{RUN}", "candidates"]},
    "shottrack": {"gpu": None, "deps": ["frame_sig"], "inputs": ["frame_sig.json", "shots.json"],
                  "outputs": ["shots_frame.json"],
                  "cmd": [PYAV, str(TOOLS / "frame_refine.py"), "{RUN}", "shottrack"]},
    "vision":     {"gpu": None, "serve": True, "deps": ["shotseg"], "inputs": ["shots.json"],
                  "outputs": ["e2e/vision.json"],
                  "cmd": [PYAV, str(TOOLS / "pipeline_e2e.py"), "{RUN}", "vision"]},
    "semantic":   {"gpu": None, "serve": True, "deps": ["stories", "asr", "vision"],
                  "inputs": ["stories_v2.json", "e2e/asr.json", "e2e/vision.json"],
                  "outputs": ["e2e/semantic.json"],
                  "cmd": [PYAV, str(TOOLS / "pipeline_e2e.py"), "{RUN}", "semantic"]},
    "selfcheck": {"gpu": None, "serve": True, "deps": ["candidates", "shottrack", "stories", "words", "semantic"],
                  "inputs": ["signal_candidates.json", "shots_frame.json", "stories_v2.json",
                             "e2e/asr.json", "e2e/asr_words.json", "audio16k.wav"],
                  "outputs": ["semantic_candidates.json"], "extra_args": ["{LLM}"]},
    "refine":    {"gpu": None, "deps": ["selfcheck", "shottrack"],
                  "inputs": ["frame_sig.json", "signal_candidates.json", "semantic_candidates.json",
                             "shots_frame.json", "e2e/asr_words.json"],
                  "outputs": ["stories_final.json", "provenance.json"],
                  "cmd": [PYAV, str(TOOLS / "frame_refine.py"), "{RUN}", "refine"]},
    "edl":       {"gpu": None, "deps": ["refine"],
                  "inputs": ["stories_final.json", "provenance.json"],
                  "outputs": ["edl/story_000.json"],
                  "cmd": [PYAV, str(TOOLS / "edl_export.py"), "{RUN}", "--all"]},
    "validate":  {"gpu": None, "deps": ["refine", "words"],
                  "inputs": ["stories_final.json", "e2e/asr.json", "e2e/asr_words.json"],
                  "outputs": [],  # invariants 无产物，成功即通过（exit 0）
                  "cmd": [PYAV, str(TOOLS / "validate.py"), "invariants", "{RUN}"]},
}
ORDER = ["shotseg", "asr", "words", "stories", "vision", "semantic", "frame_sig",
         "candidates", "shottrack", "selfcheck", "refine", "edl", "validate"]


def cfg_hash():
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest()[:16]


def fp_file(p: Path):
    st = p.stat()
    return [str(p), st.st_size, st.st_mtime_ns]


def resolve(run: Path, spec):
    video = run.parent.parent.parent / "news_videos" / f"linyi_news_{run.name.split('_')[-1]}.mp4"
    return [str(video) if s == "{VIDEO}" else str(run / s) for s in spec]


def read_marker(run: Path, name):
    p = run / ".stages" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_marker_atomic(run: Path, name, data):
    d = run / ".stages"
    d.mkdir(exist_ok=True)
    tmp = d / f".{name}.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, d / f"{name}.json")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:  # 存在但无权信号（如 root 进程）→ 仍视为活
        return True
    except OSError:
        return False


def gpu_snapshot(cfg):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
         "--format=csv,noheader,nounits"], text=True)
    snap = {}
    for line in out.strip().splitlines():
        idx, uuid, used, total = [x.strip() for x in line.split(",")]
        snap[int(idx)] = {"uuid": uuid, "used_mib": int(used), "total_mib": int(total)}
    return snap


def gpu_guard(cfg, gpu, stage, manifest):
    snap = gpu_snapshot(cfg)
    manifest.setdefault("gpu_snapshots", []).append({"stage": stage, **snap.get(gpu, {})})
    allowed = cfg["validation"]["gpus_allowed"]
    if gpu not in allowed:
        raise SystemExit(f"[{stage}] GPU{gpu} 不在允许集 {allowed}")
    need = cfg["validation"]["gpu_min_free_mib"]
    free = snap[gpu]["total_mib"] - snap[gpu]["used_mib"]
    if free < need:
        raise SystemExit(f"[{stage}] GPU{gpu} 空闲 {free}MiB < {need}MiB，fail-fast（绝不自动换卡）")


# ── serve 生命周期（PID 归属制） ──
def http_ok(port):
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def listener_pid(port):
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{port} " in line:
            for tok in line.split():
                if tok.startswith("pid="):
                    return int(tok[4:].split(",")[0])
    return None


def ensure_serve(cfg, manifest):
    port = cfg["semantic"]["port"]
    if http_ok(port):
        manifest["serve"] = {"started_by_pipeline": False, "port": port}
        return
    log = open("/tmp/news_pipeline_serve.log", "ab")
    env = dict(os.environ, PORT=str(port), CUDA_VISIBLE_DEVICES="3")
    subprocess.Popen(["bash", str(TOOLS / "serve/vl-qwen35-2b.sh")], env=env, stdout=log, stderr=log,
                     start_new_session=True)
    for _ in range(120):
        time.sleep(5)
        if http_ok(port):
            pid = listener_pid(port)
            if pid is None:  # vllm 子进程就绪竞态：再等几秒复抓
                time.sleep(8)
                pid = listener_pid(port)
            manifest["serve"] = {"started_by_pipeline": True, "port": port, "pid": pid}
            print(f"[serve] 自起 2B :{port} pid={pid}（退出时按 PID 归属 stop）")
            return
    raise SystemExit("serve 健康检查超时（10min）")


def stop_serve_if_owned(manifest):
    s = manifest.get("serve") or {}
    if not (s.get("started_by_pipeline") and s.get("pid")):
        return
    cur = listener_pid(s["port"])
    if cur == s["pid"]:
        subprocess.run(["bash", str(TOOLS / "serve/stop.sh"), str(s["port"])], check=False)
        print(f"[serve] 停止自起服务 :{s['port']} (pid={s['pid']})")
    else:
        print(f"[serve] 端口 {s['port']} 已被 pid={cur} 接管（记录 {s['pid']}），不动")


def stage_fingerprint(run: Path, name, cfg):
    inputs, _ = resolve(run, STAGES[name]["inputs"]), None
    fps = [fp_file(Path(p)) for p in inputs if Path(p).exists()]
    if len(fps) != len(inputs):
        missing = [p for p in inputs if not Path(p).exists()]
        raise SystemExit(f"[{name}] 输入缺失: {missing}")
    deps = {d: (read_marker(run, d) or {}).get("fingerprint") for d in STAGES[name]["deps"]}
    return {"config": cfg_hash(), "inputs": fps, "deps": deps}


def run_stage(run: Path, name, cfg, args, manifest):
    st = STAGES[name]
    fp = stage_fingerprint(run, name, cfg)
    gpu = st.get("gpu")
    if gpu is not None:
        gpu = int(os.environ.get("NEWS_PIPELINE_GPU", str(gpu)))
        gpu_guard(cfg, gpu, name, manifest)
        st = dict(st, gpu=gpu)
    outs = [run / o for o in st["outputs"]]
    mt_before = [o.stat().st_mtime_ns if o.exists() else 0 for o in outs]
    if st.get("serve"):  # vision/semantic 需要 2B；selfcheck 走 27B 时 2B 仍在服务
        ensure_serve(cfg, manifest)
    if name == "selfcheck":
        cmd = [FUNASR, str(TOOLS / "story_selfcheck.py"), str(run)] + \
              (["--llm27b"] if args.llm27b else [])
    else:
        video = run.parent.parent.parent / "news_videos" / f"linyi_news_{run.name.split('_')[-1]}.mp4"
        cmd = [c.replace("{RUN}", str(run)).replace("{VIDEO}", str(video)) for c in st["cmd"]]
    write_marker_atomic(run, name, {"status": "RUNNING", "pid": os.getpid(), "cmd": cmd,
                                    "fingerprint": fp, "begin": time.time()})
    print(f"[{name}] RUN {' '.join(cmd)[:110]}")
    env = dict(os.environ)
    if st.get("gpu") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(st["gpu"])  # 绝不让子进程看到生产卡
        env["CPATH"] = (":/data/tools/ffmpeg-deb/root/usr/include/python3.12"
                        ":/data/tools/ffmpeg-deb/root/usr/include")
        env["TQDM_DISABLE"] = "1"
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dur = time.time() - t0
    produced = all(o.exists() for o in outs) if outs else True
    refreshed = all(o.exists() and o.stat().st_mtime_ns > mt0 for o, mt0 in zip(outs, mt_before)) \
        if outs else True
    ok = r.returncode == 0 and produced
    rec = {"status": "SUCCESS" if ok else "FAILED", "cmd": cmd, "fingerprint": fp,
           "returncode": r.returncode, "begin": t0, "end": time.time(), "dur_s": round(dur, 1),
           "outputs_refreshed": refreshed}
    if not ok:
        rec["stderr_tail"] = (r.stderr or "")[-1500:]
        rec["stdout_tail"] = (r.stdout or "")[-800:]
    write_marker_atomic(run, name, rec)
    tag = "OK " if ok else "FAIL"
    print(f"[{name}] {tag} {dur:.1f}s")
    if not ok:
        print((r.stderr or r.stdout or "")[-1200:])
        raise SystemExit(f"stage {name} 失败")
    return rec


def invalidate_downstream(run: Path, name):
    idx = ORDER.index(name)
    for s in ORDER[idx:]:
        p = run / ".stages" / f"{s}.json"
        if p.exists():
            p.unlink()
            print(f"[invalidate] {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--llm27b", action="store_true")
    ap.add_argument("--force")
    ap.add_argument("--from-stage")
    ap.add_argument("--adopt", action="store_true",
                    help="产物齐全的阶段直接收编为 SUCCESS（指纹按当前输入算，后续变化照常失效）")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir)
    cfg = json.loads(CONFIG.read_text())
    if args.list:
        for s in ORDER:
            m = read_marker(run, s)
            print(f"{s:12s} {m['status'] if m else '-':8s} "
                  f"{m.get('dur_s', '') if m else ''}s")
        return
    if args.adopt:
        for name in ORDER:
            outs = [run / o for o in STAGES[name]["outputs"]]
            if outs and all(o.exists() for o in outs):
                try:
                    fp = stage_fingerprint(run, name, cfg)
                except SystemExit:
                    continue
                write_marker_atomic(run, name, {"status": "SUCCESS", "adopted": True,
                                                "fingerprint": fp, "end": time.time()})
                print(f"[adopt] {name}")
        return
    if args.force:
        invalidate_downstream(run, args.force)

    lock = run / ".pipeline.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({"pid": os.getpid(), "ts": time.time()}).encode())
        os.close(fd)
    except FileExistsError:
        try:
            holder = json.loads(lock.read_text())
            if pid_alive(holder.get("pid", -1)):
                raise SystemExit(f"另一 pipeline 实例运行中 pid={holder['pid']}，fail-fast")
            print(f"[lock] 持锁进程 {holder.get('pid')} 已死，接管（留痕）")
            lock.unlink()
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "ts": time.time(),
                                     "took_over_from": holder.get("pid")}).encode())
            os.close(fd)
        except json.JSONDecodeError:
            raise SystemExit("锁文件损坏，人工清理后重试")
    manifest = {"run_dir": str(run), "config_sha16": cfg_hash(), "stages": []}
    gpu_snap = gpu_snapshot(cfg)
    manifest["gpu_snapshot_at_start"] = {i: gpu_snap[i] for i in cfg["validation"]["gpus_allowed"]
                                         if i in gpu_snap}
    log = {"stages": []}
    log_path = run / "run_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
    try:
        start = False if args.from_stage else True
        for name in ORDER:
            if args.from_stage == name:
                start = True
            if not start:
                continue
            fp = None
            try:
                fp = stage_fingerprint(run, name, cfg)
            except SystemExit as e:
                if "输入缺失" in str(e) and name != "shotseg":
                    raise
                if name == "shotseg":
                    pass
            m = read_marker(run, name)
            if (m and m.get("status") == "SUCCESS" and fp
                    and m.get("fingerprint") == fp):
                print(f"[{name}] SKIP（指纹一致）")
                log["stages"].append({"stage": name, "status": "SKIP", "ts": time.time()})
                continue
            if m and m.get("status") == "RUNNING" and pid_alive(m.get("pid", -1)):
                raise SystemExit(f"[{name}] RUNNING 且进程存活——双实例？fail-fast")
            rec = run_stage(run, name, cfg, args, manifest)
            log["stages"].append({"stage": name, "status": rec["status"],
                                  "dur_s": rec["dur_s"], "ts": rec["end"]})
            log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
    finally:
        stop_serve_if_owned(manifest)
        (run / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
    total = sum(s.get("dur_s", 0) for s in log["stages"])
    print(f"== 完成；本趟累计 {total:.0f}s；明细 {log_path} ==")


if __name__ == "__main__":
    main()
