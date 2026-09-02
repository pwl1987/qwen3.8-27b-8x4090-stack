#!/usr/bin/env python3
"""RFT 沙箱并行容量压测 — 计划缺口 #5 (128 核 docker 并行单元测试).

每题一个隔离容器: --network none --read-only --cap-drop ALL, 资源限额,
内部 timeout. 度量吞吐/失败模式, 推算 RFT 全量验证负载.
"""
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

IMAGE = "python:3.12-alpine"
BASE = "/tmp/sbx"
PY = sys.executable


def build_workdir(row, d):
    os.makedirs(d, exist_ok=True)
    imports = row.get("test_imports") or []
    if isinstance(imports, str):
        imports = [imports]
    tests = row["test_list"]
    if isinstance(tests, str):
        tests = [tests]
    with open(f"{d}/solution.py", "w") as f:
        f.write(row["code"])
    with open(f"{d}/run_tests.py", "w") as f:
        f.write("import sys; sys.path.insert(0, '/work')\n")
        for imp in imports:
            f.write(f"{imp}\n")
        f.write("from solution import *\n")
        for t in tests:
            f.write(f"{t}\n")  # entries are complete assert statements
        f.write("print('ALL_PASS')\n")


def run_one(args):
    tid, d = args
    cmd = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--memory", "256m", "--memory-swap", "256m",
        "--pids-limit", "64", "--cpus", "0.5",
        "--tmpfs", "/tmp:rw,size=16m,noexec",
        "-v", f"{d}:/work:ro", "-w", "/work",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        IMAGE, "timeout", "10", "python", "-B", "run_tests.py",
    ]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        rc, out = p.returncode, p.stdout.strip()
    except subprocess.TimeoutExpired:
        rc, out = 125, "ORCHESTRATOR_TIMEOUT"
    return tid, rc, time.time() - t0, out


def main(n, par):
    df = pd.read_parquet("/data/datasets/mbpp/sanitized_test.parquet").head(n)
    shutil.rmtree(BASE, ignore_errors=True)
    jobs = []
    for _, row in df.iterrows():
        d = f"{BASE}/{row['task_id']}"
        build_workdir(row, d)
        jobs.append((int(row["task_id"]), d))

    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=par) as ex:
        for r in ex.map(run_one, jobs):
            results.append(r)
    wall = time.time() - t0

    ok = sum(1 for _, rc, _, _ in results if rc == 0)
    fails = {}
    for _, rc, _, _ in results:
        fails[rc] = fails.get(rc, 0) + 1
    times = sorted(t for _, _, t, _ in results)
    print(f"n={n} parallel={par} wall={wall:.1f}s "
          f"pass={ok}/{n} ({100*ok/n:.0f}%) rc_hist={fails}")
    print(f"per-ct: p50={times[len(times)//2]:.2f}s p90={times[int(len(times)*0.9)]:.2f}s "
          f"max={times[-1]:.2f}s | throughput={n/wall:.1f} verif/s")
    bad = [(t, rc, o[:100]) for t, rc, _, o in results if rc != 0]
    for t, rc, o in bad[:5]:
        print(f"  FAIL task={t} rc={rc}: {o}")
    json.dump({"n": n, "par": par, "wall": wall, "pass": ok,
               "rc_hist": fails, "per_ct": times},
              open(f"/data/compose/qwen27b/rft/sbx_{n}_{par}.json", "w"))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
