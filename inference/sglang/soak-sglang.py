#!/usr/bin/env python3
# 24h 浸泡采样器: SGLang(经 LB :8000) 健康/延迟/GPU/错误率 → soak 日志
# 用法: nohup python3 soak-sglang.py >/dev/null 2>&1 &
import json, time, datetime, urllib.request, subprocess
LB = "http://127.0.0.1:8000"
LOG = "/data/compose/qwen27b/soak-sglang-20260903.log"
MODEL = "/data/models/Qwen3.8-27B-FP8"
def chat():
    body = json.dumps({"model":MODEL,"messages":[{"role":"user","content":"What is 17*23? Reply one number."}],"max_tokens":8,"temperature":0}).encode()
    req = urllib.request.Request(LB+"/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
    t0=time.time()
    r=json.loads(urllib.request.urlopen(req, timeout=90).read().decode())
    return round(time.time()-t0,2), r["usage"].get("completion_tokens",0)
def sh(cmd, t=15):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=t).stdout
    except Exception as e: return f"ERR:{e}"
print("soak sampler start", flush=True)
while True:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rec = {"ts": ts}
    try:
        urllib.request.urlopen(LB+"/v1/models", timeout=6); rec["health"]="ok"
    except Exception as e: rec["health"]=f"ERR:{str(e)[:80]}"
    try:
        lat, n = chat(); rec["chat_lat_s"]=lat; rec["chat_ntok"]=n
    except Exception as e: rec["chat"]=f"ERR:{str(e)[:80]}"
    g = sh(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"], 10)
    if not g.startswith("ERR"):
        for line in g.strip().splitlines():
            i, mem, u = [x.strip() for x in line.split(",")]
            if i in ("6","7"): rec[f"gpu{i}"]=f"{mem}MiB/{u}%"
    else: rec["gpu"]=g[:60]
    logs = sh(["docker","logs","sglang-prod","--since","6m"])
    nerr = sum(1 for ln in logs.splitlines() if ("Traceback" in ln or "NCCL error" in ln or "CUDA error" in ln))
    rec["sgl_err_6m"] = nerr
    try:
        up = sh(["docker","inspect","sglang-prod","--format","{{.State.Status}}"])
        rec["container"] = up.strip()
    except Exception: pass
    with open(LOG,"a") as f: f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    time.sleep(300)
