"""P0 步速分解：单次隔离 g512 请求（temp0）前后 /metrics 差分。
输出: tok/s | tok/step | step_ms | accepted%/tok  —— 口径与臂间完全一致。
tok/s 用 usage.completion_tokens 实际计数（防提前 EOS 假象）。"""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 19627
BASE = f"http://127.0.0.1:{PORT}"


def metrics():
    m = {}
    for line in urllib.request.urlopen(BASE + "/metrics").read().decode().splitlines():
        if line.startswith(("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total",
                            "vllm:spec_decode_num_draft_tokens_total")):
            k, v = line.split()
            m[k.split("{")[0]] = float(v)
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total") and "created" not in line:
            pos = line.split('position="')[1].split('"')[0]
            m[f"pos{pos}"] = float(line.rsplit(" ", 1)[1])
    return m


def one(seed):
    prompt = (f"[bench {seed}] 请详细讲解 GPU 显存带宽对大模型推理性能的影响机制，"
              "从 GDDR 显存特性、权重读取瓶颈、KV 缓存占用、批处理摊薄、投机解码验证步等角度逐一展开论述，"
              "写成一篇结构完整的技术长文，使用 markdown 小标题与编号列表，至少覆盖八个要点。")
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 512, "temperature": 0, "seed": 4242,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "cache_salt": f"p0-step-{seed}"}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    m0 = metrics()
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    wall = time.time() - t0
    m1 = metrics()
    ntok = resp.get("usage", {}).get("completion_tokens", 512)
    dd = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    da = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    dt = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get("vllm:spec_decode_num_draft_tokens_total", 0)
    perpos = [int(m1.get(f"pos{i}", 0) - m0.get(f"pos{i}", 0)) for i in range(7)]
    prof = "/".join(str(p) for p in perpos)
    tps = ntok / wall
    tstep = da / dd + 1 if dd else float("nan")
    step_ms = wall / dd * 1000 if dd else float("nan")
    acc = 100 * da / dt if dt else float("nan")
    return tps, tstep, step_ms, acc, ntok, prof


for seed in (1, 2, 3):
    tps, tstep, step_ms, acc, ntok, prof = one(seed)
    print(f"{tps:.1f} tok/s | {tstep:.2f} tok/step | {step_ms:.1f} ms/step | {acc:.1f}%/tok | {ntok} tok | pos {prof}", flush=True)
