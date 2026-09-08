"""t3 decode 探针：复刻 ulmus t3 夹具的 decode 请求（FILLER×12+提问，temp0 seed4242，
max512，唯一 cache_salt），N 次隔离跑；每次记录 文本sha/长度/耗时/tok/s + /metrics
接受率差分。用于判别 低档boot：文本翻转(i) vs 融合机械性死亡(ii)。"""
import hashlib
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/data/sandbox/ab-vllm/repo/bench")
from ulmus_validate import make_prompt, post_json  # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 19627
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BASE = f"http://127.0.0.1:{PORT}"
import uuid  # noqa: E402

PROMPT = make_prompt(512)


def snap():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        d = a = t = None
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_drafts_total"):
                d = float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:spec_decode_num_accepted_tokens_total") and "per_pos" not in line:
                a = float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                t = float(line.rsplit(" ", 1)[1])
    return d, a, t


for i in range(N):
    d0, a0, t0 = snap()
    t_start = time.time()
    res, _ = post_json(
        BASE + "/v1/chat/completions",
        {
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 512,
            "temperature": 0,
            "seed": 4242,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_salt": "probe-" + uuid.uuid4().hex,
        },
    )
    dt = time.time() - t_start
    d1, a1, t1 = snap()
    txt = res["choices"][0]["message"]["content"]
    steps = d1 - d0
    acc = (a1 - a0) / steps if steps else 0
    accrate = 100 * (a1 - a0) / (t1 - t0) if t1 > t0 else 0
    ntok = len(txt)
    print(f"run{i}: tok={ntok} {dt:.1f}s {ntok/dt:.1f} tok/s | steps={steps:.0f} "
          f"acc/step={acc:.2f} {accrate:.1f}%/tok | sha={hashlib.sha256(txt.encode()).hexdigest()[:12]}")
    with open(f"/tmp/p0/t3probe-run{i}.txt", "w") as f:
        f.write(txt)
