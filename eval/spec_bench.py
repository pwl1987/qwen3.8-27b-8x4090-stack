#!/usr/bin/env python3
"""spec decoding 沙箱测试套件 — 对指定端点跑 short/longgen/prefill 三类测试, 输出 JSON.
用法: spec_bench.py <label> <port> <out.json>
"""
import json, sys, time, urllib.request

LABEL, PORT, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
BASE = f"http://127.0.0.1:{PORT}"

SHORT_PROMPTS = [
    "用 Python 写一个快速排序，要求原地排序并附带注释",
    "Write a TypeScript debounce function with cancel method, fully typed.",
    "写一个 bash 脚本：找出 /var/log 下 7 天内修改过且大于 100MB 的文件并输出明细",
]
LONGGEN_PROMPT = (
    "实现一个完整的、生产级的 Python LRU 缓存类 ThreadSafeLRUCache，要求：get/put/delete/"
    "clear、TTL 过期、线程安全、容量限制与 LRU 淘汰、命中/未命中统计、上下文管理器支持、"
    "完整的 pytest 单元测试。请输出完整代码与测试，不要省略任何部分，不要中断。"
)
PREFILL_FILE = "/data/compose/qwen27b/train/imatrix-calib.txt"
PREFILL_MAX_CHARS = 140_000  # ~33K tokens


def stream_chat(prompt, max_tokens):
    """流式请求, 返回 (completion_tokens, gen_tps, wall_s)。TPS=首token后到末token的斜率"""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens, "seed": 42,
        "stream": True, "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    # 思考模型: 流式先 reasoning_content 后 content, 两类块都计入生成时间
    t0 = time.time(); t_first = None; n_chunks = 0; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if t_first is None:
                        t_first = time.time()
                    n_chunks += 1
                    t_last = time.time()
    wall = time.time() - t0
    comp = (usage or {}).get("completion_tokens", n_chunks)
    gen_tps = (comp - 1) / (t_last - t_first) if t_first and comp > 1 and t_last > t_first else 0.0
    return comp, gen_tps, wall


def prefill_test():
    """一次性长 prompt (max_tokens=1), prefill tps = prompt_tokens / wall"""
    corpus = open(PREFILL_FILE, encoding="utf-8", errors="ignore").read()[:PREFILL_MAX_CHARS]
    prompt = f"以下是代码语料：\n{corpus}\n\n以上语料的最后 3 行是什么？逐字引用。"
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 16, "seed": 42,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    wall = time.time() - t0
    usage = r.get("usage", {})
    pn = usage.get("prompt_tokens", 0)
    return {"prompt_tokens": pn, "wall_s": round(wall, 2),
            "prefill_tps": round(pn / wall, 1) if pn and wall else 0}


def metrics_snapshot():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
            return r.read().decode()
    except Exception:
        return ""


def spec_counters(text):
    out = {}
    for ln in text.splitlines():
        if ln.startswith("#") or " " not in ln:
            continue
        name, _, val = ln.rpartition(" ")
        if any(k in name.lower() for k in ("spec", "draft", "ngram", "accept")):
            try:
                out[name] = float(val)
            except ValueError:
                pass
    return out


def main():
    res = {"label": LABEL, "ts": time.strftime("%F %T"), "short": [], "longgen": {}, "prefill": {}, "spec_metrics": {}}
    m0 = spec_counters(metrics_snapshot())

    for i, p in enumerate(SHORT_PROMPTS):
        comp, tps, wall = stream_chat(p, 512)
        res["short"].append({"i": i, "tokens": comp, "gen_tps": round(tps, 1), "wall_s": round(wall, 2)})
        print(f"[{LABEL}] short{i}: {comp} tok, {tps:.1f} tok/s", flush=True)

    comp, tps, wall = stream_chat(LONGGEN_PROMPT, 2048)
    res["longgen"] = {"tokens": comp, "gen_tps": round(tps, 1), "wall_s": round(wall, 2)}
    print(f"[{LABEL}] longgen: {comp} tok, {tps:.1f} tok/s", flush=True)

    res["prefill"] = prefill_test()
    print(f"[{LABEL}] prefill: {res['prefill']['prompt_tokens']} tok @ {res['prefill']['prefill_tps']} tok/s", flush=True)

    m1 = spec_counters(metrics_snapshot())
    res["spec_metrics"] = {
        "before": m0,
        "after": m1,
        "delta": {k: round(m1.get(k, 0) - m0.get(k, 0), 1) for k in set(m0) | set(m1)},
    }
    res["spec_metrics_raw_names"] = sorted(set(m1))

    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[{LABEL}] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
