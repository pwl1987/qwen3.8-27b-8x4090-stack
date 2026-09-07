#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B benchmark: SGLang TP2  vs  llama.cpp (2-card inference).
Modes:
  tps     single-stream decode tps + TTFT at a target context length
  load    N concurrent requests -> aggregate tps, p50/p99 latency, throughput
  quality lightweight spot-check (gsm8k-style math + code + needle) on the server
Both SGLang and llama.cpp expose the OpenAI /v1/chat/completions API.
"""
import argparse, json, time, sys, os, statistics, threading, concurrent.futures
import urllib.request, urllib.error

def _post(url, payload, stream=False, timeout=900):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def _sse_tokens(resp):
    """Yield (content_chars, reasoning_chars) per SSE data line; count non-empty deltas."""
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            return
        try:
            j = json.loads(d)
        except Exception:
            continue
        ch = j.get("choices") or [{}]
        delta = ch[0].get("delta", {}) if ch else {}
        c = delta.get("content") or ""
        r = delta.get("reasoning_content") or ""
        if c or r:
            yield (c, r)
    return

def make_prompt(n_tokens):
    """A deterministic filler prompt of ~n_tokens tokens (approx)."""
    base = ("The quick brown fox jumps over the lazy dog. " * 3)
    # ~6 tokens per sentence-ish; pad to approx target
    unit = "Lorem ipsum dolor sit amet consectetur adipiscing. "
    n = max(1, n_tokens // 10)
    prompt = ("Answer with a short factual sentence. Context: " + unit * n)
    return prompt

def bench_tps(base, ctx_tokens, max_tokens, model, n_warm=1, n_trials=3):
    ttfts, decs, ttfs = [], [], []
    prompt = make_prompt(ctx_tokens)
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "stream": True}
    for i in range(n_warm + n_trials):
        try:
            resp = _post(base + "/v1/chat/completions", payload, timeout=900)
        except Exception as e:
            print(f"  [trial {i}] request error: {e!r}", file=sys.stderr)
            continue
        t0 = time.time(); ttft = None; ntok = 0
        try:
            for (c, r) in _sse_tokens(resp):
                if ttft is None:
                    ttft = time.time() - t0
                ntok += 1
        except Exception as e:
            print(f"  [trial {i}] stream error: {e!r}", file=sys.stderr)
        t1 = time.time()
        if ttft is not None and ntok > 1:
            dec_time = (t1 - t0) - ttft  # 解码阶段时长 (总耗时 - 首token耗时)
            dec_tps = (ntok - 1) / dec_time if dec_time > 0 else 0.0
            if i >= n_warm:
                ttfts.append(ttft * 1000.0)
                decs.append(dec_tps)
                ttfs.append(t1 - t0)
    out = {"ctx_tokens": ctx_tokens, "max_tokens": max_tokens}
    if decs:
        out["decode_tps"] = round(statistics.mean(decs), 2)
        out["decode_tps_min"] = round(min(decs), 2)
        out["decode_tps_max"] = round(max(decs), 2)
        out["ttft_ms"] = round(statistics.mean(ttfts), 1)
        out["e2e_s"] = round(statistics.mean(ttfs), 2)
        out["n_tokens"] = ntok
    else:
        out["error"] = "no successful trials"
    return out

def _one_load_req(base, model, prompt, max_tokens):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0}
    t0 = time.time()
    try:
        resp = _post(base + "/v1/chat/completions", payload, timeout=900)
        body = json.loads(resp.read().decode())
        dt = time.time() - t0
        ntok = (body.get("usage") or {}).get("completion_tokens", max_tokens)
        return {"ok": True, "tps": ntok / dt if dt > 0 else 0, "lat_s": dt, "ntok": ntok}
    except Exception as e:
        return {"ok": False, "err": repr(e)[:120], "lat_s": time.time() - t0}

def bench_load(base, concurrency, max_tokens, model, prompt_tokens, n_rounds=1):
    prompt = make_prompt(prompt_tokens)
    all_res = []
    for rnd in range(n_rounds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_one_load_req, base, model, prompt, max_tokens) for _ in range(concurrency)]
            all_res.extend(fut.result() for fut in futs)
    ok = [r for r in all_res if r["ok"]]
    out = {"concurrency": concurrency, "n_req": len(all_res), "n_ok": len(ok), "n_fail": len(all_res) - len(ok)}
    if ok:
        lats = sorted(r["lat_s"] for r in ok)
        out["aggregate_tps"] = round(sum(r["ntok"] for r in ok) / max(min(r["lat_s"] for r in ok), 1e-6), 2)
        out["per_req_tps_mean"] = round(statistics.mean(r["tps"] for r in ok), 2)
        out["lat_p50_s"] = round(lats[len(lats)//2], 2)
        out["lat_p99_s"] = round(lats[min(len(lats)-1, int(len(lats)*0.99))], 2)
        out["lat_max_s"] = round(lats[-1], 2)
    if all_res:
        fails = [r["err"] for r in all_res if not r["ok"]]
        if fails:
            out["fail_sample"] = fails[:3]
    return out

GSM8K = [
    ("Janet has 3 apples. She buys 2 more each day for 5 days. How many apples does she have?", 13),
    ("A train travels 60 km/h for 2.5 hours. How many km does it travel?", 150),
    ("A store has 45 items. 1/3 are sold, then 8 more are added. How many remain?", 27),
]
NEEDLE = ("Remember the secret code is 41729. " * 400) + "\nWhat is the secret code? Answer with just the number."

def bench_quality(base, model):
    res = {"math_ok": 0, "math_n": 0}
    for q, ans in GSM8K:
        payload = {"model": model, "messages": [{"role": "user", "content": q + "\nAnswer with just the final number."}],
                   "max_tokens": 64, "temperature": 0}
        try:
            body = json.loads(_post(base + "/v1/chat/completions", payload).read().decode())
            txt = body["choices"][0]["message"]["content"]
            ok = str(ans) in txt
            res["math_ok"] += ok; res["math_n"] += 1
        except Exception as e:
            res["math_n"] += 1
    # needle
    try:
        payload = {"model": model, "messages": [{"role": "user", "content": NEEDLE}],
                   "max_tokens": 32, "temperature": 0}
        body = json.loads(_post(base + "/v1/chat/completions", payload).read().decode())
        txt = body["choices"][0]["message"]["content"]
        res["needle_ok"] = "41729" in txt
    except Exception as e:
        res["needle_ok"] = f"err:{e!r}"[:80]
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--mode", choices=["tps", "load", "quality", "all"], default="all")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-trials", type=int, default=3)
    args = ap.parse_args()
    out = {"base": args.base, "mode": args.mode}
    if args.mode in ("tps", "all"):
        # short + target ctx
        out["tps_2k"] = bench_tps(args.base, 2048, args.max_tokens, args.model, 0, args.n_trials)
        out["tps_8k"] = bench_tps(args.base, 8192, args.max_tokens, args.model, 1, args.n_trials)
        if args.mode == "all":
            out["tps_64k"] = bench_tps(args.base, 65536, args.max_tokens, args.model, 0, 1)
            out["tps_256k"] = bench_tps(args.base, 262144, args.max_tokens, args.model, 0, 1)
    if args.mode in ("load", "all"):
        out["load_c8"] = bench_load(args.base, 8, 128, args.model, 2048)
        out["load_c16"] = bench_load(args.base, 16, 128, args.model, 2048)
    if args.mode in ("quality", "all"):
        out["quality"] = bench_quality(args.base, args.model)
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
