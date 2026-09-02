#!/usr/bin/env python3
"""训练数据转换管线 — 13 套数据集 → ms-swift chat 格式三轨输出。

轨① sft-short.jsonl  (~22K, ≤4K tok)  短序列主轨 (代码/工具/agent 混配)
轨② sft-long.jsonl   (~2K, 8-16K tok) 长上下文轨 (完整 agent 轨迹 + 整文件 FIM 指令)
轨③ sft-fim.jsonl    (~8K)             指令式 FIM (前缀/后缀夹中间挖空), 单独训 LoRA

thinking 分轨: 代码推理轨(nemotron-code reasoning / swe-gym) 带 思考块; 工具轨/FIM 轨非 thinking。
卫生红线: /data/eval-rulers 永不入训; MinHash 去重 (Jaccard≥0.7); 课程排序 (易→难)。

用法: python convert_data.py [--out /data/compose/qwen27b/train/data]
"""
import argparse, hashlib, json, math, os, random, re, sys
from pathlib import Path

DS = Path("/data/datasets")
SEED = 42
rng = random.Random(SEED)

def tok_est(s):
    return max(1, len(s) // 4)

def msgs_text(msgs):
    return sum(tok_est(m.get("content") or "") for m in msgs if isinstance(m, dict))

def asst_with_thinking(reasoning, answer):
    """思考样本对齐 qwen3_8 协议: 换行+think 标签包裹 reasoning, 再闭合后接答案。"""
    reasoning = (reasoning or "").strip()
    if reasoning:
        return "\n" + "think" + "ing\n" + reasoning + "\n" + "think" + "ing" + "\n\n" + answer
    return answer

def _removed():
    pass

def fence(code, lang="python"):
    """包代码围栏; 已带围栏的原样返回。"""
    code = code.strip("\n")
    if code.startswith("```"):
        return code
    return f"```{lang}\n{code}\n```"

def clean_tool_msg(m):
    """tool 角色消息: 内容包进标签, 保证 assistant 侧可见。"""
    role = m.get("role")
    if role == "tool":
        return {"role": "user", "content": f"[tool_result from {m.get('name','tool')}]\n{m.get('content','')}"}
    if role == "assistant" and m.get("tool_calls"):
        tc = m["tool_calls"]
        if isinstance(tc, list):
            rendered = json.dumps(tc, ensure_ascii=False)
            m = dict(m)
            m["content"] = (m.get("content") or "") + "\n" + rendered
            del m["tool_calls"]
    return m

# ---------------------------------------------------------------- loaders → samples
# sample: {track, axis, thinking, msgs:[{role,content}], source, id}

def load_nemotron_code(n):
    """Nemotron 编程 SFT: input(msgs)+output+reasoning → thinking 轨。"""
    out = []
    fp = DS / "nemotron-pt/code/code_v1.1.jsonl"
    with open(fp) as f:
        for line in f:
            if len(out) >= n:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            inp = d.get("input") or []
            output = (d.get("output") or "").strip()
            if not output or not inp:
                continue
            user_msgs = [m for m in inp if m.get("role") in ("system", "user", "assistant")]
            if not any(m.get("role") == "user" for m in user_msgs):
                continue
            msgs = []
            sp = (d.get("system_prompt") or "").strip()
            if sp:
                msgs.append({"role": "system", "content": sp})
            msgs += [{"role": m["role"], "content": m.get("content", "")} for m in user_msgs]
            msgs.append({"role": "assistant", "content": asst_with_thinking(d.get("reasoning"), fence(output))})
            if msgs_text(msgs) > 4096:
                continue
            out.append({"track": "short", "axis": "code", "thinking": True,
                        "msgs": msgs, "source": "nemotron-code", "id": f"nem{len(out)}"})
    return out

def load_commitpackft(n):
    """CommitPackFT: bug→fix 整文件对 → 调试/重构轨 (非 thinking)。"""
    langs = sorted((DS / "commitpackft").glob("*/data.jsonl"))
    out = []
    seen = set()
    for fp in rng.sample(langs, len(langs)):
        with open(fp) as f:
            lines = f.readlines()
        rng.shuffle(lines)
        for line in lines:
            if len(out) >= n:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            old, new = d.get("old_contents") or "", d.get("new_contents") or ""
            subj = (d.get("subject") or "").strip()
            msg = (d.get("message") or "").strip()
            if not old or not new or old == new or not subj:
                continue
            if len(old) > 60000 or len(new) > 60000:
                continue
            key = hashlib.md5(new.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            user = (f"The following file has a bug described by this commit message:\n"
                    f"\"{subj}\n{msg}\"\n\n"
                    f"FILE {d.get('new_file') or d.get('old_file','')} (before the fix):\n```\n{old}\n```\n\n"
                    f"Fix the bug. Return ONLY the complete corrected file content in a single code block, no explanation.")
            out.append({"track": "short", "axis": "debug", "thinking": False,
                        "msgs": [{"role": "user", "content": user},
                                 {"role": "assistant", "content": fence(new, d.get("lang") or "python")}],
                        "source": "commitpackft", "id": f"cp{fp.parent.name}-{len(out)}"})
        if len(out) >= n:
            break
    return out

def load_magicoder(n):
    """Magicoder OSS: problem/solution → 代码指令轨。"""
    fp = sorted((DS / "magicoder-oss-75k").glob("*.jsonl"))[0]
    out = []
    with open(fp) as f:
        for line in f:
            if len(out) >= n:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            prob, sol = (d.get("problem") or "").strip(), (d.get("solution") or "").strip()
            if not prob or not sol or tok_est(prob + sol) > 4096:
                continue
            lang = d.get("lang", "python")
            out.append({"track": "short", "axis": "code", "thinking": False,
                        "msgs": [{"role": "user", "content":
                            f"Write {lang} code for the following task. Return only the code in one code block.\n\n{prob}"},
                                 {"role": "assistant", "content": fence(sol, lang)}],
                        "source": "magicoder", "id": f"mg{d.get('index', len(out))}"})
    return out

def load_swe_gym(n):
    """SWE-Gym: 实例 → 补丁生成 (thinking 轨, RFT 双用)。"""
    import pandas as pd
    df = pd.read_parquet(DS / "swe-gym/train.parquet")
    out = []
    for _, row in df.head(n * 3).iterrows():
        if len(out) >= n:
            break
        ps = (row.get("problem_statement") or "").strip()
        patch = (row.get("patch") or "").strip()
        if not ps or not patch or tok_est(ps + patch) > 4096:
            continue
        user = (f"You are an expert software engineer. A GitHub issue was filed in a repository. "
                f"Read the issue and write a patch that resolves it. Return only the unified diff.\n\n"
                f"REPO: {row.get('repo','')}\nISSUE:\n{ps}\n\nProduce the fix as a unified diff patch.")
        out.append({"track": "short", "axis": "debug", "thinking": True,
                    "msgs": [{"role": "user", "content": user},
                             {"role": "assistant", "content": asst_with_thinking(
                                 "Analyze the issue, locate the faulty code path, and design a minimal patch.", patch)}],
                    "source": "swe-gym", "id": str(row.get("instance_id", f"sg{len(out)}"))})
    return out

def load_xlam(n):
    """xlam-fc-60k: query+tools+answers → 单轮工具调用 (非 thinking)。"""
    d = json.load(open(DS / "xlam-fc-60k/xlam_function_calling_60k.json"))
    out = []
    for item in rng.sample(d, min(n * 2, len(d))):
        if len(out) >= n:
            break
        query = (item.get("query") or "").strip()
        tools = item.get("tools")
        if not query or not tools:
            continue
        try:
            tools_obj = json.loads(tools) if isinstance(tools, str) else tools
            answers = json.loads(item["answers"]) if isinstance(item.get("answers"), str) else item.get("answers")
        except Exception:
            continue
        if not answers:
            continue
        calls = "\n".join(f"{a['name']}({', '.join(f'{k}={json.dumps(v)}' for k, v in (a.get('arguments') or {}).items())})"
                          for a in answers)
        system = ("You are a helpful assistant that answers questions by calling functions. "
                  "Here are the function definitions in JSON:\n"
                  + json.dumps(tools_obj, ensure_ascii=False) +
                  "\n\nGiven the user's question, write the exact function call(s) in a python code block. "
                  "Only call defined functions. If none is relevant, say so without calling.")
        out.append({"track": "short", "axis": "tool", "thinking": False,
                    "msgs": [{"role": "system", "content": system},
                             {"role": "user", "content": query},
                             {"role": "assistant", "content": f"```python\n{calls}\n```"}],
                    "source": "xlam-fc", "id": f"xl{item.get('id', len(out))}"})
    return out

def load_toolace(n):
    """ToolACE: 多轮工具对话 {system, conversations}。"""
    d = json.load(open(DS / "toolace/data.json"))
    out = []
    for item in rng.sample(d, min(n * 2, len(d))):
        if len(out) >= n:
            break
        conv = item.get("conversations") or []
        if len(conv) < 2:
            continue
        msgs = []
        if item.get("system"):
            msgs.append({"role": "system", "content": item["system"]})
        for m in conv:
            role = m.get("from") or m.get("role")
            content = m.get("value") or m.get("content") or ""
            if role in ("human", "user") :
                msgs.append({"role": "user", "content": content})
            elif role in ("gpt", "assistant", "function", "tool"):
                r = "assistant" if role in ("gpt", "assistant") else "user"
                if r == "user":
                    content = f"[tool_result]\n{content}"
                msgs.append({"role": r, "content": content})
        if len(msgs) < 2 or not any(m["role"] == "assistant" for m in msgs):
            continue
        if msgs_text(msgs) > 4096:
            continue
        out.append({"track": "short", "axis": "tool", "thinking": False,
                    "msgs": msgs, "source": "toolace", "id": f"ta{len(out)}"})
    return out

def load_pivot(fp_name, n, axis="agent"):
    """Nemotron Pivot: {responses_create_params.input, expected_action} → agent 轨迹切片。"""
    fp = DS / "nemotron-pt" / fp_name / "train.jsonl"
    out = []
    with open(fp) as f:
        for line in f:
            if len(out) >= n:
                break
            try:
                d = json.loads(line)
                inp = (d.get("responses_create_params") or {}).get("input") or []
                act = (d.get("expected_action") or "").strip()
            except Exception:
                continue
            if not inp or not act:
                continue
            msgs = []
            for m in inp:
                role = m.get("role")
                if role in ("system", "user", "assistant"):
                    msgs.append({"role": role, "content": m.get("content") or ""})
                elif role == "tool":
                    msgs.append({"role": "user", "content": f"[tool_result]\n{m.get('content','')}"})
            if not any(m["role"] == "user" for m in msgs):
                continue
            msgs.append({"role": "assistant", "content": act})
            if msgs_text(msgs) > 4096:
                continue
            out.append({"track": "short", "axis": axis, "thinking": False,
                        "msgs": msgs, "source": fp_name, "id": str(d.get("trajectory_id", len(out)))})
    return out

def load_interactive(n, long_only=False):
    """Nemotron interactive-agent: {messages, tools} 标准 chat+tools。long_only=True 只取 ≥8K。"""
    fp = DS / "nemotron-pt/interactive-agent/interactive_agent.jsonl"
    out = []
    with open(fp) as f:
        for line in f:
            if len(out) >= n:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            msgs = [clean_tool_msg(m) for m in (d.get("messages") or [])]
            msgs = [m for m in msgs if m.get("role") in ("system", "user", "assistant") and m.get("content") is not None]
            t = msgs_text(msgs)
            if t > 16000 or t < 200:
                continue
            if not any(m["role"] == "assistant" for m in msgs):
                continue
            if (t >= 8000) != long_only:
                continue
            out.append({"track": "long" if long_only else "short",
                        "axis": "agent", "thinking": False,
                        "msgs": msgs, "tools": d.get("tools"),
                        "source": "interactive-agent", "id": f"ia{len(out)}"})
    return out

def load_menv(n, min_tok=8000):
    """MEnvData SWE 轨迹: {tools, messages, docker_image} → 长轨 + RFT 沙箱底料。"""
    fp = DS / "menv-swe-trajectory/final_trajectories.jsonl"
    out = []
    with open(fp) as f:
        for line in f:
            if len(out) >= n:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            msgs = [clean_tool_msg(m) for m in (d.get("messages") or [])]
            msgs = [m for m in msgs if m.get("role") in ("system", "user", "assistant") and m.get("content") is not None]
            if msgs_text(msgs) < min_tok:
                continue
            if not any(m["role"] == "assistant" for m in msgs):
                continue
            out.append({"track": "long", "axis": "agent", "thinking": False,
                        "msgs": msgs, "tools": d.get("tools"),
                        "source": "menv-swe", "id": f"me{len(out)}"})
    return out

def load_fim(n_per_src=2500):
    """指令式 FIM: nemotron solution / commitpackft 文件 / magicoder 解 → 前缀+后缀夹挖空。"""
    out = []
    def make_fim(text, lang, source, idx):
        lines = text.splitlines()
        if len(lines) < 10 or tok_est(text) < 600 or tok_est(text) > 6000:
            return None
        p = max(3, int(len(lines) * 0.3))
        s = max(2, int(len(lines) * 0.2))
        if p + s >= len(lines):
            return None
        prefix, middle, suffix = "\n".join(lines[:p]), "\n".join(lines[p:-s]), "\n".join(lines[-s:])
        user = (f"Complete the missing middle part of this {lang} code. Do not repeat the beginning or the end.\n"
                f"BEGIN\n{prefix}\nMIDDLE\n<MISSING>\nEND\n{suffix}\n\nOutput only the missing middle code in one code block.")
        out.append({"track": "fim", "axis": "code", "thinking": False,
                    "msgs": [{"role": "user", "content": user},
                             {"role": "assistant", "content": f"```{lang}\n{middle}\n```"}],
                    "source": source, "id": f"fim-{source}-{idx}"})
        return True
    # nemotron solutions
    cnt = 0
    with open(DS / "nemotron-pt/code/code_v1.1.jsonl") as f:
        for line in f:
            if cnt >= n_per_src:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            sol = (d.get("output") or "").strip()
            if sol.count("\n") >= 10 and make_fim(sol, "python", "nemotron", cnt) :
                cnt += 1
    # commitpackft python files
    fp = DS / "commitpackft/python/data.jsonl"
    cnt = 0
    with open(fp) as f:
        for line in f:
            if cnt >= n_per_src:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            new = d.get("new_contents") or ""
            if new.count("\n") >= 10 and make_fim(new, "python", "commitpackft", cnt):
                cnt += 1
    # magicoder
    fp = sorted((DS / "magicoder-oss-75k").glob("*.jsonl"))[0]
    cnt = 0
    with open(fp) as f:
        for line in f:
            if cnt >= n_per_src:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            sol = (d.get("solution") or "").strip()
            lang = d.get("lang", "python")
            if sol.count("\n") >= 10 and make_fim(sol, lang, "magicoder", cnt):
                cnt += 1
    return out

# ---------------------------------------------------------------- MinHash 去重

import numpy as np

_DEDUP_RNG = np.random.RandomState(42)
_DEDUP_AB = None

def _ab_pairs(nperm):
    global _DEDUP_AB
    if _DEDUP_AB is None or len(_DEDUP_AB) < nperm:
        _DEDUP_AB = _DEDUP_RNG.randint(1, (1 << 61) - 1, size=(512, 2)).astype(np.uint64)
    return _DEDUP_AB[:nperm]

def minhash_sig(text, nperm=32):
    """向量化 MinHash: ~4000 个 10 字节 n-gram 哈希 + nperm 个独立置换 (a*s+b) mod 2^64 取 min。"""
    t = re.sub(r"\s+", " ", text.lower())
    b = t.encode("utf-8", "ignore")
    L = len(b)
    if L < 12:
        return np.zeros(nperm, dtype=np.uint64)
    n = L - 9
    step = max(1, n // 4000)
    starts = np.arange(0, n, step, dtype=np.int64)
    starts = np.clip(starts, 0, L - 10)
    idx = starts[:, None] + np.arange(10)[None, :]
    grams = np.frombuffer(b, dtype=np.uint8).take(idx)
    hi = ((grams[:, 0].astype(np.uint64) << 56) | (grams[:, 1].astype(np.uint64) << 48) |
          (grams[:, 2].astype(np.uint64) << 40) | (grams[:, 3].astype(np.uint64) << 32) |
          (grams[:, 4].astype(np.uint64) << 24) | (grams[:, 5].astype(np.uint64) << 16) |
          (grams[:, 6].astype(np.uint64) << 8) | grams[:, 7].astype(np.uint64))
    lo = (grams[:, 8].astype(np.uint64) << 8) | grams[:, 9].astype(np.uint64)
    sh = hi ^ (lo * np.uint64(0x9E3779B97F4A7C15))
    ab = _ab_pairs(nperm)
    sig = np.empty(nperm, dtype=np.uint64)
    for i in range(nperm):
        sig[i] = np.min(ab[i, 0] * sh + ab[i, 1])
    return sig

def dedup(samples, threshold=0.75):
    """LSH banding 粗筛 (band 命中 → 候选) + 32 位签名精比 (Jaccard≥threshold 判重)。
    全文哈希 (user+assistant), 避免代码结尾样板导致的误判。"""
    print(f"dedup: {len(samples)} samples ...", flush=True)
    nperm, bands = 32, 8
    for s in samples:
        full = "\n".join((m.get("content") or "") for m in s["msgs"])
        s["_sig"] = minhash_sig(full[:20000], nperm=nperm)  # 截断防爆 (长轨迹)
    band_index = {}   # (band_i, key_tuple) -> kept 下标
    kept = []
    removed = 0
    for s in samples:
        sig = s["_sig"]
        cands = set()
        for i in range(bands):
            key = (i, tuple(sig[i * 4:(i + 1) * 4]))
            if key in band_index:
                cands.add(band_index[key])
        dup = False
        for ci in cands:
            inter = int(np.count_nonzero(sig == kept[ci]["_sig"]))
            if inter >= threshold * nperm:
                dup = True
                break
        if dup:
            removed += 1
            continue
        idx = len(kept)
        for i in range(bands):
            band_index.setdefault((i, tuple(sig[i * 4:(i + 1) * 4])), idx)
        kept.append(s)
    for s in kept:
        del s["_sig"]
    print(f"dedup: kept {len(kept)} / {len(samples)} (removed {removed})", flush=True)
    return kept

# ---------------------------------------------------------------- main

CURRICULUM = [("code", False), ("code", True), ("debug", False), ("debug", True),
              ("tool", False), ("agent", False), ("agent", True)]

def curriculum_sort(samples):
    rank = {(a, t): i for i, (a, t) in enumerate(CURRICULUM)}
    for i, s in enumerate(samples):
        s["_order"] = (rank.get((s["axis"], s["thinking"]), 99), i)
    samples.sort(key=lambda s: s["_order"])
    for s in samples:
        del s["_order"]
    return samples

def cap_long(sample, max_tok=14500, head_tok=7000, tail_tok=7000):
    """长轨头尾拼接: 保前 head_tok + 后 tail_tok, 中间省略标记桥接, 确保任务上下文与最终答案都在。"""
    msgs = sample["msgs"]
    if msgs_text(msgs) <= max_tok:
        return sample

    def mt(m):
        return tok_est(m.get("content") or "")

    def split_msg(m, budget=14000):
        c = m.get("content") or ""
        if tok_est(c) <= budget:
            return m
        return dict(m, content=c[: head_tok * 4] + "\n\n[…middle omitted…]\n\n" + c[-tail_tok * 4:])

    msgs = [split_msg(m) for m in msgs]
    if msgs_text(msgs) <= max_tok:
        sample["msgs"] = msgs
        return sample

    head, acc = [], 0
    for m in msgs:
        t = mt(m)
        if acc + t > head_tok and head:
            break
        head.append(m)
        acc += t
    tail, acc = [], 0
    for m in reversed(msgs):
        t = mt(m)
        if acc + t > tail_tok and tail:
            break
        tail.insert(0, m)
        acc += t
    tail_ids = set(map(id, tail))
    merged = []
    for m in head:
        if id(m) in tail_ids:
            break
        merged.append(m)
    if not merged:
        merged = [msgs[0]]
    if merged[-1]["role"] == tail[0]["role"]:
        merged.append({"role": "user", "content": "[system: middle steps omitted, continue]"})
    new_msgs = merged + tail
    if new_msgs[-1]["role"] != "assistant":
        new_msgs.append({"role": "assistant", "content": "[end]"})
    sample["msgs"] = new_msgs
    return sample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/compose/qwen27b/train/data")
    ap.add_argument("--skip-dedup", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("== loading short track ==", flush=True)
    short = []
    short += load_nemotron_code(4000)
    print(f"  nemotron-code: {len(short)}", flush=True)
    short += load_magicoder(5000)
    short += load_commitpackft(3000)
    short += load_swe_gym(2438)
    short += load_xlam(2000)
    short += load_toolace(1500)
    short += load_pivot("fc-pivot", 1000, axis="tool")
    short += load_interactive(2000, long_only=False)
    short += load_pivot("swe-pivot", 500, axis="debug")
    short += load_pivot("terminal-pivot", 500, axis="agent")
    print(f"  short total: {len(short)}", flush=True)

    print("== loading long track ==", flush=True)
    long_ = load_menv(1200)
    long_ += load_interactive(800, long_only=True)
    print(f"  long total: {len(long_)}", flush=True)

    print("== loading FIM track ==", flush=True)
    fim = load_fim()
    print(f"  fim total: {len(fim)}", flush=True)

    all_samples = short + long_ + fim
    if not args.skip_dedup:
        all_samples = dedup(all_samples)
    short = curriculum_sort([s for s in all_samples if s["track"] == "short"])
    long_ = [cap_long(s) for s in all_samples if s["track"] == "long"]
    fim = [s for s in all_samples if s["track"] == "fim"]
    rng.shuffle(long_)
    rng.shuffle(fim)

    def dump(name, samples):
        fp = out / name
        with open(fp, "w") as f:
            for s in samples:
                rec = {"messages": s["msgs"], "source": s["source"], "axis": s["axis"], "id": s["id"]}
                if s.get("tools"):
                    rec["tools"] = json.dumps(s["tools"], ensure_ascii=False)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  wrote {fp}: {len(samples)}", flush=True)

    dump("sft-short.jsonl", short)
    dump("sft-long.jsonl", long_)
    dump("sft-fim.jsonl", fim)

    # 抽检 50 条
    with open(out / "sample50.jsonl", "w") as f:
        for s in rng.sample(all_samples, min(50, len(all_samples))):
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 清单
    manifest = {
        "date": os.popen("date '+%F %T'").read().strip(),
        "short": len(short), "long": len(long_), "fim": len(fim),
        "by_source": {}, "by_axis": {},
    }
    for s in all_samples:
        manifest["by_source"][s["source"]] = manifest["by_source"].get(s["source"], 0) + 1
        manifest["by_axis"][s["axis"]] = manifest["by_axis"].get(s["axis"], 0) + 1
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: manifest[k] for k in ("short", "long", "fim", "by_source")}, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
