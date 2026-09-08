#!/usr/bin/env python3
"""semantic_gate.py — llama.cpp 生产语义 Gate（冷/暖 × 单/多轮 × cache-reuse × DFlash 草稿）。

判据口径出处: eval/vllm/p0/S4-ROOTCAUSE-20260908.md
（连贯备选续写=良性；垃圾/复读/结构破坏=真损坏——历史上从未观察到真损坏）。

四层判定（全确定性规则，零 LLM Judge）:
  EXACT        两轮逐字节一致（strip_thinking 后）
  BENIGN-DIFF  有分歧，且两侧各自: 无乱码、无复读环、结构有效、约束保留、非空，
               长度比∈[0.5,2.0]（G2 首跑前冻结，改阈值必须重跑全矩阵），公共前缀≥20 字符
  UNRESOLVED   有分歧、未触 FAIL、但不满 BENIGN 全部条件 → 人工复核，计入过门阻断
  FAIL         任一侧: 乱码 / 复读环(10-120 字周期紧邻重复≥3 次) / JSON 结构破坏 / 多轮约束丢失 / 空或过短(<10 字)
过门条件: FAIL=0 且 UNRESOLVED=0（冷-冷/冷-暖/多轮/JSON/副本/重启全轴）。
冷-冷轴的非 EXACT 记为"引擎确定性观察"，不自动认定为损坏（S4 结论）。

用法:
  python3 semantic_gate.py probe  [--base http://127.0.0.1:8085] [--ev DIR]
  python3 semantic_gate.py matrix [--base ...] [--ev DIR] [--tag run1] [--only D1@2,D2@4]
  python3 semantic_gate.py canary [--ev DIR] [--lb http://127.0.0.1:8000]
  python3 semantic_gate.py cmp    --ev DIR --a tagA --b tagB     # 跨 tag 比对（重启确定性）

证据文件一律 .jsonl/.txt/.json/.md（仓库 .gitignore 全局排除 *.log）。
"""
import json, os, re, sys, time, urllib.request

MODEL = "qwen3.8-27b"
SEED = 4242
MAXTOK = 350
# ---- 冻结阈值（G2 首跑前定死；调整=候选修改+重跑完整矩阵）----
LEN_LO, LEN_HI = 0.5, 2.0        # BENIGN-DIFF 长度比窗口
PREFIX_MIN = 20                  # 公共前缀最少字符数
REPEAT_PERIOD = (10, 120, 3)     # 复读环: 周期 10-120 字、紧邻重复 ≥3 次（非相邻的结构性重复不算）
GARBAGE_MIN = 0.85               # 常用字符占比下限
TOO_SHORT = 10                   # strip 后正文 <10 字符 = 异常
# ---- 语料与问题（D1/D2 与 vLLM 矩阵同源）----
D1_PATH = "/data/compose/qwen27b/train/imatrix-calib-v11-final.txt"
D2_PATH = "/data/compose/qwen27b/train/calib-v11-autoround.jsonl"
D2_CHARS = 60000
Q1 = "用三句话概括这份文档的主题。"
Q2 = "文档中第一段提到了什么？请引用原文连续 40 字以上。"
MARKER = "ZEBRA-3f9a2c"          # 多轮约束格的校验标记（固定，保证可跨 run 复比）

DEFAULT_EV = "/tmp/gate/semantic-gate-20260908"
ALLOWED_RE = re.compile(
    r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffefA-Za-z0-9 \t\n\r"
    r".,:;!?(){}\[\]\"'`~@#$%^&*+=<>/\\|_\-…—·‘’“”]+")


def http(base, path, body=None, timeout=900):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def strip_think(s: str) -> str:
    if "</think>" in s:
        s = s.split("</think>")[-1]
    return s.strip()


def sha12(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- 判据（全确定性）
def garbage(text: str) -> bool:
    if not text:
        return True
    ok = sum(len(m.group(0)) for m in ALLOWED_RE.finditer(text))
    return ok / len(text) < GARBAGE_MIN


def repeat_loop(text: str) -> bool:
    """紧邻周期性复读（解码坍缩环）: 存在 10-120 字周期连续重复 ≥3 次。
    仅"出现≥3 次"会把工具调用块/列表等结构性合法重复误判（D1 语料实测踩坑），
    故要求紧邻周期 text[i:i+p]==text[i+p:i+2p]==text[i+2p:i+3p]。"""
    n = len(text)
    for p in range(10, 121):
        if n < 3 * p:
            break
        for i in range(0, n - 3 * p + 1):
            if text[i:i + p] == text[i + p:i + 2 * p] == text[i + 2 * p:i + 3 * p]:
                return True
    return False


def json_ok(text: str) -> bool:
    t = strip_think(text).strip()
    try:                       # 先整体解析（最常见: 裸 JSON / JSON 数组）
        json.loads(t); return True
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)  # 再取代码围栏
    if m:
        try:
            json.loads(m.group(1).strip()); return True
        except Exception:
            return False
    for a, b in (("[", "]"), ("{", "}")):   # 最后按最外层括号截取（数组优先，防 {} 截掉 []）
        s, e = t.find(a), t.rfind(b)
        if s != -1 and e > s:
            try:
                json.loads(t[s:e + 1]); return True
            except Exception:
                continue
    return False


def text_defects(text: str, need_json=False, need_marker=False) -> str:
    """返回缺陷名（FAIL 信号），无则空串。"""
    t = strip_think(text)
    if len(t) < TOO_SHORT:
        return "empty/too-short"
    if garbage(t):
        return "garbage"
    if repeat_loop(t):
        return "repeat-loop"
    if need_json and not json_ok(t):
        return "json-broken"
    if need_marker and MARKER not in t:
        return "marker-lost"
    return ""


def common_prefix(x: str, y: str) -> int:
    n = 0
    for a, b in zip(x, y):
        if a != b:
            break
        n += 1
    return n


def classify(x: str, y: str, need_json=False, need_marker=False) -> dict:
    """四层判定。x/y 为原始 content（内部自行 strip_think）。"""
    for t, side in ((x, "A"), (y, "B")):
        d = text_defects(t, need_json, need_marker)
        if d:
            return {"verdict": "FAIL", "reason": f"{side}:{d}"}
    sx, sy = strip_think(x), strip_think(y)
    if sx == sy:
        return {"verdict": "EXACT"}
    ratio = len(sx) / max(1, len(sy))
    pre = common_prefix(sx, sy)
    if LEN_LO <= ratio <= LEN_HI and pre >= PREFIX_MIN:
        return {"verdict": "BENIGN-DIFF", "len_ratio": round(ratio, 3),
                "prefix": pre, "firstdiff_context": diff_ctx(sx, sy)}
    return {"verdict": "UNRESOLVED", "len_ratio": round(ratio, 3), "prefix": pre,
            "reason": "len-ratio 或公共前缀不满足 BENIGN 窗口", "firstdiff_context": diff_ctx(sx, sy)}


def diff_ctx(x: str, y: str) -> str:
    i = common_prefix(x, y)
    return f"@{i}: A=…{x[max(0,i-12):i+18]!r} B=…{y[max(0,i-12):i+18]!r}"


# ---------------------------------------------------------------- 请求面
_USE_TK = [True]  # chat_template_kwargs.enable_thinking 是否被服务端接受


def chat(base, messages, cache=True, max_tokens=MAXTOK):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "seed": SEED, "cache_prompt": bool(cache)}
    t0 = time.time()
    try:
        if _USE_TK[0]:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = http(base, "/v1/chat/completions", body)
    except urllib.error.HTTPError as e:
        if _USE_TK[0] and e.code == 400:  # 旧模板不支持 kwargs → 降级重试
            _USE_TK[0] = False
            body.pop("chat_template_kwargs", None)
            r = http(base, "/v1/chat/completions", body)
        else:
            raise
    msg = r["choices"][0]["message"]
    content = msg.get("content") or ""
    usage = r.get("usage") or {}
    return {"content": content, "wall": round(time.time() - t0, 2),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "stop": r["choices"][0].get("finish_reason"),
            "timings": r.get("timings")}


def tokenize_ids(base, text):
    r = http(base, "/tokenize", {"content": text})
    return r.get("ids") or r.get("tokens")


def detokenize(base, ids):
    return http(base, "/detokenize", {"tokens": ids})["content"]


def save_cell(ev, tag, name, which, text):
    p = os.path.join(ev, "cells", f"{tag}-{name}-{which}.txt")
    with open(p, "w") as f:
        f.write(text)
    return p


def jsonl(ev, fname, obj):
    with open(os.path.join(ev, fname), "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_docs():
    d1 = open(D1_PATH, errors="ignore").read()
    d2 = open(D2_PATH, errors="ignore").read()[:D2_CHARS]
    return {"D1": d1, "D2": d2}


def doc_targets(base):
    """全文 tokenize → 共享目标档网格（锚定较短文档尾部 600 步进×6，两 doc 均从头部截断同目标数）。"""
    docs = load_docs()
    toks = {name: tokenize_ids(base, text) for name, text in docs.items()}
    L = min(len(v) for v in toks.values())
    ts = [L - 600 * (6 - i) for i in range(6)]  # L-3600 … L-600
    out = {}
    for name in docs:
        out[name] = {"ids": toks[name], "L": len(toks[name]), "targets": ts}
        print(f"[tokenize] {name}: 全文 {len(toks[name])} tokens, 共享目标档 {ts} "
              f"(mod128: {[t % 128 for t in ts]})", flush=True)
    return out


# ---------------------------------------------------------------- probe
def cmd_probe(args):
    base = args["base"]; ev = args["ev"]
    os.makedirs(ev, exist_ok=True)
    rep = {"base": base, "ts": time.strftime("%F %T")}
    print("== /props ==", flush=True)
    try:
        props = http(base, "/props")
        rep["props"] = {k: props.get(k) for k in ("model_path", "model_alias", "default_generation_settings")}
        print(json.dumps(rep["props"], ensure_ascii=False)[:400], flush=True)
    except Exception as e:
        rep["props_error"] = str(e); print("props FAIL:", e, flush=True)

    print("== /tokenize + /detokenize ==", flush=True)
    try:
        ids = tokenize_ids(base, "你好，世界 Hello 123")
        back = detokenize(base, ids[:6])
        rep["tokenize"] = {"n": len(ids), "detok_head": back[:20]}
        print(f"tokenize OK n={len(ids)} detok={back[:20]!r}", flush=True)
    except Exception as e:
        rep["tokenize_error"] = str(e); print("tokenize FAIL:", e, flush=True)

    print("== chat_template_kwargs / seed / cache_prompt 接受度 ==", flush=True)
    msgs = [{"role": "user", "content": "只回复两个字：收到"}]
    r = chat(base, msgs, cache=False, max_tokens=16)
    rep["chat_ok"] = True; rep["use_template_kwargs"] = _USE_TK[0]
    rep["response_keys"] = sorted(r.keys()); rep["timings"] = r.get("timings")
    print(f"chat OK kwargs={_USE_TK[0]} keys={rep['response_keys']} timings={r.get('timings')}", flush=True)

    print("== cache_prompt:false 生效性（三段计时）==", flush=True)
    doc = load_docs()["D2"][:12000]
    m = [{"role": "user", "content": doc + "\n\n只回复：OK"}]
    t = {}
    for label, cache in (("cold1", False), ("cold2", False), ("warm_default", None)):
        body = {"model": MODEL, "messages": m, "max_tokens": 1,
                "temperature": 0, "seed": SEED}
        if cache is not None:
            body["cache_prompt"] = cache
        t0 = time.time()
        http(base, "/v1/chat/completions", body)
        t[label] = round(time.time() - t0, 2)
    t["warm_explicit"] = round(chat(base, m, cache=True, max_tokens=1)["wall"], 2)
    rep["cache_timing"] = t
    knob = t["warm_default"] < 0.6 * max(t["cold1"], 0.01) and t["cold2"] >= 0.6 * t["cold1"]
    rep["cache_prompt_knob_effective"] = bool(knob)
    print(json.dumps(t) + f"  → cache_prompt 旋钮有效: {knob}", flush=True)

    try:
        slots = http(base, "/slots")
        rep["slots_sample"] = slots
        with open(os.path.join(ev, "probe-slots.json"), "w") as f:
            json.dump(slots, f, ensure_ascii=False, indent=1)
        print("== /slots 已存 probe-slots.json ==", flush=True)
    except Exception as e:
        rep["slots_error"] = str(e); print("slots FAIL:", e, flush=True)

    try:
        with urllib.request.urlopen(base + "/metrics", timeout=60) as r:
            met = r.read().decode()
        keys = [ln.split()[0] for ln in met.splitlines()
                if ln.startswith("llamacpp:") and ("spec" in ln or "cache" in ln or "prompt" in ln)]
        rep["metric_keys"] = sorted(set(keys))
        print("metrics 计数器:", rep["metric_keys"], flush=True)
    except Exception as e:
        rep["metrics_error"] = str(e)

    with open(os.path.join(ev, "probe.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print("\nprobe 结论: cache_knob=%s tokenize=%s kwargs=%s" % (
        rep.get("cache_prompt_knob_effective"), "tokenize" in rep, _USE_TK[0]), flush=True)


# ---------------------------------------------------------------- matrix
def run_cell(base, ev, tag, name, ids, tidx, include_coldcold=True):
    """一个格: 冷路(a1,a2[,a1b,a2b]) + 暖路(b1,b2) + 暖命中实证。"""
    t = (doc_targets_cache[name]["targets"])[tidx]
    dtext = detokenize(base, ids[:t])
    m1 = [{"role": "user", "content": dtext + "\n\n" + Q1}]
    m2_of = lambda a1: m1 + [{"role": "assistant", "content": a1},
                             {"role": "user", "content": Q2}]
    res = {}
    for which, msgs, cache in (
            ("a1", m1, False), ("a2", m2_of(res["a1"]["content"]) if "a1" in res else None, False),
            ("a1b", m1, False), ("a2b", None, False),
            ("b1", m1, True), ("b2", None, True)):
        if msgs is None:
            if which == "a2":
                msgs = m2_of(res["a1"]["content"])
            elif which == "a2b":
                msgs = m2_of(res["a1b"]["content"])
            elif which == "b2":
                msgs = m2_of(res["b1"]["content"])
        res[which] = chat(base, msgs, cache=cache)
        save_cell(ev, tag, f"{name}@{t}", which, res[which]["content"])

    # 暖命中实证: 主证 = b2 自身 timings.cache_n 占比>0.8（逐 token 级）；辅证 = m1 缓存重放计时
    b2n = ((res["b2"].get("timings") or {}).get("cache_n")) or 0
    b2p = res["b2"]["prompt_tokens"] or 1
    cache_ratio = b2n / max(1, b2p)
    warm_ms = chat(base, m1, cache=True, max_tokens=1)["wall"]
    cold_wall = max(res["a1"]["wall"], res["a1b"]["wall"])
    warm_proven = cache_ratio > 0.8 or warm_ms < 0.6 * cold_wall

    cc1 = classify(res["a1"]["content"], res["a1b"]["content"]) if include_coldcold else {"verdict": "SKIP"}
    cc2 = classify(res["a2"]["content"], res["a2b"]["content"]) if include_coldcold else {"verdict": "SKIP"}
    cw1 = classify(res["a1"]["content"], res["b1"]["content"])
    cw2 = classify(res["a2"]["content"], res["b2"]["content"])

    verdicts = [v["verdict"] for v in (cc1, cc2, cw1, cw2) if v["verdict"] != "SKIP"]
    cell = "FAIL" if "FAIL" in verdicts else \
           "UNRESOLVED" if ("UNRESOLVED" in verdicts or not warm_proven) else \
           "BENIGN-DIFF" if "BENIGN-DIFF" in verdicts else "EXACT"
    entry = {"tag": tag, "doc": name, "target_tokens": t, "residue_mod_128": t % 128,
             "actual_doc_tokens": t,  # 截断即精确 token 数（/tokenize→ids[:t]）
             "prompt_tokens": {k: res[k]["prompt_tokens"] for k in res},
             "warm": {"probe_ms": warm_ms, "cold_wall_s": cold_wall, "proven": warm_proven,
                      "b2_cache_ratio": round(cache_ratio, 4)},
             "cache_n": {k: ((res[k].get("timings") or {}).get("cache_n")) for k in res},
             "cc1": cc1, "cc2": cc2, "cw1": cw1, "cw2": cw2,
             "sha": {k: sha12(strip_think(res[k]["content"])) for k in res},
             "wall": {k: res[k]["wall"] for k in res}, "cell_verdict": cell}
    jsonl(ev, "matrix.jsonl", entry)
    print(f"{name}@{t} (r{t % 128}): {cell}  cc=({cc1['verdict']},{cc2['verdict']}) "
          f"cw=({cw1['verdict']},{cw2['verdict']}) warm={warm_proven} "
          f"t={res['a1']['wall']}/{res['b1']['wall']}s", flush=True)
    if cw2["verdict"] != "EXACT" and "firstdiff_context" in cw2:
        print("   " + cw2["firstdiff_context"], flush=True)
    return entry


def run_multiturn(base, ev, tag, name, ids, tidx, cold):
    """多轮约束格: turn1 埋 ZEBRA 标记(带文档) → turn2 引用 → turn3 复述并保留标记。
    判定: 冷/暖两路缺陷向量【不对称】(一路丢标记另一路保留)才 FAIL；
    两路同样失败 = fixture 弱（如 D1 头部工具转录语料令模型无视指令）→ FIXTURE-WEAK，不计 FAIL。
    fixture 固定用 D2（语料规整，模型可遵循指令）；D1 的稳定性由 6 个残差格覆盖。"""
    t = doc_targets_cache[name]["targets"][tidx]
    dtext = detokenize(base, ids[:t])
    marker = MARKER
    m1 = [{"role": "user", "content":
           f"在本次对话的全部后续回复中，每条回复都必须原样包含校验标记 {marker}"
           f"（不得翻译、改写、省略）。先阅读下面的文档，然后用三句话概括，并在概括末尾附上校验标记。\n\n{dtext}"}]
    m2 = m1 + [{"role": "assistant", "content": ""},
               {"role": "user", "content": Q2}]
    m3 = None
    cache = not cold
    r1 = chat(base, m1, cache=cache)
    m2[1]["content"] = r1["content"]
    r2 = chat(base, m2, cache=cache)
    m3 = m2 + [{"role": "assistant", "content": r2["content"]},
               {"role": "user", "content": "最后再概括一次文档主旨，并保留校验标记。"}]
    r3 = chat(base, m3, cache=cache)
    for i, r in enumerate((r1, r2, r3), 1):
        save_cell(ev, tag, f"{name}@{t}-MT{'cold' if cold else 'warm'}", f"t{i}", r["content"])
    checks = [text_defects(r["content"], need_marker=True) for r in (r1, r2, r3)]
    return checks, (r1, r2, r3)


def mt_verdict(checks_cold, checks_warm):
    """冷/暖两路多轮约束判定: 不对称缺陷=FAIL(稳定性损坏)，对称缺陷=FIXTURE-WEAK，全清=OK。"""
    if not any(checks_cold) and not any(checks_warm):
        return "OK"
    if checks_cold == checks_warm:
        return "FIXTURE-WEAK"
    return "FAIL"


def run_json_cell(base, ev, tag, idx, payload_desc, cold):
    payload = {
        1: ("把下面这份配置清单整理成一个 JSON 对象，键为 server/port/threads/timeout/backlog，"
            "值从文本中提取；只输出 JSON，不要任何其他文字。\n\n"
            "我们的部署方案是：服务器 10.0.0.9，端口 8443，工作线程 32，超时 30 秒， backlog 2048，"
            "另外日志目录 /var/log/srv，重试 3 次。"),
        2: ("只输出一个 JSON 数组，元素为对象 {\"name\":..., \"role\":...}，内容为下面这段话里出现的"
            "角色与职责映射：张三负责编排，李四负责审核，王五负责发布。只输出 JSON。"),
    }[idx]
    m = [{"role": "user", "content": payload_desc + payload}]
    a = chat(base, m, cache=not cold)
    save_cell(ev, tag, f"JSON{idx}", "cold" if cold else "warm", a["content"])
    d = text_defects(a["content"], need_json=True)
    entry = {"tag": tag, "cell": f"JSON{idx}-{'cold' if cold else 'warm'}",
             "defect": d, "sha": sha12(strip_think(a["content"])),
             "verdict": "FAIL" if d else "OK"}
    jsonl(ev, "matrix.jsonl", entry)
    print(f"JSON{idx}[{'cold' if cold else 'warm'}]: {entry['verdict']} defect={d}", flush=True)
    return entry, a["content"]


doc_targets_cache = {}


def cmd_matrix(args):
    base = args["base"]; ev = args["ev"]; tag = args.get("tag") or "run1"
    os.makedirs(os.path.join(ev, "cells"), exist_ok=True)
    lock = os.path.join(ev, ".matrix.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            old = int((open(lock).read().strip() or "0"))
            os.kill(old, 0)
            raise SystemExit(f"另一 matrix 进程 pid={old} 正在使用此证据目录，退出")
        except (ProcessLookupError, ValueError):
            os.remove(lock)  # 陈旧锁接管
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    import atexit
    atexit.register(lambda: os.path.exists(lock) and os.remove(lock))
    info = doc_targets(base)
    doc_targets_cache.update(info)
    only = set(args["only"].split(",")) if args.get("only") else None

    entries = []
    for name in ("D1", "D2"):
        for ti in range(6):
            key = f"{name}@{ti}"
            if only and key not in only:
                continue
            entries.append(run_cell(base, ev, tag, name, info[name]["ids"], ti))
    # 多轮约束格（fixture 固定 D2，tidx=2；D1 头部语料指令敌意，稳定性已由残差格覆盖）
    if not only or "D2@MT" in only:
        checks_c, rs_c = run_multiturn(base, ev, tag, "D2", info["D2"]["ids"], 2, cold=True)
        checks_w, rs_w = run_multiturn(base, ev, tag, "D2", info["D2"]["ids"], 2, cold=False)
        verdict = mt_verdict(checks_c, checks_w)
        t = info["D2"]["targets"][2]
        entry = {"tag": tag, "cell": "D2-multiturn", "doc": "D2", "target_tokens": t,
                 "residue_mod_128": t % 128, "marker": MARKER,
                 "defects_cold": checks_c, "defects_warm": checks_w, "verdict": verdict,
                 "sha_cold": [sha12(strip_think(r["content"])) for r in rs_c],
                 "sha_warm": [sha12(strip_think(r["content"])) for r in rs_w]}
        jsonl(ev, "matrix.jsonl", entry); entries.append(entry)
        print(f"MT[D2@{t}]: {verdict} cold={checks_c} warm={checks_w}", flush=True)
    # JSON 结构格 ×2（冷、暖各一遍）
    for idx in (1, 2):
        if not only or f"JSON{idx}" in only:
            e_c, c_c = run_json_cell(base, ev, tag, idx, "", cold=True)
            e_w, c_w = run_json_cell(base, ev, tag, idx, "", cold=False)
            entries += [e_c, e_w]
            jsonl(ev, "matrix.jsonl", classify(c_c, c_w, need_json=True)
                  | {"tag": tag, "cell": f"JSON{idx}-cold-vs-warm"})

    n = {}
    for e in entries:
        v = e.get("cell_verdict") or e["verdict"]
        n[v] = n.get(v, 0) + 1
    n.setdefault("FAIL", 0); n.setdefault("UNRESOLVED", 0)
    passed = n["FAIL"] == 0 and n["UNRESOLVED"] == 0
    summ = {"tag": tag, "counts": n, "PASS": passed,
            "ts": time.strftime("%F %T")}
    with open(os.path.join(ev, f"summary-{tag}.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(f"\n== 总结 [{tag}]: {json.dumps(n, ensure_ascii=False)}  PASS={passed}", flush=True)


# ---------------------------------------------------------------- canary
def cmd_canary(args):
    ev = args["ev"]; lb = args.get("lb") or "http://127.0.0.1:8000"
    os.makedirs(os.path.join(ev, "cells"), exist_ok=True)
    reps = {"r1": "http://127.0.0.1:8081", "r2": "http://127.0.0.1:8082"}
    info = doc_targets(reps["r1"])
    name, tidx = "D2", 3
    t = info[name]["targets"][tidx]
    dtext = detokenize(reps["r1"], info[name]["ids"][:t])
    m1 = [{"role": "user", "content": dtext + "\n\n" + Q1}]
    out = {}
    for rn, base in reps.items():
        a1 = chat(base, m1, cache=False)
        m2c = m1 + [{"role": "assistant", "content": a1["content"]}, {"role": "user", "content": Q2}]
        a2 = chat(base, m2c, cache=False)
        b1 = chat(base, m1, cache=True)
        m2w = m1 + [{"role": "assistant", "content": b1["content"]}, {"role": "user", "content": Q2}]
        b2 = chat(base, m2w, cache=True)
        out[rn] = {"a1": a1, "a2": a2, "b1": b1, "b2": b2}
        for k in ("a1", "a2", "b1", "b2"):
            save_cell(ev, "canary", f"{rn}-{name}@{t}", k, out[rn][k]["content"])

    axes = {}
    for k in ("a1", "a2", "b1", "b2"):
        axes[f"replica-{k}"] = classify(out["r1"][k]["content"], out["r2"][k]["content"])
    axes["r1-cold-warm-t2"] = classify(out["r1"]["a2"]["content"], out["r1"]["b2"]["content"])
    axes["r2-cold-warm-t2"] = classify(out["r2"]["a2"]["content"], out["r2"]["b2"]["content"])

    # LB 粘滞多轮（同会话=前 512B 相同: 同 doc 前缀）+ 第二会话 key
    try:
        stats0 = http(lb, "/_lbstats")
    except Exception as e:
        stats0 = {"error": str(e)}
    m1lb = [{"role": "user", "content":
             f"在本次对话的全部后续回复中，每条回复都必须原样包含校验标记 {MARKER}。"
             f"阅读文档后用三句话概括。\n\n{dtext}"}]
    l1 = chat(lb, m1lb, cache=True)
    m2lb = m1lb + [{"role": "assistant", "content": l1["content"]}, {"role": "user", "content": Q2}]
    l2 = chat(lb, m2lb, cache=True)
    m3lb = m2lb + [{"role": "assistant", "content": l2["content"]},
                   {"role": "user", "content": "再概括一次并保留校验标记。"}]
    l3 = chat(lb, m3lb, cache=True)
    for i, r in enumerate((l1, l2, l3), 1):
        save_cell(ev, "canary", f"lb-{name}@{t}", f"t{i}", r["content"])
    try:
        stats1 = http(lb, "/_lbstats")
    except Exception as e:
        stats1 = {"error": str(e)}
    mt_defects = [text_defects(r["content"], need_marker=True) for r in (l1, l2, l3)]
    d1doc = detokenize(reps["r1"], info["D1"]["ids"][:info["D1"]["targets"][1]])
    s1 = chat(lb, [{"role": "user", "content": d1doc + "\n\n" + Q1}], cache=True)
    save_cell(ev, "canary", "lb-session2", "t1", s1["content"])

    entry = {"tag": "canary", "doc": name, "target_tokens": t, "residue_mod_128": t % 128,
             "axes": axes, "lb_multiturn_defects": mt_defects,
             "lb_mt_verdict": "FAIL" if any(mt_defects) else "OK",
             "lbstats_delta": {k: (stats1.get(k, 0) - stats0.get(k, 0))
                               if isinstance(stats0, dict) and isinstance(stats0.get(k), (int, float))
                               else None for k in set(stats1) if isinstance(stats1.get(k), (int, float))},
             "ts": time.strftime("%F %T")}
    jsonl(ev, "canary.jsonl", entry)
    print(f"canary 单元: D2@{t} (r{t % 128})", flush=True)
    for k, v in axes.items():
        print(f"  {k}: {v['verdict']}" + (f"  {v.get('firstdiff_context','')}" if v['verdict'] != 'EXACT' else ""), flush=True)
    print(f"  LB 多轮约束: {entry['lb_mt_verdict']} defects={mt_defects} lbstats_delta={entry['lbstats_delta']}", flush=True)


# ---------------------------------------------------------------- cmp（跨 tag，如重启确定性）
def cmd_cmp(args):
    ev = args["ev"]; ta, tb = args["a"], args["b"]
    import glob
    rows = {}
    for p in glob.glob(os.path.join(ev, "cells", f"{ta}-*-a1.txt")):
        stem = os.path.basename(p)[:-4]            # bootA-D1@12361-a1
        key = stem[len(ta) + 1:-3]                 # D1@12361（去尾 -a1）
        f = lambda w: os.path.join(ev, "cells", f"{ta}-{key}-{w}.txt")
        g = lambda w: os.path.join(ev, "cells", f"{tb}-{key}-{w}.txt")
        if all(os.path.exists(x) for x in (f("a1"), g("a1"))):
            need_marker = "MT" in key
            rows[key] = {
                w: classify(open(f(w)).read(), open(g(w)).read(), need_marker=need_marker)["verdict"]
                for w in ("a1", "a2", "b1", "b2")
                if os.path.exists(f(w)) and os.path.exists(g(w))}
    n = {}
    for key, verdicts in sorted(rows.items()):
        print(f"{key}: {verdicts}", flush=True)
        for v in verdicts.values():
            n[v] = n.get(v, 0) + 1
    print(f"\ncmp[{ta} vs {tb}] 汇总: {n}", flush=True)
    jsonl(ev, "matrix.jsonl", {"tag": f"cmp-{ta}-{tb}", "rows": rows, "counts": n})


def main():
    args = {"base": "http://127.0.0.1:8085", "ev": DEFAULT_EV}
    argv = sys.argv[1:]
    if not argv:
        print(__doc__); return 1
    cmd = argv[0]
    it = iter(argv[1:])
    for a in it:
        if a == "--base": args["base"] = next(it)
        elif a == "--ev": args["ev"] = next(it)
        elif a == "--tag": args["tag"] = next(it)
        elif a == "--only": args["only"] = next(it)
        elif a == "--lb": args["lb"] = next(it)
        elif a == "--a": args["a"] = next(it)
        elif a == "--b": args["b"] = next(it)
    {"probe": cmd_probe, "matrix": cmd_matrix, "canary": cmd_canary, "cmp": cmd_cmp}[cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
