#!/usr/bin/env python3
"""Qwen3.8-27B 门禁评测器 — 基线(Phase 0)与后续门禁(Phase 4/6)共用同一套尺子。

四能力轴 + 通用 + 长上下文:
  ① 代码生成  HumanEval-164  pass@1 (temp=0, 沙箱执行)
  ③ 工具调用  BFCL v3 单轮核心集 ~1950  AST 精确匹配 ground_truth
  ④ 通用推理  GSM8K-100 (temp=0, 最终数字精确匹配)
  ④ 指令遵循  IFEval-50 (约束检查器)
  长上下文    Needle@64K/195K (3 探针×3 次) + 2048-token 长生成
  MTP          /metrics 差分测接受率 (2048-token 长生成前后)

用法:
  python run_baseline.py all      --tag baseline-3.0
  python run_baseline.py humaneval bfcl gsm8k ifeval needle mtp --tag gate-4.0
  (EVAL_BASE 默认 http://127.0.0.1:8000/v1; --max 可缩减题量冒烟)

输出: /data/eval-rulers/<tag>.json  (全轴分数 + 每轴明细 + meta)
"""
import argparse, ast, concurrent.futures as cf, json, os, random, re, subprocess, sys, tempfile, time
from pathlib import Path

import requests

EVAL_BASE = os.environ.get("EVAL_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("EVAL_MODEL", "qwen3.8-27b")
RULERS = Path("/data/eval-rulers")
SEED = 42
TIMEOUT = 60

# ---------------------------------------------------------------- API

def chat(messages, temperature=0.0, max_tokens=2048, retries=3):
    url = f"{EVAL_BASE}/chat/completions"
    payload = {"model": MODEL, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens}
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)

def strip_thinking(text):
    """思考型模型: 取 </think> 之后为正文; 无标记则整体。"""
    idx = text.rfind("</think>")
    return text[idx + len("</think>"):].strip() if idx >= 0 else text.strip()

def extract_code(text, lang="python"):
    body = strip_thinking(text)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, re.S)
    if blocks:
        return max(blocks, key=len)
    if "def " in body or "import " in body:
        return body
    return ""

# ---------------------------------------------------------------- ① HumanEval

def run_humaneval(limit=0, workers=16):
    import pandas as pd
    df = pd.read_parquet(RULERS / "humaneval" / "test.parquet")
    if limit:
        df = df.head(limit)
    results = []

    def one(row):
        prompt = row["prompt"]
        try:
            resp = chat([{"role": "user", "content":
                "Complete the following Python function. Return only the code in a single python code block, no explanation.\n" + prompt}],
                temperature=0.0, max_tokens=3072)
            code = extract_code(resp)
            passed, detail = run_humaneval_test(row, code)
        except Exception as e:
            passed, detail = False, f"error: {e}"
        return {"task_id": row["task_id"], "pass": passed, "detail": detail}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, df.to_dict("records"))):
            results.append(r)
            if (i + 1) % 20 == 0:
                print(f"  humaneval {i+1}/{len(df)} pass={sum(x['pass'] for x in results)}", flush=True)
    passed = sum(r["pass"] for r in results)
    return {"pass_at_1": passed / len(results) if results else 0.0,
            "passed": passed, "total": len(results), "results": results}

def run_humaneval_test(row, code):
    if not code:
        return False, "no_code"
    entry = row["entry_point"]
    if entry not in code:
        return False, "entry_point_missing"
    test_src = row["test"]
    # test 里的 assert 调用会执行 entry_point(...)
    src = "from typing import List, Dict\n" + code + "\n" + test_src + "\n"
    with tempfile.TemporaryDirectory() as td:
        pf = Path(td) / "t.py"
        pf.write_text(src)
        try:
            p = subprocess.run([sys.executable, "-I", str(pf)], capture_output=True,
                               text=True, timeout=15, cwd=td)
            if p.returncode == 0:
                return True, "ok"
            return False, (p.stderr.strip().splitlines() or ["fail"])[-1][:200]
        except subprocess.TimeoutExpired:
            return False, "timeout"

# ---------------------------------------------------------------- ③ BFCL

BFCL_CORE = ["BFCL_v3_simple", "BFCL_v3_exec_simple", "BFCL_v3_multiple",
             "BFCL_v3_exec_multiple", "BFCL_v3_parallel", "BFCL_v3_parallel_multiple",
             "BFCL_v3_exec_parallel", "BFCL_v3_exec_parallel_multiple",
             "BFCL_v3_irrelevance", "BFCL_v3_chatable", "BFCL_v3_rest",
             "BFCL_v3_java", "BFCL_v3_javascript", "BFCL_v3_sql"]

def bfcl_render_funcs(funcs):
    out = []
    for f in funcs:
        params = f.get("parameters") or {}
        props = params.get("properties") or {}
        req = params.get("required") or []
        sig = []
        for name, p in props.items():
            t = p.get("type", "any")
            if name not in req:
                t = f"{t} = None"
            sig.append(f"{name}: {t}")
        fn = f["name"].replace(".", "_")
        out.append(f"def {fn}({', '.join(sig)}):\n    \"\"\"{f.get('description','')}\"\"\"\n    pass")
    return "\n\n".join(out)

def parse_bfcl_call(text):
    """从模型回复提取函数调用 AST Call 节点列表。"""
    body = strip_thinking(text)
    calls = []
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", body, re.S)
    candidates = blocks[-1] if blocks else body
    # 优先解析代码块; 失败则用正则兜底
    try:
        tree = ast.parse(candidates.strip())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                calls.append(node)
        if calls:
            return calls
    except Exception:
        pass
    m = re.findall(r"([A-Za-z_][\w.]*)\s*\(([^()]*)\)", body)
    for name, args in m:
        try:
            tree = ast.parse(f"{name}({args})", mode="eval")
            calls.append(tree.body)
        except Exception:
            continue
    return calls

def norm_call(call):
    if isinstance(call, ast.Call):
        f = call.func
        name = ast.unparse(f) if isinstance(f, ast.Attribute) else getattr(f, "id", ast.unparse(f))
        args = [norm_arg(a) for a in call.args]
        kwargs = {kw.arg: norm_arg(kw.value) for kw in call.keywords if kw.arg}
        return (name, tuple(args), tuple(sorted(kwargs.items())))
    return (ast.unparse(call), (), ())

def norm_arg(a):
    if isinstance(a, (ast.Constant,)):
        v = a.value
        if isinstance(v, float):
            return round(v, 6)
        return v
    if isinstance(a, ast.List):
        return [norm_arg(x) for x in a.elts]
    if isinstance(a, ast.Dict):
        return {norm_arg(k): norm_arg(v) for k, v in zip(a.keys, a.values)}
    try:
        return ast.literal_eval(ast.unparse(a))
    except Exception:
        return ast.unparse(a)

def norm_gt(gt_str):
    """ground_truth 字符串 → 规范 call 元组 (参数值可能是字符串化的 list)。"""
    gt_str = gt_str.strip()
    try:
        node = ast.parse(gt_str, mode="eval").body
        c = norm_call(node)
        if c:
            return c
    except Exception:
        pass
    return gt_str

def bfcl_score(pred_calls, pred_raw, gts, result_type):
    if not gts:
        # irrelevance/rest/chatable: 正确行为是不调用
        return not pred_calls
    # AST 路线: 双方都能规范化为 (name, args, kwargs) 元组时做集合比较
    gt_norms = [norm_gt(g) for g in gts]
    if all(isinstance(n, tuple) for n in gt_norms) and pred_calls:
        pred_norms = [n for n in (norm_call(c) for c in pred_calls) if isinstance(n, tuple)]
        return set(gt_norms) == set(pred_norms)
    # 字符串回退 (java/js/sql 语言调用、复杂参数): 归一化空白后比对
    pred_flat = " ".join(ast.unparse(c) for c in pred_calls if isinstance(c, ast.Call))
    if not pred_flat:
        body = strip_thinking(pred_raw)
        blocks = re.findall(r"```\w*\s*\n(.*?)```", body, re.S)
        pred_flat = blocks[-1] if blocks else body
    pred_norm = " ".join(pred_flat.split()).lower()
    if len(gts) == 1:
        return " ".join(gts[0].split()).lower() == pred_norm
    return all(" ".join(g.split()).lower() in pred_norm for g in gts)

def run_bfcl(limit=0, workers=8):
    rows = []
    for name in BFCL_CORE:
        fp = RULERS / "bfcl" / f"{name}.json"
        if not fp.exists():
            continue
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append((name, json.loads(line)))
    if limit:
        rows = rows[:limit]
    results = []

    def one(item):
        name, d = item
        try:
            q = ast.literal_eval(d["question"])
            if isinstance(q[0], list):
                q = q[0]
            funcs = ast.literal_eval(d["function"])
            gt = ast.literal_eval(d["ground_truth"]) if d.get("ground_truth") else []
            rtype = d.get("execution_result_type") or []
            if isinstance(rtype, list):
                rtype = "|".join(rtype)
            fdef = bfcl_render_funcs(funcs)
            msgs = [{"role": "system", "content":
                f"You are a helpful assistant that answers questions by calling functions. "
                f"Here are the function definitions:\n\n{fdef}\n\n"
                f"Given the user's question, write the exact function call(s) in a python code block. "
                f"Only call functions that are defined. If none is relevant, say so without calling."}]
            for m in q:
                if m.get("role") == "user":
                    msgs.append({"role": "user", "content": m.get("content", "")})
            resp = chat(msgs, temperature=0.0, max_tokens=1024)
            pred = parse_bfcl_call(resp)
            ok = bfcl_score(pred, resp, gt, rtype)
        except Exception as e:
            ok = False
        return {"id": d.get("id"), "file": name, "pass": ok}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, rows)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"  bfcl {i+1}/{len(rows)} pass={sum(x['pass'] for x in results)}", flush=True)
    passed = sum(r["pass"] for r in results)
    per_file = {}
    for r in results:
        per_file.setdefault(r["file"], [0, 0])
        per_file[r["file"]][1] += 1
        per_file[r["file"]][0] += int(r["pass"])
    return {"accuracy": passed / len(results) if results else 0.0,
            "passed": passed, "total": len(results),
            "per_file": {k: f"{v[0]}/{v[1]}" for k, v in per_file.items()},
            "results": results}

# ---------------------------------------------------------------- ④ GSM8K

def run_gsm8k(limit=100, workers=12):
    import pandas as pd
    limit = limit or 100
    df = pd.read_parquet(RULERS / "gsm8k" / "test.parquet")
    rng = random.Random(SEED)
    idx = rng.sample(range(len(df)), min(limit, len(df)))
    rows = df.iloc[idx].to_dict("records")
    results = []

    def one(row):
        gt = row["answer"].split("####")[-1].strip().replace("$", "").replace(",", "")
        try:
            resp = chat([{"role": "user", "content":
                row["question"] + "\n\nSolve step by step. End with the final numeric answer only."}],
                temperature=0.0, max_tokens=2048)
            nums = re.findall(r"-?\d[\d,]*\.?\d*", strip_thinking(resp).replace(",", ""))
            pred = nums[-1] if nums else None
            ok = pred is not None and float(gt) == float(pred)
        except Exception as e:
            pred, ok = None, False
        return {"q": row["question"][:60], "gt": gt, "pred": pred, "pass": ok}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, rows)):
            results.append(r)
            if (i + 1) % 25 == 0:
                print(f"  gsm8k {i+1}/{len(rows)} pass={sum(x['pass'] for x in results)}", flush=True)
    passed = sum(r["pass"] for r in results)
    return {"accuracy": passed / len(results) if results else 0.0,
            "passed": passed, "total": len(results), "results": results}

# ---------------------------------------------------------------- ④ IFEval

def ifeval_check(item, response):
    """官方 IFEval 全 25 类约束本地检查; 未覆盖返回 None(跳过该条)。"""
    body = strip_thinking(response).strip()
    if not body:
        return False
    checks = item["instruction_id_list"]
    kwlist = item.get("kwargs") or {}
    if isinstance(kwlist, dict):
        kwlist = [kwlist] * len(checks)
    n_sent = len([s for s in re.split(r"[.!?]+", body) if s.strip()])
    n_para = len([p for p in body.split("***") if p.strip()])
    bullets = len(re.findall(r"^\s*\*\s+\S", body, re.M))
    highlights = len(re.findall(r"\*[^*\n]+\*", body))
    for i, cid in enumerate(checks):
        kw = kwlist[i] if i < len(kwlist) and isinstance(kwlist[i], dict) else {}
        rel = kw.get("relation") or kw.get("let_relation") or kw.get("capital_relation")
        def _cmp(n, target):
            if rel == "less than":
                return n < target
            return n >= target
        if cid == "punctuation:no_comma":
            if "," in body: return False
        elif cid == "length_constraints:number_words":
            if not _cmp(len(body.split()), kw.get("num_words", 0)): return False
        elif cid == "length_constraints:number_sentences":
            if not _cmp(n_sent, kw.get("num_sentences", 0)): return False
        elif cid == "length_constraints:number_paragraphs":
            if n_para < kw.get("num_paragraphs", 0): return False
        elif cid == "length_constraints:nth_paragraph_first_word":
            paras = [p for p in body.split("***") if p.strip()]
            n = kw.get("nth_paragraph", 1) - 1
            if n >= len(paras) or not paras[n].strip().split() or \
               not paras[n].strip().split()[0].lower().startswith(str(kw.get("first_word", "")).lower()):
                return False
        elif cid == "keywords:existence":
            if any(str(k).lower() not in body.lower() for k in kw.get("keywords", [])): return False
        elif cid == "keywords:forbidden_words":
            if any(str(k).lower() in body.lower() for k in kw.get("forbidden_words", [])): return False
        elif cid == "keywords:frequency":
            if not _cmp(body.lower().count(str(kw.get("keyword", "")).lower()), kw.get("frequency", 0)): return False
        elif cid == "keywords:letter_frequency":
            if not _cmp(body.lower().count(str(kw.get("letter", "")).lower()), kw.get("let_frequency", 0)): return False
        elif cid == "language:response_language":
            lang = str(kw.get("language", "")).lower()
            if lang in ("chinese", "zh"):
                if not re.search(r"[\u4e00-\u9fff]", body): return False
            elif lang == "english":
                if not re.search(r"[a-zA-Z]{3,}", body): return False
        elif cid == "change_case:english_lowercase":
            if any(c.isupper() for c in body): return False
        elif cid == "change_case:english_capital":
            if any(c.islower() for c in body): return False
        elif cid == "change_case:capital_word_frequency":
            caps = len(re.findall(r"\b[A-Z]{2,}\b", body))
            if not _cmp(caps, kw.get("capital_frequency", 0)): return False
        elif cid == "startend:quotation":
            if not (body.startswith('"') and body.endswith('"')): return False
        elif cid == "startend:end_checker":
            if not body.lower().endswith(str(kw.get("end_phrase", "")).lower()): return False
        elif cid == "detectable_content:number_placeholders":
            if body.count("[") < kw.get("num_placeholders", 0): return False
        elif cid == "detectable_content:postscript":
            if str(kw.get("postscript_marker", "P.S.")).lower() not in body.lower(): return False
        elif cid == "detectable_format:number_bullet_lists":
            if bullets < kw.get("num_bullet_lists", 0): return False
        elif cid == "detectable_format:number_highlighted_sections":
            if highlights < kw.get("num_highlights", 0): return False
        elif cid == "detectable_format:title":
            if not ("<<" in body and ">>" in body): return False
        elif cid == "detectable_format:json_format":
            m = re.search(r"\{.*\}", body, re.S)
            if not m: return False
            try:
                json.loads(m.group(0))
            except Exception:
                return False
        elif cid == "detectable_format:multiple_sections":
            if body.count(str(kw.get("section_spliter", "Section"))) < kw.get("num_sections", 0): return False
        elif cid == "detectable_format:constrained_response":
            if not any(p in body for p in ("My answer is yes.", "My answer is no.", "My answer is maybe.")): return False
        elif cid == "combination:two_responses":
            if len([p for p in body.split("**************") if p.strip()]) < 2: return False
        elif cid == "combination:repeat_prompt":
            seg = body.split("***")[0]
            if str(kw.get("prompt_to_repeat", "")).strip().lower() not in seg.lower(): return False
        else:
            return None
    return True

def run_ifeval(limit=50, workers=12):
    limit = limit or 50
    items = []
    with open(RULERS / "ifeval" / "ifeval.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    rng = random.Random(SEED)
    items = rng.sample(items, min(limit, len(items)))
    results = []

    def one(item):
        try:
            resp = chat([{"role": "user", "content": item["prompt"]}],
                        temperature=0.0, max_tokens=6144)
            ok = ifeval_check(item, resp)
        except Exception as e:
            ok = False
        return {"key": item.get("key"), "pass": ok, "skipped": ok is None}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, items)):
            results.append(r)
            if (i + 1) % 10 == 0:
                print(f"  ifeval {i+1}/{len(items)}", flush=True)
    judged = [r for r in results if r["pass"] is not None]
    passed = sum(1 for r in judged if r["pass"])
    return {"accuracy": passed / len(judged) if judged else 0.0,
            "passed": passed, "judged": len(judged), "skipped": len(results) - len(judged),
            "results": results}

# ---------------------------------------------------------------- ③b XFC 工具调用轴 (toolace 抽取, 已排除训练重叠)

XFC_CALL_RE = re.compile(r"([A-Za-z0-9_ \-\.]+?)\((.*?)\)\s*(?:,|$)")

def xfc_parse_calls(text):
    text = strip_thinking(text).strip()
    mblock = re.search(r"\[.*\]", text, re.S)
    if mblock: text = mblock.group(0)
    if not (text.startswith('[') and text.endswith(']')): return None
    inner = text[1:-1]
    calls = []
    for m in XFC_CALL_RE.finditer(inner + ','):
        name = m.group(1).strip()
        args = {}
        for am in re.finditer(r"(\w+)\s*=\s*(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[^,]+)", m.group(2)):
            v = am.group(2).strip()
            if v.startswith(('"', "'")):
                try: v = ast.literal_eval(v)
                except Exception: pass
            args[am.group(1)] = v
        calls.append((name, tuple(sorted((k, str(v)) for k, v in args.items()))))
    return calls if calls else None

def run_xfc(limit=0, workers=8):
    sample = json.load(open(RULERS / "xfc" / "sample.json"))
    limit = limit or len(sample)
    sample = sample[:limit]
    def one(item):
        try:
            msgs = [{"role": "system", "content": item["system"]},
                    {"role": "user", "content": item["query"]}]
            resp = chat(msgs, temperature=0.0, max_tokens=3072)
            pred = xfc_parse_calls(resp)
            if pred is None:
                return {"pass": False, "why": "parse"}
            exp = sorted((str(n).lower(), tuple((str(k), str(v)) for k, v in a))
                         for n, a in item["expected"])
            got = sorted((n.lower(), tuple((str(k), str(v)) for k, v in a)) for n, a in pred)
            return {"pass": exp == got, "why": "" if exp == got else "mismatch"}
        except Exception as e:
            return {"pass": False, "why": str(e)[:40]}
    results = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, sample):
            results.append(r)
    passed = sum(r["pass"] for r in results)
    return {"accuracy": passed / len(results) if results else 0.0,
            "passed": passed, "total": len(results), "results": results}

# ---------------------------------------------------------------- 长上下文 Needle

FILLER = ("The warehouse ledger records shipments across the harbor district. "
          "Each crate is stamped with a route number, a weight in kilograms, "
          "and the dock where it was sealed. Clerks verify the manifests "
          "against the tide tables before the ferries depart at dawn. ")

def build_needle_context(target_tokens, seed):
    rng = random.Random(seed)
    word = FILLER.strip()
    paras = []
    total = 0
    while total < target_tokens * 1.1:
        chunk = (word * rng.randint(4, 9)).strip()
        paras.append(chunk)
        total += len(chunk) // 4
    code = f"ZEBRA-{rng.randint(10000, 99999)}"
    needle = f"IMPORTANT NOTICE: The secret verification code assigned to this shipment is {code}."
    mid = len(paras) // 2
    paras.insert(mid, needle)
    text = "\n\n".join(paras)
    return text, code

def run_needle(depths=None, trials=3, workers=6):
    depths = tuple(int(x) for x in os.environ.get("NEEDLE_DEPTHS", "64,195").split(","))
    results = {}
    for depth in depths:
        target = depth * 1000
        correct = 0
        for t in range(trials):
            ctx, code = build_needle_context(target, SEED + depth + t)
            q = [{"role": "user", "content": ctx + "\n\nQuestion: What is the secret verification code mentioned in the text above? Answer with the code only."}]
            try:
                resp = chat(q, temperature=0.0, max_tokens=64)
                ok = code in (resp or "")
            except Exception:
                ok = False
            correct += int(ok)
        results[f"needle_{depth}k"] = {"hit": correct, "trials": trials}
        print(f"  needle@{depth}K: {correct}/{trials}", flush=True)
    return results

# ---------------------------------------------------------------- 2048-token 长生成 + MTP

def run_longgen_and_mtp(mtp_metrics_url=None, tokens=2048):
    mtp_metrics_url = mtp_metrics_url or os.environ.get("MTP_METRICS_URL")
    # None=生产4副本聚合模式
    # 思考型模型: reasoning 与 content 分块返回; 低推理力度让正文占满预算, 测纯答案生成速度
    t0 = time.time()
    r = requests.post(f"{EVAL_BASE}/chat/completions",
                      json={"model": MODEL,
                            "messages": [{"role": "user", "content":
                            "Write a detailed, well-structured technical essay (in Chinese) about how speculative decoding "
                            "works in large language model inference engines. Cover the draft-verify loop, acceptance "
                            "criteria, tree attention, and practical trade-offs. Be thorough."}],
                            "temperature": 0.7, "max_tokens": 8192, "reasoning_effort": "low"},
                      timeout=900)
    r.raise_for_status()
    m1 = r.json()["choices"][0]["message"]
    resp = m1.get("content") or ""
    reasoning = m1.get("reasoning_content") or ""
    dt = time.time() - t0
    n = len(resp)
    out = {"chars": n, "reasoning_chars": len(reasoning), "seconds": round(dt, 1),
           "approx_tps": round(n / 4 / dt, 1) if dt > 0 else 0,  # 中文 ~4 char/tok
           "reached_target": n / 4 >= tokens * 0.8}
    # MTP 接受率: 4 副本 (8081-8084) /metrics 计数器求和后差分 (LB 粘性不保证落同一副本)
    def snap():
        d = a = 0
        ok = 0
        urls = [mtp_metrics_url] if mtp_metrics_url else \
               [f"http://127.0.0.1:{p}/metrics" for p in (8081, 8082, 8083, 8084)]
        for url in urls:
            try:
                txt = requests.get(url, timeout=10).text
                ok += 1
                for line in txt.splitlines():
                    if line.startswith("llamacpp:spec_decode_num_draft_tokens_total "):
                        d += float(line.rsplit(" ", 1)[1])
                    elif line.startswith("llamacpp:spec_decode_num_accepted_tokens_total "):
                        a += float(line.rsplit(" ", 1)[1])
            except Exception:
                pass
        return (d, a, ok) if ok else None
    before = snap()
    r2 = requests.post(f"{EVAL_BASE}/chat/completions",
                       json={"model": MODEL,
                             "messages": [{"role": "user", "content":
                             "Explain in detail how a database query planner estimates selectivity for range predicates, "
                             "including histograms, correlation effects, and adaptive re-optimization. Write at least 800 words."}],
                             "temperature": 0.7, "max_tokens": 8192, "reasoning_effort": "low"},
                       timeout=900)
    r2.raise_for_status()
    resp2 = r2.json()["choices"][0]["message"]["content"] or ""
    out["chars_2"] = len(resp2)
    dt2 = time.time() - t0
    after = snap()
    # 累计值 (服务启动以来全副本合计) 作为背景参考
    if after:
        out["mtp_cumulative"] = {"draft_total": round(after[0], 1), "accepted_total": round(after[1], 1),
                                 "accept_rate": round(after[1] / after[0], 4) if after[0] > 0 else None,
                                 "replicas_ok": after[2]}
    if before and after:
        dd = after[0] - before[0]
        da = after[1] - before[1]
        out["mtp"] = {"draft_delta": round(dd, 1), "accepted_delta": round(da, 1),
                      "accept_rate": round(da / dd, 4) if dd > 0 else None}
    else:
        out["mtp"] = {"error": "metrics unavailable"}
    return out

# ---------------------------------------------------------------- main

AXES = {
    "humaneval": run_humaneval,
    "xfc": run_xfc,
    "bfcl": run_bfcl,
    "gsm8k": run_gsm8k,
    "ifeval": run_ifeval,
    "needle": run_needle,
    "longgen": run_longgen_and_mtp,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("axes", nargs="+", choices=list(AXES) + ["all"])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max", type=int, default=0, help="每轴最大题量(冒烟用)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    axes = list(AXES) if "all" in args.axes else args.axes

    out = {"tag": args.tag, "model": MODEL, "endpoint": EVAL_BASE,
           "date": time.strftime("%Y-%m-%d %H:%M:%S"), "axes": {}}
    for name in axes:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] axis: {name}", flush=True)
        try:
            fn = AXES[name]
            if name in ("humaneval", "xfc", "bfcl", "gsm8k", "ifeval"):
                res = fn(limit=args.max, workers=args.workers)
            else:
                res = fn()
            res["seconds"] = round(time.time() - t0, 1)
            out["axes"][name] = res
            print(f"  -> {json.dumps({k: v for k, v in res.items() if k != 'results'}, ensure_ascii=False)[:400]}", flush=True)
        except Exception as e:
            out["axes"][name] = {"error": str(e)}
            print(f"  ERROR: {e}", flush=True)
    fp = RULERS / f"{args.tag}.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nSaved: {fp}")

if __name__ == "__main__":
    main()
