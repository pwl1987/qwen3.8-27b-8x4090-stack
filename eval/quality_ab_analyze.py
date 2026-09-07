#!/usr/bin/env python3
"""A/B 结果分析：自动检查 + 双栏对照 markdown。
自动检查（参考性）：
  - 0 长度内容计数（thinking 已关，应为 0）
  - python 代码块 ast 语法检查
  - generate 任务：拼接 python 块 exec 实跑（含模型自写的单元测试），单测失败/异常计为 FAIL
人工评分材料：每任务双栏全文落 markdown。
用法: python3 quality_ab_analyze.py [result.json] [-o quality_ab_head4_vs_head8.md]
"""
import argparse, ast, json, re, signal, sys

def code_blocks(text, lang):
    return re.findall(rf"```{lang}\w*\n(.*?)```", text, re.S)

class Timeout(Exception): pass

def run_generate_tests(text):
    """拼接所有 python 块 exec；模型自带的单测会跑。返回 (ok, detail)。"""
    blocks = code_blocks(text, "py")
    if not blocks:
        return None, "无 python 块"
    src = "\n\n".join(blocks)
    try:
        ast.parse(src)
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    def handler(s, f): raise Timeout()
    signal.signal(signal.SIGALRM, handler); signal.alarm(20)
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(src, {"__name__": "__main__"})
        return True, "exec 无异常"
    except Timeout:
        return False, "超时(>20s)"
    except AssertionError as e:
        return False, f"断言失败: {str(e)[:80]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"
    finally:
        signal.alarm(0)

def py_syntax_ok(text):
    blocks = code_blocks(text, "py")
    if not blocks:
        return None
    for b in blocks:
        try:
            ast.parse(b)
        except SyntaxError as e:
            return False, str(e)[:80]
    return True, f"{len(blocks)}块全过"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="?", default="/data/compose/qwen27b/train/quality_ab_head4_vs_head8.json")
    ap.add_argument("-o", default="/data/compose/qwen27b/train/quality_ab_head4_vs_head8.md")
    ap.add_argument("--la", default="A_int4head")
    ap.add_argument("--lb", default="B_int8head")
    args = ap.parse_args()
    rows = json.load(open(args.json_path, encoding="utf-8"))
    la, lb = args.la, args.lb
    lines = [f"# 质量 A/B 对照：int4 头（{la}）vs int8 头（{lb}）", "",
             f"> temperature=0、enable_thinking=False、max_tokens=8192 ｜ 自动检查仅供参考，最终以人工评分为准", ""]
    zero_a = zero_b = 0; gen_fail = {"a": [], "b": []}; syn_fail = {"a": [], "b": []}
    for r in rows:
        a, b = r.get(la, {}), r.get(lb, {})
        ca, cb = a.get("content", ""), b.get("content", "")
        ta, tb = a.get("usage", {}).get("completion_tokens"), b.get("usage", {}).get("completion_tokens")
        if a.get("error"): ca = f"[ERROR] {a['error']}"
        if b.get("error"): cb = f"[ERROR] {b['error']}"
        if not ca.strip(): zero_a += 1
        if not cb.strip(): zero_b += 1
        if r["task"] == "generate":
            for side, txt, key in ((la, ca, "a"), (lb, cb, "b")):
                ok, detail = run_generate_tests(txt)
                if ok is False: gen_fail[key].append(f"{r['task']}:{detail}")
        ok_a = py_syntax_ok(ca); ok_b = py_syntax_ok(cb)
        if ok_a is False: syn_fail["a"].append(r["task"])
        if ok_b is False: syn_fail["b"].append(r["task"])
        lines += [f"## {r['task']}", "",
                  f"| 侧 | 字符 | tokens | 耗时s | py语法 |",
                  f"|---|---:|---:|---:|---|",
                  f"| {la} | {len(ca)} | {ta} | {a.get('elapsed','-')} | {ok_a if ok_a is not None else '—'} |",
                  f"| {lb} | {len(cb)} | {tb} | {b.get('elapsed','-')} | {ok_b if ok_b is not None else '—'} |", "",
                  f"**{la}**:\n\n```\n{ca[:4000]}\n```" if len(ca) > 4000 else f"**{la}**:\n\n{ca}",
                  "", f"**{lb}**:\n\n```\n{cb[:4000]}\n```" if len(cb) > 4000 else f"**{lb}**:\n\n{cb}", "", "---", ""]
    lines += ["## 自动汇总", "",
              f"- 0 长度：{la}={zero_a}，{lb}={zero_b}（应全 0）",
              f"- generate 单测实跑 FAIL：{la}={gen_fail['a'] or '无'}；{lb}={gen_fail['b'] or '无'}",
              f"- python 语法 FAIL：{la}={syn_fail['a'] or '无'}；{lb}={syn_fail['b'] or '无'}"]
    open(args.o, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[-5:]))
    print("saved:", args.o)

if __name__ == "__main__":
    main()
