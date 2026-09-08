"""dflash_trace.jsonl 分析器：按 add/rm 切请求段，重建每请求的 (nct→block_len) 决策轨迹
与最终 token 序列，对比冷轮(a2)与暖轮(b2)：
  - 块长序列先分歧 → RC2（形状非确定性：恢复改变 adaptive 状态机时序）
  - 块长序列相同而 token 仍分歧 → RC1/RC3（恢复路径元数据或状态位差）
用法: trace_analyze.py <jsonl> <起始行号(1-based, 0=全文)>"""
import json
import sys

path = sys.argv[1]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 跳过此前行（别的流量）

events = []
with open(path) as f:
    for i, line in enumerate(f, 1):
        if i <= start:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass

reqs = []  # {req, add_ev, asks:[(nct,n,src,...)], toks}
cur = None
for ev in events:
    e = ev.get("ev")
    if e == "add":
        cur = {"req": ev["req"], "add": ev, "asks": [], "toks": None}
        reqs.append(cur)
    elif e == "rm":
        if cur and cur["req"] == ev["req"]:
            cur["rm"] = ev
        cur = None
    elif e == "toks":
        for r in reqs:
            if r["req"] == ev["req"]:
                r["toks"] = ev["ids"]
    elif e == "ask":
        if cur is not None:
            cur["asks"].append(ev)

print(f"请求数: {len(reqs)}")
for r in reqs:
    a = r["add"]
    n_short = sum(1 for x in r["asks"] if x.get("src") == "adaptive" and not x["long"])
    n_long = sum(1 for x in r["asks"] if x.get("src") == "adaptive" and x["long"])
    print(f"  {a['req'][:16]} prompt={a['prompt_len']} prefill={a['prefill_len']} "
          f"resume_nct={a['nct']} asks={len(r['asks'])} (short={n_short} long={n_long}) "
          f"toks={'Y' if r['toks'] else 'N'}")

if len(reqs) >= 4:
    a1, a2, b1, b2 = reqs[-4], reqs[-3], reqs[-2], reqs[-1]

    def blocks(r):
        return [(x["nct"], x["n"]) for x in r["asks"] if x.get("src") == "adaptive"]

    ba, bb = blocks(a2), blocks(b2)
    print(f"\n[a2] adaptive asks={len(ba)}  resume_nct={a2['add']['nct']}")
    print(f"[b2] adaptive asks={len(bb)}  resume_nct={b2['add']['nct']}")

    # 块长轨迹按 nct 对齐找首个分歧
    da = {n: k for n, k in ba}
    db = {n: k for n, k in bb}
    common = sorted(set(da) & set(db))
    div = next((n for n in common if da[n] != db[n]), None)
    only_a = [n for n in da if n not in db]
    only_b = [n for n in db if n not in da]
    print(f"\n块长轨迹: 共同nct={len(common)} 首个块长分歧@nct={div} "
          f"(a2={da.get(div)} b2={db.get(div)})" if div else
          f"\n块长轨迹: 共同nct={len(common)} 全部一致", )
    if not div:
        print(f"  仅a2有的nct: {only_a[:10]}{'...' if len(only_a)>10 else ''}")
        print(f"  仅b2有的nct: {only_b[:10]}{'...' if len(only_b)>10 else ''}")

    # token 序列首分歧（前提 a1==b1 保证 prompt 相同；否则只比尾部生成段）
    ta, tb = a2["toks"], b2["toks"]
    if ta and tb:
        pa = a2["add"]["prompt_len"]
        i = 0
        while i < min(len(ta), len(tb)) and ta[i] == tb[i]:
            i += 1
        print(f"\ntoken序列: a2={len(ta)} b2={len(tb)} 首分歧@idx={i} "
              f"(prompt_len={pa}, 生成位置≈{i - pa})")
        print(f"  分歧上下文 a2[{i-6}:{i+6}]={ta[max(0,i-6):i+6]}")
        print(f"  分歧上下文 b2[{i-6}:{i+6}]={tb[max(0,i-6):i+6]}")
        if div is not None and i - pa > 0:
            rel = div - a2["add"]["nct"]
            print(f"\n判别: 块长分歧@绝对nct={div} vs token分歧@绝对idx={i}")
            print("  → 块长先分歧: RC2(形状非确定性)" if div <= i else
                  "  → token先于任何块长分歧: RC1/RC3(恢复路径)")
