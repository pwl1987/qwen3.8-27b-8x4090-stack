#!/usr/bin/env python3
"""T1.1 质量 A/B：双引擎对比（默认 int4头 :19622 vs int8头 :19626，均 coding-v1.1 W4A16）。

同 prompt、temperature=0、enable_thinking 显式关闭（根治 thinking 吃满额度导致 0 字长）、
max_tokens=8192。输出 JSON 供人工评分；自动指标仅作参考（token 数/代码块/长度）。
用法: python3 quality_ab_v11.py [--a URL] [--b URL] [--label-a A_int4head] [--label-b B_int8head]
"""
import argparse, json, time, urllib.request

TASKS = [
    ("bugfix", "下面这段 Python 有一个并发 bug（计数器丢更新），请找出并修复：\n```python\nimport threading\ncount = 0\ndef worker():\n    global count\n    for _ in range(100000):\n        count += 1\nthreads = [threading.Thread(target=worker) for _ in range(4)]\n[t.start() for t in threads]\n[t.join() for t in threads]\nprint(count)\n```"),
    ("generate", "用 Python 写一个函数 parse_log_line(line: str) -> dict，解析 nginx 访问日志的 combined 格式（IP、时间、请求方法、路径、状态码、字节数、referer、UA），失败返回空 dict，附 3 个单元测试。"),
    ("refactor", "把下面这段回调式 JS 重构为 async/await，保持行为一致并说明改动点：\n```js\nfunction loadUser(id, cb) { fetch('/users/'+id).then(r=>r.json()).then(u=> cb(null,u)).catch(e=>cb(e)); }\nfunction loadOrders(user, cb) { fetch('/orders?user='+user.id).then(r=>r.json()).then(o=>cb(null,o)).catch(e=>cb(e)); }\nfunction render(id) { loadUser(id, (e,u)=>{ if(e) return console.error(e); loadOrders(u, (e2,o)=>{ if(e2) return console.error(e2); console.log(u, o); }); }); }\n```"),
    ("explain", "解释 Rust 的生命周期标注在下面函数里为什么必要，并给出一个去掉标注会编译错误的场景：\n```rust\nfn longest<'a>(x: &'a str, y: &'a str) -> &'a str { if x.len() > y.len() { x } else { y } }\n```"),
    ("sql", "有一张表 orders(id, user_id, amount, created_at)，写一条 SQL：找出 2026 年每个月的复购用户数（同月下单≥2 次的不同 user_id 数），按月排序，并解释索引建议。"),
    ("multifile", "设计一个最小 Flask 项目结构（3 个文件以内）：app.py / db.py / models.py，实现 POST /notes 创建、GET /notes 列表两个接口，SQLite 存储。只输出文件内容和一句话说明。"),
    ("bugfix2", "这段 Go 代码在遍历时删除元素，行为和预期不符，解释原因并修复：\n```go\ns := []int{1,2,3,4,5}\nfor i := range s { if s[i]%2 == 0 { s = append(s[:i], s[i+1:]...) } }\n```"),
    ("generate2", "写一个 bash 脚本 backup.sh：把 /data/compose 下最近 24h 修改过的 .yml/.env 打包成带时间戳的 tar.gz 到 /backup，成功/失败写日志，失败保留最近 3 份，多余的清理。"),
    ("debug", "一个 Python 服务偶发 MemoryError，进程 RSS 随时间线性增长。给出系统化的排查清单（工具+命令+看什么指标），并列出 3 个最常见根因。"),
    ("chinese", "用中文解释什么是数据库的 MVCC，举一个 PostgreSQL 下的具体例子说明可重复读隔离级别下的现象，最后用一段话总结对应用开发者的实践建议。"),
]

def chat(api, model, msg, timeout=900):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": msg}],
        "temperature": 0,
        "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(api + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    m = out["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "usage": out.get("usage", {}),
            "elapsed": round(time.monotonic() - t0, 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="http://127.0.0.1:19622/v1")
    ap.add_argument("--b", default="http://127.0.0.1:19626/v1")
    ap.add_argument("--model-a", default="qwen3.8-27b")
    ap.add_argument("--model-b", default="qwen3.8-27b")
    ap.add_argument("--label-a", default="A_int4head")
    ap.add_argument("--label-b", default="B_int8head")
    ap.add_argument("--out", default="/data/compose/qwen27b/train/quality_ab_head4_vs_head8.json")
    args = ap.parse_args()
    result = []
    for name, msg in TASKS:
        row = {"task": name}
        for side, api, model in ((args.label_a, args.a, args.model_a), (args.label_b, args.b, args.model_b)):
            try:
                row[side] = chat(api, model, msg)
            except Exception as e:
                row[side] = {"error": str(e)[:200]}
            print(f"{name} {side} done", flush=True)
        result.append(row)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("saved:", args.out)
    for r in result:
        a, b = r.get(args.label_a, {}), r.get(args.label_b, {})
        print(f'{r["task"]:10s} A:{len(a.get("content","")):5d}ch/{a.get("elapsed","-"):>6}s  '
              f'B:{len(b.get("content","")):5d}ch/{b.get("elapsed","-"):>6}s')

if __name__ == "__main__":
    main()
