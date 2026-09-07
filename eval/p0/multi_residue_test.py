"""多残差前缀命中矩阵：不同文档截断长度（token 残差 mod 128 各异）×两份文档，
每残差做暖/冷第二轮对比。任一分歧即判 CORRUPTION。"""
import json, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 19627
BASE = f"http://127.0.0.1:{PORT}"
D1 = open("/data/compose/qwen27b/train/imatrix-calib-v11-final.txt", errors="ignore").read()
D2 = open("/data/compose/qwen27b/train/calib-v11-autoround.jsonl", errors="ignore").read()[:60000]

def chat(messages, salt=None):
    body = {"model": "qwen3.8-27b", "messages": messages, "max_tokens": 350,
            "temperature": 0, "seed": 4242,
            "chat_template_kwargs": {"enable_thinking": False}}
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]["message"]["content"]

Q1 = "用三句话概括这份文档的主题。"
Q2 = "文档中第一段提到了什么？请引用原文连续 40 字以上。"
bad = 0
for name, doc in (("D1", D1), ("D2", D2)):
    for cut in range(39000, 41000, 600):
        d = doc[:cut]
        a1 = chat([{"role": "user", "content": d + "\n\n" + Q1}], salt=f"mr-cold-{name}{cut}-1")
        a2 = chat([{"role": "user", "content": d + "\n\n" + Q1},
                   {"role": "assistant", "content": a1},
                   {"role": "user", "content": Q2}], salt=f"mr-cold-{name}{cut}-2")
        b1 = chat([{"role": "user", "content": d + "\n\n" + Q1}])
        b2 = chat([{"role": "user", "content": d + "\n\n" + Q1},
                   {"role": "assistant", "content": b1},
                   {"role": "user", "content": Q2}])
        ok = (a1 == b1) and (a2 == b2)
        bad += not ok
        print(f"{name}@{cut}: {'CLEAN' if ok else 'WRONG'}  t1={len(b1)} t2={len(b2)}", flush=True)
        if not ok and a2 != b2:
            for i, (x, y) in enumerate(zip(a2, b2)):
                if x != y:
                    print(f"  分歧@{i}: A=…{a2[max(0,i-15):i+15]!r} B=…{b2[max(0,i-15):i+15]!r}")
                    break
print(f"\n总结: {bad} 组损坏 / 12")
