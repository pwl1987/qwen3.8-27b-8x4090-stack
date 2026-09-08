"""单格残差复测：指定 文档+截断，显式分开 a1vsb1（冷-冷）与 a2vsb2（暖命中）判定，
四份输出落盘到 /tmp/p0/cells/。配合容器内 /tmp/dflash_trace.jsonl 使用：
跑格前记下 jsonl 行数，跑完按行数切片即得本格的请求段。"""
import json
import os
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 19627
DOC = sys.argv[2]  # D1 | D2
CUT = int(sys.argv[3])
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
d = (D1 if DOC == "D1" else D2)[:CUT]
outdir = "/tmp/p0/cells"
os.makedirs(outdir, exist_ok=True)
tag = f"{DOC}{CUT}"

a1 = chat([{"role": "user", "content": d + "\n\n" + Q1}], salt=f"rc2-{tag}-1")
a2 = chat([{"role": "user", "content": d + "\n\n" + Q1},
           {"role": "assistant", "content": a1},
           {"role": "user", "content": Q2}], salt=f"rc2-{tag}-2")
b1 = chat([{"role": "user", "content": d + "\n\n" + Q1}])
b2 = chat([{"role": "user", "content": d + "\n\n" + Q1},
           {"role": "assistant", "content": b1},
           {"role": "user", "content": Q2}])

v1 = a1 == b1
v2 = a2 == b2


def firstdiff(x, y):
    for i, (c1, c2) in enumerate(zip(x, y)):
        if c1 != c2:
            return i
    return min(len(x), len(y)) if len(x) != len(y) else -1


print(f"{tag}: turn1(a1==b1)={'CLEAN' if v1 else 'WRONG'}  turn2(a2==b2)={'CLEAN' if v2 else 'WRONG'}")
if not v1:
    i = firstdiff(a1, b1)
    print(f"  t1分歧@{i}: A=…{a1[max(0,i-15):i+15]!r} B=…{b1[max(0,i-15):i+15]!r}")
if not v2:
    i = firstdiff(a2, b2)
    print(f"  t2分歧@{i}: A=…{a2[max(0,i-15):i+15]!r} B=…{b2[max(0,i-15):i+15]!r}")
print(f"  lens: a1={len(a1)} b1={len(b1)} a2={len(a2)} b2={len(b2)}")

for name, txt in (("a1", a1), ("b1", b1), ("a2", a2), ("b2", b2)):
    with open(f"{outdir}/{tag}-{name}.txt", "w") as f:
        f.write(txt)
print("VERDICT:", "CLEAN" if (v1 and v2) else "CORRUPT")
