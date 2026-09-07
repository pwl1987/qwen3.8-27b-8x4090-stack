"""前缀缓存命中正确性测试（lookup/adaptive 损坏探针，栈 start_qwen 注释方法论）。
同一段长文档，两轮对话：
  A 路（冷）: 每轮带唯一 cache_salt → 无前缀命中，输出为基准
  B 路（暖）: 同一 conversation 连发两轮 → 第二轮走前缀命中
判定: 第二轮输出 A/B 必须逐字节一致（greedy temp0）。"""
import json, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 19627
BASE = f"http://127.0.0.1:{PORT}"
DOC = open("/data/compose/qwen27b/train/imatrix-calib-v11-final.txt", errors="ignore").read()[:40000]

def chat(messages, salt=None):
    body = {"model": "qwen3.8-27b", "messages": messages, "max_tokens": 400,
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

# A 路（冷基准）
a1 = chat([{"role": "user", "content": DOC + "\n\n" + Q1}], salt="ph-cold-1")
a2 = chat([{"role": "user", "content": DOC + "\n\n" + Q1},
           {"role": "assistant", "content": a1},
           {"role": "user", "content": Q2}], salt="ph-cold-2")

# B 路（暖：同 conversation 连发，第二轮命中前缀）
b1 = chat([{"role": "user", "content": DOC + "\n\n" + Q1}])
b2 = chat([{"role": "user", "content": DOC + "\n\n" + Q1},
           {"role": "assistant", "content": b1},
           {"role": "user", "content": Q2}])

print(f"turn1 A/B 一致: {a1 == b1}  (len {len(a1)}/{len(b1)})")
print(f"turn2 A/B 一致: {a2 == b2}  (len {len(a2)}/{len(b2)})")
if a2 != b2:
    for i, (x, y) in enumerate(zip(a2, b2)):
        if x != y:
            print(f"首个分歧@{i}: A=…{a2[max(0,i-20):i+20]!r} B=…{b2[max(0,i-20):i+20]!r}")
            break
    print("PREFIX-HIT CORRUPTION" if a1 == b1 else "TURN1 已分歧(非前缀问题)")
else:
    print("PREFIX-HIT CLEAN")
