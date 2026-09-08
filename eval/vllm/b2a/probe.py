#!/usr/bin/env python3
"""B2-A arm probe: 16 FINAL prompts (all inside the B1-C A/B cache) + t3
double-run contract, against one sandbox boot. Usage: probe.py <label>"""
import json
import sys

sys.path.insert(0, "/data/repos/qwen3.8-27b-8x4090-stack/eval/vllm/b1")
from accept_ab import run_prompt, t3_full  # noqa: E402

BASE = "http://127.0.0.1:19627"
PROMPTS = [512, 300, 313, 339, 352, 391, 404, 443,
           456, 495, 508, 547, 560, 599, 612, 651]
label = sys.argv[1]

t3a = t3_full(BASE, label)
t3b = t3_full(BASE, label)
rows = [{"target_tokens": tt, **{k: v for k, v in run_prompt(BASE, tt, label).items()
                                 if k != "text"}}
        for tt in PROMPTS if tt != 512]
rows = [{"target_tokens": 512, "tok_per_step": t3a["tok_per_step"],
         "sha12": t3a["sha12"], "drafts": 0, "accepted": 0, "chars": len(t3a["text"])}] + rows
out = {
    "label": label,
    "contract": {"self_deterministic": t3a["sha12"] == t3b["sha12"],
                 "sha12": t3a["sha12"], "chars": len(t3a["text"]),
                 "semantic": t3a["semantic"]},
    "rows": rows,
}
with open(f"/tmp/b2a/probe-{label}.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
ok = [r for r in rows if r.get("tok_per_step")]
print(f"[{label}] {len(ok)}/{len(rows)} valid, det={out['contract']['self_deterministic']}, "
      f"mean tok/step={sum(r['tok_per_step'] for r in ok)/len(ok):.4f} sha={t3a['sha12']}")
