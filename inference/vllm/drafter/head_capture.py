"""头部校准 v3（对齐上游 gptq_lm_head.py 配方）：vLLM in-process 生成期钩 lm_head 输入。
vLLM 只对必要位置（预填末位+解码步）过 lm_head——钩到的正是头部真实工作分布。"""
import os, sys, json
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import torch
from vllm import LLM, SamplingParams

TARGET = "/data/models/Qwen3.8-27B-coding-v1.1-W4A16-AutoRound/merged-bf16-w4g128"
OUT = "/repo/drafter/lmhead_decode_X.pt"
llm = LLM(model=TARGET, served_model_name="qwen3.8-27b", enforce_eager=True,
          gpu_memory_utilization=0.90, max_model_len=4096, max_num_seqs=8,
          max_num_batched_tokens=2048, kv_cache_dtype="bfloat16", language_model_only=True)
core = llm.llm_engine.engine_core
core = getattr(core, "engine_core", core)
model = core.model_executor.driver_worker.worker.model_runner.model
import re
norm_names = [n for n, m in model.named_modules()
              if isinstance(m, torch.nn.Module) and n.endswith("norm")
              and not re.search(r"\d", n) and "layer" not in n and "vision" not in n and "visual" not in n]
print("final-norm 候选:", norm_names, flush=True)
rows = {}; active = {}
def mk_out_hook(key):
    def hook(m, args, output):
        x = output[0] if isinstance(output, tuple) else output
        rows.setdefault(key, []).append(x.detach().reshape(-1, x.shape[-1]).to(torch.float32).cpu())
    return hook
for n in norm_names:
    active[("norm_out", n)] = getattr(model, n.split(".")[0]) if "." not in n else None
handles = []
for n in norm_names:
    mod = dict(model.named_modules())[n]
    handles.append(mod.register_forward_hook(mk_out_hook(("norm_out", n))))
lm_head_names = [n for n, _ in model.named_modules() if n.endswith("lm_head")]
def mk_pre(key):
    def pre_hook(m, args):
        x = args[0]
        rows.setdefault(key, []).append(x.detach().reshape(-1, x.shape[-1]).to(torch.float32).cpu())
    return pre_hook
for n in lm_head_names:
    handles.append(dict(model.named_modules())[n].register_forward_pre_hook(mk_pre(("lmhead_in", n))))

recs = [json.loads(l) for l in open("/repo/drafter/data/gen.jsonl")][:400]
sp = SamplingParams(max_tokens=384, temperature=1.0, top_p=0.95, top_k=20)
CH = 32
for i in range(0, len(recs), CH):
    llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in recs[i:i+CH]], sp, use_tqdm=False)
    print(f"[{i+CH}/{len(recs)}] " + " ".join(f"{k[0]}:{k[1][-20:]}={sum(r.shape[0] for r in v)}" for k, v in rows.items()), flush=True)
for h in handles:
    h.remove()
best = max(rows.items(), key=lambda kv: sum(r.shape[0] for r in kv[1])) if rows else None
assert best is not None and sum(r.shape[0] for r in best[1]) > 0, "no hook fired: " + str({k: 0 for k in rows})
X = torch.cat(best[1], 0)
print("用钩:", best[0], "X:", tuple(X.shape))
torch.save(X, OUT)
print("saved", OUT)
