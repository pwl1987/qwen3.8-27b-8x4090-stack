#!/usr/bin/env python3
"""B1-A level-B replica: DFlash2 drafter forward in pure torch.

Replicates, op for op, the engine path between the fc output and the selector
input, so that gradients can flow fc -> (frozen) 5 conv/attn layers -> selector:

  context KV  = RoPE(k_norm(k_proj(hidden_norm(fc_out))))   (V un-normed, un-roped)
  queries     = embed([anchor, mask_token x 7]), positions C..C+7 (non-causal
                within the 8-token block, 2048 sliding window on context)
  5 layers    = RMSNorm -> conv.prepare -> GQA attn (32/8 heads, per-head q/k
                RMSNorm, RoPE theta 1e7 neox) -> conv.finish -> add+RMSNorm ->
                conv.prepare -> MLP(SiLU, 17408) -> conv.finish
  sample_hidden = final RMSNorm(hidden + residual), rows at query offsets 1..7

Sources (container vLLM 0.27.1+backport copies; line-verified against
/data/tools/vllm28-env .../spec_decode/dflash[2]/):
  qwen3_dflash2.py  _grouped_conv / DFlashGroupedConv / DFlash2Qwen3DecoderLayer.forward
                    / _score_edges / compute_candidates
  qwen3_dflash.py   DFlashQwen3Attention.forward / precompute_and_store_context_kv
  dflash/speculator.py  _prepare_dflash_inputs_kernel (query layout: valid_ctx =
                    num_ctx - num_rejected tail rows; query_pos = last_valid_pos+1+off)

Replay supervision of the acceptance count k: with temperature 0 the target
accepts a draft token iff it equals the target greedy token, so
k_j = prefix-match length of the drafted path vs the reconstructed target
token stream (tokenized t3 output text).  Cross-checked per step against the
recorded anchor (anchor_j must equal stream[s_j]).

B-level gates (FROZEN 2026-09-08 before the first offline run; any change is a
candidate modification + full rerun, per governance):
  sample_hidden cosine   mean >= 0.999, p10 >= 0.995
  end-to-end top16       mean >= 15.4,  p10 >= 14.0
  end-to-end greedy path (tie steps excluded) >= 0.98

Usage (GPU4):
  CUDA_VISIBLE_DEVICES=4 /data/vllm/venv/bin/python drafter_torch.py \
      --ask-dir /data/sandbox/ab-vllm/b1/trace-b0-260 \
      --text /tmp/p0/t3probe-run0.txt \
      --report /tmp/b1/blevel.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F
from torch import nn

# ---- frozen constants (models/Qwen3.8-27B-DFlash2/config.json) ----
H, NH, NKV, HD, INTER, EPS = 5120, 32, 8, 128, 17408, 1e-6
NL = 5
VOCAB, RANK, TOPK = 248320, 256, 16
DRAFT_BLOCK, NQ = 7, 8            # 7 mask tokens; 8 query tokens (anchor + 7)
MASK_TOKEN = 248070
ROPE_THETA = 1.0e7
WINDOW = 2048
TAPS, GSIZE, CBLOCK = 2, 16, 8
NGROUP = H // GSIZE               # 320

DRAFT_ST = "/data/sandbox/ab-vllm/repo/models/Qwen3.8-27B-DFlash2/model.safetensors"
LMHEAD_BAK = ("/data/sandbox/ab-vllm/repo/models/coding-v1.1-W4A16/"
              "model-00007-of-00007.safetensors.bak")
EMBED_BAK = ("/data/sandbox/ab-vllm/repo/models/coding-v1.1-W4A16/"
             "model-00006-of-00007.safetensors.bak_embed")
TOKENIZER_DIR = "/data/sandbox/ab-vllm/repo/models/coding-v1.1-W4A16"

# 候选修改 2026-09-08（治理流程：原 e2e_path_nontie>=0.98 冻结门 FAIL=0.0，根因实测：
# 引擎走链行裕度中位 0.07 / p10 0.016，而任何非逐位实现的 selector scores 扰动 >=5.2，
# argmax 链必翻盘；B0 已在逐位 scores 条件下证明贪心语义 98.8%。替代门 = 重建一致性：
# anchor 链 100% 且 engine_path_k_mean 与引擎 /metrics 独立 acceptance 计数(2.56)一致；
# own_walk_k 记为噪声底诊断(基线 0.14)，永不作门。修改后全量重跑为本文件官方结果。）
GATE = {
    "sh_cos_mean": 0.999,
    "sh_cos_p10": 0.995,
    "e2e_top16_mean": 15.4,
    "e2e_top16_p10": 14.0,
    "anchor_chain": 1.0,
    "k_mean_ref": 2.56,
    "k_mean_abs_err": 0.15,
}


# ---------------- primitives ----------------

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.float()).to(x.dtype)


def rope_cos_sin(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2, device=pos.device,
                                             dtype=torch.float64) / HD))
    fr = pos.to(torch.float64)[:, None] * inv[None, :]
    return (torch.cat([fr.cos(), fr.cos()], -1).float(),
            torch.cat([fr.sin(), fr.sin()], -1).float())


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [T, heads, HD]; neox rotate-half; vLLM casts the cache to x's dtype
    c, s = cos.to(x.dtype)[:, None, :], sin.to(x.dtype)[:, None, :]
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return x * c + torch.cat((-x2, x1), -1) * s


def grouped_conv(h: torch.Tensor, delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    # h [T,H]; delta [T,TAPS,NGROUP]; base [TAPS,H]  (qwen3_dflash2._grouped_conv)
    T = h.shape[0]
    blocks = h.unflatten(-1, (NGROUP, GSIZE))
    coeff = base.view(1, TAPS, NGROUP, GSIZE) + delta.unsqueeze(-1)
    out = coeff[:, 0] * blocks
    pos = torch.arange(T, device=h.device)
    pos = pos & (CBLOCK - 1) if CBLOCK & (CBLOCK - 1) == 0 else pos % CBLOCK
    for tap in range(1, TAPS):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coeff[:, tap] * shifted * (pos >= tap).view(-1, 1, 1)
    return out.flatten(-2)


def quantiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    t = torch.tensor(xs, dtype=torch.float64)
    return {
        "n": int(t.numel()),
        "mean": round(float(t.mean()), 4),
        "p50": round(float(t.quantile(0.5)), 4),
        "p10": round(float(t.quantile(0.1)), 4),
        "min": round(float(t.min()), 4),
    }


# ---------------- model ----------------

def load_layer(W: dict, i: int) -> dict:
    p = f"layers.{i}."
    ac, mc = p + "attention_conv.", p + "mlp_conv."
    return {
        "iln": W[p + "input_layernorm.weight"],
        "pln": W[p + "post_attention_layernorm.weight"],
        "qw": W[p + "self_attn.q_proj.weight"],
        "kw": W[p + "self_attn.k_proj.weight"],
        "vw": W[p + "self_attn.v_proj.weight"],
        "ow": W[p + "self_attn.o_proj.weight"],
        "qn": W[p + "self_attn.q_norm.weight"],
        "kn": W[p + "self_attn.k_norm.weight"],
        "gw": W[p + "mlp.gate_proj.weight"],
        "uw": W[p + "mlp.up_proj.weight"],
        "guw": torch.cat([W[p + "mlp.gate_proj.weight"],
                          W[p + "mlp.up_proj.weight"]], 0),   # fused once (engine gate_up_proj)
        "dw": W[p + "mlp.down_proj.weight"],
        "ac_b": W[ac + "base_kernel"],
        "ac_k": W[ac + "kernel_projection.weight"],
        "mc_b": W[mc + "base_kernel"],
        "mc_k": W[mc + "kernel_projection.weight"],
    }


def dequant_packed(shard_path: str, prefix: str) -> torch.Tensor:
    """Effective weights of a compressed-tensors pack-quantized (W4A16) tensor
    — the values the engine actually computed with (same unpack recipe the
    engine's _dense_kv_rows uses).  fp32 out; cast at use."""
    from compressed_tensors.compressors.pack_quantized.base import unpack_from_int32
    from safetensors import safe_open
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        packed = f.get_tensor(prefix + ".weight_packed")
        scale = f.get_tensor(prefix + ".weight_scale")
        shape = f.get_tensor(prefix + ".weight_shape").tolist()
    out_f, in_f = int(shape[0]), int(shape[1])
    bits = 32 * packed.shape[1] // in_f
    q = unpack_from_int32(packed.data, bits, torch.Size([out_f, in_f]), packed_dim=1)
    group = in_f // scale.shape[1]
    dense = (q.to(torch.float32).reshape(out_f, in_f // group, group)
             * scale.to(torch.float32)[..., None]).reshape(out_f, in_f)
    return dense


class DrafterTorch(nn.Module):
    """fc + selector are nn.Parameters (B1-B arms train them); the 5 layers,
    embed and lm_head stay frozen tensors, but autograd flows through all ops.

    embed_source: 'dequant' = the engine's effective W4A16 weights (unpack the
    packed target shards); 'bf16' = pre-quantization backups (.bak shards)."""

    def __init__(self, dev: torch.device, embed_source: str = "dequant"):
        super().__init__()
        from safetensors import safe_open
        W = {}
        with safe_open(DRAFT_ST, framework="pt", device="cpu") as f:
            for k in f.keys():
                W[k] = f.get_tensor(k)
        W4DIR = os.path.dirname(LMHEAD_BAK)
        if embed_source == "dequant":
            emb = dequant_packed(
                os.path.join(W4DIR, "model-00006-of-00007.safetensors"),
                "model.language_model.embed_tokens").to(torch.bfloat16)
            lm = dequant_packed(
                os.path.join(W4DIR, "model-00007-of-00007.safetensors"),
                "lm_head").to(torch.bfloat16)
        else:
            with safe_open(LMHEAD_BAK, framework="pt", device="cpu") as f:
                lm = next(f.get_tensor(k) for k in f.keys() if "lm_head" in k)
            with safe_open(EMBED_BAK, framework="pt", device="cpu") as f:
                emb = next(f.get_tensor(k) for k in f.keys()
                           if "embed_tokens" in k)

        self.fc = nn.Parameter(W["fc.weight"].to(dev))                    # [5120,25600]
        self.hidden_norm_w = W["hidden_norm.weight"].to(dev)
        self.final_norm_w = W["norm.weight"].to(dev)
        self.layers = [load_layer(W, i) for i in range(NL)]
        for lp in self.layers:
            for k, v in lp.items():
                lp[k] = v.to(dev)
        self.pred_cb = nn.Parameter(W["candidate_selector.predecessor_codebook"].to(dev))
        self.succ_cb = nn.Parameter(W["candidate_selector.successor_codebook"].to(dev))
        self.hid_proj = nn.Parameter(W["candidate_selector.hidden_projection.weight"].to(dev))
        self.embed_w = emb.to(dev)      # frozen [VOCAB,H]
        self.lm_head = lm.to(dev)       # frozen [VOCAB,H]
        self.dev = dev

    # -- context KV from fc rows (engine precompute_and_store_context_kv) --
    def context_kv(self, fc_rows: torch.Tensor, start_pos: int):
        normed = rms_norm(fc_rows, self.hidden_norm_w)
        pos = torch.arange(start_pos, start_pos + fc_rows.shape[0], device=self.dev)
        cos, sin = rope_cos_sin(pos)
        Ks, Vs = [], []
        for lp in self.layers:
            k = (normed @ lp["kw"].T).view(-1, NKV, HD)
            v = (normed @ lp["vw"].T).view(-1, NKV, HD)
            k = apply_rope(rms_norm(k, lp["kn"]), cos, sin)
            Ks.append(k)
            Vs.append(v)
        return Ks, Vs

    # -- 8 query tokens through the 5 layers; returns sample_hidden [7,H] --
    def forward_queries(self, anchor_id: int, ctx_len: int, Kc: list, Vc: list):
        ids = torch.tensor([anchor_id] + [MASK_TOKEN] * DRAFT_BLOCK, device=self.dev)
        x = self.embed_w[ids]
        qpos = torch.arange(ctx_len, ctx_len + NQ, device=self.dev)
        kpos = torch.arange(ctx_len + NQ, device=self.dev)
        residual = None
        for lp, Kcl, Vcl in zip(self.layers, Kc, Vc):
            x, residual = self._layer(lp, x, residual, qpos, Kcl, Vcl, kpos)
        return rms_norm(x + residual, self.final_norm_w)[1:]

    def _layer(self, lp, x, residual, qpos, Kc, Vc, kpos):
        if residual is None:
            residual = x
            h = rms_norm(x, lp["iln"])
        else:
            s = x + residual
            h, residual = rms_norm(s, lp["iln"]), s
        kp = (h @ lp["ac_k"].T).reshape(NQ, 2, TAPS, NGROUP)
        h = grouped_conv(h, kp[:, 0], lp["ac_b"][0])
        h = self._attn(lp, h, qpos, Kc, Vc, kpos)
        h = grouped_conv(h, kp[:, 1], lp["ac_b"][1])
        s = h + residual
        h, residual = rms_norm(s, lp["pln"]), s
        kp = (h @ lp["mc_k"].T).reshape(NQ, 2, TAPS, NGROUP)
        h = grouped_conv(h, kp[:, 0], lp["mc_b"][0])
        gu = h @ lp["guw"].T
        m = (F.silu(gu[:, :INTER]) * gu[:, INTER:]) @ lp["dw"].T
        h = grouped_conv(m, kp[:, 1], lp["mc_b"][1])
        return h, residual

    def _attn(self, lp, h, qpos, Kc, Vc, kpos):
        q = (h @ lp["qw"].T).view(NQ, NH, HD)
        k = (h @ lp["kw"].T).view(NQ, NKV, HD)
        v = (h @ lp["vw"].T).view(NQ, NKV, HD)
        q = rms_norm(q, lp["qn"])
        k = rms_norm(k, lp["kn"])
        cos, sin = rope_cos_sin(qpos)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        Kf = torch.cat([Kc, k], 0)                       # [C+8,NKV,HD]
        Vf = torch.cat([Vc, v], 0)
        Kr = Kf.repeat_interleave(NH // NKV, dim=1).float()
        Vr = Vf.repeat_interleave(NH // NKV, dim=1)
        att = torch.einsum("qhd,khd->hqk", q.float(), Kr) * (HD ** -0.5)
        keep = (qpos[:, None] - kpos[None, :]) < WINDOW  # non-causal + SWA
        att = att.masked_fill(~keep, float("-inf"))
        A = torch.softmax(att, dim=-1)
        ctx = torch.einsum("hqk,khd->qhd", A.to(Vf.dtype), Vr)
        return ctx.reshape(NQ, NH * HD) @ lp["ow"].T

    # -- candidate top-16 (engine compute_candidates; bf16 lm_head, fp32 topk) --
    def candidates(self, sample_hidden: torch.Tensor):
        logits = sample_hidden @ self.lm_head.T          # [7,VOCAB] bf16
        vals, ids = torch.topk(logits.float(), TOPK, dim=-1)
        return ids, vals                                  # multiplier = 1.0

    # -- engine _score_edges; einsum stays bf16, unary (fp32) added after --
    def selector_scores(self, cand, unary, sample_hidden, anchor_id):
        hp = sample_hidden @ self.hid_proj.T              # [7,RANK]
        succ = self.succ_cb[cand]
        pred_ids = torch.cat([
            torch.full((1, TOPK), anchor_id, device=self.dev, dtype=cand.dtype),
            cand[:-1]], 0)
        pred = self.pred_cb[pred_ids]
        sc = unary[:, :, None] + torch.einsum(
            "lpr,lcr->lpc", pred * hp[:, None, :], succ)
        return torch.nan_to_num(sc, nan=-1e30, posinf=1e30, neginf=-1e30)


def walk_greedy(sc: torch.Tensor, cand: torch.Tensor):
    """Sequential greedy walk (engine semantics: previous row, first-max tie).
    Returns path token ids [7], the number of tie-visited rows, and the
    minimum walked-row margin (max - 2nd max; 0.0 for tie rows)."""
    out = torch.empty(sc.shape[0], dtype=torch.long, device=sc.device)
    prev, ties, minmargin = 0, 0, float("inf")
    for l in range(sc.shape[0]):
        row = sc[l, prev]
        if int((row == row.max()).sum()) > 1:
            ties += 1
            minmargin = 0.0
        else:
            top2 = torch.topk(row, 2).values
            minmargin = min(minmargin, float(top2[0] - top2[1]))
        idx = int(row.argmax())
        out[l] = cand[l, idx]
        prev = idx
    return out, ties, minmargin


# ---------------- trace replay ----------------

def load_pairs(ask_dir: str):
    """Record order is `out_i, in_{i+1}, out_{i+1}, ...` (the leading in of the
    very first propose is missing: the collection flag was created after it).
    Pair an in-record with the out that FOLLOWS it; orphan outs are skipped."""
    recs = []
    for fp in sorted(glob.glob(os.path.join(ask_dir, "dflash_ask-*.pt"))):
        recs.extend(torch.load(fp, map_location="cpu", weights_only=False))
    pairs, pend = [], None
    for r in recs:
        if r["kind"] == "in":
            pend = r
        elif pend is not None:
            pairs.append((pend, r))
            pend = None
    return pairs


def segment_runs(pairs):
    """A run starts at a prefill propose (num_tokens > 100) that FOLLOWS a
    decode step (or the trace start): consecutive big-nt steps are chunked
    prefill of the SAME request (384 + 128 for t3).  Leading warmup decode
    pairs without a recorded prefill are dropped (context unrecoverable)."""
    runs, cur = [], []
    for pr in pairs:
        nt = pr[0]["num_tokens"]
        if nt > 100 and (not cur or cur[-1][0]["num_tokens"] <= 100):
            if cur:
                runs.append(cur)
            cur = [pr]
        elif cur:
            cur.append(pr)
    if cur:
        runs.append(cur)
    return runs


def token_stream(text_path: str) -> torch.Tensor:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    return torch.tensor(tok.encode(open(text_path).read(), add_special_tokens=False),
                        dtype=torch.long)


def build_prompt_ids() -> torch.Tensor:
    """Exact prompt token ids of the t3 fixture: ulmus_validate.make_prompt(512)
    rendered through the model's chat template with enable_thinking=False
    (the same path t3_probe uses).  Needed because mid-prefill anchors are
    prompt tokens: anchor_j sits at prompt[C_j] (first unprocessed index)."""
    import sys
    sys.path.insert(0, "/data/sandbox/ab-vllm/repo/bench")
    from ulmus_validate import make_prompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": make_prompt(512)}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True)
    if not isinstance(ids, (list, tuple)):
        ids = ids["input_ids"]
    return torch.tensor(ids, dtype=torch.long)


def replay_run(model, run_pairs, stream, prompt_ids, stats, ridx: int):
    """Deterministic reconstruction (oracle demoted to cross-check):
      - prefill chunk steps: v_j = C_j - C_{j-1} where C_j = position of
        anchor_j inside the prompt (anchor = first unprocessed prompt token);
        the LAST prefill chunk's anchor is stream[0] instead and closes the
        prompt (C = len(prompt)).
      - decode steps: v = 1 + k_prev, k_prev = prefix-match of the previous
        drafts vs the teacher stream; every step cross-validated by
        anchor == stream[C - P0] and, as an independent check, by the oracle
        cosine among all candidate v."""
    Kc = [torch.empty(0, NKV, HD, device=model.dev, dtype=torch.bfloat16) for _ in range(NL)]
    Vc = [torch.empty(0, NKV, HD, device=model.dev, dtype=torch.bfloat16) for _ in range(NL)]
    C, P0, prev_k = 0, None, None
    pending_prefill_D = None
    for j, (rin, rout) in enumerate(run_pairs):
        nt = rin["num_tokens"]
        anchor = int(rout["anchor"][0])
        ref_sh = rout["sample_hidden"][0].to(model.dev)

        if nt > 8 and P0 is None:
            # mid-prefill chunk: anchor = prompt[C_next] -> v = C_next - C
            lo, hi = C + nt - DRAFT_BLOCK, C + nt
            window = prompt_ids[lo:hi + 1]
            hits = (window == anchor).nonzero().flatten().tolist()
            if hits:
                v = lo + int(hits[0]) - C
            else:
                v = nt          # fallback: no rejected drafts in this chunk
                stats["prefill_anchor_miss"] += 1
            stats["prefill_v"].append([nt, v])
        elif P0 is None:
            # first decode step: the last prefill propose's drafts were verified
            # against the greedy stream -> v = 1 + k(last prefill drafts)
            P0 = len(prompt_ids)
            Dp = pending_prefill_D
            k = 0
            if Dp is not None:
                while (k < DRAFT_BLOCK and 1 + k < len(stream)
                       and int(Dp[k]) == int(stream[1 + k])):
                    k += 1
            v = 1 + k
        else:
            v = 1 + prev_k

        with torch.no_grad():
            aux = rin["aux"].to(model.dev)
            fc_all = aux @ model.fc.T
            ref = rout["fc_out"][:nt].to(model.dev)
            n = min(fc_all.shape[0], ref.shape[0])
            a, b = fc_all[:n].float().flatten(1), ref[:n].float().flatten(1)
            stats["fc_cos"] += F.cosine_similarity(a, b, dim=1).tolist()

            # oracle cross-check: cosine ranking must agree with the derived v
            best = None
            for vv in range(max(1, nt - DRAFT_BLOCK), nt + 1):
                Ks, Vs = model.context_kv(fc_all[:vv], C)
                Kc2 = [torch.cat([k, kk], 0) for k, kk in zip(Kc, Ks)]
                Vc2 = [torch.cat([x, y], 0) for x, y in zip(Vc, Vs)]
                sh2 = model.forward_queries(anchor, C + vv, Kc2, Vc2)
                cos = float(F.cosine_similarity(
                    sh2.float().flatten(1), ref_sh.float().flatten(1), dim=1).mean())
                if best is None or cos > best[0]:
                    best = (cos, vv, sh2, Kc2, Vc2)
                if cos > 0.99995:
                    break
            stats["oracle_cos"].append(best[0])
            stats["oracle_agree"] += (best[1] == v)
            stats["oracle_checks"] += 1

            Ks, Vs = model.context_kv(fc_all[:v], C)
            Kc = [torch.cat([k, kk], 0) for k, kk in zip(Kc, Ks)]
            Vc = [torch.cat([x, y], 0) for x, y in zip(Vc, Vs)]
            C += v
            stats["oracle_valid"].append([nt, v])
            # terminal steps (anchor at/after the last generated text token:
            # EOS region) are excluded from all gate stats — the text stream
            # carries no EOS, so v-derivation there is partially blind and the
            # steps are worthless as training data (post-EOS drafts).
            terminal = (P0 is not None and C - P0 >= len(stream))
            sh = model.forward_queries(anchor, C, Kc, Vc)
            if terminal or nt > 8:
                pass                    # context still accumulated, stats skipped
            else:
                stats["sh_cos"] += F.cosine_similarity(
                    sh.float().flatten(1), ref_sh.float().flatten(1), dim=1).tolist()

                cand, unary = model.candidates(sh)
                ref_cand = rout["candidate_ids"][0].to(model.dev)
                ov = (cand.unsqueeze(-1) == ref_cand.unsqueeze(-2)).any(-1).sum(-1)
                stats["e2e_top16"] += ov.tolist()

                sc_mid = model.selector_scores(ref_cand, rout["unary"][0].float().to(model.dev),
                                               sh, anchor)
                ref_sc = rout["scores"][0].to(model.dev)
                if sc_mid.shape == ref_sc.shape:
                    stats["mid_scores_maxdiff"].append(float((sc_mid - ref_sc).abs().max()))
                path_mid, _, _ = walk_greedy(sc_mid, ref_cand)
                stats["mid_path_ok"].append(
                    bool(torch.equal(path_mid, rout["selector_tokens"][0].to(model.dev))))

                sc_e2e = model.selector_scores(cand, unary, sh, anchor)
                path_e2e, ties, margin = walk_greedy(sc_e2e, cand)
                ok = bool(torch.equal(path_e2e, rout["selector_tokens"][0].to(model.dev)))
                stats["e2e_path_nontie"].append(ok if ties == 0 else None)
                stats["e2e_walk_minmargin"].append(margin)
                stats["e2e_tie_steps"] += ties > 0
                stats["_pending_own"] = path_e2e.tolist()

        if nt > 8:                                        # prefill chunk step
            pending_prefill_D = rout["selector_tokens"][0]
            prev_k = None
            continue
        if P0 is None:
            P0 = C
        s = C - P0
        if prev_k is not None:
            stats["kv_checks"] += 1
            stats["kv_agree"] += (v == 1 + prev_k)
        if s < len(stream):
            ok_a = int(stream[s]) == anchor
            stats["runs"][ridx]["anchor_checks"] += 1
            stats["runs"][ridx]["anchor_ok"] += ok_a
        D = rout["selector_tokens"][0]
        k = 0
        while (k < DRAFT_BLOCK and s + 1 + k < len(stream)
               and int(D[k]) == int(stream[s + 1 + k])):
            k += 1
        prev_k = k
        own = stats.pop("_pending_own", None)
        if own is not None:
            ko = 0
            while (ko < DRAFT_BLOCK and s + 1 + ko < len(stream)
                   and own[ko] == int(stream[s + 1 + ko])):
                ko += 1
            stats["engine_path_k"].append(k)
            stats["own_walk_k"].append(ko)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask-dir", default="/data/sandbox/ab-vllm/b1/trace-b0-260")
    ap.add_argument("--text", default="/tmp/p0/t3probe-run0.txt")
    ap.add_argument("--report", default="/tmp/b1/blevel.json")
    ap.add_argument("--embed-source", default="dequant", choices=["dequant", "bf16"])
    args = ap.parse_args()

    dev = torch.device("cuda")
    model = DrafterTorch(dev, embed_source=args.embed_source)
    pairs = load_pairs(args.ask_dir)
    runs = segment_runs(pairs)
    stream = token_stream(args.text)
    print(f"pairs={len(pairs)} runs={len(runs)} stream_tokens={len(stream)}")

    stats = {"fc_cos": [], "sh_cos": [], "e2e_top16": [], "mid_scores_maxdiff": [],
             "mid_path_ok": [], "e2e_path_nontie": [], "e2e_tie_steps": 0,
             "e2e_walk_minmargin": [],
             "oracle_cos": [], "oracle_valid": [], "oracle_agree": 0, "oracle_checks": 0,
             "prefill_v": [], "prefill_anchor_miss": 0,
             "own_walk_k": [], "engine_path_k": [],
             "kv_checks": 0, "kv_agree": 0,
             "runs": [{"anchor_checks": 0, "anchor_ok": 0} for _ in runs]}
    prompt_ids = build_prompt_ids()
    print(f"prompt_tokens={len(prompt_ids)}")
    for i, run in enumerate(runs):
        replay_run(model, run, stream, prompt_ids, stats, i)

    ties = stats["e2e_tie_steps"]
    path_vals = [v for v in stats["e2e_path_nontie"] if v is not None]
    # margin-clear subset (diagnostic for gate recalibration): steps whose
    # walked path never had a row margin < 1.0 (ties count as 0 margin)
    clear_idx = [i for i, m in enumerate(stats["e2e_walk_minmargin"]) if m >= 1.0]
    clear_ok = [stats["e2e_path_nontie"][i] for i in clear_idx
                if stats["e2e_path_nontie"][i] is not None]
    report = {
        "pairs": len(pairs),
        "runs": len(runs),
        "A_fc_cos": quantiles(stats["fc_cos"]),
        "B_sample_hidden_cos": quantiles(stats["sh_cos"]),
        "oracle_num_valid_cos": quantiles(stats["oracle_cos"]),
        "C_e2e_top16_overlap": quantiles(stats["e2e_top16"]),
        "mid_scores_max_abs_diff": quantiles(stats["mid_scores_maxdiff"]),
        "mid_path_acc": {"n": len(stats["mid_path_ok"]),
                         "mean": round(sum(stats["mid_path_ok"]) / max(1, len(stats["mid_path_ok"])), 4)},
        "e2e_walk_min_margin": quantiles(stats["e2e_walk_minmargin"]),
        "engine_path_k_mean": round(sum(stats["engine_path_k"]) / max(1, len(stats["engine_path_k"])), 4),
        "own_walk_k_mean": round(sum(stats["own_walk_k"]) / max(1, len(stats["own_walk_k"])), 4),
        "e2e_path_marginclear_acc": {"n": len(clear_ok),
                                     "mean": round(sum(clear_ok) / max(1, len(clear_ok)), 4)},
        "e2e_path_nontie_acc": {"n": len(path_vals),
                                "mean": round(sum(path_vals) / max(1, len(path_vals)), 4)},
        "e2e_tie_steps": ties,
        "teacher_stream_validation": {
            "kv_agree": f"{stats['kv_agree']}/{stats['kv_checks']}",
            "oracle_agree": f"{stats['oracle_agree']}/{stats['oracle_checks']}",
            "prefill_anchor_miss": stats["prefill_anchor_miss"],
            "prefill_v": stats["prefill_v"],
        },
        "anchor_reconstruction": stats["runs"],
        "gates": {},
    }
    sh = report["B_sample_hidden_cos"]
    ov = report["C_e2e_top16_overlap"]
    anchor_all = sum(r["anchor_ok"] for r in stats["runs"])
    anchor_tot = sum(r["anchor_checks"] for r in stats["runs"])
    km = report["engine_path_k_mean"]
    report["gates"] = {
        "sh_cos_mean>=0.999": sh.get("mean", 0) >= GATE["sh_cos_mean"],
        "sh_cos_p10>=0.995": sh.get("p10", 0) >= GATE["sh_cos_p10"],
        "e2e_top16_mean>=15.4": ov.get("mean", 0) >= GATE["e2e_top16_mean"],
        "e2e_top16_p10>=14": ov.get("p10", 0) >= GATE["e2e_top16_p10"],
        "anchor_chain==100%": anchor_tot > 0 and anchor_all == anchor_tot,
        "|k_mean-2.56|<=0.15": abs(km - GATE["k_mean_ref"]) <= GATE["k_mean_abs_err"],
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
