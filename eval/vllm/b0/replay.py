#!/usr/bin/env python3
"""B0 — DFlash2 drafter 纯 torch 数值复刻（bf16 双侧对齐）。

数据: 引擎 ask 级插桩(/tmp/dflash_ask.on)产出的 dflash_ask-*.pt:
  in  = {num_tokens, aux(num_tokens×25600 bf16 fc 输入), input_ids}
  out = {fc_out(num_tokens_padded×5120), sample_hidden(B,7,5120 drafter 5 层输出),
         candidate_ids(B,7,16), unary(B,7,16), scores(B,7,16,16 sanitized),
         selector_tokens(B,7), anchor(B)}
权重: bf16 drafter safetensors(fc/selector 码本/hidden_projection) + target bf16 lm_head
      (量化前备份分片)。注意引擎侧 lm_head 为 W4A16(Marlin)，top-16 级差异含该量化噪声——
      这是门设在 14/16 而非 16/16 的原因之一。

对齐级别:
  A fc      aux @ fc.wᵀ vs 引擎 fc_out          (余弦/最大绝对差, 逐 token)
  C top-16  sample_hidden @ lm_headᵀ → top16 vs 引擎 candidate_ids (交叠/位置)
  D selector 码本+hidden_projection 重算 scores; greedy 链与 DP 链两种走链对照
            引擎 selector_tokens; scores 张量最大绝对差
  (B = fc_out→5 层→sample_hidden 的逐层复刻属 B1 训练器; in/out 记录已含两侧张量可离线续验)

门(首批 trace 后冻结): overlap mean ≥ 14/16 且 p10 ≥ 12/16 且 min 记录; path acc ≥ 95%
用法: CUDA_VISIBLE_DEVICES=4 /data/vllm/venv/bin/python replay.py --ask-dir <dir> [--report out.md]
"""
import argparse, glob, json, math, os, sys

import torch

DRAFT_ST = "/data/sandbox/ab-vllm/repo/models/Qwen3.8-27B-DFlash2/model.safetensors"
TARGET_SHARD = ("/data/sandbox/ab-vllm/repo/models/coding-v1.1-W4A16/"
                "model-00007-of-00007.safetensors.bak")   # 量化前备份, bf16 lm_head
K, L, H = 16, 7, 5120


def load_weights(dev):
    from safetensors import safe_open
    w = {}
    with safe_open(DRAFT_ST, framework="pt") as f:
        keys = list(f.keys())
        def get(sub):
            for k in keys:
                if k.endswith(sub):
                    return f.get_tensor(k).to(dev)
            raise KeyError(sub)
        w["fc"] = get("fc.weight")                       # [5120, 25600]
        w["pred_cb"] = get("predecessor_codebook")       # [V, 256]
        w["succ_cb"] = get("successor_codebook")
        w["hid_proj"] = get("hidden_projection.weight")  # [256, 5120]
    with safe_open(TARGET_SHARD, framework="pt") as f:
        for k in f.keys():
            if k.endswith("lm_head.weight"):
                w["lm_head"] = f.get_tensor(k).to(dev)   # [V, 5120]
    return w


def load_pairs(ask_dir):
    pairs, pending_in = [], None
    for p in sorted(glob.glob(os.path.join(ask_dir, "dflash_ask-*.pt"))):
        for rec in torch.load(p, map_location="cpu", weights_only=False):
            if rec.get("kind") == "in":
                pending_in = rec
            elif rec.get("kind") == "out":
                pairs.append((pending_in, rec))
                pending_in = None
    return pairs


def quantiles(xs):
    if not xs:
        return {"n": 0}
    t = torch.tensor(xs, dtype=torch.float64)
    return {"n": len(xs), "mean": round(t.mean().item(), 4),
            "p50": round(t.quantile(0.5).item(), 4),
            "p10": round(t.quantile(0.1).item(), 4),
            "min": round(t.min().item(), 4)}


def walk_greedy(scores, cand):
    """贪心链: 每步在给定前驱行上取 argmax（scores: B,L,K,K → tokens B,L）。"""
    B, Lp, _, _ = scores.shape
    out = torch.empty(B, Lp, dtype=torch.long, device=scores.device)
    prev = None
    for l in range(Lp):
        sc = scores[:, l]                       # B,K,K (p,c)
        if prev is None:
            cur = sc[:, 0].argmax(-1)           # l=0 各 p 行等价
        else:
            cur = sc.gather(1, prev[:, None, None].expand(-1, 1, sc.shape[-1]))[:, 0].argmax(-1)
        out[:, l] = cur
        prev = cur
    return torch.gather(cand, 2, out.unsqueeze(-1)).squeeze(-1)


def walk_dp(scores, cand):
    """DP 链: 最大化 Σ_l scores[l, c_{l-1}, c_l]（含 unary，unary 在 scores 内广播于 p 维）。"""
    B, Lp, Kk, _ = scores.shape
    best = scores[:, 0, 0].clone()              # B,K — l=0 行等价
    back = []
    for l in range(1, Lp):
        trans = scores[:, l]                    # B,K(prev),K(cur)
        cand_tot = best[:, :, None] + trans     # B,Kp,Kc
        back.append(cand_tot.argmax(-1))        # B,Kc → 最优前驱
        best = cand_tot.max(-1).values
    path = [best.argmax(-1)]
    for bk in reversed(back):
        path.append(bk.gather(1, path[-1][:, None]).squeeze(1))
    idx = torch.stack(list(reversed(path)), 1)  # B,L
    return torch.gather(cand, 2, idx.unsqueeze(-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask-dir", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    dev = "cuda"
    w = load_weights(dev)
    pairs = load_pairs(args.ask_dir)
    print(f"pairs={len(pairs)}  weights: fc{tuple(w['fc'].shape)} cb{tuple(w['pred_cb'].shape)} "
          f"lm_head{tuple(w['lm_head'].shape)}")

    fc_cos, fc_mad = [], []
    ov_flat, ov_step = [], []      # 逐步逐步位置的 top16 交叠
    score_mad = []
    path_g, path_dp = [], []
    sm_hist = []
    for i, (rin, rout) in enumerate(pairs):
        if rin is None or rin.get("aux") is None:
            continue
        aux = rin["aux"].to(dev, torch.bfloat16)          # n×25600
        n = min(aux.shape[0], rout["fc_out"].shape[0])
        # A: fc
        fc_t = aux[:n] @ w["fc"].T
        fc_e = rout["fc_out"][:n].to(dev, torch.bfloat16)
        if fc_e.shape[0]:
            a, b = fc_t.float().flatten(1), fc_e.float().flatten(1)
            cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
            fc_cos += cos.tolist()
            fc_mad += (a - b).abs().max(1).values.tolist()
        # C: top-16（用引擎 record 的 sample_hidden 作输入，隔离 B 级未复刻的影响）
        sh = rout["sample_hidden"].to(dev, torch.bfloat16)      # B,7,5120
        logits = (sh @ w["lm_head"].T).float()
        tv, ti = torch.topk(logits, K, dim=-1)
        ce = rout["candidate_ids"].to(dev)
        ov = (ce.unsqueeze(-1) == ti.unsqueeze(-2)).any(-1).sum(-1).float()  # B,7 交叠数
        ov_flat += ov.flatten().tolist()
        ov_step.append(ov.mean().item())
        # D: selector scores + 走链
        cand = ce
        unary = rout["unary"].to(dev)
        anchor = rout["anchor"].to(dev)
        shp = rout["sample_hidden"].to(dev, torch.bfloat16)
        hp = shp @ w["hid_proj"].T                              # B,7,256
        succ = w["succ_cb"][cand]                               # B,7,K,256
        pid = torch.cat((anchor[:, None, None].expand(-1, 1, K), cand[:, :-1]), 1)
        pred = w["pred_cb"][pid]                                # B,7,K,256
        sc = unary[:, :, None] + torch.einsum("blpr,blcr->blpc",
                                              pred * hp[:, :, None], succ).float()
        sc = torch.nan_to_num(sc, nan=-1e30, posinf=1e30, neginf=-1e30)
        se = rout["scores"].to(dev).float()
        if se.dim() == 4 and se.shape == sc.shape:
            score_mad.append((sc - se).abs().max().item())
        sel = rout["selector_tokens"].to(dev)
        g_ok = (walk_greedy(sc, cand) == sel).all(-1)
        d_ok = (walk_dp(sc, cand) == sel).all(-1)
        path_g += g_ok.tolist(); path_dp += d_ok.tolist()
        sm_hist.append((i, int(n), float(ov.mean())))

    rep = {
        "pairs": len(pairs),
        "A_fc": {"cos": quantiles(fc_cos), "max_abs_diff": quantiles(fc_mad)},
        "C_top16_overlap": {"per_position": quantiles(ov_flat)},
        "D_scores_max_abs_diff": quantiles(score_mad),
        "D_path_greedy_acc": quantiles(path_g),
        "D_path_dp_acc": quantiles(path_dp),
        "gates": {"mean>=14/16": (quantiles(ov_flat).get("mean", 0) >= 14),
                  "p10>=12/16": (quantiles(ov_flat).get("p10", 0) >= 12),
                  "path>=0.95": (max(quantiles(path_g).get("mean", 0),
                                     quantiles(path_dp).get("mean", 0)) >= 0.95)},
    }
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
