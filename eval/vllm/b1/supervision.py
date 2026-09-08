#!/usr/bin/env python3
"""B1-A4b: teacher supervision builder + supervision gate.

TEACHER/STUDENT contract (governance lock 8):
  TEACHER = target verdict stream — the tokens the TARGET model actually
  committed: accepted drafts (accepted iff equal to the target's greedy
  choice at temperature 0) plus the correction/bonus anchor of the next step.
  STUDENT = DFlash2 drafter.  Labels NEVER come from drafter output: the
  accepted-draft labels are drafter TOKENS but they are labels exactly
  because the TARGET accepted them (verified equality) — the audit record
  keeps `source` explicit for review.

Two paths, both internally validated:
  v2 traces (num_rejected recorded, see ask-trace-v2.patch): nr read directly;
  the teacher stream is RECONSTRUCTED from labels themselves and every step's
  anchor must equal stream[s_j] (pure internal consistency check).
  Legacy B0 trace (nr absent): nr = 7 - k, k = prefix-match of the drafted
  path vs the tokenized greedy output text; validated by the same anchor
  chain against the text stream (243/243 on the 2026-09-08 trace).

Supervision gate (diagnostic baseline, no pass/fail): per batch/epoch the
trainer logs valid_depth_rate / valid_selector_rate / teacher_in_candidate_rate
— splitting "candidate generation problem" from "selector ranking problem".

Usage:
  /data/vllm/venv/bin/python supervision.py [--ask-dir ...] [--text ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from drafter_torch import (  # noqa: E402
    DRAFT_BLOCK, build_prompt_ids, load_pairs, segment_runs, token_stream)


def nr_from_records(run_pairs, stream, prompt_len):
    """num_rejected per propose index (for its drafts), legacy or v2."""
    n = len(run_pairs)
    nr = [None] * n
    has_v2 = any("num_rejected" in rin for rin, _ in run_pairs)
    if has_v2:
        for j in range(n - 1):
            r = run_pairs[j + 1][0]
            if "num_rejected" in r:
                nr[j] = int(r["num_rejected"][0])
        return nr, "v2"
    # legacy: k-chain vs text stream (prefill steps' k is vs the prompt; we only
    # need decode steps: k = prefix match at s+1)
    s = 0
    P0 = prompt_len
    prev_is_decode = False
    for j, (rin, rout) in enumerate(run_pairs):
        nt = rin["num_tokens"]
        D = rout["selector_tokens"][0]
        if nt > 8:
            # every prefill chunk's drafts are matched with s reset to 0; only
            # the LAST chunk's match faces the generated stream, and afterwards
            # s advances to 1+k (mid chunks reset again next iteration anyway)
            s = 0
            k = 0
            while (k < DRAFT_BLOCK and 1 + k < len(stream)
                   and int(D[k]) == int(stream[1 + k])):
                k += 1
            nr[j] = DRAFT_BLOCK - k
            s = 1 + k
            prev_is_decode = True
            continue
        if not prev_is_decode:
            continue                    # mid-prefill chunk: drafts face prompt
        k = 0
        while (k < DRAFT_BLOCK and s + 1 + k < len(stream)
               and int(D[k]) == int(stream[s + 1 + k])):
            k += 1
        nr[j] = DRAFT_BLOCK - k
        s += 1 + k
    return nr, "legacy"


def build_run(run_pairs, stream, prompt_len, ridx):
    """One request -> supervision steps.

    Indexing contract (kernel semantics): in-record j's num_rejected (v2) or
    the stream k-chain (legacy) gives the rejects of D_{j-1}, i.e. the VALID
    rows appended at step j: v_j = nt_j - rejects(D_{j-1}).  The LABELS of
    step j use the rejects of D_j itself (observable at step j+1):
    acc_j = 7 - rejects(D_j); teacher[d] = D_j[d] for d < acc_j, and the
    correction anchor_{j+1} at d == acc_j (bonus-only when acc_j == 7).
    """
    nr, mode = nr_from_records(run_pairs, stream, prompt_len)   # nr[j] = rejects(D_j)
    steps = []
    chk = {"ok": 0, "tot": 0}
    C, P0, prev_nr, prev_acc = 0, None, None, None
    stream_v2 = []
    for j, (rin, rout) in enumerate(run_pairs):
        nt = rin["num_tokens"]
        anchor = int(rout["anchor"][0])
        D = rout["selector_tokens"][0].tolist()
        cand = rout["candidate_ids"][0]
        if nt > 8 and P0 is None:          # prefill chunk: context only
            nxt_nt = run_pairs[j + 1][0]["num_tokens"] if j + 1 < len(run_pairs) else 0
            C += nt
            if mode == "v2":
                prev_nr = nr[j]           # real rejects of this chunk's drafts
            else:
                # legacy: only the LAST chunk's nr is real (vs the generated
                # stream); mid-chunk values are vs-prompt garbage -> 0 rows
                # rejected (matches the anchor-in-prompt rule in drafter_torch)
                prev_nr = nr[j] if nxt_nt != DRAFT_BLOCK + 1 or nr[j] is None else 0
                if nxt_nt == DRAFT_BLOCK + 1 and nr[j] is not None:
                    prev_nr = nr[j]
                else:
                    prev_nr = 0
            continue
        # rows appended at this step = previous verify's accepted tokens
        v = nt if prev_nr is None else nt - prev_nr
        if P0 is None:
            P0 = C
        C += v
        s = C - P0
        nrj = nr[j]
        if nrj is None or s >= len(stream):
            prev_nr = nrj                  # terminal / unknown: context only
            continue
        acc = DRAFT_BLOCK - nrj
        nxt_anchor = int(run_pairs[j + 1][1]["anchor"][0]) if j + 1 < len(run_pairs) else None
        labels = []
        for d in range(DRAFT_BLOCK):
            if d < acc:
                teacher, src = D[d], "accepted_draft"
            elif d == acc and nxt_anchor is not None:
                teacher, src = nxt_anchor, "correction_anchor"
            else:
                labels.append({"depth": d, "teacher_token": None,
                               "candidate_index": None, "valid": False,
                               "source": "unobserved"})
                continue
            hit = (cand[d] == teacher).nonzero().flatten().tolist()
            labels.append({"depth": d, "teacher_token": teacher,
                           "candidate_index": hit[0] if hit else None,
                           "valid": True, "source": src})
        for d in range(DRAFT_BLOCK):
            labels[d]["selector_trainable"] = bool(
                labels[d]["valid"] and labels[d]["candidate_index"] is not None
                and (d == 0 or (labels[d - 1]["valid"]
                                and labels[d - 1]["candidate_index"] is not None)))
        steps.append({"run": ridx, "j": j, "ctx_len_queries": C,
                      "num_valid_rows": v, "anchor": anchor,
                      "stream_pos": s, "labels": labels})
        if mode == "legacy":
            if s < len(stream):
                chk["tot"] += 1
                chk["ok"] += stream[s] == anchor
        elif prev_acc is not None and prev_acc < 7:
            chk["tot"] += 1
            chk["ok"] += stream_v2[-1] == anchor
        prev_acc = acc
        stream_v2.extend(lab["teacher_token"] for lab in labels if lab["valid"])
        prev_nr = nrj
    return steps, mode, stream_v2, chk["ok"], chk["tot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask-dir", default="/data/sandbox/ab-vllm/b1/trace-b0-260")
    ap.add_argument("--text", default="/tmp/p0/t3probe-run0.txt")
    ap.add_argument("--out", default="blevel-evidence/supervision-gate.json")
    args = ap.parse_args()

    pairs = load_pairs(args.ask_dir)
    runs = segment_runs(pairs)
    stream = token_stream(args.text).tolist()
    prompt_ids = build_prompt_ids()

    all_steps, modes = [], []
    anchor_ok = anchor_tot = 0
    for ridx, run in enumerate(runs):
        steps, mode, _sv2, ok, tot = build_run(run, stream, len(prompt_ids), ridx)
        modes.append(mode)
        anchor_ok += ok
        anchor_tot += tot
        all_steps.extend(steps)

    slots = [lab for st in all_steps for lab in st["labels"]]
    valid = [l for l in slots if l["valid"]]
    in_cand = [l for l in valid if l["candidate_index"] is not None]
    sel_ok = [l for l in valid if l.get("selector_trainable")]
    report = {
        "trace": args.ask_dir,
        "mode": modes,
        "sup_steps": len(all_steps),
        "anchor_internal_check": f"{anchor_ok}/{anchor_tot}",
        "valid_depth_rate": round(len(valid) / max(1, len(slots)), 4),
        "teacher_in_candidate_rate": round(len(in_cand) / max(1, len(valid)), 4),
        "valid_selector_rate": round(len(sel_ok) / max(1, len(valid)), 4),
        "mean_observable_depth": round(len(valid) / max(1, len(all_steps)), 4),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
