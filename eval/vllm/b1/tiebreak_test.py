#!/usr/bin/env python3
"""B1-A tie-break contract (A3).

Contract (FROZEN 2026-09-08):
  1. A scores row is a TIE row iff its maximum value appears more than once
     (exact fp32 equality).  A draft step is tie-affected iff any row the walk
     visits is a tie row.
  2. On NON-tie steps, torch argmax (first-max index) MUST reproduce the
     engine's recorded path 100% of the time — anything else is a replica
     semantics bug, never "numerical noise".
  3. On tie steps, the adjudication ORDER of the engine's Triton walk kernel
     is UNSPECIFIED for this contract: such steps are excluded from every
     path gate and may never be counted as training regression.

Rationale: B0 observed 3/259 mismatches between torch argmax and the engine
walk with bit-identical scores — pure tie-adjudication order.  This test
promotes that observation to an executable contract and regression-tests it
against the full 260-step B0 trace.

Usage: /data/vllm/venv/bin/python tiebreak_test.py [--ask-dir ...]
Exit code 0 = contract holds.
"""

from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from drafter_torch import load_pairs, walk_greedy  # noqa: E402


def walk_tie_contract(sc: torch.Tensor, cand: torch.Tensor):
    """Same as walk_greedy but returns (path, n_tie_rows, tie_rows_detail)."""
    out = torch.empty(sc.shape[0], dtype=torch.long)
    prev, ties = 0, 0
    for l in range(sc.shape[0]):
        row = sc[l, prev]
        is_tie = int((row == row.max()).sum()) > 1
        ties += is_tie
        idx = int(row.argmax())
        out[l] = cand[l, idx]
        prev = idx
    return out, ties


def synthetic_tests() -> bool:
    """Unit tests for tie detection and first-max argmax semantics."""
    ok = True
    # exact tie row -> detected; argmax takes the FIRST max index
    row = torch.tensor([1.0, 5.0, 5.0, 2.0])
    ok &= bool((row == row.max()).sum() == 2) and int(row.argmax()) == 1
    # near-tie (different fp32 values) -> NOT a tie
    row = torch.tensor([1.0, 5.0, 5.0 + 1e-6, 2.0])
    ok &= bool((row == row.max()).sum() == 1) and int(row.argmax()) == 2
    # chained walk: second row read at the previous index
    sc = torch.zeros(2, 4, 4)
    sc[0, 0] = torch.tensor([0.0, 3.0, 1.0, 1.0])
    sc[1, 3] = torch.tensor([9.0, 0.0, 0.0, 0.0])   # walked from prev=1 -> NOT row 3
    sc[1, 1] = torch.tensor([0.0, 0.0, 7.0, 0.0])
    cand = torch.arange(8).view(2, 4)
    path, _ = walk_greedy(sc, cand)[:2]
    ok &= int(path[0]) == 1 and int(path[1]) == 6
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask-dir", default="/data/sandbox/ab-vllm/b1/trace-b0-260")
    args = ap.parse_args()

    assert synthetic_tests(), "synthetic tie/argmax semantics tests failed"
    print("synthetic tie/argmax semantics: OK")

    pairs = load_pairs(args.ask_dir)
    # Contract scope: greedy (temp=0) requests only, i.e. pairs inside
    # segmented runs.  The leading warmup request (no recorded prefill) runs
    # with temperature>0: its walk is Gumbel-SAMPLED, not argmax — pair 0-2
    # mismatch greedily by construction and are out of scope (they are also
    # excluded from every replay/supervision use).
    from drafter_torch import segment_runs
    run_pairs = [pr for run in segment_runs(pairs) for pr in run]
    n_nontie = n_nontie_ok = n_tie = 0
    tie_examples, mismatch_examples = [], []
    for i, (rin, rout) in enumerate(run_pairs):
        sc = rout["scores"][0]                    # [7,16,16] fp32 (engine-recorded)
        cand = rout["candidate_ids"][0]
        ref = rout["selector_tokens"][0]
        path, ties = walk_tie_contract(sc, cand)
        if ties > 0:
            n_tie += 1
            if len(tie_examples) < 5:
                tie_examples.append({"pair": i, "tie_rows": ties})
            continue
        n_nontie += 1
        ok = bool(torch.equal(path, ref))
        n_nontie_ok += ok
        if not ok and len(mismatch_examples) < 5:
            mismatch_examples.append(
                {"pair": i, "nt": rin["num_tokens"],
                 "walk": path.tolist(), "engine": ref.tolist()})

    print(f"greedy-run steps total={len(run_pairs)} tie-affected={n_tie} "
          f"non-tie={n_nontie} non-tie agreement={n_nontie_ok}/{n_nontie}")
    if tie_examples:
        print("tie steps (excluded by contract):", tie_examples)
    if mismatch_examples:
        print("NON-TIE MISMATCHES (contract violation):", mismatch_examples)

    held = n_nontie > 0 and n_nontie_ok == n_nontie
    print("CONTRACT:", "HOLDS" if held else "VIOLATED")
    sys.exit(0 if held else 1)


if __name__ == "__main__":
    main()
