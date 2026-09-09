"""Apply the deployment-equivalent int4 frozen stack to a DrafterTorch model.

The stack (frozen_stack.py build) carries the 35 dequantized int4 layer
projections in bf16; overlaying swaps them into the model's layer dicts in
place.  Everything else (conv projections, norms, embed, lm_head) stays as
constructed from the baseline drafter.
"""
from __future__ import annotations

import torch

KEYS = ("qw", "kw", "vw", "ow", "gw", "uw", "dw")


def apply_stack(model, path: str) -> None:
    stack = torch.load(path, map_location="cpu", weights_only=False)
    for i, lp in enumerate(model.layers):
        for dk in KEYS:
            lp[dk] = stack[f"layers.{i}.{dk}"].to(model.dev, torch.bfloat16)
        lp["guw"] = torch.cat([lp["gw"], lp["uw"]], 0)   # engine gate_up fused
