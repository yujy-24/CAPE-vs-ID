"""No-grad monitor check for the v2.3 cape port (post-smoke, pre-retrain).

Verifies the training-time diagnostics without running to step 2000:
  - cape_loss diag: state_frac (~0.5 = official anchor pairing), swapwins,
    goal_cos; loss magnitude vs the ln(#negatives) random baseline
  - validate(): val InfoNCE under the same protocol
  - cape_counterfactual(): action sensitivity of branch outputs

Run:  python smoke_diag_v23.py [ckpt_tag]   (default _smoke)
"""
import sys

import numpy as np
import torch

from common import CAPE_BS, TAU
from modules import CAPEModel
from train import cape_counterfactual, cape_loss, group_batches, validate
from windows import WindowDataset


@torch.no_grad()
def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "_smoke"
    import os
    ck = torch.load(os.path.join("cache_v3", "ckpt", f"cape{tag}", "final.pt"),
                    map_location="cpu", weights_only=False)
    model = CAPEModel()
    model.load_state_dict(ck["model"]); model.to("cuda").eval()
    tr, va = WindowDataset("train"), WindowDataset("val")
    rng = np.random.default_rng(7)
    batches = group_batches(tr, rng)

    loss_sum, diag_acc = 0.0, dict(swapwins=0.0, goal_cos=0.0, state_frac=0.0, n=0)
    for _ in range(5):
        loss, diag = cape_loss(model, tr, next(batches), "cuda", TAU, rng)
        loss_sum += float(loss)
        for k in ("swapwins", "goal_cos", "state_frac"):
            diag_acc[k] += diag[k]
        diag_acc["n"] += 1
    n = diag_acc.pop("n")
    print(f"train loss {loss_sum/5:.4f}  (random baseline ~ln({CAPE_BS*2-1+CAPE_BS})="
          f"{np.log(CAPE_BS*2-1+CAPE_BS):.2f})")
    print({k: round(v / n, 4) for k, v in diag_acc.items()})
    print(f"val_loss  {validate(model, va, 'cape', 'cuda', 'l2', 16, TAU):.4f}")
    print(f"cf        {cape_counterfactual(model, va, 'cuda')}")


if __name__ == "__main__":
    main()
