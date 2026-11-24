"""Layer-1 metric: action-recoverability R2 for each space.

usage:
  python probe_r2.py --space frozen                 # ridge/MLP probe on raw CLS pairs
  python probe_r2.py --space cape --ckpt cache/ckpt/cape/final.pt
  python probe_r2.py --space id   --ckpt cache/ckpt/id/final.pt   # IDNet held-out R2
42 protocol: (x_k, x_{k+1}) -> a_k. ID arm is its own probe (no separate head).
"""
import argparse
import os

import numpy as np
import torch

from common import CACHE, H
from modules import CAPEModel, IDNet, StateBridge
from windows import WindowDataset


def build_pair_feats(space, ckpt=None):
    """-> (Xtr [N,1536], Atr [N,7], Xva, Ava) over consecutive window pairs."""
    tr, va = WindowDataset("train"), WindowDataset("val")
    transform = None
    net = None
    if space in ("cape", "fwd"):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = CAPEModel(); model.load_state_dict(ck["model"]); model.eval()
        bridge = StateBridge(); bridge.load_state_dict(ck["bridge"]); bridge.eval()
        with torch.no_grad():
            def transform(pooled):            # pooled [n,15,768] -> [n,256]
                return model.g(bridge(pooled))
    elif space == "id":
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        net = IDNet(); net.load_state_dict(ck["model"]); net.eval()

    def extract(ds):
        X, A = [], []
        if space == "id":
            with torch.no_grad():
                for s in range(0, len(ds), 64):
                    pl = torch.from_numpy(ds.pooled[s:s + 64].astype(np.float32))
                    aw = ds.act_win[s:s + 64]
                    prev, nxt, act = pl[:, :H], pl[:, 1:H + 1], aw[:, :H]
                    B, n, d = prev.shape
                    ah = net(prev.reshape(-1, d), nxt.reshape(-1, d)).reshape(B, n, 7)
                    X.append(ah.numpy().reshape(-1, 7))
                    A.append(act.reshape(-1, 7))
            return np.concatenate(X), np.concatenate(A)

        with torch.no_grad():
            for s in range(0, len(ds), 64):
                pl = torch.from_numpy(ds.pooled[s:s + 64].astype(np.float32))
                x = pl if transform is None else transform(pl)
                n = H   # pairs only within the window: (x_k, x_{k+1}) -> a_k, k<H
                        # (pooled has 15 slots incl. futures+rand; acts only cover H)
                X.append(torch.cat([x[:, :n], x[:, 1:n + 1]], dim=-1)
                         .reshape(-1, x.shape[-1] * 2).numpy())
                A.append(ds.act_win[s:s + 64, :H].reshape(-1, 7))
        return np.concatenate(X), np.concatenate(A)

    Xtr, Atr = extract(tr)
    Xva, Ava = extract(va)
    return Xtr, Atr, Xva, Ava


def ridge_r2(Xtr, Atr, Xva, Ava):
    # z-score features with TRAIN stats (prereg amendment 2026-11-24, before any
    # arm numbers): raw CLS norms are large -> X^T X ill-conditioned, alpha sweep
    # cannot compensate; standardization is the usual probe protocol (42 paper).
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    best = None
    for alpha in (0.1, 1.0, 10.0, 100.0):
        X1 = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        w = np.linalg.solve(X1.T @ X1 + alpha * np.eye(X1.shape[1]),
                            X1.T @ Atr).T
        pred = np.hstack([Xva, np.ones((len(Xva), 1))]) @ w.T
        r2 = 1 - ((pred - Ava) ** 2).sum(0) / ((Ava - Ava.mean(0)) ** 2).sum(0)
        score = r2.mean()
        if best is None or score > best[0]:
            best = (score, r2, alpha)
    return best[1], best[2], best[0]


def idnet_r2(Xva, Ava):     # for space=id: Xva are already IDNet predictions
    pred = Xva
    r2 = 1 - ((pred - Ava) ** 2).sum(0) / ((Ava - Ava.mean(0)) ** 2).sum(0)
    return r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=["frozen", "cape", "fwd", "id"], required=True)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    Xtr, Atr, Xva, Ava = build_pair_feats(a.space, a.ckpt)
    print(f"space={a.space}  pairs: train {len(Xtr)} / val {len(Xva)}")

    if a.space == "id":
        r2 = idnet_r2(Xva, Ava)
        print(f"IDNet held-out R2 (mean {r2.mean():.3f})  per-dim: "
              + " ".join(f"{v:.3f}" for v in r2))
        return

    r2, alpha, mean = ridge_r2(Xtr, Atr, Xva, Ava)
    print(f"ridge (alpha={alpha}) R2 (mean {mean:.3f})  per-dim: "
          + " ".join(f"{v:.3f}" for v in r2))
    out = os.path.join(CACHE, f"probe_r2_{a.space}.txt")
    with open(out, "w") as f:
        f.write(f"space={a.space} ridge_alpha={alpha}\nmean {mean:.4f}\n"
                + " ".join(f"{v:.4f}" for v in r2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
