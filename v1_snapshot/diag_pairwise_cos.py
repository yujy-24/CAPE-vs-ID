"""L2-style diversity diagnostics: how collapsed is each representation space?

part A (raw CLS, val windows):
  cos(o_T, o_{T+d}) d=1..3          -- temporal closeness
  cos(o_T, o_rand) cross-episode    -- background/scene similarity floor
  cos(o_T, o_T') cross-task same index -- task separability
part B (arm space, optional --ckpt):
  cos(u1, u2)  same window, two branches -> same outcome, different actions
  cos(u1, uswap) same ctx + foreign acts -> different outcome
  swap-wins rate = P[cos(u1,uswap) > cos(u1,u2)]   (lower is better; 0.5 = blind)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from common import DATA_ROOT, H
from modules import CAPEModel, StateBridge
from train import cape_forward, pad_actions
from windows import WindowDataset


def ep2task_map():
    dats = sorted(glob.glob(os.path.join(DATA_ROOT, "data/chunk-*/*.parquet")))
    d = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"])
                   for p in dats], ignore_index=True)
    return d.groupby("episode_index")["task_index"].first().to_dict()


def stats(x):
    x = np.asarray(x)
    return f"mean {x.mean():.4f}  p5 {np.percentile(x,5):.4f}  p95 {np.percentile(x,95):.4f}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="cache/ckpt/cape{tag}/final.pt")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = WindowDataset("val")
    idxs = np.arange(min(a.n, len(ds)))

    # ---- part A: raw CLS ----
    pl = torch.from_numpy(ds.pooled[idxs, H].astype(np.float32))
    import torch.nn.functional as F
    def cos(x, y):
        return F.cosine_similarity(x, y, dim=-1).numpy()
    for d in (1, 2, 3):
        fut = torch.from_numpy(ds.pooled[idxs, H + d].astype(np.float32))
        print(f"A cos(o_T, o_T+{d}) : {stats(cos(pl, fut))}")
    rnd = torch.from_numpy(ds.pooled[idxs, H + 4].astype(np.float32))
    print(f"A cos(o_T, o_rand)   : {stats(cos(pl, rnd))}")
    cross = torch.roll(pl, 1, 0)
    print(f"A cos(o_T, o_T'other): {stats(cos(pl, cross))}")
    e2t = ep2task_map()
    same_task = np.array([e2t[ds.meta["rand_ep"][i]] == ds.meta["task"][i] for i in idxs])
    if same_task.any():
        print(f"A   rand same-task subset ({same_task.sum()}): "
              f"{stats(cos(pl[same_task], rnd[same_task]))}")

    # ---- part B: arm space ----
    if not a.ckpt:
        return
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = CAPEModel(); model.load_state_dict(ck["model"]); model.eval().to(dev)
    bridge = StateBridge(); bridge.load_state_dict(ck["bridge"]); bridge.eval().to(dev)
    c12, c1s, wins = [], [], 0
    bs = 16
    for s in range(0, len(idxs), bs):
        ii = idxs[s:s + bs]
        ctx = torch.from_numpy(ds.ctx_patch[ii, 0].astype(np.float32)).to(dev)
        acts1, acts2 = [], []
        for i in ii:
            acts1.append(torch.from_numpy(ds.branch(int(i), 1)[1]))
            acts2.append(torch.from_numpy(ds.branch(int(i), 2)[1]))
        acts1, _ = pad_actions(acts1); acts2, _ = pad_actions(acts2)
        u1, _ = cape_forward(model, ctx, acts1.to(dev))
        u2, _ = cape_forward(model, ctx, acts2.to(dev))
        us, _ = cape_forward(model, ctx, acts2.roll(1, dims=0).to(dev))
        c12.append(F.cosine_similarity(u1, u2, dim=-1).cpu().numpy())
        c1s.append(F.cosine_similarity(u1, us, dim=-1).cpu().numpy())
        wins += (c1s[-1] > c12[-1]).sum()
    c12, c1s = np.concatenate(c12), np.concatenate(c1s)
    print(f"B cos(u1,u2) same-outcome : {stats(c12)}")
    print(f"B cos(u1,uswap) diff-acts : {stats(c1s)}")
    print(f"B swap-wins rate (lower=better, 0.5=blind): {wins/len(c12):.3f}")


if __name__ == "__main__":
    main()
