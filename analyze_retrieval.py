"""Exploratory stratification of v2.3 retrieval (registered exploratory, log 十七).

Recomputes per-query ranks under the EXACT eval_retrieval v2.3 protocol
(same seed-1234 queries, pools, own-branch exclusion), then breaks Hit@1
down by query horizon k, by task, and characterizes the WINNING key's
temporal distance to the anchor on misses (is the model confusing
temporally-close states?).

usage:
  python analyze_retrieval.py            # cape + frozen contrast
"""
from collections import defaultdict

import numpy as np
import torch

from common import HV23, IDX_T, PAST, STRIDE_V23
from modules import CAPEModel
from windows import WindowDataset, pad_branches

import sys
CKPT = sys.argv[1] if len(sys.argv) > 1 else "cache_v3/ckpt/cape/final.pt"


@torch.no_grad()
def ranks_for(arm):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = WindowDataset("val")
    N = len(ds)
    tasks = ds.meta["task"]
    model = None
    if arm == "cape":
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        model = CAPEModel()
        model.load_state_dict(ck["model"])
        model.to(dev).eval()
    kq = np.random.default_rng(1234).integers(1, HV23 + 1, size=N)
    off = kq * STRIDE_V23
    if arm == "cape":
        kb = model.encode_state(torch.from_numpy(
            ds.branch_patch[np.arange(N), HV23 - kq].astype(np.float32)).to(dev)).cpu().numpy()
        kt = model.encode_state(torch.from_numpy(
            ds.branch_patch[np.arange(N), HV23].astype(np.float32)).to(dev)).cpu().numpy()
        K = np.stack([kb, kt], 1)
    else:
        K = np.stack([ds.pooled[np.arange(N), PAST - off],
                      ds.pooled[np.arange(N), IDX_T]], 1).astype(np.float32)
    by_task = defaultdict(list)
    for i in range(N):
        by_task[int(tasks[i])].append(i)

    ranks, win_dist = [], []
    for s in range(0, N, 16):
        idxs = np.arange(s, min(s + 16, N))
        if arm == "cape":
            ctx = torch.from_numpy(
                ds.branch_patch[idxs, HV23 - kq[idxs]].astype(np.float32)).to(dev)
            acts, lens = pad_branches(
                [ds.act_win[i][PAST - o:] for i, o in zip(idxs, off[idxs])])
            q = model(ctx, acts.to(dev), lens.to(dev)).cpu().numpy()
        else:
            q = K[idxs, 0]
        for b, i in enumerate(idxs):
            pool_ix = by_task[int(tasks[i])]
            keys = K[pool_ix].reshape(-1, K.shape[-1])
            p = pool_ix.index(i)
            own1, true = 2 * p, 2 * p + 1
            qb = q[b]
            sims = keys @ qb / (np.linalg.norm(keys, axis=1) * np.linalg.norm(qb) + 1e-8)
            sims[own1] = -np.inf
            order = np.argsort(-sims)
            r = int(np.where(order == true)[0][0])
            ranks.append(r)
            if r > 0:                       # miss: temporal distance of the winner
                w = int(order[0]) // 2
                win_dist.append(abs(int(ds.meta["T"][pool_ix[w]])
                                    - int(ds.meta["T"][i])))
    return (np.array(ranks), kq, np.array(tasks), np.array(win_dist))


def report(arm, ranks, kq, tasks, win_dist):
    print(f"\n== {arm} ==")
    for k in range(1, HV23 + 1):
        m = kq == k
        print(f"  k={k} (horizon {k*STRIDE_V23/20:.1f}s, n={m.sum()}): "
              f"hit@1 {(ranks[m]==0).mean():.3f}  hit@5 {(ranks[m]<5).mean():.3f}")
    print("  per-task hit@1:", {int(t): round(float((ranks[tasks == t] == 0).mean()), 3)
                                for t in np.unique(tasks)})
    if len(win_dist):
        print(f"  miss-winner |ΔT|: median {np.median(win_dist):.0f}f "
              f"({np.median(win_dist)/20:.2f}s), p90 {np.percentile(win_dist,90):.0f}f, "
              f"<=1s {(win_dist<=20).mean():.2%}")


def main():
    for arm in ("cape", "frozen"):
        ranks, kq, tasks, wd = ranks_for(arm)
        report(arm, ranks, kq, tasks, wd)


if __name__ == "__main__":
    main()
