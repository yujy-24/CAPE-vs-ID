"""Layer-3 metric: paper-style retrieval (v2.3, CAPE §5.2 + official C.1).

  query : (ctx at T-4k, acts[T-4k..T)) -> u, k ~ U{1..5} seeded(1234)
          -- EXACTLY the training branch distribution (in-distribution);
          acts are 4k per-frame deltas, model gathers at valid length.
  pool  : per task, TWO keys per val window = {branch frame T-4k, anchor T}
          (~2 x n_windows_per_task candidates, same scale as v2's t1/t2 pool;
          paper: all frames of the task, ~420)
  hit   : rank of the query's own ANCHOR key among the pool; the query's own
          branch key (near-duplicate of the query ctx) is EXCLUDED.
  arms (all scored under the SAME v2.3 protocol, no retraining needed):
    frozen  raw CLS cosine between ctx frame and key frames (zero-action)
    cape    cos(u, encode_state(key patch))   (shared empty-action space)
    fwd     FwdModel(ctx, acts)->768 vs CLS(key)
    id      -||IDNet(pmean[T-1], pmean[key]) - a[T-1]||_1
            (1-step scoring; anchor T-1 is NOT known at query time ->
             registered diagnostic, not a fair forecast; see log entry 五)

reports Hit@1/5, MRR (micro over queries + macro over tasks).
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from common import CACHE, CKPT_DIR, HV23, IDX_T, PAST, STRIDE_V23, append_jsonl
from modules import CAPEModel, FwdModel, IDNet
from windows import WindowDataset, pad_branches


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["frozen", "cape", "fwd", "id"], required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = WindowDataset("val")
    N = len(ds)
    tasks = ds.meta["task"]

    model = None
    if a.arm != "frozen":
        ck = torch.load(os.path.join(CKPT_DIR, f"{a.arm}{a.tag}", "final.pt"),
                        map_location="cpu", weights_only=False)
        model = {"cape": CAPEModel, "fwd": FwdModel, "id": IDNet}[a.arm]()
        model.load_state_dict(ck["model"]); model.to(dev).eval()

    # query branch per window: k ~ U{1..HV23}, FIXED seed -> same queries for
    # every arm (matched protocol across arms)
    kq = np.random.default_rng(1234).integers(1, HV23 + 1, size=N)
    off = kq * STRIDE_V23                                        # 4..20 frames

    # ---- two key frames of every val window: branch frame + anchor ----
    if a.arm == "cape":
        kb = model.encode_state(
            torch.from_numpy(
                ds.branch_patch[np.arange(N), HV23 - kq].astype(np.float32)).to(dev)
        ).cpu().numpy()
        kt = model.encode_state(
            torch.from_numpy(
                ds.branch_patch[np.arange(N), HV23].astype(np.float32)).to(dev)
        ).cpu().numpy()
        K = np.stack([kb, kt], 1)                                # [N,2,256]
    else:
        feat = ds.pooled if a.arm != "id" else ds.pmean
        sb = PAST - off                                          # branch slots
        K = np.stack([feat[np.arange(N), sb], feat[np.arange(N), IDX_T]], 1
                     ).astype(np.float32)                        # [N,2,768]

    by_task = defaultdict(list)
    for i in range(N):
        by_task[int(tasks[i])].append(i)

    ranks = []
    for s in range(0, N, a.bs):
        idxs = np.arange(s, min(s + a.bs, N))
        B = len(idxs)
        if a.arm == "frozen":
            q = K[idxs, 0]                                       # CLS of ctx frame
        elif a.arm == "id":
            pass                                                 # scored below
        else:
            ctx = torch.from_numpy(
                ds.branch_patch[idxs, HV23 - kq[idxs]].astype(np.float32)).to(dev)
            acts, lens = pad_branches(
                [ds.act_win[i][PAST - o:] for i, o in zip(idxs, off[idxs])])
            q = model(ctx, acts.to(dev), lens.to(dev)).cpu().numpy()

        for b, i in enumerate(idxs):
            pool_ix = by_task[int(tasks[i])]
            keys = K[pool_ix].reshape(-1, K.shape[-1])   # [2M,D] branch,anchor
            p = pool_ix.index(i)
            own1, true = 2 * p, 2 * p + 1                # own branch (excl)/anchor
            if a.arm == "id":
                anchor = torch.from_numpy(
                    ds.pmean[i][PAST - 1].astype(np.float32)).to(dev)
                a_gt = torch.from_numpy(ds.act_win[i][PAST - 1]).to(dev)
                kk = torch.from_numpy(keys).to(dev)
                pred = model(anchor.unsqueeze(0).expand(len(kk), -1), kk)
                sims = -torch.abs(pred - a_gt.unsqueeze(0)).sum(-1).cpu().numpy()
            else:
                qb = q[b]
                sims = keys @ qb / (np.linalg.norm(keys, axis=1) * np.linalg.norm(qb) + 1e-8)
            sims[own1] = -np.inf
            order = np.argsort(-sims)
            ranks.append(int(np.where(order == true)[0][0]))

    ranks = np.array(ranks)                              # aligned with window order
    hits = [(ranks < k).mean() for k in (1, 5)]
    mrr = (1.0 / (ranks + 1)).mean()
    per_task = defaultdict(list)
    for i in range(N):
        per_task[int(tasks[i])].append(ranks[i])
    macro = {f"hit@{k}": float(np.mean([(np.array(r) < k).mean()
                                        for r in per_task.values()]))
             for k in (1, 5)}
    out = dict(arm=f"{a.arm}{a.tag}", n=int(N), protocol="v2.3",
               pool_sizes=[len(v) * 2 for v in by_task.values()],
               hit1=round(float(hits[0]), 4), hit5=round(float(hits[1]), 4),
               mrr=round(float(mrr), 4),
               macro_hit1=round(macro["hit@1"], 4), macro_hit5=round(macro["hit@5"], 4))
    print(json.dumps(out))
    append_jsonl(os.path.join(CACHE, "results", "retrieval.jsonl"), out)


if __name__ == "__main__":
    main()
