"""Layer-3 metric: paper-style retrieval (v2, CAPE §5.2 protocol).

  query : (ctx_t1, acts[t1..t2)) -> u, horizon h = t2-t1 (5..12, exactly the
          branch-2 length range -> in-distribution generalisation test)
  pool  : per task, state encodings of every val window's t1 AND t2 frames
          (~2 x n_windows_per_task candidates; paper: all frames of the task)
  hit   : rank of the query's own t2 encoding among the pool;
          the query's own t1 key is EXCLUDED (near-duplicate of the query ctx).
  arms:
    frozen  raw CLS cosine between t1 and key frames (zero-action baseline)
    cape    cos(u, encode_state(key patch))      (shared empty-action space)
    fwd     FwdModel(ctx_t1, acts)->768 vs CLS z(key)
    id      -||IDNet(pmean[t2-1], pmean[key]) - a[t2-1]||_1
            (1-step scoring; anchor t2-1 is NOT known at query time ->
             registered diagnostic, not a fair forecast; see log entry 五)

reports Hit@1/5, MRR (micro over queries + macro over tasks).
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, CKPT_DIR, PAST, append_jsonl
from modules import CAPEModel, FwdModel, IDNet
from windows import WindowDataset, pad_actions


def slot_of(ds, i, frame):
    return int(frame) - (int(ds.meta["T"][i]) - PAST)


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

    # ---- encode the two key frames of every val window ----
    if a.arm == "cape":
        k1 = np.concatenate([model.encode_state(
            torch.from_numpy(ds.ctx_patch[s:s + a.bs, 0].astype(np.float32)).to(dev)
        ).cpu().numpy() for s in range(0, N, a.bs)])
        k2 = np.concatenate([model.encode_state(
            torch.from_numpy(ds.ctx_patch[s:s + a.bs, 1].astype(np.float32)).to(dev)
        ).cpu().numpy() for s in range(0, N, a.bs)])
        K = np.stack([k1, k2], 1)                                  # [N,2,256]
    elif a.arm in ("frozen", "fwd", "id"):
        feat = ds.pooled if a.arm != "id" else ds.pmean
        s1 = np.array([slot_of(ds, i, ds.meta["t1"][i]) for i in range(N)])
        s2 = np.array([slot_of(ds, i, ds.meta["t2"][i]) for i in range(N)])
        K = np.stack([feat[np.arange(N), s1], feat[np.arange(N), s2]], 1).astype(np.float32)

    by_task = defaultdict(list)
    for i in range(N):
        by_task[int(tasks[i])].append(i)

    ranks = []
    for s in range(0, N, a.bs):
        idxs = np.arange(s, min(s + a.bs, N))
        B = len(idxs)
        if a.arm == "frozen":
            q = K[idxs, 0]
        elif a.arm == "id":
            pass                                                  # scored below
        else:
            ctx = torch.from_numpy(ds.ctx_patch[idxs, 0].astype(np.float32)).to(dev)
            acts = []
            for i in idxs:
                t1, t2 = int(ds.meta["t1"][i]), int(ds.meta["t2"][i])
                lo = t1 - (int(ds.meta["T"][i]) - PAST)
                acts.append(ds.act_win[i][lo:lo + (t2 - t1)])
            acts = pad_actions(acts)
            q = model(ctx, acts.to(dev)).cpu().numpy()

        for b, i in enumerate(idxs):
            pool_ix = by_task[int(tasks[i])]
            keys = K[pool_ix].reshape(-1, K.shape[-1])   # [2M,D] t1,t2 interleaved
            p = pool_ix.index(i)
            own1, true = 2 * p, 2 * p + 1                # own t1 (excl) / own t2
            if a.arm == "id":
                t2 = int(ds.meta["t2"][i])
                anchor = torch.from_numpy(
                    ds.pmean[i][slot_of(ds, i, t2 - 1)].astype(np.float32)).to(dev)
                a_gt = torch.from_numpy(ds.act_win[i][slot_of(ds, i, t2 - 1)]).to(dev)
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
    out = dict(arm=f"{a.arm}{a.tag}", n=int(N),
               pool_sizes=[len(v) * 2 for v in by_task.values()],
               hit1=round(float(hits[0]), 4), hit5=round(float(hits[1]), 4),
               mrr=round(float(mrr), 4),
               macro_hit1=round(macro["hit@1"], 4), macro_hit5=round(macro["hit@5"], 4))
    print(json.dumps(out))
    append_jsonl(os.path.join(CACHE, "results", "retrieval.jsonl"), out)


if __name__ == "__main__":
    main()
