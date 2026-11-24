"""L2-style diversity / collapse diagnostics (v2).

part A (raw features, val windows):
  cos(o_T, o_{T+d}) d=1..3          temporal closeness (CLS and pmean)
  cos(o_T, o_rand) cross-episode    background/scene similarity floor
  cos(o_T, o_T'other)               generic similarity floor
part B (arm space, --ckpt):
  cos(u1, u2)        branch1 vs branch2 (different ctx AND acts, same o_T)
  cos(u_b, enc(o_T)) branch prediction vs empty-action outcome key (training goal)
  cos(u2, uswap)     ctx_t2 + foreign same-episode acts -> different outcome
  swap-wins = P[cos(u2,uswap) > cos(u2,u1)]     (lower better; 0.5 = action-blind)
  cf cos(u2, u(foreign acts)) / cos(u2, u(random acts))  with ctx FIXED
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, FUTS, IDX_T, PAST
from modules import CAPEModel
from windows import WindowDataset, pad_actions, probe_window_frame


def stats(x):
    x = np.asarray(x)
    return f"mean {x.mean():.4f}  p5 {np.percentile(x,5):.4f}  p95 {np.percentile(x,95):.4f}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="e.g. cache_v2/ckpt/cape/final.pt")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = WindowDataset("val")
    idxs = np.arange(min(a.n, len(ds)))

    # ---- part A: raw features (CLS + pmean) ----
    for name, feat in (("CLS", ds.pooled), ("pmean", ds.pmean)):
        t = torch.from_numpy(feat[idxs, IDX_T].astype(np.float32))
        for d in range(1, FUTS + 1):
            fut = torch.from_numpy(feat[idxs, IDX_T + d].astype(np.float32))
            print(f"A[{name}] cos(o_T, o_T+{d}) : {stats(F.cosine_similarity(t, fut, -1).numpy())}")
        rnd = torch.from_numpy(feat[idxs, PAST + FUTS + 1].astype(np.float32))
        print(f"A[{name}] cos(o_T, o_rand)   : {stats(F.cosine_similarity(t, rnd, -1).numpy())}")
        cross = torch.roll(t, 1, 0)
        print(f"A[{name}] cos(o_T, o_T'other): {stats(F.cosine_similarity(t, cross, -1).numpy())}")

    if not a.ckpt:
        return

    # ---- part B: arm space ----
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = CAPEModel(); model.load_state_dict(ck["model"]); model.eval().to(dev)

    pm = np.load(os.path.join(CACHE, "probe_meta_val.npy"))
    key2prow = {(int(e), int(T)): r for r, (e, T) in enumerate(pm)}

    # foreign-act donors: another window of the SAME episode (hard), same length class
    rng = np.random.default_rng(0)
    c12, c1k, c2k, csw, cfr = [], [], [], [], []
    wins = 0
    bs = 16
    for s in range(0, len(idxs), bs):
        ii = idxs[s:s + bs]
        ctx1 = torch.from_numpy(ds.ctx_patch[ii, 0].astype(np.float32)).to(dev)
        ctx2 = torch.from_numpy(ds.ctx_patch[ii, 1].astype(np.float32)).to(dev)
        acts1 = [torch.from_numpy(ds.branch(int(i), 1)[1]) for i in ii]
        acts2 = [torch.from_numpy(ds.branch(int(i), 2)[1]) for i in ii]
        a1 = pad_actions(acts1); a2 = pad_actions(acts2)
        u1 = model(ctx1, a1.to(dev))
        u2 = model(ctx2, a2.to(dev))
        pT = np.stack([probe_window_frame("val", key2prow[(int(ds.meta['ep'][i]),
                                                          int(ds.meta['T'][i]))])
                       for i in ii])
        kT = model.encode_state(torch.from_numpy(pT.astype(np.float32)).to(dev))
        fidx = []
        for i in ii:
            same = [j for j in ds.ep_of[int(ds.meta["ep"][i])] if j != i]
            fidx.append(int(same[rng.integers(len(same))]) if same else int(i))
        a_f = pad_actions([ds.branch(int(j), 2)[1] for j in fidx])
        u_sw = model(ctx2, a_f.to(dev))            # same ctx_t2, foreign actions
        a_r = torch.randn_like(a2) * 0.3
        u_r = model(ctx2, a_r.to(dev))
        c12.append(F.cosine_similarity(u1, u2, -1).cpu().numpy())
        c1k.append(F.cosine_similarity(u1, kT, -1).cpu().numpy())
        c2k.append(F.cosine_similarity(u2, kT, -1).cpu().numpy())
        csw.append(F.cosine_similarity(u2, u_sw, -1).cpu().numpy())
        cfr.append(F.cosine_similarity(u2, u_r, -1).cpu().numpy())
        wins += int((csw[-1] > c12[-1]).sum())
    for nm, c in (("cos(u1,u2) same-outcome", c12),
                  ("cos(u1,enc(o_T))", c1k), ("cos(u2,enc(o_T))", c2k),
                  ("cos(u2,uswap) foreign-acts", csw),
                  ("cf cos(u2,u(random acts))", cfr)):
        print(f"B {nm:28s}: {stats(np.concatenate(c))}")
    print(f"B swap-wins rate (lower better, 0.5=blind): {wins/len(idxs):.3f}")


if __name__ == "__main__":
    main()
