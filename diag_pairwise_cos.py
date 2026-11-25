"""L2-style diversity / collapse diagnostics (v2.3).

part A (raw features, val windows):
  cos(o_T, o_{T+d}) d=1..3          temporal closeness (CLS and pmean)
  cos(o_T, o_rand) cross-episode    background/scene similarity floor
  cos(o_T, o_T'other)               generic similarity floor
part B (arm space, --ckpt; v2.3 branches k ~ U{1..5}, stride 4):
  cos(u1, u2)        two branches k1 != k2 (different ctx AND acts, same o_T)
  cos(u_b, enc(o_T)) branch prediction vs empty-action outcome key (training goal)
  cos(u2, uswap)     ctx_2 + foreign same-episode acts (same k) -> other outcome
  swap-wins = P[cos(u2,uswap) > cos(u2,u1)]     (lower better; 0.5 = action-blind)
  cf cos(u2, u(random acts))  with ctx FIXED
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common import FUTS, HV23, IDX_T, PAST
from modules import CAPEModel
from windows import WindowDataset, pad_branches


def stats(x):
    x = np.asarray(x)
    return f"mean {x.mean():.4f}  p5 {np.percentile(x,5):.4f}  p95 {np.percentile(x,95):.4f}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="e.g. cache_v3/ckpt/cape/final.pt")
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

    # ---- part B: arm space (v2.3 official-offset branches) ----
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = CAPEModel(); model.load_state_dict(ck["model"]); model.eval().to(dev)

    rng = np.random.default_rng(0)
    k1 = rng.integers(1, HV23 + 1, size=len(idxs))              # branch A
    k2 = 1 + (k1 - 1 + rng.integers(1, HV23, size=len(idxs))) % HV23   # k2 != k1
    c12, c1k, c2k, csw, cfr = [], [], [], [], []
    wins = 0
    bs = 16
    for s in range(0, len(idxs), bs):
        ii = idxs[s:s + bs]
        b1 = [ds.branch_v23(int(i), int(k)) for i, k in zip(ii, k1[s:s + bs])]
        b2 = [ds.branch_v23(int(i), int(k)) for i, k in zip(ii, k2[s:s + bs])]
        ctx1 = torch.from_numpy(np.stack([b[0] for b in b1])).to(dev)
        ctx2 = torch.from_numpy(np.stack([b[0] for b in b2])).to(dev)
        a1, l1 = pad_branches([b[1] for b in b1])
        a2, l2 = pad_branches([b[1] for b in b2])
        u1 = model(ctx1, a1.to(dev), l1.to(dev))
        u2 = model(ctx2, a2.to(dev), l2.to(dev))
        pT = torch.from_numpy(
            ds.branch_patch[ii, HV23].astype(np.float32)).to(dev)
        kT = model.encode_state(pT)
        fidx = np.empty(len(ii), dtype=int)   # foreign acts: same-episode donor, same k
        for t_, i in enumerate(ii):
            same = [j for j in ds.ep_of[int(ds.meta["ep"][i])] if j != i]
            fidx[t_] = int(same[rng.integers(len(same))]) if same else int(i)
        fb = [ds.branch_v23(int(j), int(k)) for j, k in zip(fidx, k2[s:s + bs])]
        a_f, l_f = pad_branches([b[1] for b in fb])
        u_sw = model(ctx2, a_f.to(dev), l_f.to(dev))   # same ctx_2, foreign actions
        a_r = torch.randn_like(a2) * 0.3
        u_r = model(ctx2, a_r.to(dev), l2.to(dev))
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
