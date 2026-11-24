"""Layer-2 metric: future-state retrieval Hit@1/5/10 with each arm's native scoring.

candidate pool (7): true terminal + 3 temporal neighbours (T+1..T+3)
                    + 1 manifest random + 1 in-batch terminal + 1 swapped-action (cape only)
scoring rules:
  frozen : cos(pooled[t1], pooled[cand])               (zero-action baseline)
  cape/fwd: cos(g(F(ctx, acts1)), g(bridge(pooled[cand])))
  id     : -|| IDNet(pooled[T-1], pooled[cand]) - a[T-1] ||_1
           (true candidate is a 1-step transition -> in-distribution for IDNet;
            temporal neighbours are 2..4-step; registered amendment, see log)
"""
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from common import CKPT_DIR, H
from modules import CAPEModel, IDNet, StateBridge
from train import cape_forward, pad_actions
from windows import WindowDataset


def load_models(arm, tag=""):
    if arm in ("cape", "fwd"):
        ck = torch.load(f"{CKPT_DIR}/{arm}{tag}/final.pt", map_location="cpu",
                        weights_only=False)
        m = CAPEModel(); m.load_state_dict(ck["model"]); m.eval()
        b = StateBridge(); b.load_state_dict(ck["bridge"]); b.eval()
        return m, b
    if arm == "id":
        ck = torch.load(f"{CKPT_DIR}/{arm}{tag}/final.pt", map_location="cpu",
                        weights_only=False)
        m = IDNet(); m.load_state_dict(ck["model"]); m.eval()
        return m, None
    return None, None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["frozen", "cape", "fwd", "id"], required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = WindowDataset("val")
    model, bridge = load_models(a.arm, a.tag)
    if model:
        model.to(dev)
    if bridge:
        bridge.to(dev)

    hits = np.zeros(3)          # @1 @5 @10
    top_type = Counter()
    n = 0
    order = np.arange(len(ds))
    ar = torch.arange(a.bs, device=dev)
    for s in range(0, len(ds), a.bs):
        idxs = order[s:s + a.bs]
        pl = torch.from_numpy(ds.pooled[idxs].astype(np.float32)).to(dev)   # [B,15,768]
        B = len(idxs)
        ar_b = ar[:B]

        # candidate pool per design doc §6: true + 2 temporal + 6 random
        #   (1 manifest random + 5 in-batch terminals) [+1 swapped for cape]
        cand = torch.stack([pl[:, H], pl[:, H + 1], pl[:, H + 2], pl[:, H + 4]],
                           dim=1)                                            # [B,4,768]
        rolls = torch.stack([torch.roll(pl[:, H], k, 0) for k in range(1, 6)], 1)
        cand = torch.cat([cand, rolls], 1)                                   # [B,9,768]
        types = ["true", "tn+1", "tn+2", "rand", "b1", "b2", "b3", "b4", "b5"]

        if a.arm in ("cape", "fwd"):
            ctx = torch.from_numpy(ds.ctx_patch[idxs, 0].astype(np.float32)).to(dev)
            acts1 = [torch.from_numpy(ds.branch(int(i), 1)[1]) for i in idxs]
            acts1, _ = pad_actions(acts1)
            query, _ = cape_forward(model, ctx, acts1.to(dev))
            cand_emb = model.g(bridge(cand))                                 # [B,6,256]
            if a.arm == "cape":   # swapped-action candidate: same ctx, other window's acts2
                acts2 = [torch.from_numpy(ds.branch(int(i), 2)[1]) for i in idxs]
                acts2, _ = pad_actions(acts2)
                us, _ = cape_forward(model, ctx, acts2.roll(1, dims=0).to(dev))
                cand_emb = torch.cat([cand_emb, us.unsqueeze(1)], 1)
                types.append("swap")
            sims = torch.bmm(cand_emb, query.unsqueeze(-1)).squeeze(-1)      # [B,C]
        elif a.arm == "id":
            z_prev = pl[:, H - 1]                                            # frame T-1
            a_gt = torch.from_numpy(np.stack(
                [ds.act_win[i][H - 1] for i in idxs])).to(dev)               # action at T-1
            C = cand.shape[1]
            pred = model(z_prev.unsqueeze(1).expand(-1, C, -1).reshape(-1, 768),
                         cand.reshape(-1, 768)).reshape(B, C, 7)
            sims = -torch.abs(pred - a_gt.unsqueeze(1)).sum(-1)
        else:  # frozen
            t1 = torch.tensor([ds.meta["t1"][i] - (ds.meta["T"][i] - H) for i in idxs],
                              device=dev)
            query = pl[ar_b, t1]
            sims = F.cosine_similarity(cand, query.unsqueeze(1), dim=-1)

        rank = sims.argsort(dim=1, descending=True)
        for b in range(B):
            r = int((rank[b] == 0).nonzero()[0])
            hits[0] += r < 1; hits[1] += r < 5; hits[2] += r < 10
            top_type[types[int(rank[b, 0])]] += 1
            n += 1

    print(f"[{a.arm}{a.tag}] Hit@1/5/10 = "
          f"{hits[0]/n:.3f} / {hits[1]/n:.3f} / {hits[2]/n:.3f}  (n={n})")
    print("top-1 type counts:", dict(top_type))


if __name__ == "__main__":
    main()
