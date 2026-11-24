"""Window dataset over cache shards (all arms share it -> matched data exposure)."""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from common import CACHE, H


class WindowDataset(Dataset):
    """pooled layout: [0..H]=window frames [T-H..T], [H+1..H+3]=T+1..T+3, [H+4]=rand."""

    def __init__(self, split):
        paths = sorted(glob.glob(os.path.join(CACHE, f"{split}_shard_*.npz")))
        assert paths, f"no shards for split={split} under {CACHE}; run build_cache.py first"
        cp, pl, aw = [], [], []
        meta = {k: [] for k in ("ep", "T", "t1", "t2", "task", "rand_ep", "rand_fr")}
        for p in paths:
            z = np.load(p)
            cp.append(z["ctx_patch"]); pl.append(z["pooled"]); aw.append(z["act_win"])
            for k in meta:
                meta[k].append(z[k])
        self.ctx_patch = np.concatenate(cp)   # [N,2,196,768] fp16
        self.pooled = np.concatenate(pl)      # [N,15,768] fp16
        self.act_win = np.concatenate(aw)     # [N,H,7] f32
        self.meta = {k: np.concatenate(v) for k, v in meta.items()}
        assert self.pooled.shape[1] == H + 5, self.pooled.shape

    def __len__(self):
        return len(self.act_win)

    def branch(self, i, which):
        """(ctx_patch fp32 [196,768], remaining actions [n,7], n) for branch t1/t2."""
        t = int(self.meta[f"t{which}"][i])
        n = int(self.meta["T"][i]) - t
        return (self.ctx_patch[i, which - 1].astype(np.float32),
                self.act_win[i][H - n:], n)

    def __getitem__(self, i):
        return dict(i=i,
                    ctx_patch=torch.from_numpy(self.ctx_patch[i].astype(np.float32)),
                    pooled=torch.from_numpy(self.pooled[i].astype(np.float32)),
                    act_win=torch.from_numpy(self.act_win[i]))


def id_pairs_from_window(sample, k=4, rng=None):
    """42-protocol pairs: k random consecutive (z_j, z_{j+1}) -> a_j inside the window."""
    rng = rng or np.random.default_rng()
    js = rng.choice(H, size=k, replace=False)
    z = torch.as_tensor(sample["pooled"])
    a = torch.as_tensor(sample["act_win"])
    return (torch.stack([z[j] for j in js]),
            torch.stack([z[j + 1] for j in js]),
            torch.stack([a[j] for j in js]))
