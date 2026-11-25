"""Window dataset over cache shards (all arms share it -> matched data exposure).

v2 pooled layout (29 slots): idx 0..PAST-1 -> frames [T-PAST..T-1], idx PAST ->
frame T, PAST+1..PAST+FUTS -> T+1..T+3, PAST+FUTS+1 -> random negative.
ctx_patch[0/1] = branch-1/2 context frames (t1 < t2, 0.25-0.6 s apart).
act_win [PAST,7] = actions of [T-PAST..T); branch actions slice [t->T].
pmean = mean-pooled patch features per slot (42 probe protocol).
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from common import CACHE, FUTS, H, HV23, PAST, STRIDE_V23


class WindowDataset(Dataset):
    def __init__(self, split):
        paths = sorted(glob.glob(os.path.join(CACHE, f"{split}_shard_*.npz")))
        assert paths, f"no shards for split={split} under {CACHE}; run build_cache.py first"
        cp, pl, pm, aw, gp, bp = [], [], [], [], [], []
        meta = {k: [] for k in ("ep", "T", "t1", "t2", "task", "rand_ep", "rand_fr")}
        for p in paths:
            z = np.load(p)
            cp.append(z["ctx_patch"]); pl.append(z["pooled"])
            pm.append(z["pmean"]); aw.append(z["act_win"])
            gp.append(z["goal_patch"]); bp.append(z["branch_patch"])
            for k in meta:
                meta[k].append(z[k])
        self.ctx_patch = np.concatenate(cp)   # [N,2,196,768] fp16 (t1/t2, legacy v2)
        self.goal_patch = np.concatenate(gp)  # [N,196,768] fp16 (v2.1 goal state)
        self.branch_patch = np.concatenate(bp)  # [N,HV23+1,196,768] fp16 (v2.3)
        self.pooled = np.concatenate(pl)      # [N,29,768] fp16 (CLS)
        self.pmean = np.concatenate(pm)       # [N,29,768] fp16 (patch-mean)
        self.act_win = np.concatenate(aw)     # [N,PAST,7] f32, covers [T-PAST..T)
        self.meta = {k: np.concatenate(v) for k, v in meta.items()}
        assert self.pooled.shape[1] == PAST + FUTS + 2, self.pooled.shape
        assert self.branch_patch.shape[1] == HV23 + 1, self.branch_patch.shape
        # episodes with >=2 windows: cape grouped batching pool
        self.ep_of = {}
        for i, e in enumerate(self.meta["ep"]):
            self.ep_of.setdefault(int(e), []).append(i)

    def __len__(self):
        return len(self.act_win)

    def slot(self, i, frame):
        """(pooled/pmean slot index) of an episode-local frame, or None if outside."""
        rel = int(frame) - (int(self.meta["T"][i]) - PAST)
        return rel if 0 <= rel < PAST + FUTS + 2 else None

    def branch(self, i, which):
        """(ctx patches fp32 [196,768], remaining actions [n,7] fp32, n) for branch 1/2.
        Branch which starts at t_{which} and its actions run t_{which} -> T."""
        t = int(self.meta[f"t{which}"][i])
        T = int(self.meta["T"][i])
        acts = self.act_win[i][t - (T - PAST):]         # [n,7], n = T - t
        return (self.ctx_patch[i, which - 1].astype(np.float32),
                acts.astype(np.float32), T - t)

    def branch_v23(self, i, k):
        """official-offset branch (design doc §14): k=0 -> anchor zero-horizon
        sample (anchor ctx, ZERO actions); k in 1..HV23 -> start frame
        T - k*STRIDE_V23, remaining = k*STRIDE_V23 per-frame delta actions.
        -> (ctx [196,768] fp32, acts [n,7] fp32, n)."""
        if k == 0:
            return (self.branch_patch[i, HV23].astype(np.float32),
                    np.zeros((0, 7), np.float32), 0)
        off = k * STRIDE_V23
        return (self.branch_patch[i, HV23 - k].astype(np.float32),
                self.act_win[i][PAST - off:].astype(np.float32), off)

    def probe_pairs(self):
        """consecutive (z_k, z_{k+1}) -> a_k inside the H-frame probe window,
        using patch-mean features (42 protocol). -> X [N*H, 2*768], A [N*H, 7]."""
        lo = PAST - H                                   # first probe frame slot
        z = self.pmean[:, lo:lo + H + 1].astype(np.float32)     # [N,H+1,768]
        a = self.act_win[:, lo:lo + H]                          # [N,H,7]
        X = np.concatenate([z[:, :-1], z[:, 1:]], axis=-1).reshape(-1, 2 * z.shape[-1])
        return X, a.reshape(-1, a.shape[-1])

    def __getitem__(self, i):
        return dict(i=i,
                    ctx_patch=torch.from_numpy(self.ctx_patch[i].astype(np.float32)),
                    pooled=torch.from_numpy(self.pooled[i].astype(np.float32)),
                    act_win=torch.from_numpy(self.act_win[i]))


def pad_actions(seq_list, max_len=None):
    """pad variable-length action sequences [n,7] to a common length.
    accepts numpy arrays or (fp16) tensors; always returns fp32."""
    from common import H_MAX_ACT
    seqs = [torch.as_tensor(a, dtype=torch.float32) for a in seq_list]
    n = max(len(a) for a in seqs)
    if max_len is not None:
        n = max(n, max_len)
    n = min(n, H_MAX_ACT)
    out = torch.zeros(len(seqs), n, 7)
    for i, a in enumerate(seqs):
        out[i, : len(a)] = a[:n]
    return out


def pad_branches(seq_list):
    """pad_branches + lengths (v2.3): model gathers the final token at each
    sample's LAST VALID slot (official cape_model.py valid_len-1), pads are
    never read out. -> (acts [B,n,7] fp32, lengths [B] long)."""
    seqs = [torch.as_tensor(a, dtype=torch.float32) for a in seq_list]
    n = max(len(a) for a in seqs)
    out = torch.zeros(len(seqs), n, 7)
    lengths = torch.tensor([len(a) for a in seqs], dtype=torch.long)
    for i, a in enumerate(seqs):
        out[i, : len(a)] = a
    return out, lengths


def probe_parts(split):
    """-> list of (row_offset, mmap array [n,H+1,196,768] fp16), concatenated
    row order == probe_meta_{split}.npy order."""
    paths = sorted(glob.glob(os.path.join(CACHE, f"probe_patches_{split}_*.npy")))
    assert paths, f"no probe parts for {split} under {CACHE}"
    out, off = [], 0
    for p in paths:
        a = np.load(p, mmap_mode="r")
        out.append((off, a))
        off += len(a)
    return out


def probe_window_frame(split, row, fi=-1):
    """single [196,768] fp16 frame of probe window `row` (fi=-1 -> frame T)."""
    for off, a in probe_parts(split):
        if off <= row < off + len(a):
            return a[row - off, fi]
    raise IndexError(f"probe row {row} out of range")


def id_pairs_from_window(sample, k=4, rng=None):
    """42-protocol pairs: k random consecutive (z_j, z_{j+1}) -> a_j inside the
    probe window (patch-mean features, v2)."""
    rng = rng or np.random.default_rng()
    js = rng.choice(H, size=k, replace=False)
    lo = PAST - H
    z = torch.as_tensor(sample["pmean"][lo:lo + H + 1], dtype=torch.float32)
    a = torch.as_tensor(sample["act_win"][lo:lo + H], dtype=torch.float32)
    return (torch.stack([z[j] for j in js]),
            torch.stack([z[j + 1] for j in js]),
            torch.stack([a[j] for j in js]))
