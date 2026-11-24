"""Layer-2 probe: can actions be read from the space? (v2 = 42 MLP protocol)

Primary: fresh MLP [2D -> 256 -> 128 -> 7], 3000 steps, AdamW lr 3e-4,
batch 512, 3 probe seeds; input = concat(z_k, z_{k+1}) over the H-frame probe
window; per-dim + pooled R2, MAE. Ridge (train-stat z-scored) kept secondary.

Spaces:
  frozen    pmean pairs from cache shards (42 probes mean-pooled features)
  cape/fwd  arm state encodings of probe_patches frames (empty-action readout)
  id_direct IDNet's own head on pmean pairs (native readout, no fresh probe)

All spaces use the SAME window subset for the probe (probe_keep: all val +
first --probe-train train windows, fixed in build_cache) -> matched comparison.
H3 floor control: --y-shuffle permutes action blocks within task before fit.

usage:
  python probe_r2.py --space frozen
  python probe_r2.py --space cape --ckpt cache_v2/ckpt/cape/final.pt
  python probe_r2.py --space id_direct --ckpt cache_v2/ckpt/id/final.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from common import CACHE, CKPT_DIR, H, PAST, append_jsonl, seed_all
from modules import CAPEModel, FwdModel, IDNet
from windows import WindowDataset

ENC_BS = 32          # windows per encoder pass (x (H+1) frames)


class ProbeMLP(nn.Module):
    """42 App: fresh 2-hidden MLP probe [2D -> 256 -> 128 -> 7]."""

    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 7))

    def forward(self, x):
        return self.net(x)


def load_arm(space, ckpt):
    m = {"cape": CAPEModel, "fwd": FwdModel, "id": IDNet}[space]()
    sd = torch.load(ckpt, map_location="cpu")["model"]
    m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def arm_features(space, ds, enc_cache):
    """-> dict split -> Z [n_windows, H+1, D_enc] aligned with probe rows."""
    if os.path.exists(enc_cache + ".train.npy"):
        return {s: np.load(f"{enc_cache}.{s}.npy", mmap_mode="r") for s in ("train", "val")}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_arm(space, args_ckpt()).to(dev)
    from windows import probe_parts
    out = {}
    for split in ("train", "val"):
        zs = []
        for off, arr in probe_parts(split):
            for s in range(0, len(arr), ENC_BS):
                chunk = torch.from_numpy(arr[s:s + ENC_BS].astype(np.float32)).to(dev)
                b, f, p, d = chunk.shape
                z = model.encode_state(chunk.reshape(b * f, p, d)).reshape(b, f, -1)
                zs.append(z.cpu().numpy().astype(np.float16))
        out[split] = np.concatenate(zs)
        print(f"encoded {split}: {out[split].shape}", flush=True)
    for split in ("train", "val"):
        np.save(f"{enc_cache}.{split}.npy", out[split])
    return out


_ARGS = None


def args_ckpt():
    return _ARGS.ckpt


def probe_xy(space, ds, split):
    """-> (Z [n,H+1,D], rows) for THIS split, aligned with probe_meta_{split}."""
    key2row = {}
    for i in range(len(ds)):
        key2row[(int(ds.meta["ep"][i]), int(ds.meta["T"][i]))] = i
    pm = np.load(os.path.join(CACHE, f"probe_meta_{split}.npy"))
    rows = np.array([key2row[(int(e), int(T))] for e, T in pm])
    if space == "frozen":
        Z = ds.pmean[rows][:, PAST - H:PAST + 1].astype(np.float32)
    else:
        feats = arm_features(space, ds, os.path.join(CACHE, f"probe_enc_{space}.npy"))
        Z = np.asarray(feats[split], dtype=np.float32)
    return Z, rows


def pair_xy(Z_split, rows, ds):
    """(z_k, z_{k+1}) -> a_k over the H-frame window -> X [n*H, 2D], y [n*H, 7]."""
    z, y = Z_split, ds.act_win[rows][:, PAST - H:PAST].reshape(-1, 7)
    X = np.concatenate([z[:, :-1], z[:, 1:]], axis=-1).reshape(-1, z.shape[-1] * 2)
    return X.astype(np.float32), y.astype(np.float32)


def r2_report(y, pred):
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0) + 1e-12
    per_dim = 1 - ss_res / ss_tot
    pooled = 1 - ss_res.sum() / ss_tot.sum()
    return per_dim, pooled


def y_shuffle_blocks(y, rows, tasks, seed):
    """within-task permutation of whole H-step action blocks (floor control)."""
    rng = np.random.default_rng(seed)
    y2 = y.copy().reshape(-1, H, 7)
    blocks = {}
    for b in range(len(rows)):
        blocks.setdefault(int(tasks[rows[b]]), []).append(b)
    for _, ix in blocks.items():
        perm = rng.permutation(len(ix))
        y2[ix] = y2[np.array(ix)[perm]]
    return y2.reshape(-1, 7)


def main():
    global _ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=["frozen", "cape", "fwd", "id_direct"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--y-shuffle", action="store_true")
    ap.add_argument("--ridge", action="store_true", help="also report secondary ridge")
    a = ap.parse_args()
    _ARGS = a
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds_val = WindowDataset("val")

    out = {"space": a.space, "y_shuffle": a.y_shuffle}
    if a.space == "id_direct":
        model = load_arm("id", a.ckpt).to(dev)
        Xv, yv = ds_val.probe_pairs()
        with torch.no_grad():
            zv = torch.from_numpy(Xv)
            pred = model(zv[:, :768].to(dev), zv[:, 768:].to(dev)).cpu().numpy()
        per_dim, pooled = r2_report(yv, pred)
        out.update(n_pairs=len(yv), r2_dims=[round(float(x), 4) for x in per_dim],
                   r2_pooled=round(float(pooled), 4),
                   mae=round(float(np.abs(pred - yv).mean()), 4))
        print(out)
        append_jsonl(os.path.join(CACHE, "results", "probe.jsonl"), out)
        return

    ds_tr = WindowDataset("train")
    Ztr, rtr = probe_xy(a.space, ds_tr, "train")
    Zva, rva = probe_xy(a.space, ds_val, "val")
    Xtr, ytr = pair_xy(Ztr, rtr, ds_tr)
    Xva, yva = pair_xy(Zva, rva, ds_val)
    if a.y_shuffle:
        ytr = y_shuffle_blocks(ytr, rtr, ds_tr.meta["task"], 0)
        yva = y_shuffle_blocks(yva, rva, ds_val.meta["task"], 1)

    res = []
    for sd in range(a.seeds):
        seed_all(sd)
        torch.manual_seed(sd)
        probe = ProbeMLP(Xtr.shape[1]).to(dev)
        opt = torch.optim.AdamW(probe.parameters(), lr=a.lr)
        Xt = torch.from_numpy(Xtr)
        yt = torch.from_numpy(ytr)
        rng = np.random.default_rng(sd)
        for step in range(a.steps):
            ix = rng.integers(0, len(Xt), size=a.bs)
            loss = nn.functional.mse_loss(probe(Xt[ix].to(dev)), yt[ix].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = np.concatenate([probe(torch.from_numpy(Xva[s:s + 4096]).to(dev)
                                         ).cpu().numpy()
                                   for s in range(0, len(Xva), 4096)])
        per_dim, pooled = r2_report(yva, pred)
        res.append((pooled, per_dim, float(np.abs(pred - yva).mean())))
        print(f"seed {sd}: pooled {pooled:.4f} mae {res[-1][2]:.4f}", flush=True)

    pooled = np.array([r[0] for r in res])
    per_dim = np.stack([r[1] for r in res]).mean(0)
    out.update(n_pairs=len(yva), n_train=len(ytr),
               r2_pooled_mean=round(float(pooled.mean()), 4),
               r2_pooled_std=round(float(pooled.std()), 4),
               r2_dims=[round(float(x), 4) for x in per_dim],
               mae=round(float(np.mean([r[2] for r in res])), 4))
    if a.ridge:
        mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-6
        W = np.linalg.solve((Xtr - mu).T @ (Xtr - mu) / sd_ / sd_ + 1.0 * np.eye(Xtr.shape[1]),
                            (Xtr - mu).T @ ytr / sd_)
        pred = (Xva - mu) @ W / sd_
        _, pooled_r = r2_report(yva, pred)
        out["r2_ridge"] = round(float(pooled_r), 4)
    print(out)
    append_jsonl(os.path.join(CACHE, "results", "probe.jsonl"), out)


if __name__ == "__main__":
    main()
