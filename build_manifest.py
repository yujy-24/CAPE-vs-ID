"""Build manifest.parquet: the single sample list all three arms consume (matched exposure).

v2 (design doc §11): CAPE-faithful branches -- two intermediate contexts whose
remaining action sequences have DIFFERENT lengths and both converge to o_T:
    t2 = T - U[5,12]   (branch-2 remaining 0.25-0.6 s)
    t1 = t2 - U[5,12]  (context gap  0.25-0.6 s; branch-1 remaining 0.5-1.2 s)
Task-level split (40 tasks -> 32 train / 8 val, seed 42) unchanged.
usage: python build_manifest.py
"""
import glob
import os

import numpy as np
import pandas as pd
from numpy.random import default_rng

from common import (CACHE, DATA_ROOT, FUTS, GAP_HI, GAP_LO, N_TRAIN, N_VAL,
                    SEED, T2_HI, T2_LO)


def load_tables():
    eps = sorted(glob.glob(os.path.join(DATA_ROOT, "meta/episodes/chunk-*/*.parquet")))
    ep_df = pd.concat([pd.read_parquet(p) for p in eps], ignore_index=True)
    dats = sorted(glob.glob(os.path.join(DATA_ROOT, "data/chunk-*/*.parquet")))
    cols = ["episode_index", "frame_index", "task_index"]
    d = pd.concat([pd.read_parquet(p, columns=cols) for p in dats], ignore_index=True)
    ep2task = d.groupby("episode_index")["task_index"].first()
    ep_df["task_index"] = ep_df["episode_index"].map(ep2task)
    ep_df = ep_df.sort_values("episode_index").reset_index(drop=True)
    return ep_df


def main():
    os.makedirs(CACHE, exist_ok=True)
    ep_df = load_tables()
    tasks = np.sort(ep_df["task_index"].unique())
    rng = default_rng(SEED)
    perm = rng.permutation(len(tasks))
    val_tasks = set(tasks[perm[-8:]].tolist())

    ep_df["split"] = np.where(ep_df["task_index"].isin(val_tasks), "val", "train")
    print(f"tasks: {len(tasks)} total, {len(val_tasks)} val -> "
          f"{(ep_df.split == 'train').sum()} train eps / {(ep_df.split == 'val').sum()} val eps")

    rows = []
    for split, n_target in [("train", N_TRAIN), ("val", N_VAL)]:
        pool = ep_df[ep_df.split == split].reset_index(drop=True)
        pool_eps = pool["episode_index"].to_numpy()
        cand = []
        for _, r in pool.iterrows():
            lo, hi = GAP_HI + T2_HI, int(r["length"]) - 1 - FUTS
            cand.extend((int(r["episode_index"]), t) for t in range(lo, hi))
        idx = rng.choice(len(cand), size=min(n_target, len(cand)), replace=False)
        # random negatives: any frame of any episode processed EARLIER (smaller
        # index) -> single-pass cache extraction without ordering dependency
        for i in idx:
            e, T = cand[i]
            pos = int(np.searchsorted(pool_eps, e))
            if pos == 0:
                continue
            r2_idx = int(rng.integers(0, pos))
            r2 = pool.iloc[r2_idx]
            rf = int(rng.integers(0, int(r2["length"])))
            t2 = T - int(rng.integers(T2_LO, T2_HI + 1))
            t1 = t2 - int(rng.integers(GAP_LO, GAP_HI + 1))
            rows.append(dict(split=split, episode=e, T=T, t1=t1, t2=t2,
                             task_index=int(pool.task_index[pool.episode_index == e].iloc[0]),
                             rand_episode=int(r2["episode_index"]), rand_frame=rf))

    df = pd.DataFrame(rows).sort_values(["split", "episode", "T"]).reset_index(drop=True)
    out = os.path.join(CACHE, "manifest.parquet")
    df.to_parquet(out)
    gap = df.t2 - df.t1
    rem1 = df["T"] - df.t1          # bracket: df.T is the transpose!
    print(f"saved {out}: {len(df)} windows "
          f"(train {(df.split=='train').sum()} / val {(df.split=='val').sum()})")
    print(f"branch stats: ctx gap {gap.min()}-{gap.max()} (med {gap.median():.0f}) frames "
          f"= {gap.median()/20:.2f}s; branch-1 remaining med {rem1.median():.0f} frames "
          f"= {rem1.median()/20:.2f}s")
    print(df.groupby("split")["task_index"].nunique().to_string())


if __name__ == "__main__":
    main()
