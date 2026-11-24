"""One-off: measure per-file wanted-frame concurrency + patch accumulation
(to size the OOM fix in build_cache)."""
import glob
import os
from collections import defaultdict

import av
import numpy as np
import pandas as pd

from common import CACHE, DATA_ROOT, FUTS, GAP_HI, H, PAST, T2_HI


def video_path(row):
    return os.path.join(DATA_ROOT, "videos/observation.images.image",
                        f"chunk-{int(row['videos/observation.images.image/chunk_index']):03d}",
                        f"file-{int(row['videos/observation.images.image/file_index']):03d}.mp4")


df = pd.read_parquet(os.path.join(CACHE, "manifest.parquet")).reset_index(drop=True)
eps = sorted(glob.glob(os.path.join(DATA_ROOT, "meta/episodes/chunk-*/*.parquet")))
ep_df = (pd.concat([pd.read_parquet(p) for p in eps], ignore_index=True)
         .sort_values("episode_index").reset_index(drop=True))
row_of = {int(r.episode_index): r for _, r in ep_df.iterrows()}
off_of = {int(r.episode_index): int(round(
    float(r["videos/observation.images.image/from_timestamp"]) * 20))
    for _, r in ep_df.iterrows()}

p0 = video_path(ep_df.iloc[0])
with av.open(p0) as c:
    for fr in c.decode(video=0):
        print("native frame:", fr.width, "x", fr.height)
        break

need = defaultdict(set)
need_patch = defaultdict(set)
probe_keep = np.zeros(len(df), dtype=bool)
probe_keep[(df.split == "val").to_numpy()] = True
tr = np.flatnonzero((df.split == "train").to_numpy())
probe_keep[tr[:2500]] = True
for pos, r in df.iterrows():
    e, T = int(r["episode"]), int(r["T"])
    need[e].update(range(T - PAST, T + FUTS + 1))
    pf = {int(r["t1"]), int(r["t2"])}
    if probe_keep[pos]:
        pf.update(range(T - H, T + 1))
    need_patch[e].update(pf)
    need[int(r["rand_episode"])].add(int(r["rand_frame"]))

by_file = defaultdict(list)
for e in sorted(need):
    by_file[video_path(row_of[e])].append(e)

tot_patch = 0
for p, elist in sorted(by_file.items()):
    wf = sorted({off_of[e] + f for e in elist for f in need[e]})
    events = []
    for e in elist:
        events.append((off_of[e] + min(need[e]), 1))
        events.append((off_of[e] + max(need[e]) + 1, -1))
    events.sort()
    alive = peak = 0
    for _, d in events:
        alive += d
        peak = max(peak, alive)
    npf = sum(len(need_patch[e]) for e in elist)
    tot_patch += npf
    print(os.path.basename(p), f"eps={len(elist)} wanted={len(wf)} "
          f"peak_concurrent_eps={peak} patch_frames={npf}")
print("total patch frames:", tot_patch,
      "=", round(tot_patch * 196 * 768 * 2 / 1e9, 2), "GB if all held")
