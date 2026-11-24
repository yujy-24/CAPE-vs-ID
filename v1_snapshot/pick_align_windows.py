"""Alignment gate, take 2: dump windows that actually straddle gripper events.

build_cache --dump-frames takes the FIRST k train windows (sorted by episode/T),
which all happened to be static holds -> trivially aligned, no test power.
Here we score every train window by gripper-cmd swing inside the window and
dump the top-k (distinct episodes preferred), so the grids contain real
open<->close transitions to check frame<->action alignment against.

usage: python pick_align_windows.py [--k 5]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_cache import episode_tables, seq_read, video_path
from common import CACHE, H, MANIFEST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_parquet(MANIFEST)
    df = df[df.split == "train"]
    ep_df, acts = episode_tables()
    row_of = {int(r.episode_index): r for _, r in ep_df.iterrows()}
    off_of = {int(r.episode_index): int(round(
        float(r["videos/observation.images.image/from_timestamp"]) * 20))
        for _, r in ep_df.iterrows()}

    scored = []
    for _, r in df.iterrows():
        e, T = int(r["episode"]), int(r["T"])
        g = acts[e][T - H:T, -1]          # gripper cmd over the window
        arm = acts[e][T - H:T, :3]        # xyz cmd
        swing = float(g.max() - g.min())  # 0 = flat, big = open<->close event
        motion = float(np.abs(np.diff(arm, axis=0)).mean())
        scored.append((swing, motion, e, T))
    scored.sort(reverse=True)

    out = os.path.join(CACHE, "align_check")
    os.makedirs(out, exist_ok=True)
    used_ep = set()
    n = 0
    for swing, motion, e, T in scored:
        if n >= args.k:
            break
        if e in used_ep:                  # prefer distinct episodes
            continue
        row = row_of[e]
        off = off_of[e]
        have, _ = seq_read(video_path(row), list(range(off + T - H, off + T + 1)))
        if len(have) < H + 1:
            continue
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        for i in range(H + 1):
            axes.flat[i].imshow(have[off + T - H + i]); axes.flat[i].axis("off")
            axes.flat[i].set_title(f"t={T-H+i}", fontsize=6)
        ax = axes.flat[11]
        ax.plot(range(H), acts[e][T - H:T, -1], "o-")
        ax.set_title(f"grip swing={swing:.2f}", fontsize=6)
        fig.savefig(os.path.join(out, f"grip_ep{e}_T{T}.jpg"), dpi=80)
        plt.close(fig)
        used_ep.add(e)
        n += 1
        print(f"dumped ep{e}_T{T} (grip swing {swing:.2f}, arm motion {motion:.3f})")
    print(f"align_check take2: {n} grids -> {out}/grip_*.jpg")


if __name__ == "__main__":
    main()
