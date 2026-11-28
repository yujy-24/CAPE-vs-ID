"""P0.5 video-side polarity check (read-only): does command=1 mean the same
physical gripper state across LIBERO subsets in the merged dataset?

For one episode per target task, dump frames from the middle of the longest
0-run, middle of the longest 1-run, and both sides of a 0->1 and a 1->0
transition, into a labelled grid.  Human/VLM reads the grids.

usage: python diag_gripper_polarity.py   (writes results/gripper_convention_20261128/frames/)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_cache import episode_tables, stream_frames, video_path
from common import DATA_ROOT

TARGETS = [0, 2, 8, 38]          # train ref / suspect / weak suspect / val ref
OUT = os.path.join("results", "gripper_convention_20261128", "frames")


def runs(g):
    """[(value, start, end_inclusive), ...]"""
    edges = np.flatnonzero(np.diff(g.astype(np.int8)) != 0)
    starts = np.concatenate([[0], edges + 1])
    ends = np.concatenate([edges, [len(g) - 1]])
    return [(int(g[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def pick_episode(ep_df, acts, task_name):
    eps = ep_df[ep_df.tasks == task_name]
    assert len(eps), f"no episodes for task {task_name}"
    row = eps.iloc[len(eps) // 2]                     # median-position episode
    return int(row.episode_index), row


def wanted_frames(g):
    rr = runs(g)
    r0 = max([r for r in rr if r[0] == 0], key=lambda r: r[2] - r[1], default=None)
    r1 = max([r for r in rr if r[0] == 1], key=lambda r: r[2] - r[1], default=None)
    t01 = next((i for i in range(len(g) - 1) if g[i] == 0 and g[i + 1] == 1), None)
    t10 = next((i for i in range(len(g) - 1) if g[i] == 1 and g[i + 1] == 0), None)
    picks = []
    if r0: picks += [("0run_mid", (r0[1] + r0[2]) // 2), ("0run_mid2", min(r0[2], (r0[1] + r0[2]) // 2 + 8))]
    if t01 is not None: picks += [("0->1_last0", t01), ("0->1_first1", t01 + 1)]
    if r1: picks += [("1run_mid", (r1[1] + r1[2]) // 2), ("1run_mid2", min(r1[2], (r1[1] + r1[2]) // 2 + 8))]
    if t10 is not None: picks += [("1->0_last1", t10), ("1->0_first0", t10 + 1)]
    return picks[:8]


def main():
    ep_df, acts = episode_tables()
    names = pd.read_parquet(os.path.join(DATA_ROOT, "meta/tasks.parquet"))
    os.makedirs(OUT, exist_ok=True)
    for task in TARGETS:
        name = str(names.index[task])
        ep, row = pick_episode(ep_df, acts, name)
        g = acts[ep][:, -1]
        picks = wanted_frames(g)
        off = int(round(float(row["videos/observation.images.image/from_timestamp"]) * 20))
        path = video_path(row)
        frames = {i: rgb for i, rgb in stream_frames(path, sorted(off + f for _, f in picks))}
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, (label, f) in zip(axes.flat, picks):
            ax.imshow(frames[off + f]); ax.axis("off")
            ax.set_title(f"{label}  f={f} cmd={g[f]:.0f}", fontsize=9)
        for ax in axes.flat[len(picks):]:
            ax.axis("off")
        fig.suptitle(f"task {task} ep {ep}: {name}", fontsize=10)
        fig.savefig(os.path.join(OUT, f"polarity_task{task:02d}_ep{ep}.jpg"), dpi=85)
        plt.close(fig)
        print(f"task {task} ep {ep}: {len(picks)} frames dumped ({name})")


if __name__ == "__main__":
    main()
