"""P0.5 follow-up (read-only): task 8 early-phase polarity check.

diag_gripper_polarity.py showed task 8 ep189's FIRST 0->1 transition at f=78
with the bowl apparently already inside the drawer -- impossible under the
standard convention (1=grasp) since cmd was 0 for all of f=0..78.  Dump the
early phase (f=0,15,...,75,78,79) for two task-8 episodes to decide:
inverted convention (arm carries the bowl while cmd=0) vs VLM misread
(bowl still on the counter until the transition).

usage: python diag_gripper_task8_early.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_cache import episode_tables, stream_frames, video_path
from common import DATA_ROOT

TASK = "put the black bowl in the bottom drawer of the cabinet and close it"
PICKS = [0, 15, 30, 45, 60, 75, 78, 79]
OUT = os.path.join("results", "gripper_convention_20261128", "frames")


def main():
    ep_df, acts = episode_tables()
    eps = ep_df[ep_df.tasks == TASK]
    chosen = [int(eps.iloc[i].episode_index) for i in (len(eps) // 3, 2 * len(eps) // 3)]
    for ep in chosen:
        row = ep_df[ep_df.episode_index == ep].iloc[0]
        g = acts[ep][:, -1]
        off = int(round(float(row["videos/observation.images.image/from_timestamp"]) * 20))
        frames = {i: rgb for i, rgb in stream_frames(video_path(row), sorted(off + f for f in PICKS))}
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, f in zip(axes.flat, PICKS):
            ax.imshow(frames[off + f]); ax.axis("off")
            ax.set_title(f"f={f} cmd={g[f]:.0f}", fontsize=9)
        fig.suptitle(f"task 8 ep {ep} early phase", fontsize=10)
        fig.savefig(os.path.join(OUT, f"task8_early_ep{ep}.jpg"), dpi=85)
        plt.close(fig)
        print(f"ep {ep}: frames {PICKS} dumped, cmd values {g[PICKS].astype(int).tolist()}")


if __name__ == "__main__":
    main()
