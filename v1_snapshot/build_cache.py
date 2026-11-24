"""Extract frozen features for every manifest window -> cache shards.

Design:
  * dataset has only 4 video files (AV1!); cv2 cannot decode AV1 and torchvision
    whole-file reads OOM -> PyAV (libdav1d) SEQUENTIAL streaming decode, one pass
    per file, yielding frames in index order.
  * RGB memory stays ~1 episode: as soon as the stream passes an episode's last
    wanted frame we run the backbone for that episode and free its RGB.
  * random negatives are sampled (in build_manifest) only from episodes with a
    SMALLER episode index, so their CLS is already extracted when consumed.
  * patch tokens are kept only for o_{t1}/o_{t2} of each window, freed after
    the file's assemble loop.
  * integrity gate: a window missing ANY feature is dropped+counted (never
    silently substituted); >5% dropped aborts.

Per window (fp16 unless noted):
  ctx_patch [2, 196, 768] / pooled [15, 768] / act_win [H,7] f32 / meta int cols

usage: python build_cache.py [--bs 32] [--dump-frames 5]
"""
import argparse
import glob
import os
from collections import defaultdict

import av
import numpy as np
import pandas as pd
import torch

from backbone import FrozenDINOv3
from common import CACHE, DATA_ROOT, H, MANIFEST, device

SHARD = 512


def episode_tables():
    eps = sorted(glob.glob(os.path.join(DATA_ROOT, "meta/episodes/chunk-*/*.parquet")))
    ep_df = (pd.concat([pd.read_parquet(p) for p in eps], ignore_index=True)
             .sort_values("episode_index").reset_index(drop=True))
    dats = sorted(glob.glob(os.path.join(DATA_ROOT, "data/chunk-*/*.parquet")))
    d = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "action"])
                   for p in dats], ignore_index=True)
    d = d.sort_values(["episode_index", "frame_index"])
    acts = {int(e): np.stack([np.asarray(a, dtype=np.float32) for a in g["action"]])
            for e, g in d.groupby("episode_index")}
    return ep_df, acts


def video_path(row):
    return os.path.join(DATA_ROOT, "videos/observation.images.image",
                        f"chunk-{int(row['videos/observation.images.image/chunk_index']):03d}",
                        f"file-{int(row['videos/observation.images.image/file_index']):03d}.mp4")


def seq_read(path, wanted):
    """sequential PyAV pass -> ({file-local idx: rgb}, n_decoded). AV1 via libdav1d."""
    have, i = {}, 0
    want = iter(wanted)
    nxt = next(want, None)
    with av.open(path) as c:
        for fr in c.decode(video=0):
            if i == nxt:
                have[i] = fr.to_ndarray(format="rgb24")
                nxt = next(want, None)
                if nxt is None:
                    i += 1
                    break
            i += 1
    return have, i


def stream_frames(path, wanted):
    """yield (file-local idx, rgb uint8) for sorted `wanted`, decoding sequentially."""
    want = iter(wanted)
    nxt = next(want, None)
    i = 0
    with av.open(path) as c:
        for fr in c.decode(video=0):
            if i == nxt:
                yield i, fr.to_ndarray(format="rgb24")
                nxt = next(want, None)
                if nxt is None:
                    return
            i += 1


def dump_align_checks(df, ep_df, acts, k):
    """visual spot-check: first k train windows -> jpg grid (frames + gripper cmd)."""
    try:
        import matplotlib
    except ImportError:
        print("[warn] matplotlib missing - skip align_check dump "
              "(pip install matplotlib, rerun with --dump-frames)")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    row_of = {int(r.episode_index): r for _, r in ep_df.iterrows()}
    os.makedirs(os.path.join(CACHE, "align_check"), exist_ok=True)
    n = 0
    for _, r in df[df.split == "train"].iterrows():
        if n >= k:
            break
        e, T = int(r["episode"]), int(r["T"])
        row = row_of[e]
        off = int(round(float(row["videos/observation.images.image/from_timestamp"]) * 20))
        have, _ = seq_read(video_path(row), list(range(off + T - H, off + T + 1)))
        if len(have) < H + 1:
            continue
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        for i in range(H + 1):
            axes.flat[i].imshow(have[off + T - H + i]); axes.flat[i].axis("off")
            axes.flat[i].set_title(f"t={T-H+i}", fontsize=6)
        axes.flat[11].plot(range(H), acts[e][T - H:T, -1], "o-")
        axes.flat[11].set_title("gripper cmd", fontsize=6)
        fig.savefig(os.path.join(CACHE, "align_check", f"ep{e}_T{T}.jpg"), dpi=80)
        plt.close(fig)
        n += 1
    print(f"align_check: dumped {n} jpg grids -> {CACHE}/align_check/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dump-frames", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_parquet(MANIFEST).sort_values(["split", "episode", "T"]).reset_index(drop=True)
    ep_df, acts = episode_tables()
    row_of = {int(r.episode_index): r for _, r in ep_df.iterrows()}
    off_of = {int(r.episode_index): int(round(
        float(r["videos/observation.images.image/from_timestamp"]) * 20))
        for _, r in ep_df.iterrows()}

    need = defaultdict(set)          # ep -> episode-local frames (cls)
    need_patch = defaultdict(set)    # ep -> frames needing patch tokens (t1/t2)
    for _, r in df.iterrows():
        e, T = int(r["episode"]), int(r["T"])
        need[e].update(range(T - H, T + 4))
        need_patch[e].update({int(r["t1"]), int(r["t2"])})
        need[int(r["rand_episode"])].add(int(r["rand_frame"]))

    by_file = defaultdict(list)      # video path -> [episodes] (ascending)
    for e in sorted(need):
        by_file[video_path(row_of[e])].append(e)

    feats = {}                       # (ep, frame) -> cls fp16 [768]
    patches = {}                     # (ep, frame) -> patch fp16 [196,768]
    bb = FrozenDINOv3(device=device())

    def run_backbone(e, local_rgb):
        idxs = sorted(local_rgb)
        frames = np.stack([local_rgb[i] for i in idxs])
        ps, cs = [], []
        for i in range(0, len(frames), args.bs):
            p, c = bb(torch.from_numpy(frames[i:i + args.bs]))
            ps.append(p.numpy()); cs.append(c.numpy())
        ps, cs = np.concatenate(ps), np.concatenate(cs)
        off = off_of[e]
        for j, fi in enumerate(idxs):
            f = fi - off
            feats[(e, f)] = cs[j]
            if f in need_patch[e]:
                patches[(e, f)] = ps[j]

    buf = {"train": [], "val": []}
    sid = {"train": 0, "val": 0}
    kept = {"train": 0, "val": 0}
    dropped = {"n": 0}

    def assemble(e):
        rows = df[(df.episode == e)]
        for _, r in rows.iterrows():
            T = int(r["T"])
            fs = list(range(T - H, T + 4))
            rk = (int(r["rand_episode"]), int(r["rand_frame"]))
            # integrity gate: drop the window if ANY required feature is missing.
            # (silent substitution of a neighbouring frame would poison features.)
            if (any((e, f) not in feats for f in fs) or rk not in feats
                    or (e, int(r["t1"])) not in patches or (e, int(r["t2"])) not in patches):
                dropped["n"] += 1
                continue
            rec = dict(
                ctx_patch=np.stack([patches[(e, int(r["t1"]))], patches[(e, int(r["t2"]))]]),
                pooled=np.stack([feats[(e, f)] for f in fs] + [feats[rk]]),
                act_win=acts[e][T - H:T],
                meta=(e, T, int(r["t1"]), int(r["t2"]), int(r["task_index"]),
                      int(r["rand_episode"]), int(r["rand_frame"])),
            )
            s = str(r["split"])
            buf[s].append(rec)
            kept[s] += 1
            if len(buf[s]) >= SHARD:
                save_shard(buf[s], sid[s], s); buf[s] = []; sid[s] += 1

    def flush(e, cur):
        off = off_of[e]
        local = {off + f: cur.pop(off + f) for f in need[e] if (off + f) in cur}
        if local:
            run_backbone(e, local)
        return len(need[e]) - len(local)

    for p in sorted(by_file):
        eps_in_file = sorted(by_file[p], key=lambda e: off_of[e])
        wanted = sorted({off_of[e] + f for e in eps_in_file for f in need[e]})
        # flush an episode as soon as the stream passes its last wanted frame
        pend = sorted(eps_in_file, key=lambda e: off_of[e] + max(need[e]))
        ranges = {e: off_of[e] + max(need[e]) for e in eps_in_file}
        cur, miss = {}, 0
        for idx, rgb in stream_frames(p, wanted):
            cur[idx] = rgb
            while pend and ranges[pend[0]] <= idx:
                miss += flush(pend.pop(0), cur)
        for e in pend:                      # stream ended early
            miss += flush(e, cur)
        assert not cur, f"stray frames left in {p}"
        if miss:
            print(f"[warn] {os.path.basename(p)}: {miss} wanted frames never decoded",
                  flush=True)
        for e in eps_in_file:
            assemble(e)
            for f in list(need_patch[e]):
                patches.pop((e, f), None)
        print(f"{os.path.basename(p)} done "
              f"(feats {len(feats)}, patches {len(patches)})", flush=True)
    for s in ("train", "val"):
        if buf[s]:
            save_shard(buf[s], sid[s], s)
    print(f"windows kept {kept['train']} train / {kept['val']} val, "
          f"dropped {dropped['n']}")
    if dropped["n"] > 0.05 * len(df):
        raise SystemExit(f"dropped {dropped['n']} > 5% of windows - investigate decode")

    if args.dump_frames:
        dump_align_checks(df, ep_df, acts, args.dump_frames)


def save_shard(recs, shard_id, split):
    os.makedirs(CACHE, exist_ok=True)
    m = np.array([r["meta"] for r in recs], dtype=np.int64)
    out = os.path.join(CACHE, f"{split}_shard_{shard_id:04d}.npz")
    np.savez(out,
             ctx_patch=np.stack([r["ctx_patch"] for r in recs]),
             pooled=np.stack([r["pooled"] for r in recs]),
             act_win=np.stack([r["act_win"] for r in recs]),
             ep=m[:, 0], T=m[:, 1], t1=m[:, 2], t2=m[:, 3],
             task=m[:, 4], rand_ep=m[:, 5], rand_fr=m[:, 6])
    print(f"saved {out} ({len(recs)} windows)", flush=True)


if __name__ == "__main__":
    main()
