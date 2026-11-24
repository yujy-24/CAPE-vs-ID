"""Extract frozen features for every manifest window -> cache shards (v2).

v2 changes (design doc §11):
  * pooled span = [T-PAST .. T+FUTS] + rand  (29 slots, PAST=24)
  * patch tokens kept for t1/t2 (branch contexts) AND [T-H..T] (probe window);
    the latter go to a separate mmap-able .npy (probe_patches) so the main
    shards stay small and the Dataset stays RAM-friendly.
  * pmean [29,768] = mean over patch tokens for every pooled frame
    (42 paper probes mean-pooled features, not CLS).
  * act_win stores [T-PAST..T) actions (PAST,7); branches slice t1/t2 -> T.

Decode machinery unchanged: PyAV sequential streaming, per-episode flush,
integrity gate (drop window if ANY feature missing, >5% aborts).

usage: python build_cache.py [--bs 32] [--probe-train 2500]
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
from common import CACHE, DATA_ROOT, FUTS, H, IDX_T, MANIFEST, PAST, device

SHARD = 512
N_PATCH_HIST = 256          # probe-patch flush granularity (windows); parts on
                            # disk, else the final np.stack would need ~2x11.6GB RAM


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--probe-train", type=int, default=2500,
                    help="first-k train windows (manifest order) that store probe patches")
    args = ap.parse_args()

    df = pd.read_parquet(MANIFEST).sort_values(["split", "episode", "T"]).reset_index(drop=True)
    probe_keep = np.zeros(len(df), dtype=bool)
    probe_keep[(df.split == "val").to_numpy()] = True
    tr_idx = np.flatnonzero((df.split == "train").to_numpy())
    probe_keep[tr_idx[:args.probe_train]] = True     # fixed subset, order-stable

    ep_df, acts = episode_tables()
    row_of = {int(r.episode_index): r for _, r in ep_df.iterrows()}
    off_of = {int(r.episode_index): int(round(
        float(r["videos/observation.images.image/from_timestamp"]) * 20))
        for _, r in ep_df.iterrows()}

    need = defaultdict(set)          # ep -> episode-local frames (cls + pmean)
    need_patch = defaultdict(set)    # ep -> frames needing FULL patch tokens
    for pos, r in df.iterrows():     # df was reset_index'd -> pos == row position
        e, T = int(r["episode"]), int(r["T"])
        need[e].update(range(T - PAST, T + FUTS + 1))
        pf = {int(r["t1"]), int(r["t2"])}
        if probe_keep[pos]:
            pf.update(range(T - H, T + 1))          # probe window frames
        need_patch[e].update(pf)
        need[int(r["rand_episode"])].add(int(r["rand_frame"]))
    # (rand frames already added: they occupy pooled slot PAST+FUTS+1)

    by_file = defaultdict(list)
    for e in sorted(need):
        by_file[video_path(row_of[e])].append(e)

    feats = {}                       # (ep, frame) -> cls fp16 [768]
    pmeans = {}                      # (ep, frame) -> patch-mean fp16 [768]
    patches = {}                     # (ep, frame) -> patch fp16 [196,768]
    bb = FrozenDINOv3(device=device())

    def run_backbone(e, local_rgb):
        idxs = sorted(local_rgb)
        frames = np.stack([local_rgb[i] for i in idxs])
        ps, cs, pm = [], [], []
        for i in range(0, len(frames), args.bs):
            p, c, m = bb(torch.from_numpy(frames[i:i + args.bs]))
            ps.append(p.numpy()); cs.append(c.numpy()); pm.append(m.numpy())
        ps, cs, pm = np.concatenate(ps), np.concatenate(cs), np.concatenate(pm)
        off = off_of[e]
        for j, fi in enumerate(idxs):
            f = fi - off
            feats[(e, f)] = cs[j]
            pmeans[(e, f)] = pm[j]
            if f in need_patch[e]:
                patches[(e, f)] = ps[j]

    buf = {"train": [], "val": []}
    sid = {"train": 0, "val": 0}
    kept = {"train": 0, "val": 0}
    dropped = {"n": 0}
    pbuf = {"train": [], "val": []}  # probe-patch records (same order as main buf)
    pid = {"train": 0, "val": 0}     # probe part counter
    pmeta = {"train": [], "val": []}   # (episode, T) per probe row, in order

    def assemble(e):
        rows = df[(df.episode == e)]
        for pos, r in rows.iterrows():
            T = int(r["T"])
            fs = list(range(T - PAST, T + FUTS + 1))
            rk = (int(r["rand_episode"]), int(r["rand_frame"]))
            if (any((e, f) not in feats for f in fs) or rk not in feats
                    or (e, int(r["t1"])) not in patches or (e, int(r["t2"])) not in patches):
                dropped["n"] += 1
                continue
            rec = dict(
                ctx_patch=np.stack([patches[(e, int(r["t1"]))], patches[(e, int(r["t2"]))]]),
                pooled=np.stack([feats[(e, f)] for f in fs] + [feats[rk]]),
                pmean=np.stack([pmeans[(e, f)] for f in fs] + [pmeans[rk]]),
                act_win=acts[e][T - PAST:T],
                meta=(e, T, int(r["t1"]), int(r["t2"]), int(r["task_index"]),
                      int(r["rand_episode"]), int(r["rand_frame"])),
            )
            s = str(r["split"])
            if probe_keep[pos]:
                pwin = np.stack([patches[(e, f)] for f in range(T - H, T + 1)])
                pbuf[s].append(pwin)
                pmeta[s].append((e, T))
            buf[s].append(rec)
            kept[s] += 1
            if len(buf[s]) >= SHARD:
                save_shard(buf[s], sid[s], s); buf[s] = []; sid[s] += 1
            if len(pbuf[s]) >= N_PATCH_HIST:
                save_probe_part(pbuf[s], pid[s], s); pbuf[s] = []; pid[s] += 1

    def flush(e, cur):
        off = off_of[e]
        local = {off + f: cur.pop(off + f) for f in need[e] if (off + f) in cur}
        if local:
            run_backbone(e, local)
        return len(need[e]) - len(local)

    for p in sorted(by_file):
        eps_in_file = sorted(by_file[p], key=lambda e: off_of[e])
        wanted = sorted({off_of[e] + f for e in eps_in_file for f in need[e]})
        pend = sorted(eps_in_file, key=lambda e: off_of[e] + max(need[e]))
        ranges = {e: off_of[e] + max(need[e]) for e in eps_in_file}
        cur, miss = {}, 0
        for idx, rgb in stream_frames(p, wanted):
            cur[idx] = rgb
            while pend and ranges[pend[0]] <= idx:
                e = pend.pop(0)
                miss += flush(e, cur)
                # assemble immediately + free THIS episode's patches:
                # v2 keeps ~45k patch frames total (13.6 GB) if held to file end
                # (probe windows cluster in early episodes -> file-000 alone 9 GB).
                assemble(e)
                for f in need_patch[e]:
                    patches.pop((e, f), None)
        for e in pend:
            miss += flush(e, cur)
            assemble(e)
            for f in need_patch[e]:
                patches.pop((e, f), None)
        assert not cur, f"stray frames left in {p}"
        if miss:
            print(f"[warn] {os.path.basename(p)}: {miss} wanted frames never decoded",
                  flush=True)
        print(f"{os.path.basename(p)} done (feats {len(feats)}, patches {len(patches)})",
              flush=True)
    for s in ("train", "val"):
        if buf[s]:
            save_shard(buf[s], sid[s], s)
        if pbuf[s]:
            save_probe_part(pbuf[s], pid[s], s)
        if pmeta[s]:
            meta = np.array(pmeta[s], dtype=np.int64)
            np.save(os.path.join(CACHE, f"probe_meta_{s}.npy"), meta)
            print(f"probe rows {s}: {len(meta)}", flush=True)
    print(f"windows kept {kept['train']} train / {kept['val']} val, "
          f"dropped {dropped['n']}")
    if dropped["n"] > 0.05 * len(df):
        raise SystemExit(f"dropped {dropped['n']} > 5% of windows - investigate decode")


def save_shard(recs, shard_id, split):
    os.makedirs(CACHE, exist_ok=True)
    m = np.array([r["meta"] for r in recs], dtype=np.int64)
    out = os.path.join(CACHE, f"{split}_shard_{shard_id:04d}.npz")
    np.savez(out,
             ctx_patch=np.stack([r["ctx_patch"] for r in recs]),
             pooled=np.stack([r["pooled"] for r in recs]),
             pmean=np.stack([r["pmean"] for r in recs]),
             act_win=np.stack([r["act_win"] for r in recs]),
             ep=m[:, 0], T=m[:, 1], t1=m[:, 2], t2=m[:, 3],
             task=m[:, 4], rand_ep=m[:, 5], rand_fr=m[:, 6])
    print(f"saved {out} ({len(recs)} windows)", flush=True)


def save_probe_part(pwins, part, split):
    """one chunk of probe-window patches [n, H+1, 196, 768] fp16 -> numbered
    part file; consumers iterate parts (windows.probe_parts) with mmap."""
    os.makedirs(CACHE, exist_ok=True)
    arr = np.stack(pwins)
    out = os.path.join(CACHE, f"probe_patches_{split}_{part:03d}.npy")
    np.save(out, arr)
    print(f"saved {os.path.basename(out)} {arr.shape} ({arr.nbytes/1e9:.2f} GB)",
          flush=True)


if __name__ == "__main__":
    main()
