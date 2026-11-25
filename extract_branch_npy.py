"""One-off converter: branch_patch members of cache_v3 npz shards -> a single
mmap-able branch_{split}.npy per split.

WindowDataset loads npz members eagerly (~24 GB resident with branch_patch ->
OOM in the 15 GiB WSL VM). branch frames are only touched randomly at train
time, so they become an mmap'd sidecar instead; the npz keeps its member as
the canonical copy (build_cache.py output unchanged).

Run inside WSL lawam env:  python extract_branch_npy.py
"""
import glob
import os

import numpy as np

from common import CACHE, HV23


def convert(split):
    paths = sorted(glob.glob(os.path.join(CACHE, f"{split}_shard_*.npz")))
    assert paths, f"no {split} shards under {CACHE}"
    out_path = os.path.join(CACHE, f"branch_{split}.npy")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return
    ns = []
    for p in paths:                       # cheap pass: window counts from meta
        with np.load(p) as z:
            ns.append(len(z["ep"]))
    N = sum(ns)
    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16, shape=(N, HV23 + 1, 196, 768))
    off = 0
    for p, n in zip(paths, ns):
        with np.load(p) as z:
            bp = z["branch_patch"]        # [n, HV23+1, 196, 768] fp16
            assert bp.shape[0] == n and bp.shape[1] == HV23 + 1, bp.shape
            out[off:off + n] = bp
        off += n
        print(f"  {os.path.basename(p)}: +{n} (total {off})", flush=True)
    out.flush()
    print(f"saved {out_path} ({N} windows, {out.nbytes/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    for split in ("train", "val"):
        convert(split)
