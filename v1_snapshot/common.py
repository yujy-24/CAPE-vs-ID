"""Shared config / paths / utils. Run everything inside WSL (lawam env)."""
import json
import os
import random

import numpy as np
import torch

# ---- paths (WSL side) ----
DATA_ROOT = "/home/yjy/LaWAM/dataset/libero_merged_no_noops_20hz"
BB_DIR = "/home/yjy/LaWAM/weights/dinov3-vitb16-pretrain-lvd1689m"
PROJ = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("IDVC_CACHE", os.path.join(PROJ, "cache"))
MANIFEST = os.path.join(CACHE, "manifest.parquet")
CKPT_DIR = os.path.join(CACHE, "ckpt")

# ---- experiment constants (registered in the design doc) ----
H = 10            # horizon in frames (20 fps -> 0.5 s)
FUT_DELTA = 3     # temporal-neighbour negatives: T+1 .. T+3
IMG = 224         # backbone input size -> 196 patch tokens for ViT-B/16
DIM = 768         # ViT-B/16 hidden
SEED = 42
N_TRAIN, N_VAL = 8000, 1000
STEPS, BS, LR = 20000, 8, 3e-4   # matched across arms

# pooled-feature layout inside each cached window:
#   idx 0..H        -> frames [T-H .. T]          (window, H+1 vectors)
#   idx H+1..H+3    -> frames [T+1 .. T+3]        (future temporal neighbours)
#   idx H+4         -> random negative frame (other episode)
POOL_WIN = H + 1
POOL_FUT0 = H + 1


def seed_all(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def cosine_schedule(step, total, warmup=1000):
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + np.cos(np.pi * t))
