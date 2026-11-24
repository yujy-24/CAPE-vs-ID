"""Shared config / paths / utils. Run everything inside WSL (lawam env).

v2 protocol (design doc §11 amendment, 2026-11-24): CAPE-faithful branches,
empty-action state encoding, same-episode grouped negatives, MLP probe.
"""
import json
import os
import random

import numpy as np
import torch

# ---- paths (WSL side) ----
DATA_ROOT = "/home/yjy/LaWAM/dataset/libero_merged_no_noops_20hz"
BB_DIR = "/home/yjy/LaWAM/weights/dinov3-vitb16-pretrain-lvd1689m"
DIM = 768          # DINOv3 ViT-B/16 hidden size (patch tokens + CLS)
IMG = 224          # input resolution -> (224/16)^2 = 196 patch tokens
PROJ = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = "v2"
CACHE = os.environ.get("IDVC_CACHE", os.path.join(PROJ, f"cache_{PROTOCOL}"))
MANIFEST = os.path.join(CACHE, "manifest.parquet")
CKPT_DIR = os.path.join(CACHE, "ckpt")

# ---- experiment constants ----
FPS = 20
H = 10            # probe window frames (T-H..T), unchanged from v1
SEED = 42
N_TRAIN, N_VAL = 8000, 1000

# v2 branch sampling (paper: two intermediate contexts, stride-0.4s granularity,
# remaining action sequences of different lengths converge to the same o_T):
#   t2 = T - U[T2_LO, T2_HI]      (branch-2 remaining: 5..12 steps = 0.25..0.6 s)
#   t1 = t2 - U[GAP_LO, GAP_HI]   (context gap 5..12 steps = 0.25..0.6 s)
#   branch-1 remaining = t1 -> T  (10..24 steps = 0.5..1.2 s)
T2_LO, T2_HI = 5, 12
GAP_LO, GAP_HI = 5, 12
T2_HI, GAP_HI = max(T2_LO, T2_HI), max(GAP_LO, GAP_HI)
PAST = T2_HI + GAP_HI            # 24: earliest frame we ever read = T-24
H_MAX_ACT = PAST                 # longest remaining action sequence
FUTS = 3                         # temporal-neighbour future frames T+1..T+3
POOL_LEN = PAST + 1 + FUTS + 1   # 29 pooled slots (+1 random negative)

# pooled-slot layout: idx 0..PAST-1 -> [T-PAST .. T-1], idx PAST -> T,
#   PAST+1..PAST+FUTS -> T+1..T+3, PAST+FUTS+1 -> random negative frame.
# NOTE v2: window starts at T-PAST (probe pairs still use its last H+1 frames).
IDX_T = PAST                     # slot of frame T

# ---- matched training config (design doc §4) ----
STEPS = 20000
BS = 8                           # id/fwd micro-batch
LR, WD = 1e-4, 1e-4              # v2: paper App.A (cape) / App.G range (id)
WARMUP = 2000
ID_ACCUM = 2                     # id effective batch = BS * ID_ACCUM = 16
# cape grouped batch: GROUPS episodes x PER_GROUP windows (paper: 16/group)
GROUPS, PER_GROUP = 8, 4
CAPE_BS = GROUPS * PER_GROUP     # 32
TAU = 0.07                       # paper: fixed, untuned


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


def cosine_schedule(step, total, warmup=WARMUP):
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + np.cos(np.pi * t))
