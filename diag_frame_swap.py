"""P0 frame-swap sensitivity (read-only): senior's 2026-11-28 diagnostic.

Fix the first frame s_t; replace the second frame with s_{t+j} (j in 1/2/4/8/16,
plus static j=0), a random frame from the same episode, or a random frame from
another episode (same task / different task).  If predictions barely move while
the second frame changes -- or collapse toward the training prior -- the ID head
is not reading pair change information (mean-guessing suspicion).

Evaluated on (a) same-task unseen episodes (in-distribution, reuses
diagnose_id.unseen_same_task) and (b) unseen episodes of the 8 held-out tasks
(out-of-distribution, features extracted here).  No checkpoint or cache is
modified; results are diagnostic only.

usage (WSL, repo root):
  python diag_frame_swap.py --ckpt cache_v3/ckpt/id/final.pt \
      --out results/frame_swap_20261128/report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backbone import FrozenDINOv3
from build_cache import episode_tables, stream_frames, video_path
from common import BB_DIR, CACHE, DATA_ROOT
from diagnose_id import predict, training_window_action_mean, unseen_same_task
from modules import IDNet

GAPS = (1, 2, 4, 8, 16)          # second-frame lookahead in frames (20 Hz)
MAXJ = max(GAPS)
EPS_PER_TASK = 10                # held-out episodes per task
T_LO, ANCHOR_STEP = 20, 2        # anchor frame range / stride
GROUPS = (("translation", slice(0, 3)), ("rotation", slice(3, 6)), ("gripper", slice(6, 7)))


def heldout_feature_set(cache: Path, data_root: Path, device: str) -> dict:
    """Features + actions for EPS_PER_TASK evenly spaced episodes of each
    held-out task (never trained on, so any episode is unseen)."""
    manifest = pd.read_parquet(cache / "manifest.parquet")
    train_tasks = set(manifest.loc[manifest.split == "train", "task_index"].astype(int))
    tasks_tbl = pd.read_parquet(Path(data_root) / "meta/tasks.parquet")
    ix_by_name = {str(name): int(ix) for name, ix in tasks_tbl.task_index.items()}
    held_tasks = sorted(set(ix_by_name.values()) - train_tasks)

    ep_df, acts = episode_tables()
    by_task = defaultdict(list)
    for _, row in ep_df.iterrows():
        task = ix_by_name.get(str(row.tasks))
        if task in held_tasks:
            by_task[task].append(int(row.episode_index))

    selected = []
    for task in held_tasks:
        eps = sorted(by_task[task])
        take = np.unique(np.linspace(0, len(eps) - 1, min(EPS_PER_TASK, len(eps))).astype(int))
        selected += [(task, eps[i]) for i in take]

    rows = ep_df.set_index("episode_index")
    wanted_by_file, sizes = defaultdict(dict), {}
    for _, ep in selected:
        row = rows.loc[ep]
        size = int(row["length"])
        if len(acts[ep]) != size:
            raise ValueError(f"episode {ep}: actions {len(acts[ep])} != length {size}")
        off = int(round(float(row["videos/observation.images.image/from_timestamp"]) * 20))
        sizes[ep] = size
        for frame in range(size):
            key = off + frame
            if key in wanted_by_file[video_path(row)]:
                raise ValueError(f"duplicate video frame {video_path(row)}:{key}")
            wanted_by_file[video_path(row)][key] = (ep, frame)

    features = {ep: np.empty((sizes[ep], 768), np.float32) for _, ep in selected}
    backbone = FrozenDINOv3(path=str(BB_DIR), device=device)
    for path, mapping in sorted(wanted_by_file.items()):
        rgb_batch, positions = [], []

        def flush() -> None:
            if not rgb_batch:
                return
            _, _, pmean = backbone(torch.from_numpy(np.stack(rgb_batch)))
            for (ep, frame), vec in zip(positions, pmean.numpy()):
                features[ep][frame] = vec.astype(np.float32)
            rgb_batch.clear()
            positions.clear()

        for file_frame, rgb in stream_frames(str(path), sorted(mapping)):
            rgb_batch.append(rgb)
            positions.append(mapping[file_frame])
            if len(rgb_batch) == 32:
                flush()
        flush()

    return {"features": features, "acts": acts, "task_of": {ep: t for t, ep in selected},
            "tasks": held_tasks, "episodes": [ep for _, ep in selected]}


def episode_conditions(ep: int, feats: dict, acts: dict, rng: np.random.Generator,
                       eps_by_task: dict, all_eps: list, task_of: dict) -> dict:
    """Second-frame indices per condition for one episode's anchors."""
    size = len(acts[ep])
    anchors = np.arange(T_LO, size - MAXJ, ANCHOR_STEP)
    z = feats[ep]
    out = {"anchors": anchors, "z0": z[anchors]}
    for j in GAPS:
        out[f"gap{j}"] = z[anchors + j]
    out["static"] = z[anchors]

    same = rng.integers(0, size, len(anchors))
    for _ in range(10):
        bad = np.abs(same - anchors) <= 2
        if not bad.any():
            break
        same[bad] = rng.integers(0, size, int(bad.sum()))
    out["same_ep_rand"] = z[same]
    out["_same_idx"] = same

    others = sorted(e for e in eps_by_task.get(task_of[ep], []) if e != ep)
    if others:
        picks = rng.choice(others, len(anchors))
        s = rng.integers(0, np.array([len(acts[e]) for e in picks]))
        out["cross_ep_same_task"] = np.stack([feats[e][i] for e, i in zip(picks, s)])
        out["_ces_idx"] = (picks, s)
    foreign = [e for e in all_eps if task_of[e] != task_of[ep]]
    picks = rng.choice(foreign, len(anchors))
    s = rng.integers(0, np.array([len(acts[e]) for e in picks]))
    out["cross_ep_diff_task"] = np.stack([feats[e][i] for e, i in zip(picks, s)])
    out["_ced_idx"] = (picks, s)
    return out


def r2_mae(y: np.ndarray, p: np.ndarray) -> tuple:
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    sst = np.square(y - y.mean()).sum()
    r2 = float(1 - np.square(p - y).sum() / sst) if sst > 1e-12 else None
    return r2, float(np.abs(p - y).mean())


def condition_report(z0: np.ndarray, zs: np.ndarray, pred: np.ndarray, ref: np.ndarray,
                     y_t: np.ndarray, y_arr: np.ndarray, y_mid: np.ndarray) -> dict:
    move = pred - ref
    out = {
        "input_change_l2": round(float(np.linalg.norm(zs - z0, axis=1).mean()), 4),
        "pred_move_l2": round(float(np.linalg.norm(move, axis=1).mean()), 4),
        "pred_move_by_group": {n: round(float(np.abs(move[:, d]).mean()), 4) for n, d in GROUPS},
        "grip_flip_rate": round(float(((pred[:, 6] >= 0.5) != (ref[:, 6] >= 0.5)).mean()), 4),
        "pred_mean_grip": round(float(pred[:, 6].mean()), 4),
        "pred_std_grip": round(float(pred[:, 6].std()), 4),
        "r2_vs_a_t": None, "mae_vs_a_t": None,
        "r2_vs_arrival": None, "mae_vs_arrival": None,
        "r2_vs_mid": None, "grip": {},
    }
    out["r2_vs_a_t"], out["mae_vs_a_t"] = r2_mae(y_t, pred)
    out["r2_vs_arrival"], out["mae_vs_arrival"] = r2_mae(y_arr, pred)
    out["r2_vs_mid"], _ = r2_mae(y_mid, pred)
    for key, y in (("a_t", y_t), ("arrival", y_arr)):
        r2, mae = r2_mae(y[:, 6], pred[:, 6])
        out["grip"][key] = {"r2": round(r2, 4) if r2 is not None else None,
                            "mae": round(mae, 4)}
    for key in ("r2_vs_a_t", "mae_vs_a_t", "r2_vs_arrival", "mae_vs_arrival", "r2_vs_mid"):
        out[key] = round(out[key], 4) if out[key] is not None else None
    return out


def evaluate_set(model: IDNet, feats: dict, acts: dict, task_of: dict,
                 device: str, rng: np.random.Generator) -> dict:
    eps_by_task = defaultdict(list)
    for ep, task in task_of.items():
        eps_by_task[task].append(ep)
    per_cond = defaultdict(lambda: defaultdict(list))
    n_anchors = 0
    for ep in sorted(feats):
        conds = episode_conditions(ep, feats, acts, rng, eps_by_task, sorted(feats), task_of)
        anchors = conds["anchors"]
        if not len(anchors):
            continue
        n_anchors += len(anchors)
        a = acts[ep]
        y_t = a[anchors]
        ref = predict(model, conds["z0"], conds["gap1"], device)
        per_cond["n_eps"] += 1
        for name in ("static", *(f"gap{j}" for j in GAPS), "same_ep_rand",
                     "cross_ep_same_task", "cross_ep_diff_task"):
            if name not in conds:
                continue
            zs = conds[name]
            pred = predict(model, conds["z0"], zs, device)
            if name.startswith("gap"):
                j = int(name[3:])
                y_mid, y_arr = a[anchors + (j - 1) // 2], a[anchors + j - 1]
            else:
                y_mid = y_t
                if name == "same_ep_rand":
                    s = np.clip(conds["_same_idx"], 1, None)
                    y_arr = np.stack([a[int(i) - 1] for i in s])
                else:
                    picks, s = conds["_ces_idx"] if name == "cross_ep_same_task" else conds["_ced_idx"]
                    y_arr = np.stack([acts[int(e)][max(int(i) - 1, 0)] for e, i in zip(picks, s)])
            rep = condition_report(conds["z0"], zs, pred, ref, y_t, y_arr, y_mid)
            rep["n"] = len(anchors)
            for key, val in rep.items():
                per_cond[name][key].append(val)
        # reference condition itself (movement 0 by definition)
        for name in ("gap1",):
            pred = ref
            rep = condition_report(conds["z0"], conds[name], pred, ref, y_t,
                                   a[anchors + 0], a[anchors])
            rep["n"] = len(anchors)
            for key, val in rep.items():
                per_cond[name][key].append(val)
    out = {"n_anchors": n_anchors}
    for name, cols in per_cond.items():
        if name == "n_eps":
            out["n_episodes"] = cols
            continue
        agg = {}
        for key, vals in cols.items():
            agg[key] = round(float(np.mean(vals)), 4) if isinstance(vals[0], (int, float)) else vals
        if "grip" in agg:  # nested dicts averaged manually
            agg["grip"] = {src: {k: round(float(np.mean([g[src][k] for g in cols["grip"]])), 4)
                                 for k in ("r2", "mae")} for src in ("a_t", "arrival")}
        out[name] = agg
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("cache_v3/ckpt/id/final.pt"))
    parser.add_argument("--cache", type=Path, default=Path(CACHE))
    parser.add_argument("--data-root", type=Path, default=Path(DATA_ROOT))
    parser.add_argument("--out", type=Path, default=Path("results/frame_swap_20261128/report.json"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ckpt.get("config", {}).get("arm") != "id":
        raise ValueError("checkpoint is not an ID run")
    model = IDNet().to(device).eval()
    model.load_state_dict(ckpt["model"])

    prior = training_window_action_mean(args.cache)

    same, coverage = unseen_same_task(args.cache, args.data_root)
    held = heldout_feature_set(args.cache, args.data_root, device)

    sets = {
        "same_task_unseen": evaluate_set(
            model, same["features_by_episode"], same["actions_by_episode"],
            same_task_map(same), device, np.random.default_rng(0)),
        "heldout_tasks": evaluate_set(
            model, held["features"], held["acts"], held["task_of"],
            device, np.random.default_rng(0)),
    }

    report = {"checkpoint": str(args.ckpt), "train_prior_mean": prior.tolist(),
              "anchor_config": {"t_lo": T_LO, "stride": ANCHOR_STEP, "gaps": list(GAPS),
                                "eps_per_heldout_task": EPS_PER_TASK},
              "sets": sets}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for name, block in sets.items():
        print(f"== {name} (n={block['n_anchors']}) ==")
        for cond in ("static", "gap1", "gap2", "gap4", "gap8", "gap16",
                     "same_ep_rand", "cross_ep_same_task", "cross_ep_diff_task"):
            if cond not in block:
                continue
            c = block[cond]
            print(f"  {cond:>18} move={c['pred_move_l2']:>7} flip={c['grip_flip_rate']:>6} "
                  f"dIn={c['input_change_l2']:>8} R2(a_t)={c['r2_vs_a_t']:>8} "
                  f"R2(arr)={c['r2_vs_arrival']:>8} R2(mid)={c['r2_vs_mid']:>8} "
                  f"gmean={c['pred_mean_grip']:>6}")


def same_task_map(same: dict) -> dict:
    """episode -> task index for the unseen_same_task result (per-anchor repeats)."""
    out = {}
    for ep, task in zip(same["episode"], same["task"]):
        out[int(ep)] = int(task)
    return out


if __name__ == "__main__":
    main()
