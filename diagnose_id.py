"""Read-only diagnosis of a trained ID head on clean LIBERO trajectories.

The same-task set contains only episodes absent from the training manifest.
The held-out-task set reproduces the existing 9950-pair probe protocol.
No checkpoint or cache is modified. Lag scores are diagnostic only.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backbone import FrozenDINOv3
from build_cache import stream_frames
from common import BB_DIR, CACHE, DATA_ROOT, H, PAST
from modules import IDNet


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, np.float64)
    pred = np.asarray(pred, np.float64)
    error = pred - y
    sse = np.square(error).sum(axis=0)
    sst = np.square(y - y.mean(axis=0)).sum(axis=0)
    per_dim = np.divide(sse, sst, out=np.full(7, np.nan), where=sst > 1e-12)
    return {
        "n": len(y),
        "r2_pooled": float(1 - sse.sum() / sst.sum()) if sst.sum() > 1e-12 else None,
        "r2_dims": [float(1 - x) if np.isfinite(x) else None for x in per_dim],
        "mae": float(np.abs(error).mean()),
        "mae_dims": np.abs(error).mean(axis=0).tolist(),
        "action_mean": y.mean(axis=0).tolist(),
        "action_std": y.std(axis=0).tolist(),
    }


@torch.no_grad()
def predict(model: IDNet, first: np.ndarray, second: np.ndarray,
            device: str, batch_size: int = 512) -> np.ndarray:
    out = []
    for start in range(0, len(first), batch_size):
        stop = start + batch_size
        x0 = torch.from_numpy(np.asarray(first[start:stop], np.float32)).to(device)
        x1 = torch.from_numpy(np.asarray(second[start:stop], np.float32)).to(device)
        out.append(model(x0, x1).cpu().numpy())
    return np.concatenate(out)


def read_val(cache: Path) -> dict:
    parts = defaultdict(list)
    for path in sorted(cache.glob("val_shard_*.npz")):
        with np.load(path) as item:
            for key in ("pmean", "act_win", "ep", "T", "task"):
                parts[key].append(item[key])
    if not parts:
        raise FileNotFoundError(f"no val shards in {cache}")
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    n = len(arrays["T"])
    z = arrays["pmean"][:, PAST-H:PAST+1].astype(np.float32)
    y = arrays["act_win"][:, PAST-H:PAST].astype(np.float32)
    return {
        "z0": z[:, :-1].reshape(-1, 768),
        "z1": z[:, 1:].reshape(-1, 768),
        "y": y.reshape(-1, 7),
        "previous": arrays["act_win"][:, PAST-H-1:PAST-1].reshape(-1, 7),
        "task": np.repeat(arrays["task"], H),
        "episode": np.repeat(arrays["ep"], H),
        "frame": (arrays["T"][:, None] - H + np.arange(H)).reshape(-1),
        "actions_window": arrays["act_win"],
        "windows": n,
    }


def read_source_tables(data_root: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, int]]:
    episodes = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(
        str(data_root / "meta/episodes/chunk-*/*.parquet")))], ignore_index=True)
    rows = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "task_index", "action"])
        for p in sorted(glob.glob(str(data_root / "data/chunk-*/*.parquet")))], ignore_index=True)
    rows = rows.sort_values(["episode_index", "frame_index"])
    actions = {int(ep): np.stack([np.asarray(a, np.float32) for a in block.action])
               for ep, block in rows.groupby("episode_index")}
    tasks = rows.groupby("episode_index").task_index.first().astype(int).to_dict()
    return episodes, actions, tasks


def unseen_same_task(cache: Path, data_root: Path) -> tuple[dict, dict]:
    manifest = pd.read_parquet(cache / "manifest.parquet")
    used = set(manifest.loc[manifest.split == "train", "episode"].astype(int))
    train_tasks = set(manifest.loc[manifest.split == "train", "task_index"].astype(int))
    episodes, actions, tasks = read_source_tables(data_root)
    selected = [int(ep) for ep in sorted(tasks) if ep not in used and tasks[ep] in train_tasks]
    if not selected:
        raise RuntimeError("no unseen episodes from training tasks")
    rows = episodes.set_index("episode_index")
    wanted_by_file = defaultdict(dict)
    sizes = {}
    for ep in selected:
        row = rows.loc[ep]
        size = int(row["length"])
        if len(actions[ep]) != size:
            raise ValueError(f"episode {ep}: actions {len(actions[ep])} != length {size}")
        offset = int(round(float(row["videos/observation.images.image/from_timestamp"]) * 20))
        path = data_root / "videos/observation.images.image" / (
            f"chunk-{int(row['videos/observation.images.image/chunk_index']):03d}") / (
            f"file-{int(row['videos/observation.images.image/file_index']):03d}.mp4")
        sizes[ep] = size
        for frame in range(size):
            key = offset + frame
            if key in wanted_by_file[path]:
                raise ValueError(f"duplicate video frame {path}:{key}")
            wanted_by_file[path][key] = (ep, frame)

    features = {ep: np.empty((sizes[ep], 768), np.float32) for ep in selected}
    seen = {ep: np.zeros(sizes[ep], bool) for ep in selected}
    backbone = FrozenDINOv3(path=str(BB_DIR), device="cuda" if torch.cuda.is_available() else "cpu")
    for path, mapping in sorted(wanted_by_file.items()):
        rgb_batch, positions = [], []

        def flush() -> None:
            if not rgb_batch:
                return
            _, _, pmean = backbone(torch.from_numpy(np.stack(rgb_batch)))
            for (ep, frame), vec in zip(positions, pmean.numpy()):
                features[ep][frame] = vec.astype(np.float32)
                seen[ep][frame] = True
            rgb_batch.clear()
            positions.clear()

        for file_frame, rgb in stream_frames(str(path), sorted(mapping)):
            rgb_batch.append(rgb)
            positions.append(mapping[file_frame])
            if len(rgb_batch) == 32:
                flush()
        flush()
    for ep in selected:
        if not seen[ep].all():
            raise RuntimeError(f"episode {ep}: {np.count_nonzero(~seen[ep])} frames not decoded")

    z0, z1, y, previous, task, episode, frame = [], [], [], [], [], [], []
    for ep in selected:
        size = sizes[ep]
        # t=1..size-2 permits an observed previous action and a next frame.
        t = np.arange(1, size - 1)
        z0.append(features[ep][t])
        z1.append(features[ep][t+1])
        y.append(actions[ep][t])
        previous.append(actions[ep][t-1])
        task.append(np.full(len(t), tasks[ep], int))
        episode.append(np.full(len(t), ep, int))
        frame.append(t)
    result = {"z0": np.concatenate(z0), "z1": np.concatenate(z1),
              "y": np.concatenate(y), "previous": np.concatenate(previous),
              "task": np.concatenate(task), "episode": np.concatenate(episode),
              "frame": np.concatenate(frame), "actions_by_episode": actions,
              "features_by_episode": features}
    return result, {"unseen_episodes": selected,
                    "covered_train_tasks": sorted(set(tasks[e] for e in selected)),
                    "train_tasks_total": len(train_tasks)}


def task_metrics(data: dict, pred: np.ndarray) -> dict:
    return {str(int(task)): scores(data["y"][mask], pred[mask])
            for task in np.unique(data["task"])
            for mask in [data["task"] == task]}


def lag_metrics(data: dict, pred: np.ndarray, actions_by_episode: dict,
                central_only: bool = False) -> dict:
    # Hold the exact same predictions and frames fixed for every lag.
    if central_only:
        within = np.arange(len(pred)) % H
        valid = (within >= 2) & (within <= H-3)
    else:
        lengths = np.array([len(actions_by_episode[int(e)]) for e in data["episode"]])
        valid = (data["frame"] >= 2) & (data["frame"] + 2 < lengths)
    ix = np.flatnonzero(valid)
    out = {}
    for shift in range(-2, 3):
        y = np.stack([actions_by_episode[int(data["episode"][j])][int(data["frame"][j]) + shift]
                      for j in ix])
        out[str(shift)] = scores(y, pred[ix])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(CACHE))
    parser.add_argument("--data-root", type=Path, default=Path(DATA_ROOT))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ckpt.get("config", {}).get("arm") != "id":
        raise ValueError("checkpoint is not an ID run")
    model = IDNet().to(device).eval()
    model.load_state_dict(ckpt["model"])

    val = read_val(args.cache)
    val_pred = predict(model, val["z0"], val["z1"], device)
    same, coverage = unseen_same_task(args.cache, args.data_root)
    same_pred = predict(model, same["z0"], same["z1"], device)

    manifest = pd.read_parquet(args.cache / "manifest.parquet")
    used_eps = set(manifest.loc[manifest.split == "train", "episode"].astype(int))
    train_actions = [a for ep, a in same["actions_by_episode"].items() if ep in used_eps]
    train_mean = np.concatenate(train_actions).mean(axis=0)
    source = {"same_task_unseen_episodes": same, "heldout_tasks": val}
    predictions = {"same_task_unseen_episodes": same_pred, "heldout_tasks": val_pred}
    report = {"checkpoint": str(args.ckpt), "checkpoint_config": ckpt.get("config"),
              "coverage": coverage, "train_action_mean": train_mean.tolist(), "sets": {}}
    for name, data in source.items():
        pred = predictions[name]
        entry = {"id": scores(data["y"], pred),
                 "train_mean_baseline": scores(data["y"], np.broadcast_to(train_mean, data["y"].shape)),
                 "previous_action_baseline": scores(data["y"], data["previous"]),
                 "by_task": task_metrics(data, pred)}
        if name == "same_task_unseen_episodes":
            entry["lag_diagnostic"] = lag_metrics(data, pred, same["actions_by_episode"])
        else:
            # Val windows contain the full 24-step action history, enabling
            # central-frame lags without accessing raw video or future labels.
            window = data["actions_window"]
            central = np.arange(len(pred)) % H
            keep = (central >= 2) & (central <= H-3)
            ix = np.flatnonzero(keep)
            entry["lag_diagnostic"] = {}
            for shift in range(-2, 3):
                rows = ix // H
                action_slot = PAST-H + central[ix] + shift
                target = window[rows, action_slot]
                entry["lag_diagnostic"][str(shift)] = scores(target, pred[ix])
        report["sets"][name] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"coverage": coverage,
                      "same_task": report["sets"]["same_task_unseen_episodes"]["id"],
                      "heldout": report["sets"]["heldout_tasks"]["id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
