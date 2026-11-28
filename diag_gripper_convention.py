"""P0.5 audit (read-only): is held-out task 2's gripper failure a data-side
convention artifact rather than a model failure?

Part A (no GPU): per-task action stats from the source parquets -- task name,
gripper mean / frac==0 / frac==1 / continuous fraction, mean |a| per dim group.

Part B (tiny GPU): id seed-42 head on the 9950 val probe pairs (cache pmean);
per held-out task, gripper dim: Pearson corr(pred, true), R2(pred), R2(1-pred)
inversion test, and per-dim corr signs.  Strong negative corr + (1-pred)
far above pred == inverted-convention signature.

usage (WSL, repo root):
  python diag_gripper_convention.py --ckpt cache_v3/ckpt/id/final.pt \
      --out results/gripper_convention_20261128/report.json
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import DATA_ROOT, H, PAST
from modules import IDNet


def read_actions(data_root: Path):
    rows = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index",
                                                  "task_index", "action"])
                      for p in sorted(glob.glob(str(data_root / "data/chunk-*/*.parquet")))],
                     ignore_index=True).sort_values(["episode_index", "frame_index"])
    actions = {int(ep): np.stack([np.asarray(a, np.float32) for a in block.action])
               for ep, block in rows.groupby("episode_index")}
    tasks = rows.groupby("episode_index").task_index.first().astype(int).to_dict()
    return actions, tasks


def part_a(data_root: Path) -> dict:
    tasks_tbl = pd.read_parquet(data_root / "meta/tasks.parquet")
    name_by_ix = {}
    for col in ("task", "task_name", "name"):
        if col in tasks_tbl.columns:
            name_by_ix = {i: str(s) for i, s in enumerate(tasks_tbl[col])}
            break
    actions, ep_task = read_actions(data_root)
    per_task = defaultdict(list)
    for ep, a in actions.items():
        per_task[ep_task[ep]].append(a)
    out = {}
    for t in sorted(per_task):
        a = np.concatenate(per_task[t])
        g = a[:, 6]
        out[str(t)] = {
            "name": name_by_ix.get(t, "?"),
            "n_eps": len(per_task[t]), "n_frames": len(a),
            "grip_mean": round(float(g.mean()), 4),
            "grip_frac0": round(float((g < 0.05).mean()), 4),
            "grip_frac1": round(float((g > 0.95).mean()), 4),
            "grip_frac_cont": round(float(((g >= 0.05) & (g <= 0.95)).mean()), 4),
            "mean_abs_dims": [round(float(x), 4) for x in np.abs(a).mean(0)],
        }
    return out


@torch.no_grad()
def part_b(ckpt: Path, cache: Path, device: str) -> dict:
    parts = defaultdict(list)
    for path in sorted(cache.glob("val_shard_*.npz")):
        with np.load(path) as item:
            for key in ("pmean", "act_win", "task"):
                parts[key].append(item[key])
    z = np.concatenate(parts["pmean"])[:, PAST - H:PAST + 1].astype(np.float32)
    y = np.concatenate(parts["act_win"])[:, PAST - H:PAST].reshape(-1, 7)
    task = np.repeat(np.concatenate(parts["task"]), H)
    z0, z1 = z[:, :-1].reshape(-1, 768), z[:, 1:].reshape(-1, 768)

    model = IDNet().to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    preds = []
    for s in range(0, len(z0), 2048):
        preds.append(model(torch.from_numpy(z0[s:s + 2048]).to(device),
                           torch.from_numpy(z1[s:s + 2048]).to(device)).cpu().numpy())
    pred = np.concatenate(preds)

    def r2(t, p):
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum() + 1e-12
        return float(1 - ss_res / ss_tot)

    out = {}
    for t in np.unique(task):
        m = task == t
        gt, gp = y[m, 6].astype(np.float64), pred[m, 6].astype(np.float64)
        corr = float(np.corrcoef(gt, gp)[0, 1]) if gt.std() > 1e-9 else None
        out[str(int(t))] = {
            "n": int(m.sum()),
            "grip": {"corr": round(corr, 4) if corr is not None else None,
                     "r2_pred": round(r2(gt, gp), 4),
                     "r2_inverted_1mpred": round(r2(gt, 1.0 - gp), 4),
                     "mae_pred": round(float(np.abs(gt - gp).mean()), 4),
                     "mae_inverted": round(float(np.abs(gt - (1.0 - gp)).mean()), 4),
                     "true_mean": round(float(gt.mean()), 4),
                     "true_std": round(float(gt.std()), 4),
                     "pred_mean": round(float(gp.mean()), 4),
                     "pred_std": round(float(gp.std()), 4),
                     "r2_pure_bias_floor": round(-float((gt.mean() - gp.mean()) ** 2)
                                                 / (gt.var() + 1e-12), 4)},
            "corr_dims": [round(float(np.corrcoef(y[m, d].astype(np.float64),
                          pred[m, d].astype(np.float64))[0, 1]), 4)
                          if y[m, d].std() > 1e-9 else None for d in range(7)],
            "r2_dims": [round(r2(y[m, d].astype(np.float64),
                          pred[m, d].astype(np.float64)), 4) for d in range(7)],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("cache_v3/ckpt/id/final.pt"))
    ap.add_argument("--cache", type=Path, default=Path("cache_v3"))
    ap.add_argument("--data-root", type=Path, default=Path(DATA_ROOT))
    ap.add_argument("--out", type=Path, default=Path("results/gripper_convention_20261128/report.json"))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"part_a_source_stats": part_a(a.data_root),
              "part_b_id_head": part_b(a.ckpt, a.cache, device)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for t, m in report["part_b_id_head"].items():
        g = m["grip"]
        print(f"task {t:>3} n={m['n']:>5} grip corr={g['corr']} "
              f"R2={g['r2_pred']:>8.3f} R2(1-p)={g['r2_inverted_1mpred']:>8.3f} "
              f"corr_dims={m['corr_dims']}")


if __name__ == "__main__":
    main()
