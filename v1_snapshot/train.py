"""Train one arm. usage:
  python train.py --arm cape --tau 0.1
  python train.py --arm id --form l1
  python train.py --arm fwd
  python train.py --arm cape --no-swap        # H3 ablation
Matched defaults (steps/bs/lr/seed) come from common.py.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (BS, CKPT_DIR, H, LR, SEED, STEPS, append_jsonl,
                    cosine_schedule, n_params, seed_all)
from losses import fwd_cosine_loss, id_loss, info_nce
from modules import CAPEModel, IDNet, StateBridge
from windows import WindowDataset, id_pairs_from_window


def pad_actions(seq_list):
    lengths = torch.tensor([len(a) for a in seq_list])
    out = torch.zeros(len(seq_list), H, 7)
    for i, a in enumerate(seq_list):
        out[i, : len(a)] = a
    return out, lengths


def collate(ds, idxs, arm):
    """arm-specific collate directly from the dataset (no worker pickling issues)."""
    out = dict(i=idxs)
    pooled = torch.from_numpy(ds.pooled[idxs].astype(np.float32))   # [B,15,768]
    out["pooled"] = pooled
    if arm == "id":
        rng = np.random.default_rng()
        prev, nxt, act = [], [], []
        for i in idxs:
            p, nx, a = id_pairs_from_window(
                dict(pooled=ds.pooled[i].astype(np.float32),
                     act_win=ds.act_win[i]), k=4, rng=rng)
            prev.append(torch.as_tensor(p)); nxt.append(torch.as_tensor(nx))
            act.append(torch.as_tensor(a))
        out["id"] = (torch.cat(prev), torch.cat(nxt), torch.cat(act))
        return out
    ctx1, ctx2, acts1, acts2 = [], [], [], []
    for i in idxs:
        c1, a1, _ = ds.branch(i, 1); c2, a2, _ = ds.branch(i, 2)
        ctx1.append(torch.from_numpy(c1)); ctx2.append(torch.from_numpy(c2))
        acts1.append(torch.from_numpy(a1)); acts2.append(torch.from_numpy(a2))
    out["ctx1"], out["ctx2"] = torch.stack(ctx1), torch.stack(ctx2)
    out["acts1"], _ = pad_actions(acts1)
    out["acts2"], _ = pad_actions(acts2)
    return out


def cape_forward(model, ctx, acts):
    """-> (u [B,256] projected, z [B,768] pre-projection final token)"""
    c = model.ctx(ctx)
    last, _ = model.dec(c, acts)
    return model.g(last), last


def neg_states(model, bridge, pooled):
    """2 of 3 future temporal neighbours + random frame -> g(bridge(.)) [B,3,256]"""
    fut = random.sample([H + 1, H + 2, H + 3], 2)
    neg = torch.stack([pooled[:, fut[0]], pooled[:, fut[1]], pooled[:, H + 4]], dim=1)
    return model.g(bridge(neg))


@torch.no_grad()
def validate(model, bridge, ds, arm, dev, form, tau, bs):
    model.eval()
    tot, n = 0.0, 0
    order = np.arange(len(ds))
    for s in range(0, len(ds), bs):
        b = collate(ds, order[s:s + bs], arm)
        if arm == "id":
            p, nx, a = b["id"]
            loss = id_loss(model(p.to(dev), nx.to(dev)), a.to(dev), form)
        else:
            u1, z1 = cape_forward(model, b["ctx1"].to(dev), b["acts1"].to(dev))
            if arm == "fwd":
                loss = fwd_cosine_loss(z1, b["pooled"][:, H].to(dev))
            else:
                u2, _ = cape_forward(model, b["ctx2"].to(dev), b["acts2"].to(dev))
                negs = neg_states(model, bridge, b["pooled"].to(dev))
                loss = info_nce(u1, u2, negs, tau)
        tot += loss.item() * len(b["i"]); n += len(b["i"])
    model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["cape", "id", "fwd"], required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--form", choices=["l1", "l2"], default="l1")
    ap.add_argument("--no-swap", action="store_true")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="H3 control: misalign id action labels (roll within batch)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va = WindowDataset("train"), WindowDataset("val")
    model = (IDNet() if a.arm == "id" else CAPEModel()).to(dev)
    bridge = None if a.arm == "id" else StateBridge().to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) +
                            (list(bridge.parameters()) if bridge else []),
                            lr=a.lr, weight_decay=0.05)

    name = f"{a.arm}{a.tag}"
    out_dir = os.path.join(CKPT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    append_jsonl(os.path.join(out_dir, "config.jsonl"), vars(a))
    extra = f"+bridge {n_params(bridge)/1e6:.1f}M" if bridge else ""
    print(f"[{name}] trainable {n_params(model)/1e6:.1f}M {extra} | "
          f"train {len(tr)} / val {len(va)} windows")

    rng = np.random.default_rng(a.seed)          # matched data exposure across arms
    log = os.path.join(out_dir, "metrics.jsonl")
    step, t0 = 0, time.time()
    while step < a.steps:
        perm = rng.permutation(len(tr))
        for s in range(0, len(perm), a.bs):
            if step >= a.steps:
                break
            idxs = perm[s:s + a.bs]
            b = collate(tr, idxs, a.arm)
            for gp in opt.param_groups:
                gp["lr"] = a.lr * cosine_schedule(step, a.steps)
            if a.arm == "id":
                p, nx, act = b["id"]
                if a.shuffle_labels:
                    act = act.roll(1, dims=0)      # H3: misaligned labels
                loss = id_loss(model(p.to(dev), nx.to(dev)), act.to(dev), a.form)
            else:
                u1, z1 = cape_forward(model, b["ctx1"].to(dev), b["acts1"].to(dev))
                if a.arm == "fwd":
                    loss = fwd_cosine_loss(z1, b["pooled"][:, H].to(dev))
                else:
                    u2, _ = cape_forward(model, b["ctx2"].to(dev), b["acts2"].to(dev))
                    neg_list = [neg_states(model, bridge, b["pooled"].to(dev))]
                    if not a.no_swap:
                        a_swap = b["acts2"].roll(1, dims=0).to(dev)
                        us, _ = cape_forward(model, b["ctx1"].to(dev), a_swap)
                        neg_list.append(us.unsqueeze(1))
                    negs = F.normalize(torch.cat(neg_list, dim=1), dim=-1)
                    loss = info_nce(u1, u2, negs, a.tau)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + (list(bridge.parameters()) if bridge else []), 1.0)
            opt.step()
            if step % 100 == 0:
                append_jsonl(log, dict(step=step, loss=round(float(loss.item()), 5)))
            if step and step % 2000 == 0:
                vl = validate(model, bridge, va, a.arm, dev, a.form, a.tau, a.bs)
                append_jsonl(log, dict(step=step, val_loss=round(vl, 5)))
                print(f"[{name}] step {step} loss {loss.item():.4f} val {vl:.4f}", flush=True)
            step += 1
    torch.save(dict(model=model.state_dict(),
                    bridge=None if bridge is None else bridge.state_dict(),
                    config=vars(a)),
               os.path.join(out_dir, "final.pt"))
    print(f"[{name}] done {(time.time()-t0)/60:.1f} min -> {out_dir}/final.pt")


if __name__ == "__main__":
    main()
