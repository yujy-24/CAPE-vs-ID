"""Train one arm (v2 protocol, design doc §11).

cape: same-episode grouped batches (GROUPS eps x PER_GROUP windows, paper App.A).
  positives: two branches (t1/t2, different remaining actions) -> same o_T.
  negatives: in-group empty-action state encodings (same scene, other times)
           + in-group swapped-action outcomes (paper's batch-wise action swap)
           + cross-group state encodings (random other episodes).
  State encoding = SAME ctx/decoder/g pathway with a learnable empty-action
  query (paper App.A) -> predictions and states share one space. tau=0.07.
id: L2 (42 default), within-task shuffle control, eff batch 16 (accum x2).
fwd: (ctx_t1, acts t1->T) predicts terminal CLS z_T, cosine loss
  (registered JEPA-WM-style strawman; 42's one-step FD discussed in §11).

usage:
  python train.py --arm cape
  python train.py --arm id [--form l2] [--shuffle-labels] [--seed 43]
  python train.py --arm fwd
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (BS, CKPT_DIR, GROUPS, ID_ACCUM, IDX_T, LR, PAST, PER_GROUP,
                    SEED, STEPS, TAU, WD, append_jsonl, cosine_schedule,
                    n_params, seed_all)
from losses import fwd_cosine_loss, id_loss, info_nce
from modules import CAPEModel, FwdModel, IDNet
from windows import WindowDataset, id_pairs_from_window, pad_actions


def group_batches(ds, rng, groups=GROUPS, per_group=PER_GROUP):
    """yield lists of per-episode window-id lists (each episode >= 2 windows,
    so every anchor always has same-group negatives + a swap partner)."""
    eps = [e for e, ix in ds.ep_of.items() if len(ix) >= 2]
    while True:
        sel = rng.choice(len(eps), size=min(groups, len(eps)), replace=False)
        yield [list(np.asarray(ds.ep_of[eps[gi]])[
                    rng.choice(len(ds.ep_of[eps[gi]]),
                               size=min(per_group, len(ds.ep_of[eps[gi]])),
                               replace=False)])
               for gi in sel]


def collate_id(ds, idxs, rng, shuffle_map=None):
    """k consecutive (z_j, z_{j+1}) -> a_j pairs per window (pmean features).
    shuffle_map: window -> donor window of the SAME task (H3 control); features
    stay with window i, the whole H-step action block comes from the donor --
    destroys frame<->action correspondence, the within-task permutation."""
    prev, nxt, act = [], [], []
    for i in idxs:
        src = i if shuffle_map is None else shuffle_map[i]
        p, nx, a = id_pairs_from_window(
            dict(pmean=ds.pmean[i], act_win=ds.act_win[src]), k=4, rng=rng)
        prev.append(p); nxt.append(nx); act.append(a)
    return (torch.cat(prev), torch.cat(nxt), torch.cat(act))


def cape_loss(model, ds, groups, dev, tau):
    """one grouped batch -> (loss, diagnostics). groups = per-episode lists."""
    idxs = np.array([i for g in groups for i in g])
    B = len(idxs)
    local = []                                   # local positions per group
    b = 0
    for g in groups:
        local.append(list(range(b, b + len(g)))); b += len(g)
    ctx1 = torch.from_numpy(ds.ctx_patch[idxs, 0].astype(np.float32)).to(dev)
    ctx2 = torch.from_numpy(ds.ctx_patch[idxs, 1].astype(np.float32)).to(dev)
    acts1 = pad_actions([ds.branch(int(i), 1)[1] for i in idxs])
    acts2 = pad_actions([ds.branch(int(i), 2)[1] for i in idxs])
    u1 = model(ctx1, acts1.to(dev))
    u2 = model(ctx2, acts2.to(dev))
    # state keys for every window (empty-action encoding, shared space)
    keys = model.encode_state(torch.cat([ctx1, ctx2], 0))       # [2B,256]
    k1, k2 = keys[:B], keys[B:]
    # v2.2: goal state = zero-horizon SAMPLE (paper App.A) -> a third branch
    # converging to the same o_T. Eq (3) is PAIRED: each branch pair gets its
    # own symmetric InfoNCE with its own numerator. (v2.1's shared-logsumexp
    # multi-positive saturates -- the u1<->u2 term (cos .87 @ tau .07 = e^12)
    # dominates the numerator, the goal positive gets ~1e-6 of the softmax
    # weight: goal_cos stayed -0.06 for 20k steps and retrieval stayed chance.)
    s_T = model.encode_state(
        torch.from_numpy(ds.goal_patch[idxs].astype(np.float32)).to(dev))
    allkeys = torch.cat([keys, s_T], 0)                          # 3B state-side keys
    negs, swap = [], []
    for lg in local:
        for pos, j in enumerate(lg):
            others = [x for x in lg if x != j]
            negs.append(torch.cat([k1[others], k2[others], s_T[others]]))
            swap.append(lg[(pos + 1) % len(lg)])                 # in-group roll
    u_sw = model(ctx1, acts2[swap].to(dev))                      # swapped-action outcome
    K = max(x.shape[0] for x in negs)                            # full groups: 3(S-1)
    negs = torch.stack([
        x if x.shape[0] == K else
        torch.cat([x, allkeys[torch.randperm(3 * B, device=dev)[:K - x.shape[0]]]])
        for x in negs], 0)                                       # [B,K,256]
    idx_c = torch.stack([torch.randperm(3 * B, device=dev)[:2] for _ in range(B)])
    cross = allkeys[idx_c]                                       # [B,2,256] random
    negs = torch.cat([negs, u_sw.unsqueeze(1), cross], 1)        # [B,K,256]
    negs = F.normalize(negs, dim=-1)
    l12 = info_nce(u1, u2, negs, tau)                            # branch pair, eq (3)
    l1g = info_nce(u1, s_T, negs, tau)                           # zero-horizon pairs
    l2g = info_nce(u2, s_T, negs, tau)
    loss = (l12 + l1g + l2g) / 3.0
    with torch.no_grad():
        diag = dict(swapwins=float((F.cosine_similarity(
            u1, u_sw, -1) > F.cosine_similarity(u1, u2, -1)).float().mean()),
                    goal_cos=float(F.cosine_similarity(u1, s_T, -1).mean()),
                    nce_goal=float((l1g + l2g) / 2))
    return loss, diag


@torch.no_grad()
def cape_counterfactual(model, ds, dev, n=64):
    """collapse meter: how much do u's move when ACTIONS change (ctx fixed)?
    degenerate signature: both cosines -> 1 (v1's failure mode)."""
    rng = np.random.default_rng(0)
    idxs = np.arange(min(n, len(ds)))
    ctx = torch.from_numpy(ds.ctx_patch[idxs, 1].astype(np.float32)).to(dev)
    acts_t = pad_actions([ds.branch(int(i), 2)[1] for i in idxs])
    acts_r = torch.randn_like(acts_t) * 0.3
    f_idx = []                        # foreign actions: another SAME-TASK window
    tasks = ds.meta["task"]
    for i in idxs:
        same = np.flatnonzero(tasks == tasks[i])
        same = same[same != i]
        f_idx.append(int(same[rng.integers(len(same))]) if len(same) else int(i))
    acts_f = pad_actions([ds.branch(int(j), 2)[1] for j in f_idx])
    u_t = model(ctx, acts_t.to(dev))
    u_f = model(ctx, acts_f.to(dev))
    u_r = model(ctx, acts_r.to(dev))
    return dict(cos_foreign=F.cosine_similarity(u_t, u_f, -1).mean().item(),
                cos_random=F.cosine_similarity(u_t, u_r, -1).mean().item())


@torch.no_grad()
def validate(model, ds, arm, dev, form, bs, tau=TAU):
    model.eval()
    tot, n = 0.0, 0
    rng = np.random.default_rng(0)
    if arm == "id":
        order = np.arange(len(ds))
        for s in range(0, len(ds), bs):
            p, nx, a = collate_id(ds, order[s:s + bs], rng)
            loss = id_loss(model(p.to(dev), nx.to(dev)), a.to(dev), form)
            tot += loss.item() * len(p); n += len(p)
    elif arm == "fwd":
        for s in range(0, len(ds), bs):
            idxs = np.arange(s, min(s + bs, len(ds)))
            ctx = torch.from_numpy(ds.ctx_patch[idxs, 0].astype(np.float32)).to(dev)
            acts = pad_actions([ds.branch(int(i), 1)[1] for i in idxs])
            z = model(ctx, acts.to(dev))
            tgt = torch.from_numpy(ds.pooled[idxs, IDX_T].astype(np.float32)).to(dev)
            loss = fwd_cosine_loss(z, tgt)
            tot += loss.item() * len(idxs); n += len(idxs)
    else:
        gen = group_batches(ds, rng, groups=4)
        for _ in range(16):
            loss, _ = cape_loss(model, ds, next(gen), dev, tau)
            tot += loss.item(); n += 1
    model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["cape", "id", "fwd"], required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--wd", type=float, default=WD)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--form", choices=["l1", "l2"], default="l2")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="H3 control: within-task action permutation (42 App.M)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va = WindowDataset("train"), WindowDataset("val")
    model = (IDNet() if a.arm == "id" else
             FwdModel() if a.arm == "fwd" else CAPEModel()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)

    name = f"{a.arm}{a.tag}"
    out_dir = os.path.join(CKPT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    append_jsonl(os.path.join(out_dir, "config.jsonl"), vars(a))
    print(f"[{name}] trainable {n_params(model)/1e6:.1f}M | "
          f"train {len(tr)} / val {len(va)} windows", flush=True)

    # id within-task shuffle map (window -> donor window of the SAME task)
    shuffle_map = None
    if a.arm == "id" and a.shuffle_labels:
        rng_s = np.random.default_rng(a.seed + 1000)
        shuffle_map = {}
        for task in np.unique(tr.meta["task"]):
            ix = np.flatnonzero(tr.meta["task"] == task)
            perm = rng_s.permutation(ix)
            for src, dst in zip(ix, perm):
                shuffle_map[int(src)] = int(dst)

    rng = np.random.default_rng(a.seed)          # matched data exposure across arms
    log = os.path.join(out_dir, "metrics.jsonl")
    step, t0 = 0, time.time()
    accum = ID_ACCUM if a.arm == "id" else 1

    if a.arm == "cape":
        batches = group_batches(tr, rng)
        while step < a.steps:
            groups = next(batches)
            for gp in opt.param_groups:
                gp["lr"] = a.lr * cosine_schedule(step, a.steps)
            loss, diag = cape_loss(model, tr, groups, dev, a.tau)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                append_jsonl(log, dict(step=step, loss=round(float(loss.item()), 5)))
            if step and step % 2000 == 0:
                vl = validate(model, va, a.arm, dev, a.form, a.bs, a.tau)
                cf = cape_counterfactual(model, va, dev)
                append_jsonl(log, dict(step=step, val_loss=round(vl, 5),
                                       swapwins=diag["swapwins"],
                                       goal_cos=diag["goal_cos"], **cf))
                print(f"[{name}] step {step} loss {loss.item():.4f} val {vl:.4f} "
                      f"cf {cf['cos_foreign']:.3f}/{cf['cos_random']:.3f} "
                      f"goal {diag['goal_cos']:.3f} nceg {diag['nce_goal']:.3f} "
                      f"swapwins {diag['swapwins']:.3f}",
                      flush=True)
            step += 1
    else:
        while step < a.steps:
            opt.zero_grad()
            for _ in range(accum):
                idxs = rng.permutation(len(tr))[: a.bs]
                if a.arm == "id":
                    p, nx, act = collate_id(tr, idxs, rng, shuffle_map)
                    loss = id_loss(model(p.to(dev), nx.to(dev)), act.to(dev), a.form)
                else:
                    ctx = torch.from_numpy(tr.ctx_patch[idxs, 0].astype(np.float32)).to(dev)
                    acts = pad_actions([tr.branch(int(i), 1)[1] for i in idxs])
                    z = model(ctx, acts.to(dev))
                    tgt = torch.from_numpy(tr.pooled[idxs, IDX_T].astype(np.float32)).to(dev)
                    loss = fwd_cosine_loss(z, tgt) / accum
                loss.backward()
            for gp in opt.param_groups:
                gp["lr"] = a.lr * cosine_schedule(step, a.steps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                append_jsonl(log, dict(step=step, loss=round(float(loss.item()), 5)))
            if step and step % 2000 == 0:
                vl = validate(model, va, a.arm, dev, a.form, a.bs)
                append_jsonl(log, dict(step=step, val_loss=round(vl, 5)))
                print(f"[{name}] step {step} loss {loss.item():.4f} val {vl:.4f}", flush=True)
            step += 1

    torch.save(dict(model=model.state_dict(), config=vars(a)),
               os.path.join(out_dir, "final.pt"))
    print(f"[{name}] done {(time.time()-t0)/60:.1f} min -> {out_dir}/final.pt")


if __name__ == "__main__":
    main()
