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

from common import (BS, CKPT_DIR, GROUPS, HV23, ID_ACCUM, IDX_T, LR, PAST,
                    PER_GROUP, SEED, STEPS, TAU, WD, append_jsonl,
                    cosine_schedule, n_params, seed_all)
from losses import fwd_cosine_loss, id_loss
from modules import CAPEModel, FwdModel, IDNet
from windows import WindowDataset, id_pairs_from_window, pad_actions, pad_branches


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


def cape_loss(model, ds, groups, dev, tau, rng):
    """v2.3 (design doc §14): line-by-line port of the OFFICIAL
    compute_contrastive_loss (single-GPU global mode, use_group_local_state_nce
    =False, use_action_swap=True) on our grouped batches.

    Pairing (official dataset.py:404-422): each window draws ONE pair --
    query = branch k ~ U{1..HV23} (first offset is always non-anchor);
    key = the anchor ITSELF (zero-horizon sample) with p=0.5, else a
    different non-anchor offset. Both branches converge on the same o_T.

    Negatives (official loss.py:220-236): ALL keys in the batch (same-episode
    group members = hard negatives, cross-episode keys = "random states from
    other trajectories") + derangement-swapped query-action outcomes as extra
    negative columns in BOTH directions. Single positive per row (eq 3), no
    multi-positive logsumexp."""
    idxs = np.array([i for g in groups for i in g])
    B = len(idxs)
    kq = rng.integers(1, HV23 + 1, size=B)                 # query branch 1..5
    use_anchor = rng.random(B) < 0.5                       # key = anchor (50%)
    j = rng.integers(1, HV23, size=B)                      # 1..4 -> k' != k
    kk = np.where(use_anchor, 0, 1 + (kq - 1 + j) % HV23)

    qb = [ds.branch_v23(int(i), int(k)) for i, k in zip(idxs, kq)]
    kb = [ds.branch_v23(int(i), int(k)) for i, k in zip(idxs, kk)]
    ctx_q = torch.from_numpy(np.stack([b[0] for b in qb])).to(dev)
    acts_q, len_q = pad_branches([b[1] for b in qb])
    u_q = model(ctx_q, acts_q.to(dev), len_q.to(dev))

    # key side: k=0 rows are zero-horizon samples -> encode_state (single
    # empty-action query through the same decoder == official's all-padded
    # noop branch); k>=1 rows are regular branch predictions.
    K = torch.empty(B, u_q.shape[-1], device=dev)
    ai, bi = np.flatnonzero(kk == 0), np.flatnonzero(kk > 0)
    if len(ai):
        K[ai] = model.encode_state(
            torch.from_numpy(np.stack([kb[t][0] for t in ai])).to(dev))
    if len(bi):
        ctx_b = torch.from_numpy(np.stack([kb[t][0] for t in bi])).to(dev)
        acts_b, len_b = pad_branches([kb[t][1] for t in bi])
        K[bi] = model(ctx_b, acts_b.to(dev), len_b.to(dev))

    # swapped-action outcomes: derangement over the whole query batch, each
    # row keeps its OWN context but executes another query's action sequence.
    ar = np.arange(B)
    while True:
        perm = rng.permutation(B)
        if not (perm == ar).any():
            break
    u_sw = model(ctx_q, acts_q[perm].to(dev), len_q[perm].to(dev))

    logits_q = u_q @ K.T / tau                        # diag = positives
    logits_q_sw = u_q @ u_sw.T / tau                  # swap negatives (extra cols)
    logits_k = K @ u_q.T / tau
    logits_k_sw = K @ u_sw.T / tau
    labels = torch.arange(B, device=dev)
    loss = 0.5 * (F.cross_entropy(torch.cat([logits_q, logits_q_sw], 1), labels)
                  + F.cross_entropy(torch.cat([logits_k, logits_k_sw], 1), labels))

    with torch.no_grad():
        s_T = model.encode_state(
            torch.from_numpy(ds.branch_patch[idxs, HV23].astype(np.float32)).to(dev))
        diag = dict(swapwins=float((F.cosine_similarity(u_q, u_sw, -1)
                                    > F.cosine_similarity(u_q, K, -1)).float().mean()),
                    goal_cos=float(F.cosine_similarity(u_q, s_T, -1).mean()),
                    state_frac=float(use_anchor.mean()))
    return loss, diag


@torch.no_grad()
def cape_counterfactual(model, ds, dev, n=64):
    """collapse meter: how much do u's move when ACTIONS change (ctx fixed)?
    degenerate signature: both cosines -> 1 (v1's failure mode). v2.3: draws
    k ~ U{1..HV23} branches; foreign actions come from another SAME-TASK
    window at the SAME k (matched length)."""
    rng = np.random.default_rng(0)
    idxs = np.arange(min(n, len(ds)))
    ks = rng.integers(1, HV23 + 1, size=len(idxs))
    tb = [ds.branch_v23(int(i), int(k)) for i, k in zip(idxs, ks)]
    ctx = torch.from_numpy(np.stack([b[0] for b in tb])).to(dev)
    acts_t, len_t = pad_branches([b[1] for b in tb])
    acts_r = torch.randn_like(acts_t) * 0.3
    f_idx = np.empty(len(idxs), dtype=int)   # foreign: same-task other window
    tasks = ds.meta["task"]
    for t, i in enumerate(idxs):
        same = np.flatnonzero(tasks == tasks[i])
        same = same[same != i]
        f_idx[t] = int(same[rng.integers(len(same))]) if len(same) else int(i)
    fb = [ds.branch_v23(int(j), int(k)) for j, k in zip(f_idx, ks)]
    acts_f, len_f = pad_branches([b[1] for b in fb])
    u_t = model(ctx, acts_t.to(dev), len_t.to(dev))
    u_f = model(ctx, acts_f.to(dev), len_f.to(dev))
    u_r = model(ctx, acts_r.to(dev), len_t.to(dev))
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
            loss, _ = cape_loss(model, ds, next(gen), dev, tau, rng)
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
                # v2.3: official warmup = 5% of total steps (train.py:236)
                gp["lr"] = a.lr * cosine_schedule(
                    step, a.steps, warmup=max(1, a.steps // 20))
            loss, diag = cape_loss(model, tr, groups, dev, a.tau, rng)
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
                      f"goal {diag['goal_cos']:.3f} state {diag['state_frac']:.2f} "
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
