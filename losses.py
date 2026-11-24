"""Losses: goal-convergent InfoNCE (CAPE), ID regression (L1/L2), forward-cosine."""
import torch
import torch.nn.functional as F


def info_nce(u1, u2, negs, tau=0.1):
    """u1, u2: [B, D] normalised positives (same outcome, two branches).
    negs: [B, N, D] normalised negatives for each anchor.
    symmetric CE; returns scalar."""
    pos = (u1 * u2).sum(-1, keepdim=True) / tau                    # [B,1]
    neg = torch.bmm(negs, u1.unsqueeze(-1)).squeeze(-1) / tau       # [B,N]
    logits1 = torch.cat([pos, neg], dim=1)
    l1 = F.cross_entropy(logits1, torch.zeros(len(u1), device=u1.device, dtype=torch.long))
    pos2 = pos  # symmetric similarity
    neg2 = torch.bmm(negs, u2.unsqueeze(-1)).squeeze(-1) / tau
    logits2 = torch.cat([pos2, neg2], dim=1)
    l2 = F.cross_entropy(logits2, torch.zeros(len(u2), device=u2.device, dtype=torch.long))
    return 0.5 * (l1 + l2)


def id_loss(a_hat, a, form="l1"):
    return F.l1_loss(a_hat, a) if form == "l1" else F.mse_loss(a_hat, a)


def fwd_cosine_loss(z_pred, z_target):
    return (1 - F.cosine_similarity(z_pred, z_target, dim=-1)).mean()
