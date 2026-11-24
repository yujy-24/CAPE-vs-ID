"""Trainable modules. Capacity-matched across arms (print_param_table tells)."""
import math

import torch
import torch.nn as nn

from common import DIM


class ContextEncoder(nn.Module):
    """3-layer self-attn over frozen patch tokens (CAPE's enc_ctx)."""

    def __init__(self, layers=2, dim=DIM, heads=8, ff=1536):
        super().__init__()
        self.inp = nn.Linear(DIM, dim)
        self.pos = nn.Parameter(torch.zeros(1, 196, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, ff, batch_first=True,
                                            norm_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, layers)

    def forward(self, patch):  # [B, 196, 768]
        return self.tr(self.inp(patch) + self.pos)


class DecBlock(nn.Module):
    """one decoder layer: causal self-attn over queries + cross-attn to context."""

    def __init__(self, dim, heads, ff):
        super().__init__()
        self.sa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ca = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, q, ctx, mask):
        h = self.ln1(q + self.sa(q, q, q, attn_mask=mask, need_weights=False)[0])
        h = self.ln2(h + self.ca(h, ctx, ctx, need_weights=False)[0])
        return self.ln3(h + self.ff(h))


class ActionQueryDecoder(nn.Module):
    """Causal self-attn over action queries + cross-attn to context tokens.

    v2 (paper App.A): pos table sized H_MAX_ACT (branches have variable
    remaining lengths 5..24); plus a learnable EMPTY-action query used for
    action-free state encoding through the very same pathway."""

    def __init__(self, layers=2, dim=DIM, heads=8, ff=1536, d_act=7):
        super().__init__()
        from common import H_MAX_ACT
        self.act_proj = nn.Linear(d_act, dim)
        self.pos = nn.Parameter(torch.zeros(1, H_MAX_ACT, dim))
        self.empty = nn.Parameter(torch.zeros(1, 1, dim))  # empty-action query
        nn.init.normal_(self.empty, std=0.02)
        self.blocks = nn.ModuleList([DecBlock(dim, heads, ff) for _ in range(layers)])

    def forward(self, ctx, actions, lengths=None):
        """ctx [B, 196, dim]; actions [B, n, 7] padded to n<=H_MAX_ACT.
        -> final-horizon token [B, dim] plus all tokens [B, n, dim]"""
        q = self.act_proj(actions) + self.pos[:, :actions.shape[1]]
        n = actions.shape[1]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1).to(ctx.device)
        for blk in self.blocks:
            q = blk(q, ctx, mask)
        return q[:, -1], q

    def forward_state(self, ctx):
        """empty-action state encoding: single learnable query, same blocks.
        ctx [B, 196, dim] -> state token [B, dim] (g_phi applied outside)."""
        q = self.empty.expand(ctx.shape[0], -1, -1)
        m = torch.zeros(1, 1, dtype=torch.bool, device=ctx.device)
        for blk in self.blocks:
            q = blk(q, ctx, m)
        return q[:, -1]


class ProjHead(nn.Module):
    """g_phi: d -> d_proj, L2-normalised output (CAPE's projected outcome space)."""

    def __init__(self, dim=DIM, d_proj=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, d_proj))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


class StateBridge(nn.Module):
    """pooled CLS -> trunk dim, so states and predictions share g_phi."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIM, dim), nn.GELU())

    def forward(self, cls):  # [.., 768]
        return self.net(cls)


class CAPEModel(nn.Module):
    def __init__(self, ctx_layers=2, dec_layers=2, dim=DIM):
        super().__init__()
        self.ctx = ContextEncoder(ctx_layers, dim)
        self.dec = ActionQueryDecoder(dec_layers, dim)
        self.g = ProjHead(dim)

    def forward(self, patch, actions, lengths=None):
        """patch [B, 196, 768]; actions [B, n, 7] -> u [B, 256] (normalised)"""
        c = self.ctx(patch)
        last, _ = self.dec(c, actions, lengths)
        return self.g(last)

    def encode_state(self, patch):
        """v2: action-free state token via empty-action query (paper App.A) --
        SAME ctx/decoder/g pathway as predictions => one shared space."""
        c = self.ctx(patch)
        return self.g(self.dec.forward_state(c))


class FwdModel(nn.Module):
    """forward strawman: (ctx_t1, actions t1->T) -> predict terminal CLS z_T.
    Same ctx/decoder trunk as CAPEModel; output head to 768 (no g projection),
    so params stay capacity-matched (ProjHead dim*dim vs Linear dim*768)."""

    def __init__(self, ctx_layers=2, dec_layers=2, dim=DIM):
        super().__init__()
        self.ctx = ContextEncoder(ctx_layers, dim)
        self.dec = ActionQueryDecoder(dec_layers, dim)
        self.out = nn.Linear(dim, DIM)

    def forward(self, patch, actions, lengths=None):
        last, _ = self.dec(self.ctx(patch), actions, lengths)
        return self.out(last)                                    # [B, 768]

    def encode_state(self, patch):
        """action-free readout through the same trunk (empty query + out head):
        'z with no actions yet' in the CLS space it was trained to predict."""
        return self.out(self.dec.forward_state(self.ctx(patch)))


class IDNet(nn.Module):
    """42-style inverse-dynamics head: (z_k, z_{k+1}) -> a_k, transformer over 2 tokens."""

    def __init__(self, layers=4, dim=DIM, heads=8, ff=1536, d_act=7):
        super().__init__()
        self.inp = nn.Linear(DIM, dim)
        self.pos = nn.Parameter(torch.zeros(1, 2, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, ff, batch_first=True,
                                            norm_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, d_act))

    def forward(self, z_prev, z_next):  # [B, 768] each
        x = self.inp(torch.stack([z_prev, z_next], dim=1)) + self.pos
        return self.head(self.tr(x)[:, -1])  # [B, 7]


def print_param_table():
    rows = [("CAPE(ctx2+dec2+g)", CAPEModel()),
            ("IDNet(4L)+head", IDNet())]
    for name, m in rows:
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name}: {n/1e6:.1f}M")
    return {name: sum(p.numel() for p in m.parameters()) for name, m in rows}


if __name__ == "__main__":
    print_param_table()
    m = CAPEModel()
    u = m(torch.randn(2, 196, 768), torch.randn(2, 7, 7))
    print("cape out", u.shape)
    s = m.encode_state(torch.randn(2, 196, 768))
    print("state enc", s.shape, "| unit-norm:", float(s.norm(dim=-1).mean()))
    f = FwdModel()
    print("fwd out", f(torch.randn(2, 196, 768), torch.randn(2, 7, 7)).shape)
    i = IDNet()
    print("id out", i(torch.randn(2, 768), torch.randn(2, 768)).shape)
