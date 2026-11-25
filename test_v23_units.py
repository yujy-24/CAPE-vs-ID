"""v2.3 unit checks (design doc §14): official-style last-valid-token gather,
pad slots replaced by the empty query, state encoding stays unit-norm.

Run inside WSL lawam env:  python -I test_v23_units.py
"""
import torch

from modules import CAPEModel


def main():
    torch.manual_seed(0)
    m = CAPEModel()
    m.eval()
    ctx = torch.randn(2, 196, 768)

    # a) lengths=None must equal lengths=n (backward compat for id/fwd paths)
    a_full = torch.randn(2, 9, 7)
    with torch.no_grad():
        u_none, _ = m.dec(m.ctx(ctx), a_full, None)
        u_len, _ = m.dec(m.ctx(ctx), a_full, torch.full((2,), 9, dtype=torch.long))
    print("a) none == full-length:", torch.allclose(u_none, u_len, atol=1e-6))

    # b) padded batch w/ lengths == unpadded single-sample forward
    #    (official valid_len-1 gather: pads must not alter the last valid token)
    with torch.no_grad():
        u_short, _ = m.dec(m.ctx(ctx[0:1]), a_full[0:1, :4], torch.tensor([4]))
        padded = torch.zeros(2, 9, 7)
        padded[0, :4] = a_full[0, :4]
        padded[1] = a_full[1]
        u_pad, _ = m.dec(m.ctx(ctx), padded, torch.tensor([4, 9]))
    print("b) gather picks last VALID slot:", torch.allclose(u_short[0], u_pad[0], atol=1e-5))

    # c) state encoding unit norm + distinct from a 20-step branch output
    with torch.no_grad():
        s = m.encode_state(ctx)
        u = m(ctx, torch.randn(2, 20, 7), torch.full((2,), 20))
    cos = torch.nn.functional.cosine_similarity(s, u, -1).mean()
    print(f"c) state unit-norm: {float(s.norm(dim=-1).mean()):.4f} | cos(s,u): {float(cos):.4f}")

    # d) pad_branches round trip
    from windows import pad_branches
    import numpy as np
    acts, lens = pad_branches([np.ones((4, 7), np.float32), np.zeros((0, 7), np.float32)])
    assert acts.shape == (2, 4, 7) and lens.tolist() == [4, 0]
    assert acts[1].sum() == 0
    print("d) pad_branches:", acts.shape, lens.tolist())


if __name__ == "__main__":
    main()
