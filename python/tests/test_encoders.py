import torch
from torch import nn

from zai_rl.models.encoders import run_gru_segments


def reference_segments(gru, x, h0, starts):
    """The pre-M6.0 implementation: one GRU call per span between restart steps."""
    L = x.shape[1]
    cuts = (torch.nonzero(starts[:, 1:].any(dim=0)).flatten() + 1).tolist()
    bounds = [0, *cuts, L]
    h, outs = h0, []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        h = h * (~starts[:, a]).unsqueeze(1)
        out, hn = gru(x[:, a:b], h.unsqueeze(0).contiguous())
        outs.append(out)
        h = hn[0]
    return torch.cat(outs, dim=1)


def test_packed_segments_match_the_reference_outputs_and_gradients():
    torch.manual_seed(0)
    gru = nn.GRU(7, 5, batch_first=True)
    B, L = 9, 12
    starts = torch.rand(B, L) < 0.15
    starts[0, 0] = True  # restart at t = 0
    starts[1] = False  # no restart at all
    starts[2, 1:4] = True  # consecutive one-step episodes
    starts[3, -1] = True  # restart on the last step
    for case in (starts, torch.zeros(B, L, dtype=torch.bool)):
        x = torch.randn(B, L, 7, requires_grad=True)
        h0 = torch.randn(B, 5, requires_grad=True)
        w = torch.randn(B, L, 5)

        ref = reference_segments(gru, x, h0, case)
        (ref * w).sum().backward()
        ref_grads = [x.grad.clone(), h0.grad.clone()] + [p.grad.clone() for p in gru.parameters()]
        x.grad = h0.grad = None
        gru.zero_grad()

        out = run_gru_segments(gru, x, h0, case)
        (out * w).sum().backward()
        grads = [x.grad, h0.grad] + [p.grad for p in gru.parameters()]
        gru.zero_grad()

        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)
        for g, r in zip(grads, ref_grads, strict=True):
            torch.testing.assert_close(g, r, atol=1e-5, rtol=1e-4)
