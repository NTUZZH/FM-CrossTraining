"""Micro-batching gradient equivalence (guards the attn OOM fix).

train2.py's Y2_MICROBATCH gradient accumulation must be mathematically
identical (up to float summation order) to the locked minibatch=1024 PPO
update, since the attention verdict-pool checkpoints are trained with it.
This test fails loudly if the accumulation ever drifts from the single-step
update.

Run: PYTHONPATH=.:vendor python tests/test_microbatch_equiv.py
"""
import numpy as np
import torch

from methods.policy2 import make_policy

CLIP, VAL_COEF, ENT_COEF = 0.2, 0.5, 0.01
N = 1024


def _grads(pol_state, arch, micro, batch):
    p = make_policy(arch)
    p.load_state_dict(pol_state)
    p.train()
    for prm in p.parameters():
        prm.grad = None
    pairs, mask, ctx, act, logp_old, adv, ret = batch
    sl = np.arange(N)
    n_sl = len(sl)
    for cs in range(0, n_sl, micro):
        cl = sl[cs:cs + micro]
        st = torch.as_tensor(cl)
        lp, ent, val = p.evaluate(pairs[st], mask[st], ctx[st], act[st])
        ratio = torch.exp(lp - logp_old[st])
        a = adv[st]
        pg = torch.max(-a * ratio,
                       -a * torch.clamp(ratio, 1 - CLIP, 1 + CLIP)).mean()
        vl = ((val - ret[st]) ** 2).mean()
        loss = pg + VAL_COEF * vl - ENT_COEF * ent.mean()
        (loss * (float(len(cl)) / n_sl)).backward()
    return torch.cat([prm.grad.flatten() for prm in p.parameters()
                      if prm.grad is not None])


def _batch(arch):
    pol = make_policy(arch)
    g = torch.Generator().manual_seed(1)
    fp = pol.f_pair
    fc = pol.f_ctx
    pairs = torch.randn(N, 40, fp, generator=g)
    mask = torch.ones(N, 40, dtype=torch.bool)
    mask[:, 20:] = torch.rand(N, 20, generator=g) > 0.5
    ctx = torch.randn(N, fc, generator=g)
    act = torch.randint(0, 20, (N,), generator=g)
    logp_old = torch.randn(N, generator=g)
    adv = torch.randn(N, generator=g)
    ret = torch.randn(N, generator=g)
    return pol.state_dict(), (pairs, mask, ctx, act, logp_old, adv, ret)


def test_equivalence():
    for arch in ("attn", "mlp"):
        torch.manual_seed(0)
        state, batch = _batch(arch)
        g_full = _grads(state, arch, N, batch)      # single chunk
        g_micro = _grads(state, arch, 256, batch)   # 4 accumulated chunks
        rel = (g_full - g_micro).abs().max().item() \
            / (g_full.abs().max().item() + 1e-12)
        assert rel < 1e-5, "%s micro-batching drifted: rel=%.3e" % (arch, rel)
        print("  %s: micro=256 vs full=1024 grad rel-diff %.2e" % (arch, rel))


if __name__ == "__main__":
    test_equivalence()
    print("PASS test_microbatch_equiv (attn + mlp)")
