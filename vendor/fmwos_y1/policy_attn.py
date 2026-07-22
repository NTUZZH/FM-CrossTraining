# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/policy_attn.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Attention dispatch policy: a stronger candidate scorer (P3 §4 Appendix-B upgrade).

Same I/O contract as ``fmwos.policy.DispatchPolicy`` (``forward``/``act``/
``evaluate``/``save``/``load``, masked softmax over candidates, value head) so the
PPO loop in ``fmwos.train`` needs no change -- only ``--arch attn`` swaps the
class.  The difference is the scorer body:

* per-candidate embedding = Linear(F_job + F_ctx -> d), ctx broadcast onto every
  candidate (identical input fusion to DispatchPolicy);
* 2 pre-LN multi-head self-attention blocks over the candidate SET (4 heads,
  d=64, residual + LayerNorm, position-wise FFN); the blocks are mask-aware --
  padded candidates neither attend nor are attended (key-padding mask), so a real
  candidate's representation is invariant to the number of padding slots and the
  whole scorer is permutation-equivariant over candidates;
* score head Linear(d -> 1);
* value head = masked mean-pool of the post-attention embeddings over the valid
  candidates -> Linear(d -> d) -> Linear(d -> 1).

Weight init follows DispatchPolicy conventions (PyTorch defaults for Linear /
LayerNorm / MultiheadAttention; no custom init).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import F_CTX, F_JOB, K_CAND

_NEG_INF = -1e9   # masked-logit fill (finite, avoids NaNs in softmax/entropy)


class _PreLNBlock(nn.Module):
    """Pre-LN transformer encoder block: MHSA sublayer + position-wise FFN.

    Both sublayers are residual; LayerNorm is applied to the sublayer INPUT
    (pre-norm), which trains stably for shallow stacks.  ``key_padding_mask``
    (True = ignore) keeps padded candidates out of every attention sum.
    """

    def __init__(self, d: int, heads: int, ffn_mult: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.ReLU(),
            nn.Linear(ffn_mult * d, d),
        )

    def forward(self, x, key_padding_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        h = self.ln2(x)
        x = x + self.ffn(h)
        return x


class AttnDispatchPolicy(nn.Module):
    def __init__(self, f_job: int = F_JOB, f_ctx: int = F_CTX,
                 hidden: int = 64, k_cand: int = K_CAND,
                 heads: int = 4, n_blocks: int = 2):
        super().__init__()
        self.f_job = f_job
        self.f_ctx = f_ctx
        self.hidden = hidden
        self.k_cand = k_cand
        self.heads = heads
        self.n_blocks = n_blocks

        # Per-candidate embedding (ctx broadcast, same fusion as DispatchPolicy).
        self.embed = nn.Linear(f_job + f_ctx, hidden)
        # Self-attention stack over the candidate set.
        self.blocks = nn.ModuleList(
            [_PreLNBlock(hidden, heads) for _ in range(n_blocks)])
        self.post_ln = nn.LayerNorm(hidden)
        # Score head.
        self.score = nn.Linear(hidden, 1)
        # Value head over the masked mean-pooled embedding.
        self.val1 = nn.Linear(hidden, hidden)
        self.val2 = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------ #
    def forward(self, cand, mask, ctx):
        """Return (logits [B,K], value [B]).

        cand : float32 [B, K, F_job]
        mask : bool    [B, K]   (True = real candidate)
        ctx  : float32 [B, F_ctx]
        """
        b, k, _ = cand.shape
        ctx_b = ctx.unsqueeze(1).expand(b, k, self.f_ctx)
        x = self.embed(torch.cat([cand, ctx_b], dim=-1))   # [B, K, hidden]

        # key_padding_mask: True = IGNORE. Guard the (pathological) all-masked
        # row so MultiheadAttention never averages over an empty key set (NaN):
        # such a row attends to everything but is masked out of logits + pool.
        kp = ~mask                                          # [B, K]
        all_masked = kp.all(dim=1, keepdim=True)            # [B, 1]
        kp = kp & ~all_masked
        for blk in self.blocks:
            x = blk(x, kp)
        emb = self.post_ln(x)                               # [B, K, hidden]

        logits = self.score(emb).squeeze(-1)                # [B, K]
        logits = torch.where(mask, logits, torch.full_like(logits, _NEG_INF))

        # Masked mean-pool over valid candidates for the value head.
        m = mask.to(emb.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (emb * m.unsqueeze(-1)).sum(dim=1) / denom  # [B, hidden]
        vh = F.relu(self.val1(pooled))
        value = self.val2(vh).squeeze(-1)                   # [B]
        return logits, value

    # ------------------------------------------------------------------ #
    @staticmethod
    def _masked_dist(logits, mask):
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp()
        probs = probs * mask.to(probs.dtype)
        entropy = -(probs * torch.where(mask, logp,
                                        torch.zeros_like(logp))).sum(dim=-1)
        return logp, probs, entropy

    @torch.no_grad()
    def act(self, obs, greedy: bool = False, device=None):
        """Pick an action for a single observation dict.

        Returns (action:int, logprob:float, value:float, entropy:float).
        """
        device = device or next(self.parameters()).device
        cand = torch.as_tensor(obs["cand"], dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device).unsqueeze(0)
        ctx = torch.as_tensor(obs["ctx"], dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(cand, mask, ctx)
        logp, probs, entropy = self._masked_dist(logits, mask)
        if greedy:
            action = torch.argmax(logits, dim=-1)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        a = int(action.item())
        return a, float(logp[0, a].item()), float(value.item()), float(entropy.item())

    def evaluate(self, cand, mask, ctx, actions):
        """Batched re-scoring for PPO.

        Returns (logprobs [B], entropy [B], values [B]) with gradients.
        """
        logits, value = self.forward(cand, mask, ctx)
        logp, _probs, entropy = self._masked_dist(logits, mask)
        logprobs = logp.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        return logprobs, entropy, value

    # ------------------------------------------------------------------ #
    def save(self, path):
        torch.save({"state_dict": self.state_dict(),
                    "arch": "attn",
                    "config": {"f_job": self.f_job, "f_ctx": self.f_ctx,
                               "hidden": self.hidden, "k_cand": self.k_cand,
                               "heads": self.heads, "n_blocks": self.n_blocks}},
                   path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]
        model = cls(f_job=cfg["f_job"], f_ctx=cfg["f_ctx"],
                    hidden=cfg["hidden"], k_cand=cfg["k_cand"],
                    heads=cfg.get("heads", 4), n_blocks=cfg.get("n_blocks", 2))
        model.load_state_dict(ckpt["state_dict"])
        return model
