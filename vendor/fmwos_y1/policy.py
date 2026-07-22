# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/policy.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""Learned dispatch policy: masked-softmax candidate scorer + value head (P3 §4).

Scorer: an MLP [F_job + F_ctx -> 64 -> 64 -> 1] applied to every candidate (the
context is broadcast onto each candidate), producing one score per candidate;
a masked softmax over the (padded) candidate axis gives the pick distribution.

Value head: the scorer's penultimate 64-d activation, mean-pooled over the valid
candidates, concatenated with the context, then [64+F_ctx -> 64 -> 1].

The module is device-agnostic and batches naturally because K (candidate slots)
is fixed: observations stack to cand [B, K, F_job], mask [B, K], ctx [B, F_ctx].
``act`` is used during rollout/eval; ``evaluate`` re-scores stored (obs, action)
pairs for the PPO update; ``save``/``load`` persist weights + config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import F_CTX, F_JOB, K_CAND

_NEG_INF = -1e9   # masked-logit fill (finite, avoids NaNs in softmax/entropy)


class DispatchPolicy(nn.Module):
    def __init__(self, f_job: int = F_JOB, f_ctx: int = F_CTX,
                 hidden: int = 64, k_cand: int = K_CAND):
        super().__init__()
        self.f_job = f_job
        self.f_ctx = f_ctx
        self.hidden = hidden
        self.k_cand = k_cand

        # Per-candidate scorer.
        self.enc1 = nn.Linear(f_job + f_ctx, hidden)
        self.enc2 = nn.Linear(hidden, hidden)          # penultimate embedding
        self.score = nn.Linear(hidden, 1)
        # Value head over (mean-pooled embedding, ctx).
        self.val1 = nn.Linear(hidden + f_ctx, hidden)
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
        x = torch.cat([cand, ctx_b], dim=-1)
        h = F.relu(self.enc1(x))
        emb = F.relu(self.enc2(h))                     # [B, K, hidden]
        logits = self.score(emb).squeeze(-1)           # [B, K]

        m = mask.to(logits.dtype)
        logits = torch.where(mask, logits, torch.full_like(logits, _NEG_INF))

        # Mean-pool the embedding over valid candidates for the value head.
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (emb * m.unsqueeze(-1)).sum(dim=1) / denom   # [B, hidden]
        vh = F.relu(self.val1(torch.cat([pooled, ctx], dim=-1)))
        value = self.val2(vh).squeeze(-1)              # [B]
        return logits, value

    # ------------------------------------------------------------------ #
    @staticmethod
    def _masked_dist(logits, mask):
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp()
        # Zero out padded contributions defensively (their prob is already ~0).
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
                    "config": {"f_job": self.f_job, "f_ctx": self.f_ctx,
                               "hidden": self.hidden, "k_cand": self.k_cand}}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]
        model = cls(f_job=cfg["f_job"], f_ctx=cfg["f_ctx"],
                    hidden=cfg["hidden"], k_cand=cfg["k_cand"])
        model.load_state_dict(ckpt["state_dict"])
        return model
