"""Learned pair-scoring dispatch policies v2.

Decision = pair scoring over the capped candidate-pair set of env.engine.
Each pair token is the 38-dim vector [12 order | 6 pair | 20 context]
(context broadcast onto every pair, Y1's fusion idiom). Two architecture
classes, both trained; the verdict class is chosen on development data before
any test evaluation (pre-registered rule, protocol log):

* PairMLP       : token -> MLP 128 x 128 -> scalar score; masked softmax over
                  pairs; value head = mean-pooled penultimate embedding
                  concat ctx -> 128 -> 1. (Width raised from Y1's 64 to 128
                  for the richer input; locked default.)
* PairAttention : token -> width-64 embedding -> 2 pre-LN self-attention
                  blocks (4 heads) -> per-token score; same masked softmax
                  and pooled value structure (Y1's attention scorer,
                  re-targeted at pairs).

Interface identical to Y1's policies (forward/act/evaluate/save/load), so
the PPO loop carries over unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.engine import F_CTX, F_TOTAL, K_PAIRS

_NEG_INF = -1e9


class PairMLP(nn.Module):
    ARCH = "pair_mlp"

    def __init__(self, f_pair: int = F_TOTAL, f_ctx: int = F_CTX,
                 hidden: int = 128, k_pairs: int = K_PAIRS):
        super().__init__()
        self.f_pair = f_pair
        self.f_ctx = f_ctx
        self.hidden = hidden
        self.k_pairs = k_pairs

        self.enc1 = nn.Linear(f_pair, hidden)
        self.enc2 = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1)
        self.val1 = nn.Linear(hidden + f_ctx, hidden)
        self.val2 = nn.Linear(hidden, 1)

    def forward(self, pairs, mask, ctx):
        """pairs [B,K,F], mask [B,K] bool, ctx [B,F_ctx] -> (logits, value)."""
        h = F.relu(self.enc1(pairs))
        emb = F.relu(self.enc2(h))
        logits = self.score(emb).squeeze(-1)
        logits = torch.where(mask, logits, torch.full_like(logits, _NEG_INF))
        m = mask.to(emb.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (emb * m.unsqueeze(-1)).sum(dim=1) / denom
        vh = F.relu(self.val1(torch.cat([pooled, ctx], dim=-1)))
        value = self.val2(vh).squeeze(-1)
        return logits, value

    # ------------------------------------------------------------------ #
    @staticmethod
    def _masked_dist(logits, mask):
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp() * mask.to(logits.dtype)
        entropy = -(probs * torch.where(mask, logp,
                                        torch.zeros_like(logp))).sum(dim=-1)
        return logp, probs, entropy

    @torch.no_grad()
    def act(self, obs, greedy: bool = False, device=None):
        device = device or next(self.parameters()).device
        n = max(1, int(obs.get("n") or obs["mask"].sum()))
        pairs = torch.as_tensor(obs["pairs"][:n], dtype=torch.float32,
                                device=device).unsqueeze(0)
        mask = torch.as_tensor(obs["mask"][:n], dtype=torch.bool,
                               device=device).unsqueeze(0)
        ctx = torch.as_tensor(obs["ctx"], dtype=torch.float32,
                              device=device).unsqueeze(0)
        logits, value = self.forward(pairs, mask, ctx)
        logp, probs, entropy = self._masked_dist(logits, mask)
        if greedy:
            action = torch.argmax(logits, dim=-1)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        a = int(action.item())
        return a, float(logp[0, a].item()), float(value.item()), \
            float(entropy.item())

    def evaluate(self, pairs, mask, ctx, actions):
        logits, value = self.forward(pairs, mask, ctx)
        logp, _probs, entropy = self._masked_dist(logits, mask)
        logprobs = logp.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        return logprobs, entropy, value

    # ------------------------------------------------------------------ #
    def _config(self):
        return {"f_pair": self.f_pair, "f_ctx": self.f_ctx,
                "hidden": self.hidden, "k_pairs": self.k_pairs,
                "arch": self.ARCH}

    def save(self, path):
        torch.save({"state_dict": self.state_dict(),
                    "config": self._config()}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]
        kw = {k: cfg[k] for k in ("f_pair", "f_ctx", "hidden", "k_pairs")}
        model = cls(**kw)
        model.load_state_dict(ckpt["state_dict"])
        return model


class _PreLNBlock(nn.Module):
    """Pre-LN transformer block (Y1 fmwos.policy_attn idiom, verbatim)."""

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


class PairAttention(PairMLP):
    ARCH = "pair_attn"

    def __init__(self, f_pair: int = F_TOTAL, f_ctx: int = F_CTX,
                 hidden: int = 64, k_pairs: int = K_PAIRS,
                 heads: int = 4, n_blocks: int = 2):
        nn.Module.__init__(self)
        self.f_pair = f_pair
        self.f_ctx = f_ctx
        self.hidden = hidden
        self.k_pairs = k_pairs
        self.heads = heads
        self.n_blocks = n_blocks

        self.embed = nn.Linear(f_pair, hidden)
        self.blocks = nn.ModuleList(
            [_PreLNBlock(hidden, heads) for _ in range(n_blocks)])
        self.post_ln = nn.LayerNorm(hidden)
        self.score = nn.Linear(hidden, 1)
        self.val1 = nn.Linear(hidden + f_ctx, hidden)
        self.val2 = nn.Linear(hidden, 1)

    def forward(self, pairs, mask, ctx):
        x = self.embed(pairs)
        # An all-padded row (terminal zero-obs) would NaN under attention;
        # give it one attendable slot (its logits stay masked downstream).
        kpm = ~mask
        all_pad = kpm.all(dim=1)
        if bool(all_pad.any()):
            kpm = kpm.clone()
            kpm[all_pad, 0] = False
        for blk in self.blocks:
            x = blk(x, key_padding_mask=kpm)
        emb = self.post_ln(x)
        logits = self.score(emb).squeeze(-1)
        logits = torch.where(mask, logits, torch.full_like(logits, _NEG_INF))
        m = mask.to(emb.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (emb * m.unsqueeze(-1)).sum(dim=1) / denom
        vh = F.relu(self.val1(torch.cat([pooled, ctx], dim=-1)))
        value = self.val2(vh).squeeze(-1)
        return logits, value

    def _config(self):
        cfg = super()._config()
        cfg.update({"heads": self.heads, "n_blocks": self.n_blocks})
        return cfg

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]
        kw = {k: cfg[k] for k in ("f_pair", "f_ctx", "hidden", "k_pairs",
                                  "heads", "n_blocks")}
        model = cls(**kw)
        model.load_state_dict(ckpt["state_dict"])
        return model


def make_policy(arch: str, f_pair: int | None = None):
    """f_pair overrides the token width (wait-action variant: F_TOTAL + 1,
    the extra dimension being the is-wait flag; all other locked defaults
    unchanged)."""
    if arch in ("mlp", "pair_mlp"):
        return PairMLP(f_pair=f_pair or F_TOTAL)
    if arch in ("attn", "pair_attn"):
        return PairAttention(f_pair=f_pair or F_TOTAL)
    raise ValueError("unknown arch %r" % arch)


def load_policy(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    arch = ckpt["config"].get("arch", "pair_mlp")
    cls = PairAttention if arch == "pair_attn" else PairMLP
    return cls.load(path, map_location=map_location)
