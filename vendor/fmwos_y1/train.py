# VENDORED from Paper Y1 repo (READ-ONLY reference), file src/fmwos/train.py
# Y1 local commit 25ee06200af37cc2337083694d68564162c45ab9 (2026-07-07);
# public mirror github.com/NTUZZH/FM-Scheduling @ 05f958d.
# Copied verbatim below this header; DO NOT EDIT (Y2 work order 0.6.7 / 2.1.5).
"""PPO trainer for the learned dispatch policy (notes/p3_spec.md §4).

In-process vectorized PPO (no multiprocessing -- the python envs are cheap):
16 env copies stepped in lockstep collect ~8192 transitions per update; GAE
(gamma=1.0, lambda=0.98); clipped PPO (clip 0.2, 4 epochs, minibatch 1024,
entropy 0.01, value coef 0.5, grad clip 0.5, per-batch advantage normalization);
Adam lr 3e-4; rewards already scaled 1/100 inside the env.

Instance sampling per §4: 50/50 replay-train vs generator, campuses {5,9,10,12},
sizes 150/400 (generator crew_multiplier in {0.75, 1.0} sampled 50/50, seeds
30000+).  The generator module + params (results/p2_generator/params_c*.json)
are loaded DEFENSIVELY at runtime: if either is missing the sampler falls back
to replay-only and logs a warning (generator is never imported at module top).

Dev evaluation: a fixed 32-instance replay-train set (greedy argmax), every 20
updates; the best-dev-WWT checkpoint is kept.

Curriculum (``--curriculum {v1,v2}``, default v1 = the sampling above, unchanged):
``v2`` is the contention-heavy rebalance (notes/decisions.md "Policy v2 retraining
planned").  Replay half: with prob 2/3 apply ``tightness.scale_crew`` at m in
{0.5,0.6,0.8} (uniform), else leave at m=1.0.  Generator half: crew_multiplier in
{0.5,0.6,0.8,1.0} weighted {0.2,0.25,0.25,0.3} and arrival_multiplier in
{1.0,1.25,1.5} weighted {0.5,0.3,0.2}.  The dev set is unchanged for either
curriculum; a second dev metric ``dev_wwt_tight`` (the same dev instances scaled
to m=0.6) is always reported.

CLI
---
python -m fmwos.train --seed 301 --updates 300 --out results/p3_train/seed301
python -m fmwos.train --curriculum v2 --seed 401 --out results/p3_train/v2_401
python -m fmwos.train --seed 301 --smoke --out /tmp/.../p3smoke   # 3 updates, cpu
Writes config.json, curves.csv (update, mean_train_return, dev_wwt_mean,
dev_wwt_tight, entropy, value_loss, seconds), best.pt, final.pt.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch

from .env import DispatchEnv
from .policy import DispatchPolicy
from . import validator
from . import tightness

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_INST_ROOT = os.path.join(_ROOT, "data", "processed", "instances")
_PARAM_ROOT = os.path.join(_ROOT, "results", "p2_generator")
_TRAIN_ANCHOR_MAX = "2017-12-31"
DEV_TIGHT_M = 0.6   # second dev metric: the dev instances scaled to m=0.6 (contended)


# --------------------------------------------------------------------------- #
# Instance pools                                                              #
# --------------------------------------------------------------------------- #
def _campus_dir(campus):
    return "c%02d" % int(campus)


def list_replay_train_files(campuses, sizes):
    """Sorted replay TRAIN files (window_start <= 2017-12-31) for the cells."""
    files = []
    for c in campuses:
        for s in sizes:
            pat = os.path.join(_INST_ROOT, _campus_dir(c), "replay", str(s), "*.json")
            for f in sorted(glob.glob(pat)):
                try:
                    with open(f) as fh:
                        ws = json.load(fh)["meta"]["window_start"]
                except Exception:
                    continue
                if ws <= _TRAIN_ANCHOR_MAX:
                    files.append(f)
    return files


def _load_instance(path, cache):
    inst = cache.get(path)
    if inst is None:
        with open(path) as fh:
            inst = json.load(fh)
        cache[path] = inst
    return inst


def _curriculum_knobs(curriculum):
    """Knob distributions recorded in config.json (documentation only)."""
    if curriculum == "v2":
        return {
            "replay_scale_prob": 2.0 / 3.0,
            "replay_scale_m_choices": [0.5, 0.6, 0.8],
            "gen_crew_choices": [0.5, 0.6, 0.8, 1.0],
            "gen_crew_weights": [0.2, 0.25, 0.25, 0.3],
            "gen_arrival_choices": [1.0, 1.25, 1.5],
            "gen_arrival_weights": [0.5, 0.3, 0.2],
        }
    # v1: legacy 50/50 replay/generator, generator crew in {0.75, 1.0}, no scaling
    return {
        "replay_scale_prob": 0.0,
        "gen_crew_choices": [0.75, 1.0],
        "gen_crew_weights": [0.5, 0.5],
        "gen_arrival_choices": [1.0],
        "gen_arrival_weights": [1.0],
    }


class InstanceSampler:
    """50/50 replay/generator instance sampler (defensive generator loading).

    ``curriculum='v1'`` (default) is the original behavior.  ``curriculum='v2'``
    is the contention-heavy rebalance (see module docstring / notes/decisions.md):
    replay instances are scaled with ``tightness.scale_crew`` most of the time and
    generator instances draw crew/arrival multipliers from the storm-leaning packs.
    """

    def __init__(self, campuses, sizes, seed, gen_seed_base=30000, curriculum="v1"):
        self.campuses = list(campuses)
        self.sizes = list(sizes)
        self.curriculum = str(curriculum)
        self.rng = random.Random(seed + 777)
        self.replay_files = list_replay_train_files(campuses, sizes)
        self._cache = {}
        self._gen_seed = gen_seed_base
        self.last_spec = None          # knobs of the most recent draw (v2 introspection)
        self.generator, self.params = self._load_generator()
        self.gen_enabled = self.generator is not None and bool(self.params)
        if not self.replay_files and not self.gen_enabled:
            raise RuntimeError("no replay files and no generator available to sample")

    def _load_generator(self):
        try:
            from . import generator          # imported lazily, never at top level
        except Exception as e:               # pragma: no cover
            print("[train] WARNING: generator module unavailable (%s); "
                  "falling back to replay-only." % e, file=sys.stderr)
            return None, {}
        params = {}
        for c in self.campuses:
            p = os.path.join(_PARAM_ROOT, "params_c%d.json" % int(c))
            if os.path.exists(p):
                try:
                    with open(p) as fh:
                        params[int(c)] = json.load(fh)
                except Exception:
                    pass
        if not params:
            print("[train] WARNING: no generator params found under %s; "
                  "falling back to replay-only." % _PARAM_ROOT, file=sys.stderr)
            return None, {}
        return generator, params

    def sample(self):
        if self.curriculum == "v2":
            return self._sample_v2()
        use_gen = self.gen_enabled and (self.rng.random() < 0.5)
        if not use_gen and self.replay_files:
            return _load_instance(self.rng.choice(self.replay_files), self._cache)
        if self.gen_enabled:
            campus = self.rng.choice(sorted(self.params.keys()))
            size = self.rng.choice(self.sizes)
            crew = 0.75 if self.rng.random() < 0.5 else 1.0
            seed = self._gen_seed
            self._gen_seed += 1
            return self.generator.generate(self.params[campus], size, seed,
                                            crew_multiplier=crew)
        # gen disabled and (rare) no replay chosen -> pick replay
        return _load_instance(self.rng.choice(self.replay_files), self._cache)

    # ---- curriculum v2 (contention-heavy) --------------------------------- #
    def _draw_spec_v2(self):
        """Draw one v2 episode spec (knobs only; no instance materialization).

        Replay half: prob 2/3 scale with m ~ U{0.5,0.6,0.8}, else m=1.0.
        Generator half: weighted crew_multiplier and arrival_multiplier draws.
        Cheap enough to sample in bulk for distribution tests.
        """
        use_gen = self.gen_enabled and (self.rng.random() < 0.5)
        if not use_gen and self.replay_files:
            if self.rng.random() < (2.0 / 3.0):
                m = self.rng.choice([0.5, 0.6, 0.8])
            else:
                m = 1.0
            return {"track": "replay", "crew_multiplier": m,
                    "arrival_multiplier": 1.0,
                    "path": self.rng.choice(self.replay_files)}
        if self.gen_enabled:
            crew = self.rng.choices([0.5, 0.6, 0.8, 1.0],
                                    weights=[0.2, 0.25, 0.25, 0.3])[0]
            arr = self.rng.choices([1.0, 1.25, 1.5],
                                   weights=[0.5, 0.3, 0.2])[0]
            campus = self.rng.choice(sorted(self.params.keys()))
            size = self.rng.choice(self.sizes)
            seed = self._gen_seed
            self._gen_seed += 1
            return {"track": "generator", "crew_multiplier": crew,
                    "arrival_multiplier": arr, "campus": campus,
                    "size": size, "seed": seed}
        # gen disabled and (rare) no replay chosen -> full-capacity replay
        return {"track": "replay", "crew_multiplier": 1.0,
                "arrival_multiplier": 1.0,
                "path": self.rng.choice(self.replay_files)}

    def _materialize(self, spec):
        """Turn a v2 spec into an instance dict (scaling replay copies as needed)."""
        if spec["track"] == "replay":
            inst = _load_instance(spec["path"], self._cache)
            m = spec["crew_multiplier"]
            if m != 1.0:
                inst = tightness.scale_crew(inst, m)   # deep copy; cache untouched
            return inst
        return self.generator.generate(
            self.params[spec["campus"]], spec["size"], spec["seed"],
            crew_multiplier=spec["crew_multiplier"],
            arrival_multiplier=spec["arrival_multiplier"])

    def _sample_v2(self):
        spec = self._draw_spec_v2()
        self.last_spec = spec
        return self._materialize(spec)


def load_dev_set(campuses, sizes, n, cache):
    """A fixed, deterministic, cell-stratified dev set of <= n replay-train
    instances (round-robin over (campus, size) cells so it spans the training
    distribution rather than collapsing onto whichever cell sorts first)."""
    per_cell = []
    for c in campuses:
        for s in sizes:
            pat = os.path.join(_INST_ROOT, _campus_dir(c), "replay", str(s), "*.json")
            cell = []
            for f in sorted(glob.glob(pat)):
                try:
                    with open(f) as fh:
                        ws = json.load(fh)["meta"]["window_start"]
                except Exception:
                    continue
                if ws <= _TRAIN_ANCHOR_MAX:
                    cell.append(f)
            if cell:
                per_cell.append(cell)
    picked, i = [], 0
    while len(picked) < n and per_cell:
        made_progress = False
        for cell in per_cell:
            if i < len(cell):
                picked.append(cell[i])
                made_progress = True
                if len(picked) >= n:
                    break
        if not made_progress:
            break
        i += 1
    return [_load_instance(f, cache) for f in picked]


# --------------------------------------------------------------------------- #
# Rollout helpers                                                             #
# --------------------------------------------------------------------------- #
def _stack_obs(obs_list):
    cand = np.stack([o["cand"] for o in obs_list])
    mask = np.stack([o["mask"] for o in obs_list])
    ctx = np.stack([o["ctx"] for o in obs_list])
    return cand, mask, ctx


def _batch_value(policy, obs_list, device):
    cand, mask, ctx = _stack_obs(obs_list)
    ct = torch.as_tensor(cand, dtype=torch.float32, device=device)
    mt = torch.as_tensor(mask, dtype=torch.bool, device=device)
    xt = torch.as_tensor(ctx, dtype=torch.float32, device=device)
    with torch.no_grad():
        _logits, value = policy(ct, mt, xt)
    return value.cpu().numpy()


def eval_dev(policy, dev_instances, device, feature_drop=None):
    """Mean WWT (validator) of the greedy policy over the dev set.

    ``feature_drop`` must match the training env so the policy sees the same
    (ablated) observation distribution at eval; the reward_mode is irrelevant
    here (WWT is scored by the validator, not the reward).
    """
    policy.eval()
    wwts = []
    for inst in dev_instances:
        env = DispatchEnv(inst, feature_drop=feature_drop)
        obs = env.reset()
        done = False
        while not done:
            a, _, _, _ = policy.act(obs, greedy=True, device=device)
            obs, _r, done, _info = env.step(a)
        sched = env.to_schedule("rl")
        wwts.append(validator.validate(inst, sched)["metrics"]["WWT"])
    return float(np.mean(wwts)) if wwts else float("nan")


# --------------------------------------------------------------------------- #
# PPO                                                                         #
# --------------------------------------------------------------------------- #
def _make_policy(arch):
    """Instantiate the scorer for ``arch`` ('mlp' default, 'attn' upgrade).

    Both classes expose the SAME interface (forward/act/evaluate/save/load and
    the k_cand/f_job/f_ctx attributes the PPO loop reads), so nothing else in
    the loop changes."""
    if arch == "attn":
        from .policy_attn import AttnDispatchPolicy
        return AttnDispatchPolicy()
    return DispatchPolicy()


def train(seed, updates, out_dir, smoke=False, device=None,
          reward_mode="shaped", feature_drop=None, curriculum="v1", arch="mlp"):
    if curriculum not in ("v1", "v2"):
        raise ValueError("curriculum must be 'v1' or 'v2', got %r" % (curriculum,))
    if arch not in ("mlp", "attn"):
        raise ValueError("arch must be 'mlp' or 'attn', got %r" % (arch,))
    campuses = [5, 9, 10, 12]
    if smoke:
        n_envs, steps_per_env, sizes = 2, 64, [50]
        n_dev, eval_every, minibatch = 2, 1, 128
        device = device or "cpu"
    else:
        n_envs, steps_per_env, sizes = 16, 512, [150, 400]
        n_dev, eval_every, minibatch = 32, 20, 1024
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    gamma, lam, clip = 1.0, 0.98, 0.2
    epochs, ent_coef, val_coef, max_grad = 4, 0.01, 0.5, 0.5
    lr = 3e-4

    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    sampler = InstanceSampler(campuses, sizes, seed, curriculum=curriculum)
    dev_cache = dict(sampler._cache)
    dev_set = load_dev_set(campuses, [150, 400] if not smoke else [50], n_dev, dev_cache)
    # Second dev metric (both curricula): the same dev instances at m=0.6 capacity.
    dev_set_tight = [tightness.scale_crew(inst, DEV_TIGHT_M) for inst in dev_set]

    policy = _make_policy(arch).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)

    config = {
        "seed": seed, "updates": updates, "smoke": smoke, "device": str(device),
        "n_envs": n_envs, "steps_per_env": steps_per_env, "sizes": sizes,
        "campuses": campuses, "gamma": gamma, "gae_lambda": lam, "clip": clip,
        "epochs": epochs, "minibatch": minibatch, "ent_coef": ent_coef,
        "val_coef": val_coef, "max_grad_norm": max_grad, "lr": lr,
        "reward_mode": reward_mode, "feature_drop": feature_drop,
        "n_dev": len(dev_set), "eval_every": eval_every,
        "generator_enabled": sampler.gen_enabled,
        "n_replay_train_files": len(sampler.replay_files),
        "curriculum": curriculum, "dev_tight_m": DEV_TIGHT_M,
        "curriculum_knobs": _curriculum_knobs(curriculum),
        "arch": arch,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    curves_path = os.path.join(out_dir, "curves.csv")
    with open(curves_path, "w") as fh:
        fh.write("update,mean_train_return,dev_wwt_mean,dev_wwt_tight,"
                 "entropy,value_loss,seconds\n")

    # Env slots.
    envs = [DispatchEnv(sampler.sample(), reward_mode=reward_mode,
                        feature_drop=feature_drop) for _ in range(n_envs)]
    cur_obs = [e.reset() for e in envs]
    ep_return = [0.0 for _ in range(n_envs)]

    # Checkpoint-selection metric: curriculum v2 targets contended regimes, so
    # its best.pt is chosen by the TIGHT dev metric (the default-capacity dev
    # plateaus at ~410 for any non-idling policy and cannot discriminate --
    # see decisions.md 2026-07-05 "v2 checkpoint selection").
    best_dev = float("inf")
    last_dev = eval_dev(policy, dev_set, device, feature_drop)  # baseline: every row finite
    last_dev_tight = eval_dev(policy, dev_set_tight, device, feature_drop)
    sel = last_dev_tight if curriculum == "v2" else last_dev
    if sel < best_dev:
        best_dev = sel
        policy.save(os.path.join(out_dir, "best.pt"))

    L = steps_per_env
    for update in range(updates):
        t0 = time.perf_counter()
        policy.eval()

        # Storage [L, n_envs, ...]
        b_cand = np.zeros((L, n_envs, policy.k_cand, policy.f_job), dtype=np.float32)
        b_mask = np.zeros((L, n_envs, policy.k_cand), dtype=bool)
        b_ctx = np.zeros((L, n_envs, policy.f_ctx), dtype=np.float32)
        b_act = np.zeros((L, n_envs), dtype=np.int64)
        b_logp = np.zeros((L, n_envs), dtype=np.float32)
        b_val = np.zeros((L, n_envs), dtype=np.float32)
        b_rew = np.zeros((L, n_envs), dtype=np.float32)
        b_done = np.zeros((L, n_envs), dtype=np.float32)
        completed_returns = []

        for t in range(L):
            cand, mask, ctx = _stack_obs(cur_obs)
            ct = torch.as_tensor(cand, dtype=torch.float32, device=device)
            mt = torch.as_tensor(mask, dtype=torch.bool, device=device)
            xt = torch.as_tensor(ctx, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = policy(ct, mt, xt)
                logp_all = torch.log_softmax(logits, dim=-1)
                probs = logp_all.exp() * mt.to(logits.dtype)
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
                logp_a = logp_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            acts = actions.cpu().numpy()
            b_cand[t] = cand
            b_mask[t] = mask
            b_ctx[t] = ctx
            b_act[t] = acts
            b_logp[t] = logp_a.cpu().numpy()
            b_val[t] = value.cpu().numpy()

            for i in range(n_envs):
                nobs, r, done, _info = envs[i].step(int(acts[i]))
                b_rew[t, i] = r
                b_done[t, i] = 1.0 if done else 0.0
                ep_return[i] += r
                if done:
                    completed_returns.append(ep_return[i])
                    ep_return[i] = 0.0
                    envs[i] = DispatchEnv(sampler.sample(), reward_mode=reward_mode,
                                          feature_drop=feature_drop)
                    cur_obs[i] = envs[i].reset()
                else:
                    cur_obs[i] = nobs

        # Bootstrap value for the final obs of each env.
        last_val = _batch_value(policy, cur_obs, device)

        # GAE (gamma, lambda).
        adv = np.zeros((L, n_envs), dtype=np.float32)
        lastgae = np.zeros(n_envs, dtype=np.float32)
        for t in reversed(range(L)):
            nonterminal = 1.0 - b_done[t]
            nextval = last_val if t == L - 1 else b_val[t + 1]
            delta = b_rew[t] + gamma * nextval * nonterminal - b_val[t]
            lastgae = delta + gamma * lam * nonterminal * lastgae
            adv[t] = lastgae
        ret = adv + b_val

        # Flatten.
        N = L * n_envs
        f_cand = torch.as_tensor(b_cand.reshape(N, policy.k_cand, policy.f_job), device=device)
        f_mask = torch.as_tensor(b_mask.reshape(N, policy.k_cand), device=device)
        f_ctx = torch.as_tensor(b_ctx.reshape(N, policy.f_ctx), device=device)
        f_act = torch.as_tensor(b_act.reshape(N), device=device)
        f_logp = torch.as_tensor(b_logp.reshape(N), device=device)
        f_adv = torch.as_tensor(adv.reshape(N), device=device)
        f_ret = torch.as_tensor(ret.reshape(N), device=device)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        policy.train()
        mb = min(minibatch, N)
        ent_acc, vloss_acc, n_mb = 0.0, 0.0, 0
        idx = np.arange(N)
        for _ep in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, N, mb):
                sl = idx[start:start + mb]
                st = torch.as_tensor(sl, device=device)
                logp, entropy, value = policy.evaluate(
                    f_cand[st], f_mask[st], f_ctx[st], f_act[st])
                ratio = torch.exp(logp - f_logp[st])
                a = f_adv[st]
                pg1 = -a * ratio
                pg2 = -a * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = ((value - f_ret[st]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss + val_coef * v_loss - ent_coef * ent
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
                optim.step()
                ent_acc += float(ent.item())
                vloss_acc += float(v_loss.item())
                n_mb += 1

        mean_ent = ent_acc / max(1, n_mb)
        mean_vloss = vloss_acc / max(1, n_mb)
        mean_ret = float(np.mean(completed_returns)) if completed_returns else 0.0

        is_last = (update == updates - 1)
        if (update % eval_every == 0) or is_last:
            last_dev = eval_dev(policy, dev_set, device, feature_drop)
            last_dev_tight = eval_dev(policy, dev_set_tight, device, feature_drop)
            sel = last_dev_tight if curriculum == "v2" else last_dev
            if sel < best_dev:
                best_dev = sel
                policy.save(os.path.join(out_dir, "best.pt"))
        secs = time.perf_counter() - t0
        with open(curves_path, "a") as fh:
            fh.write("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f\n"
                     % (update, mean_ret, last_dev, last_dev_tight,
                        mean_ent, mean_vloss, secs))
        sel_now = last_dev_tight if curriculum == "v2" else last_dev
        print("[u%03d] ret=%.4f dev_wwt=%.4f dev_tight=%.4f ent=%.4f vloss=%.4f %.1fs%s"
              % (update, mean_ret, last_dev, last_dev_tight, mean_ent, mean_vloss,
                 secs, "  *best" if sel_now == best_dev else ""))

    policy.save(os.path.join(out_dir, "final.pt"))
    if not os.path.exists(os.path.join(out_dir, "best.pt")):
        policy.save(os.path.join(out_dir, "best.pt"))
    print("[train] done. best dev WWT=%.4f -> %s" % (best_dev, out_dir))
    return {"best_dev_wwt": best_dev, "out_dir": out_dir}


def main(argv=None):
    ap = argparse.ArgumentParser(description="PPO trainer for the FM dispatch policy")
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--reward", type=str, default="shaped",
                    choices=["shaped", "realized", "terminal"],
                    help="reward variant for E5 ablation (default: shaped)")
    ap.add_argument("--feature-drop", type=str, default="none",
                    choices=["none", "urgency", "workload", "context"],
                    help="zero out a feature group for E5 ablation (default: none)")
    ap.add_argument("--curriculum", type=str, default="v1", choices=["v1", "v2"],
                    help="instance-sampling curriculum (v2 = contention-heavy; "
                         "default v1 = original behavior)")
    ap.add_argument("--arch", type=str, default="mlp", choices=["mlp", "attn"],
                    help="scorer architecture (default 'mlp' = DispatchPolicy, "
                         "unchanged; 'attn' = AttnDispatchPolicy, Appendix B)")
    args = ap.parse_args(argv)
    n_updates = 3 if args.smoke else args.updates
    feature_drop = None if args.feature_drop == "none" else args.feature_drop
    train(args.seed, n_updates, args.out, smoke=args.smoke, device=args.device,
          reward_mode=args.reward, feature_drop=feature_drop,
          curriculum=args.curriculum, arch=args.arch)


if __name__ == "__main__":
    main()
