"""PPO trainer v2: single conditioned policy over the flexibility ladder
(PPO hyperparameters locked from Y1, Appendix D).

Curriculum (locked; recorded in the protocol log): every episode draws
independently and uniformly
  track     ~ U{replay-train, generator}      (4 training campuses 5/9/10/12)
  m         ~ U{0.5, 0.6, 0.8, 1.0}
  structure ~ U{L0, CHAIN(0.5), CHAIN(1.0), FULL}
  eta       ~ U{1.0, 0.8}                     (eta irrelevant at L0)
Instances are materialised at m = 1.0 workload; the OVERLAY carries m
(crews scaled inside the overlay builder, Y1 convention), so replay and
generator compose identically. Generator arrival_multiplier stays 1.0
(contention comes from m; Frame U overload is an evaluation-only axis).

Dev sets (all 32 cell-stratified replay-TRAIN instances, Y1 builder):
  primary  : (m=0.6, CHAIN(1.0), eta=0.8)  -> checkpoint selection signal
  monitor1 : replay-default (L0, m=1.0)    -> the Y1 plateau, reported only
  monitor2 : (m=0.6, FULL, eta=0.8)        -> upper-envelope monitor
Checkpoint = per-seed minimum of the primary signal (Y1 curriculum-v2
lesson: default-capacity signals plateau and cannot discriminate).

PPO: lr 3e-4, gamma 1.0, GAE lambda 0.98, clip 0.2, 4 epochs, minibatch
1024, entropy 0.01, value coef 0.5, grad clip 0.5, 16 envs x 512 steps
(8192 transitions/update); updates 1,200 (doubled vs Y1's 600). Rewards
carry the /100 scale inside the env.

CLI:
  PYTHONPATH=.:vendor python -m methods.train2 --arch mlp --seed 301 \
      --updates 1200 --out results/train/mlp_seed301
  --smoke: 3 updates, tiny sizes, cpu.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("Y2_TORCH_THREADS", "2")))

from env.engine import PairDispatchEnv
from env.validator2 import validate as validate2
from methods.policy2 import make_policy
from overlays.build import build_overlay, load_crews

_REPO = Path(__file__).resolve().parents[1]
Y1_ROOT = os.environ.get("FMWOS_Y1_ROOT", str(_REPO.parent / "FM-Scheduling"))
INST_ROOT = os.path.join(Y1_ROOT, "data", "processed", "instances")
PARAM_ROOT = os.path.join(Y1_ROOT, "results", "p2_generator")
CAPACITY = os.path.join(Y1_ROOT, "results", "p1_calib", "capacity.csv")
TRAIN_ANCHOR_MAX = "2017-12-31"

CAMPUSES = [5, 9, 10, 12]
M_CHOICES = [0.5, 0.6, 0.8, 1.0]
STRUCT_CHOICES = [("dedicated", None), ("chain", 0.5), ("chain", 1.0),
                  ("full", None)]
ETA_CHOICES = [1.0, 0.8]
GEN_SEED_BASE = 30000

DEV_PRIMARY = {"structure": "chain", "phi": 1.0, "eta": 0.8, "m": 0.6}
DEV_MON_FULL = {"structure": "full", "phi": None, "eta": 0.8, "m": 0.6}
DEV_MON_L0 = {"structure": "dedicated", "phi": None, "eta": 1.0, "m": 1.0}


# --------------------------------------------------------------------------- #
# Instance + overlay pools
# --------------------------------------------------------------------------- #
def list_replay_train_files(campuses, sizes):
    files = []
    for c in campuses:
        for s in sizes:
            pat = os.path.join(INST_ROOT, "c%02d" % c, "replay", str(s),
                               "*.json")
            for f in sorted(glob.glob(pat)):
                try:
                    with open(f) as fh:
                        ws = json.load(fh)["meta"]["window_start"]
                except Exception:
                    continue
                if ws <= TRAIN_ANCHOR_MAX:
                    files.append(f)
    return files


class _LRU(OrderedDict):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        v = fn()
        self[key] = v
        if len(self) > self.cap:
            self.popitem(last=False)
        return v


class OverlayBank:
    """Deterministic overlay cache keyed by (campus, structure, phi, eta, m)."""

    def __init__(self):
        self._crews = {}
        self._cache = {}

    def crews(self, campus):
        if campus not in self._crews:
            self._crews[campus] = load_crews(CAPACITY, campus)
        return self._crews[campus]

    def get(self, campus, structure, phi, eta, m, perm_seed=None):
        key = (campus, structure, phi, eta, m, perm_seed)
        if key not in self._cache:
            self._cache[key] = build_overlay(campus, self.crews(campus),
                                             structure, phi, eta, m,
                                             perm_seed=perm_seed)
        return self._cache[key]


class CurriculumSampler:
    """Uniform mixed-flexibility episode sampler (module docstring).

    ``structures`` restricts the structure mix (specialist ablation,
    protocol log 6: e.g. [("chain", 1.0)] or [("full", None)])."""

    def __init__(self, campuses, sizes, seed, structures=None):
        self.campuses = list(campuses)
        self.sizes = list(sizes)
        self.structures = list(structures or STRUCT_CHOICES)
        self.rng = random.Random(seed + 777)
        self.replay_files = list_replay_train_files(campuses, sizes)
        self._cache = _LRU(256)
        self._gen_seed = GEN_SEED_BASE
        self.bank = OverlayBank()
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                        "vendor"))
        from fmwos_y1 import generator
        self.generator = generator
        self.params = {}
        for c in self.campuses:
            p = os.path.join(PARAM_ROOT, "params_c%d.json" % c)
            with open(p) as fh:
                self.params[c] = json.load(fh)
        if not self.replay_files:
            raise RuntimeError("no replay train files found")
        self.last_spec = None

    def _load(self, path):
        return self._cache.get_or(path, lambda: json.load(open(path)))

    def sample(self):
        rng = self.rng
        structure, phi = rng.choice(self.structures)
        eta = rng.choice(ETA_CHOICES)
        m = rng.choice(M_CHOICES)
        if rng.random() < 0.5:
            path = rng.choice(self.replay_files)
            inst = self._load(path)
            campus = inst["meta"]["campus"]
            track = "replay"
        else:
            campus = rng.choice(self.campuses)
            size = rng.choice(self.sizes)
            seed = self._gen_seed
            self._gen_seed += 1
            inst = self.generator.generate(self.params[campus], size, seed,
                                           crew_multiplier=1.0)
            track = "generator"
        ov = self.bank.get(campus, structure, phi, eta, m)
        self.last_spec = {"track": track, "campus": campus,
                          "structure": structure, "phi": phi, "eta": eta,
                          "m": m}
        return inst, ov


def load_dev_set(campuses, sizes, n):
    """Y1's fixed cell-stratified replay-train dev set (round-robin)."""
    per_cell = []
    for c in campuses:
        for s in sizes:
            pat = os.path.join(INST_ROOT, "c%02d" % c, "replay", str(s),
                               "*.json")
            cell = []
            for f in sorted(glob.glob(pat)):
                try:
                    with open(f) as fh:
                        ws = json.load(fh)["meta"]["window_start"]
                except Exception:
                    continue
                if ws <= TRAIN_ANCHOR_MAX:
                    cell.append(f)
            if cell:
                per_cell.append(cell)
    picked, i = [], 0
    while len(picked) < n and per_cell:
        progressed = False
        for cell in per_cell:
            if i < len(cell):
                picked.append(cell[i])
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
        i += 1
    return [json.load(open(f)) for f in picked]


# --------------------------------------------------------------------------- #
# Rollout helpers
# --------------------------------------------------------------------------- #
def _stack_obs(obs_list):
    pairs = np.stack([o["pairs"] for o in obs_list])
    mask = np.stack([o["mask"] for o in obs_list])
    ctx = np.stack([o["ctx"] for o in obs_list])
    return pairs, mask, ctx


def _kmax(mask_arr):
    """Valid pairs form a PREFIX of the slot axis, so the longest prefix in
    the batch bounds the compute; padding beyond it is dead weight."""
    return max(1, int(mask_arr.sum(axis=-1).max()))


def _batch_value(policy, obs_list, device):
    pairs, mask, ctx = _stack_obs(obs_list)
    k = _kmax(mask)
    with torch.no_grad():
        _l, value = policy(
            torch.as_tensor(pairs[:, :k], dtype=torch.float32, device=device),
            torch.as_tensor(mask[:, :k], dtype=torch.bool, device=device),
            torch.as_tensor(ctx, dtype=torch.float32, device=device))
    return value.cpu().numpy()


def _cpu_clone(policy):
    """Small-net CPU clone for dev evaluation: single-observation act() is
    latency-bound and ~10x faster on CPU than on a contended GPU."""
    import copy
    c = copy.deepcopy(policy).to("cpu")
    c.eval()
    return c


def eval_dev(policy, dev_instances, bank, spec, device, allow_wait=False):
    """Mean validator WWT of the greedy policy on a dev overlay cell."""
    policy.eval()
    wwts = []
    for inst in dev_instances:
        campus = inst["meta"]["campus"]
        ov = bank.get(campus, spec["structure"], spec["phi"], spec["eta"],
                      spec["m"])
        env = PairDispatchEnv(inst, ov, allow_wait=allow_wait)
        obs = env.reset()
        done = env._done
        while not done:
            a, _, _, _ = policy.act(obs, greedy=True, device=device)
            obs, _r, done, _info = env.step(a)
        sched = env.to_schedule("rl")
        wwts.append(validate2(inst, sched, ov)["metrics"]["WWT"])
    return float(np.mean(wwts)) if wwts else float("nan")


# --------------------------------------------------------------------------- #
# PPO (Y1 loop, retargeted at pair observations)
# --------------------------------------------------------------------------- #
def train(seed, updates, out_dir, arch="mlp", smoke=False, device=None,
          structures=None, wait=False):
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

    sampler = CurriculumSampler(CAMPUSES, sizes, seed,
                                structures=structures)
    bank = sampler.bank
    dev_set = load_dev_set(CAMPUSES, [150, 400] if not smoke else [50], n_dev)

    from env.engine import F_TOTAL
    policy = make_policy(arch, f_pair=(F_TOTAL + 1) if wait else None)
    if wait:
        # One-variable-changed guard: the wait variant may differ from the
        # released architecture ONLY by the extra is-wait input column
        # (hidden extra weights in the first encoder layer).
        base_n = sum(p.numel() for p in make_policy(arch).parameters())
        new_n = sum(p.numel() for p in policy.parameters())
        expected = base_n + policy.hidden
        assert new_n == expected, (
            "wait-variant parameter drift: %d != released %d + %d"
            % (new_n, base_n, policy.hidden))
    policy = policy.to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)

    config = {
        "seed": seed, "updates": updates, "smoke": smoke,
        "device": str(device), "arch": arch, "wait": bool(wait),
        "n_envs": n_envs, "steps_per_env": steps_per_env, "sizes": sizes,
        "campuses": CAMPUSES, "gamma": gamma, "gae_lambda": lam,
        "clip": clip, "epochs": epochs, "minibatch": minibatch,
        "ent_coef": ent_coef, "val_coef": val_coef,
        "max_grad_norm": max_grad, "lr": lr,
        "curriculum": {"m": M_CHOICES, "structures": sampler.structures,
                       "eta": ETA_CHOICES, "track": "50/50 replay/generator",
                       "arrival_multiplier": 1.0},
        "dev": {"primary": DEV_PRIMARY, "monitor_full": DEV_MON_FULL,
                "monitor_l0": DEV_MON_L0, "n_dev": len(dev_set),
                "eval_every": eval_every},
        "n_replay_train_files": len(sampler.replay_files),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    curves_path = os.path.join(out_dir, "curves.csv")
    with open(curves_path, "w") as fh:
        fh.write("update,mean_train_return,dev_primary,dev_l0,dev_full,"
                 "entropy,value_loss,seconds\n")

    def _fresh_env():
        inst, ov = sampler.sample()
        return PairDispatchEnv(inst, ov, allow_wait=wait)

    envs = [_fresh_env() for _ in range(n_envs)]
    cur_obs = [e.reset() for e in envs]
    ep_return = [0.0 for _ in range(n_envs)]

    best_dev = float("inf")
    _pc = _cpu_clone(policy)
    _cpu = torch.device("cpu")
    dev_primary = eval_dev(_pc, dev_set, bank, DEV_PRIMARY, _cpu,
                           allow_wait=wait)
    dev_l0 = eval_dev(_pc, dev_set, bank, DEV_MON_L0, _cpu, allow_wait=wait)
    dev_full = eval_dev(_pc, dev_set, bank, DEV_MON_FULL, _cpu,
                        allow_wait=wait)
    if dev_primary < best_dev:
        best_dev = dev_primary
        policy.save(os.path.join(out_dir, "best.pt"))

    K, FP, FC = policy.k_pairs, policy.f_pair, policy.f_ctx
    L = steps_per_env
    for update in range(updates):
        t0 = time.perf_counter()
        policy.eval()

        b_pairs = np.zeros((L, n_envs, K, FP), dtype=np.float32)
        b_mask = np.zeros((L, n_envs, K), dtype=bool)
        b_ctx = np.zeros((L, n_envs, FC), dtype=np.float32)
        b_act = np.zeros((L, n_envs), dtype=np.int64)
        b_logp = np.zeros((L, n_envs), dtype=np.float32)
        b_val = np.zeros((L, n_envs), dtype=np.float32)
        b_rew = np.zeros((L, n_envs), dtype=np.float32)
        b_done = np.zeros((L, n_envs), dtype=np.float32)
        completed_returns = []

        for t in range(L):
            pairs, mask, ctx = _stack_obs(cur_obs)
            kx = _kmax(mask)
            pt = torch.as_tensor(pairs[:, :kx], dtype=torch.float32,
                                 device=device)
            mt = torch.as_tensor(mask[:, :kx], dtype=torch.bool,
                                 device=device)
            xt = torch.as_tensor(ctx, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, value = policy(pt, mt, xt)
                logp_all = torch.log_softmax(logits, dim=-1)
                probs = logp_all.exp() * mt.to(logits.dtype)
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
                logp_a = logp_all.gather(-1,
                                         actions.unsqueeze(-1)).squeeze(-1)
            acts = actions.cpu().numpy()
            b_pairs[t] = pairs
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
                    envs[i] = _fresh_env()
                    cur_obs[i] = envs[i].reset()
                else:
                    cur_obs[i] = nobs

        last_val = _batch_value(policy, cur_obs, device)

        adv = np.zeros((L, n_envs), dtype=np.float32)
        lastgae = np.zeros(n_envs, dtype=np.float32)
        for t in reversed(range(L)):
            nonterminal = 1.0 - b_done[t]
            nextval = last_val if t == L - 1 else b_val[t + 1]
            delta = b_rew[t] + gamma * nextval * nonterminal - b_val[t]
            lastgae = delta + gamma * lam * nonterminal * lastgae
            adv[t] = lastgae
        ret = adv + b_val

        N = L * n_envs
        k_roll = _kmax(b_mask)               # longest prefix this rollout
        f_pairs = torch.as_tensor(
            np.ascontiguousarray(b_pairs.reshape(N, K, FP)[:, :k_roll]),
            device=device)
        f_mask = torch.as_tensor(
            np.ascontiguousarray(b_mask.reshape(N, K)[:, :k_roll]),
            device=device)
        f_ctx = torch.as_tensor(b_ctx.reshape(N, FC), device=device)
        f_act = torch.as_tensor(b_act.reshape(N), device=device)
        f_logp = torch.as_tensor(b_logp.reshape(N), device=device)
        f_adv = torch.as_tensor(adv.reshape(N), device=device)
        f_ret = torch.as_tensor(ret.reshape(N), device=device)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        policy.train()
        mb = min(minibatch, N)
        # Y2_MICROBATCH: gradient-accumulation chunk size. The OPTIMISATION
        # step still uses the locked minibatch of 1024 samples (losses are
        # sample-mean-weighted across chunks before the single step), so
        # the update is mathematically the locked one up to float summation
        # order; chunking only caps activation memory (needed for the
        # attention class on the shared GPU).
        micro = int(os.environ.get("Y2_MICROBATCH", "0")) or mb
        ent_acc, vloss_acc, n_mb = 0.0, 0.0, 0
        idx = np.arange(N)
        for _ep in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, N, mb):
                sl = idx[start:start + mb]
                n_sl = len(sl)
                optim.zero_grad()
                ent_w, vloss_w = 0.0, 0.0
                for cs in range(0, n_sl, micro):
                    cl = sl[cs:cs + micro]
                    st = torch.as_tensor(cl, device=device)
                    logp, entropy, value = policy.evaluate(
                        f_pairs[st], f_mask[st], f_ctx[st], f_act[st])
                    ratio = torch.exp(logp - f_logp[st])
                    a = f_adv[st]
                    pg1 = -a * ratio
                    pg2 = -a * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                    pg_loss = torch.max(pg1, pg2).mean()
                    v_loss = ((value - f_ret[st]) ** 2).mean()
                    ent = entropy.mean()
                    loss = pg_loss + val_coef * v_loss - ent_coef * ent
                    w = float(len(cl)) / n_sl
                    (loss * w).backward()
                    ent_w += float(ent.item()) * w
                    vloss_w += float(v_loss.item()) * w
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
                optim.step()
                ent_acc += ent_w
                vloss_acc += vloss_w
                n_mb += 1

        mean_ent = ent_acc / max(1, n_mb)
        mean_vloss = vloss_acc / max(1, n_mb)
        mean_ret = (float(np.mean(completed_returns))
                    if completed_returns else 0.0)

        is_last = (update == updates - 1)
        if (update % eval_every == 0) or is_last:
            _pc = _cpu_clone(policy)
            dev_primary = eval_dev(_pc, dev_set, bank, DEV_PRIMARY, _cpu,
                                   allow_wait=wait)
            dev_l0 = eval_dev(_pc, dev_set, bank, DEV_MON_L0, _cpu,
                              allow_wait=wait)
            dev_full = eval_dev(_pc, dev_set, bank, DEV_MON_FULL, _cpu,
                                allow_wait=wait)
            if dev_primary < best_dev:
                best_dev = dev_primary
                policy.save(os.path.join(out_dir, "best.pt"))
        secs = time.perf_counter() - t0
        with open(curves_path, "a") as fh:
            fh.write("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f\n"
                     % (update, mean_ret, dev_primary, dev_l0, dev_full,
                        mean_ent, mean_vloss, secs))
        print("[u%04d] ret=%.4f dev_pri=%.4f dev_l0=%.4f dev_full=%.4f "
              "ent=%.4f vloss=%.4f %.1fs%s"
              % (update, mean_ret, dev_primary, dev_l0, dev_full, mean_ent,
                 mean_vloss, secs,
                 "  *best" if dev_primary == best_dev else ""), flush=True)

    policy.save(os.path.join(out_dir, "final.pt"))
    if not os.path.exists(os.path.join(out_dir, "best.pt")):
        policy.save(os.path.join(out_dir, "best.pt"))
    with open(os.path.join(out_dir, "done.json"), "w") as fh:
        json.dump({"best_dev_primary": best_dev, "updates": updates}, fh)
    print("[train2] done. best dev(primary)=%.4f -> %s" % (best_dev, out_dir))
    return {"best_dev_primary": best_dev, "out_dir": out_dir}


def main(argv=None):
    ap = argparse.ArgumentParser(description="PPO trainer v2 (pair policies)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--updates", type=int, default=1200)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--arch", type=str, default="mlp",
                    choices=["mlp", "attn"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--structures", type=str, default=None,
                    help="restrict the curriculum structure mix: "
                         "'chain1.0' or 'full' (specialist ablation)")
    ap.add_argument("--wait", action="store_true",
                    help="wait-action variant: one extra always-legal-"
                         "while-anyone-is-busy WAIT token (E10)")
    args = ap.parse_args(argv)
    n_updates = 3 if args.smoke else args.updates
    structures = None
    if args.structures == "chain1.0":
        structures = [("chain", 1.0)]
    elif args.structures == "full":
        structures = [("full", None)]
    elif args.structures:
        raise SystemExit("unknown --structures %r" % args.structures)
    train(args.seed, n_updates, args.out, arch=args.arch, smoke=args.smoke,
          device=args.device, structures=structures, wait=args.wait)


if __name__ == "__main__":
    main()
