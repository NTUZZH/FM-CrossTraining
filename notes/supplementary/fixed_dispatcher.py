"""Recompute the flexibility dividend and capture ratio
under FIXED dispatchers (no ex-post envelope). Reuses analysis/gates.py
pooling conventions exactly. Existing result data only.

Run: PYTHONPATH=.:vendor python notes/supplementary/fixed_dispatcher.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis.gates as G

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

RANKED = G.RANKED  # edd wspt atc pfifo mor lfj_atc atc_eta
MLP_MAIN = ["v2mlp%d" % i for i in range(301, 311)]
ATTN_MAIN = ["v2attn%d" % i for i in range(401, 411)]

# Envelope pools exactly as build_all.py builds them for Gate C.
POOLS = {r: [r] for r in RANKED + ["random"]}
POOLS["policy_mlp"] = MLP_MAIN
POOLS["policy_attn"] = ATTN_MAIN

# Fixed single-policy methods we evaluate as operational dispatchers.
FIXED_RULES = ["edd", "atc", "atc_eta", "mor", "lfj_atc"]
FIXED_METHODS = {r: [r] for r in FIXED_RULES}
FIXED_METHODS["policy_mlp"] = MLP_MAIN  # pair-MLP seed pool


def pooled_twt(df_cell, methods):
    """Pooled mean TWT for one method set, mirroring gates.twt_best:
    single method -> straight mean over the cell rows; multi-method
    (seed pool) -> per-config seed-mean then mean over configs."""
    sub = df_cell[df_cell.method.isin(methods)]
    if not len(sub):
        return None
    if len(methods) > 1:
        sub = sub.copy()
        sub["cfg"] = G.config_key(sub)
        return float(sub.groupby("cfg").twt.mean().mean())
    return float(sub.twt.mean())


def load_tier1():
    df = pd.read_csv(RES / "tier1" / "results.csv")
    return G.expand_l0(df)


def cell(df, m, eta, st):
    return df[(df.m == m) & (df.eta == eta) & (df.structure == st)]


# --------------------------------------------------------------------------
# 0. Reproduction check against gate_c
# --------------------------------------------------------------------------
def reproduction_check(raw):
    gc = G.gate_c(raw, POOLS)
    f = gc["families"]["m0.6_eta1.0"]
    return gc, f


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    raw = pd.read_csv(RES / "tier1" / "results.csv")
    df = G.expand_l0(raw)

    MS = [1.0, 0.8, 0.6]
    ETAS = [1.0, 0.8]
    STRUCTS = ["dedicated", "chain", "full"]

    # ---- reproduction check ----
    gc = G.gate_c(raw, POOLS)
    print("=" * 78)
    print("REPRODUCTION CHECK (envelope Gate C via analysis/gates.py)")
    print("=" * 78)
    for key in ("m0.6_eta1.0", "m0.8_eta1.0", "m0.6_eta0.8", "m0.8_eta0.8"):
        fam = gc["families"][key]
        c = fam["cells"]
        print("%-14s L0=%.4f(%s) CHAIN=%.4f(%s) FULL=%.4f(%s) "
              "dChain=%.4f dFull=%.4f rho=%s" % (
                  key,
                  c["dedicated"]["twt_best"], c["dedicated"]["best_method"],
                  c["chain"]["twt_best"], c["chain"]["best_method"],
                  c["full"]["twt_best"], c["full"]["best_method"],
                  fam["delta_chain"], fam["delta_full"],
                  ("%.4f" % fam["rho"]) if fam["rho"] is not None else "NA"))
    rho06 = gc["families"]["m0.6_eta1.0"]["rho"]
    print("\nrho(m=0.6, eta=1.0) = %.4f  [published 0.870]  match=%s"
          % (rho06, abs(rho06 - 0.870) < 5e-4))

    # ---- Table 1: per-cell pooled TWT, every method + envelope ----
    print("\n" + "=" * 78)
    print("TABLE 1. Per-cell pooled TWT (envelope best + fixed methods)")
    print("=" * 78)
    methods_report = {name: POOLS[name] for name in
                      ["edd", "atc", "atc_eta", "mor", "lfj_atc"]}
    methods_report["policy_mlp"] = MLP_MAIN
    hdr = ["m", "eta", "struct", "envBest", "envMethod"] + \
        list(methods_report.keys())
    rows = []
    twt = {}  # (m,eta,st) -> dict method->twt, plus 'env','envm'
    for m in MS:
        for eta in ETAS:
            for st in STRUCTS:
                dc = cell(df, m, eta, st)
                ev, evm = G.twt_best(dc, POOLS)
                d = {"env": ev, "envm": evm}
                for name, meths in methods_report.items():
                    d[name] = pooled_twt(dc, meths)
                twt[(m, eta, st)] = d
                row = [m, eta, st, _f(ev), evm] + \
                    [_f(d[k]) for k in methods_report]
                rows.append(row)
    _print_table(hdr, rows)

    # ---- Table 2: fixed-dispatcher dividends and capture ratios ----
    print("\n" + "=" * 78)
    print("TABLE 2. Dividends Delta(CHAIN), Delta(FULL) and rho per method")
    print("Delta_method(S) = TWT_method(L0) - TWT_method(S), same method on L0")
    print("Guard: envelope Delta_env(FULL) >= 2% of TWT_best(L0)")
    print("=" * 78)
    hdr2 = ["m", "eta", "method", "L0", "CHAIN", "FULL",
            "dChain", "dFull", "rho", "guardOK"]
    rows2 = []
    rho_flags = []
    for m in MS:
        for eta in ETAS:
            envL0 = twt[(m, eta, "dedicated")]["env"]
            envFull = twt[(m, eta, "full")]["env"]
            env_dfull = envL0 - envFull
            guard_ok = env_dfull >= 0.02 * envL0
            # envelope row
            env_dchain = envL0 - twt[(m, eta, "chain")]["env"]
            rho_env = env_dchain / env_dfull if env_dfull else None
            rows2.append([m, eta, "ENVELOPE", _f(envL0),
                          _f(twt[(m, eta, "chain")]["env"]), _f(envFull),
                          _f(env_dchain), _f(env_dfull),
                          _f(rho_env, 4), "yes" if guard_ok else "NO"])
            for name in ["edd", "atc", "atc_eta", "mor", "lfj_atc",
                         "policy_mlp"]:
                l0 = twt[(m, eta, "dedicated")][name]
                ch = twt[(m, eta, "chain")][name]
                fu = twt[(m, eta, "full")][name]
                if None in (l0, ch, fu):
                    continue
                dch = l0 - ch
                dfu = l0 - fu
                rho = dch / dfu if dfu else None
                rows2.append([m, eta, name, _f(l0), _f(ch), _f(fu),
                              _f(dch), _f(dfu), _f(rho, 4),
                              "yes" if guard_ok else "NO"])
                if guard_ok and eta == 1.0 and rho is not None:
                    if abs(rho - 0.870) > 0.1:
                        rho_flags.append((m, eta, name, rho))
    _print_table(hdr2, rows2)

    print("\n--- FLAG: fixed-dispatcher rho differing from 0.870 by >0.1 "
          "(eta=1.0, guard OK) ---")
    if rho_flags:
        for m, eta, name, rho in rho_flags:
            print("  !! m=%s eta=%s %-11s rho=%.4f  (|d|=%.3f)"
                  % (m, eta, name, rho, abs(rho - 0.870)))
    else:
        print("  none")

    # ---- Table 3: envelope argmin per cell ----
    print("\n" + "=" * 78)
    print("TABLE 3. Envelope argmin (best method) per (m, eta, structure)")
    print("=" * 78)
    hdr3 = ["m", "eta", "dedicated", "chain", "full"]
    rows3 = []
    for m in MS:
        for eta in ETAS:
            r = [m, eta]
            for st in STRUCTS:
                r.append(twt[(m, eta, st)]["envm"])
            rows3.append(r)
    _print_table(hdr3, rows3)

    transfer_e4()
    realized_adoption()
    benefit_per_skill(twt)

    return twt


def transfer_e4():
    """Task 4: campus-2 transfer ratios TWT(struct)/TWT(dedicated) at eta=0.8,
    reproduced under the envelope, then under FIXED EDD and FIXED ATC-eta."""
    e4 = pd.read_csv(RES / "e4" / "results.csv")
    d4 = G.expand_l0(e4)
    pools4 = {r: [r] for r in RANKED}          # build_all: RANKED only (+pols)
    pools4["policy_mlp"] = MLP_MAIN
    pools4["policy_attn"] = ATTN_MAIN

    def bcell(campus, st, m, eta, methods):
        sub = d4[(d4.campus == campus) & (d4.structure == st)
                 & (d4.m == m) & (d4.eta == eta)]
        return pooled_twt(sub, methods) if isinstance(methods, list) \
            else G.twt_best(sub, methods)[0]

    print("\n" + "=" * 78)
    print("TABLE 4. Campus-2 transfer ratios (eta=0.8), TWT(S)/TWT(L0)")
    print("Published envelope: m=0.8 -> 1.00/0.76/0.72 ; m=0.6 -> 1.00/0.73/0.68")
    print("=" * 78)
    hdr = ["campus", "m", "policy", "L0", "CHAIN", "FULL",
           "r_L0", "r_CHAIN", "r_FULL"]
    rows = []
    variants = [("ENVELOPE", pools4), ("edd", ["edd"]),
                ("atc_eta", ["atc_eta"])]
    for c in (1, 2):
        for m in (0.8, 0.6):
            for name, meths in variants:
                l0 = bcell(c, "dedicated", m, 0.8, meths)
                ch = bcell(c, "chain", m, 0.8, meths)
                fu = bcell(c, "full", m, 0.8, meths)
                rows.append([c, m, name, _f(l0), _f(ch), _f(fu),
                             _f(l0 / l0, 2) if l0 else "NA",
                             _f(ch / l0, 2) if l0 else "NA",
                             _f(fu / l0, 2) if l0 else "NA"])
    _print_table(hdr, rows)


def load_overlay(name):
    return json.load(open(ROOT / "overlays" / "generated" / (name + ".json")))


def realized_adoption():
    """Task 5: nominal phi vs realized second-skill fraction and budget B,
    per verdict campus and m, for CHAIN(0.25/0.5/1.0). Skills are independent
    of eta, so eta100 overlays are used."""
    print("\n" + "=" * 78)
    print("TABLE 5. Realized adoption (CHAIN): nominal phi vs realized "
          "second-skill fraction, budget B")
    print("=" * 78)
    hdr = ["campus", "m", "head", "phi_nom", "struct", "B",
           "n_second", "realized_frac"]
    rows = []
    campuses = [5, 9, 10, 12]
    ms = [(1.0, "m100"), (0.8, "m080"), (0.6, "m060")]
    phis = [(0.25, "phi025"), (0.5, "phi050"), (1.0, "phi100")]
    for c in campuses:
        for mval, mtag in ms:
            for phi, ptag in phis:
                nm = "c%02d_chain_%s_eta100_%s" % (c, ptag, mtag)
                d = load_overlay(nm)
                techs = d["technicians"]
                head = d["headcount"]
                n2 = sum(1 for t in techs if len(t["skills"]) > 1)
                rows.append([c, mval, head, phi, "chain(%.2f)" % phi,
                             d["budget_B"], n2, "%.3f" % (n2 / head)])
    _print_table(hdr, rows)


def benefit_per_skill(twt):
    """Task 6: Delta_env(CHAIN(1.0)) and Delta_env(FULL) per realized added
    skill (budget B), at m=0.6, eta=1.0, pooled over verdict campuses."""
    print("\n" + "=" * 78)
    print("TABLE 6. Benefit per added skill, m=0.6 eta=1.0 pooled")
    print("=" * 78)
    l0 = twt[(0.6, 1.0, "dedicated")]["env"]
    ch = twt[(0.6, 1.0, "chain")]["env"]
    fu = twt[(0.6, 1.0, "full")]["env"]
    d_chain = l0 - ch
    d_full = l0 - fu
    campuses = [5, 9, 10, 12]
    Bc, Bf = {}, {}
    for c in campuses:
        Bc[c] = load_overlay("c%02d_chain_phi100_eta100_m060" % c)["budget_B"]
        Bf[c] = load_overlay("c%02d_full_eta100_m060" % c)["budget_B"]
    meanBc = np.mean(list(Bc.values()))
    meanBf = np.mean(list(Bf.values()))
    print("Delta_env(CHAIN(1.0)) = %.4f   Delta_env(FULL) = %.4f  (pooled TWT)"
          % (d_chain, d_full))
    print("Realized added skills B (m=0.6) per campus:")
    print("  CHAIN(1.0): " + ", ".join("c%02d=%d" % (c, Bc[c]) for c in
                                        campuses) + "  mean=%.1f" % meanBc)
    print("  FULL:       " + ", ".join("c%02d=%d" % (c, Bf[c]) for c in
                                       campuses) + "  mean=%.1f" % meanBf)
    print("Benefit per added skill (pooled Delta / mean B):")
    print("  CHAIN(1.0): %.4f / %.1f = %.5f TWT per added skill"
          % (d_chain, meanBc, d_chain / meanBc))
    print("  FULL:       %.4f / %.1f = %.5f TWT per added skill"
          % (d_full, meanBf, d_full / meanBf))
    print("  ratio (chain-per-skill / full-per-skill) = %.2fx"
          % ((d_chain / meanBc) / (d_full / meanBf)))


def _f(x, nd=4):
    if x is None:
        return "NA"
    return ("%.*f" % (nd, x))


def _print_table(hdr, rows):
    cols = list(zip(*([hdr] + rows))) if rows else [hdr]
    widths = [max(len(str(c)) for c in col) for col in cols]
    fmt = "  ".join("%%-%ds" % w for w in widths)
    print(fmt % tuple(hdr))
    print(fmt % tuple("-" * w for w in widths))
    for row in rows:
        print(fmt % tuple(str(c) for c in row))


if __name__ == "__main__":
    main()
