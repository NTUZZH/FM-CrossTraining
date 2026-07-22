#!/usr/bin/env python
"""E-D analysis: heterogeneous-eta dividends vs tier1 L0.

Replicates the released dividend definition (analysis.gates.twt_best): a cell's
envelope is the lowest pooled-mean TWT over methods; dividend = L0_best minus
flex_best in raw TWT units, pooled over the verdict campuses and sizes. Fixed
EDD uses the single edd method. Comparators are tier1 uniform eta=0.8, m=0.6.
Writes results/e9_etahet/summary.json.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from analysis import gates as G

RANKED = G.RANKED
METHODS = RANKED + ["random"]


def pooled_mean(sub, method):
    s = sub[sub.method == method]
    return float(s.twt.mean()) if len(s) else None


def envelope_best(sub):
    vals = {m: pooled_mean(sub, m) for m in METHODS}
    vals = {k: v for k, v in vals.items() if v is not None}
    bn = min(vals, key=vals.get)
    return vals[bn], bn


def top3(sub):
    vals = {m: pooled_mean(sub, m) for m in METHODS}
    vals = {k: v for k, v in vals.items() if v is not None}
    return [k for k, _ in sorted(vals.items(), key=lambda x: x[1])[:3]]


def main():
    het = pd.read_csv("results/e9_etahet/results.csv")
    t1 = pd.read_csv("results/tier1/results.csv")
    fam1 = G.expand_l0(t1)
    fam1 = fam1[(fam1.m == 0.6) & (fam1.eta == 0.8)]

    n_configs = het.groupby(["instance_id", "structure", "draw_seed"]).ngroups
    sanity = {
        "n_rows": int(len(het)),
        "n_configs": int(n_configs),
        "n_configs_expected": 4578,
        "n_infeasible": int((het.validator_ok == 0).sum()),
        "all_validator_ok": bool((het.validator_ok == 1).all()),
        "methods": sorted(het.method.unique().tolist()),
        "draw_seeds": sorted(int(s) for s in het.draw_seed.unique()),
        "rows_per_config": int(len(het) / n_configs),
    }

    def cell1(st):
        return fam1[fam1.structure == st]

    l0 = cell1("dedicated")
    l0_env, l0_env_m = envelope_best(l0)
    l0_edd = pooled_mean(l0, "edd")

    uni = {}
    for st in ("chain", "full"):
        e, en = envelope_best(cell1(st))
        uni[st] = {"env": e, "env_method": en, "edd": pooled_mean(cell1(st), "edd")}
    uni_delta = {
        "env_chain": l0_env - uni["chain"]["env"],
        "env_full": l0_env - uni["full"]["env"],
        "edd_chain": l0_edd - uni["chain"]["edd"],
        "edd_full": l0_edd - uni["full"]["edd"],
    }

    seeds = sorted(int(s) for s in het.draw_seed.unique())
    per_draw = {}
    for s in seeds:
        hs = het[het.draw_seed == s]
        row = {"realized_mean_eta": float(hs.mean_eta.mean())}
        for st in ("chain", "full"):
            sub = hs[hs.structure == st]
            e, en = envelope_best(sub)
            row["%s_env" % st] = e
            row["%s_env_method" % st] = en
            row["%s_edd" % st] = pooled_mean(sub, "edd")
        row["delta_env_chain"] = l0_env - row["chain_env"]
        row["delta_env_full"] = l0_env - row["full_env"]
        row["delta_edd_chain"] = l0_edd - row["chain_edd"]
        row["delta_edd_full"] = l0_edd - row["full_edd"]
        row["chain_ge_full_env"] = bool(row["delta_env_chain"] >= row["delta_env_full"])
        row["chain_ge_full_edd"] = bool(row["delta_edd_chain"] >= row["delta_edd_full"])
        per_draw[s] = row

    rank = {"uniform": {st: top3(cell1(st)) for st in ("chain", "full")}, "het": {}}
    for s in seeds:
        hs = het[het.draw_seed == s]
        rank["het"][s] = {st: top3(hs[hs.structure == st]) for st in ("chain", "full")}

    xs = [per_draw[s]["realized_mean_eta"] for s in seeds]
    corr = {}
    for key, uk in [("delta_env_chain", "env_chain"), ("delta_env_full", "env_full"),
                    ("delta_edd_chain", "edd_chain"), ("delta_edd_full", "edd_full")]:
        ys = [per_draw[s][key] - uni_delta[uk] for s in seeds]
        r = float(np.corrcoef(xs, ys)[0, 1]) if np.std(ys) > 0 else float("nan")
        corr[key] = {"movement": [round(y, 3) for y in ys], "pearson_r_n3": r}

    summary = {
        "sanity": sanity,
        "L0": {"env": l0_env, "env_method": l0_env_m, "edd": l0_edd},
        "uniform08_ref": {**uni, "delta": uni_delta},
        "per_draw": per_draw,
        "top3": rank,
        "correlation_meanEta_vs_movement_n3": {"mean_eta": xs, **corr},
        "claim_chain_ge_full": {
            "env_all_draws": bool(all(per_draw[s]["chain_ge_full_env"] for s in seeds)),
            "edd_all_draws": bool(all(per_draw[s]["chain_ge_full_edd"] for s in seeds)),
        },
    }
    with open("results/e9_etahet/summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    print("SANITY:", json.dumps(sanity))
    print("\nL0: env=%.4f (%s)  edd=%.4f" % (l0_env, l0_env_m, l0_edd))
    print("UNIFORM-0.8: env Dchain=%.2f Dfull=%.2f | edd Dchain=%.2f Dfull=%.2f"
          % (uni_delta["env_chain"], uni_delta["env_full"],
             uni_delta["edd_chain"], uni_delta["edd_full"]))
    print("\nPER-DRAW (raw TWT dividends vs L0):")
    print("seed        meanEta  ENV:Dchain Dfull ch>=fu | EDD:Dchain Dfull ch>=fu")
    for s in seeds:
        r = per_draw[s]
        print("%d  %.4f   %6.2f %6.2f  %-5s |   %6.2f %6.2f  %-5s"
              % (s, r["realized_mean_eta"], r["delta_env_chain"],
                 r["delta_env_full"], r["chain_ge_full_env"],
                 r["delta_edd_chain"], r["delta_edd_full"], r["chain_ge_full_edd"]))
    print("\nTOP-3 per cell:")
    for st in ("chain", "full"):
        print("  uniform %-5s: %s" % (st, rank["uniform"][st]))
    for s in seeds:
        for st in ("chain", "full"):
            print("  het %d %-5s: %s" % (s, st, rank["het"][s][st]))
    print("\nCORRELATION meanEta vs movement (het-uniform), n=3; meanEta=%s"
          % [round(x, 4) for x in xs])
    for k in ("delta_env_chain", "delta_env_full", "delta_edd_chain", "delta_edd_full"):
        print("  %s: move=%s r=%.3f" % (k, corr[k]["movement"], corr[k]["pearson_r_n3"]))
    print("\nCLAIM chain>=full all draws: env=%s edd=%s"
          % (summary["claim_chain_ge_full"]["env_all_draws"],
             summary["claim_chain_ge_full"]["edd_all_draws"]))
    print("wrote results/e9_etahet/summary.json")


if __name__ == "__main__":
    main()
