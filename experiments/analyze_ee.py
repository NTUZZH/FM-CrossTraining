#!/usr/bin/env python
"""E-E summary statistics. Reads the merged instrument/bigcap
results, writes results/ee_instrument/summary.json + results/ee_bigcap/
summary.json, prints a readable report. Aggregation: per-schedule metric,
then mean over instances and seeds (pool = mean over all rows of the class).
TWT convention validated against tier1 (Random FULL eta0.8 = 341.27/327.21)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RANKED = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
RANDOM_ANCHOR = {0.6: 341.27, 0.8: 327.21}


def _pool(df, kind):
    if kind == "edd":
        return df[df.method == "edd"]
    if kind == "mlp":
        return df[df.method.str.startswith("v2mlp")]
    if kind == "attn":
        return df[df.method.str.startswith("v2attn")]
    raise ValueError(kind)


def sec_share(df):
    """Per-schedule secondary share: count and base-hours weighted."""
    cnt = (df.instr_n_secondary / df.instr_n_assign)
    hrs = (df.instr_pbh_secondary / df.instr_pbh_total)
    rhrs = (df.instr_rph_secondary / df.instr_rph_total)
    return float(cnt.mean()), float(hrs.mean()), float(rhrs.mean())


def cap_stats(df):
    d = df[df.instr_decisions > 0]
    pair = (d.instr_pair_cap_binds / d.instr_decisions)
    order = (d.instr_order_cap_binds / d.instr_decisions)
    excl = (d.instr_edd_excluded / d.instr_decisions)
    return (float(pair.mean()), float(order.mean()), float(excl.mean()))


def main():
    ins = pd.read_csv(ROOT / "results/ee_instrument/results.csv")
    big = pd.read_csv(ROOT / "results/ee_bigcap/results.csv")
    tier1 = pd.read_parquet(ROOT / "results/tier1/results.parquet")

    # ------------------------------------------------------------------ #
    # 1 + 2: secondary share and cap stats, per cell (eta=0.8 fixed)
    # ------------------------------------------------------------------ #
    r3c3 = {}
    r16 = {}
    cells = [("full", 0.6), ("full", 0.8), ("chain", 0.6), ("chain", 0.8)]
    print("=" * 78)
    print("1. Secondary-assignment share (mean over instances&seeds), "
          "eta=0.8")
    print("   cell            EDD_cnt EDD_hr  | MLP_cnt MLP_hr  | "
          "ATT_cnt ATT_hr")
    for (st, m) in cells:
        cell = ins[(ins.structure == st) & (ins.m == m)]
        row = {}
        for kind in ("edd", "mlp", "attn"):
            c, h, rh = sec_share(_pool(cell, kind))
            row[kind] = {"count_share": c, "hours_share_pbh": h,
                         "hours_share_realized": rh}
        r3c3["%s_m%02d" % (st, int(m * 100))] = row
        print("   %-14s  %.3f   %.3f   | %.3f   %.3f   | %.3f   %.3f"
              % (st + "_m" + str(m), row["edd"]["count_share"],
                 row["edd"]["hours_share_pbh"], row["mlp"]["count_share"],
                 row["mlp"]["hours_share_pbh"], row["attn"]["count_share"],
                 row["attn"]["hours_share_pbh"]))

    print()
    print("2. Cap statistics (fraction of policy decisions), eta=0.8")
    print("   cell            class  pair_cap  order_cap  edd_excluded")
    for (st, m) in cells:
        cell = ins[(ins.structure == st) & (ins.m == m)]
        cellrow = {}
        for kind in ("mlp", "attn"):
            p, o, e = cap_stats(_pool(cell, kind))
            cellrow[kind] = {"pair_cap_bind_frac": p,
                             "order_cap_bind_frac": o,
                             "edd_excluded_frac": e}
            print("   %-14s  %-5s  %.4f    %.4f     %.4f"
                  % (st + "_m" + str(m), kind, p, o, e))
        r16["%s_m%02d" % (st, int(m * 100))] = cellrow

    # over-use verdict
    verdict1 = {}
    for (st, m) in cells:
        key = "%s_m%02d" % (st, int(m * 100))
        edd = r3c3[key]["edd"]["count_share"]
        mlp = r3c3[key]["mlp"]["count_share"]
        att = r3c3[key]["attn"]["count_share"]
        verdict1[key] = {"mlp_minus_edd": mlp - edd, "attn_minus_edd": att - edd}
    print()
    print("   Over-use vs EDD (count-share delta):")
    for k, v in verdict1.items():
        print("     %-14s mlp %+0.3f  attn %+0.3f"
              % (k, v["mlp_minus_edd"], v["attn_minus_edd"]))

    # ------------------------------------------------------------------ #
    # 3: large-cap verdict per FULL cell
    # ------------------------------------------------------------------ #
    print()
    print("=" * 78)
    print("3. Large-cap verdict, penalized FULL (eta=0.8). Pooled seed-mean TWT")
    largecap = {}
    for m in (0.6, 0.8):
        # best ranked rule from tier1
        tr = tier1[(tier1.structure == "full") & (tier1.eta == 0.8)
                   & (tier1.m == m)]
        rule_means = {r: tr[tr.method == r].twt.mean() for r in RANKED}
        best_rule = min(rule_means, key=rule_means.get)
        cell = big[(big.structure == "full") & (big.m == m)]
        mrow = {"random": RANDOM_ANCHOR[m],
                "best_rule": {"name": best_rule,
                              "twt": float(rule_means[best_rule])}}
        print("   -- m=%s -- Random=%.2f  best rule=%s %.2f"
              % (m, RANDOM_ANCHOR[m], best_rule, rule_means[best_rule]))
        for kind in ("mlp", "attn"):
            std = _pool(cell[cell.cap == "std"], kind).twt.mean()
            bigt = _pool(cell[cell.cap == "big"], kind).twt.mean()
            delta = bigt - std
            pct = 100.0 * delta / std if std else float("nan")
            mrow[kind] = {"twt_std": float(std), "twt_big": float(bigt),
                          "delta": float(delta), "pct": float(pct)}
            print("      %-4s  std=%.2f  big=%.2f  delta=%+0.2f (%+0.2f%%)"
                  % (kind, std, bigt, delta, pct))
        largecap["full_m%02d" % int(m * 100)] = mrow

    # ------------------------------------------------------------------ #
    # 4: sanity
    # ------------------------------------------------------------------ #
    sanity = {
        "instrument_rows": int(len(ins)),
        "instrument_infeasible": int((ins.validator_ok == 0).sum()),
        "instrument_twt_mismatch": int((ins.twt_matches == 0).sum()),
        "bigcap_rows": int(len(big)),
        "bigcap_infeasible": int((big.validator_ok == 0).sum()),
        "bigcap_std_twt_mismatch": int(
            (big[big.cap == "std"].twt_matches == 0).sum()),
        "instrument_configs": int(len(ins) // 21),
        "bigcap_configs": int(len(big) // 40),
    }
    print()
    print("4. Sanity:", json.dumps(sanity))

    ins_summary = {"cells_eta": 0.8, "r3c3_secondary_share": r3c3,
                   "r3c3_overuse_vs_edd": verdict1,
                   "r16_cap_stats": r16, "sanity": sanity}
    big_summary = {"largecap_full_penalized": largecap, "sanity": sanity}
    with open(ROOT / "results/ee_instrument/summary.json", "w") as f:
        json.dump(ins_summary, f, indent=2)
    with open(ROOT / "results/ee_bigcap/summary.json", "w") as f:
        json.dump(big_summary, f, indent=2)
    print("\nwrote results/ee_instrument/summary.json and "
          "results/ee_bigcap/summary.json")


if __name__ == "__main__":
    main()
