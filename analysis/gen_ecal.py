"""E-CAL roster-calibration sensitivity: supplement table and numbers keys.

Reads results/ecal/ (written by experiments/run_ecal.py on crew tables from
experiments/recalibrate.py) and writes paper/sections/gen_ecal.tex plus the
``ecal-`` keys of paper/numbers.json.

Conventions, all inherited from the released analysis so the rows are
comparable with the manuscript:

  rules envelope   the lowest pooled mean weighted tardiness over the seven
                   ranked rules and Random, taken per cell (analysis/gates.py
                   twt_best with rule-only pools)
  fixed EDD        the same quantities with EDD as the dispatcher in every
                   cell
  dividend         Delta(structure) = TWT_best(L0) - TWT_best(structure)
  capture ratio    rho = Delta(CHAIN(1.0)) / Delta(FULL)
  guard            a family is not evaluable when Delta(FULL) stays under 2%
                   of TWT_best(L0)
  interval         95% percentile interval from 10,000 instance-cluster
                   resamples (instances resampled within campus), seed
                   20260718, one resample applied coherently to the L0, chain
                   and full cells

The utilization column holds the demand of a 95th-percentile week over the
scaled roster hours of the calibration in question, the statistic of the
released roster table: the numerator is the released p95 weekly hours of the
campus and the denominator is 40 bh times the scaled headcount. Demand is
therefore held fixed and only the roster moves.

Usage: the coordinator calls generate(numbers, root) from analysis/build_all.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RANKED = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
ENVELOPE = RANKED + ["random"]
VERDICT_CAMPUSES = (5, 9, 10, 12)
MS = (1.0, 0.8, 0.6)
GUARD = 0.02
THRESHOLD = 0.70
BOOT_B, BOOT_SEED = 10_000, 20260718

# Ordered by how generous the roster is, so the table reads monotonically.
LABELS = ["q090_train", "released", "q095_train", "q0975_train",
          "q099_train", "q095_all"]
SLUG = {"released": "rel", "q090_train": "q90", "q095_train": "q95",
        "q0975_train": "q975", "q099_train": "q99", "q095_all": "q95all"}
PRETTY = {"released": "$p_{95}$, train (released)",
          "q090_train": "$p_{90}$, train",
          "q095_train": "$p_{95}$, train (rebuilt)",
          "q0975_train": "$p_{97.5}$, train",
          "q099_train": "$p_{99}$, train",
          "q095_all": "$p_{95}$, all years"}
PLAIN = {"released": "the released 95th-percentile train-window roster",
         "q090_train": "the 90th percentile",
         "q095_train": "the rebuilt 95th percentile",
         "q0975_train": "the 97.5th percentile",
         "q099_train": "the 99th percentile",
         "q095_all": "the 95th percentile on all recorded years"}
MTAG = {1.0: "m10", 0.8: "m08", 0.6: "m06"}


# --------------------------------------------------------------------------- #
# Rosters                                                                      #
# --------------------------------------------------------------------------- #
def _scaled_crews(crews, m):
    return {c["trade"]: max(1, int(round(c["crew"] * m))) for c in crews}


def roster_table(root):
    """{(label, m, campus) -> (headcount, p95-week utilization)}."""
    from overlays.build import load_crews
    released = {c: load_crews(root / "results/ecal/capacity_released.csv", c)
                for c in VERDICT_CAMPUSES}
    demand = {c: sum(x["volume"] for x in released[c])
              for c in VERDICT_CAMPUSES}
    out = {}
    for label in LABELS:
        path = root / "results/ecal" / ("capacity_%s.csv" % label)
        for c in VERDICT_CAMPUSES:
            crews = load_crews(path, c)
            for m in MS:
                head = sum(_scaled_crews(crews, m).values())
                out[(label, m, c)] = (head, demand[c] / (40.0 * head))
    return out


# --------------------------------------------------------------------------- #
# Cell matrices and the bootstrap                                              #
# --------------------------------------------------------------------------- #
def _cell_matrix(df, structure, m, eta, master, phi=None):
    sub = df[(df.structure == structure) & (df.m == m)]
    if structure != "dedicated":
        sub = sub[sub.eta == eta]
    if phi is not None:
        sub = sub[sub.phi == phi]
    pos = {iid: k for k, iid in enumerate(master)}
    M = np.full((len(master), len(ENVELOPE)), np.nan)
    for j, meth in enumerate(ENVELOPE):
        s = sub[sub.method == meth]
        for iid, v in zip(s.instance_id, s.twt):
            M[pos[iid], j] = v
    return M


def bootstrap_weights(campus_of, n_boot=BOOT_B, seed=BOOT_SEED):
    """(n_boot, n_instances) resample counts, instances drawn within campus."""
    rng = np.random.default_rng(seed)
    n = len(campus_of)
    W = np.zeros((n_boot, n), dtype=np.float64)
    for c in np.unique(campus_of):
        idx = np.flatnonzero(campus_of == c)
        draw = rng.integers(0, len(idx), size=(n_boot, len(idx)))
        for b in range(n_boot):
            np.add.at(W[b], idx[draw[b]], 1.0)
    return W


def cell_best(M, cols, W=None):
    """Lowest column mean (point estimate, and per-resample when W is given)."""
    sel = M[:, cols]
    point = np.nanmin(np.nanmean(sel, axis=0))
    if W is None:
        return point, None
    means = (W @ np.nan_to_num(sel)) / W.sum(axis=1, keepdims=True)
    return point, np.nanmin(means, axis=1)


def family_stats(cells, cols, W):
    """Dividends, capture ratio, guard and the interval for one family."""
    l0, l0b = cell_best(cells["dedicated"], cols, W)
    ch, chb = cell_best(cells["chain"], cols, W)
    fu, fub = cell_best(cells["full"], cols, W)
    d_ch, d_fu = l0 - ch, l0 - fu
    guard = bool(d_fu < GUARD * l0)
    rho = (d_ch / d_fu) if d_fu != 0 else float("nan")
    ci = None
    if W is not None and not guard and d_fu > 0:
        r = (l0b - chb) / (l0b - fub)
        r = r[np.isfinite(r)]
        if len(r):
            ci = (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))
    return {"l0": float(l0), "delta_chain": float(d_ch),
            "delta_full": float(d_fu), "pct": float(100.0 * d_fu / l0),
            "guard": guard, "rho": float(rho), "ci": ci}


def evaluate(root):
    """Every (calibration, m) family, on the rules envelope and fixed EDD."""
    p = root / "results/ecal/results.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(
        root / "results/ecal/results.csv")
    df = df[df.method.isin(ENVELOPE)]
    master = sorted(df[df.structure == "dedicated"].instance_id.unique())
    campus_of = np.array([int(i.split("_")[0][1:]) for i in master])
    W = bootstrap_weights(campus_of)
    env_cols = list(range(len(ENVELOPE)))
    edd_cols = [ENVELOPE.index("edd")]

    out = {}
    for label in LABELS:
        sub = df[df.calibration == label]
        for m in MS:
            cells = {"dedicated": _cell_matrix(sub, "dedicated", m, 1.0,
                                               master)}
            for eta in (1.0, 0.8):
                cells[("chain", eta)] = _cell_matrix(sub, "chain", m, eta,
                                                     master, phi=1.0)
                cells[("full", eta)] = _cell_matrix(sub, "full", m, eta,
                                                    master)
            for eta in (1.0, 0.8):
                c = {"dedicated": cells["dedicated"],
                     "chain": cells[("chain", eta)],
                     "full": cells[("full", eta)]}
                out[(label, m, eta, "env")] = family_stats(c, env_cols, W)
                out[(label, m, eta, "edd")] = family_stats(c, edd_cols, W)
    return out, len(master), len(df)


# --------------------------------------------------------------------------- #
# Formatting                                                                   #
# --------------------------------------------------------------------------- #
def _rng(vals, fmt="%.2f"):
    lo, hi = min(vals), max(vals)
    if fmt % lo == fmt % hi:
        return fmt % lo
    return "%s--%s" % (fmt % lo, fmt % hi)


def _num(x, fmt="%.1f"):
    """Format without a negative zero, with a typographic minus sign."""
    t = fmt % x
    if t.startswith("-"):
        return t[1:] if float(t) == 0.0 else "$-$" + t[1:]
    return t


def _rho_cell(f):
    if f["guard"] or f["delta_full"] <= 0:
        return "--"
    if f["ci"] is None:
        return "%.2f" % f["rho"]
    return "%.2f (%.2f--%.2f)" % (f["rho"], f["ci"][0], f["ci"][1])


def _ci_key(f, nd=3):
    if f["ci"] is None:
        return "%.*f" % (nd, f["rho"])
    return "%.*f (95\\%% CI %.*f--%.*f)" % (nd, f["rho"], nd, f["ci"][0],
                                            nd, f["ci"][1])


def _same_family(fam, roster, a, b):
    """True when two calibrations produce the same roster and the same
    dividends, so one of them is a duplicate row."""
    for m in MS:
        for c in VERDICT_CAMPUSES:
            if roster[(a, m, c)][0] != roster[(b, m, c)][0]:
                return False
        for eta in (1.0, 0.8):
            for kind in ("env", "edd"):
                fa, fb = fam[(a, m, eta, kind)], fam[(b, m, eta, kind)]
                for k in ("l0", "delta_chain", "delta_full"):
                    if abs(fa[k] - fb[k]) > 5e-3:
                        return False
    return True


def build_table(fam, roster, n_inst, labels):
    lines = ["%% Generated by analysis/gen_ecal.py from results/ecal/ "
             "-- do not edit.",
             "\\begin{table}[!htb]",
             "\\caption{Roster-calibration sensitivity. Crews are rebuilt "
             "from the raw corpus at four weekly-hours quantiles on the "
             "training window and at the 95th percentile on all recorded "
             "years, and the flexibility ladder is re-scored with the seven "
             "ranked rules and Random. $N$ = scaled headcount and "
             "utilization = demand of a 95th-percentile week over scaled "
             "roster hours, both as ranges over the four verdict campuses. "
             "$\\Delta$(FULL) is the full-flexibility dividend on the rules "
             "envelope at $\\eta = 1.0$, with its share of L0 in "
             "parentheses. A dash in a $\\rho$ column marks a family the 2\\% "
             "denominator guard declares not evaluable; where a ratio is "
             "shown, the parenthesis is a 95\\% instance-cluster bootstrap "
             "interval. $n$ = " + str(n_inst) + " instances per cell.}",
             "\\label{tab:ecal}", "\\centering", "\\footnotesize",
             "\\begin{tabular*}{\\tblwidth}{@{}l@{\\extracolsep{\\fill}}"
             "rrrrrrrr@{}}",
             "\\toprule",
             "Calibration & $m$ & $N$ & Util. & L0 TWT & $\\Delta$(FULL) "
             "& $\\rho$ (env.) & $\\rho$ (EDD) & $\\Delta$ at $\\eta{=}0.8$ "
             "(ch./full) \\\\",
             "\\midrule"]
    for label in labels:
        for k, m in enumerate(MS):
            e = fam[(label, m, 1.0, "env")]
            d = fam[(label, m, 1.0, "edd")]
            e8 = fam[(label, m, 0.8, "env")]
            heads = [roster[(label, m, c)][0] for c in VERDICT_CAMPUSES]
            utils = [roster[(label, m, c)][1] for c in VERDICT_CAMPUSES]
            lines.append(" & ".join([
                PRETTY[label] if k == 0 else "",
                "%.1f" % m,
                _rng(heads, "%d"),
                _rng(utils, "%.2f"),
                _num(e["l0"]),
                "%s (%s\\%%)" % (_num(e["delta_full"]), _num(e["pct"])),
                _rho_cell(e),
                ("--" if (d["guard"] or d["delta_full"] <= 0)
                 else "%.2f" % d["rho"]),
                "%s / %s" % (_num(e8["delta_chain"]), _num(e8["delta_full"])),
            ]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _m_phrase(ms):
    if len(ms) == len(MS):
        return "every $m$"
    parts = ["%.1f" % m for m in sorted(ms)]
    if len(parts) == 1:
        return "$m = %s$" % parts[0]
    return "$m = %s$ and $%s$" % (", ".join(parts[:-1]), parts[-1])


def _guard_sentence(fam, labels):
    """Plain-English list of the families the 2% guard declares not
    evaluable on the rules envelope at eta = 1.0."""
    parts = []
    for label in labels:
        ms = [m for m in MS if fam[(label, m, 1.0, "env")]["guard"]]
        if ms:
            parts.append("%s at %s" % (PLAIN[label], _m_phrase(ms)))
    if not parts:
        return "no calibration"
    if len(parts) == 1:
        return parts[0]
    return "%s, and %s" % (", ".join(parts[:-1]), parts[-1])


def generate(numbers: dict, root: Path):
    """Write paper/sections/gen_ecal.tex and add the ecal- keys."""
    root = Path(root)
    res_dir = root / "results/ecal"
    if not (res_dir / "results.csv").exists():
        return None
    fam, n_inst, n_rows = evaluate(root)
    roster = roster_table(root)

    # The released roster and its rebuild from the raw corpus are shown as one
    # row when they agree and as two when they do not.
    rebuilt_matches = _same_family(fam, roster, "released", "q095_train")
    labels = [x for x in LABELS if not (rebuilt_matches
                                        and x == "q095_train")]
    tex = build_table(fam, roster, n_inst, labels)
    out = root / "paper/sections/gen_ecal.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)

    numbers["ecal-n-calibrations"] = str(len(labels))
    numbers["ecal-rebuild-matches-released"] = (
        "reproduces the released roster" if rebuilt_matches
        else "differs from the released roster")
    numbers["ecal-n-instances"] = "{:,}".format(n_inst)
    numbers["ecal-n-rows"] = "{:,}".format(n_rows)

    # Per-calibration keys at the evaluable crew multiplier and elsewhere.
    for label in LABELS:
        s = SLUG[label]
        for m in MS:
            t = MTAG[m]
            e = fam[(label, m, 1.0, "env")]
            d = fam[(label, m, 1.0, "edd")]
            numbers["ecal-l0-%s-%s" % (s, t)] = "%.1f" % e["l0"]
            numbers["ecal-delta-full-%s-%s" % (s, t)] = (
                "%.2f" % e["delta_full"])
            numbers["ecal-delta-full-pct-%s-%s" % (s, t)] = (
                "%.1f\\%%" % e["pct"])
            numbers["ecal-guard-%s-%s" % (s, t)] = (
                "fires" if e["guard"] else "clears")
            if not e["guard"] and e["delta_full"] > 0:
                numbers["ecal-rho-env-%s-%s" % (s, t)] = _ci_key(e)
            if not d["guard"] and d["delta_full"] > 0:
                numbers["ecal-rho-edd-%s-%s" % (s, t)] = _ci_key(d)
            heads = [roster[(label, m, c)][0] for c in VERDICT_CAMPUSES]
            utils = [roster[(label, m, c)][1] for c in VERDICT_CAMPUSES]
            numbers["ecal-headcount-%s-%s" % (s, t)] = _rng(heads, "%d")
            numbers["ecal-util-%s-%s" % (s, t)] = _rng(utils, "%.2f")
        allu = [roster[(label, m, c)][1] for m in MS
                for c in VERDICT_CAMPUSES]
        numbers["ecal-util-range-%s" % s] = _rng(allu, "%.2f")

    # Portfolio headcount change against the released roster.
    base = sum(roster[("released", 1.0, c)][0] for c in VERDICT_CAMPUSES)
    for label, key in (("q090_train", "ecal-headcount-change-q90"),
                       ("q0975_train", "ecal-headcount-change-q975"),
                       ("q099_train", "ecal-headcount-change-q99"),
                       ("q095_all", "ecal-headcount-change-q95all")):
        tot = sum(roster[(label, 1.0, c)][0] for c in VERDICT_CAMPUSES)
        numbers[key] = "%+.0f\\%%" % (100.0 * (tot - base) / base)

    # Capture ratios over every evaluable family at eta = 1.0.
    ev_env = [fam[(lab, m, 1.0, "env")]["rho"] for lab in LABELS for m in MS
              if not fam[(lab, m, 1.0, "env")]["guard"]
              and fam[(lab, m, 1.0, "env")]["delta_full"] > 0]
    ev_edd = [fam[(lab, m, 1.0, "edd")]["rho"] for lab in LABELS for m in MS
              if not fam[(lab, m, 1.0, "edd")]["guard"]
              and fam[(lab, m, 1.0, "edd")]["delta_full"] > 0]
    if ev_env:
        numbers["ecal-rho-range-evaluable"] = _rng(ev_env, "%.2f")
        numbers["ecal-n-evaluable"] = str(len(ev_env))
        numbers["ecal-rho-min-evaluable"] = "%.3f" % min(ev_env)
    if ev_edd:
        numbers["ecal-rho-edd-range-evaluable"] = _rng(ev_edd, "%.2f")
    numbers["ecal-guard-fires-list"] = _guard_sentence(fam, labels)

    holds = bool(ev_env) and min(ev_env + ev_edd) >= THRESHOLD
    numbers["ecal-summary-sentence"] = (
        "Across every calibration the reading is the same: the dividend stays "
        "immaterial wherever capacity is adequate, and wherever the guard "
        "clears the chain captures between %s and %s of what full "
        "flexibility delivers on the rules envelope (%s to %s under fixed "
        "EDD), so the quantile and the calibration window do not carry the "
        "result."
        % ("%.2f" % min(ev_env), "%.2f" % max(ev_env),
           "%.2f" % min(ev_edd), "%.2f" % max(ev_edd))
        if holds else
        "The reading changes with the calibration: the capture ratio falls to "
        "%.2f in at least one family the guard declares evaluable, so the "
        "chaining-suffices claim is calibration-dependent."
        % min(ev_env)) if ev_env else (
        "No calibration produces a dividend the 2\\% guard declares material, "
        "so the capture ratio is not evaluable at any of them.")

    with open(res_dir / "analysis.json", "w") as f:
        json.dump({"%s|%s|%s|%s" % (lab, m, eta, kind): v
                   for (lab, m, eta, kind), v in fam.items()}, f, indent=2)
    return str(out)


if __name__ == "__main__":
    n = {}
    print(generate(n, Path(__file__).resolve().parents[1]))
    for k in sorted(n):
        print("%-34s %s" % (k, n[k]))
