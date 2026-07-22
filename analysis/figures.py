"""All manuscript figures from released result files.

Design system: Y1's frozen tokens and validated categorical palette
(paper figures render identically across the two papers); two new method
colours for the flexibility-aware rules. Vector PDF for LaTeX + PNG QA.
Figures whose inputs are missing (policy rows before training) are skipped
with a note; build_all re-runs them when data lands.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.patches import (Rectangle, FancyArrowPatch,  # noqa: E402
                                FancyBboxPatch)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"
Y1_ROOT = Path(os.environ.get("FMWOS_Y1_ROOT", ROOT.parent / "FM-Scheduling"))

# ---- Y1 design tokens (frozen) -------------------------------------------- #
# INK and INK2 are the two text inks; both are pure black so every string in
# every figure renders black. MUTE stays reserved for non-text structure only
# (tick marks, arrow shafts, light rules); never use it to colour a string.
INK = "#000000"
INK2 = "#000000"
MUTE = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURF = "#fcfcfb"

CMAP = {
    "edd": "#008300", "pfifo": "#8f8d86", "atc": "#4a3aa7",
    "wspt": "#eb6834", "mor": "#e34948", "random": "#c0beb6",
    "ga": "#eda100", "cpsat": "#e87ba4", "roll": "#1baf7a",
    "policy": "#2a78d6",
    # v2 additions (kept distinguishable from the frozen set)
    "lfj_atc": "#0e7c94",     # teal
    "atc_eta": "#8a5a00",     # brown
    "policy_attn": "#7fb2e5",  # light blue (secondary class)
}
PRETTY = {"edd": "EDD", "pfifo": "pFIFO", "atc": "ATC", "wspt": "WSPT",
          "mor": "LPT", "random": "Random", "ga": "GA", "cpsat": "CP-SAT",
          "roll": "Rolling CP-SAT", "policy": "Policy",
          "lfj_atc": "LFJ-ATC", "atc_eta": "ATC-$\\eta$",
          "policy_attn": "Policy (attn)"}

STRUCT_COLOR = {"dedicated": "#8f8d86", "chain": "#008300",
                "generalist": "#eda100", "full": "#4a3aa7"}
STRUCT_PRETTY = {"dedicated": "L0", "chain": "CHAIN", "generalist": "GEN",
                 "full": "FULL"}

MM = 1 / 25.4


def _rc():
    # Figures must carry the same typeface as the body text (Times). The
    # manuscript sets newtx, whose roman is the Nimbus/Termes clone of Times
    # New Roman; matching it here keeps every glyph in the PDF consistent.
    plt.rcParams.update({
        "font.size": 8.6, "axes.titlesize": 9.2, "axes.labelsize": 8.6,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.5,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTE, "ytick.color": MUTE,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "savefig.facecolor": SURF,
        "figure.facecolor": SURF,
    })


def figsize(w_mm, h_mm):
    return (w_mm * MM, h_mm * MM)


def style_ax(ax):
    ax.set_facecolor(SURF)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Tick MARKS stay muted (structural); tick NUMBERS must be pure black.
    # tick_params(colors=) would recolour both, so set them separately.
    ax.tick_params(length=2.2, width=0.5, color=MUTE, labelcolor=INK)
    # Never let matplotlib factor out a "+3.184e2" style offset: on the
    # narrow-range panels it turns readable tardiness values into an
    # unreadable exponent header.
    try:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    except AttributeError:
        pass


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIGDIR / (name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def _read(family):
    p = RES / family / "results.csv"
    return pd.read_csv(p) if p.exists() else None


def _expand_l0(df):
    import analysis.gates as G
    return G.expand_l0(df)


def _pool_method(df, method_or_pool):
    """Pooled mean TWT; a list = seed pool (per-config mean first)."""
    import analysis.gates as G
    if isinstance(method_or_pool, str):
        sub = df[df.method == method_or_pool]
        return float(sub.twt.mean()) if len(sub) else None
    sub = df[df.method.isin(method_or_pool)]
    if not len(sub):
        return None
    sub = sub.copy()
    sub["cfg"] = G.config_key(sub)
    return float(sub.groupby("cfg").twt.mean().mean())


def _policy_pools(df):
    mlp = sorted({m for m in df.method.unique() if isinstance(m, str)
                  and m.startswith("v2mlp") and 301 <= int(m[5:]) <= 310})
    attn = sorted({m for m in df.method.unique() if isinstance(m, str)
                   and m.startswith("v2attn") and 401 <= int(m[6:]) <= 410})
    return (mlp if len(mlp) == 10 else None,
            attn if len(attn) == 10 else None)


# ---- Instance bootstrap for point uncertainty ----------------------------- #
# A point's estimate is a pooled mean TWT over instances. Rules carry one row
# per instance; a policy pool carries ten seeds per instance, so we collapse it
# to the per-instance seed-mean first (exactly as twt_best pools it). Resampling
# these per-instance values reproduces the plotted point estimate in the mean.
BOOT_SEED = 20260718
BOOT_N = 2000


def _instance_values(sub, methods):
    """Per-instance TWT for a winning method pool inside one cell."""
    import analysis.gates as G
    w = sub[sub.method.isin(methods)]
    if not len(w):
        return None
    w = w.copy()
    w["cfg"] = G.config_key(w)
    return w.groupby("cfg").twt.mean().to_numpy()


def _boot_ci(vals, n_boot=BOOT_N, seed=BOOT_SEED, alpha=0.05):
    """95% instance-bootstrap CI of the mean (percentile method)."""
    if vals is None or len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


# Per-structure line/marker styling so the series stay distinguishable in
# grayscale legibility, not only by colour.
STRUCT_LINE = {"dedicated": ":", "chain": "-", "full": "--",
               "generalist": "none"}
STRUCT_MARKER = {"dedicated": "s", "chain": "o", "full": "^",
                 "generalist": "D"}


# --------------------------------------------------------------------------- #
def fig1_pipeline():
    """Whole-paper framework overview on an explicit grid, at print scale.

    House rules (scientific-figure-design section 0): the canvas is
    text-width wide (165 mm) so the figure prints 1:1 and source point
    sizes are printed point sizes; the footer prints at the manuscript
    body size (10 pt) and everything else as close as the four-column
    geometry allows (at body size a column holds ~20 characters per
    line). All text and every title is centred; titles and key labels
    are Title Case; the five protocol stages are all boxed at one width
    with matched heights per line count and equal arrow gaps; the four
    column frames are black hairlines. Colour system unchanged: two
    accents with one meaning each (green = cross-skill links and what
    cross-training buys, indigo = trades/structure), decision-map mini
    grayscale, priority ramp grayscale, all text pure black. Mini-plot
    values are the parent figures' real numbers (fig3 m=0.6 eta=1.0
    points; fig5's eta=0.8 pooled winner pattern; fig7's campus-2 m=0.6
    ratios). Every text artist is asserted inside its own column and
    the figure, so an edit that overflows fails fast.
    """
    _rc()
    import math as _m
    from overlays.build import build_overlay, load_crews, chain_order

    # ---- design tokens -------------------------------------------------- #
    BODY, HEAD, MICRO, FOOT = 8.2, 8.8, 7.0, 10.2   # printed pt (1:1 canvas)
    LW_SPINE, LW_RULE, LW_HAIR, LW_MESH = 0.9, 0.7, 0.5, 0.35
    NODE, ARC = "#4a3aa7", "#008300"            # indigo trades / green links
    MAP_TIE, MAP_SEP = "#d9d7d0", "#706e67"     # decision map, grayscale
    P_GREY = ["#000000", "#4a4a48", "#8f8d86", "#c9c7c0"]  # P1..P4 ink depth

    # ---- grid: equal columns, shared horizontal bands ------------------- #
    W = 165.0                                   # canvas = \linewidth, 1:1
    MARG, GAP = 3.0, 8.0
    CW = (W - 2 * MARG - 3 * GAP) / 4.0         # column width
    cx = [MARG + i * (CW + GAP) for i in range(4)]
    mid4 = [x + CW / 2.0 for x in cx]           # column centres
    PADX = 2.0                                  # frame-to-content padding
    BOX_T, BOX_B = 134.5, 8.6                   # black frames, one y for all
    Y_RULE = 124.8                              # header underline
    B_T, B_B = 121.8, 35.4                      # main-visual band
    Y_ARROW = (B_T + B_B) / 2.0                 # spine rides the band centre
    F_T = 30.6                                  # facts band, first-line top
    Y_FOOT_RULE = 5.8                           # footer rule
    COL_TOP, COL_BOT = 135.0, 9.2

    fig, ax = plt.subplots(figsize=figsize(W, 143.6))
    ax.set_xlim(0, W)
    ax.set_ylim(-6.6, 137)
    ax.axis("off")
    ax.set_position([0.0, 0.0, 1.0, 1.0])   # 1 data unit == 1 mm

    checks, other_texts, tracked = [], [], []

    def ctext(i, x, y, s, **kw):
        kw.setdefault("color", INK)
        kw.setdefault("ha", "center")
        t = ax.text(x, y, s, zorder=6, **kw)
        checks.append((t, i))
        return t

    def otext(x, y, s, **kw):
        kw.setdefault("color", INK)
        t = ax.text(x, y, s, zorder=6, **kw)
        other_texts.append(t)
        return t

    def track(artist):
        ax.add_patch(artist)
        tracked.append(artist)
        return artist

    # ---- panel frames: one BLACK hairline box per column ---------------- #
    for i in range(4):
        track(Rectangle((cx[i] - PADX, BOX_B), CW + 2 * PADX,
                        BOX_T - BOX_B, facecolor="none", edgecolor=INK,
                        linewidth=LW_HAIR, zorder=2))

    # ---- header band: two-line Title-Case headers, centred over a rule -- #
    HEADS = ["Replayed\nWork Orders",
             "Skill-Structure\nOverlays",
             "One Evaluation\nProtocol",
             "Cross-Training\nVerdicts"]
    for i, h in enumerate(HEADS):
        ctext(i, mid4[i], Y_RULE + 0.8, h, fontsize=HEAD, va="bottom",
              fontweight="bold")
        ax.plot([cx[i], cx[i] + CW], [Y_RULE, Y_RULE], color=INK,
                linewidth=LW_RULE, zorder=4)

    # ---- the spine: three gutter arrows, frame edge to frame edge ------- #
    for i in range(3):
        tracked.append(ax.annotate(
            "", xy=(cx[i + 1] - PADX - 0.4, Y_ARROW),
            xytext=(cx[i] + CW + PADX + 0.4, Y_ARROW),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=LW_SPINE,
                            mutation_scale=9, shrinkA=0, shrinkB=0),
            zorder=5))

    # ================= column 1 : replayed work orders =================== #
    # A full-width order stream on a time-arrow baseline; the window
    # bracket and the priority key sit directly beneath, so the column
    # reads as one coherent block.
    x0 = cx[0]
    base_y = 70.0
    tracked.append(ax.annotate(
        "", xy=(x0 + CW - 1.0, base_y), xytext=(x0 + 1.5, base_y),
        arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=LW_RULE,
                        mutation_scale=6, shrinkA=0, shrinkB=0), zorder=3))
    rng_pos = [0.03, 0.09, 0.15, 0.22, 0.27, 0.33, 0.40, 0.46, 0.52, 0.58,
               0.65, 0.71, 0.78, 0.84, 0.90, 0.96]
    rng_hgt = [14.0, 24.0, 10.0, 28.0, 17.0, 12.0, 25.0, 19.0, 9.0, 22.0,
               13.5, 27.0, 16.0, 10.5, 20.5, 29.0]
    rng_pri = [2, 0, 3, 1, 2, 3, 0, 2, 3, 1, 2, 0, 1, 3, 2, 0]
    bx0, bx1 = x0 + 1.5, x0 + CW - 4.0
    for p, h, pr in zip(rng_pos, rng_hgt, rng_pri):
        lx = bx0 + p * (bx1 - bx0)
        ax.plot([lx, lx], [base_y, base_y + h], color=P_GREY[pr],
                linewidth=1.0, zorder=4)
        ax.scatter([lx], [base_y + h], s=9.0, color=P_GREY[pr], zorder=5)
    # window bracket under the stream, label centred on the bracket
    ax.plot([bx0, bx0, bx1, bx1],
            [base_y - 2.4, base_y - 3.6, base_y - 3.6, base_y - 2.4],
            color=MUTE, linewidth=LW_HAIR, zorder=3)
    bmid = (bx0 + bx1) / 2.0
    ctext(0, bmid, base_y - 5.6, "Window 8 h $\\times$ 5 d",
          fontsize=MICRO, va="top")
    # priority key, directly beneath the window label
    for k in range(4):
        ax.scatter([bmid - 6.6 + k * 4.4], [56.5], s=9.0,
                   color=P_GREY[k], zorder=5)
    ctext(0, bmid, 53.0, "Priority P1$\\to$P4", fontsize=MICRO,
          va="top")
    ctext(0, mid4[0], F_T, "4,986 instances,\n6 campuses: real\nreleases, "
          "durations,\npriorities, SLAs.\nOnly the crew overlay\never "
          "changes.", fontsize=BODY, va="top")

    # ============ column 2 : the flexibility ladder (real data) ========== #
    x0 = cx[1]
    cap = str(Y1_ROOT / "results/p1_calib/capacity.csv")
    crews = load_crews(cap, 5)
    order = chain_order(crews)
    K = len(order)
    pos = {g: (_m.cos(_m.pi / 2 - 2 * _m.pi * i / K),
               _m.sin(_m.pi / 2 - 2 * _m.pi * i / K))
           for i, g in enumerate(order)}
    csize = {c["trade"]: c["crew"] for c in crews}
    smax = max(csize.values())
    # The ladder runs vertically, L0 at the top to FULL at the bottom, so
    # the column reads as the flexibility ladder itself and fills the band.
    R = 7.6
    LTOP, LPITCH = 120.6, 21.4
    minis = [("L0", build_overlay(5, crews, "dedicated", None, 1.0, 1.0),
              mid4[1], LTOP - R),
             ("CHAIN(1.0)", build_overlay(5, crews, "chain", 1.0,
                                          1.0, 1.0),
              mid4[1], LTOP - R - LPITCH),
             ("GEN", build_overlay(5, crews, "generalist", None, 1.0, 1.0),
              mid4[1], LTOP - R - 2 * LPITCH),
             ("FULL", build_overlay(5, crews, "full", None, 1.0, 1.0),
              mid4[1], LTOP - R - 3 * LPITCH)]
    for name, ov, mx, my in minis:
        arcs = set()
        for t in ov["technicians"]:
            for s in t["skills"]:
                if s != t["primary"]:
                    arcs.add((t["primary"], s))
        # Dense meshes as thin low-alpha straight chords with no heads
        # (texture, not mass); direction only on the readable chain ring.
        dense = len(arcs) > K
        style, al, lwd, msc = (("-", 0.25, LW_MESH, 3) if dense
                               else ("-|>", 0.7, 0.6, 5))
        for (g, h) in arcs:
            gx, gy = pos[g]
            hx, hy = pos[h]
            ax.annotate("", xy=(mx + hx * R * 0.86, my + hy * R * 0.86),
                        xytext=(mx + gx * R * 0.86, my + gy * R * 0.86),
                        arrowprops=dict(arrowstyle=style, color=ARC,
                                        alpha=al, lw=lwd, shrinkA=1.2,
                                        shrinkB=1.2, mutation_scale=msc),
                        zorder=3)
        for g in order:
            gx, gy = pos[g]
            ax.scatter([mx + gx * R], [my + gy * R],
                       s=7.0 + 30.0 * csize[g] / smax, color=NODE,
                       edgecolor="white", linewidth=0.4, zorder=4)
        ctext(1, mx, my - R - 1.6, "%s $\\cdot$ $B = %d$"
              % (name, ov["budget_B"]), fontsize=MICRO, va="top")
    ctext(1, mid4[1], F_T, "Node = trade,\nsize $\\propto$ crew;\narc = "
          "secondary skill.\nSwept: structure $\\Lambda$,\nadoption "
          "$\\varphi$, penalty $\\eta$,\ncrew size $m$.",
          fontsize=BODY, va="top")

    # ============ column 3 : one evaluation protocol ===================== #
    # Every stage boxed (house rule 4): one width, matched heights per
    # line count, equal arrow gaps.
    x0 = cx[2]
    mid = mid4[2]
    W_BOX = 31.0
    LINE_H = 3.6                       # BODY line pitch in mm
    PAD_Y = 1.4
    STAGES = ["Instance $\\oplus$ Overlay",
              "Pair-Selection\nEngine",
              "7 Priority Rules\n+ Random; CP-SAT,\nExact and Rolling;\n"
              "GA; Learned Pair\nPolicies, 10 Seeds",
              "Independent\nValidator: Sole\nScorer of Every Row",
              "TWT + Secondary\nMetrics"]
    heights = [s.count("\n") * LINE_H + LINE_H + 2 * PAD_Y for s in STAGES]
    A_LEN, A_GAP = 3.2, 0.4
    stack_h = sum(heights) + 4 * (A_LEN + 2 * A_GAP)
    y_cur = B_T - (B_T - B_B - stack_h) / 2.0
    for si, (s, h) in enumerate(zip(STAGES, heights)):
        track(FancyBboxPatch((mid - W_BOX / 2.0, y_cur - h), W_BOX, h,
                             boxstyle="round,pad=0,rounding_size=1.0",
                             facecolor="none", edgecolor=INK,
                             linewidth=0.6, mutation_scale=1, zorder=5))
        ctext(2, mid, y_cur - h / 2.0, s, fontsize=BODY, va="center")
        y_cur -= h
        if si < 4:
            tracked.append(ax.annotate(
                "", xy=(mid, y_cur - A_GAP - A_LEN),
                xytext=(mid, y_cur - A_GAP),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=LW_RULE,
                                mutation_scale=7, shrinkA=0, shrinkB=0),
                zorder=5))
            y_cur -= A_LEN + 2 * A_GAP
    ctext(2, mid, F_T, "Every method, every\nstructure: same\nengine, same "
          "event\nstream, same validator.", fontsize=BODY, va="top")

    # ============ column 4 : verdicts, as miniature real data ============ #
    # Three centred rows: bold Title-Case row title, mini, two-line
    # description, separated by a full air gap per row so the stack
    # breathes; the gate verdicts below stay bold as the column's focal
    # point.
    x0 = cx[3]
    mid = mid4[3]
    PLOT_W = 16.0
    ROW_T = [121.0, 93.0, 65.0]              # row tops, one pitch

    def mini_base(yb):
        ax.plot([mid - PLOT_W / 2.0, mid + PLOT_W / 2.0], [yb, yb],
                color=MUTE, linewidth=LW_HAIR, zorder=3)

    def mini_row(yr, title, desc):
        ctext(3, mid, yr, title, fontsize=HEAD, va="top",
              fontweight="bold")
        ctext(3, mid, yr - 16.4, desc, fontsize=BODY, va="top")

    # (a) design curve: fig3's real m=0.6, eta=1.0 best-pool points
    yr = ROW_T[0]
    yb = yr - 15.0
    vals = [328.97, 316.30, 314.40]          # L0, CHAIN(1.0), FULL
    vmin, vmax = 313.0, 330.0
    pxs = [mid - PLOT_W / 2.0 + 1.4 + k * 6.6 for k in range(3)]
    pys = [yb + 1.4 + 7.6 * (v - vmin) / (vmax - vmin) for v in vals]
    ax.plot([mid - PLOT_W / 2.0, mid + PLOT_W / 2.0], [pys[2], pys[2]],
            color=MUTE, linewidth=LW_HAIR, linestyle=(0, (2, 2)), zorder=3)
    mini_base(yb)
    ax.plot(pxs, pys, color=ARC, linewidth=1.0, zorder=4)
    ax.scatter(pxs, pys, s=10, color=ARC, zorder=5)
    mini_row(yr, "Design Curve",
             "chain captures most\nof full flexibility")

    # (b) decision map: fig5's eta=0.8 pooled winner pattern, grayscale
    yr = ROW_T[1]
    yb = yr - 15.0
    mini_base(yb)
    cells = [["tie", "tie", "sep"],          # m=1.0 (bottom row)
             ["tie", "sep", "sep"],          # m=0.8
             ["tie", "sep", "sep"]]          # m=0.6 (top row)
    cs = 3.2
    for r in range(3):
        for c in range(3):
            col = MAP_SEP if cells[r][c] == "sep" else MAP_TIE
            track(Rectangle((mid - 1.5 * cs + c * cs, yb + 0.5 + r * cs),
                            cs - 0.35, cs - 0.35, facecolor=col,
                            edgecolor="none", zorder=4))
    mini_row(yr, "Decision Map",
             "$\\eta = 0.8$: rules win all\ncells that separate")

    # (c) transfer: fig7's real campus-2 ratios at m=0.6
    yr = ROW_T[2]
    yb = yr - 15.0
    mini_base(yb)
    ratios = [1.00, 0.73, 0.68]
    bcol = [STRUCT_COLOR["dedicated"], STRUCT_COLOR["chain"],
            STRUCT_COLOR["full"]]
    for k, (rv, bc) in enumerate(zip(ratios, bcol)):
        track(Rectangle((mid - 6.9 + k * 4.8, yb), 3.4, 9.4 * rv,
                        facecolor=bc, edgecolor="none", zorder=4))
    mini_row(yr, "Transfer Stress Test",
             "up to 32% lower TWT\non a held-out campus")

    # ---- the two pre-committed gates, centred, verdicts bold ------------ #
    # Short gate names as in Section 6 ("chaining test", "prediction
    # test"); the caption expands both.
    ctext(3, mid, F_T, "Gate C $\\cdot$ Chaining", fontsize=BODY, va="top")
    ctext(3, mid, F_T - 4.2, "PASSED", fontsize=HEAD, va="top",
          fontweight="bold")
    ctext(3, mid, F_T - 9.8, "Gate P $\\cdot$ Prediction", fontsize=BODY,
          va="top")
    ctext(3, mid, F_T - 14.0, "NOT SUPPORTED", fontsize=HEAD, va="top",
          fontweight="bold")

    # ---- footer rule + two centred lines at manuscript body size -------- #
    ax.plot([cx[0] - PADX, cx[3] + CW + PADX], [Y_FOOT_RULE, Y_FOOT_RULE],
            color=INK, linewidth=LW_RULE, zorder=4)
    otext(W / 2.0, 3.6, "two frames: fixed headcount (design), matched "
          "offered load (decision)", fontsize=FOOT, ha="center", va="top")
    otext(W / 2.0, -1.0, "gates dated before any verdict run $\\cdot$ L0 "
          "reproduces the single-skill results bitwise", fontsize=FOOT,
          ha="center", va="top")

    # ---- assertions ----------------------------------------------------- #
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    TOL = 1.0
    for t, i in checks:
        (bxa, bya), (bxb, byc) = ax.transData.transform(
            [(cx[i], COL_BOT), (cx[i] + CW, COL_TOP)])
        tb = t.get_window_extent(renderer)
        if (tb.x0 < bxa - TOL or tb.x1 > bxb + TOL or
                tb.y0 < bya - TOL or tb.y1 > byc + TOL):
            raise RuntimeError(
                "fig1_pipeline: text %r overflows column %d (text "
                "x[%.1f,%.1f] y[%.1f,%.1f] vs column x[%.1f,%.1f] "
                "y[%.1f,%.1f])" % (t.get_text(), i, tb.x0, tb.x1, tb.y0,
                                   tb.y1, bxa, bxb, bya, byc))
    fw, fh = fig.bbox.width, fig.bbox.height
    for t, _i in checks:
        _fig_bounds(t.get_window_extent(renderer), fw, fh, TOL,
                    "text %r" % t.get_text())
    for t in other_texts:
        _fig_bounds(t.get_window_extent(renderer), fw, fh, TOL,
                    "text %r" % t.get_text())
    for p in tracked:
        _fig_bounds(p.get_window_extent(renderer), fw, fh, TOL, "artist")
    _save(fig, "fig1_pipeline")

def _fig_bounds(ext, fw, fh, tol, name):
    if (ext.x0 < -tol or ext.y0 < -tol or ext.x1 > fw + tol or
            ext.y1 > fh + tol):
        raise RuntimeError(
            "fig1_pipeline: %s drawn outside the figure "
            "(x[%.1f,%.1f] y[%.1f,%.1f] vs figure [0,%.1f]x[0,%.1f])"
            % (name, ext.x0, ext.x1, ext.y0, ext.y1, fw, fh))


def fig2_ladder():
    """Trade-level eligibility structure of the ladder on one real campus.

    Nodes = trades on a circle in chain order; a directed arc g -> h means
    some technician has primary g and secondary h. This makes the structure
    legible: L0 has no arcs, CHAIN(1.0) is a single cycle, GEN is a hub,
    FULL is complete. Node size scales with crew size; the flexibility
    budget B is annotated."""
    _rc()
    import math as _m
    from overlays.build import build_overlay, load_crews, chain_order
    cap = str(Y1_ROOT / "results/p1_calib/capacity.csv")
    campus = 5
    crews = load_crews(cap, campus)
    order = chain_order(crews)            # circle position = chain order
    K = len(order)
    pos = {g: (_m.cos(_m.pi / 2 - 2 * _m.pi * i / K),
               _m.sin(_m.pi / 2 - 2 * _m.pi * i / K))
           for i, g in enumerate(order)}
    csize = {c["trade"]: c["crew"] for c in crews}
    smax = max(csize.values())
    ovs = [("L0", build_overlay(campus, crews, "dedicated", None, 1.0, 1.0)),
           ("CHAIN(0.5)", build_overlay(campus, crews, "chain", 0.5, 1.0, 1.0)),
           ("CHAIN(1.0)", build_overlay(campus, crews, "chain", 1.0, 1.0, 1.0)),
           ("GEN", build_overlay(campus, crews, "generalist", None, 1.0, 1.0)),
           ("FULL", build_overlay(campus, crews, "full", None, 1.0, 1.0))]
    fig, axes = plt.subplots(1, 5, figsize=figsize(165, 60))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.24,
                        wspace=0.06)
    for ax, (name, ov) in zip(axes, ovs):
        ax.set_facecolor(SURF)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.5, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        # directed trade-to-trade arcs (dedup)
        arcs = set()
        for t in ov["technicians"]:
            for s in t["skills"]:
                if s != t["primary"]:
                    arcs.add((t["primary"], s))
        # Dense meshes (GEN, FULL) as thin low-alpha straight chords with
        # no heads: overlapping arrowheads clump into solid rosettes at the
        # rim and the panel reads as mass, not structure. Direction stays
        # on the sparse chain panels, where it is actually readable.
        dense = len(arcs) > len(order)
        if dense:
            style, al, lwd, kw = "-", 0.25, 0.4, {}
        else:
            style, al, lwd = "-|>", 0.6, 0.9
            kw = {"connectionstyle": "arc3,rad=0.18"}
        for (g, h) in arcs:
            x0, y0 = pos[g]
            x1, y1 = pos[h]
            ax.annotate("", xy=(x1 * 0.82, y1 * 0.82),
                        xytext=(x0 * 0.82, y0 * 0.82),
                        arrowprops=dict(arrowstyle=style, color="#008300",
                                        alpha=al, lw=lwd,
                                        shrinkA=2, shrinkB=2, **kw),
                        zorder=1)
        for g in order:
            x, y = pos[g]
            ax.scatter([x], [y], s=30 + 150 * csize[g] / smax,
                       color="#4a3aa7", edgecolor="white", linewidth=0.5,
                       zorder=3)
        ax.set_title("%s\n$B=%d$" % (name, ov["budget_B"]), fontsize=9.6,
                     pad=4)
    fig.text(0.5, 0.02, "nodes = trades (size $\\propto$ crew), sized on "
             "campus 5;\narc $g\\!\\to\\!h$ = a technician with primary $g$, "
             "secondary $h$", ha="center", va="bottom", fontsize=10.2,
             color=INK2)
    _save(fig, "fig2_ladder")


def fig3_design_curve():
    """TWT vs flexibility budget B; series = structure; panels m x eta."""
    _rc()
    tier1, tier2 = _read("tier1"), _read("tier2")
    if tier1 is None:
        return
    both = pd.concat([x for x in (tier1, tier2) if x is not None],
                     ignore_index=True)
    df = _expand_l0(both)
    mlp, attn = _policy_pools(both)
    pools = {r: [r] for r in
             ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta",
              "random"]}
    if mlp:
        pools["policy_mlp"] = mlp
    if attn:
        pools["policy_attn"] = attn

    from overlays.build import build_overlay, load_crews
    cap = str(Y1_ROOT / "results/p1_calib/capacity.csv")

    def budget(structure, phi):
        tot = 0
        for c in (5, 9, 10, 12):
            ov = build_overlay(cap and c, load_crews(cap, c), structure,
                               phi, 1.0, 1.0)
            tot += ov["budget_B"]
        return tot

    xs = {("dedicated", None): 0,
          ("chain", 0.25): budget("chain", 0.25),
          ("chain", 0.5): budget("chain", 0.5),
          ("chain", 1.0): budget("chain", 1.0),
          ("generalist", None): budget("generalist", None),
          ("full", None): budget("full", None)}

    import analysis.gates as G
    from matplotlib.lines import Line2D

    b_full = xs[("full", None)]
    b_chain1 = xs[("chain", 1.0)]
    # The budget axis is broken: every cheap structure lives below ~1.1x the
    # complete chain's budget, while FULL sits an order of magnitude out. A
    # single linear axis would leave 90 per cent of the panel empty, so the
    # far point gets its own narrow segment with break marks.
    x_lo_hi = b_chain1 * 1.18
    x_hi_lo, x_hi_hi = b_full * 0.985, b_full * 1.015

    fig = plt.figure(figsize=figsize(165, 156))
    gs = fig.add_gridspec(3, 4, width_ratios=[7, 1.5, 7, 1.5],
                          height_ratios=[10, 10, 8.0],
                          wspace=0.10, hspace=0.92,
                          left=0.085, right=0.985, top=0.865, bottom=0.135)

    for i, m in enumerate((0.8, 0.6)):
        for j, eta in enumerate((1.0, 0.8)):
            axl = fig.add_subplot(gs[j, 2 * i])
            axr = fig.add_subplot(gs[j, 2 * i + 1], sharey=axl)
            style_ax(axl)
            style_ax(axr)
            fam = df[(df.m == m) & (df.eta == eta)]
            pts = {}
            cis = {}
            for (st, phi), b in xs.items():
                sub = fam[fam.structure == st]
                if phi is not None:
                    sub = sub[sub.phi == phi]
                if not len(sub):
                    continue
                v, name = G.twt_best(sub, pools)
                if v is not None:
                    pts[(st, phi)] = (b, v)
                    cis[(st, phi)] = _boot_ci(
                        _instance_values(sub, pools[name]))

            chain_keys = sorted((k for k in pts
                                 if k[0] in ("dedicated", "chain")),
                                key=lambda k: pts[k][0])
            if chain_keys:
                bx = [pts[k][0] for k in chain_keys]
                by = [pts[k][1] for k in chain_keys]
                yerr = np.array(
                    [[by[i] - cis[k][0] for i, k in enumerate(chain_keys)],
                     [cis[k][1] - by[i] for i, k in enumerate(chain_keys)]])
                axl.errorbar(bx, by, yerr=yerr, fmt="none",
                             ecolor=STRUCT_COLOR["chain"], elinewidth=0.7,
                             capsize=1.6, capthick=0.7, alpha=0.45, zorder=2)
                axl.plot(bx, by, "-o", color=STRUCT_COLOR["chain"],
                         linewidth=1.0, markersize=2.6, mec=SURF, mew=0.3,
                         zorder=3)
            if ("generalist", None) in pts:
                b, v = pts[("generalist", None)]
                lo, hi = cis[("generalist", None)]
                # GEN sits at the complete-chain budget; nudge it clear of the
                # chain(1.0) marker so both points and their bars stay readable.
                bd = b + x_lo_hi * 0.02
                axl.errorbar([bd], [v], yerr=[[v - lo], [hi - v]], fmt="none",
                             ecolor=STRUCT_COLOR["generalist"], elinewidth=0.7,
                             capsize=1.6, capthick=0.7, alpha=0.45, zorder=4)
                axl.plot([bd], [v], "D", color=STRUCT_COLOR["generalist"],
                         markersize=3.4, mec=SURF, mew=0.4, zorder=5)
            if ("full", None) in pts:
                b, v = pts[("full", None)]
                lo, hi = cis[("full", None)]
                axr.errorbar([b], [v], yerr=[[v - lo], [hi - v]], fmt="none",
                             ecolor=STRUCT_COLOR["full"], elinewidth=0.7,
                             capsize=1.6, capthick=0.7, alpha=0.45, zorder=4)
                axr.plot([b], [v], "s", color=STRUCT_COLOR["full"],
                         markersize=3.6, mec=SURF, mew=0.4, zorder=5)
            if ("dedicated", None) in pts:
                l0 = pts[("dedicated", None)][1]
                for a in (axl, axr):
                    a.axhline(l0, color=AXIS, linewidth=0.5, linestyle=":",
                              zorder=1)

            axl.set_xlim(-x_lo_hi * 0.06, x_lo_hi)
            axr.set_xlim(x_hi_lo, x_hi_hi)
            axr.set_xticks([b_full])
            axr.set_xticklabels(["%d" % b_full])
            axr.tick_params(labelleft=False, left=False)
            axl.spines["right"].set_visible(False)
            axr.spines["left"].set_visible(False)

            # Diagonal break marks on the facing spines.
            kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=4,
                      linestyle="none", color=AXIS, mec=AXIS, mew=0.6,
                      clip_on=False)
            axl.plot([1, 1], [0, 1], transform=axl.transAxes, **kw)
            axr.plot([0, 0], [0, 1], transform=axr.transAxes, **kw)

            axl.set_title("$m=%.1f$, $\\eta=%.1f$" % (m, eta), fontsize=8.6,
                          loc="center")
            if j == 1:
                axl.set_xlabel("skill-membership budget $B$", x=0.62)
            if i == 0:
                axl.set_ylabel("pooled TWT (weighted units)")
            if m == 0.8:
                axl.text(0.03, 0.06, "no material dividend (2% guard)",
                         transform=axl.transAxes, fontsize=6.8, color=INK)

    # ---- paired-effect row: per-instance dividend vs L0, fixed EDD ------ #
    # Marginal CIs on raw means understate the design signal because the
    # structures share their instances; this row shows the PAIRED effect
    # (L0 minus structure, per instance, same dispatcher) directly.
    edd = df[df.method == "edd"].copy()
    edd["base"] = (edd.campus.astype(str) + "|" + edd["size"].astype(str)
                   + "|" + edd.track.astype(str) + "|"
                   + edd.instance_id.astype(str))
    cats = [("chain", 0.25, "C(.25)"), ("chain", 0.5, "C(.5)"),
            ("chain", 1.0, "C(1.0)"), ("generalist", None, "GEN"),
            ("full", None, "FULL")]
    for i, m in enumerate((0.8, 0.6)):
        axp = fig.add_subplot(gs[2, 2 * i:2 * i + 2])
        style_ax(axp)
        axp.axhline(0.0, color=AXIS, linewidth=0.5, linestyle=":", zorder=1)
        for eta, mk, fill in ((1.0, "o", True), (0.8, "s", False)):
            fam = edd[(edd.m == m) & (edd.eta == eta)]
            l0 = fam[fam.structure == "dedicated"].set_index("base").twt
            xs_, ys_, lo_, hi_ = [], [], [], []
            for k, (st, phi, lab) in enumerate(cats):
                sub = fam[fam.structure == st]
                if phi is not None:
                    sub = sub[sub.phi == phi]
                if not len(sub):
                    continue
                d = (l0 - sub.set_index("base").twt).dropna()
                if not len(d):
                    continue
                lo, hi = _boot_ci(d.to_numpy())
                xs_.append(k + (-0.13 if eta == 1.0 else 0.13))
                ys_.append(float(d.mean()))
                lo_.append(lo)
                hi_.append(hi)
            col = STRUCT_COLOR["chain"] if m == 0.6 else AXIS
            kwm = dict(color=col, markersize=3.4,
                       mfc=(col if fill else SURF), mec=col, mew=0.7)
            axp.errorbar(xs_, ys_,
                         yerr=[np.array(ys_) - np.array(lo_),
                               np.array(hi_) - np.array(ys_)],
                         fmt=mk, elinewidth=0.7, capsize=1.6,
                         capthick=0.7, linestyle="none", **kwm)
        axp.set_xticks(range(len(cats)))
        axp.set_xticklabels([lab for _s, _p, lab in cats])
        axp.set_xlim(-0.6, len(cats) - 0.4)
        axp.set_title("Paired Dividend vs L0, $m=%.1f$ (Fixed EDD)" % m,
                      fontsize=8.6, loc="center", pad=8.0)
        if i == 0:
            axp.set_ylabel("paired $\\Delta$TWT (units)")
        hnd = [Line2D([], [], marker="o", linestyle="none", color=INK,
                      markersize=3.4, label="$\\eta=1.0$"),
               Line2D([], [], marker="s", linestyle="none", color=INK,
                      mfc=SURF, mew=0.7, markersize=3.4,
                      label="$\\eta=0.8$")]
        axp.legend(handles=hnd, frameon=False, fontsize=7.0,
                   loc="upper left", handletextpad=0.3)

    handles = [
        Line2D([], [], color=STRUCT_COLOR["chain"], marker="o", markersize=2.6,
               linewidth=1.0, label="CHAIN($\\varphi$)"),
        Line2D([], [], color=STRUCT_COLOR["generalist"], marker="D",
               markersize=3.4, linestyle="none",
               label="GEN (Membership-Matched)"),
        Line2D([], [], color=STRUCT_COLOR["full"], marker="s", markersize=3.6,
               linestyle="none", label="FULL"),
        Line2D([], [], color=AXIS, linewidth=0.5, linestyle=":",
               label="L0 (Dedicated) Reference"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, fontsize=7.6,
               loc="lower center", bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.5, columnspacing=1.8)
    fig.text(0.5, 0.985,
             "n = 763 instances per point, pooled over 4 campuses;\n"
             "bars are 95% instance-bootstrap CIs (2000 resamples).",
             ha="center", va="top", fontsize=8.6, color=INK)
    _save(fig, "fig3_design_curve")


def fig6_frameu():
    _rc()
    e3 = _read("e3")
    if e3 is None:
        return
    df = _expand_l0(e3)
    mlp, attn = _policy_pools(e3)
    import analysis.gates as G
    import matplotlib.ticker as mticker
    fig, axes = plt.subplots(1, 2, figsize=figsize(165, 90), sharey=True)
    for ax, eta in zip(axes, (1.0, 0.8)):
        style_ax(ax)
        fam = df[df.eta == eta]
        for st in ("dedicated", "chain", "full"):
            sub = fam[fam.structure == st]
            us, ys, los, his = [], [], [], []
            for u in sorted(sub.u_target.dropna().unique()):
                cell = sub[sub.u_target == u]
                pools = {r: [r] for r in
                         ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc",
                          "atc_eta"]}
                if mlp:
                    pools["policy_mlp"] = mlp
                v, name = G.twt_best(cell, pools)
                if v is None:
                    continue
                ci = _boot_ci(_instance_values(cell, pools[name]))
                us.append(u)
                ys.append(v)
                los.append(ci[0])
                his.append(ci[1])
            if not us:
                continue
            col = STRUCT_COLOR[st]
            # 95% instance-bootstrap CI ribbon (light; 120 instances per point,
            # 30 per campus over the four campuses).
            ax.fill_between(us, los, his, color=col, alpha=0.13, linewidth=0,
                            zorder=1)
            ax.plot(us, ys, color=col, linestyle=STRUCT_LINE[st],
                    marker=STRUCT_MARKER[st], linewidth=1.0, markersize=2.8,
                    mec=SURF, mew=0.3, label=STRUCT_PRETTY[st], zorder=3)
        # A plain linear axis with round-number ticks: every cell here sits far
        # above any symlog threshold, so real values read cleanly.
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6,
                                                        steps=[1, 2, 5, 10]))
        ax.set_xlabel("offered-load utilisation $U$")
        ax.set_title("$\\eta=%.1f$" % eta, fontsize=8.6)
    axes[0].set_ylabel("best-method TWT (weighted units)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.subplots_adjust(bottom=0.28)
    fig.text(0.5, 0.02,
             "n = 120 instances per point (30 per campus $\\times$ 4 "
             "campuses);\nribbons are 95% instance-bootstrap CIs "
             "(2000 resamples).",
             ha="center", va="bottom", fontsize=8.0, color=INK)
    _save(fig, "fig6_frameu")


def fig8_training():
    """Per-seed primary development signal (the checkpoint-selection signal)
    with the two default-capacity monitors overlaid as cross-seed means and
    each seed's checkpoint marked at its primary minimum. The point the
    figure must show: the tight-capacity primary signal varies enough to
    discriminate checkpoints, whereas the L0 monitor plateaus."""
    _rc()
    import glob
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=figsize(165, 88), sharey=True)
    for ax, arch, color in ((axes[0], "mlp", CMAP["policy"]),
                            (axes[1], "attn", CMAP["policy_attn"])):
        style_ax(ax)
        n = 0
        mons = {"dev_l0": [], "dev_full": []}
        grid = None
        for d in sorted(glob.glob(str(RES / "train" / (arch + "_seed*")))):
            p = Path(d) / "curves.csv"
            if not p.exists():
                continue
            c = pd.read_csv(p)
            if not len(c):
                continue
            cp = c[c["dev_primary"].notna()]
            # per-seed primary signal (thin) + its checkpoint (argmin)
            ax.plot(cp["update"], cp["dev_primary"], color=color, alpha=0.30,
                    linewidth=0.6, zorder=2)
            imin = cp["dev_primary"].values.argmin()
            ax.scatter(cp["update"].values[imin],
                       cp["dev_primary"].values[imin], s=8, color=color,
                       edgecolor=SURF, linewidth=0.4, zorder=5)
            for k in mons:
                mons[k].append(cp.set_index("update")[k])
            grid = cp["update"].values
            n += 1
        # cross-seed monitor means: the L0 monitor is the plateau, drawn as a
        # single muted line; the FULL monitor as a second reference.
        for k, col, ls, lab in (("dev_l0", MUTE, "--", "L0 monitor (plateau)"),
                                ("dev_full", STRUCT_COLOR["full"], ":",
                                 "FULL monitor")):
            if mons[k]:
                mean = pd.concat(mons[k], axis=1).mean(axis=1)
                ax.plot(mean.index, mean.values, color=col, linestyle=ls,
                        linewidth=1.0, alpha=0.9, zorder=3)
        ax.set_title("Pair-%s (%d Seeds)"
                     % ({"mlp": "MLP", "attn": "Attn"}[arch], n),
                     fontsize=8.6)
        ax.set_xlabel("PPO update")
    axes[0].set_ylabel("development TWT")
    handles = [
        Line2D([], [], color=CMAP["policy"], linewidth=0.9,
               label="primary signal (per seed)"),
        Line2D([], [], color=INK, marker="o", linestyle="none", markersize=3,
               mec=SURF, mew=0.4, label="checkpoint (primary min)"),
        Line2D([], [], color=MUTE, linestyle="--", linewidth=1.0,
               label="L0 monitor (plateau)"),
        Line2D([], [], color=STRUCT_COLOR["full"], linestyle=":",
               linewidth=1.0, label="FULL monitor"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, fontsize=7.4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.5, columnspacing=1.6)
    fig.subplots_adjust(bottom=0.24)
    _save(fig, "fig8_training")


def build_all(numbers=None):
    fig1_pipeline()
    fig2_ladder()
    fig3_design_curve()
    fig6_frameu()
    fig8_training()
    # fig4/fig5 need policy rows; fig7/fig9 build from rules alone and
    # refresh automatically when policy rows land. Each tries independently.
    build_exhibits_final()


def build_exhibits_final():
    for fn in (fig4_gradient, fig5_decision_map, fig7_campus2, fig9_tau):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print("  (%s pending: %s)" % (fn.__name__, e))


def fig4_gradient():
    _rc()
    tier1 = _read("tier1")
    mlp, attn = _policy_pools(tier1)
    pool = mlp or attn
    if tier1 is None or pool is None:
        raise RuntimeError("policy rows not present yet")
    import analysis.gates as G
    df = _expand_l0(tier1)
    ranked = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
    fig, axes = plt.subplots(1, 2, figsize=figsize(165, 82), sharey=True)
    structs = [("dedicated", "L0"), ("chain", "CHAIN"), ("full", "FULL")]
    for ax, m in zip(axes, (0.8, 0.6)):
        style_ax(ax)
        for eta, ls in ((1.0, "-"), (0.8, "--")):
            fam = df[(df.m == m) & (df.eta == eta)]
            xs, gaps, spreads = [], [], []
            for k, (st, lbl) in enumerate(structs):
                sub = fam[fam.structure == st]
                if not len(sub):
                    continue
                best_rule = min((float(sub[sub.method == r].twt.mean())
                                 for r in ranked if len(sub[sub.method == r])),
                                default=None)
                seed_means = [float(sub[sub.method == s].twt.mean())
                              for s in pool if len(sub[sub.method == s])]
                if best_rule is None or not seed_means:
                    continue
                xs.append(k)
                gaps.append(np.mean(seed_means) - best_rule)
                spreads.append(np.std(seed_means))
            gaps = np.array(gaps)
            spreads = np.array(spreads)
            ax.fill_between(xs, gaps - spreads, gaps + spreads,
                            color=CMAP["policy"], alpha=0.12, linewidth=0)
            ax.plot(xs, gaps, ls, marker="o", color=CMAP["policy"],
                    linewidth=1.0, markersize=2.8,
                    label="$\\eta=%.1f$" % eta)
        ax.axhline(0, color=AXIS, linewidth=0.5)
        ax.set_xticks(range(len(structs)))
        ax.set_xticklabels([lbl for _s, lbl in structs])
        ax.set_title("$m=%.1f$" % m, fontsize=8.6)
    axes[0].set_ylabel("policy $-$ best rule (TWT, weighted units)")
    axes[0].legend(frameon=False)
    _save(fig, "fig4_gradient")


def fig5_decision_map():
    """Which method family wins each cell, at both efficiency settings.

    Both eta rows are shown: the learned policy's only strict wins occur at
    eta = 1, and hiding that row would misrepresent the map.
    """
    _rc()
    tier1 = _read("tier1")
    mlp, attn = _policy_pools(tier1)
    if tier1 is None or (mlp is None and attn is None):
        raise RuntimeError("policy rows not present yet")
    import analysis.gates as G
    df = _expand_l0(tier1)
    classic_rules = ["edd", "pfifo", "atc", "wspt", "mor"]
    flex_rules = ["lfj_atc", "atc_eta"]
    pool = mlp or attn
    campuses = [5, 9, 10, 12]
    structs = ["dedicated", "chain", "full"]
    ms = [1.0, 0.8, 0.6]
    etas = [1.0, 0.8]
    fig, axes = plt.subplots(2, 4, figsize=figsize(165, 98), sharey=True,
                             sharex=True)
    colors = {"classic": CMAP["edd"], "flex": CMAP["lfj_atc"],
              "policy": CMAP["policy"], "tie": "#d9d8d1"}
    for r, eta in enumerate(etas):
        d_eta = df[df.eta == eta]
        for k, c in enumerate(campuses):
            ax = axes[r][k]
            style_ax(ax)
            sub_c = d_eta[d_eta.campus == c]
            for i, st in enumerate(structs):
                for j, m in enumerate(ms):
                    cell = sub_c[(sub_c.structure == st) & (sub_c.m == m)]
                    if not len(cell):
                        continue
                    vals = {}
                    vals["classic"] = min(
                        float(cell[cell.method == x].twt.mean())
                        for x in classic_rules)
                    vals["flex"] = min(
                        float(cell[cell.method == x].twt.mean())
                        for x in flex_rules)
                    pol = cell[cell.method.isin(pool)].copy()
                    pol["cfg"] = G.config_key(pol)
                    vals["policy"] = float(
                        pol.groupby("cfg").twt.mean().mean())
                    best = min(vals.values())
                    winners = [x for x, v in vals.items() if v <= best + 1.0]
                    color = (colors["tie"] if len(winners) > 1
                             else colors[winners[0]])
                    ax.add_patch(Rectangle((i, j), 0.94, 0.94,
                                           facecolor=color, edgecolor="white",
                                           linewidth=0.8))
            ax.set_xlim(0, len(structs))
            ax.set_ylim(0, len(ms))
            ax.set_xticks([x + 0.5 for x in range(len(structs))])
            ax.set_xticklabels(["L0", "CHAIN", "FULL"])
            ax.set_yticks([x + 0.5 for x in range(len(ms))])
            ax.set_yticklabels(["$m=1$", "$m=.8$", "$m=.6$"])
            if r == 0:
                ax.set_title("Campus %d" % c, fontsize=8.6)
            if k == 0:
                ax.text(-0.42, 0.5, "$\\eta = %.1f$" % eta, rotation=90,
                        transform=ax.transAxes, va="center", ha="center",
                        fontsize=8.6, color=INK)
    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=5,
                      color=v, label=k) for k, v in
               (("Best Classical Rule", colors["classic"]),
                ("Best Flexibility-Aware Rule", colors["flex"]),
                ("Learned Policy", colors["policy"]),
                ("Tie (within $\\varepsilon = 1$)", colors["tie"]))]
    fig.subplots_adjust(bottom=0.155, top=0.93, left=0.075, right=0.985,
                        hspace=0.22, wspace=0.20)
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4,
               fontsize=7.8, bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.5, columnspacing=2.0)
    _save(fig, "fig5_decision_map")


def fig7_campus2():
    _rc()
    e4 = _read("e4")
    if e4 is None:
        raise RuntimeError("e4 rows not present yet")
    import analysis.gates as G
    df = _expand_l0(e4)
    df = df[(df.campus == 2) & (df.eta == 0.8)]
    if not len(df):
        raise RuntimeError("campus-2 rows missing")
    ranked = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]
    fig, ax = plt.subplots(figsize=figsize(165, 88))
    style_ax(ax)
    structs = ["dedicated", "chain", "full"]
    for m, ls in ((1.0, "-"), (0.8, "--"), (0.6, ":")):
        xs, ys = [], []
        base = None
        for k, st in enumerate(structs):
            cell = df[(df.structure == st) & (df.m == m)]
            if not len(cell):
                continue
            best = min(float(cell[cell.method == r].twt.mean())
                       for r in ranked)
            if st == "dedicated":
                base = best
            if base:
                xs.append(k)
                ys.append(best / base)
        ax.plot(xs, ys, ls, marker="o", color=STRUCT_COLOR["chain"],
                linewidth=1.0, markersize=2.8, label="$m=%.1f$" % m)
    ax.axhline(1.0, color=AXIS, linewidth=0.5)
    ax.set_xticks(range(len(structs)))
    ax.set_xticklabels(["L0", "CHAIN(1.0)", "FULL"])
    ax.set_ylabel("best-method TWT relative to L0")
    ax.legend(frameon=False)
    _save(fig, "fig7_campus2")


def fig9_tau():
    _rc()
    e5 = _read("e5")
    if e5 is None:
        raise RuntimeError("e5 rows not present yet")
    from scipy.stats import kendalltau
    import analysis.gates as G
    # Ranked-rule set only, matching the robustness prose in build_all: the
    # policy pool's seed noise would conflate "is the rule ordering stable"
    # with "does the policy's position jitter", and the policy's sensitivity
    # is covered separately by the cap256 variant.
    methods = ["edd", "wspt", "atc", "pfifo", "mor", "lfj_atc", "atc_eta"]

    def ranking(sub):
        # Raw means, not argsort ranks: tau-b then treats the exact
        # EDD/pFIFO tie as a tie instead of an arbitrary version-dependent
        # ordering (matches the robustness prose in build_all).
        vals = []
        for r in methods:
            s = sub[sub.method == r]
            if len(s):
                vals.append(float(s.twt.mean()))
            else:
                return None
        return np.asarray(vals)

    base = ranking(e5[e5.variant == "base"])
    variants = [v for v in e5.variant.unique()
                if v not in ("base", "cap256", "full_base",
                             "tbrand_full", "tbflex_full")]
    taus = []
    labels = []
    for v in sorted(variants):
        r = ranking(e5[e5.variant == v])
        if r is None or base is None:
            continue
        t, _p = kendalltau(base, r)
        taus.append(t)
        labels.append(v)
    # TB-ablation pair on the FULL cell references full_base
    fb = ranking(e5[e5.variant == "full_base"])
    for v in ("tbrand_full", "tbflex_full"):
        r = ranking(e5[e5.variant == v])
        if r is not None and fb is not None:
            t, _p = kendalltau(fb, r)
            taus.append(t)
            labels.append(v)
    # A one-row heatmap wasted the panel and, worse, clipped the single
    # negative value (the random tie-break) to white, so the most important
    # ablation looked like missing data. A ranked bar chart shows the whole
    # range, negatives included.
    PRETTY_V = {
        "eta075": "Efficiency $\\eta = 0.75$",
        "eta090": "Efficiency $\\eta = 0.90$",
        "sla050": "SLA Windows $\\times 0.5$",
        "sla150": "SLA Windows $\\times 1.5$",
        "crew075": "Crews $\\times 0.75$",
        "crew125": "Crews $\\times 1.25$",
        "w27931": "Weights (2,7,9,31)",
        "w4321": "Weights (4,3,2,1)",
        "perm1": "Chain Permutation 1",
        "perm2": "Chain Permutation 2",
        "perm3": "Chain Permutation 3",
        "tbflex": "Tie-Break: Most-Flexible First",
        "tbrand": "Tie-Break: Random",
        "tbflex_full": "Tie-Break: Most-Flexible (FULL)",
        "tbrand_full": "Tie-Break: Random (FULL)",
    }
    pairs = sorted(zip(taus, labels), key=lambda x: x[0])
    vals = [p[0] for p in pairs]
    names = [PRETTY_V.get(p[1], p[1]) for p in pairs]
    is_tb = [p[1].startswith("tb") for p in pairs]

    fig, ax = plt.subplots(figsize=figsize(165, 116))
    style_ax(ax)
    y = np.arange(len(vals))
    cols = [CMAP["wspt"] if tb else CMAP["policy"] for tb in is_tb]
    ax.barh(y, vals, height=0.62, color=cols, edgecolor="none", zorder=3)
    ax.axvline(1.0, color=AXIS, linewidth=0.6, linestyle=":", zorder=2)
    for i, v in enumerate(vals):
        ax.text(v + (0.025 if v >= 0 else -0.025), i, "%.2f" % v,
                va="center", ha="left" if v >= 0 else "right",
                fontsize=7.2, color=INK2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.6)
    # Left limit hugs the content: at 0 when every tau is non-negative (the
    # tick labels then sit against the bar baseline instead of across a
    # dead band), widening only if a rerun ever yields a negative value.
    vmin = min(vals)
    lo = 0.0 if vmin >= 0 else vmin - 0.10
    ax.set_xlim(lo, 1.18)
    if lo < 0:
        ax.axvline(0.0, color=AXIS, linewidth=0.6, zorder=2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Kendall $\\tau_b$ of the method ranking against the "
                  "locked default")
    ax.text(1.0, len(vals) - 0.35, "identical ranking", fontsize=7.0,
            color=INK, ha="right", va="bottom")
    handles = [
        Line2D([], [], color=CMAP["policy"], linewidth=4,
               label="Environment and Instance Perturbations"),
        Line2D([], [], color=CMAP["wspt"], linewidth=4,
               label="Technician Tie-Break Ablation"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.6, loc="lower right",
              handlelength=1.4)
    _save(fig, "fig9_tau")


if __name__ == "__main__":
    build_all()
