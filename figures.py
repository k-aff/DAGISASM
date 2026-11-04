#!/usr/bin/env python3
r"""Chart figures for the COS700 report.

    python figures.py                  # every figure
    python figures.py contrast forest  # only some

Writes a PDF and a PNG of each figure to ./figures/. Report figure numbers:

    ita          Figure 2  ITA categories and the effect of lighting
    composition  Figure 3  racial make-up of the 140k corpus
    cvar         Figure 5  average loss versus CVaR (simulated losses)
    contrast     Figure 7  FPR by race under the two test conditions
    forest       Figure 8  gaps between groups, baseline vs fairness-aware
    confusion    Figure 9  cross-source/cross-generator confusion matrices

Figures 1, 4 and 6 are TikZ drawings; their sources are in figures/tikz/.

The DATA section holds the results as counts (errors out of n), copied from
runs/models140k/swin_aug/*/eval/ and */eval_cross/. Where the report gives a
confidence interval it is used as is; the other intervals (in-distribution
rates and the gaps in Figure 8) are bootstrapped from the counts with 2 000
resamples.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

OUT = Path("figures")
RACES = ["Black", "Indian", "White"]

# =============================================================================
# DATA
# =============================================================================
# FairFace labels over all 140 000 images
COMPOSITION = {"White": 88496, "Other": 41165, "Black": 5975, "Indian": 4364}
CORPUS_TOTAL = 140000

# in-distribution test, baseline: 400 real + 400 fake per race
INDIST_FP = {"Black": (0, 400), "Indian": (0, 400), "White": (2, 400)}   # FPR
INDIST_FN = {"Black": (2, 400), "Indian": (1, 400), "White": (3, 400)}   # FNR

# cross-source/cross-generator test: (errors, n)
#   IMDb holds real faces only (false positives), SD fake faces only (misses)
CROSS = {
    "baseline": {
        "fp": {"Black": (620, 1457), "Indian": (109, 197), "White": (1757, 4845)},
        "fn": {"Black": (1264, 1709), "Indian": (492, 891), "White": (7102, 11577)},
    },
    "fair": {
        "fp": {"Black": (418, 1457), "Indian": (71, 197), "White": (1424, 4845)},
        "fn": {"Black": (1153, 1709), "Indian": (370, 891), "White": (6017, 11577)},
    },
}
# 95% intervals of the baseline IMDb FPR, from eval_cross/imdb/per_group_error.csv
CROSS_FPR_CI_REPORTED = {"Black": (0.401, 0.449), "Indian": (0.482, 0.619),
                         "White": (0.349, 0.377)}

N_BOOT, SEED = 2000, 0

# =============================================================================
# STYLE
# =============================================================================
SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, PURPLE = "#77b2fb", "#815F9D"          # categorical slots 1 and 2
GREY_BAR = "#c3c2b7"                         # "left out" / reference
PURPLE_RAMP = ["#ffffff", "#e5d9f8", "#cbb5ef", "#b290e4", "#9b69da",
               "#8d5acb", "#7d4fb5", "#6e43a2", "#5f398d"]
TINT = {"neutral": "#cea2e2"}                # correct cells in Figure 9

WIDTH = 5.4                                  # inches, the text width of the report

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "pdf.fonttype": 42, "font.size": 10,
    "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE,
})


# =============================================================================
# helpers
# =============================================================================
def num(n: int) -> str:
    """88496 -> 88 496 with a thin space, as in the report."""
    return "$" + f"{n:,}".replace(",", r"\,") + "$"


def pct(x: float, d: int = 0) -> str:
    return "$" + f"{x * 100:.{d}f}" + r"\%$"


def rate(kn) -> float:
    k, n = kn
    return k / n


def boot_rate(kn, rng) -> tuple[float, float]:
    k, n = kn
    draws = rng.binomial(n, k / n, N_BOOT) / n
    return tuple(np.percentile(draws, [2.5, 97.5]))


def boot_gap(cells: dict, rng) -> tuple[float, float]:
    draws = np.vstack([rng.binomial(n, k / n, N_BOOT) / n for k, n in cells.values()])
    gaps = draws.max(axis=0) - draws.min(axis=0)
    return tuple(np.percentile(gaps, [2.5, 97.5]))


def gap(cells: dict) -> float:
    r = [rate(v) for v in cells.values()]
    return max(r) - min(r)


def style_axes(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=4)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def bar_path(x0, x1, yc, h, rx, ry):
    """Horizontal bar: square at the baseline x0, rounded data end at x1."""
    k = 0.5523
    y0, y1 = yc - h / 2, yc + h / 2
    rx = min(rx, (x1 - x0) / 2)
    v = [(x0, y0), (x1 - rx, y0),
         (x1 - rx + k * rx, y0), (x1, y0 + ry - k * ry), (x1, y0 + ry),
         (x1, y1 - ry),
         (x1, y1 - ry + k * ry), (x1 - rx + k * rx, y1), (x1 - rx, y1),
         (x0, y1), (x0, y0)]
    c = [MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
         MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
         MPath.LINETO, MPath.CLOSEPOLY]
    return MPath(v, c)


def units_per_pt(fig, ax):
    fig.canvas.draw()
    a = ax.transData.transform([(0, 0), (1, 1)])
    ppt = fig.dpi / 72.0
    return 1 / (abs(a[1][0] - a[0][0]) / ppt), 1 / (abs(a[1][1] - a[0][1]) / ppt)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {OUT / name}.pdf / .png")


# =============================================================================
# Figure 3: corpus composition
# =============================================================================
def fig_composition():
    groups = list(COMPOSITION)
    counts = list(COMPOSITION.values())
    used = {"White", "Black", "Indian"}

    fig = plt.figure(figsize=(WIDTH, 2.5))
    ax = fig.add_axes([0.22, 0.18, 0.56, 0.74])
    ax.set_xlim(0, 90000)
    ax.set_ylim(len(groups) - 0.35, -0.90)
    xpt, ypt = units_per_pt(fig, ax)
    bh = 15 * ypt
    for i, (g, n) in enumerate(zip(groups, counts)):
        ax.add_patch(PathPatch(bar_path(0, n, i, bh, 3 * xpt, 3 * ypt),
                               fc=BLUE if g in used else GREY_BAR, ec="none", zorder=3))
        ax.text(n + 1200, i, f"{num(n)}  ({pct(n / CORPUS_TOTAL, 1)})",
                va="center", ha="left", color=INK, fontsize=11, zorder=4)

    ax.axvline(COMPOSITION["Indian"], color=INK2, lw=0.8, zorder=2)

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, color=INK2, fontsize=11)
    ax.set_xticks([0, 20000, 40000, 60000, 80000])
    ax.set_xticklabels(["0", "20K", "40K", "60K", "80K"])
    style_axes(ax, grid_axis="x")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", pad=6)
    ax.set_xlabel("Number of images", color=INK2, fontsize=10, labelpad=4)
    save(fig, "fig_composition")


# =============================================================================
# Figure 7: FPR by race, two test conditions
# =============================================================================
def fig_contrast():
    rng = np.random.default_rng(SEED)
    fig = plt.figure(figsize=(WIDTH, 2.9))
    ax = fig.add_axes([0.12, 0.13, 0.84, 0.70])
    x = np.arange(len(RACES))
    off = 0.13

    series = [
        ("In-distribution test (FFHQ real faces)", BLUE, -off,
         [rate(INDIST_FP[r]) for r in RACES], [boot_rate(INDIST_FP[r], rng) for r in RACES]),
        ("Cross-source/cross-generator test (IMDb real faces)", PURPLE, +off,
         [rate(CROSS["baseline"]["fp"][r]) for r in RACES],
         [CROSS_FPR_CI_REPORTED[r] for r in RACES]),
    ]
    for label, col, dx, vals, cis in series:
        for xi, v, (lo, hi) in zip(x, vals, cis):
            ax.plot([xi + dx, xi + dx], [lo, hi], color=col, lw=1.6,
                    solid_capstyle="round", zorder=3)
        ax.plot(x + dx, vals, "o", ms=7, color=col, mec=SURFACE, mew=1.5,
                zorder=4, label=label)
    # value labels on the second test only
    for xi, v, (lo, hi) in zip(x, series[1][3], series[1][4]):
        ax.text(xi + off + 0.07, v, pct(v), va="center", ha="left", fontsize=10, color=INK)
    ax.text(x[1] - off, 0.04, r"0 to 0.5$\%$ for every group", ha="center", va="bottom",
            fontsize=8.5, color=INK2)

    ax.set_xticks(x)
    ax.set_xticklabels(RACES, color=INK2, fontsize=10.5)
    ax.set_xlim(-0.6, len(RACES) - 0.4)
    ax.set_ylim(-0.012, 0.66)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0", "20%", "40%", "60%"])
    ax.set_ylabel("False-positive rate", color=INK2, fontsize=10)
    style_axes(ax, "y")
    ax.spines["left"].set_visible(False)
    leg = fig.legend(loc="upper left", bbox_to_anchor=(0.10, 1.0), ncol=1, frameon=False,
                     fontsize=9, handletextpad=0.4, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, "fig_contrast")


# =============================================================================
# Figure 8: gaps between groups, baseline vs fair
# =============================================================================
def fig_forest():
    rng = np.random.default_rng(SEED)
    rows = [("False-positive gap\n(IMDb real faces)", "fp"),
            ("Missed-fake gap\n(Stable Diffusion fakes)", "fn")]
    models = [("Baseline", "baseline", MUTED, +0.14), ("Fairness-aware", "fair", BLUE, -0.14)]

    fig = plt.figure(figsize=(WIDTH, 2.4))
    ax = fig.add_axes([0.27, 0.17, 0.66, 0.66])
    for yi, (_, key) in enumerate(rows):
        y = len(rows) - 1 - yi
        for name, m, col, dy in models:
            cells = CROSS[m][key]
            g = gap(cells)
            lo, hi = boot_gap(cells, rng)
            ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=1.6,
                    solid_capstyle="round", zorder=3)
            ax.plot([g], [y + dy], "o", ms=7, color=col, mec=SURFACE, mew=1.5, zorder=4,
                    label=name if yi == 0 else None)
            ax.text(hi + 0.008, y + dy, f"${g:.3f}$", va="center", ha="left",
                    fontsize=9.5, color=INK)

    ax.axvline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], color=INK2, fontsize=10)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlim(-0.005, 0.36)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["0", "0.1", "0.2", "0.3"])
    ax.set_xlabel("Gap between highest and lowest race (rate)", color=INK2, fontsize=10)
    style_axes(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    leg = fig.legend(loc="upper left", bbox_to_anchor=(0.26, 1.0), ncol=2, frameon=False,
                     fontsize=9.5, handletextpad=0.4, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, "fig_forest")


# =============================================================================
# Figure 9: cross-source/cross-generator confusion matrices
# =============================================================================
def fig_confusion():
    """Correct cells are neutral; the two error cells are shaded by their rate."""
    cmap = LinearSegmentedColormap.from_list("blue", PURPLE_RAMP)
    models = [("Baseline", "baseline"), ("Fairness-aware", "fair")]

    fig = plt.figure(figsize=(WIDTH, 3.7))
    L, R, B, T = 0.20, 0.99, 0.03, 0.90
    cw, ch = (R - L) / 3, (T - B) / 2
    for ri, (mname, m) in enumerate(models):
        for ci, race in enumerate(RACES):
            fp, n_real = CROSS[m]["fp"][race]
            fn, n_fake = CROSS[m]["fn"][race]
            mat = np.array([[n_real - fp, fp], [fn, n_fake - fn]])
            share = mat / mat.sum(axis=1, keepdims=True)
            is_err = np.array([[False, True], [True, False]])

            ax = fig.add_axes([L + ci * cw + 0.012, T - (ri + 1) * ch + 0.035,
                               cw - 0.024, ch - 0.07])
            ax.set_xlim(0, 2)
            ax.set_ylim(2, 0)
            ax.axis("off")
            g = 0.03                                  # surface gap between cells
            for i in range(2):
                for j in range(2):
                    v = share[i, j]
                    fc = cmap(v) if is_err[i, j] else TINT["neutral"]
                    tc = SURFACE if (is_err[i, j] and v > 0.45) else INK
                    ax.add_patch(Rectangle((j + g, i + g), 1 - 2 * g, 1 - 2 * g,
                                           fc=fc, ec="none"))
                    ax.text(j + 0.5, i + 0.40, num(int(mat[i, j])), ha="center",
                            va="center", fontsize=9, color=tc)
                    ax.text(j + 0.5, i + 0.68, pct(v), ha="center", va="center",
                            fontsize=8, color=tc)
            if ri == 0:
                ax.text(1, -0.30, race, ha="center", va="bottom", fontsize=10.5, color=INK)
                for j, lab in enumerate(["pred. real", "pred. fake"]):
                    ax.text(j + 0.5, -0.04, lab, ha="center", va="bottom",
                            fontsize=7.5, color=MUTED)
            if ci == 0:
                for i, lab in enumerate(["real\n(IMDb)", "fake\n(SD)"]):
                    ax.text(-0.08, i + 0.5, lab, ha="right", va="center", fontsize=7.5,
                            color=MUTED, linespacing=1.1)
        fig.text(0.015, T - (ri + 0.5) * ch, mname, ha="left", va="center",
                 fontsize=10, color=INK)
    save(fig, "fig_confusion")


# =============================================================================
# Figure 2: ITA scale and why lighting moves it
# =============================================================================
ITA_BANDS = [(-60, -30, "dark"), (-30, 10, "brown"), (10, 28, "tan"),
             (28, 41, "intermediate"), (41, 55, "light"), (55, 80, "very light")]
# dark -> very light, taken from PURPLE_RAMP
ITA_RAMP = [PURPLE_RAMP[8], PURPLE_RAMP[6], PURPLE_RAMP[4], PURPLE_RAMP[3], PURPLE_RAMP[2], PURPLE_RAMP[1]]


def ita(L, b):
    return np.degrees(np.arctan((L - 50.0) / b))


def fig_ita():
    fig = plt.figure(figsize=(WIDTH, 4.2))
    # top panel: the six ITA categories as a scale
    ax = fig.add_axes([0.10, 0.76, 0.86, 0.12])
    for (lo, hi, name), col in zip(ITA_BANDS, ITA_RAMP):
        ax.add_patch(Rectangle((lo, 0), hi - lo, 1, fc=col, ec=SURFACE, lw=1.5))
        tc = SURFACE if lo < 28 else INK
        lab = "inter-\nmediate" if name == "intermediate" else name
        ax.text((lo + hi) / 2, 0.5, lab, ha="center", va="center", fontsize=8, color=tc,
                linespacing=1.0)
    for t in (-30, 10, 28, 41, 55):
        ax.text(t, -0.18, f"${t}^\\circ$", ha="center", va="top", fontsize=8.5, color=INK2)
    ax.set_xlim(-60, 80)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.text(0.10, 0.955, "(a) ITA categories (Chardon et al.)", fontsize=10, color=INK)

    # bottom panel: ITA as the brightness L* of the same skin changes
    ax2 = fig.add_axes([0.10, 0.11, 0.86, 0.50])
    for (lo, hi, name), col in zip(ITA_BANDS, ITA_RAMP):
        ax2.axhspan(lo, hi, color=col, alpha=0.28, lw=0, zorder=0)
        ax2.text(98.5, (max(lo, -60) + min(hi, 80)) / 2, name, ha="right", va="center",
                 fontsize=8, color=INK2)
    L = np.linspace(25, 85, 300)
    for b, ls in ((12, (0, (4, 2))), (18, "-"), (24, (0, (1, 1.5)))):
        ax2.plot(L, ita(L, b), color=PURPLE, lw=1.6, ls=ls, zorder=3, label=f"$b^* = {b}$")
    # one skin colour seen in good light and in shadow
    L1, L2, b0 = 70, 45, 18
    for Lx in (L1, L2):
        ax2.plot([Lx], [ita(Lx, b0)], "o", ms=7, color=BLUE, mec=SURFACE, mew=1.5, zorder=4)
    ax2.annotate("", xy=(L2 + 1.0, ita(L2, b0) + 3), xytext=(L1 - 1.0, ita(L1, b0) - 3),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2), zorder=4)
    ax2.text(L1 - 2.2, ita(L1, b0) + 1.5, "A", fontsize=9, color=INK, ha="right", va="center")
    ax2.text(L2 - 2.2, ita(L2, b0), "B", fontsize=9, color=INK, ha="right", va="center")
    ax2.text(50, -24, f"Same skin colour ($b^* = {b0}$)\n"
             f"A: good light, $L^* = {L1}$, ITA $= {ita(L1, b0):.0f}^\\circ$ (light)\n"
             f"B: in shadow, $L^* = {L2}$, ITA $= {ita(L2, b0):.0f}^\\circ$ (brown)",
             fontsize=8, color=INK, ha="left", va="top", linespacing=1.3)
    ax2.set_xlim(25, 100)
    ax2.set_ylim(-60, 80)
    ax2.set_xticks([30, 40, 50, 60, 70, 80])
    ax2.set_yticks([-60, -30, 0, 30, 60])
    ax2.set_yticklabels([f"${v}^\\circ$" for v in (-60, -30, 0, 30, 60)])
    ax2.set_xlabel("Lightness $L^*$ of the measured pixels", color=INK2, fontsize=10)
    ax2.set_ylabel("ITA", color=INK2, fontsize=10)
    style_axes(ax2, "y")
    leg = ax2.legend(loc="upper left", frameon=False, fontsize=8.5, handlelength=2.2)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.10, 0.665, "(b) ITA changes when only the brightness changes", fontsize=10,
             color=INK)
    save(fig, "fig_ita")


# =============================================================================
# Figure 5: average loss versus CVaR, with simulated losses
# =============================================================================
def fig_cvar():
    rng = np.random.default_rng(SEED)
    losses = np.concatenate([rng.gamma(2.0, 0.12, 900), rng.gamma(6.0, 0.18, 100)])
    alpha = 0.5
    rho = np.quantile(losses, 1 - alpha)
    mean = losses.mean()
    cvar = losses[losses >= rho].mean()

    fig = plt.figure(figsize=(WIDTH, 2.6))
    ax = fig.add_axes([0.10, 0.18, 0.87, 0.70])
    bins = np.linspace(0, 2.6, 53)
    h, e = np.histogram(losses, bins)
    for k in range(len(h)):
        c = PURPLE if e[k] >= rho else GREY_BAR
        ax.bar(e[k], h[k], width=e[1] - e[0], align="edge", color=c, ec=SURFACE, lw=0.6,
               zorder=3)
    top = h.max() * 1.38
    for x, col, ls in ((mean, INK2, (0, (3, 2))), (rho, INK2, (0, (1, 1.5))), (cvar, PURPLE, "-")):
        ax.axvline(x, color=col, lw=1.0, ls=ls, zorder=4)
    ax.text(mean + 0.03, top * 0.97, f"average\n{mean:.2f}", ha="left", va="top",
            fontsize=8.5, color=INK2, linespacing=1.1)
    ax.text(rho - 0.03, top * 0.97, f"$\\rho$\n{rho:.2f}", ha="right", va="top",
            fontsize=8.5, color=INK2, linespacing=1.1)
    ax.text(cvar + 0.03, top * 0.76, f"CVaR$_{{0.5}}$\n{cvar:.2f}", ha="left", va="top",
            fontsize=8.5, color=PURPLE, linespacing=1.1)
    ax.text(1.45, top * 0.40, "CVaR$_{0.5}$ is the average of the\nworst 50% of the losses (purple)", ha="left",
            va="center", fontsize=8.5, color=INK, linespacing=1.15)
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, top)
    ax.set_xlabel("Loss of one training image (simulated values)", color=INK2, fontsize=9.5)
    ax.set_ylabel("Number of images", color=INK2, fontsize=9.5)
    style_axes(ax, "y")
    save(fig, "fig_cvar")


# =============================================================================
FIGURES = {
    "ita": fig_ita,
    "cvar": fig_cvar,
    "composition": fig_composition,
    "contrast": fig_contrast,
    "forest": fig_forest,
    "confusion": fig_confusion,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    bad = [w for w in wanted if w not in FIGURES]
    if bad:
        raise SystemExit(f"Unknown figure(s): {bad}. Choose from {list(FIGURES)}")
    for w in wanted:
        print(w)
        FIGURES[w]()