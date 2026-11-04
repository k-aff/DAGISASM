#!/usr/bin/env python3
r"""All figures for the COS700 report, in one script.

    python figures.py                 # make every figure
    python figures.py contrast forest # make only some

Writes PDF (for LaTeX) and PNG (for slides) into ./figures/. In report.tex
each figure is included as  \includegraphics[width=\textwidth]{figures/<name>.pdf}

    ita          extra     ITA categories and the effect of lighting (Figures/fig_ita.pdf)
    cvar         extra     average loss versus CVaR, simulated (Figures/fig_cvar.pdf)
    composition  Figure 1  racial make-up of the 140k corpus
    arch         Figure 2  the multimodal detector
    evaldesign   Figure 3  the two test conditions
    contrast     Figure 4  FPR by race, familiar vs new generator
    forest       Figure 5  fairness gaps, baseline vs fairness-aware
    confusion    Figure 6  cross-generator confusion matrices

Only needs numpy and matplotlib (no SciPy).

All numbers are in the DATA section below. They are the results from the
report's tables, stored as counts (errors out of n) so every rate can be
recomputed. Where the report gives a confidence interval it is used as is;
the remaining intervals (in-distribution rates and the gaps in Figure 5)
are bootstrapped here from the counts with 2,000 resamples.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

OUT = Path("figures")
RACES = ["Black", "Indian", "White"]

# =============================================================================
# DATA
# =============================================================================
# FairFace labels over all 140,000 images (Table 2)
COMPOSITION = {"White": 88496, "Other": 41165, "Black": 5975, "Indian": 4364}
CORPUS_TOTAL = 140000

# In-distribution test: 400 real + 400 fake per race (Table 3)
INDIST_FP = {"Black": (0, 400), "Indian": (0, 400), "White": (2, 400)}   # FPR
INDIST_FN = {"Black": (2, 400), "Indian": (1, 400), "White": (3, 400)}   # FNR

# Cross-generator test (Tables 4 and 5): (errors, n)
#   IMDb = real faces only -> false positives; SD = fake faces only -> misses
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
# 95% CIs reported in Table 4 (baseline, cross-generator FPR)
CROSS_FPR_CI_REPORTED = {"Black": (0.401, 0.449), "Indian": (0.482, 0.619),
                         "White": (0.349, 0.377)}

N_BOOT, SEED = 2000, 0

# =============================================================================
# STYLE (dataviz reference palette, light mode, print)
# =============================================================================
SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, PURPLE = "#77b2fb", "#815F9D"          # categorical slots 1 and 2
GREY_BAR = "#c3c2b7"                         # "left out" / reference
PURPLE_RAMP = ["#ffffff", "#e5d9f8", "#cbb5ef", "#b290e4", "#9b69da",
               "#8d5acb", "#7d4fb5", "#6e43a2", "#5f398d"]
TINT = {"neutral": "#cea2e2", "blue": "#e6f0fc", "PURPLE": "#fdeee7"}

WIDTH = 5.4                                  # inches = \textwidth in the template

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


def n_eq(n: int) -> str:
    """'n = 6 499' typeset as one maths expression."""
    return "$n = " + f"{n:,}".replace(",", r"\,") + "$"


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


# --- diagram helpers ---------------------------------------------------------
def canvas(h_in):
    fig = plt.figure(figsize=(WIDTH, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h_in / WIDTH)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, sub=None, fc=TINT["neutral"], dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=fc, ec=INK2, lw=0.8, ls=(0, (3, 2)) if dashed else "-",
                                zorder=2))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + 1.5, title, ha="center", va="bottom", fontsize=9.5,
                color=INK, zorder=3)
        ax.text(x + w / 2, cy + 0.6, sub, ha="center", va="top", fontsize=7.5,
                color=INK2, zorder=3, linespacing=1.25)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=9.5,
                color=INK, zorder=3)
    return (x, y, w, h)


def arrow(ax, p, q, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9,
                                 color=INK2, shrinkA=0, shrinkB=0, zorder=1,
                                 ls=(0, (3, 2)) if dashed else "-"))


def right(b):  return (b[0] + b[2], b[1] + b[3] / 2)
def left(b):   return (b[0], b[1] + b[3] / 2)
def top(b):    return (b[0] + b[2] / 2, b[1] + b[3])
def bottom(b): return (b[0] + b[2] / 2, b[1])


# =============================================================================
# Figure 1: corpus composition
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

    ind = COMPOSITION["Indian"]
    ax.axvline(ind, color=INK2, lw=0.8, zorder=2)
    ax.text(ind + 1200, -0.68, f"",
            va="center", ha="left", color=INK2, fontsize=10, zorder=4,
            bbox=dict(fc=SURFACE, ec="none", pad=1.5))

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g if g in used else f"{g}" for g in groups],
                       color=INK2, fontsize=11)
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
# Figure 2: architecture
# =============================================================================
def fig_arch():
    fig, ax = canvas(2.6)

    rgb = box(ax, 1, 30, 18, 12, "Face crop", r"RGB, $224 \times 224$", fc=SURFACE)
    fft = box(ax, 1, 7, 18, 12, "Spectrum", "greyscale, 2D FFT,\n" r"$\log(1+|\mathcal{F}|)$",
              fc=SURFACE)
    swin = box(ax, 24, 30, 23, 12, "Swin-Tiny", "pretrained on ImageNet,\nfine-tuned")
    cnn = box(ax, 24, 7, 23, 12, "Small CNN", "trained from scratch")
    fuse = box(ax, 53, 18.5, 20, 12, "Fusion", "concatenate,\n256-d embedding")
    head = box(ax, 79, 18.5, 20, 12, "Forgery head", "linear + sigmoid,\nfake if $p \\geq 0.5$")
    adv = box(ax, 53, 0.8, 20, 11, "Race adversary", "gradient reversal,\nfair recipe only",
              fc=SURFACE, dashed=True)

    ax.text(1, 43.2, "Image stream", fontsize=8.5, color=MUTED, va="bottom")
    ax.text(1, 5.8, "Frequency stream", fontsize=8.5, color=MUTED, va="top")

    arrow(ax, bottom(rgb), top(fft))
    arrow(ax, right(rgb), left(swin))
    arrow(ax, right(fft), left(cnn))
    arrow(ax, right(swin), (fuse[0], fuse[1] + fuse[3] * 0.75))
    arrow(ax, right(cnn), (fuse[0], fuse[1] + fuse[3] * 0.25))
    arrow(ax, right(fuse), left(head))
    arrow(ax, bottom(fuse), top(adv), dashed=True)
    save(fig, "fig_arch")


# =============================================================================
# Figure 3: evaluation design
# =============================================================================
def fig_evaldesign():
    fig, ax = canvas(2.55)
    H = 100 * 2.55 / WIDTH                        # ~47

    model = box(ax, 31, H - 10.5, 38, 9.5, "Trained detector",
                "FFHQ real + StyleGAN fake")

    ind = box(ax, 1, 10, 30, 21, "In-distribution test",
              "FFHQ real + StyleGAN fake,\n" + num(2400) + " images, both labels",
              fc=TINT["blue"])

    # container for the two unseen sources
    cx, cy, cw, chh = 37, 10, 62, 21
    ax.add_patch(FancyBboxPatch((cx, cy), cw, chh, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=TINT["PURPLE"], ec=INK2, lw=0.8, zorder=1))
    ax.text(cx + cw / 2, cy + chh - 1.6, "Cross-generator test (unseen sources)",
            ha="center", va="top", fontsize=9.5, color=INK, zorder=3)
    imdb = box(ax, cx + 2, cy + 2, 28, 13, "IMDb real faces",
               "real only, " + n_eq(6499) + ",\ngives FPR", fc=SURFACE)
    sd = box(ax, cx + 32, cy + 2, 28, 13, "Stable Diffusion fakes",
             "fake only, " + n_eq(14177) + ",\ngives FNR", fc=SURFACE)

    arrow(ax, (model[0] + 5, model[1]), top(ind))
    arrow(ax, (model[0] + model[2] - 5, model[1]), (cx + cw / 2, cy + chh))

    ax.text(ind[0] + ind[2] / 2, 8, "All metrics: accuracy, AUC,\nFPR, FNR and their gaps",
            ha="center", va="top", fontsize=8.5, color=INK2, linespacing=1.3)
    ax.text(cx + cw / 2, 8,
            "Races are compared within each source.\nThe two sources are never pooled.",
            ha="center", va="top", fontsize=8.5, color=INK2, linespacing=1.3)
    save(fig, "fig_evaldesign")


# =============================================================================
# Figure 4: FPR by race, two test conditions
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
        ("Cross-generator test (IMDb real faces)", PURPLE, +off,
         [rate(CROSS["baseline"]["fp"][r]) for r in RACES],
         [CROSS_FPR_CI_REPORTED[r] for r in RACES]),
    ]
    for label, col, dx, vals, cis in series:
        for xi, v, (lo, hi) in zip(x, vals, cis):
            ax.plot([xi + dx, xi + dx], [lo, hi], color=col, lw=1.6,
                    solid_capstyle="round", zorder=3)
        ax.plot(x + dx, vals, "o", ms=7, color=col, mec=SURFACE, mew=1.5,
                zorder=4, label=label)
    # direct labels on the cross-generator points only
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
# Figure 5: fairness gaps, baseline vs fair (forest plot)
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
# Figure 6: cross-generator confusion matrices
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
# Extra figure: ITA scale and why lighting moves it
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
# Extra figure: average loss versus CVaR (illustration with simulated losses)
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
    "arch": fig_arch,
    "evaldesign": fig_evaldesign,
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