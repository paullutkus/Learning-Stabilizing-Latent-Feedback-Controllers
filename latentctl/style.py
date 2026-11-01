"""One visual language for every figure of the paper.

Sizes are set for inclusion at width=\\linewidth in the paper (article, 10 pt, 1 in
margins: 6.5 in text width), so a figure is drawn at its printed size and fonts are
the point sizes the reader sees (body text 10 pt, captions 9 pt).  Type is Computer
Modern (cmr10 + mathtext 'cm'), the document's own face.

Colour does one job per figure and keeps its meaning across figures:

    BWD    blue    multi-step backwards loss L_bwd / latent model trained with it
    FWD    orange  decoder-free forwards loss L_fwd
    MLP    aqua    MLP latent model (Fig. exp_arm_nonlinear)
    REF    ink     the true system, full-state LQR, Algorithm 1 with known (A, B)
    OPEN   muted   open-loop spectrum, reference levels

Slots are the first three of the validated reference palette (all pairs pass the CVD
and normal-vision gates, so any two may sit side by side); every series also carries
a marker or dash, so identity never rests on colour alone.  Light tints of the same
hues are used for filled regions.
"""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import matplotlib

logging.getLogger("fontTools").setLevel(logging.ERROR)        # cmr10 subsetting chatter
warnings.filterwarnings("ignore", message=".*cmr10 font should ideally be used with mathtext.*")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- palette -----------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]          # steps 100 .. 700
BLUE_TINT, ORANGE_TINT, AQUA_TINT, GRAY_TINT = "#dbe9fa", "#fbe1d6", "#d3f0e5", "#ecebe7"

BWD, FWD, MLP, REF, OPEN = BLUE, ORANGE, AQUA, INK, MUTED

TEXT_W = 6.5            # \linewidth of the paper [in]

RC = {
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
    "font.size": 8.0, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.title_fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.labelpad": 2.5, "axes.titlepad": 4.0, "axes.titlelocation": "left",
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 3.8, "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round", "patch.linewidth": 0.0,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.4, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.35,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
}


def use():
    plt.rcParams.update(RC)


use()


# --- helpers -----------------------------------------------------------------------
def title(ax, letter, text, **kw):
    """Panel title '(a) text', left aligned."""
    ax.set_title(f"({letter})  {text}" if letter else text, loc="left", **kw)


def log_ticks(ax, axis="x", values=None, fmt="{:g}", labels=None):
    """Plain-number tick labels at the given values on a log axis (only at ``labels``, if given)."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    if values is not None:
        (ax.set_xticks if axis == "x" else ax.set_yticks)(list(values))
    show = (lambda v: True) if labels is None else (lambda v: any(np.isclose(v, l) for l in labels))
    a.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt.format(v) if show(v) else ""))
    a.set_minor_locator(mticker.NullLocator())


def pow10_ticks(ax, axis="y", numticks=6):
    """Ticks 10^k on a log axis (at most ``numticks``), labels as $10^{k}$ (and 1 for 10^0)."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(mticker.LogLocator(base=10, numticks=numticks))
    a.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: "$1$" if abs(np.log10(v)) < 1e-9 else f"$10^{{{int(round(np.log10(v)))}}}$"))
    a.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    a.set_minor_formatter(mticker.NullFormatter())


def unit_circle(ax, fill=True, lim=1.35, ticks=(-1, 0, 1)):
    """Complex-plane axes with the unit disk."""
    if fill:
        ax.add_patch(plt.Circle((0, 0), 1.0, facecolor="#f4f3f0", edgecolor="none", zorder=0))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=AXIS, lw=0.7, zorder=1)
    ax.axhline(0, color=GRID, lw=0.5, zorder=0.5)
    ax.axvline(0, color=GRID, lw=0.5, zorder=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


SPREAD = "iqr"                 # bars over data sets: "iqr" (interquartile range) or "minmax"


def spread(v, kind=None):
    """(lo, hi) of the finite values of v: interquartile range or min-max (``SPREAD``); NaNs if none."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return np.nan, np.nan
    return tuple(np.percentile(v, [25, 75])) if (kind or SPREAD) == "iqr" else (v.min(), v.max())


def errbar(ax, x, lo, hi, color, lw=0.75, cap=4.0, alpha=0.8, zorder=2.5):
    """Spread behind a line: a vertical bar from lo to hi at each x, with caps (none where lo = hi)."""
    x, lo, hi = (np.asarray(v, float) for v in (x, lo, hi))
    ok = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    ax.vlines(x[ok], lo[ok], hi[ok], color=color, lw=lw, alpha=alpha, zorder=zorder)
    for y in (lo[ok], hi[ok]):
        ax.plot(x[ok], y, ls="none", marker="_", ms=cap, mew=lw, color=color, alpha=alpha, zorder=zorder)


def handle(color, marker=None, ls="-", lw=1.4, ms=4.5, label="", **kw):
    return plt.Line2D([], [], color=color, marker=marker, ls=ls, lw=lw, ms=ms, label=label,
                      mec=kw.pop("mec", SURFACE), mew=kw.pop("mew", 0.6), **kw)


def save(fig, path):
    """Save as PDF (``path`` without extension)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf")
    plt.close(fig)
    print(f"  wrote {path}.pdf", flush=True)
