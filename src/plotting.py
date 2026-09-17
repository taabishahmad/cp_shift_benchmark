"""Publication-quality figures.

A single restrained style is applied everywhere: a muted, colour-blind-safe
palette, light gridlines, clean spines and generous whitespace. Every figure is
saved at 300 dpi as both PNG and PDF for direct inclusion in a manuscript.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Muted, colour-blind-safe palette (one hue per method).
PALETTE = {
    "Split CP":     "#4C72B0",   # blue
    "Weighted CP":  "#55A868",   # green
    "Classwise CP": "#C44E52",   # red
    "ACI":          "#8172B3",   # purple
}
MARKERS = {"Split CP": "o", "Weighted CP": "s", "Classwise CP": "^", "ACI": "D"}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def _save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_worst_vs_severity(df, cfg, xlabel="Distribution-shift severity"):
    """Worst-class coverage across the shift axis - the safety-critical headline
    for label shift, where marginal coverage can look healthy while a dangerous
    class is silently undercovered."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    target = 1 - cfg.alpha
    for method in PALETTE:
        sub = df[df.method == method].groupby("severity")["worst_class_cov"]
        m = sub.mean()
        ax.plot(m.index, m.values, color=PALETTE[method], marker=MARKERS[method], label=method)
    ax.axhline(target, ls="--", color="#444444", lw=1.3, label=f"Target ({target:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Worst-class coverage")
    ax.set_title("Worst-class coverage under increasing shift")
    ax.set_ylim(min(0.55, target - 0.35), 1.02)
    ax.legend(loc="lower left")
    _save(fig, cfg.fig_dir, "fig6_worst_class_vs_severity")


def plot_coverage_vs_severity(df, cfg, xlabel="Distribution-shift severity"):
    """Headline figure: marginal coverage versus shift level, per method."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    target = 1 - cfg.alpha
    for method in PALETTE:
        sub = df[df.method == method].groupby("severity")
        m = sub["coverage"].mean()
        sd = sub["coverage"].std().fillna(0)
        ax.plot(m.index, m.values, color=PALETTE[method], marker=MARKERS[method], label=method)
        ax.fill_between(m.index, m - sd, m + sd, color=PALETTE[method], alpha=0.12)
    ax.axhline(target, ls="--", color="#444444", lw=1.3, label=f"Target ({target:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Marginal coverage")
    ax.set_title("Coverage under increasing shift")
    ax.set_ylim(min(0.55, target - 0.35), 1.02)
    ax.legend(loc="lower left", ncol=1)
    _save(fig, cfg.fig_dir, "fig1_coverage_vs_severity")


def plot_coverage_size_tradeoff(df, cfg):
    """Coverage vs set size at the strongest shift - the efficiency picture."""
    sev = df.severity.max()
    sub = df[df.severity == sev]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for method in PALETTE:
        d = sub[sub.method == method]
        ax.scatter(d.set_size.mean(), d.coverage.mean(), s=90,
                   color=PALETTE[method], marker=MARKERS[method], label=method, zorder=3)
    ax.axhline(1 - cfg.alpha, ls="--", color="#444444", lw=1.2)
    ax.set_xlabel("Average set size  (smaller is better)")
    ax.set_ylabel("Marginal coverage")
    ax.set_title(f"Coverage-efficiency trade-off at severity {sev:g}")
    ax.legend(loc="lower right")
    _save(fig, cfg.fig_dir, "fig2_coverage_size_tradeoff")


def plot_worst_class(df, cfg):
    """Worst-class coverage at the strongest shift - the safety-critical view."""
    sev = df.severity.max()
    sub = df[df.severity == sev].groupby("method")["worst_class_cov"].mean()
    methods = [m for m in PALETTE if m in sub.index]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar(methods, [sub[m] for m in methods],
           color=[PALETTE[m] for m in methods], width=0.6)
    ax.axhline(1 - cfg.alpha, ls="--", color="#444444", lw=1.2, label=f"Target ({1-cfg.alpha:.2f})")
    ax.set_ylabel("Worst-class coverage")
    ax.set_title(f"Worst-class coverage at severity {sev:g}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, cfg.fig_dir, "fig3_worst_class_coverage")


def plot_cost(df, cfg):
    """The axis the literature omits: wall-clock cost of each repair method."""
    sub = df.groupby("method")["cost_ms"].mean()
    methods = [m for m in PALETTE if m in sub.index]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar(methods, [sub[m] for m in methods],
           color=[PALETTE[m] for m in methods], width=0.6)
    ax.set_ylabel("Calibrate + predict time (ms)")
    ax.set_title("Computational cost per method")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, cfg.fig_dir, "fig4_computational_cost")


def plot_decision_guide(guide, cfg):
    """Heatmap: recommended method per (severity, practitioner priority)."""
    priorities = list(guide.keys())
    severities = list(guide[priorities[0]].keys())
    methods = list(PALETTE.keys())
    codes = {m: i for i, m in enumerate(methods)}
    grid = np.array([[codes[guide[p][s]] for s in severities] for p in priorities])

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    cmap = mpl.colors.ListedColormap([PALETTE[m] for m in methods])
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=len(methods) - 1)
    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels([f"{s:g}" for s in severities])
    ax.set_yticks(range(len(priorities)))
    ax.set_yticklabels(priorities)
    ax.set_xlabel("Shift severity")
    ax.set_title("Recommended repair method")
    for i in range(len(priorities)):
        for j in range(len(severities)):
            ax.text(j, i, methods[grid[i, j]].replace(" CP", ""),
                    ha="center", va="center", color="white", fontsize=9)
    handles = [mpl.patches.Patch(color=PALETTE[m], label=m) for m in methods]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, cfg.fig_dir, "fig5_decision_guide")
