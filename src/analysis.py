"""Statistical analysis and paper-ready tables.

Produces, per track:
  * stats.csv       - mean +/- std of every metric, with paired-test p-values of
                      each repair method against the Split-CP baseline;
  * main_table.tex  - a LaTeX coverage table with significance markers;
  * guide_table.tex - the decision guide as a LaTeX table.

Significance uses a paired t-test across seeds (methods share the same seeds, so
the comparison is paired). With few seeds it is indicative; use >= 10 seeds for
the final manuscript.
"""

import os
import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

METHOD_ORDER = ["Split CP", "Weighted CP", "Classwise CP", "ACI"]
BASELINE = "Split CP"


def _paired_p(x, base):
    """Paired t-test p-value; NaN if undefined (too few or degenerate samples)."""
    x, base = np.asarray(x), np.asarray(base)
    if not _HAVE_SCIPY or len(x) != len(base) or len(x) < 2 or np.allclose(x, base):
        return np.nan
    if np.std(x - base) == 0:
        return np.nan
    return float(sps.ttest_rel(x, base).pvalue)


def build_stats(df):
    """Return a tidy DataFrame with aggregated metrics and p-values vs baseline."""
    rows = []
    for sev in sorted(df.severity.unique()):
        base_cov = df[(df.method == BASELINE) & (df.severity == sev)].sort_values("seed")["coverage"].values
        base_wc = df[(df.method == BASELINE) & (df.severity == sev)].sort_values("seed")["worst_class_cov"].values
        for m in [x for x in METHOD_ORDER if x in df.method.unique()]:
            d = df[(df.method == m) & (df.severity == sev)].sort_values("seed")
            rows.append({
                "method": m, "severity": sev,
                "coverage_mean": d.coverage.mean(), "coverage_std": d.coverage.std(ddof=0),
                "set_size_mean": d.set_size.mean(), "set_size_std": d.set_size.std(ddof=0),
                "worst_class_mean": d.worst_class_cov.mean(), "worst_class_std": d.worst_class_cov.std(ddof=0),
                "cost_ms_mean": d.cost_ms.mean(),
                "p_coverage_vs_split": _paired_p(d.coverage.values, base_cov),
                "p_worst_class_vs_split": _paired_p(d.worst_class_cov.values, base_wc),
            })
    return pd.DataFrame(rows)


def _stars(p):
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main_table_latex(stats_df, track, target):
    """Coverage (mean +/- std) as method x severity, with significance stars."""
    sevs = sorted(stats_df.severity.unique())
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Marginal coverage (mean\,$\pm$\,std over seeds) under %s shift. "
        r"Target $=%.2f$. Stars: significance vs.\ Split CP (\*\,$p{<}0.05$, "
        r"\*\*\,$p{<}0.01$, \*\*\*\,$p{<}0.001$, paired $t$-test).}" % (track, target),
        r"\label{tab:coverage_%s}" % track,
        r"\begin{tabular}{l" + "c" * len(sevs) + "}",
        r"\toprule",
        "Method & " + " & ".join(f"{s:g}" for s in sevs) + r" \\",
        r"\midrule",
    ]
    for m in [x for x in METHOD_ORDER if x in stats_df.method.unique()]:
        cells = []
        for s in sevs:
            r = stats_df[(stats_df.method == m) & (stats_df.severity == s)].iloc[0]
            cells.append(f"{r.coverage_mean:.3f}\\,{{\\scriptsize$\\pm${r.coverage_std:.3f}}}{_stars(r.p_coverage_vs_split)}")
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def guide_table_latex(guide, track):
    priorities = list(guide.keys())
    sevs = list(guide[priorities[0]].keys())
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Recommended repair method under %s shift, by practitioner "
        r"priority and shift level.}" % track,
        r"\label{tab:guide_%s}" % track,
        r"\begin{tabular}{l" + "c" * len(sevs) + "}",
        r"\toprule",
        "Priority & " + " & ".join(f"{float(s):g}" for s in sevs) + r" \\",
        r"\midrule",
    ]
    for p in priorities:
        lines.append(f"{p} & " + " & ".join(guide[p][s].replace(" CP", "") for s in sevs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def export(df, guide, cfg, track, out_dir):
    stats_df = build_stats(df)
    stats_df.round(4).to_csv(os.path.join(out_dir, "stats.csv"), index=False)
    with open(os.path.join(out_dir, "main_table.tex"), "w") as f:
        f.write(main_table_latex(stats_df, track, 1 - cfg.alpha))
    with open(os.path.join(out_dir, "guide_table.tex"), "w") as f:
        f.write(guide_table_latex(guide, track))
    return stats_df
