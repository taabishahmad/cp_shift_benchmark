"""Evaluation metrics for prediction sets.

Beyond marginal coverage we report the quantities that actually reveal safety
failures under shift: worst-class coverage, size-stratified coverage (SSC), and
the empty-set rate. Computational cost is measured separately in the runner.
"""

import numpy as np


def marginal_coverage(sets, y):
    return float(sets[np.arange(len(y)), y].mean())


def average_set_size(sets):
    return float(sets.sum(1).mean())


def worst_class_coverage(sets, y, n_classes):
    """Minimum per-class coverage - where the danger of a rare fault hides."""
    covs = []
    for c in range(n_classes):
        mask = y == c
        if mask.sum() > 0:
            covs.append(sets[mask, c].mean())
    return float(np.min(covs)) if covs else float("nan")


def size_stratified_coverage(sets, y, n_bins=3):
    """Worst coverage across set-size strata: exposes conditional miscoverage
    that the marginal number can mask."""
    sizes = sets.sum(1)
    covered = sets[np.arange(len(y)), y]
    edges = np.quantile(sizes, np.linspace(0, 1, n_bins + 1))
    worst = 1.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (sizes >= lo) & (sizes <= hi) if b == n_bins - 1 else (sizes >= lo) & (sizes < hi)
        if m.sum() > 0:
            worst = min(worst, float(covered[m].mean()))
    return worst


def empty_rate(sets):
    return float((sets.sum(1) == 0).mean())


def evaluate(sets, y, n_classes):
    return {
        "coverage": marginal_coverage(sets, y),
        "set_size": average_set_size(sets),
        "worst_class_cov": worst_class_coverage(sets, y, n_classes),
        "ssc": size_stratified_coverage(sets, y),
        "empty_rate": empty_rate(sets),
    }
