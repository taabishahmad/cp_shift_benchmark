"""Turn the benchmark table into a practitioner decision guide.

For each shift severity and each practitioner priority we pick the method that
best serves that priority *subject to* respecting the coverage target where
possible. This mapping is the study's headline, reusable contribution.
"""

import numpy as np

PRIORITIES = ["Coverage first", "Small sets", "Low cost"]


def build_guide(df, cfg):
    target = 1 - cfg.alpha
    methods = df.method.unique().tolist()
    severities = sorted(df.severity.unique())
    guide = {p: {} for p in PRIORITIES}

    for s in severities:
        agg = df[df.severity == s].groupby("method").agg(
            coverage=("coverage", "mean"),
            set_size=("set_size", "mean"),
            cost_ms=("cost_ms", "mean"),
        )
        valid = agg[agg.coverage >= target - 0.01]           # methods that hold coverage
        pool = valid if len(valid) else agg                  # fall back if none do

        # Coverage first: closest to (or above) target, then smallest sets.
        cov_choice = (agg.assign(gap=(agg.coverage - target).abs())
                        .sort_values(["gap", "set_size"]).index[0]
                      if not len(valid) else valid.sort_values("set_size").index[0])
        guide["Coverage first"][s] = cov_choice

        # Small sets: smallest sets among coverage-valid methods.
        guide["Small sets"][s] = pool.sort_values("set_size").index[0]

        # Low cost: cheapest among coverage-valid methods.
        guide["Low cost"][s] = pool.sort_values("cost_ms").index[0]

    return guide
