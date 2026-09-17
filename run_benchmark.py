"""End-to-end benchmark runner.

Runs two shift tracks and, for each, compares four conformal coverage-repair
methods across increasing shift, over several seeds. For every combination it
builds prediction sets, times the calibrate+predict step, and records coverage /
efficiency metrics, then produces the figures and the practitioner decision guide.

Tracks
    covariate : operating-point drift (graded noise / gain). ACI is the repair.
    label     : changing fault prevalence. Weighted (BBSE) and Classwise repair.

Run:  python run_benchmark.py
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

from config import CFG
from src import data, model, conformal, metrics, plotting, decision_guide, analysis

warnings.filterwarnings("ignore")


def _time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1e3   # milliseconds


def run_track(cfg, shift_type, net, Scal_true, ycal, Pcal, feat_cal, stats, seed, rng):
    """Evaluate all methods across the severity axis for one shift track."""
    rows = []
    for level in cfg.severities:
        Xte, yte, _ = data.get_test_domain(cfg, shift_type, level, seed, stats)
        Pte = model.predict_proba(net, Xte)
        Ste = conformal.make_scores(Pte, cfg, rng)

        # Weighted CP uses the estimator that matches the shift type.
        if shift_type == "covariate":
            feat_te = conformal.signal_features(Xte)
            w = conformal.estimate_weights(feat_cal, feat_te, seed)
        else:
            w = conformal.estimate_label_shift_weights(Pcal, ycal, Pte, cfg.n_classes)

        methods = {
            "Split CP":     lambda: conformal.split_cp(Scal_true, Ste, cfg.alpha),
            "Weighted CP":  lambda: conformal.weighted_cp(Scal_true, Ste, cfg.alpha, w),
            "Classwise CP": lambda: conformal.classwise_cp(
                Scal_true, ycal, Ste, cfg.alpha, cfg.n_classes),
            "ACI":          lambda: conformal.aci(
                Scal_true, Ste, yte, cfg.alpha, cfg.aci_gamma),
        }
        for name, fn in methods.items():
            sets, cost_ms = _time_call(fn)
            m = metrics.evaluate(sets, yte, cfg.n_classes)
            m.update(seed=seed, severity=level, method=name, cost_ms=cost_ms)
            rows.append(m)
    return rows


def run_seed(cfg, seed):
    rng = np.random.default_rng(seed)

    # source domain: train the classifier and calibrate (both balanced, severity 0)
    Xtr, ytr, stats = data.get_domain(cfg, 0.0, "train", seed)
    net = model.train_model(cfg, Xtr, ytr, seed)

    Xcal, ycal, _ = data.get_domain(cfg, 0.0, "cal", seed, stats)
    Pcal = model.predict_proba(net, Xcal)
    Scal = conformal.make_scores(Pcal, cfg, rng)
    Scal_true = Scal[np.arange(len(ycal)), ycal]
    feat_cal = conformal.signal_features(Xcal)

    out = {}
    for shift_type in cfg.shift_types:
        out[shift_type] = run_track(cfg, shift_type, net, Scal_true, ycal,
                                    Pcal, feat_cal, stats, seed, rng)
    return out


def summarise_and_plot(df, cfg, track):
    out_dir = os.path.join(cfg.out_dir, track)
    fig_dir = os.path.join(cfg.fig_dir, track)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    summary = (df.groupby(["method", "severity"])
                 .agg(coverage=("coverage", "mean"),
                      set_size=("set_size", "mean"),
                      worst_class=("worst_class_cov", "mean"),
                      ssc=("ssc", "mean"),
                      cost_ms=("cost_ms", "mean"))
                 .round(3))
    summary.to_csv(os.path.join(out_dir, "summary.csv"))

    xlabel = "Label-imbalance level" if track == "label" else "Distribution-shift severity"
    cfg.fig_dir = fig_dir                       # plotting reads cfg.fig_dir
    plotting.plot_coverage_vs_severity(df, cfg, xlabel)
    plotting.plot_worst_vs_severity(df, cfg, xlabel)
    plotting.plot_coverage_size_tradeoff(df, cfg)
    plotting.plot_worst_class(df, cfg)
    plotting.plot_cost(df, cfg)
    guide = decision_guide.build_guide(df, cfg)
    plotting.plot_decision_guide(guide, cfg)
    with open(os.path.join(out_dir, "decision_guide.json"), "w") as f:
        json.dump({p: {str(k): v for k, v in d.items()} for p, d in guide.items()}, f, indent=2)

    # statistical tests + LaTeX tables for the manuscript
    analysis.export(df, guide, cfg, track, out_dir)
    return summary


def main():
    cfg = CFG
    base_fig = cfg.fig_dir
    plotting.apply_style()
    print(f"Data source : {cfg.data_source}   |   tracks: {cfg.shift_types}")
    print(f"Target coverage : {1 - cfg.alpha:.0%}   |   score: {cfg.score_fn.upper()}")
    print(f"Seeds : {cfg.seeds}\n")

    collected = {t: [] for t in cfg.shift_types}
    for seed in cfg.seeds:
        print(f"  running seed {seed} ...")
        out = run_seed(cfg, seed)
        for t in cfg.shift_types:
            collected[t] += out[t]

    for t in cfg.shift_types:
        cfg.fig_dir = base_fig
        df = pd.DataFrame(collected[t])
        summary = summarise_and_plot(df, cfg, t)
        print(f"\n===== {t.upper()} shift — summary (averaged over seeds) =====")
        print(summary.to_string())

    print(f"\nSaved results -> {cfg.out_dir}/<track>/   figures -> {base_fig}/<track>/")


if __name__ == "__main__":
    main()
