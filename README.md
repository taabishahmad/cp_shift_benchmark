# Conformal Prediction Under Distribution Shift — A Fault-Detection Benchmark

A clean, honest benchmark of conformal-prediction **coverage-repair** methods for
safety-critical fault detection, plus two things the literature usually omits: an
apples-to-apples **computational-cost** comparison and a **practitioner decision
guide** (which method to use for which shift regime and priority).

The whole pipeline is post-hoc after one small model train, so it runs on a
**laptop CPU** in a few minutes. A GPU is used automatically if present but is not
required.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
python run_benchmark.py
```

Outputs (written per track, under `results/covariate/` and `results/label/`):

* `results.csv` — every seed × severity × method row
* `summary.csv` — averaged summary table
* `stats.csv` — mean ± std of each metric with paired-test p-values vs. Split CP
* `main_table.tex`, `guide_table.tex` — paper-ready LaTeX tables
* `decision_guide.json` — the recommended-method mapping
* `figures/<track>/*.png` and `*.pdf` — six publication-ready figures each

Default runtime is a few minutes on CPU. Edit `config.py` to scale sizes,
severities, the target coverage `alpha`, or the score function.

---

## Switching to real CWRU bearing data (for the paper)

The synthetic generator is the **controlled track**. For the real-shift track,
use the Case Western Reserve University bearing dataset, whose natural shift is
across motor loads (0/1/2/3 hp). The loader is fully implemented — you only supply
the files:

1. Download the drive-end (`_DE_time`) `.mat` files from the CWRU Bearing Data
   Center: the Normal baseline and the Inner-race, Outer-race and Ball faults.
2. Place them under `data/cwru/` so that each file's **path contains its class and
   load**. Either of these works:
   * folder layout — `data/cwru/0hp/normal/*.mat`, `data/cwru/1hp/inner/*.mat`, … ; or
   * descriptive filenames — e.g. `inner_1hp_014.mat`, `normal_0hp.mat`.
   The loader infers the class from the keywords *normal / inner / outer / ball*
   and the load from *0hp…3hp* anywhere in the path.
3. Set `data_source = "cwru"` in `config.py` and run.

The loader segments each recording into non-overlapping windows of
`signal_length`, uses load 0 hp as the source (train/calibration) and 1–3 hp as the
shifted test domains. Nothing else changes — model, conformal methods, metrics,
tests and figures are data-agnostic. (If your class keywords differ, edit
`_CWRU_CLASS_KEYWORDS` at the top of `src/data.py`.)

---

## What each method is and why it is here

| Method | Repairs | One-line rationale |
|---|---|---|
| **Split CP** | nothing (baseline) | Standard split conformal; the thing that silently breaks under shift. |
| **Weighted CP** | covariate shift | Reweights calibration scores by an estimated density ratio (Tibshirani et al., 2019). |
| **Classwise CP** | label-conditional gaps | Per-class thresholds; protects the coverage of rare, dangerous classes. |
| **ACI** | streaming / evolving shift | Online adjustment of the error rate from realised miscoverage (Gibbs & Candès, 2021). |

Score functions (`config.score_fn`): **LAC** (smallest sets), **APS** (adaptive,
better conditional coverage — the default), **RAPS** (APS with a size penalty).

## The metrics

Marginal coverage, **worst-class coverage** (where a rare fault hides),
**size-stratified coverage** (conditional-coverage failures the marginal number
masks), average set size (efficiency), empty-set rate, and wall-clock **cost**.

## Reading the result

The benchmark runs **two shift tracks**, and the honest finding is that no single
repair wins everywhere - each matches a shift type:

* **Covariate shift** (`results/covariate/`): split conformal collapses past a
  tipping point, and **online adaptation (ACI) is the method that keeps coverage**.
  Static reweighting helps only while the target stays within the calibration
  score support.
* **Label shift** (`results/label/`): marginal coverage can look perfectly
  healthy (~0.90) while a hard-to-detect fault class is **silently undercovered** -
  the dangerous failure a single marginal number hides. **Class-conditional CP (and
  ACI) restore that worst-class coverage**; see `fig6_worst_class_vs_severity`.

This "match the repair to the shift", together with the cost axis and the decision
guide, is the contribution. Importance-weighted CP giving only limited repair in
this source-only-calibration setting is itself a useful, reportable negative result.


