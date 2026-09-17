"""Conformal prediction: nonconformity scores and the coverage-repair methods.

Everything is expressed in a single unifying form. A score function turns a
softmax matrix P (N x K) into a nonconformity matrix S (N x K), where *smaller is
more conforming*. A prediction set is then simply {y : S[i, y] <= tau}. All four
methods differ only in how the threshold tau is chosen.

Methods
    split_cp     - standard split conformal (the baseline that breaks under shift)
    weighted_cp  - importance-weighted CP for covariate shift (Tibshirani et al., 2019)
    classwise_cp - class-conditional / Mondrian CP (per-class thresholds)
    aci          - Adaptive Conformal Inference for streaming shift (Gibbs & Candes, 2021)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------------------------------- #
# Nonconformity score functions
# --------------------------------------------------------------------------- #
def scores_lac(P):
    """LAC / THR score: 1 - p_y. Smallest sets, weakest conditional coverage."""
    return 1.0 - P


def scores_aps(P, rng):
    """APS score: randomised cumulative probability mass up to class y."""
    order = np.argsort(-P, axis=1)                 # classes by descending prob
    sorted_P = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(sorted_P, axis=1)
    u = rng.uniform(size=P.shape)
    # score at a sorted position = cumulative-before + u * p_here
    sorted_scores = cum - u * sorted_P
    # scatter back to original class order
    S = np.empty_like(P)
    np.put_along_axis(S, order, sorted_scores, axis=1)
    return S


def scores_raps(P, rng, lam, k_reg):
    """RAPS score: APS plus a penalty on high-rank (large-set) classes."""
    order = np.argsort(-P, axis=1)
    sorted_P = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(sorted_P, axis=1)
    u = rng.uniform(size=P.shape)
    ranks = np.arange(1, P.shape[1] + 1)[None, :]
    penalty = lam * np.maximum(0, ranks - k_reg)
    sorted_scores = cum - u * sorted_P + penalty
    S = np.empty_like(P)
    np.put_along_axis(S, order, sorted_scores, axis=1)
    return S


def make_scores(P, cfg, rng):
    if cfg.score_fn == "lac":
        return scores_lac(P)
    if cfg.score_fn == "aps":
        return scores_aps(P, rng)
    if cfg.score_fn == "raps":
        return scores_raps(P, rng, cfg.raps_lambda, cfg.raps_k_reg)
    raise ValueError(cfg.score_fn)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _conformal_quantile(cal_scores, alpha):
    """Finite-sample corrected (1 - alpha) quantile of calibration scores."""
    n = len(cal_scores)
    if n == 0:
        return np.inf
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(cal_scores, level, method="higher")


def _sets_from_threshold(S_test, tau):
    """Boolean membership matrix {y : S[i, y] <= tau}."""
    return S_test <= tau


# --------------------------------------------------------------------------- #
# Repair methods. Each returns a boolean (N_test x K) membership matrix.
# S_cal_true : nonconformity of the *true* label for each calibration point.
# --------------------------------------------------------------------------- #
def split_cp(S_cal_true, S_test, alpha):
    tau = _conformal_quantile(S_cal_true, alpha)
    return _sets_from_threshold(S_test, tau)


def classwise_cp(S_cal_true, y_cal, S_test, alpha, n_classes):
    """Per-class thresholds -> label-conditional coverage."""
    global_tau = _conformal_quantile(S_cal_true, alpha)
    sets = np.zeros(S_test.shape, dtype=bool)
    for c in range(n_classes):
        mask = y_cal == c
        tau_c = _conformal_quantile(S_cal_true[mask], alpha) if mask.sum() >= 10 else global_tau
        sets[:, c] = S_test[:, c] <= tau_c
    return sets


def estimate_weights(feat_cal, feat_test, seed, clip=20.0):
    """Density-ratio weights via a domain classifier (calibration vs test).

    Fitting on a low-dimensional summary avoids the well-known collapse of
    density-ratio estimation in high dimensions. Weights are clipped for stability.
    """
    Xd = np.vstack([feat_cal, feat_test])
    yd = np.concatenate([np.zeros(len(feat_cal)), np.ones(len(feat_test))])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xd, yd)
    p = clf.predict_proba(feat_cal)[:, 1]
    ratio = (p / (1 - p + 1e-8)) * (len(feat_cal) / len(feat_test))
    return np.clip(ratio, 1.0 / clip, clip)


def estimate_label_shift_weights(P_cal, y_cal, P_test, n_classes, clip=20.0):
    """Label-shift weights via Black-Box Shift Estimation (Lipton et al., 2018).

    Estimates the test class priors from the calibration confusion matrix and the
    mean test prediction, then weights each calibration point by the ratio of test
    to calibration prior for its class.
    """
    C = np.zeros((n_classes, n_classes))            # C[:, k] = E_cal[softmax | y = k]
    for k in range(n_classes):
        m = y_cal == k
        if m.sum() > 0:
            C[:, k] = P_cal[m].mean(0)
    mu = P_test.mean(0)                              # mean predicted distribution on test
    try:
        pi_test = np.linalg.solve(C, mu)
    except np.linalg.LinAlgError:
        pi_test = np.linalg.lstsq(C, mu, rcond=None)[0]
    pi_test = np.clip(pi_test, 1e-3, None)
    pi_test /= pi_test.sum()
    pi_cal = np.array([(y_cal == k).mean() for k in range(n_classes)])
    w = pi_test[y_cal] / (pi_cal[y_cal] + 1e-8)
    return np.clip(w, 1.0 / clip, clip)


def weighted_cp(S_cal_true, S_test, alpha, weights):
    """Importance-weighted split CP for covariate shift."""
    order = np.argsort(S_cal_true)
    s_sorted = S_cal_true[order]
    w_sorted = weights[order]
    w_norm = w_sorted / w_sorted.sum()
    cum = np.cumsum(w_norm)
    idx = np.searchsorted(cum, 1 - alpha)
    tau = s_sorted[min(idx, len(s_sorted) - 1)]
    return _sets_from_threshold(S_test, tau)


def aci(S_cal_true, S_test, y_test, alpha, gamma):
    """Adaptive Conformal Inference over a test stream.

    alpha is nudged after each observation so realised coverage tracks the
    target even as the distribution drifts. Returns the membership matrix built
    from the *time-varying* thresholds.
    """
    s_sorted = np.sort(S_cal_true)
    n = len(s_sorted)
    sets = np.zeros(S_test.shape, dtype=bool)
    a_t = alpha
    for t in range(len(S_test)):
        a_clip = float(np.clip(a_t, 1e-3, 1 - 1e-3))
        k = int(np.ceil((n + 1) * (1 - a_clip))) - 1
        tau = s_sorted[min(max(k, 0), n - 1)]
        row = S_test[t] <= tau
        sets[t] = row
        covered = row[y_test[t]]
        a_t = a_t + gamma * (alpha - (0.0 if covered else 1.0))
    return sets


def signal_features(X):
    """Cheap, low-dimensional signal statistics used for weight estimation."""
    x = X[:, 0, :]
    feats = np.stack([
        x.mean(1), x.std(1), np.sqrt((x ** 2).mean(1)),          # mean, std, RMS
        (np.abs(x).max(1) / (np.sqrt((x ** 2).mean(1)) + 1e-8)),  # crest factor
        ((x ** 4).mean(1) / ((x ** 2).mean(1) ** 2 + 1e-8)),      # kurtosis
    ], axis=1)
    return feats
