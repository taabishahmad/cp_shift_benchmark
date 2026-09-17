"""Data sources for the benchmark.

Two tracks, matching the two-benchmark design of the study:

  * "synthetic" - a physics-informed vibration-signal generator. Each fault
    class carries its own bearing characteristic frequency, and a `severity`
    knob induces a genuine *covariate shift* (changing operating load: signal
    amplitude, resonance band and noise level) while the fault semantics stay
    fixed. This is the controlled track and runs on any machine.

  * "cwru" - loader for the real Case Western Reserve University bearing data.
    The natural shift there is across motor loads (0/1/2/3 hp). Drop the .mat
    files in `cwru_dir` and set data_source="cwru" (see README).
"""

import os
import glob
import numpy as np

# Bearing characteristic frequencies (normalised, arbitrary units) that make the
# four classes physically distinguishable to the classifier.
_FAULT_FREQS = {
    0: None,    # healthy: no impulsive fault signature
    1: 0.162,   # inner-race (BPFI-like)
    2: 0.107,   # outer-race (BPFO-like)
    3: 0.070,   # ball        (BSF-like)
}

# Mild per-class signature strength. The ball fault (class 3) is the hardest and
# most safety-relevant: its weaker signature gives it lower per-class coverage,
# which is exactly what a label shift toward/away from it exposes.
_CLASS_STRENGTH = {0: 1.0, 1: 1.05, 2: 0.9, 3: 0.45}


def _make_signal(label, severity, length, rng):
    """Synthesise one vibration signal for a given fault class and shift level.

    The fault signature (impulse train at the class characteristic frequency,
    ringing at a *fixed* resonance) is preserved across all severities, so the
    fault stays detectable and p(y|x) is held fixed. `severity` only perturbs the
    operating point - gain, broadband noise and low-frequency baseline wander -
    which is a genuine covariate shift of p(x).
    """
    t = np.arange(length)
    shaft = 0.02          # base shaft rotation frequency
    resonance = 0.30      # resonance band is FIXED (keeps faults recognisable)

    sig = np.sin(2 * np.pi * shaft * t)
    sig += 0.3 * np.sin(2 * np.pi * resonance * t)

    f = _FAULT_FREQS[label]
    if f is not None:
        impulses = np.zeros(length)
        period = max(int(1.0 / f), 1)
        impulses[::period] = 1.0
        ring = np.exp(-0.02 * (t % period)) * np.cos(2 * np.pi * resonance * t)
        sig += 1.6 * _CLASS_STRENGTH[label] * impulses * ring   # class-specific signature

    # ---- covariate shift: operating-point drift, label-preserving ----
    # Graded noise is the dominant driver: it smoothly lifts the nonconformity
    # scores and so produces a gradual, monotone coverage decay. Small gain and
    # baseline terms also drift so the shift is detectable for weight estimation.
    gain = 1.0 + 0.10 * severity
    baseline = 0.15 * severity * np.sin(2 * np.pi * 0.003 * t)
    sig = gain * sig + baseline

    snr_db = 11.0 - 9.0 * severity                    # 11 dB (clean) -> 2 dB (shifted)
    p_sig = np.mean(sig ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    sig += rng.normal(0.0, np.sqrt(p_noise), size=length)
    return sig.astype(np.float32)


def _standardise(x, stats=None):
    """Per-run z-normalisation using source-domain statistics."""
    if stats is None:
        mu, sd = x.mean(), x.std() + 1e-8
        return (x - mu) / sd, (mu, sd)
    mu, sd = stats
    return (x - mu) / sd, stats


def make_synthetic(cfg, severity, n_per_class, seed, stats=None):
    """Return (X, y) for one domain at the requested shift severity (balanced)."""
    return make_synthetic_counts(cfg, severity, [n_per_class] * cfg.n_classes, seed, stats)


def make_synthetic_counts(cfg, severity, counts, seed, stats=None):
    """Like make_synthetic but with an explicit per-class sample count, used to
    induce label (prior-probability) shift while keeping p(x|y) fixed."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(cfg.n_classes):
        for _ in range(int(counts[c])):
            X.append(_make_signal(c, severity, cfg.signal_length, rng))
            y.append(c)
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    X, stats = _standardise(X, stats)
    return X[:, None, :], y, stats


def _label_shift_counts(cfg, level):
    """Per-class test counts for a given imbalance level in [0, 1].

    level 0 -> balanced (matches calibration); level 1 -> the hard-to-detect ball
    fault (class 3) dominates - the safety-relevant regime where a poorly covered
    class becomes common.
    """
    base = cfg.n_per_class_test
    mult = np.array([1 - 0.4 * level, 1.0, 1 - 0.2 * level, 1 + 4.0 * level])
    counts = np.maximum((base * mult).astype(int), cfg.label_min_per_class)
    return counts


# --------------------------------------------------------------------------- #
# Real CWRU bearing data
# --------------------------------------------------------------------------- #
# Class is inferred from keywords anywhere in the file path, so either a
# per-class folder layout (cwru/<load>hp/<class>/*.mat) or descriptive filenames
# work. Motor load (0-3 hp) is inferred from the path if present.
_CWRU_CLASS_KEYWORDS = {
    0: ["normal", "health", "baseline"],
    1: ["inner", "_ir", "/ir", "ir_"],
    2: ["outer", "_or", "/or", "or_"],
    3: ["ball", "_b_", "/b/", "ball_"],
}
_CWRU_CACHE = {}


def _infer_cwru_class(path):
    p = path.lower()
    for c, kws in _CWRU_CLASS_KEYWORDS.items():
        if any(k in p for k in kws):
            return c
    return None


def _infer_cwru_load(path):
    p = path.lower()
    for hp in range(4):
        if f"{hp}hp" in p or f"_{hp}." in p or f"{os.sep}{hp}{os.sep}" in p:
            return hp
    return None


def _read_de_signal(mat_path):
    """Return the drive-end accelerometer time series from a CWRU .mat file."""
    from scipy.io import loadmat
    mat = loadmat(mat_path)
    keys = [k for k in mat if k.endswith("DE_time")] or \
           [k for k in mat if not k.startswith("__") and np.ndim(mat[k]) >= 1]
    if not keys:
        return None
    return np.asarray(mat[keys[0]]).squeeze().astype(np.float32)


def _cwru_pool(cfg, load):
    """Dict {class: array (n_windows, L)} of all windows for one motor load."""
    key = (cfg.cwru_dir, load, cfg.signal_length)
    if key in _CWRU_CACHE:
        return _CWRU_CACHE[key]

    files = glob.glob(os.path.join(cfg.cwru_dir, "**", "*.mat"), recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No .mat files under {cfg.cwru_dir}. See README for the CWRU layout.")

    L = cfg.signal_length
    pool = {c: [] for c in range(cfg.n_classes)}
    for f in files:
        cls = _infer_cwru_class(f)
        fload = _infer_cwru_load(f)
        if cls is None or (fload is not None and fload != load):
            continue
        sig = _read_de_signal(f)
        if sig is None or sig.size < L:
            continue
        n_win = sig.size // L
        windows = sig[:n_win * L].reshape(n_win, L)     # non-overlapping windows
        pool[cls].append(windows)

    pool = {c: (np.concatenate(v) if v else np.empty((0, L), np.float32))
            for c, v in pool.items()}
    missing = [c for c, w in pool.items() if len(w) == 0]
    if missing:
        raise ValueError(
            f"CWRU load {load}hp is missing windows for classes {missing}. "
            f"Check the class keywords / folder layout (see README).")
    _CWRU_CACHE[key] = pool
    return pool


def _sample_cwru(cfg, load, counts, seed, stats=None):
    """Draw per-class windows from the CWRU pool and standardise."""
    rng = np.random.default_rng(seed)
    pool = _cwru_pool(cfg, load)
    X, y = [], []
    for c in range(cfg.n_classes):
        w = pool[c]
        idx = rng.integers(0, len(w), size=int(counts[c]))   # sample with replacement
        X.append(w[idx])
        y += [c] * int(counts[c])
    X = np.concatenate(X)
    y = np.array(y, dtype=np.int64)
    X, stats = _standardise(X, stats)
    return X[:, None, :], y, stats


# --------------------------------------------------------------------------- #
# Unified domain access (synthetic or CWRU)
# --------------------------------------------------------------------------- #
def get_domain(cfg, severity, split, seed, stats=None):
    """Train / calibration domains: always the balanced source distribution."""
    n = {"train": cfg.n_per_class_train,
         "cal": cfg.n_per_class_cal,
         "test": cfg.n_per_class_test}[split]
    sub = seed * 100 + {"train": 1, "cal": 2, "test": 3}[split]
    if cfg.data_source == "synthetic":
        return make_synthetic(cfg, severity, n, sub, stats)
    return _sample_cwru(cfg, 0, [n] * cfg.n_classes, sub, stats)   # source = load 0


def get_test_domain(cfg, shift_type, level, seed, stats):
    """Shifted test domain for a given track.

    covariate : synthetic -> shift severity = level; CWRU -> motor load 0->3 hp.
    label     : source-domain signals with class prevalence set by `level`.
    """
    sub = seed * 100 + 3
    if shift_type == "covariate":
        if cfg.data_source == "synthetic":
            return make_synthetic(cfg, level, cfg.n_per_class_test, sub, stats)
        load = int(round(level * 3))
        return _sample_cwru(cfg, load, [cfg.n_per_class_test] * cfg.n_classes, sub, stats)
    if shift_type == "label":
        counts = _label_shift_counts(cfg, level)
        if cfg.data_source == "synthetic":
            return make_synthetic_counts(cfg, 0.0, counts, sub, stats)
        return _sample_cwru(cfg, 0, counts, sub, stats)
    raise ValueError(shift_type)
