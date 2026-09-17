"""Central configuration for the benchmark.

Defaults are tuned to run comfortably on a laptop CPU in a few minutes while
still reproducing the coverage-collapse-and-repair story clearly. Scale the
sizes up (or switch DATA_SOURCE to "cwru") for the numbers you report in the paper.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- reproducibility ---
    seeds: tuple = (0, 1, 2, 3, 4)    # averaged over these; use >= 10 for the final paper

    # --- data ---
    data_source: str = "synthetic"     # "synthetic" (runs anywhere) or "cwru"
    cwru_dir: str = "data/cwru"        # folder holding CWRU .mat files (see README)
    signal_length: int = 1024
    n_classes: int = 4                 # healthy, inner-race, outer-race, ball
    n_per_class_train: int = 600
    n_per_class_cal: int = 300
    n_per_class_test: int = 300
    severities: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # 0 = source, 1 = strongest shift
    shift_types: tuple = ("covariate", "label")         # both tracks run by default
    label_min_per_class: int = 25                       # floor so rare classes stay measurable

    # --- model ---
    epochs: int = 12
    batch_size: int = 128
    lr: float = 1e-3

    # --- conformal ---
    alpha: float = 0.10                # target miscoverage -> 90% coverage
    score_fn: str = "aps"              # "lac", "aps", or "raps"
    raps_lambda: float = 0.05
    raps_k_reg: int = 2
    aci_gamma: float = 0.02            # ACI learning rate

    # --- output ---
    out_dir: str = "results"
    fig_dir: str = "figures"


CFG = Config()
