"""chemdl — Near-infrared spectral scatter-correction theory, paired LOO gating, and Wiener shrinkage.

Modules included in this paper release:
  Theory & Gating : theory (Theorems 1–4, R^2_shape decomposition, Exact-PRESS LOO gate)
                    deploy_gate (inductive 3-state deployment workflow, AURC, asymmetric cost)
  Preprocessing   : classic (SNV / MSC / EMSC / RNV / SG / Detrend / Log1p)
                    transfer (decoupled shrinkage DSSNV, inductive wrappers, PLS, SPORT)
                    diffprep (5-step differentiable preprocessing & BilevelDiffPrepRidge)
  Utilities       : select / uq (calibration selection & drift/scale sanity monitoring)
  Data Loaders    : corn / tablets / tecator / cordoba / diesel
"""

__version__ = "2.0.0"

__all__ = [
    "theory",
    "deploy_gate",
    "classic",
    "transfer",
    "diffprep",
    "select",
    "uq",
    "corn",
    "tablets",
    "tecator",
    "cordoba",
    "diesel",
]
