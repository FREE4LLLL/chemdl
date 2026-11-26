# Selective Standard Normal Variate for Near-Infrared Calibration
**Analytic Attribution, Paired Decision Gating, and Continuous Scale Shrinkage**

Companion code, SHA-256 dataset manifests, preregistered audit logs, and figure-reproduction scripts.

---

## Core Contributions

1. **Analytic Sign Criterion & Residual Shape Decomposition (Theorem 1)**:
   Derives a closed-form first-order two-channel orthogonal SNR surrogate and critical coupling threshold $r_{\text{crit}}(\sigma_a; \alpha, \beta)$, paired with unsupervised spectral-shape residual decomposition $1 - R^2_{\text{shape}}$ to separate physical scatter from Beer–Lambert chemical amplitude variance.
2. **Calibration-Only Paired Exact Allen-PRESS Three-State Gate (`C4A`/`C4B` Nested)**:
   Computes calibration-only paired residuals to emit `IMPROVE`, `WORSEN`, or `ABSTAIN` with supervised fallback. Under the revised replicate-grouped nested 5-fold primary protocol (`C4A`), one multi-year `CRAW` wheat moisture error raises `Gate + SNV` $C_{10}$ to `50.58` vs `26.07` for standalone inner CV (`C1` single-$\gamma$: `15.17`), whereas year-grouping (`C4B`) recovers `0.56`. On 13 external tasks, `Gate + Inner-CV` lowers $C_{10}$ from `33.09` to `17.53` (`4/13` answered, all net gain from `AustrianSoil pH_CaCl2`).
3. **Label-Free Surrogate Wiener-Type Shrinkage & Unified Coupling Upper Bound (Theorems 3–4)**:
   Establishes $w_{s,\text{Wiener}}^\star = 1 - R^2_{\text{shape}}$ without consuming analyte labels $y$ or $w_s$ grid search. On 30 reused holdouts under unified grouped evaluation, PLS `Wiener-Type (1-split)` matches $T_{1,0}$ (`0.7535`, severe regrets `13→9`, $C_{10}=69.44$) and trails fixed $w_s = 0.20$ (`60.09`) while running `4.9×` faster than 11-pt CV (one-sided 12-source cluster $\mathrm{UCB}_{95} = +0.0544$, missing $\delta = 0.05$, supporting $\delta = 0.08$; Ridge `3.5×` faster, $\mathrm{UCB}_{95} = +0.0279 < \delta = 0.03$), bounded above by $w_{s,\text{unified}}^\star < w_{s,\text{Wiener}}^\star$ under $r \neq 0$.
4. **Physical Identifiability Domain**:
   Maps scalar scatter fidelity ($r = 0.9975$ against $f_{\text{adj,lin}}$), wavelength-tilt collapse, and 6-source/8-group real paired/replicate scans.

---

## Quick Start

```python
import numpy as np
from chemdl.deploy_gate import fit_snv_gated_workflow
from chemdl.theory import diagnose_snv_decision, estimate_physical_params_v2
from chemdl.transfer import dssnv_apply

# 1. Calibration-only 3-state SNV gate
diag = diagnose_snv_decision(X_cal, y_cal, groups_tr=groups_cal)
print(diag["decision"], diag["delta_loo_pct"], diag["t_loo"])

# 2. End-to-end inductive gated PLS model (single-sample & batch consistent)
predict_fn, info = fit_snv_gated_workflow(X_cal, y_cal, abstain_policy="inner_cv")
y_hat = predict_fn(X_test)

# 3. Label-free Wiener shrinkage w_s* = 1 - R^2_shape
phys = estimate_physical_params_v2(X_cal, y_cal)
ws_wiener = 1.0 - phys["r2_s_shape"]
s_bar = float(np.clip(X_cal.std(axis=1), 1e-12, None).mean())
X_cal_w = dssnv_apply(X_cal, wc=1.0, ws=ws_wiener, s_bar=s_bar)
X_test_w = dssnv_apply(X_test, wc=1.0, ws=ws_wiener, s_bar=s_bar)
```

---

## Installation & Three-Tier Verification

```bash
git clone https://github.com/FREE4LLLL/chemdl.git
cd chemdl
pip install -r requirements-lock.txt   # or requirements.txt
pip install --no-deps -e .

# Tier 1 (Offline Verification, zero downloads): unit tests + frozen hashes + 26 JSONs + Tables 1-5
pytest -q -W default
python reproduce_paper.py --verify --tables

# Tier 2 (Offline Figure Regeneration): regenerate all 7 publication PNGs from archived JSONs
python reproduce_paper.py --plot-all

# Tier 3 (Full Raw-Data Re-training): fetch public datasets & re-run full revision benchmark
python scripts/fetch_data.py --verify
python _audit/fetch_round3_unseen.py --verify
python _audit/bench_revision_v2_2026_11_26.py
```

---

## Repository Structure

| Path | Description |
|---|---|
| `chemdl/theory.py` | Theorems 1–4, $R^2_{\text{shape}}$ decomposition, exact Allen-PRESS LOO, `diagnose_snv_decision` (hash-locked) |
| `chemdl/deploy_gate.py` | Inductive 3-state deployment gate, `GroupKFold` handling, AURC, $C_w$, sanity diagnostics |
| `chemdl/transfer.py` | Decoupled shrinkage `dssnv_apply`, inductive preprocessing wrappers, PLS, SPORT |
| `chemdl/diffprep.py`, `classic.py`, `select.py`, `uq.py` | 5-step differentiable preprocessing, `BilevelDiffPrepRidge`, classical baselines, selection & UQ |
| `chemdl/{corn,tablets,tecator,cordoba,diesel}.py` | Public NIR benchmark loaders |
| `data/MANIFEST.yaml`, `data/MANIFEST_ROUND3.yaml` | Full 64-hex SHA-256 checksums and open-license provenance for all public datasets |
| `scripts/fetch_data.py`, `_audit/fetch_round3_unseen.py` | Dataset download and SHA-256 verification utilities (`data_ext/` is not checked into git) |
| `_audit/` | Preregistered Phase 1/2 scripts, revision benchmarks (incl. `bench_revision_v2_2026_11_26.py`), and 26 archived JSON artifacts |
| `requirements-lock.txt` | Exact pinned environment versions (Python 3.13.9, NumPy, SciPy, scikit-learn, etc.) |
| `paper_assets/` | 4 figure-generation scripts and the 7 publication PNG figures |
| `tests/test_paper_core.py` | Self-contained test suite (no external dataset download required) |

---

## Tables & Figures Mapping

| Paper Item | Primary Script / Archived Artifact |
|---|---|
| **Fig 1** | `paper_assets/plot_manuscript_figs_1_3_4.py` (`fig_ms1_framework.png`) |
| **Table 1** | `chemdl/theory.py::estimate_physical_params_v2` |
| **Table 2 / Fig 2** | `_audit/bench_review_p0_2026_11_23.py`, `_audit/bench_theory_dssnv_2026_11_20.py`, `paper_assets/plot_fig_snv_phase_diagram.py` |
| **Fig 3** | `_audit/prereg_theory_blind_phase2_unblind.py`, `_audit/prereg_snv_gate_round2_phase2_unblind.py`, `paper_assets/plot_manuscript_figs_1_3_4.py` |
| **Table 3 / Fig 4 / Fig 5** | `_audit/bench_revision_v2_2026_11_26.py`, `_audit/res_revision_v2_2026_11_26.json`, `_audit/prereg_deploy_holdout_phase2_unblind.py`, `_audit/prereg_round3_unseen_phase2_unblind.py`, `paper_assets/plot_manuscript_figs_1_3_4.py`, `paper_assets/plot_manuscript_figs_6_7.py` |
| **Table 4 / Fig 6** | `_audit/bench_revision_v2_2026_11_26.py`, `_audit/res_revision_v2_2026_11_26.json`, `_audit/prereg_ws_closed_value_phase2_unblind.py`, `_audit/prereg_wiener_ws_phase2_unblind.py`, `paper_assets/plot_manuscript_figs_6_7.py` |
| **Table 5 / Fig 7** | `_audit/bench_r2_shape_identifiability_2026_11_25.py`, `paper_assets/plot_fig13_r2_shape_identifiability.py` |

---

## External Datasets (On-Demand Download)

Raw spectral datasets are excluded from Git to keep the repository lightweight. All benchmarks use permissive open licenses (MIT, CC0 1.0, CC-BY 4.0, or public academic benchmarks) and are verified via SHA-256:

```bash
python scripts/fetch_data.py --verify
python _audit/fetch_round3_unseen.py --verify
```

---

## License & Citation

Released under the [MIT License](LICENSE). See [CITATION.cff](CITATION.cff) for citation metadata.
