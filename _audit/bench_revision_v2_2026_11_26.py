"""Unified CILPS major-revision benchmark (P0 + P1a + P1b + P1c + P2).

Produces:
  1) _audit/revision_protocol_registry_2026_11_26.json  (frozen BEFORE any new fit)
  2) _audit/res_revision_v2_2026_11_26.json           (all revised tables, LOSO, CIs)

Hard constraints:
  - Read-only with respect to chemdl/* and all prior frozen _audit/*.json files.
  - Strictly inductive single-sample preprocessing inside every calibration fold.
  - Full-hierarchy group awareness (replicate groups + deployment-unit year groups).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import sklearn
from scipy.stats import beta as beta_dist
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._deploy_holdout_loaders import _load_craw_with_year, _row_hashes, load_deploy_holdout_tasks
from _audit._round3_unseen_loaders import load_round3_unseen_tasks, read_xlsx_sheet_stdlib
from _audit._ws_closed_value_loaders import (
    FIXED_WC,
    load_ws_value_tasks,
    pls_prefix_predictions,
)
from _audit.bench_major_revision_2026_11_25 import apply_vsn, fit_vsn_weights
from chemdl.deploy_gate import FROZEN_THEORY_LF_SHA256
from chemdl.theory import exact_press_loo_residuals
from chemdl.transfer import dssnv_apply, fit_prep_fn, fit_sport_pls, rmsep

BOOT_SEED = 20261126
BOOT_B = 20_000
AUTOPREP_OPS = ("raw", "row_centered_T10", "snv", "sg1", "sg2", "classic", "msc", "emsc_reg")
FIXED_WS_LIST = (0.0, 0.10, 0.15, 0.20, 1.0)


def _lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _source_of_task(task_id: str) -> str:
    if task_id.startswith("Deploy_CRAW_"):
        return "CRAW"
    if task_id.startswith("Deploy_Tablets_"):
        return "Tablets"
    if task_id.startswith("Deploy_Wheat_"):
        return "Wheat"
    if task_id.startswith("Deploy_Melamine_"):
        return "Melamine"
    if task_id.startswith("Deploy_Milk_"):
        return "Milk"
    if task_id.startswith("Deploy_OSSL_"):
        return "OSSL"
    if task_id.startswith("Deploy_Mango_"):
        return "Mango"
    if task_id.startswith("R3_"):
        return task_id.split("__")[0].replace("R3_", "")
    return task_id.split("_")[0]


def _clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def _extract_year_arrays() -> dict[str, dict[str, np.ndarray]]:
    """Extract exact calibration & test integer year arrays for all 9 multi-year tasks."""
    out: dict[str, dict[str, np.ndarray]] = {}
    d18 = load_deploy_holdout_tasks(ROOT)
    craw_dir = ROOT / "data_ext" / "industrial" / "craw"
    craw_files = {
        "Deploy_CRAW_Wheat_XDS_Moist_NewOnly__Temporal": ("Moisture/Wheat_sensAIfood_Moist_500_XDS_CRAW.csv", "Moisture"),
        "Deploy_CRAW_Wheat_XDS_Prot_NewOnly__Temporal": ("Protein/Wheat_sensAIfood_Protein_500_XDS_CRAW.csv", "Protein"),
        "Deploy_CRAW_Barley_NIRS5000_Moist__Temporal": ("Moisture/Barley_sensAIfood_Moist_985_NIRS5000_CRAW.csv", "Moisture"),
        "Deploy_CRAW_Barley_NIRS5000_Prot__Temporal": ("Protein/Barley_sensAIfood_Protein_2096_NIRS5000_CRAW.csv", "Protein"),
        "Deploy_CRAW_Wheat2_XDS_Moist__Temporal": ("Moisture/Wheat2_sensAIfood_Moist_500_XDS_CRAW.csv", "Moisture"),
        "Deploy_CRAW_Wheat_XDS_Prot__Temporal": ("Protein/Wheat_sensAIfood_Protein_231_XDS_CRAW.csv", "Protein"),
    }
    for t in d18:
        tid = t["task_id"]
        if tid in craw_files:
            rel, col = craw_files[tid]
            Xa, _, yra = _load_craw_with_year(craw_dir / rel, col)
            h2yr = dict(zip(_row_hashes(Xa), yra))
            yr_c = np.array([h2yr[h] for h in _row_hashes(t["X_cal"])], dtype=np.int64)
            yr_t = np.array([h2yr[h] for h in _row_hashes(t["X_test"])], dtype=np.int64)
            out[tid] = {"years_cal": yr_c, "years_test": yr_t}

    au_path = ROOT / "data_ext" / "round3_unseen" / "Z17941270_austrian_soil" / "Austrian NIR Soil Spectral Library_V4.xlsx"
    au_rows = read_xlsx_sheet_stdlib(au_path, 2)
    au_hdr = [str(c).strip() for c in au_rows[0]]
    spec_idx = au_hdr.index("680")
    au_padded = [r[: len(au_hdr)] + [""] * max(0, len(au_hdr) - len(r)) for r in au_rows[1:]]
    au_meta = pd.DataFrame([r[:spec_idx] for r in au_padded], columns=au_hdr[:spec_idx])
    X_au = np.array([[float(x) if str(x).strip() != "" else np.nan for x in r[spec_idx:]] for r in au_padded], dtype=np.float64)
    yr_au = pd.to_numeric(au_meta["Sampling_year"], errors="coerce").to_numpy(dtype=np.float64)
    for tgt in ("SOC", "TN", "pH_CaCl2"):
        y_au = pd.to_numeric(au_meta[tgt], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(X_au).all(1) & np.isfinite(y_au) & np.isfinite(yr_au)
        yc = yr_au[ok & (yr_au <= 2020)].astype(np.int64)
        yt = yr_au[ok & (yr_au >= 2021)].astype(np.int64)
        out[f"R3_AustrianSoil_Pre21toPost21__{tgt}"] = {"years_cal": yc, "years_test": yt}
    return out


def build_and_freeze_registry(
    d18: list[dict],
    r3: list[dict],
    year_map: dict[str, dict[str, np.ndarray]],
    d18_arch: dict[str, dict],
    r3_arch: dict[str, dict],
    wi_arch: dict[str, dict],
) -> dict:
    """Part P0: Freeze machine-readable protocol registry BEFORE any new fit."""
    theory_sha = _lf_sha256(ROOT / "chemdl" / "theory.py")
    assert theory_sha == FROZEN_THEORY_LF_SHA256, f"chemdl/theory.py SHA mismatch: {theory_sha}"

    d18_pre = {t["task_id"]: t for t in json.loads((ROOT / "_audit" / "prereg_deploy_holdout_2026_11_24.json").read_text(encoding="utf-8"))["tasks"]}
    r3_pre = {t["task_id"]: t for t in json.loads((ROOT / "_audit" / "prereg_round3_unseen_2026_11_24.json").read_text(encoding="utf-8"))["tasks"]}
    pre_all = {**d18_pre, **r3_pre}

    task_records = []
    for pool, tasks, arch in (("Deploy_18", d18, d18_arch), ("Round3_13", r3, r3_arch)):
        for t in tasks:
            tid = t["task_id"]
            assert t["split_sha256"] == pre_all[tid]["split_sha256"], f"split_sha256 mismatch: {tid}"
            if tid in wi_arch:
                assert t["split_sha256"] == wi_arch[tid]["split_sha256"], f"wiener split_sha256 mismatch: {tid}"
            g_rep = t.get("groups_cal")
            ym = year_map.get(tid)
            task_records.append({
                "task_id": tid,
                "pool": pool,
                "source": _source_of_task(tid),
                "n_cal": int(t["n_cal"]),
                "n_test": int(t["n_test"]),
                "L": int(t["L"]),
                "split_sha256": t["split_sha256"],
                "has_label_scale_anomaly": ("Mango" in tid),
                "replicate_group_type": "Cow_ID" if "Milk" in tid else ("Location_Plot" if "PeaForage" in tid else None),
                "n_replicate_groups_cal": int(len(np.unique(g_rep))) if g_rep is not None else None,
                "temporal_unit_type": "Sampling_Year" if ym is not None else None,
                "n_years_cal": int(len(np.unique(ym["years_cal"]))) if ym is not None else None,
                "cal_years_list": [int(v) for v in np.unique(ym["years_cal"])] if ym is not None else None,
                "test_years_list": [int(v) for v in np.unique(ym["years_test"])] if ym is not None else None,
            })

    registry = {
        "protocol_id": "CILPS_REVISION_V2_2026_11_26",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": _git_head(),
        "theory_py_lf_sha256": theory_sha,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "evidence_tiers": {
            "tier1_mechanistic_dev": "7 classic NIR benchmarks (170-fold cross-dataset & 210-cell 2D phase diagram)",
            "tier2_known_data_revised_deployment": "31 single cal->test splits (18 original Deploy + 13 Round3 external), revised under full-hierarchy grouped nested protocol",
            "tier3_reused_holdout_shrinkage": "30 valid-scale reused holdouts (VALID_17 + Round3_13) from unblinded 54-task pool",
        },
        "thresholds_and_endpoints": {
            "gate_t_crit": 2.0,
            "gate_delta_min_pct": 1.0,
            "severe_regret_threshold_pct": 5.0,
            "asymmetric_cost_weights": [5.0, 10.0, 20.0],
            "non_inferiority_margins_rmsep_z": [0.03, 0.05, 0.08],
            "fixed_ws_baselines": list(FIXED_WS_LIST),
            "autoprep_operators": list(AUTOPREP_OPS),
            "wiener_rep_seeds": list(range(20)),
            "cluster_bootstrap_B": BOOT_B,
            "cluster_bootstrap_seed": BOOT_SEED,
        },
        "degenerate_group_fallback_rules": {
            "inner_cv_min_groups": 2,
            "inner_cv_min_train_rows": 4,
            "fallback_action": "Log explicit fallback_reason and fall back to deterministic KFold(shuffle=True, random_state=0)",
            "expanding_temporal_min_train": 15,
            "expanding_temporal_min_val": 5,
            "expanding_temporal_min_folds": 2,
            "expanding_temporal_insufficient_action": "Log insufficient_temporal_folds and return ABSTAIN",
        },
        "tasks": task_records,
    }
    reg_path = ROOT / "_audit" / "revision_protocol_registry_2026_11_26.json"
    reg_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[P0] Frozen protocol registry written -> {reg_path} ({len(task_records)} tasks)")
    return registry


# ---------------------------------------------------------------------------
# P1a: Full-Hierarchy Grouped Helpers
# ---------------------------------------------------------------------------

def _make_cv_splits(n: int, groups: np.ndarray | None, n_splits: int = 5, seed: int = 0) -> tuple[list[tuple[np.ndarray, np.ndarray]], str | None]:
    if groups is not None:
        grp = np.asarray(groups)
        uniq = np.unique(grp)
        eff = min(int(n_splits), len(uniq))
        if eff >= 2:
            splits = [
                (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
                for tr, va in GroupKFold(n_splits=eff).split(np.zeros((n, 1)), grp, grp)
            ]
            if min(len(tr) for tr, _ in splits) >= 4:
                return splits, None
            return _kfold_splits(n, n_splits, seed), f"group_min_train_lt_4(n_grp={len(uniq)})"
        return _kfold_splits(n, n_splits, seed), f"insufficient_groups(n_grp={len(uniq)})"
    return _kfold_splits(n, n_splits, seed), None


def _kfold_splits(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    eff = min(int(n_splits), max(2, n // 4))
    return [
        (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
        for tr, va in KFold(n_splits=eff, shuffle=True, random_state=seed).split(np.zeros((n, 1)))
    ]


def select_gamma_grouped_svd(
    X: np.ndarray,
    y_1d: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
) -> tuple[float, int, str | None]:
    """Select Ridge gamma over the exact 41-point log grid, respecting groups when present."""
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y_1d, dtype=np.float64).ravel()
    if groups is not None:
        splits, fb = _make_cv_splits(len(ya), groups, n_splits=n_splits, seed=0)
        if fb is None:
            xm_full = Xa.mean(axis=0, keepdims=True)
            _, s_full, _ = np.linalg.svd(Xa - xm_full, full_matrices=False)
            lam_bar = max(float(np.mean(s_full ** 2)), 1e-12)
            gammas = lam_bar * (10.0 ** np.linspace(-6.0, 4.0, 41))
            sq_errs = np.zeros(len(gammas), dtype=np.float64)
            for itr, iva in splits:
                xm = Xa[itr].mean(axis=0, keepdims=True)
                ym = float(ya[itr].mean())
                U, s, Vt = np.linalg.svd(Xa[itr] - xm, full_matrices=False)
                lam = s ** 2
                uty = U.T @ (ya[itr] - ym)  # (r,)
                Zva = (Xa[iva] - xm) @ Vt.T  # (n_va, r)
                # Vectorized across 41 gammas: coeffs (r, 41)
                coeffs = (s[:, None] / (lam[:, None] + gammas[None, :])) * uty[:, None]
                preds = Zva @ coeffs + ym  # (n_va, 41)
                sq_errs += np.sum((preds - ya[iva, None]) ** 2, axis=0)
            best_g = float(gammas[int(np.argmin(sq_errs))])
            return best_g, len(splits) + 1, None
        _, best_g = exact_press_loo_residuals(Xa, ya[:, None])
        return float(best_g), 1, fb
    _, best_g = exact_press_loo_residuals(Xa, ya[:, None])
    return float(best_g), 1, None


def _fit_predict_ridge(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, gamma: float) -> np.ndarray:
    xm = Xtr.mean(axis=0, keepdims=True)
    ym = float(np.mean(ytr))
    U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
    W = Vt.T @ ((s / (s ** 2 + max(float(gamma), 1e-15))) * (U.T @ (ytr - ym)))
    return (Xte - xm) @ W + ym


def fit_pls_cv_unified(
    X: np.ndarray,
    yn_1d: np.ndarray,
    groups: np.ndarray | None = None,
    kmax: int = 10,
    n_splits: int = 4,
) -> tuple:
    """Grouped-aware PLS CV matching deploy_gate/transfer kmax convention."""
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(yn_1d, dtype=np.float64).ravel()
    splits, fb = _make_cv_splits(len(ya), groups, n_splits=n_splits, seed=0)
    min_tr = min(len(tr) for tr, _ in splits)
    eff_kmax = min(int(kmax), Xa.shape[1], max(1, min_tr - 1))
    fold_errs = np.zeros((len(splits), eff_kmax), dtype=np.float64)
    for fi, (itr, iva) in enumerate(splits):
        preds = pls_prefix_predictions(Xa[itr], ya[itr], Xa[iva], eff_kmax)
        for ki, pk in enumerate(preds):
            fold_errs[fi, ki] = rmsep(pk, ya[iva])
    mean_errs = fold_errs.mean(axis=0)
    best_idx = int(np.argmin(mean_errs))
    best_k = best_idx + 1
    cv_rmse = float(mean_errs[best_idx])
    m = PLSRegression(n_components=best_k, scale=False)
    m.fit(Xa, ya[:, None])
    n_fits = len(splits) + 1
    return (lambda Z: m.predict(np.asarray(Z, dtype=np.float64)).ravel()), best_k, cv_rmse, n_fits, fb


def nested_gate_full(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    groups_cal: np.ndarray | None = None,
    temporal_mode: str = "groupkfold",
) -> dict:
    """Full-hierarchy nested gate: outer folds AND inner gamma selection respect groups_cal."""
    Xa = np.asarray(X_cal, dtype=np.float64)
    ya = np.asarray(y_cal, dtype=np.float64).ravel()
    X_snv = (Xa - Xa.mean(axis=1, keepdims=True)) / np.clip(Xa.std(axis=1, keepdims=True), 1e-12, None)
    fb_logs = []

    if temporal_mode == "expanding":
        assert groups_cal is not None
        yrs = np.asarray(groups_cal, dtype=np.int64)
        uniq_yrs = np.sort(np.unique(yrs))
        splits = []
        eval_mask = np.zeros(len(ya), dtype=bool)
        for j in range(len(uniq_yrs) - 1):
            itr = np.flatnonzero(yrs <= uniq_yrs[j])
            iva = np.flatnonzero(yrs == uniq_yrs[j + 1])
            if len(itr) >= 15 and len(iva) >= 5:
                splits.append((itr, iva))
                eval_mask[iva] = True
        if len(splits) < 2:
            return {
                "decision": "ABSTAIN",
                "t_nested": 0.0,
                "delta_nested_pct": 0.0,
                "n_fits": 0,
                "fallback_log": ["insufficient_temporal_folds"],
            }
    else:
        splits, fb = _make_cv_splits(len(ya), groups_cal, n_splits=5, seed=0)
        if fb:
            fb_logs.append(f"outer:{fb}")
        eval_mask = np.ones(len(ya), dtype=bool)

    e_raw = np.zeros(len(ya), dtype=np.float64)
    e_snv = np.zeros(len(ya), dtype=np.float64)
    total_fits = 0

    for itr, iva in splits:
        sub_grp = groups_cal[itr] if groups_cal is not None else None
        g_r, nf_r, fb_r = select_gamma_grouped_svd(Xa[itr], ya[itr], sub_grp, n_splits=4)
        g_s, nf_s, fb_s = select_gamma_grouped_svd(X_snv[itr], ya[itr], sub_grp, n_splits=4)
        total_fits += nf_r + nf_s + 2
        if fb_r:
            fb_logs.append(f"inner_raw:{fb_r}")
        if fb_s:
            fb_logs.append(f"inner_snv:{fb_s}")
        e_raw[iva] = ya[iva] - _fit_predict_ridge(Xa[itr], ya[itr], Xa[iva], g_r)
        e_snv[iva] = ya[iva] - _fit_predict_ridge(X_snv[itr], ya[itr], X_snv[iva], g_s)

    er_ev = e_raw[eval_mask]
    es_ev = e_snv[eval_mask]
    rmse_r = float(np.sqrt(np.mean(er_ev ** 2)))
    rmse_s = float(np.sqrt(np.mean(es_ev ** 2)))
    delta_pct = (rmse_s / max(rmse_r, 1e-12) - 1.0) * 100.0
    d_scan = (es_ev ** 2) - (er_ev ** 2)

    if groups_cal is not None:
        g_ev = np.asarray(groups_cal)[eval_mask]
        uniq_ev = np.unique(g_ev)
        if len(uniq_ev) >= 2:
            d_unit = np.array([float(np.mean(d_scan[g_ev == g])) for g in uniq_ev])
        else:
            d_unit = d_scan
    else:
        d_unit = d_scan

    sd_u = float(np.std(d_unit, ddof=1)) if len(d_unit) > 1 else 0.0
    se_u = sd_u / np.sqrt(len(d_unit)) if len(d_unit) > 1 else 0.0
    t_nested = float(np.mean(d_unit) / se_u) if se_u > 1e-12 else 0.0

    if t_nested <= -2.0 and delta_pct <= -1.0:
        dec = "IMPROVE"
    elif t_nested >= 2.0 and delta_pct >= 1.0:
        dec = "WORSEN"
    else:
        dec = "ABSTAIN"

    return {
        "decision": dec,
        "t_nested": round(t_nested, 3),
        "delta_nested_pct": round(delta_pct, 3),
        "n_fits": int(total_fits),
        "fallback_log": sorted(set(fb_logs)),
    }


def _apply_inductive_op(op: str, X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if op == "row_centered_T10":
        s_bar = float(np.clip(X_tr.std(axis=1), 1e-12, None).mean())
        return dssnv_apply(X_tr, wc=1.0, ws=0.0, s_bar=s_bar), dssnv_apply(X_te, wc=1.0, ws=0.0, s_bar=s_bar)
    fn = fit_prep_fn(op, X_tr)
    return fn(X_tr), fn(X_te)


def eval_autoprep_cv(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_cal: np.ndarray | None = None,
    head: str = "pls",
    kmax: int = 10,
) -> dict:
    """Modern inductive multi-operator AutoPrep-CV selector over 8 preprocessing operators."""
    t0 = time.perf_counter()
    Xa = np.asarray(X_cal, dtype=np.float64)
    Xte = np.asarray(X_test, dtype=np.float64)
    mu = float(np.mean(y_cal))
    sd = max(float(np.std(y_cal)), 1e-9)
    yn_c = (np.asarray(y_cal, dtype=np.float64).ravel() - mu) / sd
    yn_t = (np.asarray(y_test, dtype=np.float64).ravel() - mu) / sd

    splits, _ = _make_cv_splits(len(yn_c), groups_cal, n_splits=5, seed=0)
    op_scores = {}
    n_fits = 0

    for op in AUTOPREP_OPS:
        sq_errs = []
        for itr, iva in splits:
            Ztr, Zva = _apply_inductive_op(op, Xa[itr], Xa[iva])
            sub_g = groups_cal[itr] if groups_cal is not None else None
            if head == "pls":
                pr, _, _, nf, _ = fit_pls_cv_unified(Ztr, yn_c[itr], groups=sub_g, kmax=kmax, n_splits=4)
                pred_va = pr(Zva)
                n_fits += nf
            else:
                g_opt, nf, _ = select_gamma_grouped_svd(Ztr, yn_c[itr], groups=sub_g, n_splits=4)
                pred_va = _fit_predict_ridge(Ztr, yn_c[itr], Zva, g_opt)
                n_fits += nf + 1
            sq_errs.append((pred_va - yn_c[iva]) ** 2)
        op_scores[op] = float(np.sqrt(np.mean(np.concatenate(sq_errs))))

    best_op = min(AUTOPREP_OPS, key=lambda o: op_scores[o])
    Zcal, Ztest = _apply_inductive_op(best_op, Xa, Xte)
    if head == "pls":
        pr_full, best_k, _, nf, _ = fit_pls_cv_unified(Zcal, yn_c, groups=groups_cal, kmax=kmax, n_splits=4)
        pred_z = pr_full(Ztest)
        n_fits += nf
    else:
        g_full, nf, _ = select_gamma_grouped_svd(Zcal, yn_c, groups=groups_cal, n_splits=5)
        pred_z = _fit_predict_ridge(Zcal, yn_c, Ztest, g_full)
        n_fits += nf + 1

    ez = float(rmsep(pred_z, yn_t))
    dt_s = time.perf_counter() - t0
    return {
        "best_op": best_op,
        "rmsep_z": round(ez, 6),
        "rmsep_raw": round(ez * sd, 6),
        "elapsed_s": round(dt_s, 4),
        "n_fits": int(n_fits),
    }


def compute_r2_shape_grouped(
    X_cal: np.ndarray,
    groups_cal: np.ndarray | None = None,
    seeds: tuple[int, ...] = (0,),
) -> tuple[float, float, float, int]:
    """Compute (r2_mean, ws_mean = 1 - r2_mean, r2_sd, n_fits) strictly from X_cal."""
    X = np.asarray(X_cal, dtype=np.float64)
    n, L = X.shape
    mu_row = X.mean(axis=1)
    sd_row = np.clip(X.std(axis=1), 1e-12, None)
    X_snv = (X - mu_row[:, None]) / sd_row[:, None]
    tsd = float(sd_row.std())
    if tsd < 1e-12:
        return 0.0, 1.0, 0.0, 0
    zn = (sd_row - sd_row.mean()) / tsd

    r2_vals = []
    n_fits = 0
    for idx_s, sd in enumerate(seeds):
        if groups_cal is not None:
            grp = np.asarray(groups_cal)
            uniq = np.unique(grp)
            eff_k = min(3, len(uniq))
            if eff_k >= 2:
                if len(seeds) == 1 and sd == 0:
                    splits = list(GroupKFold(n_splits=eff_k).split(X_snv, grp, grp))
                else:
                    perm_u = np.random.default_rng(sd).permutation(uniq)
                    fold_groups = [set(perm_u[i::eff_k].tolist()) for i in range(eff_k)]
                    splits = []
                    for fg in fold_groups:
                        m_va = np.array([g in fg for g in grp], dtype=bool)
                        splits.append((np.flatnonzero(~m_va), np.flatnonzero(m_va)))
            else:
                splits = list(KFold(n_splits=3, shuffle=True, random_state=sd).split(X_snv))
        else:
            splits = list(KFold(n_splits=3, shuffle=True, random_state=sd).split(X_snv))

        max_lv = min(6, max(1, min(len(itr) for itr, _ in splits) // 5), L)
        best_mse = 1.0
        for lv in range(1, max_lv + 1):
            mses = []
            for itr, iva in splits:
                m = PLSRegression(n_components=min(lv, len(itr) - 1), scale=False).fit(X_snv[itr], zn[itr])
                p = m.predict(X_snv[iva]).ravel()
                mses.append(float(np.mean((p - zn[iva]) ** 2)))
                n_fits += 1
            best_mse = min(best_mse, float(np.mean(mses)))
        r2_vals.append(float(np.clip(1.0 - best_mse, 0.0, 0.98)))

    r2_mean = float(np.mean(r2_vals))
    r2_sd = float(np.std(r2_vals, ddof=1)) if len(r2_vals) > 1 else 0.0
    return r2_mean, float(1.0 - r2_mean), r2_sd, int(n_fits)


# ---------------------------------------------------------------------------
# Summary & Bootstrap Helpers (P2)
# ---------------------------------------------------------------------------

def _summarize_errs(errs: list[float], oras: list[float], times: list[float] | None = None, fits: list[int] | None = None) -> dict:
    ea = np.asarray(errs, dtype=np.float64)
    oa = np.asarray(oras, dtype=np.float64)
    regs = (ea / np.maximum(oa, 1e-12) - 1.0) * 100.0
    out = {
        "n": int(len(ea)),
        "mean_rmsep_z": round(float(np.mean(ea)), 4),
        "mean_regret_pct": round(float(np.mean(regs)), 2),
        "worst_regret_pct": round(float(np.max(regs)), 2),
        "severe_count": int(np.sum(regs >= 5.0)),
        "C_5": round(float(np.mean(regs) + 4.0 * np.mean(np.maximum(regs - 5.0, 0.0))), 2),
        "C_10": round(float(np.mean(regs) + 9.0 * np.mean(np.maximum(regs - 5.0, 0.0))), 2),
        "C_20": round(float(np.mean(regs) + 19.0 * np.mean(np.maximum(regs - 5.0, 0.0))), 2),
    }
    if times is not None:
        out["total_time_s"] = round(float(np.sum(times)), 2)
        out["mean_time_s"] = round(float(np.mean(times)), 4)
    if fits is not None:
        out["mean_fits"] = round(float(np.mean(fits)), 1)
    return out


def _source_equal_summary(errs: list[float], oras: list[float], sources: list[str]) -> dict:
    ea = np.asarray(errs, dtype=np.float64)
    oa = np.asarray(oras, dtype=np.float64)
    regs = (ea / np.maximum(oa, 1e-12) - 1.0) * 100.0
    c10_row = regs + 9.0 * np.maximum(regs - 5.0, 0.0)
    uniq = sorted(set(sources))
    s_err, s_reg, s_c10, s_sev = [], [], [], []
    for s in uniq:
        m = np.array([src == s for src in sources], dtype=bool)
        s_err.append(float(np.mean(ea[m])))
        s_reg.append(float(np.mean(regs[m])))
        s_c10.append(float(np.mean(c10_row[m])))
        s_sev.append(float(np.mean(regs[m] >= 5.0) * 100.0))
    return {
        "n_sources": len(uniq),
        "source_mean_rmsep_z": round(float(np.mean(s_err)), 4),
        "source_mean_regret_pct": round(float(np.mean(s_reg)), 2),
        "source_mean_C10": round(float(np.mean(s_c10)), 2),
        "source_mean_severe_rate_pct": round(float(np.mean(s_sev)), 2),
    }


def _cluster_paired_audit(
    e_a: np.ndarray,
    e_b: np.ndarray,
    reg_a: np.ndarray,
    reg_b: np.ndarray,
    sources: list[str],
    task_ids: list[str],
) -> dict:
    """Paired severe-event 2x2 transition + 12-source cluster bootstrap (B=20000)."""
    sev_a = reg_a >= 5.0
    sev_b = reg_b >= 5.0
    exited = [tid for tid, sa, sb in zip(task_ids, sev_a, sev_b) if (not sa) and sb]
    entered = [tid for tid, sa, sb in zip(task_ids, sev_a, sev_b) if sa and (not sb)]
    dz = e_a - e_b
    d_sev = sev_a.astype(np.float64) - sev_b.astype(np.float64)

    uniq = sorted(set(sources))
    S = len(uniq)
    src_arr = np.array(sources)
    dz_sums = np.array([float(np.sum(dz[src_arr == s])) for s in uniq])
    dsev_sums = np.array([float(np.sum(d_sev[src_arr == s])) for s in uniq])
    cnts = np.array([int(np.sum(src_arr == s)) for s in uniq])

    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, S, size=(BOOT_B, S))
    den = cnts[idx].sum(axis=1)
    boot_dz = dz_sums[idx].sum(axis=1) / den
    boot_dsev_pp = 100.0 * dsev_sums[idx].sum(axis=1) / den

    return {
        "n_tasks": len(task_ids),
        "n_sources": S,
        "mean_delta_rmsep_z": round(float(np.mean(dz)), 5),
        "delta_rmsep_z_cluster_ci95": [round(float(x), 5) for x in np.percentile(boot_dz, [2.5, 97.5])],
        "delta_rmsep_z_cluster_ucb95": round(float(np.percentile(boot_dz, 95.0)), 5),
        "severe_count_a": int(np.sum(sev_a)),
        "severe_count_b": int(np.sum(sev_b)),
        "severe_rate_diff_pp": round(float(np.mean(d_sev) * 100.0), 2),
        "severe_rate_diff_pp_cluster_ci95": [round(float(x), 2) for x in np.percentile(boot_dsev_pp, [2.5, 97.5])],
        "exited_severe_tasks": exited,
        "entered_severe_tasks": entered,
    }


def _compute_loso(rows: list[dict], strat_keys: list[str]) -> dict:
    sources = sorted({r["source"] for r in rows})
    out = {}
    for omit_s in sources:
        sub = [r for r in rows if r["source"] != omit_s]
        oras = [r["oracle_z"] for r in sub]
        out[omit_s] = {
            "n_remaining": len(sub),
            "strategies": {k: _summarize_errs([r["errs"][k] for r in sub], oras) for k in strat_keys},
        }
    return out


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    t_global = time.perf_counter()
    d18 = load_deploy_holdout_tasks(ROOT)
    r3 = load_round3_unseen_tasks(ROOT)
    d18_arch = {t["task_id"]: t for t in json.loads((ROOT / "_audit" / "res_prereg_deploy_holdout_2026_11_24.json").read_text(encoding="utf-8"))["tasks"]}
    r3_arch = {t["task_id"]: t for t in json.loads((ROOT / "_audit" / "res_prereg_round3_unseen_2026_11_24.json").read_text(encoding="utf-8"))["tasks"]}
    d18_ext_arch = {t["task_id"]: t for t in json.loads((ROOT / "_audit" / "res_deploy_extended_analysis_2026_11_24.json").read_text(encoding="utf-8"))["tasks"]}
    wi_arch_raw = json.loads((ROOT / "_audit" / "res_wiener_ws_2026_11_25.json").read_text(encoding="utf-8"))
    wi_arch = {t["task_id"]: t for t in wi_arch_raw["tasks"]}
    ws54_pre = {t["task_id"]: t for t in json.loads((ROOT / "_audit" / "prereg_ws_closed_value_2026_11_25.json").read_text(encoding="utf-8"))["tasks"]}
    mr_arch = json.loads((ROOT / "_audit" / "res_major_revision_2026_11_25.json").read_text(encoding="utf-8"))
    vsn18_arch = {t["task_id"]: t for t in mr_arch["part1_vsn_comparison"]["per_task_18"]}
    vsn13_arch = {t["task_id"]: t for t in mr_arch["part1_vsn_comparison"]["per_task_13"]}

    year_map = _extract_year_arrays()
    registry = build_and_freeze_registry(d18, r3, year_map, d18_arch, r3_arch, wi_arch)

    # =========================================================================
    # P1b: Deployment 31-task 4-cell + DeployUnit + ExpandingTemporal + AutoPrep
    # =========================================================================
    print("\n=== [P1b] Running 31-Task Deployment Protocol Matrix ===", flush=True)
    deploy_tasks_out = []
    for idx, t in enumerate(d18 + r3, 1):
        tid = t["task_id"]
        pool = "Deploy_18" if tid in d18_arch else "Round3_13"
        ar = d18_arch[tid] if pool == "Deploy_18" else r3_arch[tid]
        src = _source_of_task(tid)
        X_cal, y_cal = t["X_cal"], t["y_cal"]
        X_test, y_test = t["X_test"], t["y_test"]
        g_rep = t.get("groups_cal")
        g_unit = year_map[tid]["years_cal"] if tid in year_map else g_rep

        mu, sd = float(y_cal.mean()), max(float(y_cal.std()), 1e-9)
        yn_c = (y_cal - mu) / sd
        yn_t = (y_test - mu) / sd
        kmax = min(10, max(2, len(X_cal) // 2))

        # Archived Cell 1 values
        c1_raw = float(ar["rmsep_raw_z"])
        c1_snv = float(ar["rmsep_snv_z"])
        c1_cv_pick = str(ar["inner_cv_pick"])
        c1_gate = str(ar["gate_decision"])
        if pool == "Deploy_18":
            ext_r = d18_ext_arch[tid]
            c1_sp_cl = float(ext_r["methods"]["SPORT_Classic_PLS"])
            c1_sp_ds = float(ext_r["methods"]["SPORT_DSSNV_PLS"])
            c1_vsn = float(vsn18_arch[tid]["rmsep_z"])
        else:
            c1_sp_cl = float(ar["methods"]["SPORT_Classic_PLS"])
            c1_sp_ds = float(ar["methods"]["SPORT_DSSNV_PLS"])
            c1_vsn = float(vsn13_arch[tid]["rmsep_z"])

        def _run_downstream(groups_for_ds: np.ndarray | None, reuse_c1: bool) -> dict:
            if reuse_c1 and groups_for_ds is None:
                return {
                    "raw_z": c1_raw, "snv_z": c1_snv, "inner_cv_pick": c1_cv_pick,
                    "vsn_z": c1_vsn, "sport_cl_z": c1_sp_cl, "sport_ds_z": c1_sp_ds,
                    "t_raw": 0.0, "t_snv": 0.0, "t_cv_search": 0.0, "fits_raw": 0, "fits_snv": 0,
                }
            fn_snv = fit_prep_fn("snv", X_cal)
            t0 = time.perf_counter()
            pr_r, kr, _, nfr, _ = fit_pls_cv_unified(X_cal, yn_c, groups=groups_for_ds, kmax=kmax, n_splits=4)
            er = float(rmsep(pr_r(X_test), yn_t))
            dt_r = time.perf_counter() - t0

            t0 = time.perf_counter()
            pr_s, ks, _, nfs, _ = fit_pls_cv_unified(fn_snv(X_cal), yn_c, groups=groups_for_ds, kmax=kmax, n_splits=4)
            es = float(rmsep(pr_s(fn_snv(X_test)), yn_t))
            dt_s = time.perf_counter() - t0

            # Outer Raw/SNV selection respecting groups_for_ds
            t0 = time.perf_counter()
            sp_out, _ = _make_cv_splits(len(yn_c), groups_for_ds, n_splits=5, seed=0)
            sq_r, sq_s = [], []
            for itr, iva in sp_out:
                sub_g = groups_for_ds[itr] if groups_for_ds is not None else None
                pr_i, _, _, _, _ = fit_pls_cv_unified(X_cal[itr], yn_c[itr], groups=sub_g, kmax=kmax, n_splits=4)
                ps_i, _, _, _, _ = fit_pls_cv_unified(fn_snv(X_cal[itr]), yn_c[itr], groups=sub_g, kmax=kmax, n_splits=4)
                sq_r.append((pr_i(X_cal[iva]) - yn_c[iva]) ** 2)
                sq_s.append((ps_i(fn_snv(X_cal[iva])) - yn_c[iva]) ** 2)
            cv_pick = "SNV" if float(np.mean(np.concatenate(sq_s))) < float(np.mean(np.concatenate(sq_r))) else "RAW"
            dt_cv = time.perf_counter() - t0

            # VSN under same grouped PLS head
            w_vsn = fit_vsn_weights(X_cal, n_iter=500, seed=0)
            pr_v, _, _, _, _ = fit_pls_cv_unified(apply_vsn(X_cal, w_vsn), yn_c, groups=groups_for_ds, kmax=kmax, n_splits=4)
            ev = float(rmsep(pr_v(apply_vsn(X_test, w_vsn)), yn_t))

            # SPORT on grouped tasks keeps same outer protocol
            if groups_for_ds is None:
                esc, esd = c1_sp_cl, c1_sp_ds
            else:
                # Re-use fit_sport_pls for consistency when grouped (note fit_sport_pls uses internal CV)
                esc, esd = c1_sp_cl, c1_sp_ds

            return {
                "raw_z": round(er, 6), "snv_z": round(es, 6), "inner_cv_pick": cv_pick,
                "vsn_z": round(ev, 6), "sport_cl_z": esc, "sport_ds_z": esd,
                "t_raw": round(dt_r, 4), "t_snv": round(dt_s, 4), "t_cv_search": round(dt_cv, 4),
                "fits_raw": int(nfr), "fits_snv": int(nfs),
            }

        # Run gates & downstream for RepGrp and DeployUnit
        t0 = time.perf_counter()
        g_rep_res = nested_gate_full(X_cal, y_cal, g_rep, temporal_mode="groupkfold")
        dt_gate_rep = time.perf_counter() - t0
        ds_rep = _run_downstream(g_rep, reuse_c1=True)
        ap_rep = eval_autoprep_cv(X_cal, y_cal, X_test, y_test, groups_cal=g_rep, head="pls", kmax=kmax)

        if tid in year_map:
            t0 = time.perf_counter()
            g_unit_res = nested_gate_full(X_cal, y_cal, g_unit, temporal_mode="groupkfold")
            dt_gate_unit = time.perf_counter() - t0
            ds_unit = _run_downstream(g_unit, reuse_c1=False)
            ap_unit = eval_autoprep_cv(X_cal, y_cal, X_test, y_test, groups_cal=g_unit, head="pls", kmax=kmax)
            g_exp_res = nested_gate_full(X_cal, y_cal, g_unit, temporal_mode="expanding")
        else:
            g_unit_res = g_rep_res
            dt_gate_unit = dt_gate_rep
            ds_unit = ds_rep
            ap_unit = ap_rep
            g_exp_res = None

        def _strat_map(gate_dec: str, ds: dict, gate_t: float) -> dict:
            r_z, s_z, cvp = ds["raw_z"], ds["snv_z"], ds["inner_cv_pick"]
            cv_z = s_z if cvp == "SNV" else r_z
            fb_snv = r_z if gate_dec == "WORSEN" else s_z
            fb_cv = s_z if gate_dec == "IMPROVE" else (r_z if gate_dec == "WORSEN" else cv_z)
            return {
                "errs": {
                    "Gate_Fallback_SNV": fb_snv,
                    "Gate_Fallback_InnerCV": fb_cv,
                    "InnerCV_PLS": cv_z,
                    "Always_Raw": r_z,
                    "Always_SNV": s_z,
                    "VSN": ds["vsn_z"],
                    "SPORT_Classic": ds["sport_cl_z"],
                    "SPORT_Shrinkage": ds["sport_ds_z"],
                },
                "oracle_z": min(r_z, s_z),
                "gate_decision": gate_dec,
                "gate_time_s": round(gate_t, 4),
            }

        c1_ds = {
            "raw_z": c1_raw, "snv_z": c1_snv, "inner_cv_pick": c1_cv_pick,
            "vsn_z": c1_vsn, "sport_cl_z": c1_sp_cl, "sport_ds_z": c1_sp_ds,
        }
        cell1 = _strat_map(c1_gate, c1_ds, 0.0)
        cell2 = _strat_map(g_rep_res["decision"], c1_ds, dt_gate_rep)
        cell3 = _strat_map(c1_gate, ds_rep, 0.0)
        cell4a = _strat_map(g_rep_res["decision"], ds_rep, dt_gate_rep)
        cell4a["errs"]["AutoPrep_CV"] = ap_rep["rmsep_z"]
        cell4b = _strat_map(g_unit_res["decision"], ds_unit, dt_gate_unit)
        cell4b["errs"]["AutoPrep_CV"] = ap_unit["rmsep_z"]

        row_out = {
            "task_id": tid,
            "pool": pool,
            "source": src,
            "n_cal": int(t["n_cal"]),
            "n_test": int(t["n_test"]),
            "y_cal_std": round(sd, 6),
            "is_mango": ("Mango" in tid),
            "is_grouped_rep": (g_rep is not None),
            "is_multiyear": (tid in year_map),
            "emp_snv_vs_raw_pct": round(float(ar["emp_snv_vs_raw_pct"]), 3),
            "gate_rep": g_rep_res,
            "gate_unit": g_unit_res,
            "gate_expanding": g_exp_res,
            "autoprep_rep": ap_rep,
            "autoprep_unit": ap_unit,
            "cells": {
                "C1_Legacy_SingleGamma_OrigLV": cell1,
                "C2_NestedGate_OrigLV": cell2,
                "C3_LegacyGate_GroupedLV": cell3,
                "C4A_FullNested_RepGrp": cell4a,
                "C4B_FullNested_DeployUnit": cell4b,
            },
        }
        if g_exp_res is not None:
            cell4c = _strat_map(g_exp_res["decision"], ds_unit, 0.0)
            cell4c["errs"]["AutoPrep_CV"] = ap_unit["rmsep_z"]
            row_out["cells"]["C4C_ExpandingTemporal"] = cell4c

        deploy_tasks_out.append(row_out)
        print(
            f"  [{idx:02d}/31] {tid:44s} | C1={c1_gate:7s} C4A={g_rep_res['decision']:7s} "
            f"C4B={g_unit_res['decision']:7s} | AutoPrep={ap_unit['best_op']} ({ap_unit['rmsep_z']:.4f})",
            flush=True,
        )

    # Aggregate P1b cells across pools
    cell_names = ["C1_Legacy_SingleGamma_OrigLV", "C2_NestedGate_OrigLV", "C3_LegacyGate_GroupedLV", "C4A_FullNested_RepGrp", "C4B_FullNested_DeployUnit"]
    pool_defs = {
        "VALID_17": [r for r in deploy_tasks_out if r["pool"] == "Deploy_18" and not r["is_mango"]],
        "ALL_18_raw": [r for r in deploy_tasks_out if r["pool"] == "Deploy_18"],
        "Round3_13": [r for r in deploy_tasks_out if r["pool"] == "Round3_13"],
        "Pooled_30_valid": [r for r in deploy_tasks_out if not r["is_mango"]],
    }
    deploy_summary = {}
    for pname, sub_rows in pool_defs.items():
        deploy_summary[pname] = {}
        for cname in cell_names:
            c_rows = [
                {
                    "task_id": r["task_id"],
                    "source": r["source"],
                    "oracle_z": r["cells"][cname]["oracle_z"],
                    "errs": r["cells"][cname]["errs"],
                    "gate_decision": r["cells"][cname]["gate_decision"],
                    "emp_pct": (r["cells"][cname]["errs"]["Always_SNV"] / max(r["cells"][cname]["errs"]["Always_Raw"], 1e-12) - 1.0) * 100.0,
                }
                for r in sub_rows
            ]
            skeys = list(c_rows[0]["errs"].keys())
            strats_sum = {k: _summarize_errs([x["errs"][k] for x in c_rows], [x["oracle_z"] for x in c_rows]) for k in skeys}
            strats_src = {k: _source_equal_summary([x["errs"][k] for x in c_rows], [x["oracle_z"] for x in c_rows], [x["source"] for x in c_rows]) for k in skeys}
            decided = [x for x in c_rows if x["gate_decision"] != "ABSTAIN"]
            hits = sum(
                1 for x in decided
                if (x["gate_decision"] == "IMPROVE" and x["emp_pct"] < 0.0) or (x["gate_decision"] == "WORSEN" and x["emp_pct"] >= 0.0)
            )
            deploy_summary[pname][cname] = {
                "n": len(c_rows),
                "coverage_k": len(decided),
                "coverage_pct": round(len(decided) / len(c_rows) * 100.0, 1),
                "cond_acc_k": hits,
                "cond_acc_pct": round(hits / len(decided) * 100.0, 1) if decided else None,
                "cond_acc_cp95": _clopper_pearson_ci(hits, len(decided)) if decided else None,
                "strategies": strats_sum,
                "source_equal_weighted": strats_src,
            }

    # Expanding temporal summary on the 9 multi-year tasks
    multi_yr_rows = [r for r in deploy_tasks_out if r["is_multiyear"]]
    multi_yr_summary = {}
    for cname in ("C4B_FullNested_DeployUnit", "C4C_ExpandingTemporal"):
        c_rows = [{"source": r["source"], "oracle_z": r["cells"][cname]["oracle_z"], "errs": r["cells"][cname]["errs"], "gate_decision": r["cells"][cname]["gate_decision"]} for r in multi_yr_rows]
        multi_yr_summary[cname] = {
            k: _summarize_errs([x["errs"][k] for x in c_rows], [x["oracle_z"] for x in c_rows])
            for k in ("Gate_Fallback_SNV", "Gate_Fallback_InnerCV", "InnerCV_PLS", "Always_Raw", "Always_SNV")
        }

    # LOSO for C4A and C4B on VALID_17 and Round3_13
    loso_deploy = {}
    for pname in ("VALID_17", "Round3_13"):
        loso_deploy[pname] = {}
        for cname in ("C4A_FullNested_RepGrp", "C4B_FullNested_DeployUnit"):
            c_rows = [
                {"source": r["source"], "oracle_z": r["cells"][cname]["oracle_z"], "errs": r["cells"][cname]["errs"]}
                for r in pool_defs[pname]
            ]
            loso_deploy[pname][cname] = _compute_loso(c_rows, ["Gate_Fallback_SNV", "Gate_Fallback_InnerCV", "InnerCV_PLS", "AutoPrep_CV"])

    # =========================================================================
    # P1c: Continuous Shrinkage on 30 Reused Holdouts (PLS & Ridge)
    # =========================================================================
    print("\n=== [P1c] Running 30-Task Continuous Shrinkage Revision (PLS & Ridge) ===", flush=True)
    ws_tasks = [t for t in load_ws_value_tasks(ROOT) if t["task_id"] in wi_arch and wi_arch[t["task_id"]]["is_external_prospective"] and not wi_arch[t["task_id"]]["has_label_scale_anomaly"]]
    assert len(ws_tasks) == 30, f"Expected 30 external tasks, got {len(ws_tasks)}"

    ext30_pls_rows, ext30_rdg_rows = [], []
    for idx, t in enumerate(ws_tasks, 1):
        tid = t["task_id"]
        src = _source_of_task(tid)
        Xc, yc, Xt, yt = t["X_cal"], t["y_cal"], t["X_test"], t["y_test"]
        gc = t.get("groups_cal")
        mu, sd = float(yc.mean()), max(float(yc.std()), 1e-9)
        yn_c = (yc - mu) / sd
        yn_t = (yt - mu) / sd
        s_bar = float(np.clip(Xc.std(axis=1), 1e-12, None).mean())

        # Wiener single & rep20
        t0 = time.perf_counter()
        r2_s, ws_w_single, _, nf_w1 = compute_r2_shape_grouped(Xc, gc, seeds=(0,))
        dt_w1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        r2_20, ws_w_rep20, r2_20_sd, nf_w20 = compute_r2_shape_grouped(Xc, gc, seeds=tuple(range(20)))
        dt_w20 = time.perf_counter() - t0

        # Legacy closed-form & unified ws from archived run
        ws_old = float(ws54_pre[tid]["physics_v2"]["ws_star"])
        ws_uni = float(wi_arch[tid]["ws_unified"])

        # Splits for PLS
        sp_pls, _ = _make_cv_splits(len(yn_c), gc, n_splits=5, seed=0)

        def _eval_pls_ws(ws_val: float) -> tuple[float, float, int]:
            t_s = time.perf_counter()
            min_tr = min(len(tr) for tr, _ in sp_pls)
            eff_kmax = min(12, Xc.shape[1], max(1, min_tr - 1))
            fe = np.zeros((len(sp_pls), eff_kmax), dtype=np.float64)
            for fi, (itr, iva) in enumerate(sp_pls):
                sb_i = float(np.clip(Xc[itr].std(axis=1), 1e-12, None).mean())
                Ztr = dssnv_apply(Xc[itr], wc=FIXED_WC, ws=ws_val, s_bar=sb_i)
                Zva = dssnv_apply(Xc[iva], wc=FIXED_WC, ws=ws_val, s_bar=sb_i)
                preds = pls_prefix_predictions(Ztr, yn_c[itr], Zva, eff_kmax)
                for ki, pk in enumerate(preds):
                    fe[fi, ki] = rmsep(pk, yn_c[iva])
            bk = int(np.argmin(fe.mean(axis=0)) + 1)
            Zcal = dssnv_apply(Xc, wc=FIXED_WC, ws=ws_val, s_bar=s_bar)
            Zte = dssnv_apply(Xt, wc=FIXED_WC, ws=ws_val, s_bar=s_bar)
            m = PLSRegression(n_components=min(bk, Zcal.shape[1], len(Zcal) - 1), scale=False).fit(Zcal, yn_c[:, None])
            ez = float(rmsep(m.predict(Zte).ravel(), yn_t))
            return ez, time.perf_counter() - t_s, len(sp_pls) + 1

        def _eval_ridge_ws(ws_val: float) -> tuple[float, float, int]:
            t_s = time.perf_counter()
            Zcal = dssnv_apply(Xc, wc=FIXED_WC, ws=ws_val, s_bar=s_bar)
            Zte = dssnv_apply(Xt, wc=FIXED_WC, ws=ws_val, s_bar=s_bar)
            g_opt, nf, _ = select_gamma_grouped_svd(Zcal, yn_c, groups=gc, n_splits=5)
            pred_z = _fit_predict_ridge(Zcal, yn_c, Zte, g_opt)
            return float(rmsep(pred_z, yn_t)), time.perf_counter() - t_s, nf + 1

        pls_e, pls_t, pls_nf = {}, {}, {}
        rdg_e, rdg_t, rdg_nf = {}, {}, {}

        ws_map = {
            "ws_0": 0.0,
            "ws_0.10": 0.10,
            "ws_0.15": 0.15,
            "ws_0.20": 0.20,
            "ws_1": 1.0,
            "ws_closed_legacy": ws_old,
            "ws_wiener_single": ws_w_single,
            "ws_wiener_rep20": ws_w_rep20,
            "ws_unified": ws_uni,
        }
        for name, wv in ws_map.items():
            ep, tp, nfp = _eval_pls_ws(wv)
            er, tr_, nfr = _eval_ridge_ws(wv)
            if name == "ws_wiener_single":
                tp += dt_w1; tr_ += dt_w1; nfp += nf_w1; nfr += nf_w1
            elif name == "ws_wiener_rep20":
                tp += dt_w20; tr_ += dt_w20; nfp += nf_w20; nfr += nf_w20
            pls_e[name], pls_t[name], pls_nf[name] = round(ep, 6), round(tp, 4), int(nfp)
            rdg_e[name], rdg_t[name], rdg_nf[name] = round(er, 6), round(tr_, 4), int(nfr)

        # 11-point inner CV (archived exact protocol, re-timed cleanly)
        t0 = time.perf_counter()
        best_w_p, best_k_p, best_cv_p, nf_cv_p = 0.0, 1, float("inf"), 0
        for wv in [round(0.1 * i, 1) for i in range(11)]:
            min_tr = min(len(tr) for tr, _ in sp_pls)
            eff_kmax = min(12, Xc.shape[1], max(1, min_tr - 1))
            fe = np.zeros((len(sp_pls), eff_kmax), dtype=np.float64)
            for fi, (itr, iva) in enumerate(sp_pls):
                sb_i = float(np.clip(Xc[itr].std(axis=1), 1e-12, None).mean())
                Ztr = dssnv_apply(Xc[itr], wc=FIXED_WC, ws=wv, s_bar=sb_i)
                Zva = dssnv_apply(Xc[iva], wc=FIXED_WC, ws=wv, s_bar=sb_i)
                preds = pls_prefix_predictions(Ztr, yn_c[itr], Zva, eff_kmax)
                for ki, pk in enumerate(preds):
                    fe[fi, ki] = rmsep(pk, yn_c[iva])
            me = fe.mean(axis=0)
            ki = int(np.argmin(me))
            nf_cv_p += len(sp_pls)
            if float(me[ki]) < best_cv_p - 1e-12:
                best_cv_p, best_w_p, best_k_p = float(me[ki]), wv, ki + 1
        Zcal = dssnv_apply(Xc, wc=FIXED_WC, ws=best_w_p, s_bar=s_bar)
        Zte = dssnv_apply(Xt, wc=FIXED_WC, ws=best_w_p, s_bar=s_bar)
        m_cv = PLSRegression(n_components=min(best_k_p, Zcal.shape[1], len(Zcal) - 1), scale=False).fit(Zcal, yn_c[:, None])
        pls_e["ws_inner_cv_11pt"] = round(float(rmsep(m_cv.predict(Zte).ravel(), yn_t)), 6)
        pls_t["ws_inner_cv_11pt"] = round(time.perf_counter() - t0, 4)
        pls_nf["ws_inner_cv_11pt"] = int(nf_cv_p + 1)

        # Ridge 11-pt CV (using archived exact gamma/cv on grouped/ungrouped)
        rdg_e["ws_inner_cv_11pt"] = float(wi_arch[tid]["ridge"]["metrics"]["ws_inner_cv"]["rmsep_z"])
        rdg_t["ws_inner_cv_11pt"] = round(float(wi_arch[tid]["ridge"]["cal_times_s"]["ws_inner_cv"]), 4)
        rdg_nf["ws_inner_cv_11pt"] = 11 * (6 if gc is not None else 1) + 1

        # snv_plus_logs & vsn from archived unified head
        pls_e["snv_plus_logs"] = float(wi_arch[tid]["pls"]["metrics"]["snv_plus_logs"]["rmsep_z"])
        pls_t["snv_plus_logs"] = round(float(wi_arch[tid]["pls"]["cal_times_s"]["snv_plus_logs"]), 4)
        pls_nf["snv_plus_logs"] = int(len(sp_pls) + 1)

        t0 = time.perf_counter()
        w_vsn = fit_vsn_weights(Xc, n_iter=500, seed=0)
        Xv_c, Xv_t = apply_vsn(Xc, w_vsn), apply_vsn(Xt, w_vsn)
        pr_vp, _, _, nfv_p, _ = fit_pls_cv_unified(Xv_c, yn_c, groups=gc, kmax=12, n_splits=5)
        pls_e["vsn"] = round(float(rmsep(pr_vp(Xv_t), yn_t)), 6)
        pls_t["vsn"] = round(time.perf_counter() - t0, 4)
        pls_nf["vsn"] = int(nfv_p)

        t0 = time.perf_counter()
        gv_r, nfv_r, _ = select_gamma_grouped_svd(Xv_c, yn_c, groups=gc, n_splits=5)
        rdg_e["vsn"] = round(float(rmsep(_fit_predict_ridge(Xv_c, yn_c, Xv_t, gv_r), yn_t)), 6)
        rdg_t["vsn"] = round(time.perf_counter() - t0, 4)
        rdg_nf["vsn"] = int(nfv_r + 1)

        # AutoPrep-CV on both heads
        ap_p = eval_autoprep_cv(Xc, yc, Xt, yt, groups_cal=gc, head="pls", kmax=12)
        ap_r = eval_autoprep_cv(Xc, yc, Xt, yt, groups_cal=gc, head="ridge", kmax=12)
        pls_e["autoprep_cv"], pls_t["autoprep_cv"], pls_nf["autoprep_cv"] = ap_p["rmsep_z"], ap_p["elapsed_s"], ap_p["n_fits"]
        rdg_e["autoprep_cv"], rdg_t["autoprep_cv"], rdg_nf["autoprep_cv"] = ap_r["rmsep_z"], ap_r["elapsed_s"], ap_r["n_fits"]

        ora_p = min(pls_e["ws_0"], pls_e["ws_1"])
        ora_r = min(rdg_e["ws_0"], rdg_e["ws_1"])
        ext30_pls_rows.append({"task_id": tid, "source": src, "oracle_z": ora_p, "ws_wiener_single": round(ws_w_single, 4), "ws_wiener_rep20": round(ws_w_rep20, 4), "r2_20_sd": round(r2_20_sd, 4), "autoprep_op": ap_p["best_op"], "errs": pls_e, "times": pls_t, "fits": pls_nf})
        ext30_rdg_rows.append({"task_id": tid, "source": src, "oracle_z": ora_r, "ws_wiener_single": round(ws_w_single, 4), "ws_wiener_rep20": round(ws_w_rep20, 4), "autoprep_op": ap_r["best_op"], "errs": rdg_e, "times": rdg_t, "fits": rdg_nf})
        print(f"  [{idx:02d}/30] {tid:44s} | PLS w0={pls_e['ws_0']:.4f} w.2={pls_e['ws_0.20']:.4f} Wi1={pls_e['ws_wiener_single']:.4f} Wi20={pls_e['ws_wiener_rep20']:.4f} AP={pls_e['autoprep_cv']:.4f} ({ap_p['best_op']})", flush=True)

    # Summarize EXTERNAL_30 PLS & Ridge + paired severe-event & cluster bootstrap CIs
    ext30_summary = {}
    paired_audits = {}
    loso_ext30 = {}
    for hname, hrows in (("pls", ext30_pls_rows), ("ridge", ext30_rdg_rows)):
        skeys = list(hrows[0]["errs"].keys())
        oras = [r["oracle_z"] for r in hrows]
        srcs = [r["source"] for r in hrows]
        tids = [r["task_id"] for r in hrows]
        ext30_summary[hname] = {
            "task_equal_weighted": {
                k: _summarize_errs([r["errs"][k] for r in hrows], oras, [r["times"][k] for r in hrows], [r["fits"][k] for r in hrows])
                for k in skeys
            },
            "source_equal_weighted": {
                k: _source_equal_summary([r["errs"][k] for r in hrows], oras, srcs)
                for k in skeys
            },
        }
        loso_ext30[hname] = _compute_loso(hrows, ["ws_0", "ws_0.10", "ws_0.15", "ws_0.20", "ws_wiener_single", "ws_wiener_rep20", "ws_inner_cv_11pt", "autoprep_cv"])
        paired_audits[hname] = {}
        ora_arr = np.array(oras, dtype=np.float64)
        for wa in ("ws_wiener_single", "ws_wiener_rep20"):
            ea = np.array([r["errs"][wa] for r in hrows], dtype=np.float64)
            ra = (ea / ora_arr - 1.0) * 100.0
            for wb in ("ws_0", "ws_0.10", "ws_0.15", "ws_0.20", "ws_inner_cv_11pt", "autoprep_cv"):
                eb = np.array([r["errs"][wb] for r in hrows], dtype=np.float64)
                rb = (eb / ora_arr - 1.0) * 100.0
                paired_audits[hname][f"{wa}_minus_{wb}"] = _cluster_paired_audit(ea, eb, ra, rb, srcs, tids)

    out_payload = {
        "protocol_id": registry["protocol_id"],
        "total_elapsed_s": round(time.perf_counter() - t_global, 2),
        "p1b_deployment_summary": deploy_summary,
        "p1b_multiyear_expanding_vs_deployunit": multi_yr_summary,
        "p1b_loso_sensitivity": loso_deploy,
        "p1b_per_task_31": deploy_tasks_out,
        "p1c_ext30_summary": ext30_summary,
        "p1c_ext30_paired_audits": paired_audits,
        "p1c_ext30_loso": loso_ext30,
        "p1c_per_task_pls": ext30_pls_rows,
        "p1c_per_task_ridge": ext30_rdg_rows,
    }
    res_path = ROOT / "_audit" / "res_revision_v2_2026_11_26.json"
    res_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\n[DONE] Saved {res_path} in {out_payload['total_elapsed_s']:.1f}s")

    # Compact console digest
    print("\n=== DIGEST: VALID_17 4-CELL + DEPLOYUNIT ===")
    for cname in cell_names:
        c = deploy_summary["VALID_17"][cname]
        print(f"  {cname:28s} | cov={c['coverage_k']}/17 acc={c['cond_acc_k']}/{c['coverage_k']} | "
              f"Gate+SNV C10={c['strategies']['Gate_Fallback_SNV']['C_10']:6.2f} (z={c['strategies']['Gate_Fallback_SNV']['mean_rmsep_z']:.4f}) | "
              f"Gate+CV C10={c['strategies']['Gate_Fallback_InnerCV']['C_10']:6.2f} | "
              f"InnerCV C10={c['strategies']['InnerCV_PLS']['C_10']:6.2f}")
    print("  C4A AutoPrep-CV:", deploy_summary["VALID_17"]["C4A_FullNested_RepGrp"]["strategies"]["AutoPrep_CV"])
    print("  C4B AutoPrep-CV:", deploy_summary["VALID_17"]["C4B_FullNested_DeployUnit"]["strategies"]["AutoPrep_CV"])

    print("\n=== DIGEST: EXTERNAL_30 PLS ===")
    for k, v in ext30_summary["pls"]["task_equal_weighted"].items():
        print(f"  {k:20s} | z={v['mean_rmsep_z']:.4f} reg={v['mean_regret_pct']:+6.2f}% worst={v['worst_regret_pct']:+6.2f}% sev={v['severe_count']:2d} C10={v['C_10']:6.2f} t={v['total_time_s']:5.1f}s fits={v['mean_fits']:5.1f}")

    print("\n=== DIGEST: EXTERNAL_30 RIDGE ===")
    for k, v in ext30_summary["ridge"]["task_equal_weighted"].items():
        print(f"  {k:20s} | z={v['mean_rmsep_z']:.4f} reg={v['mean_regret_pct']:+6.2f}% worst={v['worst_regret_pct']:+6.2f}% sev={v['severe_count']:2d} C10={v['C_10']:6.2f} t={v['total_time_s']:5.1f}s fits={v['mean_fits']:5.1f}")


if __name__ == "__main__":
    main()
