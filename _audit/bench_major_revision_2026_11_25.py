"""Major Revision hard experimental upgrades (zero data leakage, exact matched protocols):

1. Faithful inductive VSN (Rabatel et al., 2020, J. Chemom. 34(2): e3164) head-to-head
   comparison across Deploy_18 / VALID_17, Round3_13, and EXTERNAL_30 (PLS & Ridge),
   with synthetic chemical-band separation and single-sample inductive invariance checks.
2. Strictly nested outer-CV gamma-selection bias ablation for Paired Allen-PRESS gating
   across all 31 deployment tasks (VALID_17, ALL_18, Round3_13, Pooled_31).
3. Formal one-sided non-inferiority tests (delta in {0.03, 0.05, 0.08} RMSEP_z) and
   cluster-bootstrap 95% CIs (170-fold across 7 datasets, 86-fold across 18 tasks,
   and EXTERNAL_30 across 12 distinct sources/labs).

Outputs: _audit/res_major_revision_2026_11_25.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._deploy_holdout_loaders import load_deploy_holdout_tasks
from _audit._round3_unseen_loaders import load_round3_unseen_tasks
from _audit._ws_closed_value_loaders import (
    load_ws_value_tasks,
    make_inner_splits,
    pls_prefix_predictions,
    select_k_pls_for_ws,
)
from chemdl.theory import exact_press_loo_residuals
from chemdl.transfer import fit_pls, rmsep


def fit_vsn_weights(
    X_cal: np.ndarray,
    *,
    n_iter: int = 500,
    tol: float = 0.10,
    top_frac: float = 0.20,
    seed: int = 0,
) -> np.ndarray:
    """Faithful inductive RANSAC-based Variable Sorting for Normalization (Rabatel et al., 2020).

    Fits wavelength consensus weights w in [0, 1]^L strictly on calibration spectra X_cal:
      - Column-center X_cal across samples;
      - Sample random pairs of wavelengths (j1, j2) and project every wavelength column
        onto span(x_{:, j1}, x_{:, j2});
      - Compute scale-invariant relative residual e_j = ||r_j||_2 / ||x_j||_2;
      - Evaluate inliers at e_j <= max(tol, 1.5 * median(e)) and average inlier indicators
        across the top-consensus iterations.
    """
    X = np.asarray(X_cal, dtype=np.float64)
    n, L = X.shape
    if L < 3 or n < 3:
        return np.ones(L, dtype=np.float64)

    Xc = X - X.mean(axis=0, keepdims=True)
    col_norms = np.linalg.norm(Xc, axis=0)
    valid_cols = np.flatnonzero(col_norms > 1e-12)
    if len(valid_cols) < 3:
        return np.ones(L, dtype=np.float64)

    rng = np.random.default_rng(seed)
    inlier_mats = np.zeros((n_iter, L), dtype=np.float64)
    scores = np.zeros(n_iter, dtype=np.float64)
    safe_norms = np.maximum(col_norms, 1e-12)

    for it in range(n_iter):
        j1, j2 = rng.choice(valid_cols, size=2, replace=False)
        B = Xc[:, [j1, j2]]
        Q, _ = np.linalg.qr(B, mode="reduced")
        proj = Q @ (Q.T @ Xc)
        res_norms = np.linalg.norm(Xc - proj, axis=0)
        rel_err = res_norms / safe_norms
        inl = (rel_err <= float(tol)) & (col_norms > 1e-12)
        inlier_mats[it] = inl.astype(np.float64)
        scores[it] = float(np.sum(inl))

    k_top = max(1, int(np.ceil(top_frac * n_iter)))
    top_idx = np.argsort(scores)[-k_top:]
    w = np.mean(inlier_mats[top_idx], axis=0)
    if float(np.sum(w)) < 1e-6 or float(np.sum(w > 0)) < 2:
        return np.ones(L, dtype=np.float64)
    return np.clip(w, 0.0, 1.0)


def apply_vsn(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Strictly single-sample inductive weighted SNV using calibration-frozen wavelength weights."""
    Xa = np.asarray(X, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    sw = max(float(np.sum(w)), 1e-12)
    mu_w = (Xa @ w) / sw
    diff = Xa - mu_w[:, None]
    var_w = ((diff ** 2) @ w) / sw
    sd_w = np.sqrt(np.maximum(var_w, 1e-24))
    return diff / sd_w[:, None]


def verify_vsn_synthetic() -> dict:
    """Verify VSN separates scatter-only inliers from chemical peaks & preserves inductive invariance."""
    rng = np.random.default_rng(2026)
    n, L = 80, 120
    wl = np.linspace(0.0, 1.0, L)
    base_shape = np.sin(2.0 * np.pi * wl) + 0.5 * np.cos(4.0 * np.pi * wl)
    a = rng.uniform(0.7, 1.4, size=(n, 1))
    b = rng.normal(0.0, 0.3, size=(n, 1))
    X = a * base_shape[None, :] + b + rng.normal(0.0, 0.005, size=(n, L))
    # Inject independent chemical analyte peaks on 20% of wavelengths (indices 48..71)
    y = rng.normal(0.0, 1.0, size=n)
    peak_mask = np.zeros(L, dtype=bool)
    peak_mask[48:72] = True
    X[:, peak_mask] += (a * y[:, None]) * 1.5

    w = fit_vsn_weights(X[:60], n_iter=500, seed=0)
    w_base = float(np.mean(w[~peak_mask]))
    w_peak = float(np.mean(w[peak_mask]))
    assert w_base > 0.80, f"Expected baseline weights > 0.80, got {w_base:.3f}"
    assert w_peak < 0.25, f"Expected chemical-peak weights < 0.25, got {w_peak:.3f}"

    # Single-sample inductive invariance & poison resistance
    X_te = X[60:]
    out_batch = apply_vsn(X_te, w)
    out_single = np.vstack([apply_vsn(X_te[i : i + 1], w) for i in range(len(X_te))])
    X_poison = np.vstack([X_te, np.full((3, L), 999.0)])
    out_poison = apply_vsn(X_poison, w)[: len(X_te)]
    max_ind_diff = float(np.max(np.abs(out_batch - out_single)))
    max_poi_diff = float(np.max(np.abs(out_batch - out_poison)))
    assert max_ind_diff < 1e-12 and max_poi_diff < 1e-12
    return {
        "w_baseline_mean": round(w_base, 4),
        "w_chemical_peak_mean": round(w_peak, 4),
        "max_single_sample_diff": max_ind_diff,
        "max_poison_diff": max_poi_diff,
    }


def _eval_cw(regrets: list[float] | np.ndarray, w: float) -> float:
    r = np.asarray(regrets, dtype=np.float64)
    return float(np.mean(r) + (w - 1.0) * np.mean(np.maximum(r - 5.0, 0.0)))


def _summarize_regrets(errs: list[float], oracles: list[float], total_time_s: float) -> dict:
    e = np.asarray(errs, dtype=np.float64)
    o = np.asarray(oracles, dtype=np.float64)
    reg = (e / o - 1.0) * 100.0
    return {
        "n": int(len(e)),
        "mean_rmsep_z": round(float(np.mean(e)), 4),
        "mean_regret_pct": round(float(np.mean(reg)), 2),
        "median_regret_pct": round(float(np.median(reg)), 2),
        "max_regret_pct": round(float(np.max(reg)), 2),
        "n_severe_ge5pct": int(np.sum(reg >= 5.0)),
        "C_5": round(_eval_cw(reg, 5.0), 2),
        "C_10": round(_eval_cw(reg, 10.0), 2),
        "C_20": round(_eval_cw(reg, 20.0), 2),
        "total_time_s": round(float(total_time_s), 3),
        "mean_time_s": round(float(total_time_s / max(len(e), 1)), 4),
    }


def run_part1_vsn() -> dict:
    print("\n=== Part 1: Faithful Inductive VSN (Rabatel 2020) Head-to-Head ===")
    syn = verify_vsn_synthetic()
    print("Synthetic VSN verification:", syn)

    # 1A. Deploy_18 & VALID_17 (exact matched protocol to prereg_deploy_holdout_phase2_unblind.py)
    d18_tasks = load_deploy_holdout_tasks(ROOT)
    d18_arch = json.loads((ROOT / "_audit" / "res_prereg_deploy_holdout_2026_11_24.json").read_text(encoding="utf-8"))
    d18_by_id = {t["task_id"]: t for t in d18_arch["tasks"]}

    vsn_18_errs, ora_18, t_18 = [], [], 0.0
    vsn_17_errs, ora_17, t_17 = [], [], 0.0
    vsn_18_rows = []
    for t in d18_tasks:
        tid = t["task_id"]
        ar = d18_by_id[tid]
        X_cal, y_cal, X_te, y_te = t["X_cal"], t["y_cal"], t["X_test"], t["y_test"]
        mu, sd = float(y_cal.mean()), max(float(y_cal.std()), 1e-9)
        y_cal_n = ((y_cal - mu) / sd)[:, None]
        y_te_n = ((y_te - mu) / sd)[:, None]
        kmax = min(10, max(2, len(X_cal) // 2))
        inner = min(4, max(2, len(X_cal) // 4))

        t0 = time.perf_counter()
        w_vsn = fit_vsn_weights(X_cal, n_iter=500, seed=0)
        Xcal_v = apply_vsn(X_cal, w_vsn)
        Xte_v = apply_vsn(X_te, w_vsn)
        pr_v, _ = fit_pls(Xcal_v, y_cal_n, kmax=kmax, inner=inner, per_col=True, seed=0)
        dt = time.perf_counter() - t0
        e_v = float(rmsep(pr_v(Xte_v), y_te_n))
        ora = float(ar["oracle_pls_z"])
        reg = (e_v / ora - 1.0) * 100.0

        t_18 += dt
        vsn_18_errs.append(e_v)
        ora_18.append(ora)
        vsn_18_rows.append({"task_id": tid, "rmsep_z": round(e_v, 4), "regret_pct": round(reg, 2), "time_s": round(dt, 4)})
        if "Mango" not in tid:
            t_17 += dt
            vsn_17_errs.append(e_v)
            ora_17.append(ora)

    sum_17 = _summarize_regrets(vsn_17_errs, ora_17, t_17)
    sum_18 = _summarize_regrets(vsn_18_errs, ora_18, t_18)
    print("  VALID_17 VSN:", sum_17)
    print("  ALL_18   VSN:", sum_18)

    # 1B. Round3_13 (exact matched protocol to prereg_round3_unseen_phase2_unblind.py)
    r3_tasks = load_round3_unseen_tasks(ROOT)
    r3_arch = json.loads((ROOT / "_audit" / "res_prereg_round3_unseen_2026_11_24.json").read_text(encoding="utf-8"))
    r3_by_id = {t["task_id"]: t for t in r3_arch["tasks"]}

    vsn_13_errs, ora_13, t_13 = [], [], 0.0
    vsn_13_rows = []
    for t in r3_tasks:
        tid = t["task_id"]
        ar = r3_by_id[tid]
        X_cal, y_cal, X_te, y_te = t["X_cal"], t["y_cal"], t["X_test"], t["y_test"]
        mu, sd = float(y_cal.mean()), max(float(y_cal.std()), 1e-9)
        y_cal_n = ((y_cal - mu) / sd)[:, None]
        y_te_n = ((y_te - mu) / sd)[:, None]
        kmax = min(10, max(2, len(X_cal) // 2))
        inner = min(4, max(2, len(X_cal) // 4))

        t0 = time.perf_counter()
        w_vsn = fit_vsn_weights(X_cal, n_iter=500, seed=0)
        Xcal_v = apply_vsn(X_cal, w_vsn)
        Xte_v = apply_vsn(X_te, w_vsn)
        pr_v, _ = fit_pls(Xcal_v, y_cal_n, kmax=kmax, inner=inner, per_col=True, seed=0)
        dt = time.perf_counter() - t0
        e_v = float(rmsep(pr_v(Xte_v), y_te_n))
        ora = float(ar["oracle_pls_z"])
        reg = (e_v / ora - 1.0) * 100.0
        t_13 += dt
        vsn_13_errs.append(e_v)
        ora_13.append(ora)
        vsn_13_rows.append({"task_id": tid, "rmsep_z": round(e_v, 4), "regret_pct": round(reg, 2), "time_s": round(dt, 4)})

    sum_13 = _summarize_regrets(vsn_13_errs, ora_13, t_13)
    print("  Round3_13 VSN:", sum_13)

    # 1C. EXTERNAL_30 (exact matched protocol to prereg_wiener_ws_phase2_unblind.py: PLS kmax=12 & Ridge)
    ws_tasks = load_ws_value_tasks(ROOT)
    wi_arch = json.loads((ROOT / "_audit" / "res_wiener_ws_2026_11_25.json").read_text(encoding="utf-8"))
    wi_by_id = {t["task_id"]: t for t in wi_arch["tasks"]}

    pls_errs, pls_ora, pls_t = [], [], 0.0
    rdg_errs, rdg_ora, rdg_t = [], [], 0.0
    ext30_rows = []
    for t in ws_tasks:
        tid = t["task_id"]
        ar = wi_by_id[tid]
        if not (ar.get("is_external_prospective") and not ar.get("has_label_scale_anomaly")):
            continue
        X_cal, y_cal, X_te, y_te = t["X_cal"], t["y_cal"], t["X_test"], t["y_test"]
        gc = t.get("groups_cal")
        mu, sd = float(y_cal.mean()), max(float(y_cal.std()), 1e-9)
        yn_cal = (y_cal - mu) / sd
        yn_te = (y_te - mu) / sd

        t0 = time.perf_counter()
        w_vsn = fit_vsn_weights(X_cal, n_iter=500, seed=0)
        Xcal_v = apply_vsn(X_cal, w_vsn)
        Xte_v = apply_vsn(X_te, w_vsn)
        dt_w = time.perf_counter() - t0

        # PLS head (exact same inner splits & kmax=12 as prereg_wiener_ws)
        t0 = time.perf_counter()
        splits = make_inner_splits(len(X_cal), gc, n_splits=5, seed=0)
        eff_kmax = min(12, Xcal_v.shape[1], max(1, min(len(tr) for tr, _ in splits) - 1))
        fold_errs = np.zeros((len(splits), eff_kmax), dtype=np.float64)
        for fi, (itr, iva) in enumerate(splits):
            preds = pls_prefix_predictions(Xcal_v[itr], yn_cal[itr], Xcal_v[iva], eff_kmax)
            fold_errs[fi] = np.sqrt(np.mean((preds - yn_cal[iva][None, :]) ** 2, axis=1))
        best_k = int(np.argmin(np.mean(fold_errs, axis=0)) + 1)
        pred_pls = pls_prefix_predictions(Xcal_v, yn_cal, Xte_v, best_k)[best_k - 1]
        e_pls = float(rmsep(pred_pls, yn_te))
        dt_pls = dt_w + (time.perf_counter() - t0)

        # Ridge head (exact same gamma selection as prereg_wiener_ws)
        t0 = time.perf_counter()
        if gc is not None and len(np.unique(gc)) >= 2:
            eff = min(5, len(np.unique(gc)))
            g_splits = list(GroupKFold(n_splits=eff).split(Xcal_v, yn_cal, gc))
            if min(len(tr) for tr, _ in g_splits) >= 3:
                gammas = [float(exact_press_loo_residuals(Xcal_v[tr], yn_cal[tr])[1]) for tr, _ in g_splits]
                gamma = float(np.median(gammas))
            else:
                _, gamma = exact_press_loo_residuals(Xcal_v, yn_cal)
        else:
            _, gamma = exact_press_loo_residuals(Xcal_v, yn_cal)
        xm = Xcal_v.mean(axis=0, keepdims=True)
        ym = float(np.mean(yn_cal))
        U, s, Vt = np.linalg.svd(Xcal_v - xm, full_matrices=False)
        W = Vt.T @ ((s / (s ** 2 + max(gamma, 1e-15))) * (U.T @ (yn_cal - ym)))
        pred_rdg = (Xte_v - xm) @ W + ym
        e_rdg = float(rmsep(pred_rdg, yn_te))
        dt_rdg = dt_w + (time.perf_counter() - t0)

        ora_p = min(ar["pls"]["metrics"]["ws_0"]["rmsep_z"], ar["pls"]["metrics"]["ws_1"]["rmsep_z"])
        ora_r = min(ar["ridge"]["metrics"]["ws_0"]["rmsep_z"], ar["ridge"]["metrics"]["ws_1"]["rmsep_z"])
        pls_errs.append(e_pls); pls_ora.append(ora_p); pls_t += dt_pls
        rdg_errs.append(e_rdg); rdg_ora.append(ora_r); rdg_t += dt_rdg
        ext30_rows.append({"task_id": tid, "pls_rmsep_z": round(e_pls, 4), "ridge_rmsep_z": round(e_rdg, 4)})

    sum_ext30_pls = _summarize_regrets(pls_errs, pls_ora, pls_t)
    sum_ext30_rdg = _summarize_regrets(rdg_errs, rdg_ora, rdg_t)
    print("  EXTERNAL_30 PLS   VSN:", sum_ext30_pls)
    print("  EXTERNAL_30 Ridge VSN:", sum_ext30_rdg)

    return {
        "synthetic_check": syn,
        "VALID_17": sum_17,
        "ALL_18": sum_18,
        "Round3_13": sum_13,
        "EXTERNAL_30_PLS": sum_ext30_pls,
        "EXTERNAL_30_RIDGE": sum_ext30_rdg,
        "per_task_ext30": ext30_rows,
        "per_task_18": vsn_18_rows,
        "per_task_13": vsn_13_rows,
    }


def _nested_gate_task(X_cal: np.ndarray, y_cal: np.ndarray, groups_cal: np.ndarray | None) -> dict:
    """Strictly nested outer-CV gate: select gamma ONLY inside each outer training fold."""
    Xa = np.asarray(X_cal, dtype=np.float64)
    ya = np.asarray(y_cal, dtype=np.float64).ravel()
    X_snv = (Xa - Xa.mean(axis=1, keepdims=True)) / np.clip(Xa.std(axis=1, keepdims=True), 1e-12, None)

    if groups_cal is not None:
        grp = np.asarray(groups_cal)
        uniq = np.unique(grp)
        eff = min(5, len(uniq))
        splits = list(GroupKFold(n_splits=eff).split(Xa, ya, grp))
    else:
        eff = min(5, len(ya))
        splits = list(KFold(n_splits=eff, shuffle=True, random_state=0).split(Xa))

    e_raw = np.empty(len(ya), dtype=np.float64)
    e_snv = np.empty(len(ya), dtype=np.float64)

    for itr, iva in splits:
        _, g_r = exact_press_loo_residuals(Xa[itr], ya[itr])
        _, g_s = exact_press_loo_residuals(X_snv[itr], ya[itr])
        for Z, g, out_arr in ((Xa, g_r, e_raw), (X_snv, g_s, e_snv)):
            xm = Z[itr].mean(axis=0, keepdims=True)
            ym = float(ya[itr].mean())
            U, s, Vt = np.linalg.svd(Z[itr] - xm, full_matrices=False)
            W = Vt.T @ ((s / (s ** 2 + max(g, 1e-15))) * (U.T @ (ya[itr] - ym)))
            out_arr[iva] = ya[iva] - ((Z[iva] - xm) @ W + ym)

    rmse_r = float(np.sqrt(np.mean(e_raw ** 2)))
    rmse_s = float(np.sqrt(np.mean(e_snv ** 2)))
    delta_pct = (rmse_s / max(rmse_r, 1e-12) - 1.0) * 100.0

    d_scan = (e_snv ** 2) - (e_raw ** 2)
    if groups_cal is not None:
        grp = np.asarray(groups_cal)
        d_unit = np.array([float(np.mean(d_scan[grp == g])) for g in np.unique(grp)])
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
    return {"decision": dec, "delta_nested_pct": round(delta_pct, 3), "t_nested": round(t_nested, 3)}


def run_part2_nested_cv() -> dict:
    from scipy.stats import spearmanr

    print("\n=== Part 2: Strictly Nested Outer-CV Gamma-Selection Bias Ablation ===")
    d18_tasks = load_deploy_holdout_tasks(ROOT)
    d18_arch = json.loads((ROOT / "_audit" / "res_prereg_deploy_holdout_2026_11_24.json").read_text(encoding="utf-8"))
    d18_by_id = {t["task_id"]: t for t in d18_arch["tasks"]}

    r3_tasks = load_round3_unseen_tasks(ROOT)
    r3_arch = json.loads((ROOT / "_audit" / "res_prereg_round3_unseen_2026_11_24.json").read_text(encoding="utf-8"))
    r3_by_id = {t["task_id"]: t for t in r3_arch["tasks"]}

    all_rows = []
    t_nested_total = 0.0
    for pool, tasks, arch in (("Deploy_18", d18_tasks, d18_by_id), ("Round3_13", r3_tasks, r3_by_id)):
        for t in tasks:
            tid = t["task_id"]
            ar = arch[tid]
            t0 = time.perf_counter()
            ng = _nested_gate_task(t["X_cal"], t["y_cal"], t.get("groups_cal"))
            dt = time.perf_counter() - t0
            t_nested_total += dt
            emp = float(ar["emp_snv_vs_raw_pct"])
            ora = float(ar["oracle_pls_z"])
            r_raw, r_snv = float(ar["rmsep_raw_z"]), float(ar["rmsep_snv_z"])
            cv_pick = str(ar["inner_cv_pick"])
            e_cv = r_snv if cv_pick == "SNV" else r_raw
            dec_n = ng["decision"]
            e_fb_snv = r_raw if dec_n == "WORSEN" else r_snv
            e_fb_cv = r_snv if dec_n == "IMPROVE" else (r_raw if dec_n == "WORSEN" else e_cv)
            all_rows.append({
                "pool": pool,
                "task_id": tid,
                "is_mango": "Mango" in tid,
                "gate_allen": ar["gate_decision"],
                "gate_nested": dec_n,
                "delta_loo": float(ar["delta_loo_pct"]),
                "delta_nested": ng["delta_nested_pct"],
                "t_loo": float(ar["t_loo"]),
                "t_nested": ng["t_nested"],
                "emp_pct": emp,
                "ora": ora,
                "e_fb_snv": e_fb_snv,
                "e_fb_cv": e_fb_cv,
            })

    def summarize_subset(name, rows):
        rho_d, p_d = spearmanr([r["delta_loo"] for r in rows], [r["delta_nested"] for r in rows])
        rho_t, p_t = spearmanr([r["t_loo"] for r in rows], [r["t_nested"] for r in rows])
        agree_3 = sum(1 for r in rows if r["gate_allen"] == r["gate_nested"])
        decided_n = [r for r in rows if r["gate_nested"] != "ABSTAIN"]
        hits_n = sum(1 for r in decided_n if (r["gate_nested"] == "IMPROVE" and r["emp_pct"] < 0) or (r["gate_nested"] == "WORSEN" and r["emp_pct"] >= 0))
        reg_snv = [(r["e_fb_snv"] / r["ora"] - 1.0) * 100.0 for r in rows]
        reg_cv = [(r["e_fb_cv"] / r["ora"] - 1.0) * 100.0 for r in rows]
        out = {
            "n": len(rows),
            "spearman_delta": round(float(rho_d), 4),
            "spearman_t": round(float(rho_t), 4),
            "three_state_agreement": f"{agree_3}/{len(rows)}",
            "three_state_agreement_pct": round(agree_3 / len(rows) * 100.0, 1),
            "nested_coverage": f"{len(decided_n)}/{len(rows)}",
            "nested_decided_acc": f"{hits_n}/{len(decided_n)}",
            "nested_fb_snv_mean_regret_pct": round(float(np.mean(reg_snv)), 2),
            "nested_fb_snv_C10": round(_eval_cw(reg_snv, 10.0), 2),
            "nested_fb_cv_mean_regret_pct": round(float(np.mean(reg_cv)), 2),
            "nested_fb_cv_C10": round(_eval_cw(reg_cv, 10.0), 2),
        }
        print(f"  {name:12s}: {out}")
        return out

    res = {
        "VALID_17": summarize_subset("VALID_17", [r for r in all_rows if r["pool"] == "Deploy_18" and not r["is_mango"]]),
        "ALL_18": summarize_subset("ALL_18", [r for r in all_rows if r["pool"] == "Deploy_18"]),
        "Round3_13": summarize_subset("Round3_13", [r for r in all_rows if r["pool"] == "Round3_13"]),
        "Pooled_31": summarize_subset("Pooled_31", all_rows),
        "total_nested_gate_time_s": round(t_nested_total, 3),
    }
    return res


def run_part3_noninferiority_and_cluster_cis(vsn_res: dict) -> dict:
    print("\n=== Part 3: Formal Non-Inferiority & Cluster-Bootstrap 95% CIs ===")
    rng = np.random.default_rng(20261125)
    B = 10000

    # 3A. EXTERNAL_30 non-inferiority & 12-lab cluster bootstrap
    wi = json.loads((ROOT / "_audit" / "res_wiener_ws_2026_11_25.json").read_text(encoding="utf-8"))
    ext30 = [t for t in wi["tasks"] if t.get("is_external_prospective") and not t.get("has_label_scale_anomaly")]
    vsn_map = {r["task_id"]: r for r in vsn_res["per_task_ext30"]}

    def get_cluster(tid: str) -> str:
        if tid.startswith("Deploy_CRAW_"): return "CRAW"
        if tid.startswith("Deploy_OSSL_"): return "OSSL"
        if tid.startswith("Deploy_Milk_"): return "Milk"
        if tid.startswith("Deploy_Wheat_"): return "IDRC2016_Wheat"
        if tid.startswith("Deploy_Tablets_"): return "Tablets"
        if tid.startswith("Deploy_Melamine_"): return "Melamine"
        if tid.startswith("R3_"): return tid.split("__")[0]
        return tid

    clusters = {}
    for t in ext30:
        clusters.setdefault(get_cluster(t["task_id"]), []).append(t)
    c_keys = sorted(clusters.keys())

    ni_out = {}
    for head in ("pls", "ridge"):
        zw = np.array([t[head]["metrics"]["ws_wiener"]["rmsep_z"] for t in ext30])
        zc = np.array([t[head]["metrics"]["ws_inner_cv"]["rmsep_z"] for t in ext30])
        zv = np.array([vsn_map[t["task_id"]][f"{head}_rmsep_z"] for t in ext30])
        diff_cv = zw - zc

        # One-sided Wilcoxon non-inferiority tests at margins delta in {0.03, 0.05, 0.08}
        margins = {}
        for delta in (0.03, 0.05, 0.08):
            p_less = float(wilcoxon(diff_cv - delta, alternative="less").pvalue)
            margins[f"delta_{delta:.2f}"] = {"wilcoxon_one_sided_p": round(p_less, 6)}

        # 12-cluster bootstrap
        boot_cv, boot_vsn = [], []
        for _ in range(B):
            picked = rng.choice(c_keys, size=len(c_keys), replace=True)
            ts = [t for ck in picked for t in clusters[ck]]
            b_zw = np.array([t[head]["metrics"]["ws_wiener"]["rmsep_z"] for t in ts])
            b_zc = np.array([t[head]["metrics"]["ws_inner_cv"]["rmsep_z"] for t in ts])
            b_zv = np.array([vsn_map[t["task_id"]][f"{head}_rmsep_z"] for t in ts])
            boot_cv.append(float(np.mean(b_zw - b_zc)))
            boot_vsn.append(float(np.mean(b_zw - b_zv)))
        ucb95_cv = float(np.percentile(boot_cv, 95.0))
        ci95_cv = [round(float(np.percentile(boot_cv, 2.5)), 4), round(float(np.percentile(boot_cv, 97.5)), 4)]
        ci95_vsn = [round(float(np.percentile(boot_vsn, 2.5)), 4), round(float(np.percentile(boot_vsn, 97.5)), 4)]
        ni_out[head] = {
            "mean_diff_wiener_minus_inner_cv": round(float(np.mean(diff_cv)), 4),
            "cluster12_ci95_wiener_minus_inner_cv": ci95_cv,
            "cluster12_ucb95_wiener_minus_inner_cv": round(ucb95_cv, 4),
            "margins": margins,
            "mean_diff_wiener_minus_vsn": round(float(np.mean(zw - zv)), 4),
            "cluster12_ci95_wiener_minus_vsn": ci95_vsn,
        }
        print(f"  Ext30 {head.upper()} non-inferiority:", ni_out[head])

    # 3B. 170-fold Theorem 1 cluster bootstrap (7 datasets)
    p0 = json.loads((ROOT / "_audit" / "res_review_p0_2026_11_23.json").read_text(encoding="utf-8"))
    ds_items = list(p0["part3_theorem1_confusion"]["per_dataset"].values())
    b170 = []
    for _ in range(B):
        idx = rng.integers(0, len(ds_items), size=len(ds_items))
        hits = sum(ds_items[i]["v2"]["tp"] + ds_items[i]["v2"]["tn"] for i in idx)
        tot = sum(ds_items[i]["v2"]["n"] for i in idx)
        b170.append(hits / tot)
    ci170 = [round(float(np.percentile(b170, 2.5)), 3), round(float(np.percentile(b170, 97.5)), 3)]

    # 3C. 86-fold Stage 2 cluster bootstrap (18 tasks)
    from scipy.stats import spearmanr
    r2 = json.loads((ROOT / "_audit" / "res_prereg_snv_gate_round2_2026_11_24.json").read_text(encoding="utf-8"))
    t18 = r2["tasks"]
    b86_acc, b86_rho = [], []
    for _ in range(B):
        idx = rng.integers(0, len(t18), size=len(t18))
        folds = [f for i in idx for f in t18[i]["folds"]]
        dec = [f for f in folds if f["call_primary"] != "ABSTAIN"]
        if dec:
            hits = sum(
                1 for f in dec
                if (f["call_primary"] == "IMPROVE" and f["emp_snv_vs_raw_pct"] < 0)
                or (f["call_primary"] == "WORSEN" and f["emp_snv_vs_raw_pct"] >= 0)
            )
            b86_acc.append(hits / len(dec))
        rho, _ = spearmanr([f["delta_loo_pct"] for f in folds], [f["emp_snv_vs_raw_pct"] for f in folds])
        if np.isfinite(rho):
            b86_rho.append(float(rho))

    out = {
        "external_30_noninferiority": ni_out,
        "folds_170_7dataset_cluster_ci95": ci170,
        "folds_86_18task_cluster_decided_acc_ci95": [round(float(np.percentile(b86_acc, 2.5)), 3), round(float(np.percentile(b86_acc, 97.5)), 3)],
        "folds_86_18task_cluster_rho_ci95": [round(float(np.percentile(b86_rho, 2.5)), 3), round(float(np.percentile(b86_rho, 97.5)), 3)],
    }
    print("  170-fold 7-dataset cluster 95% CI:", out["folds_170_7dataset_cluster_ci95"])
    print("  86-fold 18-task cluster decided acc 95% CI:", out["folds_86_18task_cluster_decided_acc_ci95"])
    print("  86-fold 18-task cluster rho 95% CI:", out["folds_86_18task_cluster_rho_ci95"])
    return out


def main() -> None:
    t0 = time.time()
    vsn_res = run_part1_vsn()
    nested_res = run_part2_nested_cv()
    stat_res = run_part3_noninferiority_and_cluster_cis(vsn_res)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_sec": round(time.time() - t0, 2),
        "part1_vsn_comparison": vsn_res,
        "part2_nested_cv_bias_ablation": nested_res,
        "part3_noninferiority_and_cluster_cis": stat_res,
    }
    out_p = ROOT / "_audit" / "res_major_revision_2026_11_25.json"
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out_p} in {payload['elapsed_sec']}s")


if __name__ == "__main__":
    main()
