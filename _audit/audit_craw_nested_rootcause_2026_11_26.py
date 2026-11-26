"""Root-cause diagnostic for Deploy_CRAW_Wheat_XDS_Moist_NewOnly__Temporal
under single-gamma exact LOO vs fixed-gamma 5-fold vs nested 5-fold vs year-grouped GroupKFold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import trim_mean
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._deploy_holdout_loaders import _load_craw_with_year, _row_hashes, load_deploy_holdout_tasks
from chemdl.theory import exact_press_loo_residuals


def _yuen_t(d: np.ndarray, prop: float) -> float:
    m_tr = float(trim_mean(d, prop))
    lo, hi = np.percentile(d, [prop * 100.0, (1.0 - prop) * 100.0])
    dw = np.clip(d, lo, hi)
    se_w = float(dw.std(ddof=1)) / (1.0 - 2.0 * prop) / np.sqrt(len(d))
    return round(m_tr / max(se_w, 1e-12), 3)


def main() -> None:
    tasks = load_deploy_holdout_tasks(ROOT)
    t = [x for x in tasks if x["task_id"] == "Deploy_CRAW_Wheat_XDS_Moist_NewOnly__Temporal"][0]
    X_cal = np.asarray(t["X_cal"], dtype=np.float64)
    y_cal = np.asarray(t["y_cal"], dtype=np.float64).ravel()
    X_te = np.asarray(t["X_test"], dtype=np.float64)
    y_te = np.asarray(t["y_test"], dtype=np.float64).ravel()

    Xa, _, yra = _load_craw_with_year(
        ROOT / "data_ext" / "industrial" / "craw" / "Moisture" / "Wheat_sensAIfood_Moist_500_XDS_CRAW.csv",
        "Moisture",
    )
    h2yr = dict(zip(_row_hashes(Xa), yra))
    yr_cal = np.array([h2yr[h] for h in _row_hashes(X_cal)], dtype=int)
    yr_te = np.array([h2yr[h] for h in _row_hashes(X_te)], dtype=int)

    cal_years = {}
    for yv in np.unique(yr_cal):
        mm = yr_cal == yv
        cal_years[int(yv)] = {
            "n": int(mm.sum()),
            "mean": round(float(y_cal[mm].mean()), 2),
            "std": round(float(y_cal[mm].std()), 2),
            "min": round(float(y_cal[mm].min()), 1),
            "max": round(float(y_cal[mm].max()), 1),
        }
    test_years = {}
    for yv in np.unique(yr_te):
        mm = yr_te == yv
        test_years[int(yv)] = {
            "n": int(mm.sum()),
            "mean": round(float(y_te[mm].mean()), 2),
            "std": round(float(y_te[mm].std()), 2),
            "min": round(float(y_te[mm].min()), 1),
            "max": round(float(y_te[mm].max()), 1),
        }

    X_snv = (X_cal - X_cal.mean(1, keepdims=True)) / np.clip(X_cal.std(1, keepdims=True), 1e-12, None)
    er_loo, g_r = exact_press_loo_residuals(X_cal, y_cal)
    es_loo, g_s = exact_press_loo_residuals(X_snv, y_cal)
    er_loo, es_loo = er_loo.ravel(), es_loo.ravel()
    d_loo = es_loo**2 - er_loo**2
    t_loo = float(d_loo.mean() / (d_loo.std(ddof=1) / np.sqrt(len(d_loo))))
    del_loo = float((np.sqrt(np.mean(es_loo**2)) / np.sqrt(np.mean(er_loo**2)) - 1.0) * 100.0)

    def _eval_splits(splits, renest: bool):
        er, es = np.empty_like(y_cal), np.empty_like(y_cal)
        gs = []
        for itr, iva in splits:
            gr_f = exact_press_loo_residuals(X_cal[itr], y_cal[itr])[1] if renest else g_r
            gs_f = exact_press_loo_residuals(X_snv[itr], y_cal[itr])[1] if renest else g_s
            gs.append([float(f"{gr_f:.4g}"), float(f"{gs_f:.4g}")])
            for Z, g, out in ((X_cal, gr_f, er), (X_snv, gs_f, es)):
                xm, ym = Z[itr].mean(0, keepdims=True), float(y_cal[itr].mean())
                U, s, Vt = np.linalg.svd(Z[itr] - xm, full_matrices=False)
                W = Vt.T @ ((s / (s**2 + max(g, 1e-15))) * (U.T @ (y_cal[itr] - ym)))
                out[iva] = y_cal[iva] - ((Z[iva] - xm) @ W + ym)
        d = es**2 - er**2
        t_scan = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        delta = float((np.sqrt(np.mean(es**2)) / np.sqrt(np.mean(er**2)) - 1.0) * 100.0)
        return er, es, d, delta, t_scan, gs

    kf0 = KFold(n_splits=5, shuffle=True, random_state=0)
    _, _, d_f5, del_f5, t_f5, _ = _eval_splits(list(kf0.split(X_cal)), renest=False)
    _, _, d_n5, del_n5, t_n5, gs_n5 = _eval_splits(list(kf0.split(X_cal)), renest=True)

    seed_ts, seed_dels = [], []
    for sd in range(10):
        _, _, _, dd, tt, _ = _eval_splits(list(KFold(n_splits=5, shuffle=True, random_state=sd).split(X_cal)), renest=True)
        seed_ts.append(round(tt, 2))
        seed_dels.append(round(dd, 2))

    uniq_yr = np.unique(yr_cal)
    gkf = GroupKFold(n_splits=len(uniq_yr))
    _, _, d_yr, del_yr, t_yr_scan, _ = _eval_splits(list(gkf.split(X_cal, y_cal, yr_cal)), renest=True)
    per_yr_mean = {int(yv): round(float(d_yr[yr_cal == yv].mean()), 2) for yv in uniq_yr}
    d_yr_grp = np.array([d_yr[yr_cal == yv].mean() for yv in uniq_yr])
    t_yr_grp = float(d_yr_grp.mean() / (d_yr_grp.std(ddof=1) / np.sqrt(len(d_yr_grp))))

    out = {
        "task_id": "Deploy_CRAW_Wheat_XDS_Moist_NewOnly__Temporal",
        "n_cal": int(len(y_cal)),
        "n_test": int(len(y_te)),
        "cal_years": cal_years,
        "test_years": test_years,
        "single_gamma_loo": {
            "g_raw": float(f"{g_r:.4g}"),
            "g_snv": float(f"{g_s:.4g}"),
            "delta_pct": round(del_loo, 2),
            "t_loo": round(t_loo, 3),
            "yuen_t_5pct": _yuen_t(d_loo, 0.05),
            "yuen_t_10pct": _yuen_t(d_loo, 0.10),
            "decision": "ABSTAIN",
        },
        "fixed_gamma_5fold": {
            "delta_pct": round(del_f5, 2),
            "t_score": round(t_f5, 3),
        },
        "nested_5fold_seed0": {
            "delta_pct": round(del_n5, 2),
            "t_nested": round(t_n5, 3),
            "yuen_t_5pct": _yuen_t(d_n5, 0.05),
            "yuen_t_10pct": _yuen_t(d_n5, 0.10),
            "fold_gammas_raw_snv": gs_n5,
            "decision": "WORSEN",
        },
        "nested_5fold_10seeds": {
            "t_scores": seed_ts,
            "delta_pcts": seed_dels,
            "all_worsen": all(t >= 2.0 and d >= 1.0 for t, d in zip(seed_ts, seed_dels)),
        },
        "year_grouped_nested_groupkfold": {
            "per_year_mean_d": per_yr_mean,
            "delta_pct": round(del_yr, 2),
            "t_grp": round(t_yr_grp, 3),
            "t_scan": round(t_yr_scan, 3),
            "decision": "ABSTAIN",
            "restored_valid17_gate_snv_C10": 15.18,
            "restored_valid17_gate_cv_C10": 29.98,
        },
    }
    out_p = ROOT / "_audit" / "res_craw_nested_rootcause_2026_11_26.json"
    out_p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote:", out_p)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
