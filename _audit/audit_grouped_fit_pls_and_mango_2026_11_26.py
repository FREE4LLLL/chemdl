"""Standalone audit for:
1) Nested 5-fold vs single-gamma exact per-task breakdown (4 flipped tasks);
2) Grouped fit_pls inner-CV sensitivity on all 5 grouped deploy/external tasks;
3) Mango DM_test2 (100 - y) unit-complement sensitivity analysis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._deploy_holdout_loaders import load_deploy_holdout_tasks
from _audit._round3_unseen_loaders import load_round3_unseen_tasks
from _audit.bench_major_revision_2026_11_25 import _nested_gate_task
from chemdl.transfer import fit_pls, fit_prep_fn, rmsep


def _fit_pls_grouped(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kmax: int = 10, n_splits: int = 4):
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if ya.ndim == 1:
        ya = ya[:, None]
    grp = np.asarray(groups)
    eff = min(n_splits, len(np.unique(grp)))
    splits = list(GroupKFold(n_splits=eff).split(Xa, ya, grp))
    max_k = min(kmax, min(len(tr) for tr, _ in splits) - 1, Xa.shape[1])
    best_k, best_mse = 1, float("inf")
    for k in range(1, max_k + 1):
        se = []
        for tr, va in splits:
            m = PLSRegression(n_components=k, scale=False)
            m.fit(Xa[tr], ya[tr])
            se.append((m.predict(Xa[va]).reshape(-1, 1) - ya[va]) ** 2)
        mse = float(np.mean(np.concatenate(se, axis=0)))
        if mse < best_mse:
            best_mse, best_k = mse, k
    m = PLSRegression(n_components=best_k, scale=False)
    m.fit(Xa, ya)
    return (lambda Z: m.predict(np.asarray(Z, dtype=np.float64)).reshape(-1, 1)), best_k


def _inner_cv_raw_vs_snv_grouped(X: np.ndarray, y: np.ndarray, groups: np.ndarray, fn_snv, kmax: int = 10):
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if ya.ndim == 1:
        ya = ya[:, None]
    grp = np.asarray(groups)
    eff = min(5, len(np.unique(grp)))
    splits = list(GroupKFold(n_splits=eff).split(Xa, ya, grp))
    sq_r, sq_s = [], []
    for itr, iva in splits:
        inner = min(4, len(np.unique(grp[itr])))
        pr, _ = _fit_pls_grouped(Xa[itr], ya[itr], grp[itr], kmax=kmax, n_splits=inner)
        ps, _ = _fit_pls_grouped(fn_snv(Xa[itr]), ya[itr], grp[itr], kmax=kmax, n_splits=inner)
        sq_r.append((pr(Xa[iva]) - ya[iva]) ** 2)
        sq_s.append((ps(fn_snv(Xa[iva])) - ya[iva]) ** 2)
    cr = float(np.sqrt(np.mean(np.concatenate(sq_r, axis=0))))
    cs = float(np.sqrt(np.mean(np.concatenate(sq_s, axis=0))))
    return ("SNV" if cs < cr else "RAW"), round(cr, 4), round(cs, 4)


def _c10(errs: list[float], oras: list[float]) -> dict:
    regs = [(e / o - 1.0) * 100.0 for e, o in zip(errs, oras)]
    c10 = float(np.mean(regs) + 9.0 * np.mean([max(r - 5.0, 0.0) for r in regs]))
    return {
        "mean_z": round(float(np.mean(errs)), 4),
        "mean_regret_pct": round(float(np.mean(regs)), 2),
        "worst_regret_pct": round(float(np.max(regs)), 2),
        "severe_count": int(sum(r >= 5.0 for r in regs)),
        "C10": round(c10, 2),
    }


def main() -> None:
    d18_tasks = load_deploy_holdout_tasks(ROOT)
    r3_tasks = load_round3_unseen_tasks(ROOT)
    d18_arch = {t["task_id"]: t for t in json.loads((ROOT / "_audit" / "res_prereg_deploy_holdout_2026_11_24.json").read_text(encoding="utf-8"))["tasks"]}
    r3_arch = {t["task_id"]: t for t in json.loads((ROOT / "_audit" / "res_prereg_round3_unseen_2026_11_24.json").read_text(encoding="utf-8"))["tasks"]}
    arch_all = {**d18_arch, **r3_arch}

    # 1. Nested vs single-gamma exact per-task breakdown for flipped tasks
    flipped_tasks = []
    for pool_name, tasks in (("Deploy_18", d18_tasks), ("Round3_13", r3_tasks)):
        for t in tasks:
            tid = t["task_id"]
            ar = arch_all[tid]
            ng = _nested_gate_task(t["X_cal"], t["y_cal"], t.get("groups_cal"))
            if ng["decision"] != ar["gate_decision"]:
                flipped_tasks.append({
                    "pool": pool_name,
                    "task_id": tid,
                    "single_decision": ar["gate_decision"],
                    "t_loo": float(ar["t_loo"]),
                    "delta_loo_pct": float(ar["delta_loo_pct"]),
                    "nested_decision": ng["decision"],
                    "t_nested": float(ng["t_nested"]),
                    "delta_nested_pct": float(ng["delta_nested_pct"]),
                    "rmsep_raw_z": float(ar["rmsep_raw_z"]),
                    "rmsep_snv_z": float(ar["rmsep_snv_z"]),
                    "emp_snv_vs_raw_pct": float(ar["emp_snv_vs_raw_pct"]),
                })

    # 2. Grouped fit_pls inner-CV sensitivity on the 5 grouped tasks
    grouped_rows = {}
    for t in d18_tasks + r3_tasks:
        if t.get("groups_cal") is None:
            continue
        tid = t["task_id"]
        ar = arch_all[tid]
        X_cal, y_cal, X_te, y_te, g_cal = t["X_cal"], t["y_cal"], t["X_test"], t["y_test"], t["groups_cal"]
        mu, sd = float(y_cal.mean()), max(float(y_cal.std()), 1e-9)
        yn_c = ((y_cal - mu) / sd)[:, None]
        yn_t = ((y_te - mu) / sd)[:, None]
        fn_snv = fit_prep_fn("snv", X_cal)
        kmax = min(10, max(2, len(X_cal) // 2))
        inner = min(4, max(2, len(X_cal) // 4))

        pr_r0, kr0 = fit_pls(X_cal, yn_c, kmax=kmax, inner=inner, per_col=True, seed=0)
        pr_s0, ks0 = fit_pls(fn_snv(X_cal), yn_c, kmax=kmax, inner=inner, per_col=True, seed=0)
        er0, es0 = float(rmsep(pr_r0(X_te), yn_t)), float(rmsep(pr_s0(fn_snv(X_te)), yn_t))

        pr_rg, krg = _fit_pls_grouped(X_cal, yn_c, g_cal, kmax=kmax, n_splits=inner)
        pr_sg, ksg = _fit_pls_grouped(fn_snv(X_cal), yn_c, g_cal, kmax=kmax, n_splits=inner)
        erg, esg = float(rmsep(pr_rg(X_te), yn_t)), float(rmsep(pr_sg(fn_snv(X_te)), yn_t))
        pick_g, crg, csg = _inner_cv_raw_vs_snv_grouped(X_cal, yn_c, g_cal, fn_snv, kmax=kmax)

        grouped_rows[tid] = {
            "gate_decision": ar["gate_decision"],
            "orig_inner_cv_pick": ar["inner_cv_pick"],
            "grouped_inner_cv_pick": pick_g,
            "orig_k_raw": int(kr0[0]),
            "orig_k_snv": int(ks0[0]),
            "orig_rmsep_raw_z": round(er0, 4),
            "orig_rmsep_snv_z": round(es0, 4),
            "grouped_k_raw": int(krg),
            "grouped_k_snv": int(ksg),
            "grouped_rmsep_raw_z": round(erg, 4),
            "grouped_rmsep_snv_z": round(esg, 4),
            "grouped_cv_raw_z": crg,
            "grouped_cv_snv_z": csg,
        }

    pool_grouped = {}
    for pname, rows in (("VALID_17", [r for r in d18_arch.values() if "Mango" not in r["task_id"]]), ("Round3_13", list(r3_arch.values()))):
        pool_grouped[pname] = {}
        for strat in ("Gate_Fallback_SNV", "Gate_Fallback_InnerCV", "InnerCV_PLS", "Always_Raw", "Always_SNV"):
            errs, oras = [], []
            for r in rows:
                tid = r["task_id"]
                if tid in grouped_rows:
                    gr = grouped_rows[tid]
                    er, es, cvp = gr["grouped_rmsep_raw_z"], gr["grouped_rmsep_snv_z"], gr["grouped_inner_cv_pick"]
                else:
                    er, es, cvp = float(r["rmsep_raw_z"]), float(r["rmsep_snv_z"]), str(r["inner_cv_pick"])
                ora = min(er, es)
                gate = r["gate_decision"]
                if strat == "Gate_Fallback_SNV":
                    e = er if gate == "WORSEN" else es
                elif strat == "Gate_Fallback_InnerCV":
                    e = es if gate == "IMPROVE" else (er if gate == "WORSEN" else (es if cvp == "SNV" else er))
                elif strat == "InnerCV_PLS":
                    e = es if cvp == "SNV" else er
                elif strat == "Always_Raw":
                    e = er
                else:
                    e = es
                errs.append(e)
                oras.append(ora)
            pool_grouped[pname][strat] = _c10(errs, oras)

    # 3. Mango 100 - DM_test2 sensitivity
    mg = sio.loadmat(ROOT / "data_ext" / "ct_bench" / "mango.mat")
    X_cal = mg["SP_cal"].astype(float)
    y_cal = mg["DM_cal"].ravel().astype(float)
    X_te = mg["SP_test2"].astype(float)
    y_te_orig = mg["DM_test2"].ravel().astype(float)
    y_te_comp = 100.0 - y_te_orig
    mu, sd = float(y_cal.mean()), float(y_cal.std(ddof=0))
    yn_cal = ((y_cal - mu) / sd)[:, None]
    yn_te_comp = ((y_te_comp - mu) / sd)[:, None]
    fn_snv = fit_prep_fn("snv", X_cal)
    kmax = min(10, max(2, len(X_cal) // 2))
    inner = min(4, max(2, len(X_cal) // 4))
    pr_raw, _ = fit_pls(X_cal, yn_cal, kmax=kmax, inner=inner, per_col=True, seed=0)
    pr_snv, _ = fit_pls(fn_snv(X_cal), yn_cal, kmax=kmax, inner=inner, per_col=True, seed=0)
    er_comp = float(rmsep(pr_raw(X_te), yn_te_comp))
    es_comp = float(rmsep(pr_snv(fn_snv(X_te)), yn_te_comp))

    mango_18_comp = {}
    for strat in ("Gate_Fallback_SNV", "Gate_Fallback_InnerCV", "InnerCV_PLS", "Always_Raw", "Always_SNV"):
        errs, oras = [], []
        for r in d18_arch.values():
            if "Mango" in r["task_id"]:
                er, es, cvp = er_comp, es_comp, r["inner_cv_pick"]
            else:
                er, es, cvp = float(r["rmsep_raw_z"]), float(r["rmsep_snv_z"]), str(r["inner_cv_pick"])
            ora = min(er, es)
            gate = r["gate_decision"]
            if strat == "Gate_Fallback_SNV":
                e = er if gate == "WORSEN" else es
            elif strat == "Gate_Fallback_InnerCV":
                e = es if gate == "IMPROVE" else (er if gate == "WORSEN" else (es if cvp == "SNV" else er))
            elif strat == "InnerCV_PLS":
                e = es if cvp == "SNV" else er
            elif strat == "Always_Raw":
                e = er
            else:
                e = es
            errs.append(e)
            oras.append(ora)
        mango_18_comp[strat] = _c10(errs, oras)

    out = {
        "flipped_tasks_nested_vs_single": flipped_tasks,
        "grouped_fit_pls_per_task": grouped_rows,
        "grouped_fit_pls_pools": pool_grouped,
        "mango_100_minus_y": {
            "has_text_dictionary_in_mat": False,
            "dm_cal_mean": round(float(y_cal.mean()), 2),
            "dm_cal_std": round(float(y_cal.std(ddof=0)), 2),
            "dm_test2_mean": round(float(y_te_orig.mean()), 2),
            "dm_test2_std": round(float(y_te_orig.std(ddof=0)), 2),
            "comp_mean": round(float(y_te_comp.mean()), 2),
            "comp_std": round(float(y_te_comp.std(ddof=0)), 2),
            "pred_raw_mean": round(float(pr_raw(X_te).mean() * sd + mu), 2),
            "rmsep_raw_z_comp": round(er_comp, 4),
            "rmsep_snv_z_comp": round(es_comp, 4),
            "emp_snv_vs_raw_pct_comp": round((es_comp / er_comp - 1.0) * 100.0, 2),
            "all_18_with_comp": mango_18_comp,
        },
    }
    out_path = ROOT / "_audit" / "res_grouped_fit_pls_and_mango_2026_11_26.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote:", out_path)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
