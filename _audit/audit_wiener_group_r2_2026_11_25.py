"""伴随审计（Codex E2）：对全部 8 项含重复个体/地块/光谱组别的任务，用 GroupKFold 重算内层 R²_shape 并比较留出误差。

说明：
  - 保留 Phase 1 锁单 `0f5fcc4`（行级 KFold R²_shape）不变以维持预注册链条；
  - 本脚本仅用训练集 (X_cal, y_cal, groups_cal) 计算严格按 groups_cal 分组的 `GroupKFold` R²_shape_group
    与 `w_wiener_group = 1 - r2_s_shape_group`，按完全相同的内层 CV（PLS k / Ridge gamma）
    比较留出集 RMSEP_z 变化及对 EXTERNAL_30 / VALID_53 汇总的影响。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._ws_closed_value_loaders import (
    FIXED_WC,
    fit_eval_pls_at_ws,
    fit_eval_ridge_at_ws,
    load_ws_value_tasks,
    make_inner_splits,
    select_gamma_ridge_for_ws,
    select_k_pls_for_ws,
)


def _compute_group_r2_shape(X_cal: np.ndarray, groups_cal: np.ndarray) -> tuple[float, int]:
    X = np.asarray(X_cal, dtype=np.float64)
    grp = np.asarray(groups_cal)
    n, L = X.shape
    mu_row = X.mean(axis=1)
    sd_row = np.clip(X.std(axis=1), 1e-12, None)
    X_snv = (X - mu_row[:, None]) / sd_row[:, None]
    uniq = np.unique(grp)
    eff_k = min(3, len(uniq))
    assert eff_k >= 2
    splits = list(GroupKFold(n_splits=eff_k).split(X_snv, grp, grp))
    max_lv_shape = min(6, max(1, min(len(itr) for itr, _ in splits) // 5), L)

    tsd = float(sd_row.std())
    if tsd < 1e-12:
        return 0.0, int(len(uniq))
    zn = (sd_row - sd_row.mean()) / tsd
    best_mse = 1.0
    for lv in range(1, max_lv_shape + 1):
        mses = []
        for itr, iva in splits:
            m = PLSRegression(n_components=min(lv, len(itr) - 1), scale=False).fit(X_snv[itr], zn[itr])
            p = m.predict(X_snv[iva]).ravel()
            mses.append(float(np.mean((p - zn[iva]) ** 2)))
        best_mse = min(best_mse, float(np.mean(mses)))
    return float(np.clip(1.0 - best_mse, 0.0, 0.98)), int(len(uniq))


def main() -> None:
    res_prev = json.loads((ROOT / "_audit" / "res_wiener_ws_2026_11_25.json").read_text(encoding="utf-8"))
    pre_prev = json.loads((ROOT / "_audit" / "prereg_wiener_ws_2026_11_25.json").read_text(encoding="utf-8"))
    pre_map = {r["task_id"]: r for r in pre_prev["tasks"]}
    res_map = {r["task_id"]: r for r in res_prev["tasks"]}

    tasks = [t for t in load_ws_value_tasks(ROOT) if t["groups_cal"] is not None]
    assert len(tasks) == 8, f"Expected 8 grouped tasks, got {len(tasks)}"

    rows = []
    for t in tasks:
        tid = t["task_id"]
        Xc, yc, gc = t["X_cal"], t["y_cal"], t["groups_cal"]
        Xt, yt = t["X_test"], t["y_test"]
        r2_row = float(pre_map[tid]["physics_frozen"]["r2_s_shape"])
        ws_row = float(pre_map[tid]["ws_wiener"])

        r2_grp, n_uniq = _compute_group_r2_shape(Xc, gc)
        ws_grp = round(float(1.0 - r2_grp), 6)

        mu = float(yc.mean())
        sd = max(float(yc.std()), 1e-9)
        yn = (yc - mu) / sd
        splits = make_inner_splits(len(Xc), gc, n_splits=5, seed=0)

        k_grp, _ = select_k_pls_for_ws(Xc, yn, ws_grp, splits, kmax=12, wc=FIXED_WC)
        e_pls_grp = fit_eval_pls_at_ws(Xc, yc, Xt, yt, ws_grp, k_grp, wc=FIXED_WC)["rmsep_z"]
        g_grp, _ = select_gamma_ridge_for_ws(Xc, yn, ws_grp, groups_cal=gc, wc=FIXED_WC)
        e_ridge_grp = fit_eval_ridge_at_ws(Xc, yc, Xt, yt, ws_grp, g_grp, wc=FIXED_WC)["rmsep_z"]

        e_pls_row = float(res_map[tid]["pls"]["metrics"]["ws_wiener"]["rmsep_z"])
        e_ridge_row = float(res_map[tid]["ridge"]["metrics"]["ws_wiener"]["rmsep_z"])

        rows.append(
            {
                "task_id": tid,
                "suite": t["suite"],
                "n_cal": int(len(Xc)),
                "n_unique_groups_cal": int(n_uniq),
                "r2_s_shape_row_kfold": round(r2_row, 4),
                "r2_s_shape_group_kfold": round(r2_grp, 4),
                "ws_wiener_row": round(ws_row, 4),
                "ws_wiener_group": round(ws_grp, 4),
                "delta_ws_wiener": round(ws_grp - ws_row, 4),
                "pls_rmsep_z_row": round(e_pls_row, 6),
                "pls_rmsep_z_group": round(e_pls_grp, 6),
                "delta_pls_rmsep_z": round(e_pls_grp - e_pls_row, 6),
                "ridge_rmsep_z_row": round(e_ridge_row, 6),
                "ridge_rmsep_z_group": round(e_ridge_grp, 6),
                "delta_ridge_rmsep_z": round(e_ridge_grp - e_ridge_row, 6),
            }
        )
        print(
            f"{tid:40s} | grps={n_uniq:3d} | R2: {r2_row:.3f}->{r2_grp:.3f} | "
            f"ws: {ws_row:.3f}->{ws_grp:.3f} | "
            f"PLS: {e_pls_row:.4f}->{e_pls_grp:.4f} ({e_pls_grp-e_pls_row:+.4f}) | "
            f"Ridge: {e_ridge_row:.4f}->{e_ridge_grp:.4f} ({e_ridge_grp-e_ridge_row:+.4f})"
        )

    out = {
        "audit": "E2_GROUPKFOLD_R2_SHAPE_SENSITIVITY",
        "n_grouped_tasks": len(rows),
        "max_abs_delta_ws_wiener": round(max(abs(r["delta_ws_wiener"]) for r in rows), 4),
        "max_abs_delta_pls_rmsep_z": round(max(abs(r["delta_pls_rmsep_z"]) for r in rows), 6),
        "max_abs_delta_ridge_rmsep_z": round(max(abs(r["delta_ridge_rmsep_z"]) for r in rows), 6),
        "tasks": rows,
    }
    out_p = ROOT / "_audit" / "res_wiener_group_r2_audit_2026_11_25.json"
    out_p.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"\nSaved E2 GroupKFold sensitivity audit -> {out_p}")


if __name__ == "__main__":
    main()
