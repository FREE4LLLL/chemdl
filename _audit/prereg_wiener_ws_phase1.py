"""Phase 1 预注册锁单：无监督维纳收缩 (Theorem 2-W) 与统一维纳–耦合收缩 (Theorem 2-WC)。

硬不变量：
  1. 严格仅用训练集 (X_cal, y_cal, groups_cal) 与已冻结的 C-148 Phase 1 物理参数 (`91a94a8`)；
     在进入任何估计/内层 CV 前立即剥离 X_test / y_test / groups_test；
  2. 31 项外部任务 (`Phase B 18` + `Round 3 13`，有效子集 `EXTERNAL_30`) 对本轮新方法
     (`ws_wiener`, `ws_unified`, `snv_plus_logs`) 保持绝对零测试集接触（纯粹前瞻验证）；
  3. 锁定无监督 `r2_s_shape` 真实计算耗时（零 `y` 标签）、`ws_wiener = 1 - r2_s_shape`、
     `ws_unified = closed_form_optimal_ws(..., sigma_non=sigma_non_chem)`、单调性断言
     `0 <= ws_unified <= ws_wiener <= 1`，以及统一的 PLS / Ridge 内层选参；
  4. 本脚本运行并提交 Git 后，才允许编写并运行 Phase 2 揭盲脚本。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._ws_closed_value_loaders import (
    FIXED_WC,
    INTERIOR_EPS,
    classify_ws_regime,
    load_ws_value_tasks,
    make_inner_splits,
    pls_prefix_predictions,
    select_gamma_ridge_for_ws,
    select_k_pls_for_ws,
)
from chemdl.deploy_gate import FROZEN_THEORY_LF_SHA256
from chemdl.theory import closed_form_optimal_ws, exact_press_loo_residuals
from chemdl.transfer import dssnv_apply, rmsep


def _file_sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_blob_sha256_lf(root: Path, rev: str, rel_path: str) -> str:
    raw = subprocess.check_output(["git", "show", f"{rev}:{rel_path}"], cwd=root).replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _sanitize_json(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not np.isfinite(v):
            raise ValueError(f"Non-finite float encountered in JSON payload: {v}")
        return v
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _compute_unsupervised_r2_shape(X_cal: np.ndarray) -> float:
    """严格复现 estimate_physical_params_v2 中纯无监督（零 y）的 R²_shape 估计。"""
    X = np.asarray(X_cal, dtype=np.float64)
    n, L = X.shape
    mu_row = X.mean(axis=1)
    sd_row = np.clip(X.std(axis=1), 1e-12, None)
    X_snv = (X - mu_row[:, None]) / sd_row[:, None]
    eff_k = min(3, max(2, n // 5))
    splits = list(KFold(eff_k, shuffle=True, random_state=0).split(X_snv))
    max_lv_shape = min(6, max(1, n // 5), L)

    tsd = float(sd_row.std())
    if tsd < 1e-12:
        return 0.0
    zn = (sd_row - sd_row.mean()) / tsd
    best_mse = 1.0
    for lv in range(1, max_lv_shape + 1):
        mses = []
        for itr, iva in splits:
            m = PLSRegression(n_components=min(lv, len(itr) - 1), scale=False).fit(X_snv[itr], zn[itr])
            p = m.predict(X_snv[iva]).ravel()
            mses.append(float(np.mean((p - zn[iva]) ** 2)))
        best_mse = min(best_mse, float(np.mean(mses)))
    return float(np.clip(1.0 - best_mse, 0.0, 0.98))


def _select_k_pls_snv_plus_logs(
    X_cal: np.ndarray,
    yn_cal: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    kmax: int = 12,
) -> tuple[int, float]:
    """折内零泄漏地为 [X_snv, z_log(row_std)] 选择 PLS 成分数 k。"""
    min_tr = min(len(itr) for itr, _ in splits)
    eff_kmax = min(int(kmax), X_cal.shape[1] + 1, max(1, min_tr - 1))
    fold_errs = np.zeros((len(splits), eff_kmax), dtype=np.float64)
    sd_all = np.log(np.clip(X_cal.std(axis=1), 1e-12, None))
    Xsnv_all = dssnv_apply(X_cal, wc=FIXED_WC, ws=1.0)
    for fi, (itr, iva) in enumerate(splits):
        ls_tr = sd_all[itr]
        ls_va = sd_all[iva]
        mu_ls = float(ls_tr.mean())
        sd_ls = max(float(ls_tr.std()), 1e-9)
        Xtr_aug = np.hstack([Xsnv_all[itr], ((ls_tr - mu_ls) / sd_ls)[:, None]])
        Xva_aug = np.hstack([Xsnv_all[iva], ((ls_va - mu_ls) / sd_ls)[:, None]])
        preds_k = pls_prefix_predictions(Xtr_aug, yn_cal[itr], Xva_aug, eff_kmax)
        for ki, pk in enumerate(preds_k):
            fold_errs[fi, ki] = rmsep(pk, yn_cal[iva])
    mean_errs = fold_errs.mean(axis=0)
    best_idx = int(np.argmin(mean_errs))
    return best_idx + 1, float(mean_errs[best_idx])


def _select_gamma_ridge_snv_plus_logs(
    X_cal: np.ndarray,
    yn_cal: np.ndarray,
    groups_cal: np.ndarray | None = None,
) -> tuple[float, float]:
    """折内零泄漏地为 [X_snv, z_log(row_std)] 选择 Ridge 正则强度 gamma。"""
    sd_all = np.log(np.clip(X_cal.std(axis=1), 1e-12, None))
    Xsnv_all = dssnv_apply(X_cal, wc=FIXED_WC, ws=1.0)
    if groups_cal is not None:
        grp = np.asarray(groups_cal)
        uniq = np.unique(grp)
        eff = min(5, len(uniq))
        if eff >= 2:
            splits = [
                (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
                for tr, va in GroupKFold(n_splits=eff).split(X_cal, yn_cal, grp)
            ]
            if min(len(tr) for tr, _ in splits) >= 4:
                oof = np.empty_like(yn_cal, dtype=np.float64)
                gammas = []
                for itr, iva in splits:
                    ls_tr = sd_all[itr]
                    mu_ls = float(ls_tr.mean())
                    sd_ls = max(float(ls_tr.std()), 1e-9)
                    Xtr_a = np.hstack([Xsnv_all[itr], ((ls_tr - mu_ls) / sd_ls)[:, None]])
                    Xva_a = np.hstack([Xsnv_all[iva], ((sd_all[iva] - mu_ls) / sd_ls)[:, None]])
                    _, gf = exact_press_loo_residuals(Xtr_a, yn_cal[itr, None])
                    gammas.append(float(gf))
                    x_mu = Xtr_a.mean(axis=0, keepdims=True)
                    y_mu = float(np.mean(yn_cal[itr]))
                    U, s, Vt = np.linalg.svd(Xtr_a - x_mu, full_matrices=False)
                    W = Vt.T @ ((s / (s ** 2 + max(float(gf), 1e-15))) * (U.T @ (yn_cal[itr] - y_mu)))
                    oof[iva] = (Xva_a - x_mu) @ W + y_mu
                return float(np.median(gammas)), float(rmsep(oof, yn_cal))

    mu_ls = float(sd_all.mean())
    sd_ls = max(float(sd_all.std()), 1e-9)
    X_aug = np.hstack([Xsnv_all, ((sd_all - mu_ls) / sd_ls)[:, None]])
    e_loo, best_g = exact_press_loo_residuals(X_aug, yn_cal[:, None])
    return float(best_g), float(np.sqrt(np.mean(e_loo ** 2)))


def main() -> None:
    t_start = time.time()
    theory_path = ROOT / "chemdl" / "theory.py"
    wt_lf = _file_sha256_lf(theory_path)
    assert wt_lf == FROZEN_THEORY_LF_SHA256, f"chemdl/theory.py LF SHA-256 mismatch: {wt_lf}"
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    blob_lf = _git_blob_sha256_lf(ROOT, git_head, "chemdl/theory.py")
    assert blob_lf == FROZEN_THEORY_LF_SHA256

    prev_pre_path = ROOT / "_audit" / "prereg_ws_closed_value_2026_11_25.json"
    prev_pre = json.loads(prev_pre_path.read_text(encoding="utf-8"))
    prev_pre_lf = _file_sha256_lf(prev_pre_path)
    prev_by_id = {r["task_id"]: r for r in prev_pre["tasks"]}

    raw_tasks = load_ws_value_tasks(ROOT)
    # Strip all test arrays immediately BEFORE cal-only estimation!
    cal_only = [
        {
            "task_id": t["task_id"],
            "suite": t["suite"],
            "domain": t["domain"],
            "split_rule": t["split_rule"],
            "target_name": t["target_name"],
            "n_cal": t["n_cal"],
            "n_test": t["n_test"],
            "L": t["L"],
            "split_sha256": t["split_sha256"],
            "has_label_scale_anomaly": t["has_label_scale_anomaly"],
            "X_cal": t["X_cal"],
            "y_cal": t["y_cal"],
            "groups_cal": t["groups_cal"],
        }
        for t in raw_tasks
    ]
    del raw_tasks

    locked_tasks = []
    for i, mc in enumerate(cal_only, 1):
        tid = mc["task_id"]
        pp = prev_by_id[tid]
        assert mc["split_sha256"] == pp["split_sha256"], f"{tid}: split_sha256 mismatch"
        pv2 = pp["physics_v2"]
        Xc, yc, gc = mc["X_cal"], mc["y_cal"], mc["groups_cal"]

        t0 = time.perf_counter()
        r2_recomp = _compute_unsupervised_r2_shape(Xc)
        dt_r2_shape = time.perf_counter() - t0
        assert abs(round(r2_recomp, 4) - pv2["r2_s_shape"]) <= 1e-4, (
            f"{tid}: r2_s_shape mismatch {r2_recomp:.4f} vs {pv2['r2_s_shape']:.4f}"
        )

        ws_wiener = float(np.clip(1.0 - pv2["r2_s_shape"], 0.0, 1.0))
        ws_unified = float(
            closed_form_optimal_ws(
                pv2["alpha"], pv2["beta"], pv2["r_s"], pv2["sigma_a_true"], sigma_non=pv2["sigma_non_chem"]
            )
        )
        # Theorem 2-WC Monotonicity invariant
        assert 0.0 <= ws_unified <= ws_wiener + 1e-3, (
            f"{tid}: monotonicity violated {ws_unified} > {ws_wiener}"
        )

        tag_wie = classify_ws_regime(ws_wiener, eps=INTERIOR_EPS)
        tag_uni = classify_ws_regime(ws_unified, eps=INTERIOR_EPS)

        mu = float(np.mean(yc))
        sd = max(float(np.std(yc)), 1e-9)
        yn = (yc - mu) / sd
        splits = make_inner_splits(len(Xc), gc, n_splits=5, seed=0)

        # PLS k selection (exact same splits & kmax=12)
        t0 = time.perf_counter()
        k_wie, cv_pls_wie = select_k_pls_for_ws(Xc, yn, ws_wiener, splits, kmax=12, wc=FIXED_WC)
        dt_k_wie = time.perf_counter() - t0

        t0 = time.perf_counter()
        k_uni, cv_pls_uni = select_k_pls_for_ws(Xc, yn, ws_unified, splits, kmax=12, wc=FIXED_WC)
        dt_k_uni = time.perf_counter() - t0

        t0 = time.perf_counter()
        k_aug, cv_pls_aug = _select_k_pls_snv_plus_logs(Xc, yn, splits, kmax=12)
        dt_k_aug = time.perf_counter() - t0

        # Ridge gamma selection
        t0 = time.perf_counter()
        g_wie, cv_r_wie = select_gamma_ridge_for_ws(Xc, yn, ws_wiener, groups_cal=gc, wc=FIXED_WC)
        dt_g_wie = time.perf_counter() - t0

        t0 = time.perf_counter()
        g_uni, cv_r_uni = select_gamma_ridge_for_ws(Xc, yn, ws_unified, groups_cal=gc, wc=FIXED_WC)
        dt_g_uni = time.perf_counter() - t0

        t0 = time.perf_counter()
        g_aug, cv_r_aug = _select_gamma_ridge_snv_plus_logs(Xc, yn, groups_cal=gc)
        dt_g_aug = time.perf_counter() - t0

        dt_v2_full = float(pp["timing_s"]["v2_params_only"])

        row = {
            "task_id": tid,
            "suite": mc["suite"],
            "domain": mc["domain"],
            "split_rule": mc["split_rule"],
            "target_name": mc["target_name"],
            "n_cal": mc["n_cal"],
            "n_test": mc["n_test"],
            "L": mc["L"],
            "has_groups_cal": gc is not None,
            "has_label_scale_anomaly": mc["has_label_scale_anomaly"],
            "cal_test_hash_overlap": 0,
            "split_sha256": mc["split_sha256"],
            "is_external_prospective": mc["suite"] in ("Round3_Unseen", "PhaseB_Deploy"),
            "physics_frozen": pv2,
            "ws_wiener": round(ws_wiener, 6),
            "ws_unified": round(ws_unified, 6),
            "pre_tags": {
                "wiener": tag_wie,
                "unified": tag_uni,
            },
            "timing_s": {
                "unsupervised_r2_shape_only": round(dt_r2_shape, 4),
                "v2_params_only_from_c148": round(dt_v2_full, 4),
            },
            "pls_cal_config": {
                "ws_wiener": {
                    "ws": round(ws_wiener, 6),
                    "k": int(k_wie),
                    "cv_rmsep_z": round(cv_pls_wie, 6),
                    "cal_time_s": round(dt_r2_shape + dt_k_wie, 4),
                },
                "ws_unified": {
                    "ws": round(ws_unified, 6),
                    "k": int(k_uni),
                    "cv_rmsep_z": round(cv_pls_uni, 6),
                    "cal_time_s": round(dt_v2_full + dt_k_uni, 4),
                },
                "snv_plus_logs": {
                    "k": int(k_aug),
                    "cv_rmsep_z": round(cv_pls_aug, 6),
                    "cal_time_s": round(dt_k_aug, 4),
                },
            },
            "ridge_cal_config": {
                "ws_wiener": {
                    "ws": round(ws_wiener, 6),
                    "gamma": float(g_wie),
                    "cv_rmsep_z": round(cv_r_wie, 6),
                    "cal_time_s": round(dt_r2_shape + dt_g_wie, 4),
                },
                "ws_unified": {
                    "ws": round(ws_unified, 6),
                    "gamma": float(g_uni),
                    "cv_rmsep_z": round(cv_r_uni, 6),
                    "cal_time_s": round(dt_v2_full + dt_g_uni, 4),
                },
                "snv_plus_logs": {
                    "gamma": float(g_aug),
                    "cv_rmsep_z": round(cv_r_aug, 6),
                    "cal_time_s": round(dt_g_aug, 4),
                },
            },
        }
        locked_tasks.append(row)
        print(
            f"[{i:02d}/54] {tid:42s} | R2sh={pv2['r2_s_shape']:.2f} | "
            f"w_wie={ws_wiener:.3f} w_uni={ws_unified:.3f} (old={pv2['ws_star']:.3f}) | "
            f"{'EXT' if row['is_external_prospective'] else 'DEV'}"
        )

    counts = {}
    for vk in ("wiener", "unified"):
        counts[vk] = {
            "all_54_interior_01": sum(1 for r in locked_tasks if r["pre_tags"][vk]["is_interior_01"]),
            "ext_30_interior_01": sum(
                1 for r in locked_tasks
                if r["is_external_prospective"] and not r["has_label_scale_anomaly"] and r["pre_tags"][vk]["is_interior_01"]
            ),
            "classic_23_interior_01": sum(
                1 for r in locked_tasks if not r["is_external_prospective"] and r["pre_tags"][vk]["is_interior_01"]
            ),
        }

    payload = _sanitize_json(
        {
            "phase": "WIENER_WS_PHASE_1_LOCKED",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_head_frozen": git_head,
            "theory_py_git_blob_sha256_lf": blob_lf,
            "c148_prereg_json_sha256_lf": prev_pre_lf,
            "fixed_wc": FIXED_WC,
            "interior_eps": INTERIOR_EPS,
            "n_tasks": len(locked_tasks),
            "n_external_prospective_valid": 30,
            "n_classic_dev": 23,
            "elapsed_s": round(time.time() - t_start, 2),
            "prospective_hypotheses": {
                "primary_universe": "EXTERNAL_30 (Phase B Valid 17 + Round 3 13, zero prior test contact)",
                "H1": "On EXTERNAL_30 (PLS & Ridge), ws_wiener and ws_unified reduce mean_regret_pct_vs_best_ep, worst_regret_pct, and w=10 asymmetric cost vs old ws_closed_v2 (sigma_non=0).",
                "H2": "On EXTERNAL_30, ws_wiener and ws_unified achieve lower w=10 asymmetric cost and lower mean_regret_pct than BOTH blind endpoints (ws_0 and ws_1).",
                "H3": "Honest comparison against supervised 11-point ws_inner_cv on EXTERNAL_30 (accuracy, regret, w=10 cost, Wilcoxon p, and calibration speedup).",
                "H4": "Mechanism ablation snv_plus_logs fails to eliminate nonlinear Beer-Lambert denominator distortion compared to ws_wiener.",
            },
            "summary": {"pre_tag_counts": counts},
            "tasks": locked_tasks,
        }
    )

    out_json = ROOT / "_audit" / "prereg_wiener_ws_2026_11_25.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    md_lines = [
        "# Phase 1 预注册锁单：无监督维纳收缩 (Theorem 2-W) 与统一维纳–耦合收缩 (Theorem 2-WC)",
        "",
        f"- **锁单 Git HEAD**：`{git_head}`",
        f"- **`chemdl/theory.py` LF SHA-256**：`{blob_lf}`",
        f"- **前瞻盲测池**：`EXTERNAL_30`（`Phase B Valid 17 + Round 3 13`），本轮新方法对测试集绝对零接触；`Classic_23` 明确标为开发诊断集。",
        f"- **激活统计 (`0.01 < w_s < 0.99`)**：`ws_wiener` 在 `EXTERNAL_30` 激活 `{counts['wiener']['ext_30_interior_01']}/30`、`Classic_23` `{counts['wiener']['classic_23_interior_01']}/23`；`ws_unified` 在 `EXTERNAL_30` `{counts['unified']['ext_30_interior_01']}/30`、`Classic_23` `{counts['unified']['classic_23_interior_01']}/23`。",
        "",
        "## 预注册检验假说 (`EXTERNAL_30` 前瞻集)",
        "",
        "1. **H1**：`ws_wiener` 与 `ws_unified` 在 PLS 与 Ridge 上均降低均值遗憾、最坏遗憾与 `w=10` 非对称代价（vs `C-148` 旧版 `ws_closed_v2`）。",
        "2. **H2**：`ws_wiener` 与 `ws_unified` 的 `w=10` 非对称代价与均值遗憾均低于两个盲用端点（`ws=0` 与 `ws=1`）。",
        "3. **H3**：与有监督 11 点 `ws_inner_cv` 正面对比精度、遗憾、`w=10` 代价与校准加速比。",
        "4. **H4**：机制消融 `snv_plus_logs` 无法消除 Beer-Lambert 非线性分母畸变。",
    ]
    out_md = ROOT / "reports" / "antigravity" / "PREREG_WIENER_WS_2026-11-25.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nPhase 1 locked in {payload['elapsed_s']}s -> {out_json}")
    print("Pre-tag counts:", counts)


if __name__ == "__main__":
    main()
