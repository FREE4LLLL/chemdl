"""SNV 预处决策部署工作流、Risk-Coverage 评估、非对称成本与发布异常防护接口。

本模块严格以只读方式调用冻结的 `chemdl.theory.diagnose_snv_decision` 与
`chemdl.uq.BatchDriftMonitor`，不改变底层物理/PRESS 判据本身，旨在将研究期门控
封装为工程可交付的端到端校准与监测接口：
  - `fit_snv_gated_workflow`：支持 4 种弃权兜底策略（`snv` / `raw` / `inner_cv` / `escalate_bilevel`）
  - `diagnose_deploy_sanity`：光谱分布漂移（SPE/MD/Energy/Amp）+ 可选标签量纲/范围偏移告警
  - `compute_risk_coverage_curve`：按 `t_crit` 扫描选择性风险-覆盖率曲线与 AURC
  - `compute_asymmetric_regret`：量化严重遗憾（如 >= +5%）在非对称代价下的真实工程价值
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

from chemdl.diffprep import fit_bilevel_diffprep
from chemdl.theory import diagnose_snv_decision
from chemdl.transfer import fit_pls, fit_prep_fn
from chemdl.uq import BatchDriftMonitor

# Phase B 历史锁单指纹（384ac59 ~ 487a61b）与 P0-4 修订后的现行冻结理论指纹（Layer 2 数值完全不变）
PHASE_B_FROZEN_THEORY_LF_SHA256 = "81a1c2d63d875bb4a5dff0ace090709b1ffeb159b47806fb12bb49cc372e9d14"
FROZEN_THEORY_LF_SHA256 = "66831a6b1ec7b79be048ebd0712dfae367aa9639bc3bcf9f27cded04fdc8bd76"

VALID_ABSTAIN_POLICIES = ("snv", "raw", "inner_cv", "escalate_bilevel")


def _as_2d_spectra(X: Any, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须为二维数组 (N, L)，实际 ndim={arr.ndim}, shape={arr.shape}")
    if arr.shape[1] < 1:
        raise ValueError(f"{name} 波长/通道数必须 >= 1，实际 shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 存在 NaN 或 Inf")
    return arr


def _inner_cv_select_snv_or_raw(
    Xa: np.ndarray,
    yn: np.ndarray,
    groups_cal: np.ndarray | None,
    seed: int,
) -> dict:
    """严格仅在校准集内部比较 Raw_PLS 与 SNV_PLS，用于 ABSTAIN 时的 inner_cv 策略。"""
    n = len(Xa)
    use_group = False
    if groups_cal is not None:
        grp = np.asarray(groups_cal)
        uniq = np.unique(grp)
        eff = min(5, len(uniq))
        if eff >= 2:
            cand_splits = list(GroupKFold(n_splits=eff).split(Xa, yn, grp))
            if min(len(itr) for itr, _ in cand_splits) >= 3:
                splits = cand_splits
                use_group = True
    if not use_group:
        n_splits = min(5, max(2, n // 2))
        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(Xa))

    sq_raw, sq_snv = [], []
    for fi, (itr, iva) in enumerate(splits):
        kmax = min(10, max(2, len(itr) // 2))
        inner = min(4, max(2, len(itr) // 4))
        pr_r, _ = fit_pls(Xa[itr], yn[itr], kmax=kmax, inner=inner, per_col=True, seed=seed + fi)
        fn_s = fit_prep_fn("snv", Xa[itr])
        pr_s, _ = fit_pls(fn_s(Xa[itr]), yn[itr], kmax=kmax, inner=inner, per_col=True, seed=seed + fi)
        sq_raw.append((pr_r(Xa[iva]) - yn[iva]) ** 2)
        sq_snv.append((pr_s(fn_s(Xa[iva])) - yn[iva]) ** 2)

    cv_raw = float(np.sqrt(np.mean(np.concatenate(sq_raw, axis=0))))
    cv_snv = float(np.sqrt(np.mean(np.concatenate(sq_snv, axis=0))))
    delta_pct = float((cv_snv / max(cv_raw, 1e-12) - 1.0) * 100.0)
    pick = "SNV" if cv_snv < cv_raw else "RAW"
    return {
        "cv_mode": "group_cv" if use_group else "kfold_cv",
        "cv_rmsep_raw_z": round(cv_raw, 4),
        "cv_rmsep_snv_z": round(cv_snv, 4),
        "delta_cv_pct": round(delta_pct, 2),
        "inner_cv_pick": pick,
    }


def fit_snv_gated_workflow(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    groups_cal: np.ndarray | None = None,
    abstain_policy: str = "snv",
    t_crit: float = 2.0,
    min_effect_pct: float = 1.0,
    kmax: int = 10,
    inner: int = 4,
    seed: int = 0,
    bilevel_steps: int = 40,
    bilevel_max_opt_n: int = 800,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """端到端拟合带弃权工作流的 SNV 决策校准管线。

    路由规则：
      - ``IMPROVE`` -> ``SNV_PLS``
      - ``WORSEN``  -> ``Raw_PLS``
      - ``ABSTAIN`` -> 按 ``abstain_policy`` 路由：
          ``"snv"``（保守偏 SNV）、``"raw"``（保守保留 Raw）、
          ``"inner_cv"``（折内 CV 裁决）或 ``"escalate_bilevel"``（升级可微双层预处理）。
    """
    t0 = time.time()
    if abstain_policy not in VALID_ABSTAIN_POLICIES:
        raise ValueError(f"abstain_policy 必须属于 {VALID_ABSTAIN_POLICIES}，实际 {abstain_policy!r}")

    Xa = _as_2d_spectra(X_cal, "X_cal")
    if Xa.shape[0] < 4:
        raise ValueError(f"X_cal 至少需要 4 条样本，实际 {Xa.shape[0]}")

    y_raw = np.asarray(y_cal, dtype=np.float64)
    if not np.all(np.isfinite(y_raw)):
        raise ValueError("y_cal 存在 NaN 或 Inf")
    was_1d = y_raw.ndim == 1
    ya = y_raw[:, None] if was_1d else y_raw
    if ya.ndim != 2 or ya.shape[0] != Xa.shape[0]:
        raise ValueError(f"y_cal 样本数与 X_cal 不匹配：{ya.shape} vs {Xa.shape}")

    gate_report = diagnose_snv_decision(
        Xa,
        ya,
        groups_tr=groups_cal,
        t_crit=float(t_crit),
        min_effect_pct=float(min_effect_pct),
        compute_physics=True,
    )
    decision = str(gate_report["decision"])

    ymu = ya.mean(axis=0, keepdims=True)
    ysd = np.clip(ya.std(axis=0, keepdims=True), 1e-9, None)
    yn = (ya - ymu) / ysd

    eff_kmax = min(int(kmax), max(2, len(Xa) // 2))
    eff_inner = min(int(inner), max(2, len(Xa) // 4))
    inner_cv_info: dict[str, Any] | None = None

    if decision == "IMPROVE":
        route_choice = "SNV"
    elif decision == "WORSEN":
        route_choice = "RAW"
    else:
        if abstain_policy == "snv":
            route_choice = "SNV"
        elif abstain_policy == "raw":
            route_choice = "RAW"
        elif abstain_policy == "inner_cv":
            inner_cv_info = _inner_cv_select_snv_or_raw(Xa, yn, groups_cal, seed=seed)
            route_choice = str(inner_cv_info["inner_cv_pick"])
        else:
            route_choice = "BILEVEL"

    n_feat = Xa.shape[1]
    n_tgt = ya.shape[1]

    if route_choice == "SNV":
        fn_snv = fit_prep_fn("snv", Xa)
        pr_snv, ks = fit_pls(fn_snv(Xa), yn, kmax=eff_kmax, inner=eff_inner, per_col=True, seed=seed)
        selected_pipeline = "SNV_PLS"
        model_details: dict[str, Any] = {"pls_ncomp": ks}

        def _pred_norm(X_in: np.ndarray) -> np.ndarray:
            return np.asarray(pr_snv(fn_snv(X_in)), dtype=np.float64).reshape(len(X_in), n_tgt)

    elif route_choice == "RAW":
        pr_raw, ks = fit_pls(Xa, yn, kmax=eff_kmax, inner=eff_inner, per_col=True, seed=seed)
        selected_pipeline = "Raw_PLS"
        model_details = {"pls_ncomp": ks}

        def _pred_norm(X_in: np.ndarray) -> np.ndarray:
            return np.asarray(pr_raw(X_in), dtype=np.float64).reshape(len(X_in), n_tgt)

    else:
        pr_bi, bi_state = fit_bilevel_diffprep(
            Xa,
            yn,
            steps=int(bilevel_steps),
            max_opt_n=int(bilevel_max_opt_n),
            seed=int(seed),
        )
        selected_pipeline = "BilevelDiffPrep"
        model_details = {"bilevel_state": bi_state if isinstance(bi_state, dict) else {"summary": str(bi_state)}}

        def _pred_norm(X_in: np.ndarray) -> np.ndarray:
            return np.asarray(pr_bi(X_in), dtype=np.float64).reshape(len(X_in), n_tgt)

    def predict_fn(X_new: np.ndarray) -> np.ndarray:
        Xn = _as_2d_spectra(X_new, "X_new")
        if Xn.shape[1] != n_feat:
            raise ValueError(f"X_new 光谱通道数 {Xn.shape[1]} 与校准集 {n_feat} 不匹配")
        if Xn.shape[0] == 0:
            return np.empty((0,) if was_1d else (0, n_tgt), dtype=np.float64)
        pred_n = _pred_norm(Xn)
        pred_orig = pred_n * ysd + ymu
        return pred_orig[:, 0] if was_1d else pred_orig

    drift_monitor = BatchDriftMonitor().fit(Xa)
    cal_y_stats = {
        "n_cal": int(Xa.shape[0]),
        "n_targets": int(n_tgt),
        "mean": [round(float(v), 6) for v in ya.mean(axis=0)],
        "std": [round(float(v), 6) for v in ya.std(axis=0)],
        "min": [round(float(v), 6) for v in ya.min(axis=0)],
        "max": [round(float(v), 6) for v in ya.max(axis=0)],
    }
    report = {
        "abstain_policy": abstain_policy,
        "selected_pipeline": selected_pipeline,
        "gate_report": gate_report,
        "inner_cv_info": inner_cv_info,
        "cal_y_stats": cal_y_stats,
        "model_details": model_details,
        "drift_monitor": drift_monitor,
        "fit_elapsed_sec": round(time.time() - t0, 4),
    }
    return predict_fn, report


def diagnose_deploy_sanity(
    X_cal: np.ndarray,
    X_new: np.ndarray,
    *,
    y_cal: np.ndarray | None = None,
    y_new: np.ndarray | None = None,
    label_z_shift_warn: float = 3.0,
) -> dict:
    """部署安全体检：检测光谱域分布漂移（无监督）与可选标签量纲/单位系统偏移。"""
    Xa = _as_2d_spectra(X_cal, "X_cal")
    Xn = _as_2d_spectra(X_new, "X_new")
    if Xa.shape[1] != Xn.shape[1]:
        raise ValueError(f"光谱宽度不匹配：X_cal={Xa.shape[1]} vs X_new={Xn.shape[1]}")
    if Xa.shape[0] < 3 or Xn.shape[0] < 1:
        raise ValueError(f"样本数不足：X_cal={Xa.shape[0]}, X_new={Xn.shape[0]}")

    drift = BatchDriftMonitor().fit(Xa).assess_batch(Xn)
    spectrum_drift_status = str(drift["status"])

    label_check: dict[str, Any] = {
        "checked": False,
        "label_scale_shift": False,
        "mean_shift_z": None,
        "std_ratio": None,
        "range_disjoint": False,
    }
    if y_cal is not None and y_new is not None:
        yc = np.asarray(y_cal, dtype=np.float64).ravel()
        yn = np.asarray(y_new, dtype=np.float64).ravel()
        if yc.size == 0 or yn.size == 0:
            raise ValueError("y_cal 与 y_new 不能为空")
        if not (np.all(np.isfinite(yc)) and np.all(np.isfinite(yn))):
            raise ValueError("y_cal 或 y_new 存在 NaN/Inf")
        yc_sd = max(float(np.std(yc)), 1e-9)
        mean_shift_z = float(abs(float(np.mean(yn)) - float(np.mean(yc))) / yc_sd)
        std_ratio = float(float(np.std(yn)) / yc_sd)
        range_disjoint = bool(float(np.max(yn)) < float(np.min(yc)) or float(np.min(yn)) > float(np.max(yc)))
        label_scale_shift = bool(mean_shift_z >= float(label_z_shift_warn) or range_disjoint)
        label_check = {
            "checked": True,
            "label_scale_shift": label_scale_shift,
            "mean_shift_z": round(mean_shift_z, 3),
            "std_ratio": round(std_ratio, 3),
            "range_disjoint": range_disjoint,
            "y_cal_mean": round(float(np.mean(yc)), 4),
            "y_new_mean": round(float(np.mean(yn)), 4),
        }

    overall_status = "LABEL_SHIFT_ALERT" if label_check["label_scale_shift"] else spectrum_drift_status
    return {
        "status": overall_status,
        "spectrum_drift_status": spectrum_drift_status,
        "drift_score": float(drift["drift_score"]) if math.isfinite(float(drift["drift_score"])) else None,
        "raw_ratio": float(drift["raw_ratio"]),
        "spe_ratio": float(drift["spe_ratio"]),
        "mahalanobis_ratio": float(drift["mahalanobis_ratio"]),
        **label_check,
    }


def compute_asymmetric_regret(
    errs: np.ndarray,
    oracles: np.ndarray,
    *,
    severe_threshold_pct: float = 5.0,
    penalty_weights: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0),
) -> dict:
    """计算相对 Oracle 的均值/最坏遗憾以及不同严重遗憾惩罚倍数的非对称部署成本。"""
    ea = np.asarray(errs, dtype=np.float64).ravel()
    oa = np.asarray(oracles, dtype=np.float64).ravel()
    if ea.shape != oa.shape or ea.size == 0:
        raise ValueError(f"errs 与 oracles 形状不匹配或为空：{ea.shape} vs {oa.shape}")
    ratios = ea / np.clip(oa, 1e-12, None)
    regrets = (ratios - 1.0) * 100.0
    excess = np.maximum(0.0, regrets - float(severe_threshold_pct))

    weighted_cost = {}
    for w in penalty_weights:
        wf = float(w)
        cost = regrets + (wf - 1.0) * excess
        weighted_cost[str(w)] = round(float(np.mean(cost)), 3)

    return {
        "n": int(ea.size),
        "severe_threshold_pct": float(severe_threshold_pct),
        "mean_regret_pct": round(float(np.mean(regrets)), 3),
        "median_regret_pct": round(float(np.median(regrets)), 3),
        "max_regret_pct": round(float(np.max(regrets)), 3),
        "n_severe_ge_thr": int(np.sum(regrets >= float(severe_threshold_pct))),
        "geom_ratio_vs_oracle": round(float(np.exp(np.mean(np.log(np.clip(ratios, 1e-12, None))))), 4),
        "weighted_cost_by_w": weighted_cost,
    }


def compute_risk_coverage_curve(
    t_values: np.ndarray,
    delta_loo_pcts: np.ndarray,
    emp_pcts: np.ndarray,
    err_raw: np.ndarray,
    err_snv: np.ndarray,
    *,
    min_effect_pct: float = 1.0,
    t_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0),
) -> dict:
    """跨置信阈值 `t_crit` 计算选择性风险-覆盖率曲线（Risk-Coverage）及端到端遗憾演化。"""
    tv = np.asarray(t_values, dtype=np.float64).ravel()
    dv = np.asarray(delta_loo_pcts, dtype=np.float64).ravel()
    ev = np.asarray(emp_pcts, dtype=np.float64).ravel()
    er = np.asarray(err_raw, dtype=np.float64).ravel()
    es = np.asarray(err_snv, dtype=np.float64).ravel()
    n = int(tv.size)
    if not (dv.size == ev.size == er.size == es.size == n) or n == 0:
        raise ValueError("所有输入数组必须等长且非空")

    oracles = np.minimum(er, es)
    y_true_all = (ev < 0.0).astype(int)
    points = []

    for tc in t_grid:
        t_thr = float(tc)
        calls = []
        for t_i, d_i in zip(tv, dv):
            if t_i <= -t_thr and d_i <= -float(min_effect_pct):
                calls.append("IMPROVE")
            elif t_i >= t_thr and d_i >= float(min_effect_pct):
                calls.append("WORSEN")
            else:
                calls.append("ABSTAIN")

        dec_mask = np.array([c != "ABSTAIN" for c in calls], dtype=bool)
        n_dec = int(np.sum(dec_mask))
        cov = round(n_dec / n, 4)

        if n_dec > 0:
            yt = y_true_all[dec_mask]
            yp = np.array([1 if calls[idx] == "IMPROVE" else 0 for idx in np.flatnonzero(dec_mask)], dtype=int)
            acc_val = float(np.mean(yt == yp))
            sign_acc: float | None = round(acc_val, 4)
            sel_risk = round(1.0 - acc_val, 4)
            tp = int(np.sum((yt == 1) & (yp == 1)))
            tn = int(np.sum((yt == 0) & (yp == 0)))
            fp = int(np.sum((yt == 0) & (yp == 1)))
            fn = int(np.sum((yt == 1) & (yp == 0)))
            denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc: float | None = round(((tp * tn) - (fp * fn)) / math.sqrt(denom), 4) if denom > 0.0 else 0.0
        else:
            sign_acc = None
            sel_risk = 0.0
            mcc = None

        errs_fb_snv = np.array([er[i] if calls[i] == "WORSEN" else es[i] for i in range(n)], dtype=np.float64)
        errs_fb_raw = np.array([es[i] if calls[i] == "IMPROVE" else er[i] for i in range(n)], dtype=np.float64)
        reg_snv = (errs_fb_snv / np.clip(oracles, 1e-12, None) - 1.0) * 100.0
        reg_raw = (errs_fb_raw / np.clip(oracles, 1e-12, None) - 1.0) * 100.0

        points.append(
            {
                "t_crit": t_thr,
                "coverage": cov,
                "n_decided": n_dec,
                "sign_acc": sign_acc,
                "selective_risk": sel_risk,
                "mcc": mcc,
                "fallback_snv_mean_regret_pct": round(float(np.mean(reg_snv)), 3),
                "fallback_snv_max_regret_pct": round(float(np.max(reg_snv)), 3),
                "fallback_snv_n_severe": int(np.sum(reg_snv >= 5.0)),
                "fallback_raw_mean_regret_pct": round(float(np.mean(reg_raw)), 3),
                "fallback_raw_max_regret_pct": round(float(np.max(reg_raw)), 3),
                "fallback_raw_n_severe": int(np.sum(reg_raw >= 5.0)),
            }
        )

    # 梯形积分计算 AURC（以 (0.0, 0.0) 为锚点，仅保留有覆盖点并按 coverage 升序）
    rc_pairs = sorted([(p["coverage"], p["selective_risk"]) for p in points if p["n_decided"] > 0], key=lambda x: (x[0], x[1]))
    # 去重同一 coverage 取最小风险对应代表点，确保严格非降横坐标
    by_cov: dict[float, list[float]] = {}
    for c_val, r_val in rc_pairs:
        by_cov.setdefault(c_val, []).append(r_val)
    curve_xy = [(0.0, 0.0)] + [(c_val, float(np.mean(r_list))) for c_val, r_list in sorted(by_cov.items())]
    aurc = 0.0
    for idx in range(len(curve_xy) - 1):
        c1, r1 = curve_xy[idx]
        c2, r2 = curve_xy[idx + 1]
        aurc += (c2 - c1) * 0.5 * (r1 + r2)

    return {
        "n_total": n,
        "min_effect_pct": float(min_effect_pct),
        "aurc_selective_error": round(float(aurc), 4),
        "points": points,
    }
