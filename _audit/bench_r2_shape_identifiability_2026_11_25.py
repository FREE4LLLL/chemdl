"""验证 1 - R²_shape 物理可识别性与物理边界的受控及真实配对光谱实验（C-150）。

包含四部分：
  - Part A1: 固定真实化学成分的 5 套 NIR 光谱上注入纯标量乘性散射 (a_i = exp(eps_i))，
             检验 1 - R²_shape 是否单调追踪独立真值散射方差占比 f_true 与基线修正理论值 f_adj；
  - Part A2: 对抗性负对照——注入波长耦合倾斜散射 a_i(lambda) = exp(eps_i * (1 + gamma * lambda_tilde))，
             检验当漂移同时改变谱形与幅值时 1 - R²_shape 的系统性低估坍塌边界；
  - Part A3: 在 Corn_m5 (Oil / Moisture) 与 SWRI_Diesel (BP50) 独立留出集上沿标量散射扫描检验
             ws_wiener = 1 - R²_shape 是否追踪测试集最优 w_s* 并降低相对两端点的遗憾；
  - Part B:  8 套未修饰真实配对/重复扫描 NIR 光谱（零人工注入）的组内散射占比 f_within 与
             单谱 1 - R²_shape 对照及差异机制诊断（颗粒间形态散射 vs 波长耦合仪器漂移）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._round3_unseen_loaders import read_xlsx_sheet_stdlib
from _audit._ws_closed_value_loaders import WS_GRID, _stratified_70_30_split
from chemdl.cordoba import load_cordoba_all
from chemdl.corn import load_corn
from chemdl.diesel import load_diesel
from chemdl.tablets import load_tablets
from chemdl.theory import estimate_physical_params_v2
from chemdl.transfer import dssnv_apply, rmsep

SEEDS = [0, 1, 2, 3, 4]
SIGMA_LEVELS_A1 = [0.00, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.25]
TILT_GAMMAS_A2 = [0.00, 0.25, 0.50, 1.00]
SIGMA_TILT_A2 = 0.10
SIGMA_LEVELS_A3 = [0.00, 0.02, 0.05, 0.10, 0.20]
SPLIT_SEED_A3 = 20261125


def raw_and_capped_r2_shape(
    X: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 3,
    max_lv: int = 6,
) -> dict:
    """与 chemdl/theory.py::estimate_physical_params_v2 完全一致地计算 R²_shape，
    同时返回未截断 (r2_raw) 与 0.98 截断 (r2_cap98 / ws_wiener) 结果。
    """
    Xa = np.asarray(X, dtype=np.float64)
    n, L = Xa.shape
    mu_row = Xa.mean(axis=1)
    sd_row = np.clip(Xa.std(axis=1), 1e-12, None)
    X_snv = (Xa - mu_row[:, None]) / sd_row[:, None]

    if groups is not None:
        g_arr = np.asarray(groups)
        n_uniq = int(len(np.unique(g_arr)))
        eff_k = min(n_splits, n_uniq)
        splits = list(GroupKFold(n_splits=eff_k).split(X_snv, groups=g_arr))
    else:
        eff_k = min(n_splits, max(2, n // 5))
        splits = list(KFold(eff_k, shuffle=True, random_state=0).split(X_snv))

    max_lv_shape = min(max_lv, max(1, n // 5), L)
    tsd = float(sd_row.std())
    if tsd < 1e-12:
        return {
            "r2_raw": 0.0,
            "r2_cap98": 0.0,
            "one_minus_r2_raw": 1.0,
            "ws_wiener": 1.0,
            "best_lv": 0,
        }

    zn = (sd_row - sd_row.mean()) / tsd
    best_mse = 1.0
    best_lv = 0
    for lv in range(1, max_lv_shape + 1):
        mses = []
        for itr, iva in splits:
            n_comp = min(lv, len(itr) - 1)
            if n_comp < 1:
                continue
            m = PLSRegression(n_components=n_comp, scale=False).fit(X_snv[itr], zn[itr])
            p = m.predict(X_snv[iva]).ravel()
            mses.append(float(np.mean((p - zn[iva]) ** 2)))
        if mses:
            mse = float(np.mean(mses))
            if mse < best_mse:
                best_mse = mse
                best_lv = int(lv)

    r2_raw = float(max(1.0 - best_mse, 0.0))
    r2_cap98 = float(min(r2_raw, 0.98))
    return {
        "r2_raw": r2_raw,
        "r2_cap98": r2_cap98,
        "one_minus_r2_raw": float(1.0 - r2_raw),
        "ws_wiener": float(1.0 - r2_cap98),
        "best_lv": best_lv,
    }


def _fit_pls_ws(X_cal: np.ndarray, y_cal: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, ws: float) -> float:
    """固定 wc=1.0 与固定 5 折内层 CV PLS (kmax=12) 评估测试集 RMSEP_z。"""
    Xc = np.asarray(X_cal, dtype=np.float64)
    yc = np.asarray(y_cal, dtype=np.float64).ravel()
    Xt = np.asarray(X_test, dtype=np.float64)
    yt = np.asarray(y_test, dtype=np.float64).ravel()

    n_cal, L = Xc.shape
    k_in = min(5, max(3, n_cal // 12))
    splits = list(KFold(n_splits=k_in, shuffle=True, random_state=42).split(Xc))
    kmax = min(12, max(2, min(len(itr) for itr, _ in splits) - 2), L)

    y_sd = float(np.std(yc))
    if y_sd < 1e-12:
        y_sd = 1.0

    best_k, best_cv = 1, float("inf")
    for k in range(1, kmax + 1):
        preds = np.empty(n_cal, dtype=np.float64)
        for itr, iva in splits:
            s_bar = float(np.mean(Xc[itr].std(axis=1)))
            Ztr = dssnv_apply(Xc[itr], wc=1.0, ws=ws, s_bar=s_bar)
            Zva = dssnv_apply(Xc[iva], wc=1.0, ws=ws, s_bar=s_bar)
            m = PLSRegression(n_components=min(k, Ztr.shape[0] - 1, Ztr.shape[1]), scale=False).fit(Ztr, yc[itr])
            preds[iva] = m.predict(Zva).ravel()
        cv_err = rmsep(yc, preds)
        if cv_err < best_cv:
            best_cv = cv_err
            best_k = k

    s_bar_full = float(np.mean(Xc.std(axis=1)))
    Z_cal = dssnv_apply(Xc, wc=1.0, ws=ws, s_bar=s_bar_full)
    Z_tst = dssnv_apply(Xt, wc=1.0, ws=ws, s_bar=s_bar_full)
    m_full = PLSRegression(n_components=min(best_k, Z_cal.shape[0] - 1, Z_cal.shape[1]), scale=False).fit(Z_cal, yc)
    y_hat = m_full.predict(Z_tst).ravel()
    return float(rmsep(yt, y_hat) / y_sd)


def _one_way_anova_within_fraction(values: np.ndarray, group_ids: np.ndarray) -> dict:
    """非平衡单因素随机效应 ANOVA 估计组内物理变异占比 sigma_w^2 / (sigma_w^2 + sigma_b^2)。"""
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(group_ids)
    uniq, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    keep_g = counts >= 2
    if not np.any(keep_g):
        return {"f_within_anova": 0.0, "n_groups": 0, "n_obs": 0}
    mask = keep_g[inv]
    v = v[mask]
    g = g[mask]
    uniq, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    k = len(uniq)
    N = len(v)
    grand_mean = float(np.mean(v))
    g_means = np.bincount(inv, weights=v) / counts
    ss_between = float(np.sum(counts * (g_means - grand_mean) ** 2))
    ss_within = float(np.sum((v - g_means[inv]) ** 2))
    ms_between = ss_between / max(k - 1, 1)
    ms_within = ss_within / max(N - k, 1)
    n0 = float((N - np.sum(counts ** 2) / N) / max(k - 1, 1))
    sigma2_w = ms_within
    sigma2_b = max(0.0, (ms_between - ms_within) / max(n0, 1e-12))
    frac = float(sigma2_w / max(sigma2_w + sigma2_b, 1e-15))
    return {
        "f_within_anova": frac,
        "sigma2_within": float(sigma2_w),
        "sigma2_between": float(sigma2_b),
        "n_groups": int(k),
        "n_obs": int(N),
        "mean_reps": float(N / k),
    }


def _load_base_datasets() -> list[tuple[str, np.ndarray]]:
    corn = load_corn()
    dsl = load_diesel()
    ok_d = np.isfinite(dsl["X"]).all(axis=1)
    tab = load_tablets(trim_saturated=True)
    cord = load_cordoba_all()["wheat"]
    ok_w = np.isfinite(cord["X"]).all(axis=1) & (cord["years"] == 1998)
    wa1 = sio.loadmat(ROOT / "data_ext" / "ct_bench" / "wheatA1_20048.mat", squeeze_me=True, struct_as_record=False)
    return [
        ("Corn_m5", np.asarray(corn["m5"], dtype=np.float64)),
        ("SWRI_Diesel", np.asarray(dsl["X"][ok_d], dtype=np.float64)),
        ("Tablets_X1_cal", np.asarray(tab["X1"][tab["part"] == 0], dtype=np.float64)),
        ("Cordoba_Wheat_1998", np.asarray(cord["X"][ok_w], dtype=np.float64)),
        ("Wheat_A1_cal", np.asarray(wa1["X"], dtype=np.float64)),
    ]


def _summary_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    pr = float(pearsonr(p, t)[0]) if np.std(p) > 1e-12 and np.std(t) > 1e-12 else 0.0
    sr = float(spearmanr(p, t)[0]) if np.std(p) > 1e-12 and np.std(t) > 1e-12 else 0.0
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    bias = float(np.mean(p - t))
    slope, intercept = (np.polyfit(t, p, 1) if np.std(t) > 1e-12 else (0.0, float(np.mean(p))))
    return {
        "pearson_r": pr,
        "spearman_rho": sr,
        "mae": mae,
        "rmse": rmse,
        "mean_bias": bias,
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
    }


def run_part_a1_and_a2(base_datasets: list[tuple[str, np.ndarray]]) -> tuple[dict, dict]:
    a1_per_dataset: dict[str, dict] = {}
    a2_per_dataset: dict[str, dict] = {}
    pooled_raw, pooled_wiener, pooled_f_true, pooled_f_adj = [], [], [], []
    pooled_f_pop_log, pooled_f_adj_pop_log = [], []
    pooled_f_pop_lin, pooled_f_adj_pop_lin = [], []
    a2_pooled_by_gamma: dict[str, list[dict]] = {f"{g:.2f}": [] for g in TILT_GAMMAS_A2}
    a2_dom4_by_gamma: dict[str, list[dict]] = {f"{g:.2f}": [] for g in TILT_GAMMAS_A2}

    for name, X0 in base_datasets:
        n, L = X0.shape
        sd0 = np.clip(X0.std(axis=1), 1e-12, None)
        log_sd0 = np.log(sd0)
        var_log_s0 = float(np.var(log_sd0))
        var_s0 = float(np.var(sd0))
        e_s0_sq = float(np.mean(sd0 ** 2))
        base_res = raw_and_capped_r2_shape(X0)
        r2_0_raw = base_res["r2_raw"]
        u0_raw = base_res["one_minus_r2_raw"]

        # Part A1: Pure scalar multiplicative scatter sweep
        levels_a1 = []
        ds_raw, ds_wie, ds_ft, ds_fa = [], [], [], []
        ds_fpl, ds_fal, ds_fplin, ds_falin = [], [], [], []
        for sig in SIGMA_LEVELS_A1:
            seed_rows = []
            for seed in SEEDS:
                if sig == 0.0:
                    eps = np.zeros(n, dtype=np.float64)
                    X_obs = X0.copy()
                else:
                    rng = np.random.default_rng(10000 + seed * 100 + int(round(sig * 1000)))
                    eps = rng.normal(0.0, sig, size=n)
                    eps = eps - np.mean(eps)
                    X_obs = np.exp(eps)[:, None] * X0

                sd_obs = np.clip(X_obs.std(axis=1), 1e-12, None)
                log_sd_obs = np.log(sd_obs)
                var_log_obs = float(np.var(log_sd_obs))
                var_eps = float(np.var(eps))
                f_true_log = 0.0 if sig == 0.0 else float(var_eps / max(var_log_obs, 1e-15))
                f_adj_raw = float(1.0 - (1.0 - f_true_log) * r2_0_raw)

                f_pop_log = 0.0 if sig == 0.0 else float(var_eps / max(var_log_s0 + var_eps, 1e-15))
                f_adj_pop_log = float(1.0 - (1.0 - f_pop_log) * r2_0_raw)
                a = np.exp(eps)
                mu_a = float(np.mean(a))
                var_a = float(np.var(a))
                num_lin = var_a * e_s0_sq
                den_lin = num_lin + (mu_a ** 2) * var_s0
                f_pop_lin = 0.0 if sig == 0.0 else float(num_lin / max(den_lin, 1e-15))
                f_adj_pop_lin = float(1.0 - (1.0 - f_pop_lin) * r2_0_raw)

                est = raw_and_capped_r2_shape(X_obs)
                seed_rows.append(
                    {
                        "f_true_log": f_true_log,
                        "f_adj_raw": f_adj_raw,
                        "f_pop_log": f_pop_log,
                        "f_adj_pop_log": f_adj_pop_log,
                        "f_pop_lin": f_pop_lin,
                        "f_adj_pop_lin": f_adj_pop_lin,
                        "one_minus_r2_raw": est["one_minus_r2_raw"],
                        "ws_wiener": est["ws_wiener"],
                        "best_lv": est["best_lv"],
                    }
                )
            m_ft = float(np.mean([r["f_true_log"] for r in seed_rows]))
            m_fa = float(np.mean([r["f_adj_raw"] for r in seed_rows]))
            m_fpl = float(np.mean([r["f_pop_log"] for r in seed_rows]))
            m_fal = float(np.mean([r["f_adj_pop_log"] for r in seed_rows]))
            m_fplin = float(np.mean([r["f_pop_lin"] for r in seed_rows]))
            m_falin = float(np.mean([r["f_adj_pop_lin"] for r in seed_rows]))
            m_raw = float(np.mean([r["one_minus_r2_raw"] for r in seed_rows]))
            sd_raw = float(np.std([r["one_minus_r2_raw"] for r in seed_rows], ddof=1)) if sig > 0 else 0.0
            m_wie = float(np.mean([r["ws_wiener"] for r in seed_rows]))
            levels_a1.append(
                {
                    "sigma_inj": float(sig),
                    "f_true_log_mean": m_ft,
                    "f_true_log_max_seed": float(max(r["f_true_log"] for r in seed_rows)),
                    "f_adj_raw_mean": m_fa,
                    "f_pop_log_mean": m_fpl,
                    "f_adj_pop_log_mean": m_fal,
                    "f_pop_lin_mean": m_fplin,
                    "f_adj_pop_lin_mean": m_falin,
                    "one_minus_r2_raw_mean": m_raw,
                    "one_minus_r2_raw_sd": sd_raw,
                    "ws_wiener_mean": m_wie,
                }
            )
            ds_ft.append(m_ft)
            ds_fa.append(m_fa)
            ds_fpl.append(m_fpl)
            ds_fal.append(m_fal)
            ds_fplin.append(m_fplin)
            ds_falin.append(m_falin)
            ds_raw.append(m_raw)
            ds_wie.append(m_wie)
            pooled_f_true.append(m_ft)
            pooled_f_adj.append(m_fa)
            pooled_f_pop_log.append(m_fpl)
            pooled_f_adj_pop_log.append(m_fal)
            pooled_f_pop_lin.append(m_fplin)
            pooled_f_adj_pop_lin.append(m_falin)
            pooled_raw.append(m_raw)
            pooled_wiener.append(m_wie)

        a1_per_dataset[name] = {
            "n": int(n),
            "L": int(L),
            "baseline_r2_raw": float(r2_0_raw),
            "baseline_one_minus_r2_raw": float(u0_raw),
            "levels": levels_a1,
            "metrics_raw_vs_f_true": _summary_metrics(ds_raw, ds_ft),
            "metrics_raw_vs_f_adj": _summary_metrics(ds_raw, ds_fa),
            "metrics_wiener_vs_f_adj": _summary_metrics(ds_wie, ds_fa),
            "metrics_raw_vs_f_pop_lin": _summary_metrics(ds_raw, ds_fplin),
            "metrics_raw_vs_f_adj_pop_lin": _summary_metrics(ds_raw, ds_falin),
        }

        # Part A2: Adversarial wavelength-dependent tilt negative control
        lam_tilde = np.linspace(-1.0, 1.0, L, dtype=np.float64)
        levels_a2 = []
        for gamma in TILT_GAMMAS_A2:
            seed_rows = []
            for seed in SEEDS:
                rng = np.random.default_rng(20000 + seed * 100 + int(round(gamma * 100)))
                eps = rng.normal(0.0, SIGMA_TILT_A2, size=n)
                eps = eps - np.mean(eps)
                gain = np.exp(eps[:, None] * (1.0 + gamma * lam_tilde[None, :]))
                X_tilt = gain * X0
                sd_tilt = np.clip(X_tilt.std(axis=1), 1e-12, None)
                delta_log = np.log(sd_tilt) - log_sd0
                var_dlog = float(np.var(delta_log))
                f_true_log = float(var_dlog / max(np.var(np.log(sd_tilt)), 1e-15))
                f_adj_raw = float(1.0 - (1.0 - f_true_log) * r2_0_raw)

                f_pop_log = float(var_dlog / max(var_log_s0 + var_dlog, 1e-15))
                f_adj_pop_log = float(1.0 - (1.0 - f_pop_log) * r2_0_raw)
                a_eff = sd_tilt / sd0
                mu_a = float(np.mean(a_eff))
                var_a = float(np.var(a_eff))
                num_lin = var_a * e_s0_sq
                den_lin = num_lin + (mu_a ** 2) * var_s0
                f_pop_lin = float(num_lin / max(den_lin, 1e-15))
                f_adj_pop_lin = float(1.0 - (1.0 - f_pop_lin) * r2_0_raw)

                est = raw_and_capped_r2_shape(X_tilt)
                seed_rows.append(
                    {
                        "f_true_log": f_true_log,
                        "f_adj_raw": f_adj_raw,
                        "f_pop_log": f_pop_log,
                        "f_adj_pop_log": f_adj_pop_log,
                        "f_pop_lin": f_pop_lin,
                        "f_adj_pop_lin": f_adj_pop_lin,
                        "one_minus_r2_raw": est["one_minus_r2_raw"],
                        "ws_wiener": est["ws_wiener"],
                    }
                )
            row_g = {
                "gamma_tilt": float(gamma),
                "f_true_log_mean": float(np.mean([r["f_true_log"] for r in seed_rows])),
                "f_true_log_max_seed": float(max(r["f_true_log"] for r in seed_rows)),
                "f_adj_raw_mean": float(np.mean([r["f_adj_raw"] for r in seed_rows])),
                "f_pop_log_mean": float(np.mean([r["f_pop_log"] for r in seed_rows])),
                "f_adj_pop_log_mean": float(np.mean([r["f_adj_pop_log"] for r in seed_rows])),
                "f_pop_lin_mean": float(np.mean([r["f_pop_lin"] for r in seed_rows])),
                "f_adj_pop_lin_mean": float(np.mean([r["f_adj_pop_lin"] for r in seed_rows])),
                "one_minus_r2_raw_mean": float(np.mean([r["one_minus_r2_raw"] for r in seed_rows])),
                "one_minus_r2_raw_sd": float(np.std([r["one_minus_r2_raw"] for r in seed_rows], ddof=1)),
                "ws_wiener_mean": float(np.mean([r["ws_wiener"] for r in seed_rows])),
                "bias_vs_f_true": float(
                    np.mean([r["one_minus_r2_raw"] - r["f_true_log"] for r in seed_rows])
                ),
                "bias_vs_f_adj": float(
                    np.mean([r["one_minus_r2_raw"] - r["f_adj_raw"] for r in seed_rows])
                ),
                "bias_vs_f_pop_log": float(
                    np.mean([r["one_minus_r2_raw"] - r["f_pop_log"] for r in seed_rows])
                ),
                "bias_vs_f_pop_lin": float(
                    np.mean([r["one_minus_r2_raw"] - r["f_pop_lin"] for r in seed_rows])
                ),
                "bias_vs_f_adj_pop_lin": float(
                    np.mean([r["one_minus_r2_raw"] - r["f_adj_pop_lin"] for r in seed_rows])
                ),
            }
            levels_a2.append(row_g)
            a2_pooled_by_gamma[f"{gamma:.2f}"].append(row_g)
            if name != "Wheat_A1_cal":
                a2_dom4_by_gamma[f"{gamma:.2f}"].append(row_g)
        a2_per_dataset[name] = {"n": int(n), "L": int(L), "levels": levels_a2}

    def _agg_a2(by_gamma: dict[str, list[dict]]) -> dict[str, dict]:
        return {
            gk: {
                "f_true_log_mean": float(np.mean([r["f_true_log_mean"] for r in rows])),
                "f_adj_raw_mean": float(np.mean([r["f_adj_raw_mean"] for r in rows])),
                "f_pop_log_mean": float(np.mean([r["f_pop_log_mean"] for r in rows])),
                "f_adj_pop_log_mean": float(np.mean([r["f_adj_pop_log_mean"] for r in rows])),
                "f_pop_lin_mean": float(np.mean([r["f_pop_lin_mean"] for r in rows])),
                "f_adj_pop_lin_mean": float(np.mean([r["f_adj_pop_lin_mean"] for r in rows])),
                "one_minus_r2_raw_mean": float(np.mean([r["one_minus_r2_raw_mean"] for r in rows])),
                "ws_wiener_mean": float(np.mean([r["ws_wiener_mean"] for r in rows])),
                "bias_vs_f_true": float(np.mean([r["bias_vs_f_true"] for r in rows])),
                "bias_vs_f_adj": float(np.mean([r["bias_vs_f_adj"] for r in rows])),
                "bias_vs_f_pop_log": float(np.mean([r["bias_vs_f_pop_log"] for r in rows])),
                "bias_vs_f_pop_lin": float(np.mean([r["bias_vs_f_pop_lin"] for r in rows])),
                "bias_vs_f_adj_pop_lin": float(np.mean([r["bias_vs_f_adj_pop_lin"] for r in rows])),
            }
            for gk, rows in by_gamma.items()
        }

    part_a1 = {
        "n_datasets": len(base_datasets),
        "n_seeds": len(SEEDS),
        "sigma_levels": SIGMA_LEVELS_A1,
        "pooled_n_points": len(pooled_raw),
        "pooled_metrics_raw_vs_f_true": _summary_metrics(pooled_raw, pooled_f_true),
        "pooled_metrics_raw_vs_f_adj": _summary_metrics(pooled_raw, pooled_f_adj),
        "pooled_metrics_wiener_vs_f_adj": _summary_metrics(pooled_wiener, pooled_f_adj),
        "pooled_metrics_raw_vs_f_pop_log": _summary_metrics(pooled_raw, pooled_f_pop_log),
        "pooled_metrics_raw_vs_f_adj_pop_log": _summary_metrics(pooled_raw, pooled_f_adj_pop_log),
        "pooled_metrics_raw_vs_f_pop_lin": _summary_metrics(pooled_raw, pooled_f_pop_lin),
        "pooled_metrics_raw_vs_f_adj_pop_lin": _summary_metrics(pooled_raw, pooled_f_adj_pop_lin),
        "pooled_metrics_wiener_vs_f_adj_pop_lin": _summary_metrics(pooled_wiener, pooled_f_adj_pop_lin),
        "datasets": a1_per_dataset,
    }
    part_a2 = {
        "sigma_inj": SIGMA_TILT_A2,
        "gamma_levels": TILT_GAMMAS_A2,
        "pooled_by_gamma": _agg_a2(a2_pooled_by_gamma),
        "pooled_by_gamma_4_dominated": _agg_a2(a2_dom4_by_gamma),
        "datasets": a2_per_dataset,
    }
    return part_a1, part_a2


def run_part_a3_downstream() -> dict:
    corn = load_corn()
    dsl = load_diesel()
    tasks_a3 = []
    for col_name in ("Oil", "Moisture"):
        j = list(corn["cols"]).index(col_name)
        tr, te = _stratified_70_30_split(corn["m5"], corn["y"][:, j], seed=SPLIT_SEED_A3 + j)
        tasks_a3.append((f"Corn_m5__{col_name}", corn["m5"][tr], corn["y"][tr, j], corn["m5"][te], corn["y"][te, j]))

    j_bp = list(dsl["cols"]).index("BP50")
    ok_d = np.isfinite(dsl["X"]).all(axis=1) & np.isfinite(dsl["y"][:, j_bp])
    Xd, yd = dsl["X"][ok_d], dsl["y"][ok_d, j_bp]
    tr_d, te_d = _stratified_70_30_split(Xd, yd, seed=SPLIT_SEED_A3 + j_bp)
    tasks_a3.append(("SWRI_Diesel__BP50", Xd[tr_d], yd[tr_d], Xd[te_d], yd[te_d]))

    out_tasks = {}
    all_ws_wie, all_ws_ora = [], []
    all_reg_wie, all_reg_ws0, all_reg_ws1 = [], [], []

    for tid, Xc0, yc, Xt0, yt in tasks_a3:
        level_rows = []
        for sig in SIGMA_LEVELS_A3:
            seed_rows = []
            for seed in SEEDS:
                if sig == 0.0:
                    Xc, Xt = Xc0.copy(), Xt0.copy()
                else:
                    rng_c = np.random.default_rng(30000 + seed * 100 + int(round(sig * 1000)))
                    rng_t = np.random.default_rng(40000 + seed * 100 + int(round(sig * 1000)))
                    ec = rng_c.normal(0.0, sig, size=len(Xc0))
                    et = rng_t.normal(0.0, sig, size=len(Xt0))
                    ec -= ec.mean()
                    et -= et.mean()
                    Xc = np.exp(ec)[:, None] * Xc0
                    Xt = np.exp(et)[:, None] * Xt0

                est = raw_and_capped_r2_shape(Xc)
                ws_wie = est["ws_wiener"]
                err_0 = _fit_pls_ws(Xc, yc, Xt, yt, 0.0)
                err_1 = _fit_pls_ws(Xc, yc, Xt, yt, 1.0)
                err_wie = _fit_pls_ws(Xc, yc, Xt, yt, ws_wie)

                grid_errs = [err_0] + [_fit_pls_ws(Xc, yc, Xt, yt, w) for w in WS_GRID[1:-1]] + [err_1]
                ora_idx = int(np.argmin(grid_errs))
                ws_ora = float(WS_GRID[ora_idx])
                best_ep = min(err_0, err_1)
                seed_rows.append(
                    {
                        "ws_wiener": ws_wie,
                        "ws_oracle": ws_ora,
                        "err_ws0": err_0,
                        "err_ws1": err_1,
                        "err_wiener": err_wie,
                        "err_oracle": float(grid_errs[ora_idx]),
                        "regret_wie_pct": float((err_wie / best_ep - 1.0) * 100.0),
                        "regret_ws0_pct": float((err_0 / best_ep - 1.0) * 100.0),
                        "regret_ws1_pct": float((err_1 / best_ep - 1.0) * 100.0),
                    }
                )
            row = {
                "sigma_inj": float(sig),
                "ws_wiener_mean": float(np.mean([r["ws_wiener"] for r in seed_rows])),
                "ws_oracle_mean": float(np.mean([r["ws_oracle"] for r in seed_rows])),
                "err_ws0_mean": float(np.mean([r["err_ws0"] for r in seed_rows])),
                "err_ws1_mean": float(np.mean([r["err_ws1"] for r in seed_rows])),
                "err_wiener_mean": float(np.mean([r["err_wiener"] for r in seed_rows])),
                "err_oracle_mean": float(np.mean([r["err_oracle"] for r in seed_rows])),
                "regret_wie_pct_mean": float(np.mean([r["regret_wie_pct"] for r in seed_rows])),
                "regret_ws0_pct_mean": float(np.mean([r["regret_ws0_pct"] for r in seed_rows])),
                "regret_ws1_pct_mean": float(np.mean([r["regret_ws1_pct"] for r in seed_rows])),
            }
            level_rows.append(row)
            all_ws_wie.append(row["ws_wiener_mean"])
            all_ws_ora.append(row["ws_oracle_mean"])
            all_reg_wie.extend([r["regret_wie_pct"] for r in seed_rows])
            all_reg_ws0.extend([r["regret_ws0_pct"] for r in seed_rows])
            all_reg_ws1.extend([r["regret_ws1_pct"] for r in seed_rows])
        out_tasks[tid] = level_rows

    return {
        "tasks": out_tasks,
        "sigma_levels": SIGMA_LEVELS_A3,
        "summary": {
            "pearson_ws_wiener_vs_oracle": float(pearsonr(all_ws_wie, all_ws_ora)[0]),
            "spearman_ws_wiener_vs_oracle": float(spearmanr(all_ws_wie, all_ws_ora)[0]),
            "mean_regret_pct_ws_wiener": float(np.mean(all_reg_wie)),
            "mean_regret_pct_ws0": float(np.mean(all_reg_ws0)),
            "mean_regret_pct_ws1": float(np.mean(all_reg_ws1)),
            "worst_regret_pct_ws_wiener": float(np.max(all_reg_wie)),
            "worst_regret_pct_ws0": float(np.max(all_reg_ws0)),
            "worst_regret_pct_ws1": float(np.max(all_reg_ws1)),
        },
    }


def _paired_diagnostics(name: str, category: str, X1: np.ndarray, X2: np.ndarray) -> dict:
    """无修饰真实配对光谱（X1, X2）的组内物理散射占比与单谱 1-R²_shape 对照诊断。"""
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    n, L = X1.shape
    s1 = np.clip(X1.std(axis=1), 1e-12, None)
    s2 = np.clip(X2.std(axis=1), 1e-12, None)
    ls1, ls2 = np.log(s1), np.log(s2)

    f_paired_diff = float((0.5 * np.var(ls1 - ls2, ddof=1)) / max(np.var(np.concatenate([ls1, ls2]), ddof=1), 1e-15))
    ls1_c = ls1 - float(np.mean(ls1))
    ls2_c = ls2 - float(np.mean(ls2))
    f_paired_centered_denom = float(
        (0.5 * np.var(ls1 - ls2, ddof=1)) / max(np.var(np.concatenate([ls1_c, ls2_c]), ddof=1), 1e-15)
    )
    gids = np.concatenate([np.arange(n), np.arange(n)])
    anova = _one_way_anova_within_fraction(np.concatenate([ls1, ls2]), gids)
    anova_centered = _one_way_anova_within_fraction(np.concatenate([ls1_c, ls2_c]), gids)

    e1 = raw_and_capped_r2_shape(X1)
    e2 = raw_and_capped_r2_shape(X2)
    X_pool = np.vstack([X1, X2])
    ep = raw_and_capped_r2_shape(X_pool, groups=gids)

    # 诊断跨仪器尺度差是否与光谱形态耦合（若耦合高，则 1-R²_shape 会将跨仪器尺度差异吸收进 R²_shape）
    snv1 = (X1 - X1.mean(1, keepdims=True)) / s1[:, None]
    snv2 = (X2 - X2.mean(1, keepdims=True)) / s2[:, None]
    dshape = snv2 - snv1
    dscale = ls2 - ls1
    d_sd = float(np.std(dscale))
    if d_sd < 1e-12:
        delta_shape_r2 = 0.0
    else:
        zn_d = (dscale - dscale.mean()) / d_sd
        eff_k = min(3, max(2, n // 5))
        max_lv = min(6, max(1, n // 5), L)
        splits = list(KFold(eff_k, shuffle=True, random_state=0).split(dshape))
        best_mse = 1.0
        for lv in range(1, max_lv + 1):
            mses = []
            for itr, iva in splits:
                m = PLSRegression(n_components=min(lv, len(itr) - 1), scale=False).fit(dshape[itr], zn_d[itr])
                p = m.predict(dshape[iva]).ravel()
                mses.append(float(np.mean((p - zn_d[iva]) ** 2)))
            best_mse = min(best_mse, float(np.mean(mses)))
        delta_shape_r2 = float(max(1.0 - best_mse, 0.0))

    return {
        "dataset": name,
        "kind": "paired_instruments",
        "category": category,
        "n_samples": int(n),
        "L": int(L),
        "f_within_log_std": anova["f_within_anova"],
        "f_within_paired_halfvar": f_paired_diff,
        "f_within_paired_centered_denom": f_paired_centered_denom,
        "f_within_anova_centered": anova_centered["f_within_anova"],
        "between_instrument_mean_offset_log": float(np.mean(ls2) - np.mean(ls1)),
        "one_minus_r2_raw_x1": e1["one_minus_r2_raw"],
        "one_minus_r2_raw_x2": e2["one_minus_r2_raw"],
        "ws_wiener_x1": e1["ws_wiener"],
        "one_minus_r2_raw_pooled_groupcv": ep["one_minus_r2_raw"],
        "delta_scale_from_delta_shape_cv_r2": delta_shape_r2,
        "gap_x1_minus_f_within": float(e1["one_minus_r2_raw"] - anova["f_within_anova"]),
        "gap_x1_minus_f_paired_centered": float(e1["one_minus_r2_raw"] - f_paired_centered_denom),
    }


def run_part_b_real_paired() -> list[dict]:
    from _audit.selfcheck_external_2026_11_19 import load_idrc2016_wheat

    out = []
    tab = load_tablets(trim_saturated=True)
    out.append(_paired_diagnostics("Tablets_X1_vs_X2", "solid_tablet", tab["X1"], tab["X2"]))

    corn = load_corn()
    out.append(_paired_diagnostics("Corn_m5_vs_mp5", "solid_kernel", corn["m5"], corn["mp5"]))
    out.append(_paired_diagnostics("Corn_m5_vs_mp6", "solid_kernel", corn["m5"], corn["mp6"]))

    idrc = load_idrc2016_wheat()
    out.append(_paired_diagnostics("Wheat_IDRC16_A_vs_B", "solid_kernel", idrc["X_A"], idrc["X_B"]))
    out.append(_paired_diagnostics("Wheat_IDRC16_A_vs_C", "solid_kernel", idrc["X_A"], idrc["X_C"]))

    sdop = sio.loadmat(ROOT / "data_ext" / "ct_bench" / "sdop_nir_data.mat")
    out.append(_paired_diagnostics("SDOP_spec1_vs_spec2", "liquid_small_n", sdop["spec1"], sdop["spec2"]))

    # OSSL Soil cross-device scans (filtered to exact repeated-sample population before ANOVA & shape)
    dfn = pd.read_csv(ROOT / "data_ext" / "ct_bench" / "neospectra_nir_v1.2.csv.gz")
    spec_cols = [c for c in dfn.columns if c.startswith("scan_nir.")]
    X_os = dfn[spec_cols].to_numpy(dtype=np.float64)
    g_os = dfn["id.sample_local_c"].astype(str).to_numpy()
    dev_os = dfn["scan.nir.model.name_utf8_txt"].astype(str).to_numpy()
    ok_os = np.isfinite(X_os).all(axis=1)
    X_os, g_os, dev_os = X_os[ok_os], g_os[ok_os], dev_os[ok_os]

    _, inv_os, cnt_os = np.unique(g_os, return_inverse=True, return_counts=True)
    rep_mask = cnt_os[inv_os] >= 2
    n_singleton_excluded = int(np.sum(~rep_mask))
    X_rep, g_rep, dev_rep = X_os[rep_mask], g_os[rep_mask], dev_os[rep_mask]

    ls_rep = np.log(np.clip(X_rep.std(axis=1), 1e-12, None))
    anova_os = _one_way_anova_within_fraction(ls_rep, g_rep)
    ls_dev_c = ls_rep.copy()
    for d in np.unique(dev_rep):
        m_d = dev_rep == d
        ls_dev_c[m_d] -= float(np.mean(ls_dev_c[m_d]))
    anova_os_devc = _one_way_anova_within_fraction(ls_dev_c, g_rep)

    gd_key = np.array([f"{g}__{d}" for g, d in zip(g_os, dev_os)])
    anova_os_samedev = _one_way_anova_within_fraction(np.log(np.clip(X_os.std(axis=1), 1e-12, None)), gd_key)
    dpc = pd.DataFrame({"g": g_rep, "d": dev_rep}).groupby("g")["d"].nunique()

    _, first_idx = np.unique(g_rep, return_index=True)
    e_os1 = raw_and_capped_r2_shape(X_rep[np.sort(first_idx)])
    e_os_grp = raw_and_capped_r2_shape(X_rep, groups=g_rep)
    out.append(
        {
            "dataset": "OSSL_Soil_Repacked",
            "display_label": "OSSL_Soil_CrossDevice",
            "kind": "cross_device_replicates",
            "category": "solid_powder",
            "n_samples": anova_os["n_groups"],
            "n_scans": anova_os["n_obs"],
            "n_singleton_excluded": n_singleton_excluded,
            "n_devices": int(len(np.unique(dev_rep))),
            "devices_per_sample_range": [int(dpc.min()), int(dpc.max())],
            "L": int(X_rep.shape[1]),
            "f_within_log_std": anova_os["f_within_anova"],
            "f_within_device_centered": anova_os_devc["f_within_anova"],
            "same_device_subset": {
                "n_pairs": anova_os_samedev["n_groups"],
                "n_scans": anova_os_samedev["n_obs"],
                "f_within_anova": anova_os_samedev["f_within_anova"],
            },
            "one_minus_r2_raw_x1": e_os1["one_minus_r2_raw"],
            "ws_wiener_x1": e_os1["ws_wiener"],
            "one_minus_r2_raw_pooled_groupcv": e_os_grp["one_minus_r2_raw"],
            "gap_x1_minus_f_within": float(e_os1["one_minus_r2_raw"] - anova_os["f_within_anova"]),
            "gap_x1_minus_f_device_centered": float(e_os1["one_minus_r2_raw"] - anova_os_devc["f_within_anova"]),
        }
    )

    # PeaForage 2019_GN plot replicates
    s0 = read_xlsx_sheet_stdlib(ROOT / "data_ext" / "round3_unseen" / "Z7809305_pea_forage" / "2019_GN_Hyperspectral&GroundReferenceData.xlsx", 0)
    X_pea_list, g_pea_list = [], []
    for r in s0[2:]:
        if len(r) < 10 or str(r[0]).strip() == "":
            continue
        loc = str(r[0]).strip()
        plot = int(float(r[1]))
        X_pea_list.append([float(x) for x in r[3:]])
        g_pea_list.append(f"{loc}_{plot}")
    X_pea = np.asarray(X_pea_list, dtype=np.float64)
    g_pea = np.asarray(g_pea_list)
    anova_pea = _one_way_anova_within_fraction(np.log(np.clip(X_pea.std(axis=1), 1e-12, None)), g_pea)
    _, first_pea = np.unique(g_pea, return_index=True)
    e_pea1 = raw_and_capped_r2_shape(X_pea[np.sort(first_pea)])
    e_pea_grp = raw_and_capped_r2_shape(X_pea, groups=g_pea)
    out.append(
        {
            "dataset": "PeaForage_2019GN_Rep",
            "kind": "field_plot_replicates",
            "category": "solid_forage",
            "n_samples": anova_pea["n_groups"],
            "n_scans": anova_pea["n_obs"],
            "L": int(X_pea.shape[1]),
            "f_within_log_std": anova_pea["f_within_anova"],
            "one_minus_r2_raw_x1": e_pea1["one_minus_r2_raw"],
            "ws_wiener_x1": e_pea1["ws_wiener"],
            "one_minus_r2_raw_pooled_groupcv": e_pea_grp["one_minus_r2_raw"],
            "gap_x1_minus_f_within": float(e_pea1["one_minus_r2_raw"] - anova_pea["f_within_anova"]),
        }
    )
    return out


def main() -> None:
    t0 = time.perf_counter()
    base_datasets = _load_base_datasets()

    # 精确对齐校验：raw_and_capped_r2_shape 必须与 chemdl/theory.py 生产函数完全一致
    X_check = base_datasets[0][1]
    ref_v2 = estimate_physical_params_v2(X_check, np.arange(len(X_check), dtype=np.float64))
    chk = raw_and_capped_r2_shape(X_check)
    assert abs(round(chk["r2_cap98"], 4) - ref_v2["r2_s_shape"]) < 1e-12, (
        f"r2_s_shape mismatch: {round(chk['r2_cap98'], 4)} vs {ref_v2['r2_s_shape']}"
    )

    part_a1, part_a2 = run_part_a1_and_a2(base_datasets)
    part_a3 = run_part_a3_downstream()
    part_b = run_part_b_real_paired()

    elapsed = time.perf_counter() - t0
    res = {
        "experiment_id": "R2_SHAPE_IDENTIFIABILITY_2026_11_25",
        "elapsed_sec": round(elapsed, 2),
        "n_distinct_sources_part_b": 6,
        "metric_definitions": {
            "f_true_log": "Finite-sample perturbation ratio Var(delta_log)/Var(log s_obs); can exceed 1.0 when Cov(log s_0, delta_log) < 0.",
            "f_pop_log": "Bounded [0,1] population log-scale scatter share Var(delta_log)/(Var(log s_0) + Var(delta_log)).",
            "f_pop_lin": "Bounded [0,1] same-scale linear-s population scatter share Var(a)*E[s_0^2] / (Var(a)*E[s_0^2] + E[a]^2*Var(s_0)).",
            "f_adj_pop_lin": "Baseline-adjusted linear-s population scatter share 1 - (1 - f_pop_lin)*R2_shape_0.",
        },
        "part_a1_scalar_sweep": part_a1,
        "part_a2_tilt_negative_control": part_a2,
        "part_a3_downstream_alignment": part_a3,
        "part_b_real_paired_replicates": part_b,
    }
    out_p = ROOT / "_audit" / "res_r2_shape_identifiability_2026_11_25.json"
    out_p.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    p1_ft = part_a1["pooled_metrics_raw_vs_f_true"]
    p1_fa = part_a1["pooled_metrics_raw_vs_f_adj"]
    print(f"Saved {out_p} in {elapsed:.1f}s")
    print(
        f"Part A1 pooled vs f_true: r={p1_ft['pearson_r']:.4f}, rho={p1_ft['spearman_rho']:.4f}, MAE={p1_ft['mae']:.4f}"
    )
    print(
        f"Part A1 pooled vs f_adj : r={p1_fa['pearson_r']:.4f}, rho={p1_fa['spearman_rho']:.4f}, MAE={p1_fa['mae']:.4f}"
    )
    print("Part A2 tilt pooled:", {k: round(v["one_minus_r2_raw_mean"], 4) for k, v in part_a2["pooled_by_gamma"].items()})
    print("Part A3 summary:", {k: round(v, 4) for k, v in part_a3["summary"].items()})
    for b in part_b:
        print(
            f"Part B {b['dataset']:22s} f_within={b['f_within_log_std']:.4f} "
            f"1-R2_raw={b['one_minus_r2_raw_x1']:.4f} pooled={b['one_minus_r2_raw_pooled_groupcv']:.4f} "
            f"gap={b['gap_x1_minus_f_within']:+.4f}"
        )


if __name__ == "__main__":
    main()
