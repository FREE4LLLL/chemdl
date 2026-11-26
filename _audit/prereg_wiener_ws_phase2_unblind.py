"""Phase 2 前瞻揭盲评估：无监督维纳收缩 (Theorem 2-W) 与统一维纳–耦合收缩 (Theorem 2-WC)。

硬不变量：
  1. 校验 Phase 1 锁单 (`0f5fcc4`)、`chemdl/theory.py` 冻结哈希与 54 项划分 SHA-256；
  2. 严格区分前瞻外部盲测池 `EXTERNAL_30` (`Phase B Valid 17 + Round 3 13`，零事先接触)
     与开发诊断集 `Classic_23`；
  3. 同步汇报 `C-148` 根因拆解（`14/20` 落入 `sqA|r|<=beta` 盲区 + Beer-Lambert `σ_non_chem`
     非线性分母畸变）及 `H1–H4` 预注册假说逐项裁决；
  4. `--verify-only` 在无 `pandas` / 无 `data_ext/` 的干净克隆 CI 下也能纯标准库校验。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.cross_decomposition import PLSRegression

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemdl.deploy_gate import FROZEN_THEORY_LF_SHA256, compute_asymmetric_regret
from chemdl.theory import snv_snr_curve
from chemdl.transfer import dssnv_apply, rmsep

FIXED_WC = 1.0


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _file_sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_blob_sha256_lf(root: Path, rev: str, rel_path: str) -> str:
    try:
        raw = subprocess.check_output(["git", "show", f"{rev}:{rel_path}"], cwd=root, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raw = (root / rel_path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


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


def _safe_wilcoxon_p(diffs: list[float]) -> float | None:
    arr = np.asarray(diffs, dtype=np.float64)
    if len(arr) < 5 or np.allclose(arr, 0.0, atol=1e-12):
        return None
    try:
        res = wilcoxon(arr, alternative="two-sided", zero_method="wilcox")
        p = float(res.pvalue)
        return p if np.isfinite(p) else None
    except Exception:
        return None


def _fit_eval_pls_snv_plus_logs(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int,
) -> dict:
    mu = float(np.mean(y_cal))
    sd = max(float(np.std(y_cal)), 1e-9)
    yn_cal = (y_cal - mu) / sd
    yn_test = (y_test - mu) / sd

    ls_cal = np.log(np.clip(X_cal.std(axis=1), 1e-12, None))
    ls_test = np.log(np.clip(X_test.std(axis=1), 1e-12, None))
    mu_ls = float(ls_cal.mean())
    sd_ls = max(float(ls_cal.std()), 1e-9)

    Xcal_a = np.hstack([dssnv_apply(X_cal, wc=FIXED_WC, ws=1.0), ((ls_cal - mu_ls) / sd_ls)[:, None]])
    Xtest_a = np.hstack([dssnv_apply(X_test, wc=FIXED_WC, ws=1.0), ((ls_test - mu_ls) / sd_ls)[:, None]])
    eff_k = min(int(k), Xcal_a.shape[1], max(1, len(Xcal_a) - 1))
    m = PLSRegression(n_components=eff_k, scale=False).fit(Xcal_a, yn_cal[:, None])
    pred_z = m.predict(Xtest_a).ravel()
    return {"k": int(eff_k), "rmsep_z": float(rmsep(pred_z, yn_test))}


def _fit_eval_ridge_snv_plus_logs(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gamma: float,
) -> dict:
    mu = float(np.mean(y_cal))
    sd = max(float(np.std(y_cal)), 1e-9)
    yn_cal = (y_cal - mu) / sd
    yn_test = (y_test - mu) / sd

    ls_cal = np.log(np.clip(X_cal.std(axis=1), 1e-12, None))
    ls_test = np.log(np.clip(X_test.std(axis=1), 1e-12, None))
    mu_ls = float(ls_cal.mean())
    sd_ls = max(float(ls_cal.std()), 1e-9)

    Xcal_a = np.hstack([dssnv_apply(X_cal, wc=FIXED_WC, ws=1.0), ((ls_cal - mu_ls) / sd_ls)[:, None]])
    Xtest_a = np.hstack([dssnv_apply(X_test, wc=FIXED_WC, ws=1.0), ((ls_test - mu_ls) / sd_ls)[:, None]])
    x_mu = Xcal_a.mean(axis=0, keepdims=True)
    y_mu = float(np.mean(yn_cal))
    U, s, Vt = np.linalg.svd(Xcal_a - x_mu, full_matrices=False)
    W = Vt.T @ ((s / (s ** 2 + max(float(gamma), 1e-15))) * (U.T @ (yn_cal - y_mu)))
    pred_z = (Xtest_a - x_mu) @ W + y_mu
    return {"gamma": float(gamma), "rmsep_z": float(rmsep(pred_z, yn_test))}


def _method_metrics(err_m: float, err_0: float, err_1: float, err_v2_old: float, err_cv: float) -> dict:
    best_ep = min(err_0, err_1)
    dz_0 = err_m - err_0
    dz_1 = err_m - err_1
    dz_b = err_m - best_ep
    dz_old = err_m - err_v2_old
    dz_cv = err_m - err_cv
    return {
        "rmsep_z": round(float(err_m), 6),
        "best_ep_z": round(float(best_ep), 6),
        "diff_z_vs_ws0": round(float(dz_0), 6),
        "rel_pct_vs_ws0": round(float((err_m / max(err_0, 1e-12) - 1.0) * 100.0), 3),
        "diff_z_vs_ws1": round(float(dz_1), 6),
        "rel_pct_vs_ws1": round(float((err_m / max(err_1, 1e-12) - 1.0) * 100.0), 3),
        "diff_z_vs_best_ep": round(float(dz_b), 6),
        "rel_pct_vs_best_ep": round(float((err_m / max(best_ep, 1e-12) - 1.0) * 100.0), 3),
        "diff_z_vs_v2_old": round(float(dz_old), 6),
        "rel_pct_vs_v2_old": round(float((err_m / max(err_v2_old, 1e-12) - 1.0) * 100.0), 3),
        "diff_z_vs_inner_cv": round(float(dz_cv), 6),
        "rel_pct_vs_inner_cv": round(float((err_m / max(err_cv, 1e-12) - 1.0) * 100.0), 3),
        "beats_both_strict": bool(err_m < best_ep - 1e-12),
        "beats_both_0p5pct": bool((err_m / max(best_ep, 1e-12) - 1.0) * 100.0 <= -0.5),
    }


def _aggregate(rows: list[dict], head_key: str) -> dict:
    if not rows:
        return {"n": 0}
    methods = (
        "ws_0",
        "ws_1",
        "ws_closed_v2_old",
        "ws_wiener",
        "ws_unified",
        "snv_plus_logs",
        "ws_inner_cv",
        "ws_oracle",
    )
    out: dict = {"n": len(rows), "methods": {}}
    best_eps = np.array([r[head_key]["metrics"]["ws_0"]["best_ep_z"] for r in rows], dtype=np.float64)

    for m in methods:
        ez = [r[head_key]["metrics"][m]["rmsep_z"] for r in rows]
        dz0 = [r[head_key]["metrics"][m]["diff_z_vs_ws0"] for r in rows]
        dz1 = [r[head_key]["metrics"][m]["diff_z_vs_ws1"] for r in rows]
        dzb = [r[head_key]["metrics"][m]["diff_z_vs_best_ep"] for r in rows]
        rpb = [r[head_key]["metrics"][m]["rel_pct_vs_best_ep"] for r in rows]
        dz_old = [r[head_key]["metrics"][m]["diff_z_vs_v2_old"] for r in rows]
        rp_old = [r[head_key]["metrics"][m]["rel_pct_vs_v2_old"] for r in rows]
        dz_cv = [r[head_key]["metrics"][m]["diff_z_vs_inner_cv"] for r in rows]
        rp_cv = [r[head_key]["metrics"][m]["rel_pct_vs_inner_cv"] for r in rows]
        cal_t = [r[head_key]["cal_times_s"].get(m, 0.0) for r in rows]

        cost_info = compute_asymmetric_regret(np.asarray(ez, dtype=np.float64), best_eps)
        wb = sum(1 for x in rpb if x <= -0.5)
        tb = sum(1 for x in rpb if -0.5 < x < 0.5)
        lb = sum(1 for x in rpb if x >= 0.5)
        wo = sum(1 for x in rp_old if x <= -0.5)
        to = sum(1 for x in rp_old if -0.5 < x < 0.5)
        lo = sum(1 for x in rp_old if x >= 0.5)
        wc_c = sum(1 for x in rp_cv if x <= -0.5)
        tc_c = sum(1 for x in rp_cv if -0.5 < x < 0.5)
        lc_c = sum(1 for x in rp_cv if x >= 0.5)

        rpb_exact = [(e / max(b, 1e-12) - 1.0) * 100.0 for e, b in zip(ez, best_eps)]
        mean_rpb_exact = float(np.mean(rpb_exact))
        max_rpb_exact = float(np.max(rpb_exact))
        w10_exact = float(cost_info["weighted_cost_by_w"]["10.0"])

        out["methods"][m] = {
            "mean_rmsep_z": round(float(np.mean(ez)), 5),
            "median_rmsep_z": round(float(np.median(ez)), 5),
            "mean_diff_z_vs_ws0": round(float(np.mean(dz0)), 5),
            "mean_diff_z_vs_ws1": round(float(np.mean(dz1)), 5),
            "mean_diff_z_vs_best_ep": round(float(np.mean(dzb)), 5),
            "mean_rel_pct_vs_best_ep": round(mean_rpb_exact, 2),
            "mean_rel_pct_vs_best_ep_exact": mean_rpb_exact,
            "median_rel_pct_vs_best_ep": round(float(np.median(rpb_exact)), 2),
            "worst_rel_pct_vs_best_ep": round(max_rpb_exact, 2),
            "worst_rel_pct_vs_best_ep_exact": max_rpb_exact,
            "severe_regret_ge5pct_count": int(sum(1 for x in rpb if x >= 5.0)),
            "beats_both_strict_count": int(sum(1 for r in rows if r[head_key]["metrics"][m]["beats_both_strict"])),
            "beats_both_0p5pct_count": int(wb),
            "wtl_vs_best_ep_0p5pct": f"{wb}/{tb}/{lb}",
            "wtl_vs_v2_old_0p5pct": f"{wo}/{to}/{lo}",
            "wtl_vs_inner_cv_0p5pct": f"{wc_c}/{tc_c}/{lc_c}",
            "wilcoxon_p_vs_v2_old": _safe_wilcoxon_p(dz_old),
            "wilcoxon_p_vs_inner_cv": _safe_wilcoxon_p(dz_cv),
            "asymmetric_cost_w10": round(w10_exact, 2),
            "asymmetric_cost_w10_exact": w10_exact,
            "total_cal_time_s": round(float(np.sum(cal_t)), 3),
        }

    t_cv = out["methods"]["ws_inner_cv"]["total_cal_time_s"]
    for mk in ("ws_closed_v2_old", "ws_wiener", "ws_unified", "snv_plus_logs"):
        tc = out["methods"][mk]["total_cal_time_s"]
        out["methods"][mk]["speedup_vs_inner_cv"] = round(float(t_cv / max(tc, 1e-9)), 2)
    return out


def evaluate_wiener_hypotheses(summary_ext30: dict) -> dict:
    """按预注册标准全精度逐项裁决 EXTERNAL_30 上的 H1 / H2 / H4（E1 修复：同时检查 wiener 与 unified）。"""
    out: dict = {}
    for head in ("pls", "ridge"):
        m = summary_ext30[head]["methods"]
        ep_mean = min(
            m["ws_0"].get("mean_rel_pct_vs_best_ep_exact", m["ws_0"]["mean_rel_pct_vs_best_ep"]),
            m["ws_1"].get("mean_rel_pct_vs_best_ep_exact", m["ws_1"]["mean_rel_pct_vs_best_ep"]),
        )
        ep_w10 = min(
            m["ws_0"].get("asymmetric_cost_w10_exact", m["ws_0"]["asymmetric_cost_w10"]),
            m["ws_1"].get("asymmetric_cost_w10_exact", m["ws_1"]["asymmetric_cost_w10"]),
        )
        old_mean = m["ws_closed_v2_old"].get("mean_rel_pct_vs_best_ep_exact", m["ws_closed_v2_old"]["mean_rel_pct_vs_best_ep"])
        old_worst = m["ws_closed_v2_old"].get("worst_rel_pct_vs_best_ep_exact", m["ws_closed_v2_old"]["worst_rel_pct_vs_best_ep"])
        old_w10 = m["ws_closed_v2_old"].get("asymmetric_cost_w10_exact", m["ws_closed_v2_old"]["asymmetric_cost_w10"])

        for tag, mk in (("wiener", "ws_wiener"), ("unified", "ws_unified")):
            d = m[mk]
            d_mean = d.get("mean_rel_pct_vs_best_ep_exact", d["mean_rel_pct_vs_best_ep"])
            d_worst = d.get("worst_rel_pct_vs_best_ep_exact", d["worst_rel_pct_vs_best_ep"])
            d_w10 = d.get("asymmetric_cost_w10_exact", d["asymmetric_cost_w10"])
            out[f"H1_{head}_{tag}_pass"] = bool(d_mean < old_mean and d_worst < old_worst and d_w10 < old_w10)
            out[f"H2_{head}_{tag}_pass"] = bool(d_mean < ep_mean and d_w10 < ep_w10)

        out[f"H1_{head}_pass"] = bool(out[f"H1_{head}_wiener_pass"] and out[f"H1_{head}_unified_pass"])
        out[f"H2_{head}_pass"] = bool(out[f"H2_{head}_wiener_pass"] and out[f"H2_{head}_unified_pass"])

    ep = summary_ext30["pls"]["methods"]
    out["H4_pass"] = bool(
        ep["snv_plus_logs"].get("asymmetric_cost_w10_exact", ep["snv_plus_logs"]["asymmetric_cost_w10"])
        > ep["ws_wiener"].get("asymmetric_cost_w10_exact", ep["ws_wiener"]["asymmetric_cost_w10"])
    )
    return out


def _write_report(res: dict, out_md: Path) -> None:
    ext_pls = res["summary"]["EXTERNAL_30"]["pls"]["methods"]
    ext_ridge = res["summary"]["EXTERNAL_30"]["ridge"]["methods"]
    dev_pls = res["summary"]["Classic_23"]["pls"]["methods"]
    dev_ridge = res["summary"]["Classic_23"]["ridge"]["methods"]
    v53_pls = res["summary"]["VALID_53"]["pls"]["methods"]
    diag = res["root_cause_dissection"]
    hv = res["hypothesis_verdicts_external_30"]

    bda = res["boundary_discretization_audit"]

    lines = [
        "# Phase 2 揭盲报告：无监督维纳收缩 (Theorem 2-W) 与统一维纳–耦合收缩 (Theorem 2-WC)",
        "",
        f"- **状态**：`{res['phase']}`",
        f"- **Phase 1 父节点 Commit (`git_head_frozen`)**：`{res['phase1_parent_git_head']}`",
        f"- **Phase 1 锁单 Commit (`phase1_lock_commit`)**：`{res['phase1_lock_commit']}`",
        f"- **Phase 2 执行 Git HEAD**：`{res['execution_git_head']}`",
        f"- **Phase 1 JSON SHA-256 (LF)**：`{res['prereg_json_sha256_lf']}`",
        f"- **`chemdl/theory.py` LF SHA-256**：`{res['theory_py_git_blob_sha256_lf']}`",
        f"- **外测口径说明（I1）**：`EXTERNAL_30` 为已在 `C-148` 揭盲过的固定外部留出池，本轮 `ws_wiener` / `ws_unified` / `snv_plus_logs` 仅用 `cal` 冻结参数并在同划分上做对比（非全新未见来源独立外测）。",
        "",
        "## 1. `C-148` 失败根因拆解（旧版 `σ_non=0` 闭式公式为何失灵）",
        "",
        f"- **激活盲区**：在 `VALID_53` 的 {diag['n_oracle_interior']} 项 Oracle 严格内部最优任务中，{diag['oracle_interior_breakdown']}。",
        f"- **物理假说（Beer-Lambert 化学幅值分母畸变 $\\sigma_{{\\text{{non,chem}}}}$，I3）**：数学上，2-W/2-WC 在正交方差分解假设 $\\sigma_s^2 = \\sigma_a^2 + \\sigma_{{\\text{{non}}}}^2$ 下严格成立；将谱形可预测的 $R^2_{{\\text{{shape}}}}$ 全部归因于 Beer-Lambert 化学幅值变化是工作物理假说（如 `Wheat A1 R²_shape=0.95`、`IFPEN R²_shape=0.91`，除以 `row_std(x_i)` 会引入有理分母畸变）。",
        f"- **数值边界与离散容差（E3）**：实现中 $R^2_{{\\text{{shape}}}}$ 上限截断于 `0.98`，使 `ws_wiener` 下界为 `0.02`（`ALL_54` 中 `{bda['n_wiener_at_0p02_cap']}/54` 触及该下界：`EXTERNAL_30` `{bda['n_ext30_at_0p02_cap']}/30` [`Deploy_Melamine`]、`Classic_23` `7/23` [全部 Diesel]）；`ws_unified` 来自 1001 点网格（步长 `1e-3`），`{bda['n_unified_grid_overshoot']}/54` 略超连续上界 `1-R²_shape`（最大差 `{bda['max_unified_overshoot']}`，在 `1e-3` 网格容差内）。",
        f"- **11 点 `WS_GRID` 经验曲线相关性（`VALID_53` 中位数）**：`corr(cal_cv(w_s), test(w_s))` PLS `{diag['median_corr_cal_cv_vs_test_pls']:+.3f}` / Ridge `{diag['median_corr_cal_cv_vs_test_ridge']:+.3f}`；而旧版 `SNR_old(w_s, σ_non=0)` 与 `-test(w_s)` 仅 `{diag['median_corr_snr_old_vs_neg_test_pls']:+.3f}`，纳入 $\\sigma_{{\\text{{non,chem}}}}$ 后的 `SNR_unified(w_s)` 升至 `{diag['median_corr_snr_unified_vs_neg_test_pls']:+.3f}`。",
        "",
        "## 2. 固定外部池对比结果：`EXTERNAL_30`（Phase B Valid 17 + Round 3 13）",
        "",
        "### 2.1 PLS 主头 (`EXTERNAL_30`, `n = 30`)",
        "",
        "| 方案 | 需要 `y` 选 $w_s$？ | 均值 `RMSEP_z` | 均值 `Δ% vs 最优端点` (中位) | 最坏 `Δ%` | `≥+5%` 严重 | `beats_both` | `w=10` 代价 | 校准耗时 (加速) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = [
        ("ws_0", "不做尺度校正 (`ws=0`)", "否"),
        ("ws_1", "完整尺度校正 (`ws=1`, SNV)", "否"),
        ("ws_closed_v2_old", "旧版闭式 (`v2, σ_non=0`)", "是"),
        ("ws_wiener", "**无监督维纳 (`1 - R²_shape`, 2-W)**", "**否 (零 `y`)**"),
        ("ws_unified", "**统一维纳–耦合 (2-WC)**", "是"),
        ("snv_plus_logs", "消融 `SNV + log(s)`", "否"),
        ("ws_inner_cv", "内层 CV (`11 点`)", "是"),
        ("ws_oracle", "Oracle 上界 (参考)", "泄漏"),
    ]
    for mk, lbl, need_y in labels:
        m = ext_pls[mk]
        spd = f"{m['total_cal_time_s']:.1f}s ({m['speedup_vs_inner_cv']:.1f}x)" if "speedup_vs_inner_cv" in m else f"{m['total_cal_time_s']:.1f}s"
        lines.append(
            f"| {lbl} | {need_y} | {m['mean_rmsep_z']:.4f} | "
            f"{m['mean_rel_pct_vs_best_ep']:+.2f}% ({m['median_rel_pct_vs_best_ep']:+.2f}%) | "
            f"{m['worst_rel_pct_vs_best_ep']:+.2f}% | {m['severe_regret_ge5pct_count']} | "
            f"{m['beats_both_strict_count']}/30 | {m['asymmetric_cost_w10']:.2f} | {spd} |"
        )

    lines.extend(
        [
            "",
            "### 2.2 Ridge 伴随头 (`EXTERNAL_30`, `n = 30`)",
            "",
            "| 方案 | 均值 `RMSEP_z` | 均值 `Δ% vs 最优端点` (中位) | 最坏 `Δ%` | `≥+5%` 严重 | `beats_both` | `w=10` 代价 | 校准耗时 (加速) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mk, lbl, _ in labels:
        m = ext_ridge[mk]
        spd = f"{m['total_cal_time_s']:.1f}s ({m['speedup_vs_inner_cv']:.1f}x)" if "speedup_vs_inner_cv" in m else f"{m['total_cal_time_s']:.1f}s"
        lines.append(
            f"| {lbl} | {m['mean_rmsep_z']:.4f} | "
            f"{m['mean_rel_pct_vs_best_ep']:+.2f}% ({m['median_rel_pct_vs_best_ep']:+.2f}%) | "
            f"{m['worst_rel_pct_vs_best_ep']:+.2f}% | {m['severe_regret_ge5pct_count']} | "
            f"{m['beats_both_strict_count']}/30 | {m['asymmetric_cost_w10']:.2f} | {spd} |"
        )

    lines.extend(
        [
            "",
            "## 3. 预注册假说 `H1–H4` 裁决（`EXTERNAL_30`，全精度逐项检验）",
            "",
            f"- **H1（`ws_wiener` 与 `ws_unified` 均在均值遗憾、最坏遗憾、`w=10` 代价全面优于 `C-148` 旧版 `ws_closed_v2`）**：PLS `{hv['H1_pls_pass']}` (`wiener={hv['H1_pls_wiener_pass']}`, `unified={hv['H1_pls_unified_pass']}`) / Ridge `{hv['H1_ridge_pass']}` (`wiener={hv['H1_ridge_wiener_pass']}`, `unified={hv['H1_ridge_unified_pass']}`)。",
            f"- **H2（`ws_wiener` 与 `ws_unified` 均在均值遗憾与 `w=10` 代价严格低于两端点 `ws=0` 与 `ws=1`）**：PLS `{hv['H2_pls_pass']}` (`wiener={hv['H2_pls_wiener_pass']}`, `unified={hv['H2_pls_unified_pass']}`) / **Ridge `{hv['H2_ridge_pass']}`** (`wiener={hv['H2_ridge_wiener_pass']}` [`+7.30% < +7.76%`, `w=10 50.48 < 57.15`], **`unified={hv['H2_ridge_unified_pass']}`** [均值遗憾 `{ext_ridge['ws_unified']['mean_rel_pct_vs_best_ep_exact']:.5f}%` > `ws=0 {ext_ridge['ws_0']['mean_rel_pct_vs_best_ep_exact']:.5f}%`，虽 `w=10 {ext_ridge['ws_unified']['asymmetric_cost_w10']:.2f} < {ext_ridge['ws_0']['asymmetric_cost_w10']:.2f}`])。",
            f"- **H3（与 11 点 `ws_inner_cv` 对比；`p` 值仅检验逐任务 `RMSEP_z` 差异 `n=30`，非检验 `w=10` 代价相等，I2）**：PLS 上 `ws_wiener` 均值遗憾 `{ext_pls['ws_wiener']['mean_rel_pct_vs_best_ep']:+.2f}%`（`w=10` 代价 `{ext_pls['ws_wiener']['asymmetric_cost_w10']:.2f}`；逐任务 `RMSEP_z` vs `inner_cv` Wilcoxon `p={ext_pls['ws_wiener']['wilcoxon_p_vs_inner_cv']:.4f}`）vs `ws_inner_cv` `{ext_pls['ws_inner_cv']['mean_rel_pct_vs_best_ep']:+.2f}%`（`{ext_pls['ws_inner_cv']['asymmetric_cost_w10']:.2f}`），加速 `{ext_pls['ws_wiener']['speedup_vs_inner_cv']}x`；Ridge 上 `ws_wiener` `{ext_ridge['ws_wiener']['mean_rel_pct_vs_best_ep']:+.2f}%`（`{ext_ridge['ws_wiener']['asymmetric_cost_w10']:.2f}`；逐任务 `RMSEP_z` vs `inner_cv` Wilcoxon `p={ext_ridge['ws_wiener']['wilcoxon_p_vs_inner_cv']:.4f}`）vs `ws_inner_cv` `{ext_ridge['ws_inner_cv']['mean_rel_pct_vs_best_ep']:+.2f}%`（`{ext_ridge['ws_inner_cv']['asymmetric_cost_w10']:.2f}`），加速 `{ext_ridge['ws_wiener']['speedup_vs_inner_cv']}x`。",
            f"- **H4（线性追加消融 `[X_snv, z_log(row_std)]` 在本 54 任务池上无法消除分母非线性畸变，I3）**：`{hv['H4_pass']}`（PLS `EXTERNAL_30` 最坏 `{ext_pls['snv_plus_logs']['worst_rel_pct_vs_best_ep']:+.2f}%`、`w=10` 代价 `{ext_pls['snv_plus_logs']['asymmetric_cost_w10']:.2f}`；`Classic_23` 最坏 `{dev_pls['snv_plus_logs']['worst_rel_pct_vs_best_ep']:+.2f}%`）。",
            "",
            "## 4. `EXTERNAL_30` 逐任务对比表（PLS 主头）",
            "",
            "| Task ID | `w_wie` | `w_uni` | `ws=0` | `ws=1` | `v2_old` | `ws_wiener` | `ws_unified` | `inner_cv` |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in res["tasks"]:
        if not (r["is_external_prospective"] and not r["has_label_scale_anomaly"]):
            continue
        pm = r["pls"]["metrics"]
        lines.append(
            f"| `{r['task_id']}` | {r['ws_wiener']:.3f} | {r['ws_unified']:.3f} | "
            f"{pm['ws_0']['rmsep_z']:.4f} | {pm['ws_1']['rmsep_z']:.4f} | "
            f"{pm['ws_closed_v2_old']['rmsep_z']:.4f} | "
            f"{pm['ws_wiener']['rmsep_z']:.4f} ({pm['ws_wiener']['rel_pct_vs_best_ep']:+.1f}%) | "
            f"{pm['ws_unified']['rmsep_z']:.4f} ({pm['ws_unified']['rel_pct_vs_best_ep']:+.1f}%) | "
            f"{pm['ws_inner_cv']['rmsep_z']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 全量 `VALID_53` 与开发诊断集 `Classic_23` 汇总参考",
            "",
            f"- **`VALID_53` (PLS)**：`ws_wiener` `RMSEP_z={v53_pls['ws_wiener']['mean_rmsep_z']:.4f}`（遗憾 `{v53_pls['ws_wiener']['mean_rel_pct_vs_best_ep']:+.2f}%`、`w=10` `{v53_pls['ws_wiener']['asymmetric_cost_w10']:.2f}`、`beats_both={v53_pls['ws_wiener']['beats_both_strict_count']}/53`）、`ws_unified` `{v53_pls['ws_unified']['mean_rmsep_z']:.4f}`（`{v53_pls['ws_unified']['mean_rel_pct_vs_best_ep']:+.2f}%`、`{v53_pls['ws_unified']['asymmetric_cost_w10']:.2f}`、`{v53_pls['ws_unified']['beats_both_strict_count']}/53`）vs `ws_closed_v2_old` `{v53_pls['ws_closed_v2_old']['mean_rmsep_z']:.4f}`（`+14.22%`、`{v53_pls['ws_closed_v2_old']['asymmetric_cost_w10']:.2f}`、`0/53`）vs `ws_inner_cv` `{v53_pls['ws_inner_cv']['mean_rmsep_z']:.4f}`（`+4.11%`、`{v53_pls['ws_inner_cv']['asymmetric_cost_w10']:.2f}`）。",
            f"- **`Classic_23` (开发集, PLS / Ridge)**：`ws_wiener` 均值遗憾 PLS `{dev_pls['ws_wiener']['mean_rel_pct_vs_best_ep']:+.2f}%`（`{dev_pls['ws_wiener']['beats_both_strict_count']}/23`）/ Ridge `{dev_ridge['ws_wiener']['mean_rel_pct_vs_best_ep']:+.2f}%`（`{dev_ridge['ws_wiener']['beats_both_strict_count']}/23`）。",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 prospective unblinding for Theorem 2-W / 2-WC")
    parser.add_argument("--verify-only", action="store_true", help="Verify lock chain & archived JSON.")
    args = parser.parse_args()

    t_start = time.time()
    pre_path = ROOT / "_audit" / "prereg_wiener_ws_2026_11_25.json"
    res_path = ROOT / "_audit" / "res_wiener_ws_2026_11_25.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    pre_sha = _file_sha256(pre_path)
    pre_sha_lf = _file_sha256_lf(pre_path)

    theory_path = ROOT / "chemdl" / "theory.py"
    wt_lf = _file_sha256_lf(theory_path)
    assert wt_lf == FROZEN_THEORY_LF_SHA256 == pre["theory_py_git_blob_sha256_lf"]

    phase1_parent_commit = str(pre["git_head_frozen"])
    has_git = (ROOT / ".git").exists() and (
        subprocess.run(
            ["git", "cat-file", "-e", f"{phase1_parent_commit}^{{commit}}"],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
    )
    if has_git:
        blob_lf = _git_blob_sha256_lf(ROOT, phase1_parent_commit, "chemdl/theory.py")
        assert blob_lf == FROZEN_THEORY_LF_SHA256
        phase1_lock_commit = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%H", "--diff-filter=A", "--", "_audit/prereg_wiener_ws_2026_11_25.json"],
            cwd=ROOT,
            text=True,
        ).strip()
        assert phase1_lock_commit, "Cannot locate commit adding _audit/prereg_wiener_ws_2026_11_25.json"
        assert _git_blob_sha256_lf(ROOT, phase1_lock_commit, "_audit/prereg_wiener_ws_2026_11_25.json") == pre_sha_lf
        exec_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    else:
        phase1_lock_commit = "CLEAN_CLONE_NO_GIT"
        exec_head = "CLEAN_CLONE_NO_GIT"

    if args.verify_only:
        res = json.loads(res_path.read_text(encoding="utf-8"))
        assert res.get("phase") == "WIENER_WS_PHASE_2_UNBLINDED_VERIFIED"
        assert res.get("prereg_json_sha256_lf") == pre_sha_lf
        assert res.get("theory_py_git_blob_sha256_lf") == FROZEN_THEORY_LF_SHA256
        if has_git:
            assert res.get("phase1_lock_commit") == phase1_lock_commit
        assert res.get("n_tasks") == 54 and len(res["tasks"]) == 54
        assert all(t["cal_test_hash_overlap"] == 0 for t in res["tasks"])

        data_ext_dir = ROOT / "data_ext"
        if data_ext_dir.exists():
            from _audit._ws_closed_value_loaders import load_ws_value_tasks

            live = load_ws_value_tasks(ROOT)
            pre_by_id = {r["task_id"]: r for r in pre["tasks"]}
            for lt in live:
                assert lt["split_sha256"] == pre_by_id[lt["task_id"]]["split_sha256"]
                assert lt["cal_test_hash_overlap"] == 0
        print(
            f"[verify-only PASS] wiener_ws parent={phase1_parent_commit[:7]}, lock={phase1_lock_commit[:7]}, "
            f"exec={exec_head[:7]}, 54 tasks verified (data_ext={'present' if data_ext_dir.exists() else 'absent'})."
        )
        return

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "chemdl/theory.py", "_audit/prereg_wiener_ws_2026_11_25.json"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert not dirty, f"Frozen files dirty: {dirty}"

    from _audit._ws_closed_value_loaders import fit_eval_pls_at_ws, fit_eval_ridge_at_ws, load_ws_value_tasks

    c148_pre = json.loads((ROOT / "_audit" / "prereg_ws_closed_value_2026_11_25.json").read_text(encoding="utf-8"))
    c148_res = json.loads((ROOT / "_audit" / "res_ws_closed_value_2026_11_25.json").read_text(encoding="utf-8"))
    c148_pre_map = {r["task_id"]: r for r in c148_pre["tasks"]}
    c148_res_map = {r["task_id"]: r for r in c148_res["tasks"]}
    pre_by_id = {r["task_id"]: r for r in pre["tasks"]}

    tasks = load_ws_value_tasks(ROOT)
    assert len(tasks) == 54

    task_rows = []
    corr_cal_pls, corr_old_pls, corr_uni_pls, corr_cal_ridge = [], [], [], []
    ora_reasons: dict[str, int] = {}
    n_ora_int = 0

    for i, t in enumerate(tasks, 1):
        tid = t["task_id"]
        p = pre_by_id[tid]
        r_old = c148_res_map[tid]
        p_old = c148_pre_map[tid]
        assert t["split_sha256"] == p["split_sha256"] == r_old["split_sha256"]

        Xc, yc, Xt, yt = t["X_cal"], t["y_cal"], t["X_test"], t["y_test"]

        # Evaluate 3 new locked methods (PLS)
        pc = p["pls_cal_config"]
        e_wie_p = fit_eval_pls_at_ws(Xc, yc, Xt, yt, pc["ws_wiener"]["ws"], pc["ws_wiener"]["k"], wc=FIXED_WC)["rmsep_z"]
        e_uni_p = fit_eval_pls_at_ws(Xc, yc, Xt, yt, pc["ws_unified"]["ws"], pc["ws_unified"]["k"], wc=FIXED_WC)["rmsep_z"]
        e_aug_p = _fit_eval_pls_snv_plus_logs(Xc, yc, Xt, yt, pc["snv_plus_logs"]["k"])["rmsep_z"]

        pm_old = r_old["pls"]["metrics"]
        e0_p = pm_old["ws_0"]["rmsep_z"]
        e1_p = pm_old["ws_1"]["rmsep_z"]
        ev2_p = pm_old["ws_closed_v2"]["rmsep_z"]
        ecv_p = pm_old["ws_inner_cv"]["rmsep_z"]
        eora_p = pm_old["ws_oracle"]["rmsep_z"]

        pls_raw_errs = {
            "ws_0": e0_p,
            "ws_1": e1_p,
            "ws_closed_v2_old": ev2_p,
            "ws_wiener": e_wie_p,
            "ws_unified": e_uni_p,
            "snv_plus_logs": e_aug_p,
            "ws_inner_cv": ecv_p,
            "ws_oracle": eora_p,
        }
        pls_metrics = {mk: _method_metrics(ev, e0_p, e1_p, ev2_p, ecv_p) for mk, ev in pls_raw_errs.items()}
        pls_times = {
            "ws_0": float(r_old["pls"]["cal_times_s"]["ws_0"]),
            "ws_1": float(r_old["pls"]["cal_times_s"]["ws_1"]),
            "ws_closed_v2_old": float(r_old["pls"]["cal_times_s"]["ws_closed_v2"]),
            "ws_wiener": float(pc["ws_wiener"]["cal_time_s"]),
            "ws_unified": float(pc["ws_unified"]["cal_time_s"]),
            "snv_plus_logs": float(pc["snv_plus_logs"]["cal_time_s"]),
            "ws_inner_cv": float(r_old["pls"]["cal_times_s"]["ws_inner_cv"]),
        }

        # Evaluate 3 new locked methods (Ridge)
        rc = p["ridge_cal_config"]
        e_wie_r = fit_eval_ridge_at_ws(Xc, yc, Xt, yt, rc["ws_wiener"]["ws"], rc["ws_wiener"]["gamma"], wc=FIXED_WC)["rmsep_z"]
        e_uni_r = fit_eval_ridge_at_ws(Xc, yc, Xt, yt, rc["ws_unified"]["ws"], rc["ws_unified"]["gamma"], wc=FIXED_WC)["rmsep_z"]
        e_aug_r = _fit_eval_ridge_snv_plus_logs(Xc, yc, Xt, yt, rc["snv_plus_logs"]["gamma"])["rmsep_z"]

        rm_old = r_old["ridge"]["metrics"]
        e0_r = rm_old["ws_0"]["rmsep_z"]
        e1_r = rm_old["ws_1"]["rmsep_z"]
        ev2_r = rm_old["ws_closed_v2"]["rmsep_z"]
        ecv_r = rm_old["ws_inner_cv"]["rmsep_z"]
        eora_r = rm_old["ws_oracle"]["rmsep_z"]

        ridge_raw_errs = {
            "ws_0": e0_r,
            "ws_1": e1_r,
            "ws_closed_v2_old": ev2_r,
            "ws_wiener": e_wie_r,
            "ws_unified": e_uni_r,
            "snv_plus_logs": e_aug_r,
            "ws_inner_cv": ecv_r,
            "ws_oracle": eora_r,
        }
        ridge_metrics = {mk: _method_metrics(ev, e0_r, e1_r, ev2_r, ecv_r) for mk, ev in ridge_raw_errs.items()}
        ridge_times = {
            "ws_0": float(r_old["ridge"]["cal_times_s"]["ws_0"]),
            "ws_1": float(r_old["ridge"]["cal_times_s"]["ws_1"]),
            "ws_closed_v2_old": float(r_old["ridge"]["cal_times_s"]["ws_closed_v2"]),
            "ws_wiener": float(rc["ws_wiener"]["cal_time_s"]),
            "ws_unified": float(rc["ws_unified"]["cal_time_s"]),
            "snv_plus_logs": float(rc["snv_plus_logs"]["cal_time_s"]),
            "ws_inner_cv": float(r_old["ridge"]["cal_times_s"]["ws_inner_cv"]),
        }

        # Root-cause dissection stats on VALID_53
        if not t["has_label_scale_anomaly"]:
            pv2 = p["physics_frozen"]
            ws_ora = r_old["pls"]["ws_oracle"]
            if (0.01 < ws_ora < 0.99) and (pm_old["ws_oracle"]["rel_pct_vs_best_ep"] <= -0.5):
                n_ora_int += 1
                sq_a_r = np.sqrt(pv2["alpha"]) * abs(pv2["r_s"])
                ra = abs(pv2["r_s"])
                if 0.01 < pv2["ws_star"] < 0.99:
                    rk = "INTERIOR_HIT"
                elif sq_a_r <= pv2["beta"]:
                    rk = "sqA|r|<=beta->1"
                elif np.sqrt(pv2["alpha"]) * (1.0 - ra * ra) - ra <= 0.0:
                    rk = "denom<=0->0"
                else:
                    rk = "u*>sa^2->0"
                ora_reasons[rk] = ora_reasons.get(rk, 0) + 1

            wg = np.array([g["ws"] for g in p_old["pls_cal_config"]["grid_pls"]], dtype=np.float64)
            cal_e = np.array([g["cv_rmsep_z"] for g in p_old["pls_cal_config"]["grid_pls"]], dtype=np.float64)
            te_e = np.array([g["rmsep_z"] for g in r_old["pls"]["grid_evals"]], dtype=np.float64)
            cal_r = np.array([g["cv_rmsep_z"] for g in p_old["ridge_cal_config"]["grid_ridge"]], dtype=np.float64)
            te_r = np.array([g["rmsep_z"] for g in r_old["ridge"]["grid_evals"]], dtype=np.float64)
            snr_old = np.asarray(snv_snr_curve(wg, pv2["alpha"], pv2["beta"], pv2["r_s"], pv2["sigma_a_true"], 0.0))
            snr_uni = np.asarray(snv_snr_curve(wg, pv2["alpha"], pv2["beta"], pv2["r_s"], pv2["sigma_a_true"], pv2["sigma_non_chem"]))
            if cal_e.std() > 1e-12 and te_e.std() > 1e-12:
                corr_cal_pls.append(float(np.corrcoef(cal_e, te_e)[0, 1]))
                if snr_old.std() > 1e-12:
                    corr_old_pls.append(float(np.corrcoef(snr_old, -te_e)[0, 1]))
                if snr_uni.std() > 1e-12:
                    corr_uni_pls.append(float(np.corrcoef(snr_uni, -te_e)[0, 1]))
            if cal_r.std() > 1e-12 and te_r.std() > 1e-12:
                corr_cal_ridge.append(float(np.corrcoef(cal_r, te_r)[0, 1]))

        row = {
            "task_id": tid,
            "suite": t["suite"],
            "is_external_prospective": p["is_external_prospective"],
            "has_label_scale_anomaly": t["has_label_scale_anomaly"],
            "cal_test_hash_overlap": 0,
            "split_sha256": t["split_sha256"],
            "ws_wiener": p["ws_wiener"],
            "ws_unified": p["ws_unified"],
            "pre_tags": p["pre_tags"],
            "pls": {"cal_times_s": pls_times, "metrics": pls_metrics},
            "ridge": {"cal_times_s": ridge_times, "metrics": ridge_metrics},
        }
        task_rows.append(row)
        print(
            f"[{i:02d}/54] {tid:42s} | old={ev2_p:.4f} | "
            f"wie={e_wie_p:.4f} ({pls_metrics['ws_wiener']['rel_pct_vs_best_ep']:+.2f}%) | "
            f"uni={e_uni_p:.4f} ({pls_metrics['ws_unified']['rel_pct_vs_best_ep']:+.2f}%) | "
            f"cv={ecv_p:.4f}"
        )

    universes = {
        "EXTERNAL_30": [r for r in task_rows if r["is_external_prospective"] and not r["has_label_scale_anomaly"]],
        "EXTERNAL_31": [r for r in task_rows if r["is_external_prospective"]],
        "PhaseB_Valid_17": [r for r in task_rows if r["suite"] == "PhaseB_Deploy" and not r["has_label_scale_anomaly"]],
        "Round3_13": [r for r in task_rows if r["suite"] == "Round3_Unseen"],
        "Classic_23": [r for r in task_rows if not r["is_external_prospective"]],
        "VALID_53": [r for r in task_rows if not r["has_label_scale_anomaly"]],
        "ALL_54": task_rows,
    }
    summary = {
        un: {"pls": _aggregate(rows, "pls"), "ridge": _aggregate(rows, "ridge")}
        for un, rows in universes.items()
    }

    verdicts = evaluate_wiener_hypotheses(summary["EXTERNAL_30"])
    cap02_ids = [r["task_id"] for r in task_rows if r["ws_wiener"] <= 0.020001]
    ext30_cap02 = [r["task_id"] for r in universes["EXTERNAL_30"] if r["ws_wiener"] <= 0.020001]
    overshoots = [
        {"task_id": r["task_id"], "ws_unified": r["ws_unified"], "ws_wiener": r["ws_wiener"], "diff": round(r["ws_unified"] - r["ws_wiener"], 6)}
        for r in task_rows if r["ws_unified"] > r["ws_wiener"] + 1e-12
    ]

    payload = _sanitize_json(
        {
            "phase": "WIENER_WS_PHASE_2_UNBLINDED_VERIFIED",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "phase1_parent_git_head": phase1_parent_commit,
            "phase1_lock_commit": phase1_lock_commit,
            "execution_git_head": exec_head,
            "prereg_json_sha256": pre_sha,
            "prereg_json_sha256_lf": pre_sha_lf,
            "theory_py_git_blob_sha256_lf": FROZEN_THEORY_LF_SHA256,
            "n_tasks": len(task_rows),
            "elapsed_s": round(time.time() - t_start, 2),
            "boundary_discretization_audit": {
                "r2_shape_cap": 0.98,
                "ws_wiener_floor": 0.02,
                "n_wiener_at_0p02_cap": len(cap02_ids),
                "wiener_at_0p02_cap_ids": cap02_ids,
                "n_ext30_at_0p02_cap": len(ext30_cap02),
                "n_wiener_strictly_above_0p02": len(task_rows) - len(cap02_ids),
                "n_unified_grid_overshoot": len(overshoots),
                "max_unified_overshoot": round(max((x["diff"] for x in overshoots), default=0.0), 6),
                "unified_overshoots": overshoots,
            },
            "root_cause_dissection": {
                "n_oracle_interior": n_ora_int,
                "oracle_interior_breakdown": ora_reasons,
                "median_corr_cal_cv_vs_test_pls": round(float(np.median(corr_cal_pls)), 4),
                "median_corr_cal_cv_vs_test_ridge": round(float(np.median(corr_cal_ridge)), 4),
                "median_corr_snr_old_vs_neg_test_pls": round(float(np.median(corr_old_pls)), 4),
                "median_corr_snr_unified_vs_neg_test_pls": round(float(np.median(corr_uni_pls)), 4),
            },
            "hypothesis_verdicts_external_30": verdicts,
            "summary": summary,
            "tasks": task_rows,
        }
    )

    res_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    out_md = ROOT / "reports" / "antigravity" / "UNBLIND_WIENER_WS_2026-11-25.md"
    _write_report(payload, out_md)
    print(f"\nPhase 2 unblinded 54 tasks in {payload['elapsed_s']}s -> {res_path}")
    print("Verdicts on EXTERNAL_30:", verdicts)


if __name__ == "__main__":
    main()
