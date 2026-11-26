"""回应 REVIEW_2026-11-23.md 全部 P0/P1 定量核验（可复现入库脚本）：

1. P0-1（7 经典 NIR 库 `tuned-SG` 与 `SNV+tuned-SG` 强基线对照，严格同 3×5 CV seed 100..102 + 同精确 LOO Ridge 头）
   + P1-4（同时输出原物理量纲 RMSEP、R²、RPD，与 z-score pooled RMSEP 并列）；
2. P0-1 #3（Tecator 2.2× 增益自由度拆解消融：Raw / SNV / tuned-SG / SNV+tuned-SG / FOD-only / SNV-opt-only / Bilevel / MK-Bilevel）；
3. P0-2（定理 1 v1/v2 完整 170 折 2×2 混淆矩阵、Balanced Accuracy、MCC、全局多数类 vs 逐库先知多数类、以及弃权区覆蓋/精度）；
4. Appendix B（Melamine 4 台仪器 n=300 预算下 `fixed SG1(11,3,d1)` vs `tuned-SG (d∈{0,1,2})` vs `tuned-SG-deriv (d∈{1,2})` 拆解）。

结果落盘至 `_audit/res_review_p0_2026_11_23.json`。
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import savgol_filter
import scipy.io
import torch

from chemdl.cordoba import load_cordoba_all
from chemdl.corn import load_corn
from chemdl.diesel import load_diesel
from chemdl.diffprep import (
    _best_loo_log_gamma_eigh,
    exact_loo_ridge_residuals,
    fit_bilevel_diffprep,
    fit_mk_bilevel_diffprep,
)
from chemdl.tablets import load_tablets
from chemdl.tecator import load_tecator
from chemdl.theory import estimate_physical_params, estimate_physical_params_v2
from chemdl.transfer import PREPS, fit_pls, rmsep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SG_GRID = [
    (w, min(3, w - 1), d)
    for w in (5, 7, 9, 11, 15, 21, 31)
    for d in (0, 1, 2)
    if d <= min(3, w - 1)
]
SG_DERIV_GRID = [cfg for cfg in SG_GRID if cfg[2] >= 1]


def _apply_sg(X: np.ndarray, w: int, poly: int, d: int) -> np.ndarray:
    if d == 0 and w == 1:
        return X.copy()
    w_eff = min(w, X.shape[1] if X.shape[1] % 2 == 1 else X.shape[1] - 1)
    p_eff = min(poly, w_eff - 1)
    return savgol_filter(X, window_length=w_eff, polyorder=p_eff, deriv=d, axis=1)


LG_GRID = np.linspace(-7.0, 2.5, 20)


def _fit_exact_loo_ridge_np(Z_tr: np.ndarray, Y_tr: np.ndarray, Z_va: np.ndarray):
    """与 BilevelDiffPrepRidge / fit_loo_ridge_fixed 逐位一致的精确含截距 Allen-PRESS LOO 岭回归预测。"""
    Z_t = torch.from_numpy(np.asarray(Z_tr, dtype=np.float64))
    Y_t = torch.from_numpy(np.asarray(Y_tr, dtype=np.float64))
    loo_val, lg = _best_loo_log_gamma_eigh(Z_t, Y_t, LG_GRID)
    gamma = float(np.exp(lg))
    e, zm, zs, da = exact_loo_ridge_residuals(Z_t, Y_t, torch.tensor(gamma, dtype=torch.float64))
    Zt = (Z_t - zm) / zs
    W = (Zt.t() @ da).numpy()
    Zva_n = (np.asarray(Z_va, dtype=np.float64) - zm.numpy()) / zs.numpy()
    pred = Zva_n @ W
    return pred, float(loo_val), gamma


def _fit_tuned_sg_ridge(X_tr: np.ndarray, Y_tr: np.ndarray, X_va: np.ndarray,
                        grid=SG_GRID, pre_snv: bool = False):
    """折内精确 LOO 筛选 SG (window, poly, deriv)，严格零测试集泄漏，与 Bilevel 完全同回归头。"""
    Xtr_in = PREPS["snv"](X_tr) if pre_snv else X_tr
    Xva_in = PREPS["snv"](X_va) if pre_snv else X_va
    best_loo = float("inf")
    best_cfg = None
    best_Ztr = None
    Y_t = torch.from_numpy(np.asarray(Y_tr, dtype=np.float64))
    for (w, p, d) in grid:
        Ztr = _apply_sg(Xtr_in, w, p, d)
        Z_t = torch.from_numpy(np.asarray(Ztr, dtype=np.float64))
        loo_val, _ = _best_loo_log_gamma_eigh(Z_t, Y_t, LG_GRID)
        if loo_val < best_loo:
            best_loo = loo_val
            best_cfg = (w, p, d)
            best_Ztr = Ztr
    Zva = _apply_sg(Xva_in, *best_cfg)
    pred, _, _ = _fit_exact_loo_ridge_np(best_Ztr, Y_tr, Zva)
    return pred, best_cfg


def _phys_metrics(Y_true_raw: np.ndarray, Y_pred_raw: np.ndarray, col_names: list[str]):
    out = {}
    for j, cname in enumerate(col_names):
        yt = Y_true_raw[:, j]
        yp = Y_pred_raw[:, j]
        e = float(np.sqrt(np.mean((yp - yt) ** 2)))
        sd = float(np.std(yt, ddof=0))
        r2 = float(1.0 - np.sum((yp - yt) ** 2) / max(np.sum((yt - yt.mean()) ** 2), 1e-12))
        rpd = float(sd / max(e, 1e-12))
        out[cname] = {"rmsep": round(e, 4), "r2": round(r2, 4), "rpd": round(rpd, 2)}
    return out


def part1_tuned_sg_and_physical_units():
    print("\n=== Part 1: 7 Classic Datasets tuned-SG & SNV+tuned-SG (3x5 CV) + Physical Units ===")
    t = load_tecator()
    c = load_corn()
    tab = load_tablets()
    cb = load_cordoba_all()
    di = load_diesel()
    ok_d = np.isfinite(di["X"]).all(1) & np.isfinite(di["y"][:, 1])

    datasets = [
        ("Tecator", t["X"], t["y"], ["water_%", "fat_%", "protein_%"]),
        ("Corn_m5", c["m5"], c["y"], ["moisture_%", "oil_%", "protein_%", "starch_%"]),
        ("Tablets_X1", tab["X1"][tab["part"] == 0], tab["y"][tab["part"] == 0],
         ["weight_mg", "hardness", "assay_mg"]),
        ("Cordoba_wheat", cb["wheat"]["X"], cb["wheat"]["y"], ["moisture_%", "protein_%"]),
        ("Cordoba_barley", cb["barley"]["X"], cb["barley"]["y"], ["moisture_%", "protein_%"]),
        ("Cordoba_maize", cb["maize"]["X"], cb["maize"]["y"], ["moisture_%", "protein_%"]),
        ("SWRI_Diesel_CN", di["X"][ok_d][:200], di["y"][ok_d][:200, [1]], ["cetane_number"]),
    ]

    ref_mk = json.load(open(os.path.join(ROOT, "_audit", "res_mk_bilevel_and_sport_fix_2026_11_21.json"),
                            encoding="utf-8"))["datasets"]

    res = {}
    for name, X, Y, cnames in datasets:
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]

        sq_z = {"tuned_sg": [], "snv_tuned_sg": [], "raw_ridge": [], "snv_ridge": []}
        y_true_all = []
        y_pred_raw = {k: [] for k in sq_z}
        cfgs_sg, cfgs_snv_sg = [], []

        for rep in range(3):
            perm = np.random.default_rng(100 + rep).permutation(len(Y))
            for va in np.array_split(perm, 5):
                tr = np.setdiff1d(perm, va)
                mu, sd = Y[tr].mean(0), np.clip(Y[tr].std(0), 1e-9, None)
                Ytr, Yva = (Y[tr] - mu) / sd, (Y[va] - mu) / sd

                p_tsg, cfg1 = _fit_tuned_sg_ridge(X[tr], Ytr, X[va], grid=SG_GRID, pre_snv=False)
                p_stsg, cfg2 = _fit_tuned_sg_ridge(X[tr], Ytr, X[va], grid=SG_GRID, pre_snv=True)
                p_raw, _, _ = _fit_exact_loo_ridge_np(X[tr], Ytr, X[va])
                p_snv, _, _ = _fit_exact_loo_ridge_np(PREPS["snv"](X[tr]), Ytr, PREPS["snv"](X[va]))

                cfgs_sg.append(str(cfg1))
                cfgs_snv_sg.append(str(cfg2))
                y_true_all.append(Y[va])
                for k, P in [("tuned_sg", p_tsg), ("snv_tuned_sg", p_stsg),
                             ("raw_ridge", p_raw), ("snv_ridge", p_snv)]:
                    sq_z[k].append((P - Yva) ** 2)
                    y_pred_raw[k].append(P * sd + mu)

        yt_cat = np.concatenate(y_true_all, axis=0)
        z_pooled = {k: round(float(np.sqrt(np.mean(np.concatenate(sq_z[k], axis=0)))), 4) for k in sq_z}
        bi_ref = ref_mk[name]["pooled_rmsep"]["bilevel"]
        mk_ref = ref_mk[name]["pooled_rmsep"]["mk_bilevel"]
        sp_ref = ref_mk[name]["pooled_rmsep"]["sport_fix"]
        best_sg_both = min(z_pooled["tuned_sg"], z_pooled["snv_tuned_sg"])

        # Top selected SG configs
        from collections import Counter
        top_sg = Counter(cfgs_sg).most_common(3)
        top_ssg = Counter(cfgs_snv_sg).most_common(3)

        res[name] = {
            "z_pooled": {
                **z_pooled,
                "bilevel_ref": bi_ref,
                "mk_bilevel_ref": mk_ref,
                "sport_fix_ref": sp_ref,
            },
            "mk_vs_tuned_sg_pct": round((mk_ref / z_pooled["tuned_sg"] - 1.0) * 100.0, 2),
            "mk_vs_best_tuned_sg_or_snv_sg_pct": round((mk_ref / best_sg_both - 1.0) * 100.0, 2),
            "top_sg_cfgs": top_sg,
            "top_snv_sg_cfgs": top_ssg,
            "physical_units_tuned_sg": _phys_metrics(yt_cat, np.concatenate(y_pred_raw["tuned_sg"], 0), cnames),
            "physical_units_snv_tuned_sg": _phys_metrics(yt_cat, np.concatenate(y_pred_raw["snv_tuned_sg"], 0), cnames),
        }
        print(
            f"  {name:16s} | MK={mk_ref:.4f} BI={bi_ref:.4f} | "
            f"tunedSG={z_pooled['tuned_sg']:.4f} SNV+tunedSG={z_pooled['snv_tuned_sg']:.4f} | "
            f"SPORT={sp_ref:.4f} | MK vs best(tSG,SNV+tSG)={res[name]['mk_vs_best_tuned_sg_or_snv_sg_pct']:+.2f}%"
        )
    return res


def part2_tecator_and_corn_physical_units_and_ablation():
    print("\n=== Part 2: Tecator 2.2x Ablation & Physical Units (Bilevel / MK-Bilevel) ===")
    t = load_tecator()
    c = load_corn()
    tab = load_tablets()
    di = load_diesel()
    ok_d = np.isfinite(di["X"]).all(1) & np.isfinite(di["y"][:, 1])

    # 2a. Tecator degree-of-freedom ablation (exact 3x5 CV seeds 100..102)
    X, Y = np.asarray(t["X"], np.float64), np.asarray(t["y"], np.float64)
    cnames = ["water_%", "fat_%", "protein_%"]
    sq = {"raw": [], "snv": [], "tuned_sg": [], "snv_tuned_sg": [],
          "bilevel_no_sg": [], "bilevel_full": [], "mk_bilevel": []}
    y_true_all, y_pred_bi, y_pred_mk = [], [], []

    for rep in range(3):
        perm = np.random.default_rng(100 + rep).permutation(len(Y))
        for va in np.array_split(perm, 5):
            tr = np.setdiff1d(perm, va)
            mu, sd = Y[tr].mean(0), np.clip(Y[tr].std(0), 1e-9, None)
            Ytr, Yva = (Y[tr] - mu) / sd, (Y[va] - mu) / sd

            p_raw, _, _ = _fit_exact_loo_ridge_np(X[tr], Ytr, X[va])
            p_snv, _, _ = _fit_exact_loo_ridge_np(PREPS["snv"](X[tr]), Ytr, PREPS["snv"](X[va]))
            p_tsg, _ = _fit_tuned_sg_ridge(X[tr], Ytr, X[va], grid=SG_GRID, pre_snv=False)
            p_stsg, _ = _fit_tuned_sg_ridge(X[tr], Ytr, X[va], grid=SG_GRID, pre_snv=True)

            # Bilevel without SG branch (SNV-opt + FOD only)
            p_bnosg, _ = fit_bilevel_diffprep(X[tr], Ytr, steps=60, seed=rep, use_sg_branch=False)
            # Full Bilevel (SNV-opt + FOD + SG)
            p_bfull, info_bi = fit_bilevel_diffprep(X[tr], Ytr, steps=60, seed=rep, physics_prior=True)
            p_mk, _ = fit_mk_bilevel_diffprep(X[tr], Ytr, steps=60, seed=rep,
                                              physics_prior=True, stage1_model=info_bi["model"])

            y_true_all.append(Y[va])
            P_bf = p_bfull(X[va]).reshape(Yva.shape)
            P_mk = p_mk(X[va]).reshape(Yva.shape)
            y_pred_bi.append(P_bf * sd + mu)
            y_pred_mk.append(P_mk * sd + mu)

            for k, P in [("raw", p_raw), ("snv", p_snv), ("tuned_sg", p_tsg),
                         ("snv_tuned_sg", p_stsg),
                         ("bilevel_no_sg", p_bnosg(X[va]).reshape(Yva.shape)),
                         ("bilevel_full", P_bf), ("mk_bilevel", P_mk)]:
                sq[k].append((P - Yva) ** 2)

    tec_z = {k: round(float(np.sqrt(np.mean(np.concatenate(sq[k], 0)))), 4) for k in sq}
    yt_cat = np.concatenate(y_true_all, 0)
    tec_phys_bi = _phys_metrics(yt_cat, np.concatenate(y_pred_bi, 0), cnames)
    tec_phys_mk = _phys_metrics(yt_cat, np.concatenate(y_pred_mk, 0), cnames)
    print("  Tecator ablation z-pooled:", tec_z)
    print("  Tecator MK-Bilevel physical units:", tec_phys_mk)

    # 2b. Physical units for Corn_m5, Tablets_X1, SWRI_Diesel_CN (MK-Bilevel)
    extra_phys = {"Tecator_MK": tec_phys_mk, "Tecator_BI": tec_phys_bi}
    for dname, Xd, Yd, cn in [
        ("Corn_m5", c["m5"], c["y"], ["moisture_%", "oil_%", "protein_%", "starch_%"]),
        ("Tablets_X1", tab["X1"][tab["part"] == 0], tab["y"][tab["part"] == 0],
         ["weight_mg", "hardness", "assay_mg"]),
        ("SWRI_Diesel_CN", di["X"][ok_d][:200], di["y"][ok_d][:200, [1]], ["cetane_number"]),
    ]:
        Xd, Yd = np.asarray(Xd, np.float64), np.asarray(Yd, np.float64)
        yt_l, yp_l = [], []
        for rep in range(3):
            perm = np.random.default_rng(100 + rep).permutation(len(Yd))
            for va in np.array_split(perm, 5):
                tr = np.setdiff1d(perm, va)
                mu, sd = Yd[tr].mean(0), np.clip(Yd[tr].std(0), 1e-9, None)
                Ytr = (Yd[tr] - mu) / sd
                p_bi, info_bi = fit_bilevel_diffprep(Xd[tr], Ytr, steps=60, seed=rep, physics_prior=True)
                p_mk, _ = fit_mk_bilevel_diffprep(Xd[tr], Ytr, steps=60, seed=rep,
                                                  physics_prior=True, stage1_model=info_bi["model"])
                yt_l.append(Yd[va])
                yp_l.append(p_mk(Xd[va]).reshape(Yd[va].shape) * sd + mu)
        extra_phys[dname + "_MK"] = _phys_metrics(np.concatenate(yt_l, 0), np.concatenate(yp_l, 0), cn)
        print(f"  {dname} MK-Bilevel physical units:", extra_phys[dname + "_MK"])

    return {"tecator_ablation_z": tec_z, "physical_units_mk": extra_phys}


def _confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = len(y_true)
    acc = (tp + tn) / max(n, 1)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    bac = 0.5 * (tpr + tnr) if (tp + fn > 0 and tn + fp > 0) else float("nan")
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": round(float(acc), 4),
        "tpr": round(float(tpr), 4),
        "tnr": round(float(tnr), 4),
        "balanced_acc": round(float(bac), 4) if np.isfinite(bac) else None,
        "mcc": round(float(mcc), 4),
    }


def part3_theorem1_confusion_matrix():
    print("\n=== Part 3: Theorem 1 Exact Confusion Matrix, Balanced Acc & MCC (170 checks) ===")
    t = load_tecator()
    c = load_corn()
    tab = load_tablets()
    cb = load_cordoba_all()
    m = scipy.io.loadmat(os.path.join(ROOT, "data_ext", "diesel", "SWRI_Diesel_NIR.mat"))
    Xd = np.asarray(m["diesel_spec"][0, 0]["data"], np.float64)
    Yd = np.asarray(m["diesel_prop"][0, 0]["data"], np.float64)[:, [1]]
    ok = np.isfinite(Xd).all(1) & np.isfinite(Yd).all(1)
    Xd, Yd = Xd[ok][:200], Yd[ok][:200]

    datasets = [
        ("Tecator", t["X"], t["y"]),
        ("Corn_m5", c["m5"], c["y"]),
        ("Tablets_X1", tab["X1"][tab["part"] == 0], tab["y"][tab["part"] == 0]),
        ("Cordoba_wheat", cb["wheat"]["X"], cb["wheat"]["y"]),
        ("Cordoba_barley", cb["barley"]["X"], cb["barley"]["y"]),
        ("Cordoba_maize", cb["maize"]["X"], cb["maize"]["y"]),
        ("SWRI_Diesel_CN", Xd, Yd),
    ]

    all_y, all_p1, all_p2, all_pred2_pct, all_emp_pct = [], [], [], [], []
    per_ds = {}
    per_ds_oracle_hits = 0

    for name, X, Y in datasets:
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        M = Y.shape[1]
        ys, p1s, p2s = [], [], []

        for rep in range(2):
            perm = np.random.default_rng(100 + rep).permutation(len(Y))
            for va in np.array_split(perm, 5):
                tr = np.setdiff1d(perm, va)
                mu, sd = Y[tr].mean(0), np.clip(Y[tr].std(0), 1e-9, None)
                Ytr, Yva = (Y[tr] - mu) / sd, (Y[va] - mu) / sd
                info1 = estimate_physical_params(X[tr], Ytr)
                info2 = estimate_physical_params_v2(X[tr], Ytr)
                Xtr_s, Xva_s = PREPS["snv"](X[tr]), PREPS["snv"](X[va])
                for j in range(M):
                    pr_raw, _ = fit_pls(X[tr], Ytr[:, [j]], kmax=10, inner=4, seed=rep)
                    pr_snv, _ = fit_pls(Xtr_s, Ytr[:, [j]], kmax=10, inner=4, seed=rep)
                    er = rmsep(pr_raw(X[va]), Yva[:, [j]])
                    es = rmsep(pr_snv(Xva_s), Yva[:, [j]])
                    yt = int(es < er)
                    c1, c2 = info1["per_col"][j], info2["per_col"][j]
                    pr1 = int(c1["pred_snv_vs_raw_pct"] < 0.0)
                    pr2 = int(c2["pred_snv_vs_raw_pct_v2"] < 0.0)
                    ys.append(yt); p1s.append(pr1); p2s.append(pr2)
                    all_y.append(yt); all_p1.append(pr1); all_p2.append(pr2)
                    all_pred2_pct.append(float(c2["pred_snv_vs_raw_pct_v2"]))
                    all_emp_pct.append((es / max(er, 1e-12) - 1.0) * 100.0)

        ys_a = np.array(ys)
        maj = max(int(ys_a.sum()), int((1 - ys_a).sum()))
        per_ds_oracle_hits += maj
        per_ds[name] = {
            "actual_snv_win_rate": round(float(ys_a.mean()), 4),
            "per_ds_majority_acc": round(maj / len(ys_a), 4),
            "v1": _confusion_metrics(ys_a, p1s),
            "v2": _confusion_metrics(ys_a, p2s),
        }

    all_y_a = np.array(all_y)
    all_p1_a = np.array(all_p1)
    all_p2_a = np.array(all_p2)
    all_pred2_a = np.array(all_pred2_pct)

    global_maj_acc = max(float(all_y_a.mean()), float(1.0 - all_y_a.mean()))
    per_ds_oracle_acc = per_ds_oracle_hits / len(all_y_a)

    abstain = {}
    for thr in (0.5, 1.0, 2.0, 5.0):
        mask = np.abs(all_pred2_a) >= thr
        if mask.sum() > 0:
            abstain[f"abs_pred_ge_{thr}pct"] = {
                "coverage": round(float(mask.mean()), 4),
                **_confusion_metrics(all_y_a[mask], all_p2_a[mask]),
            }

    out = {
        "total_checks": len(all_y_a),
        "global_constant_majority_acc": round(global_maj_acc, 4),
        "per_dataset_oracle_majority_acc": round(per_ds_oracle_acc, 4),
        "overall_v1": _confusion_metrics(all_y_a, all_p1_a),
        "overall_v2": _confusion_metrics(all_y_a, all_p2_a),
        "abstain_tiers_v2": abstain,
        "per_dataset": per_ds,
    }
    print(f"  Global constant majority ('always SNV'): {out['global_constant_majority_acc']:.4f}")
    print(f"  Per-dataset oracle majority:             {out['per_dataset_oracle_majority_acc']:.4f}")
    print("  Overall v1:", out["overall_v1"])
    print("  Overall v2:", out["overall_v2"])
    print("  Abstain tiers v2:", abstain)
    return out


def part4_melamine_sg_decomposition():
    print("\n=== Part 4: Melamine 4-Instrument SG Decomposition (n=300, 5-fold) ===", flush=True)
    m = scipy.io.loadmat(
        os.path.join(ROOT, "data_ext", "ct_bench", "Melamine_Dataset.mat"),
        squeeze_me=True,
        struct_as_record=False,
    )
    ds = m["Melamine_Dataset"]
    mel_data = {}
    for inst in ("R562", "R568", "R861", "R862"):
        s = getattr(ds, inst)
        X = np.asarray(s.X1, np.float64)
        y = np.asarray(s.Y, np.float64).ravel()
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        mel_data[inst] = (X[ok], y[ok])

    out = {}
    from collections import Counter
    for inst, (X_full, y_full) in mel_data.items():
        X_full = np.asarray(X_full, np.float64)
        y_full = np.asarray(y_full, np.float64).reshape(-1, 1)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y_full), size=min(300, len(y_full)), replace=False)
        X, Y = X_full[idx], y_full[idx]

        perm = np.random.default_rng(100).permutation(len(Y))
        sq = {"raw_pls": [], "raw_ridge": [], "fixed_sg1_11_d1_pls": [],
              "fixed_sg1_11_d1_ridge": [], "tuned_sg_all_ridge": [], "tuned_sg_deriv_only_ridge": []}
        cfgs_all, cfgs_d = [], []
        for va in np.array_split(perm, 5):
            tr = np.setdiff1d(perm, va)
            mu, sd = Y[tr].mean(0), np.clip(Y[tr].std(0), 1e-9, None)
            Ytr, Yva = (Y[tr] - mu) / sd, (Y[va] - mu) / sd

            p_rpls, _ = fit_pls(X[tr], Ytr, kmax=14, inner=5, seed=0)
            p_sg1pls, _ = fit_pls(_apply_sg(X[tr], 11, 3, 1), Ytr, kmax=14, inner=5, seed=0)
            p_rr, _, _ = _fit_exact_loo_ridge_np(X[tr], Ytr, X[va])
            p_sg1r, _, _ = _fit_exact_loo_ridge_np(_apply_sg(X[tr], 11, 3, 1), Ytr, _apply_sg(X[va], 11, 3, 1))
            p_tsg_all, ca = _fit_tuned_sg_ridge(X[tr], Ytr, X[va], grid=SG_GRID, pre_snv=False)
            p_tsg_d, cd = _fit_tuned_sg_ridge(X[tr], Ytr, X[va], grid=SG_DERIV_GRID, pre_snv=False)

            cfgs_all.append(str(ca))
            cfgs_d.append(str(cd))
            for k, P in [
                ("raw_pls", p_rpls(X[va]).reshape(Yva.shape)),
                ("raw_ridge", p_rr),
                ("fixed_sg1_11_d1_pls", p_sg1pls(_apply_sg(X[va], 11, 3, 1)).reshape(Yva.shape)),
                ("fixed_sg1_11_d1_ridge", p_sg1r),
                ("tuned_sg_all_ridge", p_tsg_all),
                ("tuned_sg_deriv_only_ridge", p_tsg_d),
            ]:
                sq[k].append((P - Yva) ** 2)

        vals = {k: round(float(np.sqrt(np.mean(np.concatenate(sq[k], 0)))), 4) for k in sq}
        out[inst] = {
            "rmsep_z": vals,
            "top_tuned_all": Counter(cfgs_all).most_common(2),
            "top_tuned_deriv_only": Counter(cfgs_d).most_common(2),
        }
        print(
            f"  {inst} | rawPLS={vals['raw_pls']:.4f} rawR={vals['raw_ridge']:.4f} | "
            f"fixSG1_PLS={vals['fixed_sg1_11_d1_pls']:.4f} fixSG1_R={vals['fixed_sg1_11_d1_ridge']:.4f} | "
            f"tSG(d=0..2)={vals['tuned_sg_all_ridge']:.4f} {out[inst]['top_tuned_all']} | "
            f"tSG(d=1..2)={vals['tuned_sg_deriv_only_ridge']:.4f} {out[inst]['top_tuned_deriv_only']}"
        )
    return out


def main():
    t0 = time.time()
    result = {
        "part1_tuned_sg_7datasets": part1_tuned_sg_and_physical_units(),
        "part2_tecator_ablation_and_phys": part2_tecator_and_corn_physical_units_and_ablation(),
        "part3_theorem1_confusion": part3_theorem1_confusion_matrix(),
        "part4_melamine_sg_decomp": part4_melamine_sg_decomposition(),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = os.path.join(ROOT, "_audit", "res_review_p0_2026_11_23.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nDone in {result['elapsed_sec']}s -> {out_path}")


if __name__ == "__main__":
    main()
