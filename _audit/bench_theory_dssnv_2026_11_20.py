"""验证解析最优散射校正定理（chemdl/theory.py）：
Part 1: 在 C-061 固定方差 std(a)=0.35 耦合曲线（Wheat / Barley / Diesel）上对比定理 1 预测与实测变号曲线；
Part 2: 在 7 大真实公开 NIR 库（3×5 严格嵌套 CV）上对比 零网格搜索的闭式解析 DSSNV (fit_analytical_dssnv_pls)
        vs Raw vs 经典 SNV vs 内层网格 CV-DSSNV (fit_dssnv_pls)。
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.io
from scipy import stats

from chemdl.cordoba import load_cordoba_all
from chemdl.corn import load_corn
from chemdl.tablets import load_tablets
from chemdl.tecator import load_tecator
from chemdl.theory import (
    closed_form_optimal_ws,
    fit_analytical_dssnv_pls,
    snv_critical_coupling,
    snv_theoretical_gain_pct,
)
from chemdl.transfer import PREPS, fit_dssnv_pls, fit_pls, rmsep


def load_diesel():
    m = scipy.io.loadmat(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_ext", "diesel", "SWRI_Diesel_NIR.mat"))
    X = np.asarray(m["diesel_spec"][0, 0]["data"], np.float64)
    y = np.asarray(m["diesel_prop"][0, 0]["data"], np.float64)[:, [1]]
    ok = np.isfinite(X).all(1) & np.isfinite(y).all(1)
    return X[ok][:200], y[ok][:200]


def verify_c061_theorem():
    ext = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_audit", "res_selfcheck_external_2026_11_19.json")))["fixed_var_snv_coupling"]
    sig_a = 0.35
    alpha = 40.0
    out = {}
    print("=== Part 1: Theorem 1 & 2 vs C-061 Fixed-Variance Coupling Curves ===", flush=True)
    for key, lbl in [("cordoba_wheat", "Cordoba_Wheat"), ("cordoba_barley", "Cordoba_Barley"), ("swri_diesel_CN", "SWRI_Diesel_CN")]:
        d = ext[key]
        rs = [0.0, 0.2, 0.4, 0.6, 0.8]
        emp = [float(d[f"r={rv:.1f}"]["gain_pct"]) for rv in rs]
        rm_snv0 = float(d["r=0.0"]["snv"])
        beta = max(0.5, (1.0 / max(rm_snv0 ** 2, 1e-4)) - 1.0)
        theo = [round(snv_theoretical_gain_pct(alpha, beta, rv, sig_a), 2) for rv in rs]
        ws_opt = [round(closed_form_optimal_ws(alpha, beta, rv, sig_a), 3) for rv in rs]
        r_crit = round(snv_critical_coupling(alpha, beta, sig_a), 3)
        corr = round(float(np.corrcoef(emp, theo)[0, 1]), 4)
        sign_match = int(sum((e >= 0) == (t >= 0) for e, t in zip(emp, theo)))
        out[lbl] = {
            "beta_est": round(beta, 2),
            "r_crit": r_crit,
            "r_grid": rs,
            "empirical_gain_pct": [round(x, 2) for x in emp],
            "theoretical_gain_pct": theo,
            "closed_form_ws_star": ws_opt,
            "pearson_r": corr,
            "sign_agreement_at_r0_and_r08": bool((emp[0] < 0) == (theo[0] < 0) and (emp[-1] > 0) == (theo[-1] > 0)),
        }
        print(f"  {lbl:15s}: beta={beta:4.1f}, r_crit={r_crit:.3f} | pearson_r={corr:.4f} | sign(r=0,r=0.8) match={out[lbl]['sign_agreement_at_r0_and_r08']} | ws*={ws_opt}", flush=True)
    return out


def verify_7_real_datasets():
    t = load_tecator()
    c = load_corn()
    tab = load_tablets()
    cb = load_cordoba_all()
    Xd, Yd = load_diesel()
    datasets = [
        ("Tecator", t["X"], t["y"]),
        ("Corn_m5", c["m5"], c["y"]),
        ("Tablets_X1", tab["X1"][tab["part"] == 0], tab["y"][tab["part"] == 0]),
        ("Cordoba_wheat", cb["wheat"]["X"], cb["wheat"]["y"]),
        ("Cordoba_barley", cb["barley"]["X"], cb["barley"]["y"]),
        ("Cordoba_maize", cb["maize"]["X"], cb["maize"]["y"]),
        ("SWRI_Diesel_CN", Xd, Yd),
    ]
    print("\n=== Part 2: Zero-CV Closed-Form Analytical DSSNV vs Raw / SNV / CV-DSSNV (3×5 CV) ===", flush=True)
    out = {}
    for name, X, Y in datasets:
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        errs = {k: [] for k in ("raw", "snv", "cv_dssnv", "theory_dssnv")}
        oof = {k: [] for k in errs}
        ws_list, wc_list = [], []
        t_cv, t_th = 0.0, 0.0
        for rep in range(3):
            perm = np.random.default_rng(100 + rep).permutation(len(Y))
            for va in np.array_split(perm, 5):
                tr = np.setdiff1d(perm, va)
                mu, sd = Y[tr].mean(0), np.clip(Y[tr].std(0), 1e-9, None)
                Ytr, Yva = (Y[tr] - mu) / sd, (Y[va] - mu) / sd

                for pk in ("raw", "snv"):
                    fn = PREPS[pk]
                    p, _ = fit_pls(fn(X[tr]), Ytr, per_col=True)
                    yh = p(fn(X[va]))
                    errs[pk].append(float(rmsep(yh, Yva)))
                    oof[pk].append(np.mean((yh - Yva) ** 2, axis=1))

                ts = time.time()
                p_cv, _ = fit_dssnv_pls(X[tr], Ytr, seed=rep)
                t_cv += time.time() - ts
                yh_cv = p_cv(X[va])
                errs["cv_dssnv"].append(float(rmsep(yh_cv, Yva)))
                oof["cv_dssnv"].append(np.mean((yh_cv - Yva) ** 2, axis=1))

                ts = time.time()
                p_th, info_th = fit_analytical_dssnv_pls(X[tr], Ytr, seed=rep)
                t_th += time.time() - ts
                yh_th = p_th(X[va])
                errs["theory_dssnv"].append(float(rmsep(yh_th, Yva)))
                oof["theory_dssnv"].append(np.mean((yh_th - Yva) ** 2, axis=1))
                ws_list.append(float(np.mean([cfg[1] for cfg in info_th["cfgs"]])))
                wc_list.append(float(np.mean([cfg[0] for cfg in info_th["cfgs"]])))

        m = {k: round(float(np.mean(v)), 4) for k, v in errs.items()}
        e_th = np.concatenate(oof["theory_dssnv"])
        p_vs_snv = round(float(stats.wilcoxon(e_th, np.concatenate(oof["snv"])).pvalue), 5)
        p_vs_raw = round(float(stats.wilcoxon(e_th, np.concatenate(oof["raw"])).pvalue), 5)
        out[name] = {
            "rmsep": m,
            "vs_raw_pct": round((m["theory_dssnv"] / m["raw"] - 1.0) * 100.0, 2),
            "vs_snv_pct": round((m["theory_dssnv"] / m["snv"] - 1.0) * 100.0, 2),
            "vs_cv_dssnv_pct": round((m["theory_dssnv"] / m["cv_dssnv"] - 1.0) * 100.0, 2),
            "p_oof_vs_snv": p_vs_snv,
            "p_oof_vs_raw": p_vs_raw,
            "mean_wc_star": round(float(np.mean(wc_list)), 3),
            "mean_ws_star": round(float(np.mean(ws_list)), 3),
            "speedup_vs_cv": round(t_cv / max(t_th, 1e-6), 2),
        }
        print(
            f"  {name:15s}: raw={m['raw']:.4f} snv={m['snv']:.4f} cv_dssnv={m['cv_dssnv']:.4f} "
            f"THEORY={m['theory_dssnv']:.4f} (vs_snv={out[name]['vs_snv_pct']:+.1f}%, p={p_vs_snv:.4f}; "
            f"wc*={out[name]['mean_wc_star']:.2f}, ws*={out[name]['mean_ws_star']:.2f}, {out[name]['speedup_vs_cv']:.1f}x faster)",
            flush=True,
        )
    return out


def main():
    t0 = time.time()
    res = {
        "c061_theorem_verification": verify_c061_theorem(),
        "real_7_datasets": verify_7_real_datasets(),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_audit", "res_theory_dssnv_2026_11_20.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {p} ({res['elapsed_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
