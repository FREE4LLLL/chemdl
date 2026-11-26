"""论文图：SNV 决策的 (sigma_a, r) 相图 —— 定理 1 解析边界 vs 受控注入实测。

两面板：
  A. (sigma_a, r) 平面上的实测相对增益 Delta% = (RMSEP_snv / RMSEP_raw - 1) * 100，
     填色热图 + 实测零增益等值线（黑实线），叠加定理 1 的解析临界曲线 r_crit(sigma_a)
     （白色虚线），并标注符号一致率。
  B. 定理 1 预测 Delta% vs 实测 Delta% 散点（按数据集着色），标 Pearson r 与 RMSE。

注入协议（与 _audit/selfcheck_external_2026_11_19.py::eval_fixed_var_snv_coupling 同构）：
  eps 严格正交于标准化浓度 ys 后单位化，z = r*ys + sqrt(1-r^2)*eps，a = 1 + sigma_a*z,
  Xi = a[:, None] * X0  =>  corr(a, ys) = r 精确、std(a) = sigma_a 精确。

用法：
  python paper_assets/plot_fig_snv_phase_diagram.py            # 跑实验并绘图
  python paper_assets/plot_fig_snv_phase_diagram.py --plot-only  # 复用已有 JSON 重绘
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from chemdl.cordoba import load_cordoba_all  # noqa: E402
from chemdl.theory import (  # noqa: E402
    estimate_physical_params,
    snv_critical_coupling,
    snv_theoretical_gain_pct,
)
from chemdl.transfer import PREPS, fit_pls, rmsep  # noqa: E402

DATA_EXT = os.path.join(ROOT, "data_ext")
OUT_PNG = os.path.join(HERE, "fig_snv_phase_diagram.png")
OUT_JSON = os.path.join(ROOT, "_audit", "res_snv_phase_diagram_2026_11_24.json")

SIGMA_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
R_GRID = tuple(round(0.1 * i, 1) for i in range(10))  # 0.0 .. 0.9
REPS = 3
FOLDS = 5
MIN_SCALE = 0.10  # 乘性倍率下界：a<0 会翻转谱符号，产生非物理的伪爆炸

NAVY = "#1E6091"
RED = "#B03A2E"


def _load_datasets() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """载入三个覆盖不同 (alpha, beta) 区域的公开库：低散射谷物 / 强散射固体 / 透明液体。"""
    from chemdl.diesel import load_diesel
    from chemdl.tecator import load_tecator

    out = []
    d = load_cordoba_all()["wheat"]
    out.append(("Cordoba wheat", np.asarray(d["X"], np.float64), np.asarray(d["y"], np.float64)))

    t = load_tecator()
    out.append(
        (
            "Tecator (fat)",
            np.asarray(t["X"], np.float64),
            np.asarray(t["y"], np.float64)[:, [1]],
        )
    )

    dz = load_diesel()
    cols = [str(c) for c in dz["cols"]]
    cn_idx = next((i for i, c in enumerate(cols) if c.strip().upper() == "CN"), 1)
    Xd = np.asarray(dz["X"], np.float64)
    yd = np.asarray(dz["y"], np.float64)[:, [cn_idx]]
    ok = np.isfinite(Xd).all(1) & np.isfinite(yd).all(1)
    Xd, yd = Xd[ok], yd[ok]
    idx = np.random.default_rng(7).choice(len(yd), min(200, len(yd)), replace=False)
    out.append(("SWRI Diesel CN", Xd[idx], yd[idx]))
    return out


def _inject(
    X0: np.ndarray, ys: np.ndarray, sigma_a: float, r: float, seed: int = 51
) -> tuple[np.ndarray, float]:
    """固定 std(a)=sigma_a、corr(a, y)=r 的乘性散射注入。

    乘性倍率必须为正（a<=0 会翻转谱符号并产生非物理的伪爆炸），因此对每个格点做拒绝重采样：
    重抽 eps、重新正交化，直到 min(a) >= MIN_SCALE。返回注入后矩阵与实际 min(a)。
    """
    n = len(ys)
    for k in range(400):
        rng = np.random.default_rng(seed + 1000 * k)
        eps = rng.normal(0.0, 1.0, size=n)
        eps = eps - (np.dot(eps, ys) / np.dot(ys, ys)) * ys
        eps = eps / eps.std()
        z = r * ys + math.sqrt(max(1.0 - r * r, 0.0)) * eps
        z = z / z.std()
        a = 1.0 + sigma_a * z
        if float(a.min()) >= MIN_SCALE:
            return a[:, None] * X0, float(a.min())
    # 无法满足正性约束时退回截断（记录在 JSON 的 min_scale_actual 中）
    a = np.clip(1.0 + sigma_a * z, MIN_SCALE, None)
    return a[:, None] * X0, float(a.min())


def _cv(Xin: np.ndarray, ys: np.ndarray, prep: str) -> float:
    fn = PREPS[prep]
    n = len(ys)
    vals = []
    for rep in range(REPS):
        perm = np.random.default_rng(900 + rep).permutation(n)
        for va in np.array_split(perm, FOLDS):
            tr = np.setdiff1d(perm, va)
            pred, _ = fit_pls(fn(Xin[tr]), ys[tr][:, None], kmax=10, inner=4, per_col=False)
            vals.append(rmsep(pred(fn(Xin[va])), ys[va][:, None]))
    return float(np.mean(vals))


def run_experiment() -> dict:
    t0 = time.time()
    records = []
    for name, X0, y0 in _load_datasets():
        if y0.ndim == 1:
            y0 = y0[:, None]
        ok = np.isfinite(X0).all(1) & np.isfinite(y0).all(1)
        X0, y0 = X0[ok], y0[ok]
        ys = (y0[:, 0] - y0[:, 0].mean()) / y0[:, 0].std()

        phys = estimate_physical_params(X0, y0)
        alpha = float(phys["alpha"])
        beta = float(phys["per_col"][0]["beta"])
        print(f"[{name}] n={len(ys)} L={X0.shape[1]} alpha={alpha:.2f} beta={beta:.2f}", flush=True)

        base_raw = _cv(X0, ys, "raw")
        base_snv = _cv(X0, ys, "snv")
        print(
            f"  baseline (no injection): raw={base_raw:.4f} snv={base_snv:.4f} "
            f"delta={(base_snv / base_raw - 1) * 100:+.2f}%",
            flush=True,
        )

        for sa in SIGMA_GRID:
            for r in R_GRID:
                Xi, min_a = _inject(X0, ys, sa, r)
                raw = _cv(Xi, ys, "raw")
                snv = _cv(Xi, ys, "snv")
                meas = (snv / raw - 1.0) * 100.0
                pred = float(
                    snv_theoretical_gain_pct(alpha, beta, r, sa, sigma_non=0.0)
                )
                records.append(
                    {
                        "dataset": name,
                        "sigma_a": round(sa, 3),
                        "r": round(r, 3),
                        "alpha": round(alpha, 3),
                        "beta": round(beta, 3),
                        "min_scale_actual": round(min_a, 4),
                        "rmsep_raw": round(raw, 6),
                        "rmsep_snv": round(snv, 6),
                        "measured_pct": round(meas, 3),
                        "predicted_pct": round(pred, 3),
                        "r_crit": round(snv_critical_coupling(alpha, beta, sa), 4),
                    }
                )
                print(
                    f"  sa={sa:.2f} r={r:.1f} min_a={min_a:.2f} "
                    f"meas={meas:+7.2f}% pred={pred:+7.2f}%",
                    flush=True,
                )

    # 汇总统计
    meas = np.array([d["measured_pct"] for d in records])
    pred = np.array([d["predicted_pct"] for d in records])
    sign_ok = float(np.mean(np.sign(meas) == np.sign(pred)))
    pearson = float(np.corrcoef(pred, meas)[0, 1])
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "sigma_grid": list(SIGMA_GRID),
            "r_grid": list(R_GRID),
            "reps": REPS,
            "folds": FOLDS,
            "injection": "a = 1 + sigma_a * z, corr(z, ys)=r exact, std(z)=1; Xi = a[:,None]*X0",
            "estimator": "Alpha/Beta from estimate_physical_params on the uninjected data",
            "caveat": "alpha/beta are estimated once per dataset from the uninjected spectra; "
                      "the analytic curve is therefore a theory prediction, not a fit to the injected grid",
        },
        "summary": {
            "n_cells": len(records),
            "sign_agreement": round(sign_ok, 4),
            "pearson_pred_vs_measured": round(pearson, 4),
            "rmse_pct": round(float(np.sqrt(np.mean((pred - meas) ** 2))), 3),
        },
        "records": records,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(
        f"\nsaved {OUT_JSON}\ncells={len(records)} sign_agreement={sign_ok:.3f} "
        f"pearson={pearson:.4f} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return payload


def plot(payload: dict) -> None:
    records = payload["records"]
    datasets = sorted({d["dataset"] for d in records})

    plt.rcParams.update(
        {
            "font.size": 10.2,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.2,
            "legend.fontsize": 8.4,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    from matplotlib.lines import Line2D  # noqa: E402
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.8))
    xs = np.array(SIGMA_GRID)
    ys = np.array(R_GRID)

    # ---------- Panel (a)..(c): 逐库相图（实测零线与该库解析曲线可直接比对） ----------
    panel_tags = ["(a)", "(b)", "(c)"]
    for k, name in enumerate(datasets):
        ax = axes[k]
        sub = [d for d in records if d["dataset"] == name]
        zmat = np.full((len(R_GRID), len(SIGMA_GRID)), np.nan)
        for i, r in enumerate(R_GRID):
            for j, sa in enumerate(SIGMA_GRID):
                vals = [d["measured_pct"] for d in sub if d["r"] == r and d["sigma_a"] == sa]
                if vals:
                    zmat[i, j] = float(vals[0])

        vmax = float(np.nanmax(np.abs(zmat))) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        mesh = ax.pcolormesh(xs, ys, zmat, cmap="RdBu_r", norm=norm, shading="nearest")

        # 实测零增益等值线（移除旋转 clabel，改入图例避免文字遮挡）
        drew = bool(float(np.nanmin(zmat)) <= 0.0 <= float(np.nanmax(zmat)))
        if drew:
            ax.contour(xs, ys, zmat, levels=[0.0], colors="k", linewidths=1.8)

        # 该库自己的定理 1 临界曲线
        alpha, beta = sub[0]["alpha"], sub[0]["beta"]
        rcrit = [min(0.99, snv_critical_coupling(alpha, beta, float(s), clip=False)) for s in SIGMA_GRID]
        cap_tag = " [cap]" if alpha >= 199.9 else ""
        ax.plot(xs, rcrit, ls="--", lw=1.6, color="#111111",
                label=f"Theory $r_{{\\mathrm{{crit}}}}$ ($\\alpha$={alpha:.0f}{cap_tag}, $\\beta$={beta:.1f})")

        handles, labels = ax.get_legend_handles_labels()
        if drew:
            handles.insert(0, Line2D([0], [0], color="k", lw=1.8))
            labels.insert(0, "Measured $\\Delta = 0$")
        ax.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=8.4,
        )

        ax.set_title(f"{panel_tags[k]} {name}")
        ax.set_xlabel("$\\sigma_a$")
        if k == 0:
            ax.set_ylabel("coupling  $r$")
        ax.set_ylim(0.0, 1.04)
        cb = fig.colorbar(mesh, ax=ax, pad=0.025)
        cb.set_label("$\\Delta$ (%)", fontsize=8.8)
        if not drew:
            ax.text(0.5, 0.99, "$r_{\\mathrm{crit}} > 1$ (all cells beneficial)", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8.0, color="#111111",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CCCCCC", alpha=0.92))

    # ---------- Panel (d): 预测 vs 实测 ----------
    axB = axes[3]
    palette = {datasets[0]: NAVY, datasets[1]: "#E67E22", datasets[2]: "#27AE60"}
    for name in datasets:
        sub = [d for d in records if d["dataset"] == name]
        axB.scatter(
            [d["predicted_pct"] for d in sub],
            [d["measured_pct"] for d in sub],
            s=18, alpha=0.8, edgecolor="white", linewidth=0.3,
            color=palette.get(name, "0.4"), label=name,
        )
    lim = max(
        max(abs(d["predicted_pct"]) for d in records),
        max(abs(d["measured_pct"]) for d in records),
    ) * 1.08
    axB.plot([-lim, lim], [-lim, lim], ls=":", lw=1.0, color="0.45", label="$y=x$")
    axB.axhline(0.0, lw=0.7, color="0.6")
    axB.axvline(0.0, lw=0.7, color="0.6")
    axB.set_xlim(-lim, lim)
    axB.set_ylim(-lim, lim)
    axB.set_xlabel("Theorem 1 prediction  $\\Delta$ (%)")
    axB.set_ylabel("Measured  $\\Delta$ (%)")
    s = payload["summary"]
    axB.set_title(
        f"(d) Pooled parity ($r$={s['pearson_pred_vs_measured']:.3f}, "
        f"sign {s['sign_agreement'] * 100:.1f}%)"
    )
    axB.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"saved {OUT_PNG}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="reuse existing JSON, skip the simulation")
    args = ap.parse_args()
    if args.plot_only:
        with open(OUT_JSON, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = run_experiment()
    plot(payload)


if __name__ == "__main__":
    main()
