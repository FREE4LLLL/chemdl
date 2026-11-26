"""从最新 `C-144` 修复后的 `_audit/res_industrial_vs_dl_2026_11_21.json` 零手抄重绘
`paper_assets/fig11_industrial_vs_dl_benchmark.png` 与 `paper_assets/fig12_industrial_scaling_and_cost.png`（300 DPI）。
"""
from __future__ import annotations

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IND_JSON = os.path.join(ROOT, "_audit", "res_industrial_vs_dl_2026_11_21.json")
OUT_FIG11 = os.path.join(ROOT, "paper_assets", "fig11_industrial_vs_dl_benchmark.png")
OUT_FIG12 = os.path.join(ROOT, "paper_assets", "fig12_industrial_scaling_and_cost.png")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def plot_fig11(ind: dict):
    label_map = [
        ("Mango_DM_Seasonal", "Mango Seasonal"),
        ("CRAW_Wheat_Temporal", "CRA-W 27-yr OOD"),
        ("OSSL_Soil_MEMS", "OSSL Soil GroupSplit"),
        ("Milk_Transmittance_Sess", "Raw Milk Transmittance"),
        ("Melamine_InDomain_R861", "Melamine Same-Recipe (R861)"),
        ("Melamine_CrossInst_562_861", "Melamine Cross-Recipe (R562->R861)"),
    ]
    task_order = [
        (k, f"{lb}\n({ind['tasks'][k]['n_tr']:,} → {ind['tasks'][k]['n_te']:,}, L={ind['tasks'][k]['L']})")
        for k, lb in label_map
    ]
    show_methods = [
        ("Raw_PLS", "Raw-PLS", "#9e9e9e"),
        ("SNV_PLS", "SNV-PLS", "#607d8b"),
        ("DL_1D_ResNet", "1D-ResNet", "#ef9a9a"),
        ("DL_DeepSpectra", "DeepSpectra", "#e57373"),
        ("DL_ResMLP", "ResMLP", "#f44336"),
        ("ChemDL_MK_Bilevel", "MK-Bilevel (Linear)", "#1e88e5"),
        ("ChemDL_DiffPrep_CNN", "DiffPrep+CNN", "#43a047"),
        ("ChemDL_MK_ResNet", "MK+ResNet (Hybrid)", "#0d47a1"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), dpi=300)
    axes = axes.ravel()
    for ax, (tkey, title) in zip(axes, task_order):
        ms = ind["tasks"][tkey]["methods"]
        vals = [ms[k]["std_pooled_rmsep"] for k, _, _ in show_methods]
        labels = [lb for _, lb, _ in show_methods]
        colors = [cl for _, _, cl in show_methods]
        y_pos = np.arange(len(show_methods))
        bars = ax.barh(y_pos, vals, color=colors, edgecolor="#222222", linewidth=0.5, height=0.72)
        best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor("#ff9800")
        bars[best_idx].set_linewidth(2.0)

        xmax = max(vals) * 1.24
        ax.set_xlim(0, xmax)
        for i, v in enumerate(vals):
            star = " *" if i == best_idx else ""
            ax.text(v + xmax * 0.015, i, f"{v:.4f}{star}", va="center", fontsize=8.5,
                    fontweight="bold" if i == best_idx else "normal")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Standardized Pooled RMSEP (lower is better)", fontsize=8.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold")

    fig.suptitle(
        "Industrial-Scale Benchmark (5 Databases, 34,411 Spectra): ChemDL vs End-to-End Deep Learning",
        fontsize=12.5, fontweight="bold", y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG11, dpi=300)
    plt.close(fig)
    print(f"Saved -> {OUT_FIG11}")


def plot_fig12(ind: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), dpi=300)

    # Panel A: Mango scaling curve
    ax = axes[0]
    m_sc = ind["scaling_curves"]["Mango_DM_Seasonal"]
    ns_m = sorted(int(k) for k in m_sc.keys())
    series = [
        ("Raw_PLS", "Raw-PLS", "#78909c", "--", "o"),
        ("DL_DeepSpectra", "DeepSpectra (DL)", "#e53935", "-", "s"),
        ("DL_1D_ResNet", "1D-ResNet (DL)", "#fb8c00", ":", "^"),
        ("ChemDL_MK_Bilevel", "MK-Bilevel (Linear)", "#1e88e5", "-", "D"),
        ("ChemDL_MK_ResNet", "MK+ResNet (Hybrid)", "#0d47a1", "-", "P"),
    ]
    for k, lb, cl, ls, mk in series:
        ys = [m_sc[str(n)][k] for n in ns_m]
        ax.plot(ns_m, ys, label=lb, color=cl, linestyle=ls, marker=mk, linewidth=2.0, markersize=6)
    ax.set_xscale("log")
    ax.set_xticks(ns_m)
    ax.set_xticklabels([str(n) for n in ns_m], fontsize=8.5)
    ax.set_xlabel("Calibration Set Size n (log scale)", fontsize=9.5)
    ax.set_ylabel("Dry Matter RMSEP (%)", fontsize=9.5)
    ax.set_title("A. Mango Seasonal (L=103): DL Crossover n* ~ 6k-10k", fontsize=10.0, fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper right", frameon=True)

    # Panel B: CRAW 27-yr OOD scaling curve
    ax = axes[1]
    c_sc = ind["scaling_curves"]["CRAW_Wheat_Temporal"]
    ns_c = sorted(int(k) for k in c_sc.keys())
    for k, lb, cl, ls, mk in [
        ("SNV_PLS", "SNV-PLS", "#607d8b", "--", "o"),
        ("DL_ResMLP", "ResMLP (Best Pure DL)", "#e53935", "-", "s"),
        ("DL_1D_ResNet", "1D-ResNet (DL)", "#fb8c00", ":", "^"),
        ("ChemDL_MK_Bilevel", "MK-Bilevel (Linear)", "#1e88e5", "-", "D"),
        ("ChemDL_MK_ResNet", "MK+ResNet (Hybrid)", "#0d47a1", "-", "P"),
    ]:
        ys = [c_sc[str(n)][k] for n in ns_c]
        ax.plot(ns_c, ys, label=lb, color=cl, linestyle=ls, marker=mk, linewidth=2.0, markersize=6)
    ax.set_xscale("log")
    ax.set_xticks(ns_c)
    ax.set_xticklabels([str(n) for n in ns_c], fontsize=8.5)
    ax.set_xlabel("Calibration Set Size n (log scale)", fontsize=9.5)
    ax.set_ylabel("Wheat Protein RMSEP (%)", fontsize=9.5)
    ax.set_title("B. CRA-W 27-yr OOD (L=700): No DL Crossover", fontsize=10.0, fontweight="bold")
    ax.legend(fontsize=7.8, loc="center left", frameon=True)

    # Panel C: Mean normalized RMSEP vs Mean CPU Training Time (Pareto Frontier across 6 tasks)
    ax = axes[2]
    tkeys = list(ind["tasks"].keys())
    m_meta = [
        ("Raw_PLS", "Raw-PLS", "#9e9e9e", "o", (1.12, 0.002)),
        ("SNV_PLS", "SNV-PLS", "#607d8b", "o", (1.12, -0.004)),
        ("SG1_PLS", "SG1-PLS", "#b0bec5", "o", (1.12, 0.008)),
        ("ChemDL_Bilevel", "Bilevel", "#42a5f5", "D", (0.82, 0.014)),
        ("ChemDL_MK_Bilevel", "MK-Bilevel", "#1e88e5", "D", (1.14, 0.002)),
        ("ChemDL_MK_ResNet", "MK+ResNet", "#0d47a1", "P", (1.12, 0.002)),
        ("ChemDL_DiffPrep_CNN", "DiffPrep+CNN", "#43a047", "P", (0.72, -0.025)),
        ("DL_ResMLP", "ResMLP", "#ef5350", "s", (1.12, 0.003)),
        ("DL_1D_ResNet", "1D-ResNet", "#e53935", "s", (1.08, 0.015)),
        ("DL_DeepSpectra", "DeepSpectra", "#c62828", "s", (0.80, 0.020)),
        ("DL_SpectraFormer", "SpectraFormer", "#8e24aa", "^", (0.58, -0.032)),
    ]
    xs_all, ys_all = [], []
    for mk, lb, cl, marker, (x_mul, dy) in m_meta:
        mean_err = float(np.mean([ind["tasks"][t]["methods"][mk]["std_pooled_rmsep"] for t in tkeys]))
        mean_sec = float(np.mean([ind["tasks"][t]["methods"][mk]["train_sec"] for t in tkeys]))
        xs_all.append(mean_sec)
        ys_all.append(mean_err)
        ax.scatter(mean_sec, mean_err, color=cl, marker=marker, s=85, edgecolor="#222222", linewidth=0.6, zorder=4)
        ax.text(mean_sec * x_mul, mean_err + dy, lb, fontsize=7.8,
                fontweight="bold" if mk.startswith("ChemDL_MK") else "normal")
    ax.set_xscale("log")
    ax.set_xlim(min(xs_all) * 0.65, max(xs_all) * 1.45)
    ax.set_ylim(min(ys_all) - 0.035, max(ys_all) + 0.05)
    ax.set_xlabel("Mean CPU Training Time per Task (s, log scale)", fontsize=9.5)
    ax.set_ylabel("6-Task Mean std_RMSEP (lower is better)", fontsize=9.5)
    ax.set_title("C. Accuracy-Cost Pareto Frontier (6 Industrial Tasks)", fontsize=10.0, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_FIG12, dpi=300)
    plt.close(fig)
    print(f"Saved -> {OUT_FIG12}")


def main():
    ind = json.load(open(IND_JSON, encoding="utf-8"))
    plot_fig11(ind)
    plot_fig12(ind)


if __name__ == "__main__":
    main()
