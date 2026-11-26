"""Generate Fig 13: Empirical Identifiability & Physical Boundary of 1 - R^2_shape (C-150)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_PATH = ROOT / "_audit" / "res_r2_shape_identifiability_2026_11_25.json"
OUT_PATH = ROOT / "paper_assets" / "fig13_r2_shape_identifiability.png"


def main() -> None:
    data = json.loads(RES_PATH.read_text(encoding="utf-8"))
    a1 = data["part_a1_scalar_sweep"]
    a2 = data["part_a2_tilt_negative_control"]
    pb = data["part_b_real_paired_replicates"]

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=300)

    # Panel (a): Pure scalar multiplicative scatter sweep across 5 real chemistries
    ax = axes[0]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]
    markers = ["o", "s", "^", "D", "v"]
    short_names = {
        "Corn_m5": "Corn m5",
        "SWRI_Diesel": "SWRI Diesel",
        "Tablets_X1_cal": "Tablets X1",
        "Cordoba_Wheat_1998": "Cordoba Wheat",
        "Wheat_A1_cal": "Wheat A1",
    }
    ax.plot([0, 1], [0, 1], "k--", lw=1.1, alpha=0.65, label="Identity ($y = x$)")
    for idx, (k, v) in enumerate(a1["datasets"].items()):
        xs = [lv["f_adj_pop_lin_mean"] for lv in v["levels"]]
        ys = [lv["one_minus_r2_raw_mean"] for lv in v["levels"]]
        ax.plot(
            xs,
            ys,
            marker=markers[idx],
            color=colors[idx],
            lw=1.4,
            ms=4.8,
            alpha=0.9,
            label=short_names.get(k, k),
        )
    m_fa = a1["pooled_metrics_raw_vs_f_adj_pop_lin"]
    m_ft = a1["pooled_metrics_raw_vs_f_pop_lin"]
    ax.text(
        0.04,
        0.84,
        f"vs $f_{{\\mathrm{{adj,lin}}}}$: $r={m_fa['pearson_r']:.3f}$, MAE$={m_fa['mae']:.3f}$\n"
        f"vs $f_{{\\mathrm{{pop,lin}}}}$: $r={m_ft['pearson_r']:.3f}$, MAE$={m_ft['mae']:.3f}$",
        transform=ax.transAxes,
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Same-Scale Population Adjusted Scatter Share ($f_{\\mathrm{adj,lin}}$)")
    ax.set_ylabel("Estimated $1 - R^2_{\\mathrm{shape}}$ (raw)")
    ax.set_title("(a) Fixed-Chemistry Scalar Scatter Sweep")
    ax.legend(loc="lower right", frameon=True, framealpha=0.85)

    # Panel (b): Adversarial wavelength-tilt negative control
    ax = axes[1]
    gammas = a2["gamma_levels"]
    for idx, (k, v) in enumerate(a2["datasets"].items()):
        ys = [lv["one_minus_r2_raw_mean"] for lv in v["levels"]]
        ax.plot(
            gammas,
            ys,
            marker=markers[idx],
            color=colors[idx],
            lw=1.2,
            ms=4.2,
            alpha=0.65,
            label=short_names.get(k, k),
        )
    pooled_g = [a2["pooled_by_gamma"][f"{g:.2f}"]["one_minus_r2_raw_mean"] for g in gammas]
    pooled_ft = [a2["pooled_by_gamma"][f"{g:.2f}"]["f_pop_log_mean"] for g in gammas]
    ax.plot(gammas, pooled_g, "k-o", lw=2.3, ms=6.0, label="Pooled $1 - R^2_{\\mathrm{shape}}$")
    ax.plot(gammas, pooled_ft, "k:", lw=1.6, label="Pooled $f_{\\mathrm{pop}}$ (log-std)")
    ax.set_xticks(gammas)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Wavelength-Tilt Coupling Strength ($\\gamma$)")
    ax.set_ylabel("Estimated $1 - R^2_{\\mathrm{shape}}$ (raw)")
    ax.set_title("(b) Adversarial Wavelength-Tilt Control ($\\sigma_{\\mathrm{inj}} = 0.10$)")
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.44), frameon=True, framealpha=0.90, fontsize=7.4)

    # Panel (c): Real unmodified paired/replicate NIR spectra
    ax = axes[2]
    ax.plot([0, 1], [0, 1], "k--", lw=1.1, alpha=0.65, label="$y = x$")
    short_b = {
        "Tablets_X1_vs_X2": "Tablets",
        "Corn_m5_vs_mp5": "Corn mp5",
        "Corn_m5_vs_mp6": "Corn mp6",
        "Wheat_IDRC16_A_vs_B": "Wheat A/B",
        "Wheat_IDRC16_A_vs_C": "Wheat A/C",
        "SDOP_spec1_vs_spec2": "SDOP (liq)",
        "OSSL_Soil_Repacked": "OSSL (cross-dev)",
        "PeaForage_2019GN_Rep": "PeaForage",
    }
    label_pos = {
        "OSSL_Soil_Repacked": (0.06, 0.365),
        "Wheat_IDRC16_A_vs_B": (0.27, 0.215),
        "Wheat_IDRC16_A_vs_C": (0.27, 0.165),
        "Corn_m5_vs_mp5": (0.25, 0.118),
        "Corn_m5_vs_mp6": (0.25, 0.074),
        "Tablets_X1_vs_X2": (0.14, 0.016),
        "SDOP_spec1_vs_spec2": (0.40, 0.046),
        "PeaForage_2019GN_Rep": (0.66, 0.445),
    }
    seen_solid = False
    seen_liq = False
    for row in pb:
        fx = row.get("f_within_paired_centered_denom", row.get("f_within_device_centered", row["f_within_log_std"]))
        fy = row["one_minus_r2_raw_x1"]
        is_liq = row["category"] == "liquid_small_n"
        c = "#d62728" if is_liq else "#1f77b4"
        mk = "X" if is_liq else "o"
        lbl = None
        if is_liq and not seen_liq:
            lbl = "Liquid cross-inst (SDOP)"
            seen_liq = True
        elif not is_liq and not seen_solid:
            lbl = "Solid / Canopy pairs"
            seen_solid = True
        ax.scatter(fx, fy, color=c, marker=mk, s=55, zorder=3, label=lbl)
        tx, ty = label_pos.get(row["dataset"], (fx + 0.03, fy + 0.02))
        ax.annotate(
            short_b.get(row["dataset"], row["dataset"]),
            (fx, fy),
            xytext=(tx, ty),
            fontsize=7.7,
            va="center",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=1.0),
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.75),
        )

    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.03, 0.55)
    ax.set_xlabel("Within-Sample Fraction (Centered Paired / ANOVA)")
    ax.set_ylabel("Single-Scan $1 - R^2_{\\mathrm{shape}}$ (raw)")
    ax.set_title("(c) Real Paired/Replicate NIR (No Injection)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.90, fontsize=7.5)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
