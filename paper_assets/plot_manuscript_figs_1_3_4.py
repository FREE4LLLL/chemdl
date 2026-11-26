"""Generate Manuscript Figures 1, 3, and 4 at 300 DPI from verified audit JSONs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "_audit"
OUT = ROOT / "paper_assets"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.2,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


def _draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    fc: str = "#FFFFFF",
    ec: str = "#2C3E50",
    lw: float = 1.4,
    pad: float = 0.008,
    title_color: str = "#1A252F",
    title_size: float = 9.0,
    body_size: float = 7.6,
) -> tuple[float, float, float, float]:
    """Draw a rounded box and return its exact outer padded edges (left, bottom, right, top)."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(box)
    cx = x + w / 2.0
    if body:
        n_lines = body.count("\n") + 1
        title_y = y + h * (0.72 if n_lines >= 2 else 0.67)
        body_y = y + h * (0.31 if n_lines >= 2 else 0.29)
        ax.text(
            cx,
            title_y,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            zorder=4,
        )
        ax.text(
            cx,
            body_y,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color="#34495E",
            linespacing=1.25,
            zorder=4,
        )
    else:
        ax.text(
            cx,
            y + h / 2.0,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            zorder=4,
        )
    return (x - pad, y - pad, x + w + pad, y + h + pad)


def _draw_ortho(
    ax,
    pts: list[tuple[float, float]],
    color: str = "#2C3E50",
    lw: float = 1.6,
    ls: str = "-",
) -> None:
    """Draw strictly orthogonal Manhattan polyline ending with a crisp visible arrowhead."""
    if len(pts) >= 3:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, solid_capstyle="round", zorder=5)
    p0, p1 = pts[-2], pts[-1]
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>,head_length=6.8,head_width=4.0",
        color=color,
        linewidth=lw,
        linestyle=ls,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=6,
    )
    ax.add_patch(arr)


def make_fig_ms1_framework() -> Path:
    """Fig 1: Planar, zero-crossing 3-column orthogonal schematic of the framework."""
    fig, ax = plt.subplots(figsize=(14.4, 5.7), dpi=300)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Stage backdrop columns (leaving clean gaps & bottom gutter y=0.02..0.09)
    stage_specs = [
        (0.015, 0.115, 0.300, 0.845, "#EEF5FB", "#B7D1EA", "Stage I: Calibration-Only Physical Attribution"),
        (0.345, 0.115, 0.300, 0.845, "#F4F8F2", "#BDD7B6", "Stage II: Paired Exact Allen-PRESS LOO Test"),
        (0.695, 0.115, 0.290, 0.845, "#FDF7EE", "#E6C9A2", "Stage III: Three-State Gate & Deployment"),
    ]
    for sx, sy, sw, sh, fc, ec, lbl in stage_specs:
        panel = FancyBboxPatch(
            (sx, sy),
            sw,
            sh,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.1,
            linestyle="--",
            zorder=1,
        )
        ax.add_patch(panel)
        ax.text(sx + 0.012, sy + sh - 0.014, lbl, fontsize=8.5, fontweight="bold", color="#2C3E50", ha="left", va="top", zorder=2)

    # ---------------- Column 1: Stage I ----------------
    b_cal = _draw_box(
        ax, 0.042, 0.725, 0.246, 0.145,
        "Calibration Set $(X_{\\mathrm{cal}}, y_{\\mathrm{cal}})$",
        "Strictly inductive (no target $X_T$)",
        fc="#FFFFFF", ec="#1F4E79", lw=1.7, title_size=9.1, body_size=7.6,
    )
    b_shape = _draw_box(
        ax, 0.042, 0.455, 0.246, 0.160,
        "Shape Decomposition",
        "3-fold CV PLS ($X_{\\mathrm{SNV}} \\to s_i$)\n$1 - R^2_{\\mathrm{shape}} \\in [0.02, 1.0]$",
        fc="#FFFFFF", ec="#2874A6", title_size=9.0, body_size=7.6,
    )
    b_sign = _draw_box(
        ax, 0.030, 0.155, 0.126, 0.180,
        "Analytic Criterion",
        "$r^2 < r_{\\mathrm{crit}}^2$\n(Thm 1 Attribution)",
        fc="#FFFFFF", ec="#2874A6", title_size=8.4, body_size=7.2,
    )
    b_wiener = _draw_box(
        ax, 0.174, 0.155, 0.128, 0.180,
        "Wiener Shrinkage",
        "$w_{s,\\mathrm{Wiener}}^\\star = 1 - R^2_{\\mathrm{shape}}$\n(Thms 3–4)",
        fc="#FFFFFF", ec="#2874A6", title_size=8.4, body_size=7.2,
    )

    # Stage I internal arrows
    cal_cx = 0.5 * (b_cal[0] + b_cal[2])
    _draw_ortho(ax, [(cal_cx, b_cal[1]), (cal_cx, b_shape[3])], color="#2874A6")
    y_split1 = 0.395
    _draw_ortho(
        ax,
        [(cal_cx, b_shape[1]), (cal_cx, y_split1), (0.5 * (b_sign[0] + b_sign[2]), y_split1), (0.5 * (b_sign[0] + b_sign[2]), b_sign[3])],
        color="#2874A6",
    )
    _draw_ortho(
        ax,
        [(cal_cx, b_shape[1]), (cal_cx, y_split1), (0.5 * (b_wiener[0] + b_wiener[2]), y_split1), (0.5 * (b_wiener[0] + b_wiener[2]), b_wiener[3])],
        color="#2874A6",
    )

    # ---------------- Column 2: Stage II ----------------
    b_raw = _draw_box(
        ax, 0.365, 0.560, 0.120, 0.155,
        "Raw Branch",
        "Exact Allen-PRESS\n(or Group-CV)",
        fc="#FFFFFF", ec="#2E7D32", title_size=8.6, body_size=7.3,
    )
    b_snv = _draw_box(
        ax, 0.505, 0.560, 0.120, 0.155,
        "SNV Branch",
        "Exact Allen-PRESS\n(or Group-CV)",
        fc="#FFFFFF", ec="#2E7D32", title_size=8.6, body_size=7.3,
    )
    b_loo = _draw_box(
        ax, 0.368, 0.200, 0.254, 0.205,
        "Paired LOO Score",
        "$d_i = (e_i^{\\mathrm{snv}})^2 - (e_i^{\\mathrm{raw}})^2$\nCompute $t_{\\mathrm{LOO}}$ and $\\Delta_{\\mathrm{LOO}}$ (%)",
        fc="#FFFFFF", ec="#2E7D32", title_size=9.0, body_size=7.4,
    )

    # Calibration Set -> Stage II top bus (y = cal_cy)
    cal_cy = 0.5 * (b_cal[1] + b_cal[3])
    raw_cx = 0.5 * (b_raw[0] + b_raw[2])
    snv_cx = 0.5 * (b_snv[0] + b_snv[2])
    _draw_ortho(ax, [(b_cal[2], cal_cy), (raw_cx, cal_cy), (raw_cx, b_raw[3])], color="#2E7D32")
    _draw_ortho(ax, [(b_cal[2], cal_cy), (snv_cx, cal_cy), (snv_cx, b_snv[3])], color="#2E7D32")

    # Stage II branches -> Paired LOO Test
    y_merge2 = 0.480
    loo_cx = 0.5 * (b_loo[0] + b_loo[2])
    _draw_ortho(ax, [(raw_cx, b_raw[1]), (raw_cx, y_merge2), (loo_cx, y_merge2), (loo_cx, b_loo[3])], color="#2E7D32")
    _draw_ortho(ax, [(snv_cx, b_snv[1]), (snv_cx, y_merge2), (loo_cx, y_merge2), (loo_cx, b_loo[3])], color="#2E7D32")

    # ---------------- Column 3: Stage III ----------------
    b_imp = _draw_box(
        ax, 0.715, 0.710, 0.250, 0.155,
        "IMPROVE → Apply SNV",
        "$t_{\\mathrm{LOO}} \\leq -2.0$ and $\\Delta_{\\mathrm{LOO}} \\leq -1.0\\%$",
        fc="#E8F4FD", ec="#1F77B4", lw=1.6, title_color="#0D47A1", title_size=8.9, body_size=7.5,
    )
    b_wor = _draw_box(
        ax, 0.715, 0.460, 0.250, 0.155,
        "WORSEN → Keep Raw",
        "$t_{\\mathrm{LOO}} \\geq +2.0$ and $\\Delta_{\\mathrm{LOO}} \\geq +1.0\\%$",
        fc="#FDECEC", ec="#D62728", lw=1.6, title_color="#8E1B1B", title_size=8.9, body_size=7.5,
    )
    b_abs = _draw_box(
        ax, 0.715, 0.155, 0.250, 0.215,
        "ABSTAIN → Safe Fallback",
        "Default SNV / Keep Raw / Inner-CV\nor Bilevel DiffPrep + Shift Audit",
        fc="#FFF8E1", ec="#D9822B", lw=1.6, title_color="#7A4900", title_size=8.8, body_size=7.4,
    )

    # Clean orthogonal trunk at x = 0.670 from Paired LOO Test -> Stage III
    x_trunk = 0.670
    loo_cy = 0.5 * (b_loo[1] + b_loo[3])
    _draw_ortho(
        ax,
        [(b_loo[2], loo_cy), (x_trunk, loo_cy), (x_trunk, 0.5 * (b_imp[1] + b_imp[3])), (b_imp[0], 0.5 * (b_imp[1] + b_imp[3]))],
        color="#1F77B4",
    )
    _draw_ortho(
        ax,
        [(b_loo[2], loo_cy), (x_trunk, loo_cy), (x_trunk, 0.5 * (b_wor[1] + b_wor[3])), (b_wor[0], 0.5 * (b_wor[1] + b_wor[3]))],
        color="#D62728",
    )
    _draw_ortho(
        ax,
        [(b_loo[2], loo_cy), (x_trunk, loo_cy), (x_trunk, 0.5 * (b_abs[1] + b_abs[3])), (b_abs[0], 0.5 * (b_abs[1] + b_abs[3]))],
        color="#D9822B",
    )

    # Standalone continuous shrinkage callout (not wired as ABSTAIN fallback in deploy_gate.py)
    wiener_cx = 0.5 * (b_wiener[0] + b_wiener[2])
    y_rail = 0.048
    _draw_ortho(
        ax,
        [(wiener_cx, b_wiener[1]), (wiener_cx, y_rail), (0.395, y_rail)],
        color="#2874A6",
        lw=1.4,
        ls="--",
    )
    ax.text(
        0.400,
        y_rail,
        "Standalone label-free continuous scaling: $w_{s,\\mathrm{Wiener}}^\\star = 1 - R^2_{\\mathrm{shape}}$ (evaluated separately in Sec. 4.4)",
        ha="left",
        va="center",
        fontsize=7.6,
        color="#1F4E79",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#B7D1EA", linewidth=0.9),
        zorder=7,
    )

    out_path = OUT / "fig_ms1_framework.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_fig_ms3_stage2_scatter() -> Path:
    """Fig 3: Two-stage pre-registered blind evaluation scatter plots."""
    r1 = json.loads((AUDIT / "res_prereg_theory_blind_2026_11_24.json").read_text(encoding="utf-8"))
    r2 = json.loads((AUDIT / "res_prereg_snv_gate_round2_2026_11_24.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=300)

    # Panel (a): Stage 1 (N = 24 tasks)
    wheat = [t for t in r1["tasks"] if t["dataset_group"].startswith("IDRC2016_Wheat")]
    other = [t for t in r1["tasks"] if not t["dataset_group"].startswith("IDRC2016_Wheat")]

    ax1.axhline(0.0, color="#444444", lw=0.9, ls="--", zorder=1)
    ax1.axvline(0.0, color="#444444", lw=0.9, ls="--", zorder=1)
    ax1.scatter(
        [t["pred_mean_pct"] for t in other],
        [t["emp_pooled_pct"] for t in other],
        s=44,
        color="#3B75AF",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.88,
        label=f"Other NIR tasks (n = {len(other)})",
        zorder=3,
    )
    ax1.scatter(
        [t["pred_mean_pct"] for t in wheat],
        [t["emp_pooled_pct"] for t in wheat],
        s=58,
        marker="s",
        color="#D62728",
        edgecolor="white",
        linewidth=0.5,
        label=f"Whole-grain wheat (n = {len(wheat)})",
        zorder=4,
    )
    ax1.set_xlim(-37.0, 5.5)
    ax1.set_ylim(-38.0, 162.0)
    ax1.set_xlabel("Legacy Single-Scalar Proxy Prediction (%)")
    ax1.set_ylabel("Measured SNV vs Raw RMSEP Change (%)")
    ax1.set_title("(a) Stage 1: Legacy Single-Scalar Proxy (N = 24 Tasks)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax1.text(
        0.04,
        0.95,
        "Spearman $\\rho = -0.382$\n$\\geq +5\\%$ harm capture: 0/8\n4 wheat tasks miscalled IMPROVE",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
    )

    # Panel (b): Stage 2 (N = 86 folds)
    folds = [f for t in r2["tasks"] for f in t["folds"]]
    grp_styles = [
        ("IMPROVE", "#1F77B4", "o", 42, 0.88, "IMPROVE (n = 20)"),
        ("WORSEN", "#D62728", "^", 48, 0.90, "WORSEN (n = 10)"),
        ("ABSTAIN", "#7F8C8D", "o", 26, 0.45, "ABSTAIN (n = 56)"),
    ]
    ax2.axvspan(-1.0, 1.0, color="#EEEEEE", alpha=0.85, zorder=0, label="$|\\Delta_{\\mathrm{LOO}}| < 1\\%$ band")
    ax2.axhline(0.0, color="#444444", lw=0.9, ls="--", zorder=1)
    ax2.axvline(0.0, color="#444444", lw=0.9, ls="--", zorder=1)
    for call, col, mk, sz, alp, lbl in grp_styles:
        sub = [f for f in folds if f["call_primary"] == call]
        ax2.scatter(
            [f["delta_loo_pct"] for f in sub],
            [f["emp_snv_vs_raw_pct"] for f in sub],
            s=sz,
            marker=mk,
            color=col,
            alpha=alp,
            edgecolor="white" if call != "ABSTAIN" else "#555555",
            linewidth=0.5,
            label=lbl,
            zorder=3 if call != "ABSTAIN" else 2,
        )
    ax2.set_xlim(-17.0, 64.0)
    ax2.set_ylim(-35.0, 82.0)
    ax2.set_xlabel("Calibration Paired Allen-PRESS $\\Delta_{\\mathrm{LOO}}$ (%)")
    ax2.set_ylabel("Measured SNV vs Raw RMSEP Change (%)")
    ax2.set_title("(b) Stage 2: Paired Exact Allen-PRESS LOO (N = 86 Folds)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.8)
    ax2.text(
        0.04,
        0.95,
        "Spearman $\\rho = +0.533$\nCoverage 34.9% (30/86)\nDecided Acc 73.3% (MCC 0.433)\nDedup 79.2% | Unseen 83.3%",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
    )

    out_path = OUT / "fig_ms3_stage2_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_fig_ms4_deploy_regret_rc() -> Path:
    """Fig 4: 17 valid-scale deployment asymmetric cost C_10 (incl. VSN) and selective Risk-Coverage."""
    ext = json.loads((AUDIT / "res_deploy_extended_analysis_2026_11_24.json").read_text(encoding="utf-8"))
    mr = json.loads((AUDIT / "res_major_revision_2026_11_25.json").read_text(encoding="utf-8"))
    methods_d = dict(ext["subsets"]["VALID_17_NO_DMTEST2"]["methods"])
    vsn17 = mr["part1_vsn_comparison"]["VALID_17"]
    methods_d["VSN_Rabatel2020"] = {
        "weighted_cost_by_w": {"10.0": float(vsn17["C_10"])},
        "mean_regret_pct": float(vsn17["mean_regret_pct"]),
        "n_severe_ge_thr": int(vsn17["n_severe_ge5pct"]),
    }

    order = [
        ("Gate_Fallback_SNV", "Gate + SNV Fallback", "#0D47A1"),
        ("Gate_Fallback_InnerCV", "Gate + Inner-CV Fallback", "#42A5F5"),
        ("InnerCV_PLS", "Inner-CV (Raw/SNV)", "#78909C"),
        ("Always_Raw", "Always Raw", "#9E9E9E"),
        ("SPORT_DSSNV_PLS", "Shrinkage SPORT", "#FB8C00"),
        ("SPORT_Classic_PLS", "Classic SPORT", "#F57C00"),
        ("Always_SNV", "Always SNV", "#D32F2F"),
        ("VSN_Rabatel2020", "VSN (Rabatel 2020)", "#8E24AA"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.0), dpi=300, gridspec_kw={"width_ratios": [1.22, 1.0]})

    # Panel (a): Horizontal C_10 bars
    y_pos = np.arange(len(order))
    c10_vals = [methods_d[k]["weighted_cost_by_w"]["10.0"] for k, _, _ in order]
    regs = [methods_d[k]["mean_regret_pct"] for k, _, _ in order]
    sevs = [methods_d[k]["n_severe_ge_thr"] for k, _, _ in order]
    cols = [c for _, _, c in order]

    bars = ax1.barh(y_pos, c10_vals, color=cols, height=0.64, edgecolor="#222222", linewidth=0.6)
    bars[0].set_edgecolor("#FF9800")
    bars[0].set_linewidth(2.0)

    for i, (c10, reg, sev) in enumerate(zip(c10_vals, regs, sevs)):
        ax1.text(
            c10 + 4.5,
            i,
            f"{c10:.2f}  ({reg:+.2f}%, {sev} sev)",
            va="center",
            ha="left",
            fontsize=8.6,
            fontweight="bold" if i == 0 else "normal",
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([lbl for _, lbl, _ in order], fontsize=9.2)
    ax1.invert_yaxis()
    ax1.set_xlim(0.0, 355.0)
    ax1.set_xlabel("Asymmetric Severe-Regret Cost $C_{10}$ (Lower is Better)", fontsize=9.6)
    ax1.set_title("(a) 17 Valid-Scale Deployments: Cost $C_{10}$ & Severe Regrets", fontsize=10.5)
    ax1.grid(axis="x", alpha=0.25)

    # Panel (b): Selective Risk-Coverage curves (unclipped full range)
    rc18 = ext["subsets"]["ALL_18"]["risk_coverage"]
    rc17 = ext["subsets"]["VALID_17_NO_DMTEST2"]["risk_coverage"]
    pts18 = sorted(rc18["points"], key=lambda p: p["coverage"])
    pts17 = sorted(rc17["points"], key=lambda p: p["coverage"])

    ax2.plot(
        [p["coverage"] * 100.0 for p in pts17],
        [p["selective_risk"] * 100.0 for p in pts17],
        "o-",
        color="#0D47A1",
        lw=2.2,
        ms=4.8,
        label=f"17 valid-scale tasks (AURC = {rc17['aurc_selective_error']:.4f})",
    )
    ax2.plot(
        [p["coverage"] * 100.0 for p in pts18],
        [p["selective_risk"] * 100.0 for p in pts18],
        "s--",
        color="#78909C",
        lw=1.7,
        ms=4.0,
        label=f"All 18 tasks incl. Mango DM (AURC = {rc18['aurc_selective_error']:.4f})",
    )

    op18 = next(p for p in pts18 if abs(p["t_crit"] - 2.0) < 1e-6)
    op17 = next(p for p in pts17 if abs(p["t_crit"] - 2.0) < 1e-6)
    ax2.scatter([op18["coverage"] * 100.0], [op18["selective_risk"] * 100.0], marker="*", s=180, color="#D32F2F", zorder=5)
    ax2.scatter([op17["coverage"] * 100.0], [op17["selective_risk"] * 100.0], marker="*", s=180, color="#00695C", zorder=5)

    ax2.annotate(
        f"$t = 2.0$ ($N=18$)\nCov {op18['coverage']*100:.1f}%, Risk {op18['selective_risk']*100:.1f}%",
        xy=(op18["coverage"] * 100.0, op18["selective_risk"] * 100.0),
        xytext=(3.5, 11.5),
        fontsize=7.8,
        fontweight="bold",
        color="#8E1B1B",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.88),
        arrowprops=dict(arrowstyle="-", color="#8E1B1B", lw=0.9),
        zorder=6,
    )
    ax2.annotate(
        f"$t = 2.0$ ($N=17$)\nCov {op17['coverage']*100:.1f}%, Risk 0.0%",
        xy=(op17["coverage"] * 100.0, op17["selective_risk"] * 100.0),
        xytext=(58.0, 4.5),
        fontsize=7.8,
        fontweight="bold",
        color="#004D40",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.88),
        arrowprops=dict(arrowstyle="-", color="#004D40", lw=0.9),
        zorder=6,
    )

    ax2.set_xlim(0.0, 96.0)
    ax2.set_ylim(-3.5, 56.0)
    ax2.set_xlabel("Gate Decision Coverage (%)")
    ax2.set_ylabel("Selective Sign-Error Risk (%)")
    ax2.set_title("(b) Selective Risk-Coverage Tradeoff Across $t_{\\mathrm{crit}}$")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92)

    out_path = OUT / "fig_ms4_deploy_regret_rc.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    for fn in (make_fig_ms1_framework, make_fig_ms3_stage2_scatter, make_fig_ms4_deploy_regret_rc):
        p = fn()
        print(f"Saved {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
