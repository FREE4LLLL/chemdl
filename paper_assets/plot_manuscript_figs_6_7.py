"""Generate publication-standard Manuscript Figures 5 and 6 at 300 DPI."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "_audit"
OUT = ROOT / "paper_assets"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

_SHORT_TASK = {
    "Deploy_Wheat_A1_Official__Protein": "Wheat A1 (Protein)",
    "Deploy_Wheat_A3_Official__Protein": "Wheat A3 (Protein)",
    "Deploy_Melamine_R568_to_R862__CloudPoint": "Melamine (CloudPt)",
    "Deploy_CRAW_Wheat_XDS_Prot_NewOnly__Temporal": "CRAW Wht1 (Prot New)",
    "Deploy_OSSL_Holdout__EC": "OSSL (EC)",
    "Deploy_Milk_CowGroupHoldout__Urea": "Milk (Urea)",
    "Deploy_CRAW_Barley_NIRS5000_Moist__Temporal": "CRAW Bar (Moist)",
    "Deploy_CRAW_Wheat_XDS_Moist_NewOnly__Temporal": "CRAW Wht1 (Moist New)",
    "Deploy_CRAW_Wheat_XDS_Prot__Temporal": "CRAW Wht1 (Protein)",
    "Deploy_Tablets_X2_Official__Assay": "Tablets X2 (Assay)",
    "Deploy_CRAW_Wheat2_XDS_Moist__Temporal": "CRAW Wht2 (Moist)",
    "Deploy_Milk_CowGroupHoldout__SCC": "Milk (SCC)",
    "Deploy_OSSL_Holdout__P_ext": "OSSL (P_ext)",
    "Deploy_CRAW_Barley_NIRS5000_Prot__Temporal": "CRAW Bar (Protein)",
    "Deploy_OSSL_Holdout__Mg_ext": "OSSL (Mg_ext)",
    "Deploy_OSSL_Holdout__Al_dith": "OSSL (Al_dith)",
    "Deploy_OSSL_Holdout__WR1500": "OSSL (WR1500)",
    "Deploy_Mango_Official_SP_test2__DM": "Mango (DM) [Excluded: suspected mismatch]",
}

_DEC_BADGE = {"WORSEN": "[W]", "IMPROVE": "[I]", "ABSTAIN": "[A]"}


def make_fig_ms6_deploy_per_task() -> Path:
    """Horizontal two-panel Cleveland dot/lollipop plot across 18 deployments."""
    data = json.loads((AUDIT / "res_prereg_deploy_holdout_2026_11_24.json").read_text(encoding="utf-8"))
    rows = []
    for t in data["tasks"]:
        o = float(t["oracle_pls_z"])
        s = t["strategies_same_pls"]
        r_gate = (float(s["Gate_Fallback_SNV"]) / o - 1.0) * 100.0
        r_raw = (float(s["Always_Raw"]) / o - 1.0) * 100.0
        r_snv = (float(s["Always_SNV"]) / o - 1.0) * 100.0
        rows.append((r_snv, r_raw, r_gate, t["gate_decision"], t["task_id"]))

    # Sort valid-scale 17 tasks descending by Always_SNV regret; isolate Mango (DM) at bottom
    valid_rows = [r for r in rows if r[4] != "Deploy_Mango_Official_SP_test2__DM"]
    excl_rows = [r for r in rows if r[4] == "Deploy_Mango_Official_SP_test2__DM"]
    valid_rows.sort(key=lambda x: (-x[0], -x[1]))
    rows = valid_rows + excl_rows
    n = len(rows)
    y = np.array(list(range(len(valid_rows))) + [len(valid_rows) + 0.55], dtype=float)

    snv_vals = np.array([r[0] for r in rows])
    raw_vals = np.array([r[1] for r in rows])
    gate_vals = np.array([r[2] for r in rows])
    ylabels = [f"{_SHORT_TASK.get(r[4], r[4])}  {_DEC_BADGE.get(r[3], '')}" for r in rows]

    fig = plt.figure(figsize=(12.2, 6.8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.15], wspace=0.06)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharey=ax_l)

    c_gate, c_raw, c_snv = "#1565C0", "#E67E22", "#C0392B"

    # Left panel: 0% to 35% (uniform 5% ticks)
    ax_l.set_xlim(-2.2, 35.5)
    ax_l.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax_l.set_xticklabels(["0%", "5%", "10%", "15%", "20%", "25%", "30%", "35%"])
    ax_l.axvspan(5.0, 35.5, color="#FDEDEC", alpha=0.45, zorder=0)
    ax_l.axvline(0.0, color="#90A4AE", linewidth=0.9, zorder=1)
    ax_l.axvline(5.0, color="#B71C1C", linestyle="--", linewidth=1.25, zorder=2)

    # Right panel: 45% to 168% (uniform 50% ticks)
    ax_r.set_xlim(45.0, 174.0)
    ax_r.set_xticks([50, 100, 150])
    ax_r.set_xticklabels(["50%", "100%", "150%"])
    ax_r.axvspan(45.0, 174.0, color="#FDEDEC", alpha=0.45, zorder=0)
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(axis="y", which="both", left=False, labelleft=False)

    for ax in (ax_l, ax_r):
        ax.grid(axis="x", linestyle=":", alpha=0.45, zorder=0)
        for yi in y:
            ax.axhline(yi, color="#ECEFF1", linewidth=0.8, zorder=0)
        sep_y = len(valid_rows) - 0.15
        ax.axhspan(sep_y, y[-1] + 0.65, color="#EEEEEE", alpha=0.65, zorder=0)
        ax.axhline(sep_y, color="#607D8B", linestyle="--", linewidth=1.1, zorder=2)

    # Plot points and horizontal stems
    for i in range(n):
        yi = y[i]
        # Subtle vertical offset when multiple points overlap near 0% or identical values
        # Gate+SNV at center yi; Always_Raw at yi + 0.13 if near Gate; Always_SNV at yi - 0.13 if near Gate/Raw
        y_gate = yi
        y_raw = yi + (0.14 if abs(raw_vals[i] - gate_vals[i]) < 1.2 else 0.0)
        y_snv = yi - (0.14 if (abs(snv_vals[i] - gate_vals[i]) < 1.2 or abs(snv_vals[i] - raw_vals[i]) < 1.2) else 0.0)

        # Horizontal stems from 0 to max value in left panel (clipped at 35.5 if goes to right panel)
        max_row = max(gate_vals[i], raw_vals[i], snv_vals[i])
        if max_row > 0.2:
            ax_l.plot([0.0, min(max_row, 35.5)], [yi, yi], color="#CFD8DC", linewidth=1.1, zorder=1)
        if max_row > 45.0:
            ax_r.plot([45.0, max_row], [yi, yi], color="#CFD8DC", linewidth=1.1, linestyle="-", zorder=1)

        # Always Raw
        if raw_vals[i] <= 35.5:
            ax_l.scatter(raw_vals[i], y_raw, marker="s", s=46, color=c_raw, edgecolor="#A04000", linewidth=0.7, zorder=4)
        else:
            ax_r.scatter(raw_vals[i], y_raw, marker="s", s=46, color=c_raw, edgecolor="#A04000", linewidth=0.7, zorder=4)
            ax_r.text(raw_vals[i] + 3.2, y_raw, f"+{raw_vals[i]:.1f}%", va="center", ha="left",
                      fontsize=7.9, fontweight="bold", color="#A04000", zorder=6)

        # Always SNV
        if snv_vals[i] <= 35.5:
            ax_l.scatter(snv_vals[i], y_snv, marker="^", s=54, color=c_snv, edgecolor="#7B241C", linewidth=0.7, zorder=5)
        else:
            ax_r.scatter(snv_vals[i], y_snv, marker="^", s=54, color=c_snv, edgecolor="#7B241C", linewidth=0.7, zorder=5)
            ax_r.text(snv_vals[i] + 3.2, y_snv, f"+{snv_vals[i]:.1f}%", va="center", ha="left",
                      fontsize=7.9, fontweight="bold", color="#7B241C", zorder=6)

        # Gate + SNV fallback (always in left panel, clearly visible at 0%!)
        ax_l.scatter(gate_vals[i], y_gate, marker="o", s=56, color=c_gate, edgecolor="#0D47A1", linewidth=0.8, zorder=6)

        # Explicit annotations for key points in left panel
        if gate_vals[i] >= 5.0:
            ax_l.text(gate_vals[i] - 0.9, y_gate - 0.30, f"Gate +{gate_vals[i]:.1f}%",
                      va="bottom", ha="right", fontsize=7.6, fontweight="bold", color="#0D47A1",
                      bbox=dict(boxstyle="round,pad=0.12", facecolor="#FFFFFF", edgecolor="none", alpha=0.85), zorder=7)
        if 4.0 <= raw_vals[i] <= 7.0:
            ax_l.text(raw_vals[i] + 0.85, y_raw, f"+{raw_vals[i]:.1f}%",
                      va="center", ha="left", fontsize=7.4, fontweight="bold", color="#A04000", zorder=7)

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(ylabels, fontsize=8.4)
    ax_l.set_ylim(y[-1] + 0.65, -1.15)
    ax_l.text(
        5.5, -0.72, "Severe regret threshold (+5%)",
        color="#B71C1C", fontsize=8.1, fontweight="bold", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="#FFFFFF", edgecolor="none", alpha=0.88), zorder=7,
    )

    # Diagonal break marks // on bottom spine between panels
    d = 0.012
    ax_l.plot((1.0 - d * 0.5, 1.0 + d * 0.5), (-d, +d), transform=ax_l.transAxes, color="#2C3E50", clip_on=False, lw=1.1)
    ax_r.plot((-d, +d), (-d, +d), transform=ax_r.transAxes, color="#2C3E50", clip_on=False, lw=1.1)

    fig.supxlabel("Test Regret vs. Oracle Endpoints (%)", y=0.025, fontsize=9.5, fontweight="bold")
    fig.suptitle(
        "Per-Task Regret: 17 Valid-Scale Deployments (Plus Excluded Suspected Label-Scale Mismatch)",
        y=0.975, fontsize=10.4, fontweight="bold",
    )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c_gate, markeredgecolor="#0D47A1", markersize=7.5,
               label="Gate + SNV fallback (1 severe, 17-task mean +1.98%)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=c_raw, markeredgecolor="#A04000", markersize=6.8,
               label="Always Raw (8 severe, 17-task mean +9.14%)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=c_snv, markeredgecolor="#7B241C", markersize=7.5,
               label="Always SNV (4 severe, 17-task mean +23.79%)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.935), ncol=3,
               frameon=True, framealpha=0.94, edgecolor="#CFD8DC", fontsize=8.0)
    fig.text(0.015, 0.025, "Gate calls: [W]=WORSEN, [I]=IMPROVE, [A]=ABSTAIN", fontsize=7.6, color="#546E7A")

    fig.subplots_adjust(left=0.25, right=0.97, top=0.87, bottom=0.09)
    out_p = OUT / "fig_ms6_deploy_per_task.png"
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_p} ({out_p.stat().st_size} bytes)")
    return out_p


def make_fig_ms7_ws_asymmetric_cost() -> Path:
    """Two-panel row-aligned horizontal comparison on 30 reused PLS holdouts (revised grouped protocol)."""
    data = json.loads((AUDIT / "res_revision_v2_2026_11_26.json").read_text(encoding="utf-8"))
    methods = data["p1c_ext30_summary"]["pls"]["task_equal_weighted"]

    order = [
        ("ws_0", "Row-Centered ($T_{1,0},\\ w_s = 0$)"),
        ("ws_0.10", "Fixed Shrinkage ($w_s = 0.10$)"),
        ("ws_0.15", "Fixed Shrinkage ($w_s = 0.15$)"),
        ("ws_0.20", "Fixed Shrinkage ($w_s = 0.20$)"),
        ("ws_1", "Full SNV ($T_{1,1},\\ w_s = 1$)"),
        ("ws_closed_legacy", "Legacy Closed-Form ($\\sigma_{\\mathrm{non}}=0$)"),
        ("ws_wiener_single", "Wiener-Type (1-split, $1-R^2_{\\mathrm{shape}}$)"),
        ("ws_wiener_rep20", "Wiener-Type (20-seed avg)"),
        ("ws_inner_cv_11pt", "11-pt Inner CV (Ref.)"),
        ("autoprep_cv", "AutoPrep-CV (8-op)"),
    ]

    labels = [lbl for _, lbl in order]
    costs = [float(methods[k]["C_10"]) for k, _ in order]
    times = [float(methods[k]["total_time_s"]) for k, _ in order]
    sevs = [int(methods[k]["severe_count"]) for k, _ in order]

    y = np.arange(len(order))
    colors = ["#90A4AE", "#78909C", "#607D8B", "#455A64", "#B0BEC5", "#C0392B", "#00796B", "#4DB6AC", "#37474F", "#2E7D32"]
    edge_colors = ["#455A64", "#37474F", "#263238", "#102027", "#455A64", "#7B241C", "#004D40", "#00695C", "#102027", "#1B5E20"]
    edge_widths = [0.9, 0.9, 0.9, 1.3, 0.9, 1.1, 2.0, 1.0, 1.1, 1.1]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.2, 5.5), dpi=300, sharey=True,
                                     gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.10})

    # Panel (a): Asymmetric Cost C_10
    bars_a = ax_a.barh(y, costs, height=0.60, color=colors, edgecolor=edge_colors, linewidth=edge_widths, zorder=3)
    ax_a.set_xlim(0, 228)
    ax_a.set_xlabel("Asymmetric Severe-Regret Cost $C_{10}$ (lower is better)", fontweight="bold")
    ax_a.set_title("(a) Deployment Cost $C_{10}$", fontweight="bold", pad=8)
    ax_a.grid(axis="x", linestyle=":", alpha=0.45, zorder=0)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=8.4)
    ax_a.invert_yaxis()

    for idx, (bar, val) in enumerate(zip(bars_a, costs)):
        fw = "bold" if order[idx][0] in ("ws_wiener_single", "ws_0.20", "ws_closed_legacy") else "normal"
        col = "#004D40" if order[idx][0] == "ws_wiener_single" else ("#7B241C" if order[idx][0] == "ws_closed_legacy" else "#263238")
        ax_a.text(val + 3.5, bar.get_y() + bar.get_height() / 2.0, f"{val:.2f}",
                  va="center", ha="left", fontsize=8.2, fontweight=fw, color=col, zorder=5)

    ax_a.annotate(
        "−61.3% vs Legacy\n(fixed 0.20 = 60.09)",
        xy=(104.0, 6.0),
        xytext=(158.0, 7.6),
        ha="center", va="center",
        fontsize=7.9, fontweight="bold", color="#004D40",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#E0F2F1", edgecolor="#00796B", linewidth=1.0),
        arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color="#00796B", lw=1.4, connectionstyle="arc3,rad=-0.18"),
        zorder=7,
    )

    # Panel (b): Calibration Time (s) + Severe Regret Count
    bars_b = ax_b.barh(y, times, height=0.60, color=colors, edgecolor=edge_colors, linewidth=edge_widths, zorder=3)
    ax_b.set_xlim(0, 210)
    ax_b.set_xlabel("Total Calibration Time (s)", fontweight="bold")
    ax_b.set_title("(b) Calibration Runtime & Severe Failures", fontweight="bold", pad=8)
    ax_b.grid(axis="x", linestyle=":", alpha=0.45, zorder=0)
    ax_b.tick_params(axis="y", which="both", left=False)

    for idx, (bar, tm, sv) in enumerate(zip(bars_b, times, sevs)):
        fw = "bold" if order[idx][0] in ("ws_wiener_single", "ws_inner_cv_11pt", "autoprep_cv") else "normal"
        col = "#004D40" if order[idx][0] == "ws_wiener_single" else "#263238"
        ax_b.text(tm + 3.0, bar.get_y() + bar.get_height() / 2.0, f"{tm:.1f} s ({sv}/30 sev)",
                  va="center", ha="left", fontsize=8.0, fontweight=fw, color=col, zorder=5)

    ax_b.annotate(
        "4.9× faster than 11-pt CV\n(17× vs AutoPrep; 9/30 sev)",
        xy=(68.0, 6.0),
        xytext=(144.0, 3.2),
        ha="center", va="center",
        fontsize=7.9, fontweight="bold", color="#004D40",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#E0F2F1", edgecolor="#00796B", linewidth=1.0),
        arrowprops=dict(arrowstyle="-|>", mutation_scale=12, color="#00796B", lw=1.4, connectionstyle="arc3,rad=0.15"),
        zorder=7,
    )

    fig.suptitle(
        "Continuous Scaling & Fixed-Weight Baselines on 30 Reused Holdouts (PLS Head)",
        y=0.98, fontsize=10.4, fontweight="bold",
    )
    fig.subplots_adjust(left=0.26, right=0.98, top=0.86, bottom=0.12)
    out_p = OUT / "fig_ms7_ws_asymmetric_cost.png"
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_p} ({out_p.stat().st_size} bytes)")
    return out_p


if __name__ == "__main__":
    make_fig_ms6_deploy_per_task()
    make_fig_ms7_ws_asymmetric_cost()
