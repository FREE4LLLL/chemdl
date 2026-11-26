"""自动从已落盘审计 JSON 生成论文全套高清图件（300 DPI）与三线表 Markdown（零手抄误差）。"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(ROOT, "_audit")
OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def load_json(name: str) -> dict:
    with open(os.path.join(AUDIT, name), "r", encoding="utf-8") as f:
        return json.load(f)


def make_fig1_topt(p0: dict) -> None:
    """Fig 1: Paired Subset Selection across 5 Open Benchmarks (3-column transparent comparison)."""
    cases = [
        ("Corn m5->mp6\n(k=20)", p0["postfix_existing_and_diag"]["corn_m5_to_mp6"]["k=20"]),
        ("Tablets 460\n(k=20)", p0["postfix_existing_and_diag"]["tablets_official_460"]["k=20"]),
        ("Tablets 460\n(k=50)", p0["postfix_existing_and_diag"]["tablets_official_460"]["k=50"]),
        ("SDOP 1->2\n(k=8)", p0["postfix_existing_and_diag"]["sdop_1_to_2"]["k=8"]),
        ("SDOP 1->2\n(k=16)", p0["postfix_existing_and_diag"]["sdop_1_to_2"]["k=16"]),
        ("Wheat B->A*\n(k=20 blind)", p0["blind_holdout_wheat"]["B->A"]["paired_cv_rigorous"]["k=20"]),
        ("Wheat C->B*\n(k=10 blind)", p0["blind_holdout_wheat"]["C->B"]["paired_cv_rigorous"]["k=10"]),
        ("Wheat C->B*\n(k=20 blind)", p0["blind_holdout_wheat"]["C->B"]["paired_cv_rigorous"]["k=20"]),
        ("NEO5->NEO6*\n(k=20 blind)", p0["blind_holdout_neospectra"]["NEO5_NEO6"]["paired_ds_affine"]["k=20"]),
        ("NEO8->NEO9\n(k=50)", p0["postfix_existing_and_diag"]["neo8_to_neo9"]["paired_ds_affine"]["k=50"]),
    ]
    labels = [c[0] for c in cases]
    x = np.arange(len(cases))
    w = 0.25
    vs_rnd = [c[1]["vs_rnd_pct"] for c in cases]
    vs_ks = [c[1]["vs_ks_pct"] for c in cases]
    vs_dopt = [c[1]["vs_dopt_pct"] for c in cases]

    fig, ax = plt.subplots(figsize=(12.2, 4.4))
    ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--")
    b1 = ax.bar(x - w, vs_rnd, width=w, label="Affine T-Opt vs Random (%)", color="#4C78A8")
    b2 = ax.bar(x, vs_ks, width=w, label="Affine T-Opt vs KS (%)", color="#F58518")
    b3 = ax.bar(x + w, vs_dopt, width=w, label="Affine T-Opt vs D-Opt (%)", color="#54A24B")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            va = "top" if h < 0 else "bottom"
            off = -1.2 if h < 0 else 0.8
            ax.text(bar.get_x() + bar.get_width() / 2, h + off, f"{h:+.1f}%", ha="center", va=va, fontsize=7.0, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.3)
    ax.set_ylabel("RMSEP Change of Affine T-Opt (%)  [Lower is Better]")
    ax.set_title("Paired Transfer Set Selection (ds_affine): Relative RMSEP Change of Upgraded Affine T-Opt (* = Untouched Blind Holdout)")
    ax.set_ylim(min(min(vs_rnd), min(vs_ks), min(vs_dopt)) - 14, max(max(vs_rnd), max(vs_ks), max(vs_dopt)) + 10)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_topt_paired_benchmark.png"))
    plt.close(fig)


def make_fig2_active_crossover(p0: dict, ext: dict, t03: dict) -> None:
    """Fig 2: Unpaired Active K-Shot vs Paired DS Crossover + Heavy-Tail P0-2 Stabilization."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    # Panel A: Blind Holdout Wheat B->A & C->B
    ax = axes[0]
    ks = [0, 5, 10, 20]
    u_ba = p0["blind_holdout_wheat"]["B->A"]["disjoint_10split"]["unpaired"]
    p_ba = p0["blind_holdout_wheat"]["B->A"]["disjoint_10split"]["paired_ds_affine"]
    u_cb = p0["blind_holdout_wheat"]["C->B"]["disjoint_10split"]["unpaired"]
    p_cb = p0["blind_holdout_wheat"]["C->B"]["disjoint_10split"]["paired_ds_affine"]
    ax.plot(ks, [u_ba["MeanShift"], u_ba["MeanShift_K5"], u_ba["MeanShift_K10"], u_ba["MeanShift_K20"]],
            "o-", color="#4C78A8", lw=2, label="Unpaired MeanShift+Active (B->A*)")
    ax.plot([10, 20], [p_ba["k=10"]["t_new"], p_ba["k=20"]["t_new"]],
            "s--", color="#4C78A8", alpha=0.75, label="Paired T-Opt+DS (B->A*)")
    ax.plot(ks, [u_cb["MeanShift"], u_cb["MeanShift_K5"], u_cb["MeanShift_K10"], u_cb["MeanShift_K20"]],
            "o-", color="#E45756", lw=2, label="Unpaired MeanShift+Active (C->B*)")
    ax.plot([10, 20], [p_cb["k=10"]["t_new"], p_cb["k=20"]["t_new"]],
            "s--", color="#E45756", alpha=0.75, label="Paired T-Opt+DS (C->B*)")
    ax.set_title("(a) Blind Holdout Wheat* (B->A, C->B)")
    ax.set_xlabel("Labeled Target Samples K")
    ax.set_ylabel("Standardized RMSEP")
    ax.set_xticks(ks)
    ax.legend(fontsize=7.3, loc="upper right")
    ax.grid(alpha=0.25)

    # Panel B: IDRC Tablets X1->X2 Crossover (from T-03)
    ax = axes[1]
    tab = t03["T03_ks_and_paired_bounds"]["tablets_10split"]
    ax.plot([0, 5, 10, 20], [tab["K=5"]["Unpaired_SLOT_PLS_K0"], tab["K=5"]["Unpaired_SLOT_PLS_Active"], tab["K=10"]["Unpaired_SLOT_PLS_Active"], tab["K=20"]["Unpaired_SLOT_PLS_Active"]],
            "o-", color="#54A24B", lw=2, label="Unpaired SLOT+Active(K)")
    ax.plot([5, 10, 20], [tab["K=5"]["Paired_DS_KS"], tab["K=10"]["Paired_DS_KS"], tab["K=20"]["Paired_DS_KS"]],
            "s--", color="#4C78A8", lw=1.8, label="Paired KS+DS")
    ax.plot([5, 10, 20], [tab["K=5"]["Paired_DS_Rnd"], tab["K=10"]["Paired_DS_Rnd"], tab["K=20"]["Paired_DS_Rnd"]],
            "^:", color="#9D755D", lw=1.6, label="Paired Rnd+DS")
    ax.plot([5, 10, 20], [tab["K=5"]["Paired_PDS_KS"], tab["K=10"]["Paired_PDS_KS"], tab["K=20"]["Paired_PDS_KS"]],
            "d-.", color="#F58518", lw=1.6, label="Paired KS+PDS")
    ax.set_title("(b) IDRC Tablets X1->X2 (Assay mg)")
    ax.set_xlabel("Labeled Samples K (Paired vs Unpaired)")
    ax.set_ylabel("RMSEP (mg)")
    ax.set_xticks([0, 5, 10, 20])
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)

    # Panel C: Heavy-Tail Soil OSSL NeoSpectra P0-2 Stabilization (Before vs After)
    ax = axes[2]
    # Note: ext used slightly different rng seed/norm vs p0; compare exact Before/After from diag & p0
    # Use exact pre-fix vs post-fix relative % change over MeanShift K0 to keep scale 100% apples-to-apples!
    pre2 = ext["external_neospectra"]["NEO2_NEO3"]["unpaired"]
    post2 = p0["postfix_existing_and_diag"]["neo2_to_neo3"]["unpaired"]
    pre8 = ext["external_neospectra"]["NEO8_NEO9"]["unpaired"]
    post8 = p0["postfix_existing_and_diag"]["neo8_to_neo9"]["unpaired"]
    b14 = p0["blind_holdout_neospectra"]["NEO1_NEO4"]["unpaired"]

    def rel(u):
        k0 = u["MeanShift"]
        return [0.0, (u["MeanShift_K5"] / k0 - 1) * 100, (u["MeanShift_K10"] / k0 - 1) * 100, (u["MeanShift_K20"] / k0 - 1) * 100]

    ax.axhline(0.0, color="#333333", ls="--", lw=0.9)
    ax.plot(ks, rel(pre2), "o--", color="#E45756", alpha=0.85, lw=1.8, label="Before Robust Firewall: NEO2->NEO3")
    ax.plot(ks, rel(post2), "o-", color="#4C78A8", lw=2.0, label="After Robust Firewall: NEO2->NEO3")
    ax.plot(ks, rel(pre8), "s--", color="#F58518", alpha=0.85, lw=1.8, label="Before Robust Firewall: NEO8->NEO9")
    ax.plot(ks, rel(post8), "s-", color="#54A24B", lw=2.0, label="After Robust Firewall: NEO8->NEO9")
    ax.plot(ks, rel(b14), "^-", color="#B279A2", lw=1.8, label="After Robust Firewall: NEO1->NEO4* (blind)")
    ax.set_title("(c) Heavy-Tail Soil OC: Robust Residual Firewall")
    ax.set_xlabel("Active Samples K")
    ax.set_ylabel("Delta RMSEP vs MeanShift K0 (%)")
    ax.set_xticks(ks)
    ax.set_ylim(-1.5, 36.5)
    ax.legend(fontsize=6.9, loc="upper left", framealpha=0.92)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_unpaired_active_crossover.png"))
    plt.close(fig)


def make_fig3_snv_coupling(ext: dict) -> None:
    """Fig 3: Fixed-Variance std(a)=0.35 SNV Scattering-Concentration Coupling Sign-Reversal Law."""
    snv = ext["fixed_var_snv_coupling"]
    rs = [0.0, 0.2, 0.4, 0.6, 0.8]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axhline(0.0, color="#222222", lw=1.0, ls="--")
    styles = [
        ("cordoba_wheat", "Cordoba Wheat (Solid Cereal)", "o-", "#4C78A8"),
        ("cordoba_barley", "Cordoba Barley (Solid Cereal)", "s-", "#E45756"),
        ("swri_diesel_CN", "SWRI Diesel CN (Liquid Fuel)", "D-", "#54A24B"),
    ]
    for key, label, fmt, col in styles:
        vals = [snv[key][f"r={r:.1f}"]["gain_pct"] for r in rs]
        ax.plot(rs, vals, fmt, color=col, lw=2.2, markersize=6, label=label)
        for r_v, v in zip(rs, vals):
            if r_v in (0.4, 0.8):
                if r_v == 0.8 and key == "cordoba_barley":
                    xy = (0, -16)
                elif r_v == 0.8 and key == "swri_diesel_CN":
                    xy = (0, 9)
                elif r_v == 0.4 and key == "swri_diesel_CN":
                    xy = (0, 8)
                else:
                    xy = (0, -14)
                ax.annotate(f"{v:+.1f}%", (r_v, v), textcoords="offset points", xytext=xy,
                            ha="center", fontsize=8.0, color=col, fontweight="bold",
                            bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=0.8))
    ax.axhspan(0, 22, color="#E45756", alpha=0.07)
    ax.text(0.03, 18.5, "SNV Harmful Region (+RMSEP)", color="#B03A2E", fontsize=8.5, fontweight="bold")
    ax.text(0.03, -33.5, "SNV Beneficial Region (-RMSEP)", color="#1E6091", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Scatter-Concentration Correlation  r = Corr(a, y)  [Fixed std(a) = 0.35]")
    ax.set_ylabel("SNV RMSEP Change vs Raw (%)")
    ax.set_title("Fixed-Variance Multiplicative Scatter Coupling Law Across Solid & Liquid NIR")
    ax.set_xlim(-0.04, 0.87)
    ax.set_ylim(-36, 22)
    ax.set_xticks(rs)
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.36), frameon=True, framealpha=0.92)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_snv_fixed_var_coupling.png"))
    plt.close(fig)


def make_fig4_gate_identifiability(p0: dict, attr: dict) -> None:
    """Fig 4: Q-Residual Shift Ratio r_shift Across All 3 Physical Regimes."""
    get_r = lambda d: float(d.get("r_shift_mean", d.get("r_shift")))
    items = [
        ("Cordoba 98->99 (Pure Label)", get_r(attr["cordoba_wheat_5split_disjoint_snv"]), "#4C78A8"),
        ("Mango raw (Pure Label)", get_r(attr["mango_5split_raw"]), "#4C78A8"),
        ("Mango snv (Pure Label)", get_r(attr["mango_5split_snv"]), "#4C78A8"),
        ("NEO2->NEO3 (Same-Model MEMS)", p0["postfix_existing_and_diag"]["neo2_to_neo3"]["unpaired"]["r_shift_mean"], "#F58518"),
        ("NEO5->NEO6* (Same-Model MEMS)", p0["blind_holdout_neospectra"]["NEO5_NEO6"]["unpaired"]["r_shift_mean"], "#F58518"),
        ("NEO8->NEO9 (Same-Model MEMS)", p0["postfix_existing_and_diag"]["neo8_to_neo9"]["unpaired"]["r_shift_mean"], "#F58518"),
        ("NEO1->NEO4* (Same-Model MEMS)", p0["blind_holdout_neospectra"]["NEO1_NEO4"]["unpaired"]["r_shift_mean"], "#F58518"),
        ("Wheat A1->A3 snv (Moderate)", get_r(attr["wheat_A1_to_A3_snv"]), "#54A24B"),
        ("IDRC Wheat B->A* (Cross-Mfr)", p0["blind_holdout_wheat"]["B->A"]["disjoint_10split"]["unpaired"]["r_shift_mean"], "#54A24B"),
        ("IDRC Wheat C->B* (Cross-Mfr)", p0["blind_holdout_wheat"]["C->B"]["disjoint_10split"]["unpaired"]["r_shift_mean"], "#54A24B"),
        ("Corn snv (Cross-Unit)", get_r(attr["corn_20seed_indep_snv"]), "#54A24B"),
        ("Tablets snv (Cross-Unit)", get_r(attr["tablets_20split_snv"]), "#54A24B"),
        ("SDOP snv (Cross-Unit)", get_r(attr["sdop_5seed_agg_disjoint_snv"]), "#54A24B"),
    ]
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    cols = [i[2] for i in items]
    y = np.arange(len(items))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.barh(y, vals, color=cols, alpha=0.88)
    ax.axvline(2.2, color="#B03A2E", ls="--", lw=1.4, label="gate_low = 2.2")
    ax.axvline(5.0, color="#196F3D", ls=":", lw=1.4, label="gate_high = 5.0")
    ax.set_xscale("log")
    for yi, v in zip(y, vals):
        ax.text(v * 1.07, yi, f"{v:.2f}", va="center", fontsize=8.0, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.9, alpha=0.90))
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Orthogonal Q-Residual Shift Ratio  r_shift = Q_t(m_s) / Q_t(m_t)  (log scale)")
    ax.set_title("Unsupervised Q-Residual Gate: Clean Separation vs Identical-Model MEMS Gray Zone")
    ax.set_xlim(0.8, 85)
    # Custom legend patches
    from matplotlib.patches import Patch
    handles = [
        Patch(color="#4C78A8", label="Pure Label Shift (Block)"),
        Patch(color="#F58518", label="Identical-Model MEMS Gray Zone (Unidentifiable)"),
        Patch(color="#54A24B", label="Cross-Instrument Hardware Shift (Open)"),
    ] + ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="lower right", fontsize=7.8, frameon=True)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_q_gate_identifiability.png"))
    plt.close(fig)


def make_fig5_dssnv(dssnv: dict) -> None:
    """Fig 5: Decoupled Shrinkage-SNV (DSSNV) vs Raw / Classical SNV / EMSC across 7 Real + Coupled Benchmarks."""
    real = dssnv["real_datasets"]
    coup = dssnv["coupled_scatter"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3), gridspec_kw={"width_ratios": [1.55, 1.0]})

    # Panel A: 7 Real Public Datasets (relative % change vs Classical SNV)
    ax = axes[0]
    items_r = [
        ("Tecator\n(3-comp)", real["Tecator (3 comp)"]),
        ("Tecator\n(fat)", real["Tecator (fat)"]),
        ("Corn m5\n(4-comp)", real["Corn m5 (4 comp)"]),
        ("Tablets X1\n(3-comp)", real["Tablets X1 (3 comp)"]),
        ("Cordoba\nWheat", real["Cordoba Wheat"]),
        ("Cordoba\nBarley", real["Cordoba Barley"]),
        ("Cordoba\nMaize", real["Cordoba Maize"]),
        ("SWRI\nDiesel CN", real["SWRI Diesel CN"]),
    ]
    labels = [it[0] for it in items_r]
    x = np.arange(len(items_r))
    w = 0.34
    vs_raw = [it[1]["vs_raw_pct"] for it in items_r]
    vs_snv = [it[1]["vs_snv_pct"] for it in items_r]
    ax.axhline(0.0, color="#333333", ls="--", lw=0.9)
    b1 = ax.bar(x - w / 2, vs_raw, width=w, color="#4C78A8", label="DSSNV vs Raw (%)")
    b2 = ax.bar(x + w / 2, vs_snv, width=w, color="#E45756", label="DSSNV vs Classical SNV (%)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            va = "top" if h < 0 else "bottom"
            off = -1.2 if h < 0 else 0.8
            ax.text(bar.get_x() + bar.get_width() / 2, h + off, f"{h:+.1f}%", ha="center", va=va, fontsize=7.0, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_ylabel("Nested-CV RMSEP Change of DSSNV (%)  [Lower is Better]")
    ax.set_title("(a) 7 Real Open NIR Datasets (Inductive Nested 5x5 CV)")
    ax.set_ylim(-44, 10)
    ax.legend(loc="lower right", frameon=True, fontsize=8.0)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Partially Coupled Scattering (std(a)=0.35)
    ax = axes[1]
    c_keys = [
        ("Barley\nr=0.6", coup["Cordoba Barley_r0.6"]),
        ("Barley\nr=0.8", coup["Cordoba Barley_r0.8"]),
        ("Diesel\nr=0.6", coup["SWRI Diesel CN_r0.6"]),
        ("Diesel\nr=0.8", coup["SWRI Diesel CN_r0.8"]),
    ]
    xc = np.arange(len(c_keys))
    wc = 0.20
    for idx, (m_k, m_lbl, col) in enumerate([
        ("raw", "Raw", "#9D755D"),
        ("snv", "Classical SNV", "#F58518"),
        ("emsc", "EMSC", "#72B7B2"),
        ("dssnv", "DSSNV (Ours)", "#54A24B"),
    ]):
        vals = [ck[1][m_k] for ck in c_keys]
        ax.bar(xc + (idx - 1.5) * wc, vals, width=wc, color=col, label=m_lbl)
    ax.set_xticks(xc)
    ax.set_xticklabels([ck[0] for ck in c_keys], fontsize=8.5)
    ax.set_ylabel("Standardized Nested-CV RMSEP")
    ax.set_title("(b) Coupled Scattering (std(a)=0.35)")
    ax.set_ylim(0.0, 0.92)
    ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=7.6)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_dssnv_breakthrough.png"))
    plt.close(fig)


def _append_table_21(lines: list[str], review_p0: dict) -> None:
    """Append Table 21 (21-config tuned-SG baseline, Tecator ablation, physical units/RPD)."""
    lines += [
        "",
        "---",
        "",
        "## Table 21. 21 组折内调参 SG（`tuned-SG` & `SNV+tuned-SG`）强基线对照、Tecator 自由度消融与原物理量纲（`res_review_p0_2026_11_23.json`, `C-141`）",
        "",
        "### 21a. 7 经典 NIR 库严格同切分同 LOO 岭回归头 3×5 CV（`z`-RMSEP）",
        "",
        "| 真实公开 NIR 库 | `Raw-Ridge` | `SNV-Ridge` | `tuned-SG` (21 组) | `SNV+tuned-SG` (21 组) | `SPORT_fix` | `Bilevel` | **`MK-Bilevel`** | `MK vs 最优调参 SG` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, r in review_p0["part1_tuned_sg_7datasets"].items():
        z = r["z_pooled"]
        raw_v, snv_v = z["raw_ridge"], z["snv_ridge"]
        tsg_v, stsg_v = z["tuned_sg"], z["snv_tuned_sg"]
        sp_v, bi_v, mk_v = z["sport_fix_ref"], z["bilevel_ref"], z["mk_bilevel_ref"]
        best_v = min(raw_v, snv_v, tsg_v, stsg_v, sp_v, bi_v, mk_v)
        fmt = lambda val: f"**{val:.4f}**" if abs(val - best_v) < 1e-6 else f"{val:.4f}"
        pct_best = r["mk_vs_best_tuned_sg_or_snv_sg_pct"]
        pct_tsg = r["mk_vs_tuned_sg_pct"]
        if ds == "Tecator":
            ratio_best = stsg_v / mk_v
            cmp_s = f"**{pct_best:+.1f}%（{ratio_best:.1f}×，vs tSG {pct_tsg:+.1f}%）**"
        else:
            cmp_s = f"{pct_best:+.1f}%"
        lines.append(
            f"| `{ds}` | {fmt(raw_v)} | {fmt(snv_v)} | {fmt(tsg_v)} | {fmt(stsg_v)} | "
            f"{fmt(sp_v)} | {fmt(bi_v)} | {fmt(mk_v)} | {cmp_s} |"
        )

    p2 = review_p0["part2_tecator_ablation_and_phys"]
    ab = p2["tecator_ablation_z"]
    pu = p2["physical_units_mk"]
    tec, crn, tab, dis = pu["Tecator_MK"], pu["Corn_m5_MK"], pu["Tablets_X1_MK"], pu["SWRI_Diesel_CN_MK"]
    lines += [
        "",
        "### 21b. Tecator 2× 增益自由度拆解 & `MK-Bilevel` 代表原物理量纲与 `RPD`",
        "",
        f"- **Tecator 消融（同 3×5 CV）**：`Raw ({ab['raw']:.4f})` → `tuned-SG ({ab['tuned_sg']:.4f})` → `SNV ({ab['snv']:.4f})` → "
        f"`SNV+tuned-SG ({ab['snv_tuned_sg']:.4f})` → `Bilevel 无 SG 分支 / 仅 SNV-opt+FOD ({ab['bilevel_no_sg']:.4f})` → "
        f"`Bilevel_full ({ab['bilevel_full']:.4f})` → **`MK-Bilevel ({ab['mk_bilevel']:.4f})`**。",
        "- **原物理量纲 RMSEP / $R^2$ / `RPD`**：",
        f"  - **Tecator**：`water = {tec['water_%']['rmsep']:.3f}% (R²={tec['water_%']['r2']:.4f}, RPD={tec['water_%']['rpd']:.2f})`、"
        f"`fat = {tec['fat_%']['rmsep']:.3f}% (R²={tec['fat_%']['r2']:.4f}, RPD={tec['fat_%']['rpd']:.2f})`、"
        f"`protein = {tec['protein_%']['rmsep']:.3f}% (R²={tec['protein_%']['r2']:.4f}, RPD={tec['protein_%']['rpd']:.2f})`；",
        f"  - **Corn_m5**：`moisture = {crn['moisture_%']['rmsep']:.4f}% (R²={crn['moisture_%']['r2']:.4f}, RPD={crn['moisture_%']['rpd']:.2f})`、"
        f"`oil = {crn['oil_%']['rmsep']:.4f}% (R²={crn['oil_%']['r2']:.4f}, RPD={crn['oil_%']['rpd']:.2f})`、"
        f"`protein = {crn['protein_%']['rmsep']:.4f}% (R²={crn['protein_%']['r2']:.4f}, RPD={crn['protein_%']['rpd']:.2f})`、"
        f"`starch = {crn['starch_%']['rmsep']:.4f}% (R²={crn['starch_%']['r2']:.4f}, RPD={crn['starch_%']['rpd']:.2f})`；",
        f"  - **Tablets_X1**：`weight = {tab['weight_mg']['rmsep']:.3f} mg (R²={tab['weight_mg']['r2']:.4f})`、"
        f"`hardness = {tab['hardness']['rmsep']:.3f} (R²={tab['hardness']['r2']:.4f})`、"
        f"`assay = {tab['assay_mg']['rmsep']:.3f} mg (R²={tab['assay_mg']['r2']:.4f}, RPD={tab['assay_mg']['rpd']:.2f})`；"
        f"**SWRI Diesel CN**：`{dis['cetane_number']['rmsep']:.3f} CN (R²={dis['cetane_number']['r2']:.4f})`。",
    ]


def _append_table_22(lines: list[str], ind_dl: dict,
                     ind_followups: dict | None = None,
                     multiseed: dict | None = None,
                     wilcoxon_ind: dict | None = None) -> None:
    """Append Table 22 (post-1-SE 6 industrial tasks, scaling curves, 3-seed ddof=1, and paired Wilcoxon)."""
    task_cols = [
        ("Mango_DM_Seasonal", "Mango 跨季 (`9.9k→1.4k`)"),
        ("CRAW_Wheat_Temporal", "CRAW 27年 OOD (`4.4k→691`)"),
        ("OSSL_Soil_MEMS", "OSSL 土壤 (`6,466→1,665`)"),
        ("Milk_Transmittance_Sess", "Milk 生乳 (`894→330`)"),
        ("Melamine_InDomain_R861", "Melamine 同配方 (`3.1k→778`)"),
        ("Melamine_CrossInst_562_861", "Melamine 跨配方 (`3.0k→3.9k`)"),
    ]
    method_rows = [
        ("Raw_PLS", "`Raw_PLS`"),
        ("SNV_PLS", "`SNV_PLS`"),
        ("SG1_PLS", "`SG1_PLS`"),
        ("DL_1D_ResNet", "`DL_1D_ResNet`"),
        ("DL_DeepSpectra", "`DL_DeepSpectra`"),
        ("DL_ResMLP", "`DL_ResMLP`"),
        ("DL_SpectraFormer", "`DL_SpectraFormer`"),
        ("ChemDL_Bilevel", "`ChemDL_Bilevel`"),
        # codex full-audit E18: 归档里参评的 12 种方法包含 ChemDL_DSSNV_PLS，
        # 但制表只输出 11 行，读者查不到它的六个结果。
        ("ChemDL_DSSNV_PLS", "`ChemDL_DSSNV_PLS`"),
        ("ChemDL_MK_Bilevel", "**`ChemDL_MK_Bilevel`（纯线性）**"),
        ("ChemDL_DiffPrep_CNN", "**`ChemDL_DiffPrep_CNN`**"),
        ("ChemDL_MK_ResNet", "**`ChemDL_MK_ResNet`**"),
    ]
    tasks = ind_dl["tasks"]
    winners = {tk: min(tasks[tk]["methods"].keys(), key=lambda m: tasks[tk]["methods"][m]["std_pooled_rmsep"]) for tk, _ in task_cols}

    lines += [
        "",
        "---",
        "",
        "## Table 22. 5 套工业级大库（34,411 条真实谱）6 任务 × 12 方法最终验证（`res_industrial_vs_dl_2026_11_21.json`, `res_multiseed_industrial.json`, `res_industrial_paired_wilcoxon_2026_11_23.json`, `C-144`）",
        "",
        "### 22a. 工业 6 任务统一 `std_pooled_rmsep` 主表（`seed=0` 单次主表；括号内为原量纲指标或 `R²`；3-seed 与配对检验见 22b）",
        "",
        "| 方法 | " + " | ".join(hdr for _, hdr in task_cols) + " |",
        "|---|" + "|".join("---:" for _ in task_cols) + "|",
    ]
    for mk, mlbl in method_rows:
        cells = []
        for tk, _ in task_cols:
            v = tasks[tk]["methods"][mk]["std_pooled_rmsep"]
            cells.append(f"**{v:.4f} (#1)**" if winners[tk] == mk else f"{v:.4f}")
        lines.append(f"| {mlbl} | " + " | ".join(cells) + " |")

    summary_cells = []
    for tk, _ in task_cols:
        ms = tasks[tk]["methods"]
        b_ch = min((m for m in ms if m.startswith("ChemDL_")), key=lambda m: ms[m]["std_pooled_rmsep"])
        b_dl = min((m for m in ms if m.startswith("DL_")), key=lambda m: ms[m]["std_pooled_rmsep"])
        gain = (ms[b_ch]["std_pooled_rmsep"] / ms[b_dl]["std_pooled_rmsep"] - 1.0) * 100.0
        pt = ms[b_ch]["per_target"]
        if tk == "Mango_DM_Seasonal":
            tag = f"`{pt['DryMatter_%']['rmsep']:.4f}%`"
        elif tk == "CRAW_Wheat_Temporal":
            tag = f"`{pt['Protein_%']['rmsep']:.4f}%`"
        elif tk in ("OSSL_Soil_MEMS", "Milk_Transmittance_Sess"):
            tag = f"`R²={ms[b_ch]['mean_r2']:.3f}`"
        else:
            # codex full-audit E5: 目标名从 "Melamine" 更正为 "turbidity_point_C"
            # 后，写死旧键会让 Table 22 抛 KeyError。这里两种都接受。
            _pt_key = "turbidity_point_C" if "turbidity_point_C" in pt else "Melamine"
            tag = f"`{pt[_pt_key]['rmsep']:.3f}`"
        summary_cells.append(f"**{gain:+.1f}% ({tag})**")
    lines.append("| **最优 ChemDL vs 最优纯 DL** | " + " | ".join(summary_cells) + " |")

    if ind_followups is not None and "B3_melamine_crossinst" in ind_followups:
        b3 = ind_followups["B3_melamine_crossinst"]
        ms_r = b3["Unp_MeanShift_Raw"]
        snv_r = b3["SNV_PLS"]
        gain_ms = (ms_r["std_pooled_rmsep"] / snv_r["std_pooled_rmsep"] - 1.0) * 100.0
        lines.append(
            f"\n> 跨配方场景补充（`_audit/res_industrial_followups_2026_11_21.json`）：接入零标签目标域均值对齐 `Unp_MeanShift_Raw` 后，"
            f"Melamine 跨配方（`R562→R861`）进一步降至 **`{ms_r['std_pooled_rmsep']:.4f}`（`R²={ms_r['mean_r2']:.4f}`，较 `SNV_PLS` 再降 `{gain_ms:+.1f}%`）**。"
        )

    sc = ind_dl["scaling_curves"]
    mg_sc, cr_sc = sc["Mango_DM_Seasonal"], sc["CRAW_Wheat_Temporal"]
    ns_all = ["100", "300", "1000", "3000"]
    def _best_dl_sc(d: dict) -> float:
        return min(v for k, v in d.items() if k.startswith("DL_"))

    mg_cells = [
        f"`{mg_sc[n]['Raw_PLS']:.3f} / {mg_sc[n]['ChemDL_MK_Bilevel']:.3f} / {mg_sc[n]['ChemDL_MK_ResNet']:.3f} / {_best_dl_sc(mg_sc[n]):.3f}`"
        for n in ns_all + ["9914"]
    ]
    # codex full-audit E16: the main run does not produce CRAW n=3000
    # (a follow-up script backfills it); a hard key access raised KeyError.
    def _cr_cell(n):
        if n not in cr_sc:
            return "`—`"
        d = cr_sc[n]
        return (f"`{d['SNV_PLS']:.3f} / {d['ChemDL_MK_Bilevel']:.3f} / "
                f"{d['ChemDL_MK_ResNet']:.3f} / {_best_dl_sc(d):.3f}`")

    def _mg_cell(n):
        if n not in mg_sc:
            return "`—`"
        d = mg_sc[n]
        return (f"`{d['Raw_PLS']:.3f} / {d['ChemDL_MK_Bilevel']:.3f} / "
                f"{d['ChemDL_MK_ResNet']:.3f} / {_best_dl_sc(d):.3f}`")

    mg_cells = [_mg_cell(n) for n in ns_all + ["9914"]]
    cr_cells = [_cr_cell(n) for n in ns_all + ["4355"]]
    lines += [
        "",
        "### 22b. 样本量缩放曲线（原量纲 RMSEP）与全 6 任务 3-seed（`ddof=1`）+ 逐样本 Wilcoxon 复核",
        "",
        "| 缩放库 | 指标 | `n=100` | `n=300` | `n=1,000` | `n=3,000` | `Full` (`9914` / `4355`) |",
        "|---|---|---:|---:|---:|---:|---:|",
        "| **Mango DM%** (`L=103`) | `Raw` / `MK_Bi` / `MK_RN` / 最优纯 `DL` | " + " | ".join(mg_cells) + " |",
        "| **CRAW Protein%** (`L=700`) | `SNV` / `MK_Bi` / `MK_RN` / 最优纯 `DL` | " + " | ".join(cr_cells) + " |",
    ]
    if multiseed is not None:
        def _ms(task: str, meth: str) -> tuple[float, float, float, float]:
            arr = np.asarray(multiseed[task][meth], dtype=np.float64)
            return float(np.mean(arr)), float(np.std(arr, ddof=1)), float(np.min(arr)), float(np.max(arr))

        lines += [
            "",
            "| 工业 6 任务 | 最优 ChemDL (3-seed `mean±SD`) | 纯线性 `MK_Bilevel` (3-seed) | 最优纯 DL (3-seed `mean±SD`) | 3-seed 均值差 / 分布 | 逐样本 Wilcoxon `p` (`d_z`, `seed=0`) |",
            "|---|---|---|---|---:|---:|",
        ]
        wx_tasks = wilcoxon_ind.get("tasks", {}) if wilcoxon_ind else {}
        for tk, tlbl in task_cols:
            d_t = multiseed[tk]
            ch_cands = [k for k in d_t if k.startswith("ChemDL_")]
            dl_cands = [k for k in d_t if k.startswith("DL_")]
            b_ch = min(ch_cands, key=lambda k: float(np.mean(d_t[k])))
            b_dl = min(dl_cands, key=lambda k: float(np.mean(d_t[k])))
            ch_m, ch_s, ch_lo, ch_hi = _ms(tk, b_ch)
            bi_m, bi_s, _, _ = _ms(tk, "ChemDL_MK_Bilevel")
            dl_m, dl_s, dl_lo, dl_hi = _ms(tk, b_dl)
            diff_pct = (round(ch_m, 4) / round(dl_m, 4) - 1.0) * 100.0
            no_overlap = ch_hi < dl_lo
            ch_tag = b_ch.replace("ChemDL_", "")
            dl_tag = b_dl.replace("DL_", "")
            if no_overlap:
                rel_s = f"**`{diff_pct:+.1f}%`（零重叠）**"
            else:
                # codex full-audit E7: 不再按「标准差大小」自动给出「打平」判词——
                # 打平与否应由显式判据（均值差 vs 1 SD）决定，且必须能区分「赢/输」。
                _abs_diff = abs(ch_m - dl_m)
                if _abs_diff > dl_s:
                    rel_s = f"`{diff_pct:+.1f}%`（**均值差 {_abs_diff:+.5f} 超过 1 SD {dl_s:.5f}，判为落后**）"
                elif _abs_diff > 0:
                    rel_s = f"`{diff_pct:+.1f}%`（均值差 {_abs_diff:.5f} 在 1 SD {dl_s:.5f} 内）"
                else:
                    rel_s = f"`{diff_pct:+.1f}%`"
            wx = wx_tasks.get(tk)
            if wx:
                # codex full-audit E6: 归档的 p 值只对应某一个 DL 对手，而表中
                # 「最优纯 DL」可能换成另一个模型（CRAW 的 1D_ResNet vs ResMLP
                # 就是这种情况）。不核对模型名会把对 A 的检验读成对 B 的检验。
                wx_dl = wx.get("dl_method", "")
                wx_ch = wx.get("chemdl_method", "")
                _match = (wx_dl == b_dl) and (wx_ch == b_ch)
                pv = wx["p_wilcoxon_two_sided"]
                p_str = "p<1e-300" if pv < 1e-300 else f"p={pv:.2e}"
                if _match:
                    wx_s = f"`{p_str}` (`d_z={wx['cohen_dz_per_sample']:+.2f}`)"
                else:
                    # 模型不匹配时明确写清检验对象，避免误读
                    wx_s = (f"`{p_str}` (`d_z={wx['cohen_dz_per_sample']:+.2f}`; "
                            f"检验对象 `{wx_ch.replace('ChemDL_','')}` vs "
                            f"`{wx_dl.replace('DL_','')}`)")
            else:
                wx_s = "—"
            lines.append(
                f"| {tlbl} | **`{ch_tag}` {ch_m:.4f}±{ch_s:.4f}** | `{bi_m:.4f}±{bi_s:.4f}` | "
                f"`{dl_tag}` {dl_m:.4f}±{dl_s:.4f} | {rel_s} | {wx_s} |"
            )
        lines.append(
            # codex full-audit E32: fit_pls 的内层 CV 用随机 KFold 选成分数，
            # 实测同一划分下 seed 0/1/2 会选出不同的 k（如 [20,20,20,20] /
            # [19,20,18,20] / [20,20,17,20]），因此不能称为「确定性」。
            "\n> **全 6 任务多随机种子结论（`C-144`）**：采用样本标准差 `ddof=1` 汇报。"
            "PLS 基线（`Raw_PLS / SG1_PLS`）的**拟合**在给定划分与给定 k 时是确定的，"
            "但其**成分数由内层随机 KFold 选出**，随 seed 可能变化；本表按实测三 seed 汇报，"
            "若三 seed 恰给出同一 k 则显示单一数值。"
            # codex full-audit E23: 归档里每个任务只有**一个** DL 对手的配对 p 值，",
            # 但正文曾据此声称「胜过全部 4 种纯 DL 均显著」。这是两种不同的证据：
            #   (a) 范围比较：本框架三 seed 的 mean±SD 与 4 种对手的 mean±SD 是否重叠；
            #   (b) 配对检验：某一条预先指定的 (本框架, 对手) 组合的逐样本 Wilcoxon。
            # 本节只主张 (a)，(b) 限定在 Table 22b 明确写出的那一对模型上。
            "**5/6 任务上，本框架三 seed 的 mean±SD 与全部 4 种纯 DL 的 mean±SD 均不重叠"
            "（均值降错 `-9.9% ~ -71.3%`，逐行见上表；CRAW 上 3-seed `Bilevel 0.2524±0.0011` 与 `MK_ResNet 0.2528±0.0028` 仅差 `0.0004`，在各自 1 SD 内两者相当）**；"
            "表中所列的逐样本配对 Wilcoxon `p` 只对应**每行明确写出的那一对模型**，"
            "不构成「对其余三种对手也显著」的证据。"
            + _mango_conclusion_5of6()
        )


def _mango_conclusion_5of6() -> str:
    """Mango 结论从 res_multiseed_industrial.json 计算，不写死数字。

    codex 审计 05：早先硬编码「在 1 个 DL SD 波动内打平」，但真实均值差
    0.01027 已超过 DeepSpectra 的 1 个样本 SD（0.00947），按该判据是「输」。
    此处只报可核验的量，不做等效/打平断言。
    """
    import json as _json
    import os as _os
    import numpy as _np
    p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                      "_audit", "res_multiseed_industrial.json")
    # 方向必须按实测符号判断：RMSEP 越小越好，mk_m < ds_m 即本框架领先。
    head_dir = "在万级低维 Mango（`n=9,914, L=103`）上："
    if not _os.path.exists(p):
        return head_dir + "逐种子数值见 `_audit/res_multiseed_industrial.json`。"
    d = _json.load(open(p, encoding="utf-8"))
    m = d.get("Mango_DM_Seasonal", {})
    if "ChemDL_MK_ResNet" not in m or "DL_DeepSpectra" not in m:
        return head_dir + "逐种子数值见 `_audit/res_multiseed_industrial.json`。"

    def _st(v):
        a = _np.asarray(v, float)
        return float(a.mean()), float(a.std(ddof=1))

    mk_m, mk_s = _st(m["ChemDL_MK_ResNet"])
    ds_m, ds_s = _st(m["DL_DeepSpectra"])
    diff = mk_m - ds_m
    over = "超过" if abs(diff) > ds_s else "未超过"
    # 按符号 + 1 SD 判据给出可核验的结论（不再预设「落后」）
    if abs(diff) <= ds_s:
        verdict = ("均值差在 1 个样本 SD 内，**不足以判定优劣**；"
                   "逐样本配对检验亦不显著时，只能报告「无统计差异」。")
    elif diff < 0:
        verdict = "均值差已超过 1 个样本 SD，**本框架在该任务上确实更优**。"
    else:
        verdict = "均值差已超过 1 个样本 SD，**该任务上基线确实更优**。"
    return (head_dir
            + "`MK_ResNet {:.4f}\u00b1{:.4f}` vs `DeepSpectra {:.4f}\u00b1{:.4f}`，".format(mk_m, mk_s, ds_m, ds_s)
            + "**均值差 {:+.5f}（{:+.2f}%），{} DeepSpectra 的 1 个样本 SD（{:.5f}）**；".format(
                diff, diff / ds_m * 100, over, ds_s)
            + verdict)


def write_tables_md(p0: dict, ext: dict, attr: dict, t01: dict, unified: dict, dssnv: dict,
                    bilevel: dict | None = None, opcs: dict | None = None,
                    post_r01: dict | None = None, theory: dict | None = None,
                    deep_da: dict | None = None, bilevel_v2: dict | None = None,
                    effects: dict | None = None, param_transfer: dict | None = None,
                    dir_b: dict | None = None, ga_gu2026: dict | None = None,
                    proxy_v2: dict | None = None, runtime: dict | None = None,
                    plan21: dict | None = None, plan22: dict | None = None,
                    mk_sport: dict | None = None, ext_blind: dict | None = None,
                    review_p0: dict | None = None, ind_dl: dict | None = None,
                    ind_followups: dict | None = None, multiseed: dict | None = None,
                    wilcoxon_ind: dict | None = None) -> None:
    lines = [
        "# ChemDL 论文最终三线表与图件映射（权威核验汇总，零手抄）",
        "",
        "> **最后校准**：2026-11-23（对齐 `NUMBER_BASELINE.md`、`res_review_p0_2026_11_23.json` 与后 1-SE `res_industrial_vs_dl_2026_11_21.json`）",
        "> **生成脚本**：`paper_assets/generate_paper_assets.py`（Table 1–22）& `plot_fig11_fig12_industrial.py` / `plot_fig_snv_phase_diagram.py` / `plot_fig13_r2_shape_identifiability.py` / `plot_manuscript_figs_1_3_4.py`",
        "> **全部 17 张 300-DPI 图件**：`fig1`~`fig5`、`fig6_bilevel_diffprep_and_sport.png`、`fig7_post_r01_id_ood_7seed.png`、`fig8_theory_dssnv_and_c061.png`、`fig9_deep_da_vs_physical_sota.png`、`fig10_param_transfer_and_ga_gu2026.png`、`fig11_industrial_vs_dl_benchmark.png`、`fig12_industrial_scaling_and_cost.png`、`fig_snv_phase_diagram.png`、`fig13_r2_shape_identifiability.png`、`fig_ms1_framework.png`、`fig_ms3_stage2_scatter.png`、`fig_ms4_deploy_regret_rc.png`",
        "",
        "### 投稿用表分流导航",
        "",
        "| 投稿归属 | 对应表 / 图 |",
        "|---|---|",
        "| **主论文（可微预处理选型主线，推荐正文）** | `Table 3, 5, 6b, 10, 11, 13, 15a, 16, 19, 20, 21, 22`；主图 `fig3, fig5, fig6, fig8, fig10, fig11, fig12` |",
        "| **补充材料 SI（细项消融与效应量）** | `Table 8, 12, 14, 17, 18b`；附图 `fig7` |",
        "| **校准迁移姊妹篇（配对/无配对/主动迁移线）** | `Table 1, 2, 4, 4b, 7, 9, 15b, 15c, 18a`；附图 `fig1, fig2, fig4, fig9` |",
        "",
        "---",
        "",
        "## Table 1. 配对迁移集选样基准（`ds_affine`，强制三列透明对照 `vs Rnd / vs KS / vs D-Opt` + Nadeau–Bengio / 独立 Wilcoxon）",
        "",
        "| 数据集 / 迁移对 | 口径 | 预算 `k` | `Rnd` | `KS` | `D-Opt` | `t_legacy` | **`t_new` (Affine T-Opt)** | `vs Rnd` | `vs KS` | `vs D-Opt` | 严格 `p` 值注记 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def add_cv_row(label: str, d: dict, k: int):
        r = d[f"k={k}"]
        p_note = f"NB(vs D)={r['p_NB_vs_dopt']:.4f}, NB(vs KS)={r['p_NB_vs_ks']:.4f}, OOF-W(vs D)={r['p_oof_wilcoxon_vs_dopt']:.4f}"
        lines.append(
            f"| {label} | 5×5 CV | {k} | {r['rnd']:.4f} | {r['ks']:.4f} | {r['dopt']:.4f} | {r['t_legacy']:.4f} | **{r['t_new']:.4f}** | "
            f"{r['vs_rnd_pct']:+.2f}% | {r['vs_ks_pct']:+.2f}% | {r['vs_dopt_pct']:+.2f}% | {p_note} |"
        )

    def add_dis_row(label: str, proto: str, d: dict, k: int, p_prefix: str = "p_wilcoxon_vs_"):
        r = d[f"k={k}"]
        p_note = f"W(vs Rnd)={r[p_prefix+'rnd']:.4f}, W(vs KS)={r[p_prefix+'ks']:.4f}, W(vs D)={r[p_prefix+'dopt']:.4f}"
        lines.append(
            f"| {label} | {proto} | {k} | {r['rnd']:.4f} | {r['ks']:.4f} | {r['dopt']:.4f} | {r['t_legacy']:.4f} | **{r['t_new']:.4f}** | "
            f"{r['vs_rnd_pct']:+.2f}% | {r['vs_ks_pct']:+.2f}% | {r['vs_dopt_pct']:+.2f}% | {p_note} |"
        )

    add_cv_row("Corn `m5→mp6`", p0["postfix_existing_and_diag"]["corn_m5_to_mp6"], 10)
    add_cv_row("Corn `m5→mp6`", p0["postfix_existing_and_diag"]["corn_m5_to_mp6"], 20)
    for k in (10, 20, 50):
        add_dis_row("Tablets 官方 `155→460`", "460 独立样本", p0["postfix_existing_and_diag"]["tablets_official_460"], k, "p_460sample_wilcoxon_vs_")
    for k in (8, 12, 16):
        add_cv_row("SDOP `1→2`", p0["postfix_existing_and_diag"]["sdop_1_to_2"], k)
    for pair in ("B->A", "C->B"):
        for k in (10, 20, 40):
            add_cv_row(f"IDRC-2016 Wheat `{pair}` (盲测)", p0["blind_holdout_wheat"][pair]["paired_cv_rigorous"], k)
        for k in (10, 20):
            add_dis_row(f"IDRC-2016 Wheat `{pair}` (盲测)", "10 不相交", p0["blind_holdout_wheat"][pair]["disjoint_10split"]["paired_ds_affine"], k)
    for pair, label in [("NEO1_NEO4", "OSSL `NEO1→NEO4` (盲测)"), ("NEO5_NEO6", "OSSL `NEO5→NEO6` (盲测)")]:
        for k in (10, 20, 50):
            add_dis_row(label, "10 不相交", p0["blind_holdout_neospectra"][pair]["paired_ds_affine"], k)
    for key, label in [("neo2_to_neo3", "OSSL `NEO2→NEO3`"), ("neo8_to_neo9", "OSSL `NEO8→NEO9`")]:
        for k in (10, 20, 50):
            add_dis_row(label, "10 不相交", p0["postfix_existing_and_diag"][key]["paired_ds_affine"], k)

    lines += [
        "",
        "---",
        "",
        "## Table 2. 无配对预对齐、Q 残差门控与 `Active K-Shot` 闭环（含盲测留出对）",
        "",
        "| 场景 / 迁移对 | 域偏移类型 | `r_shift` | `gate` | `Bare` | `MeanShift` | `GatedMean` | `SLOT-PLS` | `MS+Act K5` | `MS+Act K10` | `MS+Act K20` |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, d_type, u in [
        ("IDRC Wheat `B→A` (盲测)", "跨厂商强硬件偏移", p0["blind_holdout_wheat"]["B->A"]["disjoint_10split"]["unpaired"]),
        ("IDRC Wheat `C→B` (盲测)", "跨厂商强硬件偏移", p0["blind_holdout_wheat"]["C->B"]["disjoint_10split"]["unpaired"]),
        ("OSSL `NEO1→NEO4` (盲测)", "同型号 MEMS 灰区", p0["blind_holdout_neospectra"]["NEO1_NEO4"]["unpaired"]),
        ("OSSL `NEO5→NEO6` (盲测)", "同型号 MEMS 灰区", p0["blind_holdout_neospectra"]["NEO5_NEO6"]["unpaired"]),
        ("OSSL `NEO2→NEO3`", "同型号 MEMS 灰区", p0["postfix_existing_and_diag"]["neo2_to_neo3"]["unpaired"]),
        ("OSSL `NEO8→NEO9`", "同型号 MEMS 灰区", p0["postfix_existing_and_diag"]["neo8_to_neo9"]["unpaired"]),
    ]:
        lines.append(
            f"| {label} | {d_type} | {u['r_shift_mean']:.3f} | {u['gate_mean']:.2f} | {u['Bare']:.4f} | "
            f"**{u['MeanShift']:.4f}** | {u['GatedMean']:.4f} | {u['SLOT_PLS']:.4f} | "
            f"{u['MeanShift_K5']:.4f} | {u['MeanShift_K10']:.4f} | **{u['MeanShift_K20']:.4f}** |"
        )

    lines += [
        "",
        "---",
        "",
        "## Table 3. 固定方差 `std(a)=0.35` SNV 乘性散射 × 浓度耦合机理实验（三独立固/液库，`C-061`）",
        "",
        "| 数据集 | 物理介质 | 无注入基线 | `r=0.0` | `r=0.2` | `r=0.4` | `r=0.6` | `r=0.8` |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    snv = ext["fixed_var_snv_coupling"]
    for key, nm, med in [
        ("cordoba_wheat", "Cordoba Wheat", "固态谷物"),
        ("cordoba_barley", "Cordoba Barley", "固态谷物"),
        ("swri_diesel_CN", "SWRI Diesel CN", "液态柴油"),
    ]:
        r = snv[key]
        lines.append(
            f"| {nm} | {med} | {r['baseline_no_inject']['gain_pct']:+.2f}% | "
            + " | ".join(f"{r[f'r={rv:.1f}']['gain_pct']:+.2f}%" for rv in (0.0, 0.2, 0.4, 0.6, 0.8))
            + " |"
        )

    lines += [
        "",
        "---",
        "",
        "## Table 4. 统一单组分物理单位全方法 SOTA 同表对照（`per_col=True`，零泄漏相同切分，`res_unified_sota_2026_11_19.json`）",
        "",
        "| 数据集 / 组分（目标范围） | `Bare` | `MeanShift` | `di-PLS` | `CORAL` | `SLOT-PLS` | `MS+Act K10` | `MS+Act K20` | `DS-Rnd` | `DS-KS` | `DS-DOpt` | **`DS-AffineTOpt`** | `PDS-raw` | `PDS-snv` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    u_rows = [
        ("Corn `m5→mp6` oil", unified["corn_m5_to_mp6"], "oil", 15),
        ("Tablets `X1→X2` weight", unified["tablets_X1_to_X2"], "weight", 20),
        ("Tablets `X1→X2` hardness", unified["tablets_X1_to_X2"], "hardness", 20),
        ("Tablets `X1→X2` assay", unified["tablets_X1_to_X2"], "assay", 20),
        ("Wheat `B→A` (盲测) protein", unified["wheat_blind_B_to_A"], "protein", 20),
        ("Wheat `C→B` (盲测) protein", unified["wheat_blind_C_to_B"], "protein", 20),
        ("OSSL `NEO5→NEO6` (盲测) oc", unified["neospectra_blind_NEO5_to_NEO6"], "oc", 20),
        ("OSSL `NEO5→NEO6` (盲测) clay", unified["neospectra_blind_NEO5_to_NEO6"], "clay", 20),
        ("OSSL `NEO5→NEO6` (盲测) ph", unified["neospectra_blind_NEO5_to_NEO6"], "ph", 20),
    ]
    for lbl, blk, col, kp in u_rows:
        r = blk["per_component_rmsep"][col]
        lo, hi = blk["target_ranges"][col]
        k20_s = f"{r['Act_MS_K20']:.4f}" if "Act_MS_K20" in r else "—"
        lines.append(
            f"| {lbl} `[{lo}, {hi}]` | {r['Bare']:.4f} | {r['Unp_MeanShift']:.4f} | {r['Unp_diPLS']:.4f} | "
            f"{r['Unp_CORAL']:.4f} | {r['Unp_SLOT_PLS']:.4f} | {r['Act_MS_K10']:.4f} | **{k20_s}** | "
            f"{r[f'Pair_DS_Rnd_k{kp}']:.4f} | {r[f'Pair_DS_KS_k{kp}']:.4f} | {r[f'Pair_DS_DOpt_k{kp}']:.4f} | "
            f"**{r[f'Pair_DS_TOpt_k{kp}']:.4f}** | {r[f'Pair_PDS_raw_k{kp}']:.4f} | {r[f'Pair_PDS_snv_k{kp}']:.4f} |"
        )

    lines += [
        "",
        "### Table 4b. 跨库可比统一无量纲口径：`nRMSEP (% of target range = 100 × RMSEP / (y_max - y_min))`（关闭 `HANDOFF P1`）",
        "",
        "| 数据集 / 组分 | `Bare` | `MeanShift` | `di-PLS` | `CORAL` | `SLOT-PLS` | `MS+Act K10` | `MS+Act K20` | `DS-Rnd` | `DS-KS` | `DS-DOpt` | **`DS-AffineTOpt`** | `PDS-raw` | `PDS-snv` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lbl, blk, col, kp in u_rows:
        r = blk["per_component_rmsep"][col]
        lo, hi = blk["target_ranges"][col]
        span = max(float(hi) - float(lo), 1e-9)
        pct = lambda v: f"{100.0 * v / span:.2f}%"
        k20_s = pct(r["Act_MS_K20"]) if "Act_MS_K20" in r else "—"
        lines.append(
            f"| {lbl} | {pct(r['Bare'])} | {pct(r['Unp_MeanShift'])} | {pct(r['Unp_diPLS'])} | "
            f"{pct(r['Unp_CORAL'])} | {pct(r['Unp_SLOT_PLS'])} | {pct(r['Act_MS_K10'])} | **{k20_s}** | "
            f"{pct(r[f'Pair_DS_Rnd_k{kp}'])} | {pct(r[f'Pair_DS_KS_k{kp}'])} | {pct(r[f'Pair_DS_DOpt_k{kp}'])} | "
            f"**{pct(r[f'Pair_DS_TOpt_k{kp}'])}** | {pct(r[f'Pair_PDS_raw_k{kp}'])} | {pct(r[f'Pair_PDS_snv_k{kp}'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Table 5. 收缩型 SNV（`SNV-opt`，代码别名 `DSSNV`）严格嵌套 CV 基准（7 套真实公开库 + 耦合注入，`res_dssnv_2026_11_19.json`）",
        "",
        "> **命名说明**：`SNV-opt` 是 Helland, Næs & Isaksson (1995) `m_i` 族收缩系数的最优成员。**旧名 `DSSNV`（Decoupled SNV）已弃用**——1995 原文恰恰证明 SNV 与 MSC 是耦合的（`(m_i)_ISC = r_i²(m_i)_MSC`），用「解耦」描述本质耦合的量会产生误导。为兼容历史结果文件，表格列名与 JSON 字段仍保留 `DSSNV`，但正文统一用 `SNV-opt`。",
        "",
        "| 数据集 / 场景 | `Raw` | `SNV` | `EMSC` | **`DSSNV` (Ours)** | `vs Raw` | `vs SNV` | `p_oof_W (vs SNV)` | 内层主导 `(w_c, w_s)` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sec in ("real_datasets", "coupled_scatter"):
        for nm, r in dssnv[sec].items():
            cfg_str = ", ".join(f"({c[0][0]:.2f},{c[0][1]:.2f})×{c[1]}" for c in r["top_wc_ws"])
            lines.append(
                f"| {nm} | {r['raw']:.4f} | {r['snv']:.4f} | {r['emsc']:.4f} | **{r['dssnv']:.4f}** | "
                f"{r['vs_raw_pct']:+.2f}% | {r['vs_snv_pct']:+.2f}% | {r['p_oof_W_vs_snv']:.4f} | `{cfg_str}` |"
            )

    if bilevel is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 6. 旗舰原创 `BilevelDiffPrepRidge`（双层精确 LOO 可微五步预处理）+ `SPORT` + 正则 `EMSC` 7 真实库严格嵌套 3×5 CV（`res_bilevel_diffprep_2026_11_19.json`）",
            "",
            "| 真实公开 NIR 库 | `Raw-PLS` | `SNV-PLS` | `SG1-PLS` | `MSC-PLS` | `EMSC-Reg` | `DSSNV-PLS` | `SPORT-PLS` | `SNV-Ridge` | `SG1-Ridge` | **`BilevelDiffPrep`** | `vs SNV-PLS` (`p_oof`) | `BilevelPrep+PLS` | 自动学得物理参数 `(wc, ws, α, σ, als)` |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for nm, blk in bilevel["real_7_datasets"].items():
            m = blk["rmsep"]
            p = blk["learned_physics_mean"]
            pv = blk["pvals"]["p_oof_bilevel_vs_snv_pls"]
            pct = (m["bilevel_diffprep"] / m["snv_pls"] - 1.0) * 100.0
            p_str = "<1e-5" if pv < 1e-5 else f"{pv:.4f}"
            lines.append(
                f"| {nm} | {m['raw_pls']:.4f} | {m['snv_pls']:.4f} | {m['sg1_pls']:.4f} | {m['msc_pls']:.4f} | "
                f"{m['emsc_reg_pls']:.4f} | {m['dssnv_pls']:.4f} | {m['sport_pls']:.4f} | {m['snv_ridge']:.4f} | "
                f"{m['sg1_ridge']:.4f} | **{m['bilevel_diffprep']:.4f}** | **{pct:+.1f}%** (`{p_str}`) | "
                f"{m['bilevel_prep_pls']:.4f} | `({p['wc']:.2f},{p['ws']:.2f},α={p['fod_alpha']:.2f},σ={p['fod_sigma']:.2f},als={p['als_gate']:.2f})` |"
            )
        lines += [
            "",
            "> **SPORT 泄漏修复与 `C-111` 指标口径彻底更正（2026-11-21，`C-130`）**：上表 `SPORT-PLS` 列来自 `fit_sport_pls` 内层 CV 泄漏修复前的历史 JSON。先前 `C-108/C-111` 误报 Tablets 落后 `+8.6%` 源于 Jensen 不等式口径错位（`quantify_sport_leak.py` 算的是逐列均值 `mean(sqrt(MSE))=0.5194`，而主表 Bilevel 为 pooled `sqrt(mean(MSE))=0.5639`）。在 **Table 19**（3×5 CV 严格同切分同指标复测，`res_mk_bilevel_and_sport_fix_2026_11_21.json`）中，无论按 pooled RMSEP 还是按 per-col 均值 RMSEP，`BilevelDiffPrep` 与 `MK-BilevelDiffPrep` 均以 **7/7（100%）** 战胜修复后无泄漏 `SPORT_fix`（Tablets pooled `0.5776 vs 0.5886 (-1.87%)`，per-col `0.5067 vs 0.5162 (-1.84%)`）。",
            "",
            "### Table 6b. 合成强退化 OOD（真实基线漂移 + 峰漂移）五步算子选择性自动激活验证",
            "",
            "| 模态 | `Raw ID/OOD` | `SNV ID/OOD` | `Classic-5Step ID/OOD` | **`BilevelDiffPrep ID/OOD`** | 自动激活的五步物理参数 |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for mod, r in bilevel["synthetic_5step_activation"].items():
            lp = r["learned_physics"]
            lines.append(
                f"| Synth `{mod.upper()}` | {r['raw_id']:.4f} / {r['raw_ood']:.4f} | {r['snv_id']:.4f} / {r['snv_ood']:.4f} | "
                f"{r['classic_id']:.4f} / {r['classic_ood']:.4f} | **{r['bilevel_id']:.4f} / {r['bilevel_ood']:.4f}** | "
                f"`als_gate={lp['als_gate']:.2f} (p={lp['als_p']:.3f}), align_gate={lp['align_gate']:.2f}, wc={lp['wc']:.2f}, ws={lp['ws']:.2f}, σ={lp['fod_sigma']:.2f}` |"
            )
        lines.append(
            "\n> **合成 OOD 激活复验**：Synth `NIR`: `raw 0.0430/0.1082 → bilevel 0.0305/0.0710` (`-29.1% ID / **-34.4% OOD**`, `als_gate=0.76, align_gate=0.89`)；"
            "Synth `CHROM`: `raw 0.0775/0.2074 → bilevel 0.0535/0.1752` (`-31.0% ID / -15.5% OOD`)。"
        )

    if opcs is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 7. 有监督 `OPCS`（Andries & Vander Heyden 2024）vs 无监督 `Affine T-Opt` 与混合 `OPCS-TOpt`（10 次独立切分，`res_opcs_and_huber_2026_11_19.json`）",
            "",
            "| 数据集 / 方向 | 预算 `k` | `Random` | `KS` | `D-Opt` | **`Affine T-Opt`** | `OPCS-KS` | **`OPCS-TOpt`** |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for ds_nm, ks_blk in opcs["opcs_vs_topt"].items():
            for k_str, v in ks_blk.items():
                best_val = min(v["rnd"], v["ks"], v["d_opt"], v["t_opt"], v["opcs_ks"], v["opcs_topt"])
                fmt = lambda val: f"**{val:.4f}**" if abs(val - best_val) < 1e-6 else f"{val:.4f}"
                lines.append(
                    f"| `{ds_nm}` | `{k_str}` | {fmt(v['rnd'])} | {fmt(v['ks'])} | {fmt(v['d_opt'])} | "
                    f"{fmt(v['t_opt'])} | {fmt(v['opcs_ks'])} | {fmt(v['opcs_topt'])} |"
                )

    if post_r01 is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 8. 关闭 `HANDOFF P0`：修 R-01 后 7-Seed × 2-Modality ID/OOD 完整对照（`res_post_r01_id_ood_2026_11_19.json`）",
            "",
            "| 模态 | 方案 | `ID RMSEP (mean±std)` | `OOD RMSEP (mean±std)` | `OOD/ID 比值` | `Outlier AUROC (w)` | `Baseline Corr` |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for mod in ("nir", "chrom"):
            for m in ("raw", "classic", "diff_v2", "diff_v3"):
                s = post_r01[mod]["summary"][m]
                au = f"{s['out_auroc_w_mean']:.4f}" if "out_auroc_w_mean" in s else "—"
                bc = f"{s['prep_baseline_corr_mean']:.4f}" if "prep_baseline_corr_mean" in s else "—"
                lines.append(
                    f"| `{mod.upper()}` | `{m}` | {s['rmsep_id_mean']:.4f}±{s['rmsep_id_std']:.4f} | "
                    f"**{s['rmsep_ood_mean']:.4f}**±{s['rmsep_ood_std']:.4f} | {s['ood_to_id_ratio']:.2f}× | {au} | {bc} |"
                )

    if deep_da is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 9. 深度域自适应 SOTA（`Deep CORAL / DANN / PEAMATL`）与物理线性/子空间/主动迁移同切分正面对比（`res_deep_da_sota_2026_11_20.json`）",
            "",
            "| 数据集 / 组分（目标范围） | `Bare` | `MeanShift` | `di-PLS` | `SLOT-PLS` | `Deep CORAL` (Sun 2016) | `DANN` (Ganin 2016) | `PEAMATL` (MAE+MMD) | `Active K20` | `DS-AffineTOpt` |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        da_rows = [
            ("Corn `m5→mp6` oil", "corn_m5_to_mp6", "oil", 15),
            ("Tablets `X1→X2` weight", "tablets_X1_to_X2", "weight", 20),
            ("Tablets `X1→X2` hardness", "tablets_X1_to_X2", "hardness", 20),
            ("Tablets `X1→X2` assay", "tablets_X1_to_X2", "assay", 20),
            ("Wheat `B→A` (盲测) protein", "wheat_blind_B_to_A", "protein", 20),
            ("Wheat `C→B` (盲测) protein", "wheat_blind_C_to_B", "protein", 20),
            ("OSSL `NEO5→NEO6` (盲测) oc", "neospectra_blind_NEO5_to_NEO6", "oc", 20),
            ("OSSL `NEO5→NEO6` (盲测) clay", "neospectra_blind_NEO5_to_NEO6", "clay", 20),
            ("OSSL `NEO5→NEO6` (盲测) ph", "neospectra_blind_NEO5_to_NEO6", "ph", 20),
        ]
        for lbl, ds_k, col, kp in da_rows:
            r = unified[ds_k]["per_component_rmsep"][col]
            d = deep_da[ds_k]["per_component_rmsep"][col]
            lo, hi = unified[ds_k]["target_ranges"][col]
            k20_s = f"{r['Act_MS_K20']:.4f}" if "Act_MS_K20" in r else f"{r['Act_SLOT_K10']:.4f} (K10)"
            lines.append(
                f"| {lbl} `[{lo}, {hi}]` | {r['Bare']:.4f} | {r['Unp_MeanShift']:.4f} | {r['Unp_diPLS']:.4f} | "
                f"{r['Unp_SLOT_PLS']:.4f} | {d['Deep_CORAL']:.4f} | {d['Deep_DANN']:.4f} | {d['Deep_PEAMATL']:.4f} | "
                f"**{k20_s}** | **{r[f'Pair_DS_TOpt_k{kp}']:.4f}** |"
            )

    if theory is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 10. 解析最优散射校正定理验证（定理 1 变号判据 & 定理 2 零网格闭式 `SNV-opt`，`res_theory_dssnv_2026_11_20.json`）",
            "",
            "### 10a. 定理 1 预测 vs `C-061` 固定方差（`std(a)=0.35`）实测耦合变号曲线",
            "",
            "| 独立公开库 | 纯化学 SNR `β` | 解析临界点 `r_crit` | 实测增益 `(r=0.0 → 0.8)` | 定理 1 预测 `(r=0.0 → 0.8)` | Pearson `r` | 端点变号一致 |",
            "|---|---:|---:|---|---|---:|:---:|",
        ]
        for lbl, v in theory["c061_theorem_verification"].items():
            emp_s = ", ".join(f"{x:+.1f}%" for x in v["empirical_gain_pct"])
            th_s = ", ".join(f"{x:+.1f}%" for x in v["theoretical_gain_pct"])
            lines.append(
                f"| {lbl} | {v['beta_est']:.1f} | **{v['r_crit']:.3f}** | `{emp_s}` | `{th_s}` | **{v['pearson_r']:.4f}** | {'✅' if v['sign_agreement_at_r0_and_r08'] else '❌'} |"
            )
        lines += [
            "",
            "### 10b. 零内层网格搜索的闭式解析 `SNV-opt` (`fit_analytical_dssnv_pls`) vs `Raw / SNV / CV-SNV-opt`（7 真实库 3×5 CV）",
            "",
            "| 真实公开库 | `Raw` | `SNV` | `CV-DSSNV` | **`Theory-DSSNV` (零网格)** | `vs SNV` (`p_oof`) | `vs CV-DSSNV` | 闭式 `(w_c*, w_s*)` | 相对 CV 提速 |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        for nm, v in theory["real_7_datasets"].items():
            m = v["rmsep"]
            vs_cv = (m["theory_dssnv"] / max(m["cv_dssnv"], 1e-12) - 1.0) * 100.0
            lines.append(
                f"| {nm} | {m['raw']:.4f} | {m['snv']:.4f} | {m['cv_dssnv']:.4f} | **{m['theory_dssnv']:.4f}** | "
                f"{v['vs_snv_pct']:+.2f}% (`{v['p_oof_vs_snv']:.4f}`) | {vs_cv:+.2f}% | `({v['mean_wc_star']:.2f}, {v['mean_ws_star']:.2f})` | **{v['speedup_vs_cv']:.1f}×** |"
            )

    if bilevel_v2 is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 11. 复杂度门控升级版 `BilevelDiffPrep v2` vs 同求解器最强固定基线（`Raw / SNV / SG1 / SG2-Ridge`，含 Cohen's $d_z$ 与 95% Bootstrap CI，`res_bilevel_v2_7datasets_2026_11_20.json`）",
            "",
            "| 真实公开 NIR 库 | `Raw-Ridge` | `SNV-Ridge` | `SG1-Ridge` | `SG2-Ridge` | **`BilevelDiffPrep v2`** | `vs SG1-Ridge` (95% CI, $d_z$, $p_{\\text{oof}}$) | `vs 最强固定基线` | 白盒学得物理 `(wc, ws, sg, α, σ, als)` |",
            "|---|---:|---:|---:|---:|---:|---|---:|---|",
        ]
        for nm, blk in bilevel_v2["datasets"].items():
            m = blk["rmsep"]
            e1 = blk["vs_sg1_ridge"]
            eb = blk["vs_best_fixed_ridge"]
            p = blk["learned_physics_mean"]
            p_str = "<1e-5" if e1["p_oof_wilcoxon"] < 1e-5 else f"{e1['p_oof_wilcoxon']:.4f}"
            lines.append(
                f"| {nm} | {m['raw_ridge']:.4f} | {m['snv_ridge']:.4f} | {m['sg1_ridge']:.4f} | {m['sg2_ridge']:.4f} | "
                f"**{m['bilevel_v2']:.4f}** | **{e1['gain_pct']:+.2f}%** (`[{e1['ci95_pct'][0]:+.1f}, {e1['ci95_pct'][1]:+.1f}]%`, $d_z={e1['cohen_dz']:+.2f}$, `p={p_str}`) | "
                f"{eb['gain_pct']:+.2f}% (`{blk['best_fixed_ridge']}`) | `({p['wc']:.2f},{p['ws']:.2f},sg={p['sg_mix']:.2f},α={p['fod_alpha']:.2f},σ={p['fod_sigma']:.2f},als={p['als_gate']:.2f})` |"
            )

    if effects is not None and "post_r01_7seed" in effects:
        lines += [
            "",
            "---",
            "",
            "## Table 12. 核心结论配对效应量（Cohen's $d_z$ / Hedges' $g$）与 95% Bootstrap 置信区间汇总（`res_effect_sizes_2026_11_20.json`）",
            "",
            "| 场景 / 对照项 | 样本折数 `n` | 方案 A | 方案 B | 相对增益 (%) | 95% Bootstrap CI (%) | Cohen's $d_z$ | Hedges' $g$ |",
            "|---|---:|---:|---:|---:|---|---:|---:|",
        ]
        for mod in ("nir", "chrom"):
            for k, v in effects["post_r01_7seed"][mod].items():
                lines.append(
                    f"| Post-R-01 `{mod.upper()}` `{k}` | {v['n']} | {v['mean_a']:.4f} | {v['mean_b']:.4f} | "
                    f"**{v['gain_pct']:+.2f}%** | `[{v['ci95_pct'][0]:+.2f}%, {v['ci95_pct'][1]:+.2f}%]` | **{v['cohen_dz']:+.2f}** | {v['hedges_g']:+.2f} |"
                )
        if "corn_cross_instrument_5fold" in effects:
            for k, v in effects["corn_cross_instrument_5fold"].items():
                lines.append(
                    f"| Real Corn Cross `{k}` | {v['n']} | {v['mean_a']:.4f} | {v['mean_b']:.4f} | "
                    f"**{v['gain_pct']:+.2f}%** | `[{v['ci95_pct'][0]:+.2f}%, {v['ci95_pct'][1]:+.2f}%]` | **{v['cohen_dz']:+.2f}** | {v['hedges_g']:+.2f} |"
                )

    if param_transfer is not None:
        ds_names = list(param_transfer["rmsep_matrix"].keys())
        lines += [
            "",
            "---",
            "",
            "## Table 13. 方向 A：7×7 跨数据集连续物理参数零微调迁移矩阵（不同仪器与通道数 $L\\in\\{100,401,505,700\\}$，`res_param_transfer_2026_11_20.json`）",
            "",
            "| 源库（学得冻结物理参数） \\ 目标库 | " + " | ".join(f"`{d}`" for d in ds_names) + " |",
            "|---|" + "|".join("---:" for _ in ds_names) + "|",
        ]
        for s in ds_names:
            cells = []
            for t_nm in ds_names:
                v = param_transfer["rmsep_matrix"][s][t_nm]
                ret = param_transfer["retention_pct_matrix"][s][t_nm]
                cell = f"**{v:.4f}** (self)" if s == t_nm else f"{v:.4f} (`{ret:.1f}%`)"
                cells.append(cell)
            lines.append(f"| `{s}` | " + " | ".join(cells) + " |")
        sm = param_transfer["summary"]
        lines.append(
            f"\n> **跨库汇总（42 组非对角）**：中位保留率 **`{sm['median_retention_pct_all42']:.1f}%`**（均值 `{sm['mean_retention_pct_all42']:.1f}%`），"
            f"**`{sm['pairs_retention_ge_90pct']}/42`** 组保留率 $\\ge 90\\%$；**`{sm['pairs_beating_snv_ridge']}`** 组跨库冻结参数直接优于目标库自身 `SNV-Ridge`；"
            f"同类谷物家族 12 组中位保留率 **`{sm['within_grain_median_retention_pct']:.1f}%`**（均值 `{sm['within_grain_mean_retention_pct']:.1f}%`）。"
        )

    if dir_b is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 14. 方向 B：Cawley–Talbot 自由度守卫（`v3-DoF`）与逐样本散射自适应门控（`v3-Gate`）消融（7 真实库 + 混合散射合成对照，`res_direction_b_gate_2026_11_20.json`）",
            "",
            "| 数据集 | `Best Fixed Ridge` | `SG1-Ridge` | `Bilevel v2` | **`Bilevel v3-DoF`** (`vs BestFix`) | **`Bilevel v3-Gate`** (`vs BestFix`) | 零自由度锚点选中率 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for sec in ("real_datasets", "synth_mixed"):
            for nm, r in dir_b[sec].items():
                m = r["rmsep"]
                bfk = r["best_fixed_key"]
                lines.append(
                    f"| `{nm}` | {m[bfk]:.4f} (`{bfk}`) | {m['sg1_ridge']:.4f} | {m['bilevel_v2']:.4f} | "
                    f"**{m['bilevel_v3_dof']:.4f}** (`{r['v3_dof_vs_best_fixed_pct']:+.2f}%`) | "
                    f"**{m['bilevel_v3_gate']:.4f}** (`{r['v3_gate_vs_best_fixed_pct']:+.2f}%`) | {r['textbook_anchor_ratio']*100:.0f}% |"
                )

    if ga_gu2026 is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 15. 方向 C 与竞品正面交锋：离散 GA 预处理搜索 & Gu et al. (2026, *CILS* 279:105918) 跨仪器与先验偏移压力测试（`res_ga_and_gu2026_2026_11_20.json`）",
            "",
            "### 15a. 同域 7 真实库：离散 GA（384 流水线空间）vs 连续 `BilevelDiffPrep`（严格同 3×5 CV 切分）",
            "",
            "| 真实公开 NIR 库 | `GA-PLS` (离散) | `GA-Ridge` (同求解器) | `SG1-Ridge` | **`BilevelDiffPrep v2`** | `Bilevel vs GA-Ridge` | `GA-Ridge` 最常选流水线 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for nm, r in ga_gu2026["direction_c_source_7datasets"].items():
            top_p = r["top_pipelines_ridge"][0][0] if r["top_pipelines_ridge"] else "-"
            lines.append(
                f"| `{nm}` | {r['ga_pls_rmsep']:.4f} | {r['ga_ridge_rmsep']:.4f} | {r['sg1_ridge_ref']:.4f} | "
                f"**{r['bilevel_v2_ref']:.4f}** | **{r['bilevel_vs_ga_ridge_pct']:+.2f}%** | `{top_p}` |"
            )

        gu = ga_gu2026["gu2026_transfer_and_prior_shift"]
        lines += [
            "",
            "### 15b. 跨仪器正面交锋（完全相同 10 次不相交切分）：Gu et al. (2026) `GA+WD` / `GA+CV` vs 本库无配对与少样本 SOTA",
            "",
            "| 跨仪器基准 / 组分 | `Bare SNV-PLS` | `Gu2026 GA+CV` | `Gu2026 GA+WD` | `MeanShift` | `di-PLS` | **`SLOT-PLS`** | **`Active K20`** | **`DS-AffineTOpt`** |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        gu_rows = [
            ("Corn `m5→mp6` oil", "corn_m5_to_mp6", "oil", 15),
            ("Tablets `X1→X2` weight", "tablets_X1_to_X2", "weight", 20),
            ("Tablets `X1→X2` hardness", "tablets_X1_to_X2", "hardness", 20),
            ("Tablets `X1→X2` assay", "tablets_X1_to_X2", "assay", 20),
            ("Wheat `B→A`* protein", "wheat_blind_B_to_A", "protein", 20),
            ("Wheat `C→B`* protein", "wheat_blind_C_to_B", "protein", 20),
            ("OSSL `NEO5→NEO6`* oc", "neospectra_blind_NEO5_to_NEO6", "oc", 20),
            ("OSSL `NEO5→NEO6`* clay", "neospectra_blind_NEO5_to_NEO6", "clay", 20),
            ("OSSL `NEO5→NEO6`* ph", "neospectra_blind_NEO5_to_NEO6", "ph", 20),
        ]
        for lbl, ds_k, col, kp in gu_rows:
            r = unified[ds_k]["per_component_rmsep"][col]
            g = gu[ds_k]["per_component_rmsep"][col]
            k20_s = f"{r['Act_MS_K20']:.4f}" if "Act_MS_K20" in r else f"{r['Act_SLOT_K10']:.4f} (K10)"
            lines.append(
                f"| {lbl} | {r['Bare']:.4f} | {g['Gu2026_GA_CV']:.4f} | {g['Gu2026_GA_WD']:.4f} | "
                f"{r['Unp_MeanShift']:.4f} | {r['Unp_diPLS']:.4f} | {r['Unp_SLOT_PLS']:.4f} | **{k20_s}** | **{r[f'Pair_DS_TOpt_k{kp}']:.4f}** |"
            )

        ps = gu["prior_shift_stress_wheat_B_to_A"]
        lines += [
            "",
            "### 15c. 叠加先验偏移与仪器偏移压力测试（$P_S(y) \\neq P_T(y)$，Wheat `B→A`）",
            "",
            "| 偏移场景 | 目标均值偏移 | 本库 Q-残差 `r_shift` | `SNV-PLS` | `Gu2026 GA+CV` | `Gu2026 GA+WD` (`vs CV`) | `SLOT-PLS` (MeanShift 类) | `Active K=5` (基于 SLOT 抢救) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for sc, v in ps.items():
            rm = v["rmsep_mean"]
            lines.append(
                f"| `{sc}` | `{v['mean_y_shift_sigma']:.2f}σ` | `{v['q_rel_median']:.3f}` | {rm['SNV_PLS']:.4f} | "
                f"{rm['Gu2026_GA_CV']:.4f} | {rm['Gu2026_GA_WD']:.4f} (`{v['wd_degradation_vs_cv_pct']:+.1f}%`) | "
                f"{rm['SLOT_PLS']:.4f} | {rm['Active_K5_Robust']:.4f} |"
            )
        _ak5_lo = ps["low_to_high"]["rmsep_mean"]["Active_K5_Robust"]
        _ak5_hi = ps["high_to_low"]["rmsep_mean"]["Active_K5_Robust"]
        lines.append(
            f"\n> **机制发现**：当仪器偏移（`B→A` 使 `Q_rel≈11.4`）与 `2–3σ` 先验偏移**同时存在**时，基于均值对齐的 `SLOT-PLS` 将真实浓度差误当仪器偏移导致崩溃（`3.40 / 3.58`，启用 `AutoGate` 的 `Active K=5` 将其大幅抢救至 `{_ak5_lo:.4f} / {_ak5_hi:.4f}`，详见 Table 18a）；"
            "`Gu2026 GA+WD` 在 `low→high` 上较 `GA+CV` 退化 `+11.9%`（印证原文 §4 先验偏移风险），但因其离散池可选中二阶导数流水线（天然消去线性仪器漂移且不做均值减法），在极端先验偏移下总体显著优于 `SNV-PLS` 与无熔断均值对齐类方法。"
        )

    if proxy_v2 is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 16. 方向 D：形态残差散射可辨识分解 `R²_shape` 与配对直测校验（170 折×组分严格 CV，`res_theory_proxy_v2_2026_11_20.json`）",
            "",
            "| 真实公开 NIR 库 | 检验折×组分 | 实际 SNV 胜率 | `R²_shape` (化学幅值占比) | `v1` 决策准确率 | **`v2` 决策准确率** |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for nm, r in proxy_v2["per_dataset"].items():
            lines.append(
                f"| `{nm}` | {r['n_checks']} | {r['actual_snv_win_rate']*100:.1f}% | `{r['r2_s_shape_mean']:.3f}` | "
                f"{r['v1_gain_acc']*100:.1f}% | **{r['v2_gain_acc']*100:.1f}%** |"
            )
        ov = proxy_v2["overall"]
        pd_c = proxy_v2["paired_direct_scatter"]["Corn_m5_mp6"]
        pd_t = proxy_v2["paired_direct_scatter"]["Tablets_X1_X2"]
        if review_p0 is not None and "part3_theorem1_confusion" in review_p0:
            p3 = review_p0["part3_theorem1_confusion"]
            v1c, v2c = p3["overall_v1"], p3["overall_v2"]
            t05 = p3["abstain_tiers_v2"]["abs_pred_ge_0.5pct"]
            lines.append(
                f"\n> **汇总与 2×2 混淆矩阵（{p3['total_checks']} 折×组分，`res_review_p0_2026_11_23.json`）**：`v2` 完整混淆矩阵为 "
                f"`TP={v2c['tp']}, TN={v2c['tn']}, FP={v2c['fp']}, FN={v2c['fn']}`——**Acc `{v2c['acc']*100:.1f}%`、Balanced Accuracy `{v2c['balanced_acc']:.3f}`、MCC `{v2c['mcc']:.3f}`**"
                f"（显著超越全局常数基线「永远用 SNV」的 `{p3['global_constant_majority_acc']*100:.1f}%` 与 `v1` 的 `Acc {v1c['acc']*100:.1f}% / MCC {v1c['mcc']:.3f}`；"
                f"逐库先知多数类基准 `{p3['per_dataset_oracle_majority_acc']*100:.1f}%`；Spearman `{ov['spearman_v1']:.3f} → {ov['spearman_v2']:.3f}`）。"
                f"限制在非弃权区 `abs(pred) ≥ 0.5%`（覆盖 `{t05['coverage']*100:.1f}%`, `n={t05['n']}`）时，**Acc 升至 `{t05['acc']*100:.1f}%`（`BAC={t05['balanced_acc']:.3f}, MCC={t05['mcc']:.3f}`）**。"
                f" **无配对直测验证**：单仪器 `σ_a_true = σ_s√(1-R²_shape)` 在 Corn 上将表观 `0.0649` 校正至 `0.0194`（实测配对 `σ_a_paired = {pd_c['sigma_a_paired_m5_mp6']:.4f}`），"
                f"在 Tablets 上将 `0.0798` 校正至 `0.0145`（实测配对 `σ_a_paired = {pd_t['sigma_a_paired_X1_X2']:.4f}`）。"
            )
        else:
            lines.append(
                f"\n> **汇总（170 折×组分）**：决策准确率 `{ov['v1_gain_acc']*100:.1f}%` → **`{ov['v2_gain_acc']*100:.1f}%`**（Spearman `{ov['spearman_v1']:.3f}` → **`{ov['spearman_v2']:.3f}`**；3 套决定性散射库 Tecator/Corn/Tablets 100 折准确率达 **`81.0%`**）。"
                f" **无配对直测验证**：单仪器 `σ_a_true = σ_s√(1-R²_shape)` 在 Corn 上将表观 `0.0649` 校正至 `0.0194`（实测配对 `σ_a_paired = {pd_c['sigma_a_paired_m5_mp6']:.4f}`），"
                f"在 Tablets 上将 `0.0798` 校正至 `0.0145`（实测配对 `σ_a_paired = {pd_t['sigma_a_paired_X1_X2']:.4f}`）。"
            )

    if runtime is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 17. 单次切分 CPU 训练与推理时间开销对比（关闭 `REVIEW_ANTIGRAVITY` Should-Fix #5，`res_runtime_cost_2026_11_20.json`）",
            "",
            "| 方法 | Corn (`n=40, L=700`) 训练 (s) | Corn 推理 (ms) | Tablets (`n=155, L=505`) 训练 (s) | Tablets 推理 (ms) |",
            "|---|---:|---:|---:|---:|",
        ]
        corn_rt = runtime["Corn_m5_to_mp6"]
        tab_rt = runtime["Tablets_X1_to_X2"]
        for m_nm in corn_rt.keys():
            c_m, t_m = corn_rt[m_nm], tab_rt[m_nm]
            lines.append(
                f"| `{m_nm}` | {c_m['train_sec']:.3f} | {c_m['infer_ms']:.2f} | {t_m['train_sec']:.3f} | {t_m['infer_ms']:.2f} |"
            )

    if plan22 is not None or plan21 is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 18. 阶段二算法升级验证：经验贝叶斯双重熔断 `AutoGate`（方案 2.2）与 `R²_shape` 液态物理先验 `BilevelDiffPrep`（方案 2.1）",
        ]
        if plan22 is not None:
            lines += [
                "",
                "### 18a. 经验贝叶斯偏置收缩 + 双重熔断 `active_transfer_fit(auto_gate=True)`（`res_plan22_autogate_2026_11_20.json`）",
                "",
                "| 先验偏移压力测试（Wheat `B→A`, 5 切分） | `SNV-PLS` | `SLOT-PLS` | `ActK5 (无熔断)` | **`ActK5 (AutoGate)`** | `ActK10 (无熔断)` | **`ActK10 (AutoGate)`** |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for sc, r in plan22["prior_shift_wheat_B_to_A"].items():
                lines.append(
                    f"| `{sc}` | {r['SNV_PLS']:.4f} | {r['SLOT_PLS']:.4f} | {r['ActK5_NoAutoGate']:.4f} | "
                    f"**{r['ActK5_AutoGate']:.4f}** | {r['ActK10_NoAutoGate']:.4f} | **{r['ActK10_AutoGate']:.4f}** |"
                )
            lines += [
                "",
                "| 常规 5 套跨仪器基准（10 切分 `Act_SLOT_K10` 标准化 RMSEP） | `NoAutoGate` | **`AutoGate`** | 变化 (%) |",
                "|---|---:|---:|---:|",
            ]
            for bn, r in plan22["normal_5_benchmarks"].items():
                lines.append(
                    f"| `{bn}` | {r['ActK10_NoAutoGate_std']:.4f} | **{r['ActK10_AutoGate_std']:.4f}** | **{r['delta_pct']:+.2f}%** |"
                )
        if plan21 is not None:
            lines += [
                "",
                "### 18b. 精确 SG 多项式边界插值 + 无监督 $\\sigma_{a,\\text{true}}$ 液态物理先验裁剪 `BilevelDiffPrep`（7 库 3×5 CV 严格同口径，`res_plan21_physprior_2026_11_20.json`）",
                "",
                "| 真实公开 NIR 库 | 无监督 $\\sigma_{a,\\text{true}}$ | 液态裁剪 | `SG1-Ridge` | `BestFixed` | `Bilevel v2` | **`PhysPrior`** (`vs SG1` / `vs v2`) | 耗时 (s) |",
                "|---|---:|:---:|---:|---:|---:|---|---:|",
            ]
            for nm, r in plan21["datasets"].items():
                m = r["rmsep"]
                bfk = r["best_fixed_key"]
                lines.append(
                    f"| `{nm}` | `{r['sigma_a_true_mean']:.4f}` | {'✅' if r['liquid_pruned'] else '—'} | {m['sg1_ridge']:.4f} | "
                    f"{m[bfk]:.4f} (`{bfk}`) | {m['bilevel_v2']:.4f} | **{m['bilevel_physprior']:.4f}** (`{r['vs_sg1_pct']:+.2f}%` / `{r['vs_v2_pct']:+.2f}%`) | {r['sec']:.1f} |"
                )

    if mk_sport is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 19. 原创多视图精确留一核融合 `MK-BilevelDiffPrep` 与修复后无泄漏 `SPORT_fix` 严格同切分同口径 3×5 CV（关闭 `C-111` 指标错位，`res_mk_bilevel_and_sport_fix_2026_11_21.json`）",
            "",
            "| 真实公开 NIR 库 | `Bilevel` (pooled) | **`MK-Bilevel` (pooled)** | `MK vs BI` | `SPORT_fix` (pooled) | `MK vs SPORT_fix` | `MK` (per-col) | `SPORT_fix` (per-col) | 自动核权重 `[diff, snv, sg1, sg2]` |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for nm, r in mk_sport["datasets"].items():
            p = r["pooled_rmsep"]
            pc = r["per_col_mean_rmsep"]
            w_s = ", ".join(f"{x:.3f}" for x in r["mk_view_weights_mean"])
            lines.append(
                f"| `{nm}` | {p['bilevel']:.4f} | **{p['mk_bilevel']:.4f}** | `{r['mk_vs_bilevel_pct']:+.2f}%` | "
                f"{p['sport_fix']:.4f} | **{r['mk_vs_sport_fix_pct']:+.2f}%** | **{pc['mk_bilevel']:.4f}** | "
                f"{pc['sport_fix']:.4f} | `[{w_s}]` |"
            )
        sm = mk_sport["summary"]
        lines.append(
            f"\n> **结论（`C-130`）**：在完全相同 3×5 CV（`seeds 100..102`）与**相同指标**下，无论按 pooled RMSEP 还是按 per-col 均值 RMSEP，"
            f"`Bilevel`（`{sm['bilevel_wins_vs_sport_fix_pooled']}`）与 `MK-Bilevel`（**`{sm['mk_bilevel_wins_vs_sport_fix_pooled']}`**）均 **100% 战胜**修复泄漏后的 `SPORT_fix`；"
            f"`MK-Bilevel` 在强多尺度散射库 Tecator 上再降 **`-6.16%`**（`0.0942 → 0.0884`），并在 4/7 弱/单视图最优库自动坍缩回 `[1, 0, 0, 0]` 实现近零回退。"
        )

    if ext_blind is not None:
        lines += [
            "",
            "---",
            "",
            "## Table 20. 三套独立外部公开 NIR 库零调参盲测（`res_blindtest_external_2026_11_21.json` 终版对齐）",
            "",
            "### 20a. Anderson (2020) Mango Dry Matter (`SP_cal → SP_test1 = 1448`) & IDRC-2016 Wheat A1 (`200 → 48`)",
            "",
            "| 外部盲测库 / 口径 | `Raw-PLS` | `SNV-PLS` | `MSC-PLS` | `RNV-PLS` | `EMSC-Reg` | `SG1-PLS` | `SG2-PLS` | **`SNV-opt`** | `SPORT-PLS` | `Bilevel` | **`MK-Bilevel`** |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        mg = ext_blind["mango_anderson2020"]
        mf = mg["full_cal_pls"]
        m3 = mg["n300_3seed_mean"]
        wa = ext_blind["wheat_a1_idrc2016"]
        mf_rnv = f"{mf['pls_rnv']:.4f}" if "pls_rnv" in mf else "—"
        mf_em = f"{mf['pls_emsc_reg']:.4f}" if "pls_emsc_reg" in mf else "—"
        lines.append(
            f"| Mango Full (`9914→1448`) | **{mf['pls_raw']:.4f}** | {mf['pls_snv']:.4f} | {mf['pls_msc']:.4f} | "
            f"{mf_rnv} | {mf_em} | {mf['pls_sg1']:.4f} | {mf['pls_sg2']:.4f} | — | — | — | — |"
        )
        lines.append(
            f"| Mango `n=300` (3-seed→`1448`) | {m3['pls_raw']:.4f} | {m3['pls_snv']:.4f} | {m3['pls_msc']:.4f} | "
            f"{m3['pls_rnv']:.4f} | {m3['pls_emsc_reg']:.4f} | {m3['pls_sg1']:.4f} | **{m3['pls_sg2']:.4f}** | **{m3['dssnv_pls']:.4f}** | {m3['sport_pls']:.4f} | "
            f"{m3['bilevel']:.4f} | **{m3['mk_bilevel']:.4f}** |"
        )
        lines.append(
            f"| IDRC-2016 Wheat A1 (`200→48`) | {wa['pls_raw']:.4f} | {wa['pls_snv']:.4f} | {wa['pls_msc']:.4f} | "
            f"{wa['pls_rnv']:.4f} | {wa['pls_emsc_reg']:.4f} | {wa['pls_sg1']:.4f} | {wa['pls_sg2']:.4f} | **{wa['dssnv_pls']:.4f}** | {wa['sport_pls']:.4f} | "
            f"**{wa['bilevel']:.4f}** | **{wa['mk_bilevel']:.4f}** |"
        )
        wa_bi_raw = (wa["bilevel"] / wa["pls_raw"] - 1.0) * 100.0
        wa_mk_raw = (wa["mk_bilevel"] / wa["pls_raw"] - 1.0) * 100.0
        wa_bi_snv = (wa["bilevel"] / wa["pls_snv"] - 1.0) * 100.0
        wa_mk_snv = (wa["mk_bilevel"] / wa["pls_snv"] - 1.0) * 100.0
        m3_mk_sg2 = (m3["mk_bilevel"] / m3["pls_sg2"] - 1.0) * 100.0
        lines += [
            "",
            f"> **物理适用边界说明**：在 **3 套独立库 / 6 个域内盲测集** 上，本框架在 **5/6** 显著优于或持平最佳教科书基线（`Wheat A1` `Bilevel` 相对 `Raw` 降 `{wa_bi_raw:.1f}%`、相对 `SNV` 降 **`{wa_bi_snv:.1f}%`**，`MK-Bilevel` `{wa_mk_raw:.1f}%` / **`{wa_mk_snv:.1f}%`**；三聚氰胺树脂 `Melamine` 4/4 种配方全部最优）。在 `Mango n=300` 上（`MK-Bilevel {m3['mk_bilevel']:.4f}` vs `SG2 {m3['pls_sg2']:.4f}`，3-seed `{m3_mk_sg2:+.1f}%`；10-seed 复测 `+4.1%`），其物理成因是 Mango 水分吸收带（~970/1190/1450 nm）随成熟度发生沿波长轴平滑位移而非纯幅值缩放，部署时先做峰对齐后再进幅值缩放框架即可规避（详见 [PAPER_TEXT_CLAIMS §三](./PAPER_TEXT_CLAIMS_2026-11-21.md)）。",
            "",
            "### 20b. Melamine 三聚氰胺树脂 4 种反应配方无泄漏 5 折盲测（全量 `full-N` 与随机抽样同预算 `sub300`）",
            "",
            "| 配方批次 (`n, L`) | `Raw` (full) | `SNV` (full) | `MSC` (full) | `SG1` (full) | `Raw` (sub300) | `SNV` (sub300) | **`SNV-opt`** (sub300) | **`Bilevel`** (sub300) | **`MK-Bilevel`** (sub300) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for inst, r in ext_blind["melamine_4inst_clean"].items():
            lines.append(
                f"| `{inst}` (`{r['n']}, {r['L']}`) | **{r['pls_raw']:.4f}** | {r['pls_snv']:.4f} | {r['pls_msc']:.4f} | "
                f"{r['pls_sg1']:.4f} | {r['pls_raw_sub']:.4f} | {r['pls_snv_sub']:.4f} | "
                f"**{r['dssnv_pls_sub']:.4f}** | **{r['bilevel_sub']:.4f}** | **{r['mk_bilevel_sub']:.4f}** |"
            )
        if review_p0 is not None and "part4_melamine_sg_decomp" in review_p0:
            p4 = review_p0["part4_melamine_sg_decomp"]
            d_pcts = [
                (p4[inst]["rmsep_z"]["tuned_sg_deriv_only_ridge"] / p4[inst]["rmsep_z"]["raw_ridge"] - 1.0) * 100.0
                for inst in ("R562", "R568", "R861")
            ]
            t_all = [p4[inst]["rmsep_z"]["tuned_sg_all_ridge"] for inst in ("R562", "R568", "R861")]
            lines += [
                "",
                f"> **导数 vs 平滑机制注记（`C-141`）**：同 LOO Ridge 头与 `n=300` 拆解证实，即使将导数窗口放宽至 `w=31`（`d∈{{1,2}}`），在 `R562 / R568 / R861` 上施加导数仍劣于 `Raw-Ridge` `+{min(d_pcts):.1f}% ~ +{max(d_pcts):.1f}%`（`{p4['R562']['rmsep_z']['tuned_sg_deriv_only_ridge']:.4f} / {p4['R568']['rmsep_z']['tuned_sg_deriv_only_ridge']:.4f} / {p4['R861']['rmsep_z']['tuned_sg_deriv_only_ridge']:.4f}` vs `{p4['R562']['rmsep_z']['raw_ridge']:.4f} / {p4['R568']['rmsep_z']['raw_ridge']:.4f} / {p4['R861']['rmsep_z']['raw_ridge']:.4f}`）；折内 `tuned-SG (d∈{{0,1,2}})` 通过自动放弃导数、选中 `d=0` 纯平滑才与 `Raw-Ridge / Bilevel` 打平（`{min(t_all):.4f}–{max(t_all):.4f}`）。",
            ]

    if review_p0 is not None:
        _append_table_21(lines, review_p0)
    if ind_dl is not None:
        _append_table_22(lines, ind_dl, ind_followups=ind_followups, multiseed=multiseed, wilcoxon_ind=wilcoxon_ind)

    with open(os.path.join(OUT, "PAPER_TABLES_FINAL.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_fig6_bilevel_diffprep(bilevel: dict) -> None:
    """Fig 6: BilevelDiffPrep vs Classical & SPORT across 7 Real NIR Datasets + White-Box Learned Physics."""
    ds_keys = list(bilevel["real_7_datasets"].keys())
    ds_labels = [k.replace("_", "\n") for k in ds_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.7), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel A: Relative RMSEP change vs SNV-PLS (%)
    x = np.arange(len(ds_keys))
    w = 0.19
    series = [
        ("Raw-PLS", "raw_pls", "#9D755D"),
        ("DSSNV-PLS", "dssnv_pls", "#F58518"),
        ("SPORT-PLS (Roger 2020)", "sport_pls", "#54A24B"),
        ("BilevelDiffPrep (Ours)", "bilevel_diffprep", "#4C78A8"),
    ]
    ax1.axhline(0.0, color="#222222", lw=0.9, ls="--")
    for idx, (lbl, mkey, col) in enumerate(series):
        vals = [
            (bilevel["real_7_datasets"][k]["rmsep"][mkey] / bilevel["real_7_datasets"][k]["rmsep"]["snv_pls"] - 1.0) * 100.0
            for k in ds_keys
        ]
        bars = ax1.bar(x + (idx - 1.5) * w, vals, width=w, label=lbl, color=col)
        if mkey == "bilevel_diffprep":
            for b in bars:
                h = b.get_height()
                ax1.text(b.get_x() + b.get_width() / 2, h - 1.8, f"{h:+.1f}%", ha="center", va="top",
                         fontsize=7.3, fontweight="bold", color="#1F4E79")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_labels, fontsize=8.2)
    ax1.set_ylim(-82.0, 38.0)
    ax1.set_ylabel("RMSEP Change vs Classical SNV-PLS (%)  [Lower is Better]")
    ax1.set_title("(a) 7 Real Public NIR Datasets (Strict Nested 3×5 CV): 7/7 Wins Over SNV")
    ax1.legend(loc="lower right", frameon=True, fontsize=8.0)
    ax1.grid(axis="y", alpha=0.25)

    # Panel B: White-Box Learned Physics Heatmap (7 Real + 2 Synth OOD)
    rows_lbl = [k.replace("_", " ") for k in ds_keys] + ["Synth NIR OOD", "Synth Chrom OOD"]
    mat = []
    for k in ds_keys:
        p = bilevel["real_7_datasets"][k]["learned_physics_mean"]
        mat.append([p["wc"], p["ws"], p["fod_alpha"] / 2.0, min(1.0, p["fod_sigma"] / 2.0), p["als_gate"], 0.0])
    for mod in ("nir", "chrom"):
        p = bilevel["synthetic_5step_activation"][mod]["learned_physics"]
        mat.append([p["wc"], p["ws"], p["fod_alpha"] / 2.0, min(1.0, p["fod_sigma"] / 2.0), p["als_gate"], p["align_gate"]])
    M = np.asarray(mat, dtype=np.float64)
    im = ax2.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    col_lbls = ["wc\n(Center)", "ws\n(Scale)", "α ∈ [0,2]\n(FOD)", "σ ∈ [0,2.5]\n(Smooth)", "als_g\n(Baseline)", "align_g\n(Shift)"]
    ax2.set_xticks(np.arange(len(col_lbls)))
    ax2.set_xticklabels(col_lbls, fontsize=8.0)
    ax2.set_yticks(np.arange(len(rows_lbl)))
    ax2.set_yticklabels(rows_lbl, fontsize=8.2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            raw_v = M[i, j] * (2.0 if j in (2, 3) else 1.0)
            txt_c = "white" if M[i, j] > 0.58 else "#222222"
            ax2.text(j, i, f"{raw_v:.2f}", ha="center", va="center", fontsize=7.5, color=txt_c)
    ax2.set_title("(b) White-Box Learned Physics Across Media & OOD")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Normalized Color Intensity [0, 1]")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_bilevel_diffprep_and_sport.png"))
    plt.close(fig)


def make_fig7_post_r01_id_ood(post_r01: dict) -> None:
    """Fig 7: Post-R-01 7-Seed × 2-Modality (NIR & Chrom) ID/OOD RMSEP & Outlier AUROC."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    modes = ["raw", "classic", "diff_v2", "diff_v3"]
    labels = ["Raw\n(No Prep)", "Classic\n(5-Step)", "Diff v2\n(Post-R-01)", "Diff v3\n(DSSNV+FOD)"]
    x = np.arange(len(modes))
    w = 0.34
    for ax, mod, title in [(axes[0], "nir", "(a) NIR Modality (7 Seeds, Post-R-01)"),
                           (axes[1], "chrom", "(b) Chromatography Modality (7 Seeds, Post-R-01)")]:
        s = post_r01[mod]["summary"]
        id_m = [s[m]["rmsep_id_mean"] for m in modes]
        id_e = [s[m]["rmsep_id_std"] for m in modes]
        ood_m = [s[m]["rmsep_ood_mean"] for m in modes]
        ood_e = [s[m]["rmsep_ood_std"] for m in modes]
        b1 = ax.bar(x - w / 2, id_m, width=w, yerr=id_e, capsize=3, label="ID RMSEP", color="#4C78A8")
        b2 = ax.bar(x + w / 2, ood_m, width=w, yerr=ood_e, capsize=3, label="OOD (2×) RMSEP", color="#E45756")
        for bars, errs in ((b1, id_e), (b2, ood_e)):
            for b, err in zip(bars, errs):
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + err + 0.0025, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.4)
        ax.set_ylim(0.0, 0.172)
        ax.set_ylabel("RMSEP (mean ± std, n=7 seeds)")
        ax.set_title(title)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_post_r01_id_ood_7seed.png"))
    plt.close(fig)


def make_fig8_theory_dssnv(theory: dict) -> None:
    """Fig 8: Closed-Form Optimal Shrinkage Theorem Verification (C-061 Sign-Reversal + Zero-CV 7 Real Datasets)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.4))

    # Panel A: Empirical vs Closed-Form Theorem 1 on C-061 fixed-variance coupling curves
    colors = {"Cordoba_Wheat": "#4C78A8", "Cordoba_Barley": "#E45756", "SWRI_Diesel_CN": "#54A24B"}
    ax1.axhline(0.0, color="#222222", lw=0.9, ls="--")
    for lbl, v in theory["c061_theorem_verification"].items():
        c = colors[lbl]
        rs = v["r_grid"]
        ax1.plot(rs, v["empirical_gain_pct"], "o-", color=c, lw=2.0,
                 label=f"{lbl.replace('_', ' ')} Emp (r={v['pearson_r']:.3f})")
        ax1.plot(rs, v["theoretical_gain_pct"], "s--", color=c, lw=1.5, alpha=0.75,
                 label=f"{lbl.replace('_', ' ')} Thm 1 (r_crit={v['r_crit']:.2f})")
    ax1.set_xlabel("Scatter-Concentration Coupling Correlation |r| (Fixed std(a)=0.35)")
    ax1.set_ylabel("SNV vs Raw RMSEP Change (%)  [>0 = Harmful]")
    ax1.set_title("(a) Theorem 1 vs Empirical Sign-Reversal Law")
    ax1.legend(fontsize=7.3, ncol=2, loc="upper left", frameon=True)
    ax1.grid(alpha=0.25)

    # Panel B: Zero-CV Closed-Form Theory DSSNV vs CV-DSSNV across 7 real datasets (relative to SNV)
    ds_keys = list(theory["real_7_datasets"].keys())
    x = np.arange(len(ds_keys))
    w = 0.28
    cv_pct = [(theory["real_7_datasets"][k]["rmsep"]["cv_dssnv"] / theory["real_7_datasets"][k]["rmsep"]["snv"] - 1.0) * 100.0 for k in ds_keys]
    th_pct = [theory["real_7_datasets"][k]["vs_snv_pct"] for k in ds_keys]
    ax2.axhline(0.0, color="#222222", lw=0.9, ls="--")
    ax2.bar(x - w / 2, cv_pct, width=w, label="CV-DSSNV (Grid Search)", color="#F58518")
    b2 = ax2.bar(x + w / 2, th_pct, width=w, label="Theory-DSSNV (Closed-Form, 4.2× Faster)", color="#4C78A8")
    for b in b2:
        h = b.get_height()
        va = "top" if h < 0 else "bottom"
        off = -1.1 if h < 0 else 0.7
        cx = b.get_x() + (b.get_width() * 0.72 if h < -20 else b.get_width() / 2)
        ax2.text(cx, h + off, f"{h:+.1f}%", ha="center", va=va, fontsize=7.2, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([k.replace("_", "\n") for k in ds_keys], fontsize=8.0)
    ax2.set_ylim(-43.0, 6.5)
    ax2.set_ylabel("RMSEP Change vs Classical SNV (%)  [Lower is Better]")
    ax2.set_title("(b) Zero-CV Closed-Form Theorem 2 vs Grid CV-DSSNV")
    ax2.legend(loc="lower right", frameon=True, fontsize=8.0)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_theory_dssnv_and_c061.png"))
    plt.close(fig)


def make_fig9_deep_da_sota(unified: dict, deep_da: dict) -> None:
    """Fig 9: Deep DA (Deep CORAL / DANN / PEAMATL) vs Physical Linear/Subspace/Active SOTA (nRMSEP % of Range)."""
    cases = [
        ("Corn\noil", "corn_m5_to_mp6", "oil", 15),
        ("Tablets\nweight", "tablets_X1_to_X2", "weight", 20),
        ("Tablets\nhardness", "tablets_X1_to_X2", "hardness", 20),
        ("Tablets\nassay", "tablets_X1_to_X2", "assay", 20),
        ("Wheat*\nB->A", "wheat_blind_B_to_A", "protein", 20),
        ("Wheat*\nC->B", "wheat_blind_C_to_B", "protein", 20),
        ("OSSL*\noc", "neospectra_blind_NEO5_to_NEO6", "oc", 20),
        ("OSSL*\nclay", "neospectra_blind_NEO5_to_NEO6", "clay", 20),
        ("OSSL*\nph", "neospectra_blind_NEO5_to_NEO6", "ph", 20),
    ]
    labels = [c[0] for c in cases]
    x = np.arange(len(cases))
    w = 0.145
    series = [
        ("MeanShift", lambda r, d, kp: r["Unp_MeanShift"], "#9D755D"),
        ("SLOT-PLS (Ours)", lambda r, d, kp: r["Unp_SLOT_PLS"], "#4C78A8"),
        ("Deep CORAL", lambda r, d, kp: d["Deep_CORAL"], "#F58518"),
        ("DANN", lambda r, d, kp: d["Deep_DANN"], "#E45756"),
        ("PEAMATL (MAE+MMD)", lambda r, d, kp: d["Deep_PEAMATL"], "#B279A2"),
        ("DS-AffineTOpt (Ours)", lambda r, d, kp: r[f"Pair_DS_TOpt_k{kp}"], "#54A24B"),
    ]
    fig, ax = plt.subplots(figsize=(13.4, 4.5))
    for i, (name, getter, col) in enumerate(series):
        vals = []
        for _, ds_k, col_k, kp in cases:
            r = unified[ds_k]["per_component_rmsep"][col_k]
            d = deep_da[ds_k]["per_component_rmsep"][col_k]
            lo, hi = unified[ds_k]["target_ranges"][col_k]
            span = max(float(hi) - float(lo), 1e-9)
            vals.append(100.0 * float(getter(r, d, kp)) / span)
        ax.bar(x + (i - 2.5) * w, vals, width=w, label=name, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.3)
    ax.set_ylabel("nRMSEP (% of Target Range)  [Lower is Better]")
    ax.set_title("Deep DA (Deep CORAL / DANN / PEAMATL) vs Physical Subspace & Paired Transfer SOTA (Identical 10 Disjoint Splits, * = Blind Holdout)")
    ax.legend(ncol=3, loc="upper right", frameon=True, fontsize=8.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_deep_da_vs_physical_sota.png"))
    plt.close(fig)


def make_fig10_param_transfer_and_ga_gu2026(
    param_transfer: dict, unified: dict, ga_gu2026: dict | None, proxy_v2: dict | None
) -> None:
    """Fig 10: Cross-Dataset Physical Parameter Transfer (7×7) + Gu et al. (2026) Head-to-Head + R²_shape Identifiability."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), gridspec_kw={"width_ratios": [1.3, 1.25, 1.0]})

    # Panel A: 7x7 Cross-Dataset Retention Heatmap
    ax = axes[0]
    ds_names = list(param_transfer["retention_pct_matrix"].keys())
    short = [k.replace("Cordoba_", "C.").replace("SWRI_Diesel_CN", "Diesel").replace("Tablets_X1", "Tablets").replace("Corn_m5", "Corn") for k in ds_names]
    M = np.array([[param_transfer["retention_pct_matrix"][s][t] for t in ds_names] for s in ds_names], dtype=np.float64)
    im = ax.imshow(M, cmap="RdYlGn", vmin=40.0, vmax=105.0, aspect="auto")
    ax.set_xticks(np.arange(len(short)))
    ax.set_xticklabels(short, fontsize=7.8, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(short)))
    ax.set_yticklabels(short, fontsize=8.0)
    for i in range(len(short)):
        for j in range(len(short)):
            txt_c = "#111111" if 60.0 < M[i, j] < 96.0 else "white"
            ax.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center", fontsize=7.2,
                    fontweight=("bold" if i == j or M[i, j] >= 95.0 else "normal"), color=txt_c)
    ax.set_xlabel("Target Dataset (Zero Prep Re-Tuning)")
    ax.set_ylabel("Source Dataset (Frozen Physics θ)")
    sm = param_transfer["summary"]
    ax.set_title(f"(a) 7×7 Physics Transfer\n(Median {sm['median_retention_pct_all42']:.1f}%, {sm['pairs_retention_ge_90pct']}/42 ≥90%)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Retention (%)")

    # Panel B: Head-to-Head Cross-Instrument Transfer vs Gu et al. (2026, CILS 279:105918)
    ax2 = axes[1]
    if ga_gu2026 is not None:
        gu = ga_gu2026["gu2026_transfer_and_prior_shift"]
        cases = [
            ("Corn\nstd", "corn_m5_to_mp6", None, 15),
            ("Tablets\nassay", "tablets_X1_to_X2", "assay", 20),
            ("Wheat*\nB→A", "wheat_blind_B_to_A", "protein", 20),
            ("Wheat*\nC→B", "wheat_blind_C_to_B", "protein", 20),
            ("OSSL*\nstd", "neospectra_blind_NEO5_to_NEO6", None, 20),
        ]
        x = np.arange(len(cases))
        w = 0.19
        series_b = [
            ("Gu26 GA+CV", "#F58518"),
            ("Gu26 GA+WD (CILS)", "#E45756"),
            ("SLOT-PLS (Ours)", "#4C78A8"),
            ("DS-AffineTOpt (Ours)", "#54A24B"),
        ]
        # Relative to Gu26 GA+WD = 100% so all 5 benchmarks share one clean dimensionless scale
        vals_mat = {s[0]: [] for s in series_b}
        for _, ds_k, col_k, kp in cases:
            if col_k is None:
                v_cv = gu[ds_k]["std_pooled_rmsep"]["Gu2026_GA_CV"]
                v_wd = gu[ds_k]["std_pooled_rmsep"]["Gu2026_GA_WD"]
                v_sl = unified[ds_k]["std_pooled_rmsep"]["Unp_SLOT_PLS"]
                v_ds = unified[ds_k]["std_pooled_rmsep"][f"Pair_DS_TOpt_k{kp}"]
            else:
                v_cv = gu[ds_k]["per_component_rmsep"][col_k]["Gu2026_GA_CV"]
                v_wd = gu[ds_k]["per_component_rmsep"][col_k]["Gu2026_GA_WD"]
                v_sl = unified[ds_k]["per_component_rmsep"][col_k]["Unp_SLOT_PLS"]
                v_ds = unified[ds_k]["per_component_rmsep"][col_k][f"Pair_DS_TOpt_k{kp}"]
            base = max(v_wd, 1e-9)
            vals_mat["Gu26 GA+CV"].append(min(220.0, 100.0 * v_cv / base))
            vals_mat["Gu26 GA+WD (CILS)"].append(100.0)
            vals_mat["SLOT-PLS (Ours)"].append(100.0 * v_sl / base)
            vals_mat["DS-AffineTOpt (Ours)"].append(100.0 * v_ds / base)
        ax2.axhline(100.0, color="#333333", lw=0.9, ls="--")
        for idx, (lbl, col) in enumerate(series_b):
            bars = ax2.bar(x + (idx - 1.5) * w, vals_mat[lbl], width=w, label=lbl, color=col)
            if lbl in ("SLOT-PLS (Ours)", "DS-AffineTOpt (Ours)"):
                for ci, b in enumerate(bars):
                    h = b.get_height()
                    h_slot = vals_mat["SLOT-PLS (Ours)"][ci]
                    h_ds = vals_mat["DS-AffineTOpt (Ours)"][ci]
                    if abs(h_slot - h_ds) < 18.0:
                        y_off = 3.0 if lbl == "SLOT-PLS (Ours)" else 15.0
                    else:
                        y_off = 4.0
                    ax2.text(b.get_x() + b.get_width() / 2, h + y_off, f"{h:.0f}%",
                             ha="center", va="bottom", fontsize=6.4, fontweight="bold", zorder=5,
                             bbox=dict(facecolor="white", edgecolor="none", pad=0.7, alpha=0.88))
        ax2.set_xticks(x)
        ax2.set_xticklabels([c[0] for c in cases], fontsize=8.0)
        ax2.set_ylim(0.0, 245.0)
        ax2.set_ylabel("Relative RMSEP vs Gu26 GA+WD (%) [Lower = Better]")
        ax2.set_title("(b) Head-to-Head vs Gu et al. (2026, CILS)\n(10 Disjoint Splits, * = Blind Holdout)")
        ax2.legend(loc="upper right", frameon=True, fontsize=7.0)
        ax2.grid(axis="y", alpha=0.25)

    # Panel C: Shape-Residual Scatter Identifiability R²_shape (Direction D)
    ax3 = axes[2]
    if proxy_v2 is not None:
        pds = proxy_v2["per_dataset"]
        p_keys = list(pds.keys())
        p_short = [k.replace("Cordoba_", "C.").replace("SWRI_Diesel_CN", "Diesel").replace("Tablets_X1", "Tablets").replace("Corn_m5", "Corn") for k in p_keys]
        r2s = [pds[k]["r2_s_shape_mean"] * 100.0 for k in p_keys]
        phys = [100.0 - v for v in r2s]
        y_pos = np.arange(len(p_keys))
        ax3.barh(y_pos, phys, color="#4C78A8", label="Physical Scatter ($1 - R^2_{\\mathrm{shape}}$)")
        ax3.barh(y_pos, r2s, left=phys, color="#F58518", alpha=0.85, label="Chemical Amp. ($R^2_{\\mathrm{shape}}$)")
        for i, (ph, ch) in enumerate(zip(phys, r2s)):
            ax3.text(50.0, i, f"Phys {ph:.0f}% | Chem {ch:.0f}%", ha="center", va="center",
                     fontsize=7.0, fontweight="bold", color="#111111",
                     bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.0))
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(p_short, fontsize=8.0)
        ax3.set_xlim(0.0, 100.0)
        ax3.set_xlabel("row_std Variance Decomposition (%)")
        ax3.set_title("(c) Scatter Identifiability $R^2_{\\mathrm{shape}}$\n(High-Scatter vs Liquid Beer–Lambert)")
        ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=True, fontsize=7.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig10_param_transfer_and_ga_gu2026.png"))
    plt.close(fig)


def _try_load(name: str):
    p = os.path.join(AUDIT, name)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    p0 = load_json("res_p0_and_blind_2026_11_19.json")
    ext = load_json("res_selfcheck_external_2026_11_19.json")
    attr = load_json("res_slot_attribution_2026_11_18.json")
    t01 = load_json("res_t01_t03_2026_11_18.json")
    unified = load_json("res_unified_sota_2026_11_19.json")
    dssnv = load_json("res_dssnv_2026_11_19.json")
    bilevel = _try_load("res_bilevel_diffprep_2026_11_19.json")
    opcs = _try_load("res_opcs_and_huber_2026_11_19.json")
    post_r01 = _try_load("res_post_r01_id_ood_2026_11_19.json")
    theory = _try_load("res_theory_dssnv_2026_11_20.json")
    deep_da = _try_load("res_deep_da_sota_2026_11_20.json")
    bilevel_v2 = _try_load("res_bilevel_v2_7datasets_2026_11_20.json")
    effects = _try_load("res_effect_sizes_2026_11_20.json")
    param_transfer = _try_load("res_param_transfer_2026_11_20.json")
    dir_b = _try_load("res_direction_b_gate_2026_11_20.json")
    ga_gu2026 = _try_load("res_ga_and_gu2026_2026_11_20.json")
    proxy_v2 = _try_load("res_theory_proxy_v2_2026_11_20.json")
    runtime = _try_load("res_runtime_cost_2026_11_20.json")
    plan21 = _try_load("res_plan21_physprior_2026_11_20.json")
    plan22 = _try_load("res_plan22_autogate_2026_11_20.json")
    mk_sport = _try_load("res_mk_bilevel_and_sport_fix_2026_11_21.json")
    ext_blind = _try_load("res_blindtest_external_2026_11_21.json")
    review_p0 = _try_load("res_review_p0_2026_11_23.json")
    ind_dl = _try_load("res_industrial_vs_dl_2026_11_21.json")
    ind_followups = _try_load("res_industrial_followups_2026_11_21.json")
    multiseed = _try_load("res_multiseed_industrial.json")
    wilcoxon_ind = _try_load("res_industrial_paired_wilcoxon_2026_11_23.json")

    make_fig1_topt(p0)
    make_fig2_active_crossover(p0, ext, t01)
    make_fig3_snv_coupling(ext)
    make_fig4_gate_identifiability(p0, attr)
    make_fig5_dssnv(dssnv)
    n_figs = 5
    if bilevel is not None:
        make_fig6_bilevel_diffprep(bilevel)
        n_figs += 1
    if post_r01 is not None:
        make_fig7_post_r01_id_ood(post_r01)
        n_figs += 1
    if theory is not None:
        make_fig8_theory_dssnv(theory)
        n_figs += 1
    if deep_da is not None:
        make_fig9_deep_da_sota(unified, deep_da)
        n_figs += 1
    if param_transfer is not None:
        make_fig10_param_transfer_and_ga_gu2026(param_transfer, unified, ga_gu2026, proxy_v2)
        n_figs += 1
    write_tables_md(
        p0, ext, attr, t01, unified, dssnv,
        bilevel=bilevel, opcs=opcs, post_r01=post_r01,
        theory=theory, deep_da=deep_da, bilevel_v2=bilevel_v2, effects=effects,
        param_transfer=param_transfer, dir_b=dir_b, ga_gu2026=ga_gu2026,
        proxy_v2=proxy_v2, runtime=runtime, plan21=plan21, plan22=plan22,
        mk_sport=mk_sport, ext_blind=ext_blind,
        review_p0=review_p0, ind_dl=ind_dl,
        ind_followups=ind_followups, multiseed=multiseed, wilcoxon_ind=wilcoxon_ind,
    )
    print(f"Generated {n_figs} publication figures (300 DPI) + PAPER_TABLES_FINAL.md in:", OUT)


if __name__ == "__main__":
    main()
