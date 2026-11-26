"""One-command verification, table summary, and figure regeneration for the paper.

Usage:
  python reproduce_paper.py --verify    # Verify frozen theory SHA-256, 26 archived JSONs, 7 PNGs, and prereg locks
  python reproduce_paper.py --tables    # Print headline Tables 1-5 summary directly from archived JSONs
  python reproduce_paper.py --plot-all  # Regenerate all 7 publication figures (Figs 1-7) from archived JSONs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemdl.deploy_gate import FROZEN_THEORY_LF_SHA256

REQUIRED_JSONS = [
    "prereg_deploy_holdout_2026_11_24.json",
    "prereg_round3_unseen_2026_11_24.json",
    "prereg_snv_gate_round2_2026_11_24.json",
    "prereg_theory_blind_2026_11_24.json",
    "prereg_wiener_ws_2026_11_25.json",
    "prereg_ws_closed_value_2026_11_25.json",
    "res_deploy_extended_analysis_2026_11_24.json",
    "res_major_revision_2026_11_25.json",
    "res_mk_bilevel_and_sport_fix_2026_11_21.json",
    "res_prereg_deploy_holdout_2026_11_24.json",
    "res_prereg_round3_unseen_2026_11_24.json",
    "res_prereg_snv_gate_round2_2026_11_24.json",
    "res_prereg_theory_blind_2026_11_24.json",
    "res_press_e1_fix_86folds_2026_11_24.json",
    "res_r2_shape_identifiability_2026_11_25.json",
    "res_review_p0_2026_11_23.json",
    "res_selfcheck_external_2026_11_19.json",
    "res_snv_phase_diagram_2026_11_24.json",
    "res_theory_dssnv_2026_11_20.json",
    "res_wiener_group_r2_audit_2026_11_25.json",
    "res_wiener_ws_2026_11_25.json",
    "res_ws_closed_value_2026_11_25.json",
    "res_craw_nested_rootcause_2026_11_26.json",
    "res_grouped_fit_pls_and_mango_2026_11_26.json",
    "revision_protocol_registry_2026_11_26.json",
    "res_revision_v2_2026_11_26.json",
]

REQUIRED_FIGS = [
    "fig_ms1_framework.png",
    "fig_snv_phase_diagram.png",
    "fig_ms3_stage2_scatter.png",
    "fig_ms4_deploy_regret_rc.png",
    "fig_ms6_deploy_per_task.png",
    "fig_ms7_ws_asymmetric_cost.png",
    "fig13_r2_shape_identifiability.png",
]


def _lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def cmd_verify() -> None:
    theory_sha = _lf_sha256(ROOT / "chemdl" / "theory.py")
    assert theory_sha == FROZEN_THEORY_LF_SHA256, (
        f"chemdl/theory.py LF SHA-256 mismatch: {theory_sha} != {FROZEN_THEORY_LF_SHA256}"
    )
    print(f"[OK] chemdl/theory.py LF SHA-256 verified: {theory_sha}")

    for name in REQUIRED_JSONS:
        p = ROOT / "_audit" / name
        assert p.exists(), f"Missing JSON: {p}"
        json.loads(p.read_text(encoding="utf-8"))
    print(f"[OK] {len(REQUIRED_JSONS)}/{len(REQUIRED_JSONS)} archived JSON artifacts valid.")

    for name in REQUIRED_FIGS:
        p = ROOT / "paper_assets" / name
        assert p.exists() and p.stat().st_size > 10_000, f"Missing/empty figure: {p}"
    print(f"[OK] {len(REQUIRED_FIGS)}/{len(REQUIRED_FIGS)} publication PNG figures present.")

    for script in [
        "_audit/prereg_wiener_ws_phase2_unblind.py",
        "_audit/prereg_ws_closed_value_phase2_unblind.py",
    ]:
        subprocess.run([sys.executable, script, "--verify-only"], cwd=ROOT, check=True)
    assert (ROOT / "_audit" / "bench_major_revision_2026_11_25.py").exists()
    assert (ROOT / "_audit" / "bench_revision_v2_2026_11_26.py").exists()
    assert (ROOT / "_audit" / "audit_craw_nested_rootcause_2026_11_26.py").exists()
    assert (ROOT / "_audit" / "audit_grouped_fit_pls_and_mango_2026_11_26.py").exists()
    assert (ROOT / "requirements-lock.txt").exists()
    print("[PASS] All frozen hashes, preregistration locks, and paper artifacts verified.")


def cmd_tables() -> None:
    p0 = json.loads((ROOT / "_audit" / "res_review_p0_2026_11_23.json").read_text(encoding="utf-8"))
    pd_res = json.loads((ROOT / "_audit" / "res_snv_phase_diagram_2026_11_24.json").read_text(encoding="utf-8"))
    d18 = json.loads((ROOT / "_audit" / "res_prereg_deploy_holdout_2026_11_24.json").read_text(encoding="utf-8"))
    r3 = json.loads((ROOT / "_audit" / "res_prereg_round3_unseen_2026_11_24.json").read_text(encoding="utf-8"))
    ws54 = json.loads((ROOT / "_audit" / "res_ws_closed_value_2026_11_25.json").read_text(encoding="utf-8"))
    wi30 = json.loads((ROOT / "_audit" / "res_wiener_ws_2026_11_25.json").read_text(encoding="utf-8"))
    id_res = json.loads((ROOT / "_audit" / "res_r2_shape_identifiability_2026_11_25.json").read_text(encoding="utf-8"))

    v2 = p0["part3_theorem1_confusion"]["overall_v2"]
    print(f"\n[Table 2] 170-fold Theorem 1 (with R^2_shape): Acc={v2['acc']:.4f}, BAC={v2['balanced_acc']:.4f}, MCC={v2['mcc']:.4f}")
    pd_all = pd_res["summary"]
    print(f"[Table 2] 210-cell 2D phase diagram: Sign={pd_all['sign_agreement']:.4f}, Pearson r={pd_all['pearson_pred_vs_measured']:.4f}")

    g18 = d18["gate_decision_eval"]["new_paired_loo_primary"]["all_signs"]
    s18 = d18["deployment_strategy_eval"]["ALL_18"]["same_pls"]
    print(
        f"\n[Table 3] Deploy-18 Gate: coverage={g18['coverage']:.3f}, decided_acc={g18['acc']:.4f}, "
        f"Gate+SNV mean_regret={s18['Gate_Fallback_SNV']['mean_regret_pct_vs_oracle']:+.2f}%"
    )
    g13 = r3["gate_decision_eval"]["new_paired_loo_primary"]["all_signs"]
    s13 = r3["deployment_strategy_eval"]
    print(
        f"[Table 3] Round3-13 (6 unseen labs): coverage={g13['coverage']:.3f}, decided_acc={g13['acc']:.4f}, "
        f"Gate+InnerCV mean_regret={s13['Gate_Fallback_InnerCV']['mean_regret_pct']:+.2f}% (C_10={s13['Gate_Fallback_InnerCV']['weighted_cost_by_w']['10.0']:.2f})"
    )

    ext30 = wi30["summary"]["EXTERNAL_30"]["pls"]["methods"]
    print(
        f"\n[Table 4] 54-task closed-form interior: {ws54['pre_tag_counts']['v2_alg']['interior_01']}/{ws54['n_tasks']}; "
        f"Ext-30 PLS Wiener C_10={ext30['ws_wiener']['asymmetric_cost_w10_exact']:.2f} vs "
        f"OldClosed C_10={ext30['ws_closed_v2_old']['asymmetric_cost_w10_exact']:.2f}"
    )

    a1 = id_res["part_a1_scalar_sweep"]["pooled_metrics_raw_vs_f_adj_pop_lin"]
    a2 = id_res["part_a2_tilt_negative_control"]["pooled_by_gamma"]
    print(
        f"\n[Table 5] Scalar sweep vs f_adj_lin: r={a1['pearson_r']:.4f}, MAE={a1['mae']:.4f}; "
        f"Tilt collapse: gamma=0 ({a2['0.00']['one_minus_r2_raw_mean']:.4f}) -> gamma=1 ({a2['1.00']['one_minus_r2_raw_mean']:.4f})"
    )

    mr = json.loads((ROOT / "_audit" / "res_major_revision_2026_11_25.json").read_text(encoding="utf-8"))
    vsn17 = mr["part1_vsn_comparison"]["VALID_17"]
    vsn13 = mr["part1_vsn_comparison"]["Round3_13"]
    vsn_pls = mr["part1_vsn_comparison"]["EXTERNAL_30_PLS"]
    vsn_rdg = mr["part1_vsn_comparison"]["EXTERNAL_30_RIDGE"]
    p2_31 = mr["part2_nested_cv_bias_ablation"]["Pooled_31"]
    ni_pls = mr["part3_noninferiority_and_cluster_cis"]["external_30_noninferiority"]["pls"]
    ni_rdg = mr["part3_noninferiority_and_cluster_cis"]["external_30_noninferiority"]["ridge"]
    print(
        f"\n[Major Revision] VSN VALID_17 C_10={vsn17['C_10']:.2f} ({vsn17['mean_regret_pct']:+.2f}%), "
        f"Round3_13 C_10={vsn13['C_10']:.2f}; Ext-30 PLS C_10={vsn_pls['C_10']:.2f}, Ridge C_10={vsn_rdg['C_10']:.2f}"
    )
    print(
        f"[Major Revision] Nested-CV 31-task 3-state agreement={p2_31['three_state_agreement']} "
        f"({p2_31['three_state_agreement_pct']:.1f}%); "
        f"Non-inf UCB95: Ridge={ni_rdg['cluster12_ucb95_wiener_minus_inner_cv']:+.4f} "
        f"(p_0.03={ni_rdg['margins']['delta_0.03']['wilcoxon_one_sided_p']:.2e}), "
        f"PLS={ni_pls['cluster12_ucb95_wiener_minus_inner_cv']:+.4f} "
        f"(p_0.08={ni_pls['margins']['delta_0.08']['wilcoxon_one_sided_p']:.2e})"
    )

    rv2 = json.loads((ROOT / "_audit" / "res_revision_v2_2026_11_26.json").read_text(encoding="utf-8"))
    v17_c4a = rv2["p1b_deployment_summary"]["VALID_17"]["C4A_FullNested_RepGrp"]
    v17_c4b = rv2["p1b_deployment_summary"]["VALID_17"]["C4B_FullNested_DeployUnit"]
    r13_c4a = rv2["p1b_deployment_summary"]["Round3_13"]["C4A_FullNested_RepGrp"]
    print(
        f"\n[Revision v2] VALID_17 C4A: cov={v17_c4a['coverage_k']}/17, acc={v17_c4a['cond_acc_k']}/{v17_c4a['coverage_k']} "
        f"({v17_c4a['cond_acc_pct']:.1f}%), Gate+SNV C10={v17_c4a['strategies']['Gate_Fallback_SNV']['C_10']:.2f}, "
        f"InnerCV C10={v17_c4a['strategies']['InnerCV_PLS']['C_10']:.2f}, "
        f"AutoPrep-CV C10={v17_c4a['strategies']['AutoPrep_CV']['C_10']:.2f}; "
        f"C4B Gate+SNV C10={v17_c4b['strategies']['Gate_Fallback_SNV']['C_10']:.2f}"
    )
    print(
        f"[Revision v2] Round3_13 C4A: cov={r13_c4a['coverage_k']}/13, acc={r13_c4a['cond_acc_k']}/{r13_c4a['coverage_k']} "
        f"({r13_c4a['cond_acc_pct']:.1f}%), Gate+InnerCV C10={r13_c4a['strategies']['Gate_Fallback_InnerCV']['C_10']:.2f} "
        f"vs InnerCV C10={r13_c4a['strategies']['InnerCV_PLS']['C_10']:.2f}"
    )
    e_pls = rv2["p1c_ext30_summary"]["pls"]["task_equal_weighted"]
    e_rdg = rv2["p1c_ext30_summary"]["ridge"]["task_equal_weighted"]
    print(
        f"[Revision v2] Ext-30 PLS C10: ws0={e_pls['ws_0']['C_10']:.2f}, ws0.20={e_pls['ws_0.20']['C_10']:.2f}, "
        f"Wiener1={e_pls['ws_wiener_single']['C_10']:.2f}, 11ptCV={e_pls['ws_inner_cv_11pt']['C_10']:.2f}, "
        f"AutoPrep={e_pls['autoprep_cv']['C_10']:.2f}"
    )
    print(
        f"[Revision v2] Ext-30 Ridge C10: ws0={e_rdg['ws_0']['C_10']:.2f}, ws0.20={e_rdg['ws_0.20']['C_10']:.2f}, "
        f"Wiener1={e_rdg['ws_wiener_single']['C_10']:.2f}, 11ptCV={e_rdg['ws_inner_cv_11pt']['C_10']:.2f}, "
        f"AutoPrep={e_rdg['autoprep_cv']['C_10']:.2f}"
    )


def cmd_plot_all() -> None:
    cmds = [
        [sys.executable, "paper_assets/plot_manuscript_figs_1_3_4.py"],
        [sys.executable, "paper_assets/plot_fig_snv_phase_diagram.py", "--plot-only"],
        [sys.executable, "paper_assets/plot_manuscript_figs_6_7.py"],
        [sys.executable, "paper_assets/plot_fig13_r2_shape_identifiability.py"],
    ]
    for cmd in cmds:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"[PASS] Regenerated all {len(REQUIRED_FIGS)} publication figures.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce verification, tables, and figures for the paper.")
    parser.add_argument("--verify", action="store_true", help="Verify frozen hashes, JSONs, PNGs, and prereg locks.")
    parser.add_argument("--tables", action="store_true", help="Print headline Tables 1-5 summary from archived JSONs.")
    parser.add_argument("--plot-all", action="store_true", help="Regenerate all 7 publication figures from archived JSONs.")
    args = parser.parse_args()

    if not (args.verify or args.tables or args.plot_all):
        args.verify = True
        args.tables = True

    if args.verify:
        cmd_verify()
    if args.tables:
        cmd_tables()
    if args.plot_all:
        cmd_plot_all()


if __name__ == "__main__":
    main()
