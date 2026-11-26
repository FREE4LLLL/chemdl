"""Round 3 · Phase 2：验证 Phase 1 锁单完整性并在 6 个全新外部 Zenodo NIR 实验室（13 项任务）上揭盲。

同台评估：
  - 同 PLS 头 6 种部署策略：Always_Raw / Always_SNV / InnerCV_PLS / Gate_Fallback_Raw / Gate_Fallback_SNV / Gate_Fallback_InnerCV
  - 多预处理正交融合先例：SPORT_Classic_PLS / SPORT_DSSNV_PLS
  - 异构升级工作流：Gate_Escalate_Bilevel（ABSTAIN 时升级 BilevelDiffPrep）
  - Risk-Coverage (AURC)、非对称严重遗憾惩罚代价 (w=1..10)、运行耗时与安全体检 (diagnose_deploy_sanity)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._blind_theory_loaders import file_sha256, file_sha256_lf, git_blob_sha256
from _audit._round3_unseen_loaders import load_round3_unseen_tasks
from _audit.prereg_snv_gate_round2_phase2_unblind import _eval_method
from _audit.prereg_theory_blind_phase2_unblind import _sanitize_json
from chemdl.deploy_gate import (
    FROZEN_THEORY_LF_SHA256,
    compute_asymmetric_regret,
    compute_risk_coverage_curve,
    diagnose_deploy_sanity,
)
from chemdl.diffprep import fit_bilevel_diffprep
from chemdl.transfer import fit_pls, fit_prep_fn, fit_sport_pls, rmsep


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 3 zero-contact external NIR Phase 2 unblinding")
    parser.add_argument("--verify-only", action="store_true", help="Verify lock chain & archived Round 3 JSON.")
    args = parser.parse_args()

    t_start = time.time()
    pre_path = ROOT / "_audit" / "prereg_round3_unseen_2026_11_24.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    pre_sha = file_sha256(pre_path)

    theory_path = ROOT / "chemdl" / "theory.py"
    locked_commit = str(pre["git_head_frozen"])
    frozen_blob_lf = git_blob_sha256(ROOT, locked_commit, "chemdl/theory.py", lf_normalize=True)
    assert frozen_blob_lf == pre["theory_py_git_blob_sha256_lf"] == FROZEN_THEORY_LF_SHA256
    wt_lf = file_sha256_lf(theory_path)
    exec_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    tasks = load_round3_unseen_tasks(ROOT)
    assert len(tasks) == pre["n_tasks"] == 13
    pre_by_id = {r["task_id"]: r for r in pre["tasks"]}

    if args.verify_only:
        res = json.loads((ROOT / "_audit" / "res_prereg_round3_unseen_2026_11_24.json").read_text(encoding="utf-8"))
        for t in tasks:
            p = pre_by_id[t["task_id"]]
            assert t["split_sha256"] == p["split_sha256"] and t["cal_test_hash_overlap"] == 0
        print(
            f"[verify-only PASS] Round 3 Unseen locked={locked_commit[:7]}, exec={exec_head[:7]}, "
            f"{len(res['tasks'])} tasks across {res['n_distinct_labs']} external labs verified."
        )
        return

    assert wt_lf == frozen_blob_lf, f"chemdl/theory.py LF SHA-256 mismatch: {wt_lf} vs {frozen_blob_lf}"
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "chemdl/theory.py", "_audit/prereg_round3_unseen_2026_11_24.json"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert not dirty, f"Frozen Round 3 files dirty: {dirty}"

    task_rows = []
    for i, t in enumerate(tasks, 1):
        tid = t["task_id"]
        p = pre_by_id[tid]
        assert t["split_sha256"] == p["split_sha256"], f"{tid}: split_sha256 mismatch"
        assert t["cal_test_hash_overlap"] == 0

        X_cal, y_cal = t["X_cal"], t["y_cal"]
        X_test, y_test = t["X_test"], t["y_test"]
        mu = float(y_cal.mean())
        sd = max(float(y_cal.std()), 1e-9)
        y_cal_n = ((y_cal - mu) / sd)[:, None]
        y_test_n = ((y_test - mu) / sd)[:, None]

        kmax = min(10, max(2, len(X_cal) // 2))
        inner = min(4, max(2, len(X_cal) // 4))

        t0 = time.perf_counter()
        pr_raw, _ = fit_pls(X_cal, y_cal_n, kmax=kmax, inner=inner, per_col=True, seed=0)
        e_raw = float(rmsep(pr_raw(X_test), y_test_n))
        dt_raw = time.perf_counter() - t0

        t0 = time.perf_counter()
        fn_snv = fit_prep_fn("snv", X_cal)
        pr_snv, _ = fit_pls(fn_snv(X_cal), y_cal_n, kmax=kmax, inner=inner, per_col=True, seed=0)
        e_snv = float(rmsep(pr_snv(fn_snv(X_test)), y_test_n))
        dt_snv = time.perf_counter() - t0

        t0 = time.perf_counter()
        pr_sp_cl, _ = fit_sport_pls(
            X_cal, y_cal_n, blocks=("snv", "sg1", "sg2", "raw"), kmax=kmax, inner=inner, seed=0
        )
        e_sp_cl = float(rmsep(np.asarray(pr_sp_cl(X_test), dtype=np.float64).reshape(-1, 1), y_test_n))
        dt_sp_cl = time.perf_counter() - t0

        t0 = time.perf_counter()
        pr_sp_ds, _ = fit_sport_pls(
            X_cal, y_cal_n, blocks=("dssnv", "sg1", "sg2", "raw"), kmax=kmax, inner=inner, seed=0
        )
        e_sp_ds = float(rmsep(np.asarray(pr_sp_ds(X_test), dtype=np.float64).reshape(-1, 1), y_test_n))
        dt_sp_ds = time.perf_counter() - t0

        emp_pct = (e_snv / max(e_raw, 1e-12) - 1.0) * 100.0
        oracle = min(e_raw, e_snv)
        gate = str(p["gate_decision"])
        cv_pick = str(p["inner_cv_pick"])

        e_cv = e_snv if cv_pick == "SNV" else e_raw
        e_fb_raw = e_snv if gate == "IMPROVE" else e_raw
        e_fb_snv = e_raw if gate == "WORSEN" else e_snv
        e_fb_cv = e_snv if gate == "IMPROVE" else (e_raw if gate == "WORSEN" else e_cv)

        if gate == "IMPROVE":
            e_esc = e_snv
            esc_src = "SNV_PLS"
            dt_esc = dt_snv
        elif gate == "WORSEN":
            e_esc = e_raw
            esc_src = "Raw_PLS"
            dt_esc = dt_raw
        else:
            t0 = time.perf_counter()
            pr_bi, _ = fit_bilevel_diffprep(X_cal, y_cal_n, steps=40, max_opt_n=800, seed=0)
            e_esc = float(rmsep(np.asarray(pr_bi(X_test), dtype=np.float64).reshape(-1, 1), y_test_n))
            esc_src = "BilevelDiffPrep"
            dt_esc = time.perf_counter() - t0

        sanity = diagnose_deploy_sanity(X_cal, X_test, y_cal=y_cal, y_new=y_test)

        methods_z = {
            "Always_Raw": round(e_raw, 4),
            "Always_SNV": round(e_snv, 4),
            "InnerCV_PLS": round(e_cv, 4),
            "Gate_Fallback_Raw": round(e_fb_raw, 4),
            "Gate_Fallback_SNV": round(e_fb_snv, 4),
            "Gate_Fallback_InnerCV": round(e_fb_cv, 4),
            "SPORT_Classic_PLS": round(e_sp_cl, 4),
            "SPORT_DSSNV_PLS": round(e_sp_ds, 4),
            "Gate_Escalate_Bilevel": round(e_esc, 4),
        }
        runtimes = {
            "Always_Raw": round(dt_raw, 4),
            "Always_SNV": round(dt_snv, 4),
            "SPORT_Classic_PLS": round(dt_sp_cl, 4),
            "SPORT_DSSNV_PLS": round(dt_sp_ds, 4),
            "Gate_Escalate_Bilevel": round(dt_esc, 4),
        }
        row = {
            "task_id": tid,
            "lab_id": t["lab_id"],
            "license": t["license"],
            "doi": t["doi"],
            "split_rule": t["split_rule"],
            "n_cal": t["n_cal"],
            "n_test": t["n_test"],
            "L": t["L"],
            "cal_test_hash_overlap": 0,
            "gate_decision": gate,
            "delta_loo_pct": float(p["delta_loo_pct"]),
            "t_loo": float(p["t_loo"]),
            "inner_cv_pick": cv_pick,
            "delta_cv_pct": float(p["delta_cv_pct"]),
            "legacy_v2_call_0_5": str(p["legacy_v2_call_0_5"]),
            "legacy_v2_pred_pct": float(p["legacy_v2_pred_pct"]),
            "rmsep_raw_z": round(e_raw, 4),
            "rmsep_snv_z": round(e_snv, 4),
            "emp_snv_vs_raw_pct": round(emp_pct, 2),
            "oracle_pls_z": round(oracle, 4),
            "methods": methods_z,
            "escalated_model": esc_src,
            "runtime_sec": runtimes,
            "sanity": sanity,
        }
        task_rows.append(row)
        print(
            f"[{i:02d}/{len(tasks)}] {tid:38s} | GATE={gate:7s} cv={cv_pick:3s} | "
            f"Raw={e_raw:.4f} SNV={e_snv:.4f} ({emp_pct:+6.2f}%) | "
            f"FbSNV={e_fb_snv:.4f} SP_D={e_sp_ds:.4f} EscBi={e_esc:.4f} | {sanity['status']}",
            flush=True,
        )

    emps = np.array([r["emp_snv_vs_raw_pct"] for r in task_rows], dtype=np.float64)
    dloo = np.array([r["delta_loo_pct"] for r in task_rows], dtype=np.float64)
    leg_p = np.array([r["legacy_v2_pred_pct"] for r in task_rows], dtype=np.float64)

    gate_eval = {
        "new_paired_loo_primary": _eval_method([r["gate_decision"] for r in task_rows], dloo, emps),
        "new_loo_sign_zero": _eval_method(["IMPROVE" if x < 0 else "WORSEN" for x in dloo], dloo, emps),
        "inner_cv_pls": _eval_method(
            ["IMPROVE" if r["inner_cv_pick"] == "SNV" else "WORSEN" for r in task_rows],
            np.array([r["delta_cv_pct"] for r in task_rows], dtype=np.float64),
            emps,
        ),
        "legacy_v2_tau_0_5": _eval_method([r["legacy_v2_call_0_5"] for r in task_rows], leg_p, emps),
    }

    oracles = np.array([r["oracle_pls_z"] for r in task_rows], dtype=np.float64)
    raw_a = np.array([r["methods"]["Always_Raw"] for r in task_rows], dtype=np.float64)
    cv_a = np.array([r["methods"]["InnerCV_PLS"] for r in task_rows], dtype=np.float64)
    strategy_eval = {}
    for mname in (
        "Always_Raw",
        "Always_SNV",
        "InnerCV_PLS",
        "Gate_Fallback_Raw",
        "Gate_Fallback_SNV",
        "Gate_Fallback_InnerCV",
        "SPORT_Classic_PLS",
        "SPORT_DSSNV_PLS",
        "Gate_Escalate_Bilevel",
    ):
        arr = np.array([r["methods"][mname] for r in task_rows], dtype=np.float64)
        asym = compute_asymmetric_regret(arr, oracles, severe_threshold_pct=5.0, penalty_weights=(1.0, 2.0, 5.0, 10.0))
        strategy_eval[mname] = {
            "mean_rmsep_z": round(float(np.mean(arr)), 4),
            "median_rmsep_z": round(float(np.median(arr)), 4),
            "wtl_vs_always_raw": [
                int(np.sum(arr < raw_a - 1e-12)),
                int(np.sum(np.abs(arr - raw_a) <= 1e-12)),
                int(np.sum(arr > raw_a + 1e-12)),
            ],
            "wtl_vs_inner_cv": [
                int(np.sum(arr < cv_a - 1e-12)),
                int(np.sum(np.abs(arr - cv_a) <= 1e-12)),
                int(np.sum(arr > cv_a + 1e-12)),
            ],
            **asym,
        }

    rc_curve = compute_risk_coverage_curve(
        t_values=np.array([r["t_loo"] for r in task_rows], dtype=np.float64),
        delta_loo_pcts=dloo,
        emp_pcts=emps,
        err_raw=raw_a,
        err_snv=np.array([r["methods"]["Always_SNV"] for r in task_rows], dtype=np.float64),
    )

    out = _sanitize_json(
        {
            "phase": "ROUND3_PHASE_2_UNBLINDED_VERIFIED",
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "prereg_commit_locked": locked_commit,
            "execution_git_head": exec_head,
            "prereg_json_sha256": pre_sha,
            "theory_py_git_blob_sha256_lf": frozen_blob_lf,
            "theory_py_worktree_sha256_lf": wt_lf,
            "n_tasks": len(task_rows),
            "n_distinct_labs": pre["n_distinct_labs"],
            "labs": pre["labs"],
            "elapsed_sec": round(time.time() - t_start, 2),
            "gate_decision_eval": gate_eval,
            "deployment_strategy_eval": strategy_eval,
            "risk_coverage": rc_curve,
            "sanity_summary": {
                "n_label_shift_alert": sum(1 for r in task_rows if r["sanity"]["label_scale_shift"]),
                "alert_tasks": [r["task_id"] for r in task_rows if r["sanity"]["label_scale_shift"]],
                "n_recalibrate": sum(1 for r in task_rows if r["sanity"]["spectrum_drift_status"] == "RECALIBRATE"),
            },
            "tasks": task_rows,
        }
    )

    out_json = ROOT / "_audit" / "res_prereg_round3_unseen_2026_11_24.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)

    gp = gate_eval["new_paired_loo_primary"]["all_signs"]
    gz = gate_eval["new_loo_sign_zero"]["all_signs"]
    gcv = gate_eval["inner_cv_pls"]["all_signs"]
    gleg = gate_eval["legacy_v2_tau_0_5"]["all_signs"]

    md_lines = [
        "# Round 3 零接触外部 Zenodo NIR 实验室预注册揭盲报告（6 实验室 / 13 项任务）",
        "",
        f"- **Phase 1 锁定提交**：`{locked_commit}`（执行时 HEAD：`{exec_head}`，13 项 `split_sha256` 100% 校验且 `cal/test` 光谱重叠全部为 `0`）",
        f"- **冻结理论 LF SHA-256**：`chemdl/theory.py` = `{frozen_blob_lf}` · 总耗时 `{out['elapsed_sec']:.2f}s`",
        f"- **预注册锁单**：[`PREREG_ROUND3_UNSEEN_2026-11-24.md`](./PREREG_ROUND3_UNSEEN_2026-11-24.md)",
        "",
        "## 一、SNV 决策质量对比（13 项全新外部实验室任务）",
        "",
        "| 方法 | Cov | Acc (命中/总数) | TPR / TNR | BAC | MCC | Spearman ρ (p 值) |",
        "|---|---|---|---:|---:|---:|---|",
        f"| **Paired-LOO 主门控 (`|t|>=2, |d|>=1%`)** | {gp['coverage']*100:.1f}% ({gp['n']}/13) | **{gp['acc']}** ({gp['tp']+gp['tn']}/{gp['n']}) | {gp['tpr']} / {gp['tnr']} | {gp['balanced_acc']} | {gp['mcc']} | {gate_eval['new_paired_loo_primary']['spearman_rho']:+.4f} (p={gate_eval['new_paired_loo_primary']['spearman_p']:.3g}) |",
        f"| Paired-LOO 零弃权 (`dLOO < 0`) | 100.0% (13/13) | {gz['acc']} ({gz['tp']+gz['tn']}/13) | {gz['tpr']} / {gz['tnr']} | {gz['balanced_acc']} | {gz['mcc']} | {gate_eval['new_loo_sign_zero']['spearman_rho']:+.4f} (p={gate_eval['new_loo_sign_zero']['spearman_p']:.3g}) |",
        f"| InnerCV_PLS (校准内 5 折选择) | 100.0% (13/13) | {gcv['acc']} ({gcv['tp']+gcv['tn']}/13) | {gcv['tpr']} / {gcv['tnr']} | {gcv['balanced_acc']} | {gcv['mcc']} | {gate_eval['inner_cv_pls']['spearman_rho']:+.4f} (p={gate_eval['inner_cv_pls']['spearman_p']:.3g}) |",
        f"| 旧版 `v2` (`τ = 0.5%`) | {gleg['coverage']*100:.1f}% ({gleg['n']}/13) | {gleg['acc']} ({gleg['tp']+gleg['tn']}/{gleg['n']}) | {gleg['tpr']} / {gleg['tnr']} | {gleg['balanced_acc']} | {gleg['mcc']} | {gate_eval['legacy_v2_tau_0_5']['spearman_rho']:+.4f} (p={gate_eval['legacy_v2_tau_0_5']['spearman_p']:.3g}) |",
        "",
        f"## 二、端到端部署策略与非对称代价对比（AURC = {rc_curve['aurc_selective_error']:.4f}）",
        "",
        "| 策略（同 PLS / SPORT / 异构） | mean_z | 几何均比 | 均值遗憾 | 最坏遗憾 | 严重数(>=5%) | w=10 加权代价 | 胜/平/负 vs Raw |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mname in (
        "Always_Raw",
        "Always_SNV",
        "InnerCV_PLS",
        "SPORT_Classic_PLS",
        "SPORT_DSSNV_PLS",
        "Gate_Fallback_Raw",
        "Gate_Fallback_SNV",
        "Gate_Fallback_InnerCV",
        "Gate_Escalate_Bilevel",
    ):
        s = strategy_eval[mname]
        w_str = "/".join(str(x) for x in s["wtl_vs_always_raw"])
        md_lines.append(
            f"| `{mname}` | {s['mean_rmsep_z']:.4f} | {s['geom_ratio_vs_oracle']:.4f} | {s['mean_regret_pct']:+.2f}% | "
            f"{s['max_regret_pct']:+.2f}% | {s['n_severe_ge_thr']} | {s['weighted_cost_by_w']['10.0']:.2f} | {w_str} |"
        )

    md_lines += [
        "",
        "## 三、逐任务揭盲明细（13 项）",
        "",
        "| # | Task ID | GATE | CV | Raw_z | SNV_z | d_emp | FbSNV | FbCV | SP_Classic | SP_DSSNV | EscBi | Sanity |",
        "|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(task_rows, 1):
        m = r["methods"]
        md_lines.append(
            f"| {i} | `{r['task_id']}` | **{r['gate_decision']}** | {r['inner_cv_pick']} | {r['rmsep_raw_z']:.4f} | {r['rmsep_snv_z']:.4f} | "
            f"**{r['emp_snv_vs_raw_pct']:+.2f}%** | {m['Gate_Fallback_SNV']:.4f} | {m['Gate_Fallback_InnerCV']:.4f} | "
            f"{m['SPORT_Classic_PLS']:.4f} | {m['SPORT_DSSNV_PLS']:.4f} | {m['Gate_Escalate_Bilevel']:.4f} | {r['sanity']['status']} |"
        )

    out_md = ROOT / "reports" / "antigravity" / "UNBLIND_ROUND3_UNSEEN_2026-11-24.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nSaved JSON: {out_json}\nSaved MD:   {out_md}")


if __name__ == "__main__":
    main()
