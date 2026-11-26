"""Phase B · Phase 2：验证冻结身份并在 18 项严格独立留出（cal -> test）任务上揭盲评估真实端到端部署效果。

对比两类对照口径（严格分栏，不混比较）：
  1. 同模型头（fit_pls）六种部署策略：
     Always_Raw / Always_SNV / InnerCV_PLS / Gate_Fallback_Raw / Gate_Fallback_SNV / Gate_Fallback_InnerCV
  2. 异构升级策略（单独列报）：
     Gate_Escalate_Bilevel（IMPROVE->SNV_PLS, WORSEN->Raw_PLS, ABSTAIN->fit_bilevel_diffprep）
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._blind_theory_loaders import (
    file_sha256,
    file_sha256_lf,
    git_blob_sha256,
)
from _audit._deploy_holdout_loaders import load_deploy_holdout_tasks
from _audit.prereg_snv_gate_round2_phase2_unblind import _eval_method
from _audit.prereg_theory_blind_phase2_unblind import _sanitize_json
from chemdl.diffprep import fit_bilevel_diffprep
from chemdl.transfer import fit_pls, fit_prep_fn, rmsep


def _strategy_summary(errs: np.ndarray, oracles: np.ndarray, raw_errs: np.ndarray, snv_errs: np.ndarray, cv_errs: np.ndarray) -> dict:
    ratios = errs / np.clip(oracles, 1e-12, None)
    regrets = (ratios - 1.0) * 100.0
    return {
        "n": int(len(errs)),
        "mean_rmsep_z": round(float(np.mean(errs)), 4),
        "geom_ratio_vs_oracle": round(float(np.exp(np.mean(np.log(np.clip(ratios, 1e-12, None))))), 4),
        "mean_regret_pct_vs_oracle": round(float(np.mean(regrets)), 2),
        "median_regret_pct_vs_oracle": round(float(np.median(regrets)), 2),
        "max_regret_pct_vs_oracle": round(float(np.max(regrets)), 2),
        "n_severe_regrets_ge5pct": int(np.sum(regrets >= 5.0)),
        "n_oracle_exact_hits": int(np.sum(np.abs(errs - oracles) <= 1e-12)),
        "wtl_vs_always_raw": [
            int(np.sum(errs < raw_errs - 1e-12)),
            int(np.sum(np.abs(errs - raw_errs) <= 1e-12)),
            int(np.sum(errs > raw_errs + 1e-12)),
        ],
        "wtl_vs_always_snv": [
            int(np.sum(errs < snv_errs - 1e-12)),
            int(np.sum(np.abs(errs - snv_errs) <= 1e-12)),
            int(np.sum(errs > snv_errs + 1e-12)),
        ],
        "wtl_vs_inner_cv": [
            int(np.sum(errs < cv_errs - 1e-12)),
            int(np.sum(np.abs(errs - cv_errs) <= 1e-12)),
            int(np.sum(errs > cv_errs + 1e-12)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B Deploy Holdout Phase 2 unblinding")
    parser.add_argument("--verify-only", action="store_true", help="Verify lock chain & existing result JSON.")
    args = parser.parse_args()

    t0 = time.time()
    pre_path = ROOT / "_audit" / "prereg_deploy_holdout_2026_11_24.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    pre_sha = file_sha256(pre_path)

    theory_path = ROOT / "chemdl" / "theory.py"
    locked_commit = str(pre["git_head_frozen"])
    frozen_blob_lf = git_blob_sha256(ROOT, locked_commit, "chemdl/theory.py", lf_normalize=True)
    assert frozen_blob_lf == pre["theory_py_git_blob_sha256_lf"]
    wt_lf = file_sha256_lf(theory_path)
    exec_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    tasks = load_deploy_holdout_tasks(ROOT)
    assert len(tasks) == pre["n_tasks"] == 18
    pre_by_id = {r["task_id"]: r for r in pre["tasks"]}

    if args.verify_only:
        res = json.loads((ROOT / "_audit" / "res_prereg_deploy_holdout_2026_11_24.json").read_text(encoding="utf-8"))
        for t in tasks:
            p = pre_by_id[t["task_id"]]
            assert t["split_sha256"] == p["split_sha256"] and t["cal_test_hash_overlap"] == 0
        print(f"[verify-only PASS] Deploy Holdout locked={locked_commit[:7]}, exec={exec_head[:7]}, {len(res['tasks'])} tasks verified.")
        return

    assert wt_lf == frozen_blob_lf, f"chemdl/theory.py LF SHA-256 mismatch: {wt_lf} vs {frozen_blob_lf}"
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "chemdl/theory.py", "_audit/prereg_deploy_holdout_2026_11_24.json"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert not dirty, f"Frozen deploy files dirty: {dirty}"

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

        pr_raw, _ = fit_pls(X_cal, y_cal_n, kmax=kmax, inner=inner, per_col=True, seed=0)
        fn_snv = fit_prep_fn("snv", X_cal)
        pr_snv, _ = fit_pls(fn_snv(X_cal), y_cal_n, kmax=kmax, inner=inner, per_col=True, seed=0)

        e_raw = float(rmsep(pr_raw(X_test), y_test_n))
        e_snv = float(rmsep(pr_snv(fn_snv(X_test)), y_test_n))
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
            esc_source = "SNV_PLS"
        elif gate == "WORSEN":
            e_esc = e_raw
            esc_source = "Raw_PLS"
        else:
            pr_bi, _ = fit_bilevel_diffprep(X_cal, y_cal_n, steps=40, max_opt_n=800, seed=0)
            pred_bi = np.asarray(pr_bi(X_test), dtype=np.float64).reshape(-1, 1)
            e_esc = float(rmsep(pred_bi, y_test_n))
            esc_source = "BilevelDiffPrep"

        row = {
            "task_id": tid,
            "stratum": t["stratum"],
            "domain": t["domain"],
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
            "strategies_same_pls": {
                "Always_Raw": round(e_raw, 4),
                "Always_SNV": round(e_snv, 4),
                "InnerCV_PLS": round(e_cv, 4),
                "Gate_Fallback_Raw": round(e_fb_raw, 4),
                "Gate_Fallback_SNV": round(e_fb_snv, 4),
                "Gate_Fallback_InnerCV": round(e_fb_cv, 4),
            },
            "hetero_workflow": {
                "Gate_Escalate_Bilevel": round(e_esc, 4),
                "escalated_model": esc_source,
            },
        }
        task_rows.append(row)
        print(
            f"[{i:02d}/{len(tasks)}] {tid:44s} | GATE={gate:7s} cv={cv_pick:3s} | "
            f"Raw={e_raw:.4f} SNV={e_snv:.4f} ({emp_pct:+6.2f}%) | "
            f"FbCV={e_fb_cv:.4f} EscBi={e_esc:.4f} ({esc_source})",
            flush=True,
        )

    # Gate decision metrics vs legacy v2 across 18 strictly independent holdouts
    emps = np.array([r["emp_snv_vs_raw_pct"] for r in task_rows], dtype=np.float64)
    dloo = np.array([r["delta_loo_pct"] for r in task_rows], dtype=np.float64)
    leg_p = np.array([r["legacy_v2_pred_pct"] for r in task_rows], dtype=np.float64)
    gate_eval = {
        "new_paired_loo_primary": _eval_method([r["gate_decision"] for r in task_rows], dloo, emps),
        "new_loo_sign_zero": _eval_method(["IMPROVE" if x < 0 else "WORSEN" for x in dloo], dloo, emps),
        "inner_cv_pls": _eval_method(["IMPROVE" if r["inner_cv_pick"] == "SNV" else "WORSEN" for r in task_rows], np.array([r["delta_cv_pct"] for r in task_rows]), emps),
        "legacy_v2_tau_0_5": _eval_method([r["legacy_v2_call_0_5"] for r in task_rows], leg_p, emps),
    }

    def _agg_strategies(sub: list[dict]) -> dict:
        orac = np.array([r["oracle_pls_z"] for r in sub], dtype=np.float64)
        raw_a = np.array([r["strategies_same_pls"]["Always_Raw"] for r in sub], dtype=np.float64)
        snv_a = np.array([r["strategies_same_pls"]["Always_SNV"] for r in sub], dtype=np.float64)
        cv_a = np.array([r["strategies_same_pls"]["InnerCV_PLS"] for r in sub], dtype=np.float64)
        res = {"same_pls": {}, "heterogeneous": {}}
        for sk in ("Always_Raw", "Always_SNV", "InnerCV_PLS", "Gate_Fallback_Raw", "Gate_Fallback_SNV", "Gate_Fallback_InnerCV"):
            arr = np.array([r["strategies_same_pls"][sk] for r in sub], dtype=np.float64)
            res["same_pls"][sk] = _strategy_summary(arr, orac, raw_a, snv_a, cv_a)
        esc_a = np.array([r["hetero_workflow"]["Gate_Escalate_Bilevel"] for r in sub], dtype=np.float64)
        res["heterogeneous"]["Gate_Escalate_Bilevel"] = _strategy_summary(esc_a, orac, raw_a, snv_a, cv_a)
        return res

    strata_order = [
        ("ALL_18", task_rows),
        ("A_Unseen_Spectra_Temporal", [r for r in task_rows if r["stratum"] == "A_Unseen_Spectra_Temporal"]),
        ("B_CRAW_UnseenSeries_Temporal", [r for r in task_rows if r["stratum"] == "B_CRAW_UnseenSeries_Temporal"]),
        ("C_Official_CrossInst_CowGroup", [r for r in task_rows if r["stratum"] == "C_Official_CrossInst_CowGroup"]),
        ("D_OSSL_UnseenProps_Holdout", [r for r in task_rows if r["stratum"] == "D_OSSL_UnseenProps_Holdout"]),
    ]
    strategy_eval = {name: _agg_strategies(sub) for name, sub in strata_order}

    out = _sanitize_json(
        {
            "phase": "DEPLOY_PHASE_2_UNBLINDED_VERIFIED",
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "prereg_commit_locked": locked_commit,
            "execution_git_head": exec_head,
            "prereg_json_sha256": pre_sha,
            "theory_py_git_blob_sha256_lf": frozen_blob_lf,
            "theory_py_worktree_sha256_lf": wt_lf,
            "n_tasks": len(task_rows),
            "elapsed_sec": round(time.time() - t0, 2),
            "gate_decision_eval": gate_eval,
            "deployment_strategy_eval": strategy_eval,
            "tasks": task_rows,
        }
    )

    out_json = ROOT / "_audit" / "res_prereg_deploy_holdout_2026_11_24.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)

    # 撰写严谨揭盲部署对比报告
    all_s = strategy_eval["ALL_18"]
    g_p = gate_eval["new_paired_loo_primary"]["all_signs"]
    g_z = gate_eval["new_loo_sign_zero"]["all_signs"]
    g_leg = gate_eval["legacy_v2_tau_0_5"]["all_signs"]
    g_cv = gate_eval["inner_cv_pls"]["all_signs"]

    lines = [
        "# Phase B 独立留出端到端部署对照揭盲报告（18 项单次 cal → test 任务）",
        "",
        f"- **Phase 1 锁定提交**：`{locked_commit}`（执行时 HEAD：`{exec_head}`，18 项 `split_sha256` 100% 校验且 `cal/test` 字节重叠全部为 `0`）",
        f"- **冻结理论源码 LF SHA-256**：[`chemdl/theory.py`](../../chemdl/theory.py) = `{frozen_blob_lf}`",
        f"- **预注册锁定对照**：[`PREREG_DEPLOY_HOLDOUT_2026-11-24.md`](./PREREG_DEPLOY_HOLDOUT_2026-11-24.md) · 耗时 `{out['elapsed_sec']}s`",
        "",
        "## 一、SNV 决策质量对比（18 项严格独立留出，每项单次划分、零标签交叉）",
        "",
        "| 方法 | Cov | Acc (命中/总数) | TPR / TNR | BAC | MCC | Spearman ρ (p 值) |",
        "|---|---|---|---:|---:|---:|---|",
        f"| **Paired-LOO 主门控 (`|t|≥2, |Δ|≥1%`)** | {g_p['coverage']:.1%} ({g_p['n']}/18) | **{g_p['acc']}** ({g_p['tp']+g_p['tn']}/{g_p['n']}) | {g_p['tpr']} / {g_p['tnr']} | {g_p['balanced_acc']} | {g_p['mcc']} | {gate_eval['new_paired_loo_primary']['spearman_rho']:+.4f} (p={gate_eval['new_paired_loo_primary']['spearman_p']:.3g}) |",
        f"| Paired-LOO 零弃权 (`Δ_LOO < 0`) | 100.0% (18/18) | {g_z['acc']} ({g_z['tp']+g_z['tn']}/18) | {g_z['tpr']} / {g_z['tnr']} | {g_z['balanced_acc']} | {g_z['mcc']} | {gate_eval['new_loo_sign_zero']['spearman_rho']:+.4f} (p={gate_eval['new_loo_sign_zero']['spearman_p']:.3g}) |",
        f"| InnerCV_PLS (5 折校准内选择) | 100.0% (18/18) | {g_cv['acc']} ({g_cv['tp']+g_cv['tn']}/18) | {g_cv['tpr']} / {g_cv['tnr']} | {g_cv['balanced_acc']} | {g_cv['mcc']} | {gate_eval['inner_cv_pls']['spearman_rho']:+.4f} (p={gate_eval['inner_cv_pls']['spearman_p']:.3g}) |",
        f"| 旧版 `v2` (`τ = 0.5%`) | {g_leg['coverage']:.1%} ({g_leg['n']}/18) | {g_leg['acc']} ({g_leg['tp']+g_leg['tn']}/{g_leg['n']}) | {g_leg['tpr']} / {g_leg['tnr']} | {g_leg['balanced_acc']} | {g_leg['mcc']} | {gate_eval['legacy_v2_tau_0_5']['spearman_rho']:+.4f} (p={gate_eval['legacy_v2_tau_0_5']['spearman_p']:.3g}) |",
        "",
        "## 二、同模型头（fit_pls）端到端部署策略汇总（N = 18）",
        "",
        "| 策略 | 均值 RMSEP_z | 相对 Oracle 几何均比 | 相对 Oracle 均值遗憾 | 最大最坏遗憾 | 严重遗憾数 (≥+5%) | Oracle 命中 | 胜/平/负 vs Raw |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sk in ("Always_Raw", "Always_SNV", "InnerCV_PLS", "Gate_Fallback_Raw", "Gate_Fallback_SNV", "Gate_Fallback_InnerCV"):
        m = all_s["same_pls"][sk]
        wtl = "/".join(str(x) for x in m["wtl_vs_always_raw"])
        lines.append(
            f"| `{sk}` | {m['mean_rmsep_z']:.4f} | {m['geom_ratio_vs_oracle']:.4f} | {m['mean_regret_pct_vs_oracle']:+.2f}% "
            f"| {m['max_regret_pct_vs_oracle']:+.2f}% | {m['n_severe_regrets_ge5pct']} | {m['n_oracle_exact_hits']}/18 | {wtl} |"
        )

    esc_m = all_s["heterogeneous"]["Gate_Escalate_Bilevel"]
    lines.extend([
        "",
        "## 三、异构升级策略独立对照（`ABSTAIN` 升级至 `BilevelDiffPrep`，不与纯 PLS 同台混比）",
        "",
        "| 策略 | 均值 RMSEP_z | 相对 PLS Oracle 几何均比 | 相对 PLS Oracle 均值遗憾 | 最大遗憾 | 胜/平/负 vs `InnerCV_PLS` |",
        "|---|---:|---:|---:|---:|---|",
        f"| `Gate_Escalate_Bilevel` | {esc_m['mean_rmsep_z']:.4f} | {esc_m['geom_ratio_vs_oracle']:.4f} | {esc_m['mean_regret_pct_vs_oracle']:+.2f}% | {esc_m['max_regret_pct_vs_oracle']:+.2f}% | {'/'.join(str(x) for x in esc_m['wtl_vs_inner_cv'])} |",
        "",
        "## 四、逐任务端到端部署实测明细（18 项）",
        "",
        "| # | Task ID | 分层 | GATE | InnerCV | Raw_z | SNV_z | Δemp | FbRaw | FbSNV | FbCV | EscBi |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for idx, r in enumerate(task_rows, 1):
        sp = r["strategies_same_pls"]
        eb = r["hetero_workflow"]["Gate_Escalate_Bilevel"]
        lines.append(
            f"| {idx} | `{r['task_id']}` | `{r['stratum']}` | **{r['gate_decision']}** | {r['inner_cv_pick']} "
            f"| {r['rmsep_raw_z']:.4f} | {r['rmsep_snv_z']:.4f} | **{r['emp_snv_vs_raw_pct']:+.2f}%** "
            f"| {sp['Gate_Fallback_Raw']:.4f} | {sp['Gate_Fallback_SNV']:.4f} | {sp['Gate_Fallback_InnerCV']:.4f} | {eb:.4f} |"
        )
    lines.append("")

    out_md = ROOT / "reports" / "antigravity" / "UNBLIND_DEPLOY_HOLDOUT_2026-11-24.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDeploy Phase 2 complete -> {out_json} & {out_md} ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
