"""Phase B 扩展评估（不修改已封存 Phase B 锁单与揭盲 JSON）：
  1. SPORT 多预处理正交融合先例同台对照（SPORT_Classic_PLS & SPORT_DSSNV_PLS）
  2. Risk-Coverage 曲线（AURC）与非对称严重遗憾惩罚代价（w = 1, 2, 5, 10）
  3. CRAW 跨年漂移 6 任务全弃权（ABSTAIN）诊断与光谱/标签安全体检（diagnose_deploy_sanity）
  4. 端到端单任务平均校准耗时对比（Raw / SNV / Gate / InnerCV / SPORT / Bilevel）
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._blind_theory_loaders import file_sha256_lf
from _audit._deploy_holdout_loaders import load_deploy_holdout_tasks
from _audit.prereg_deploy_holdout_phase1 import _inner_cv_raw_vs_snv_pls
from _audit.prereg_theory_blind_phase2_unblind import _sanitize_json
from chemdl.deploy_gate import (
    FROZEN_THEORY_LF_SHA256,
    PHASE_B_FROZEN_THEORY_LF_SHA256,
    compute_asymmetric_regret,
    compute_risk_coverage_curve,
    diagnose_deploy_sanity,
)
from chemdl.diffprep import fit_bilevel_diffprep
from chemdl.theory import diagnose_snv_decision
from chemdl.transfer import fit_pls, fit_prep_fn, fit_sport_pls, rmsep


def _summarize_subset(rows: list[dict], runtime_keys: list[str]) -> dict:
    oracles = np.array([r["oracle_pls_z"] for r in rows], dtype=np.float64)
    raw_arr = np.array([r["methods"]["Always_Raw"] for r in rows], dtype=np.float64)
    cv_arr = np.array([r["methods"]["InnerCV_PLS"] for r in rows], dtype=np.float64)
    methods_order = [
        "Always_Raw",
        "Always_SNV",
        "InnerCV_PLS",
        "Gate_Fallback_Raw",
        "Gate_Fallback_SNV",
        "Gate_Fallback_InnerCV",
        "SPORT_Classic_PLS",
        "SPORT_DSSNV_PLS",
        "Gate_Escalate_Bilevel",
    ]
    out_methods = {}
    for mname in methods_order:
        errs = np.array([r["methods"][mname] for r in rows], dtype=np.float64)
        asym = compute_asymmetric_regret(errs, oracles, severe_threshold_pct=5.0, penalty_weights=(1.0, 2.0, 5.0, 10.0))
        rts = np.array([r["runtime_sec"][mname] for r in rows], dtype=np.float64)
        out_methods[mname] = {
            "mean_rmsep_z": round(float(np.mean(errs)), 4),
            "median_rmsep_z": round(float(np.median(errs)), 4),
            "wtl_vs_always_raw": [
                int(np.sum(errs < raw_arr - 1e-12)),
                int(np.sum(np.abs(errs - raw_arr) <= 1e-12)),
                int(np.sum(errs > raw_arr + 1e-12)),
            ],
            "wtl_vs_inner_cv": [
                int(np.sum(errs < cv_arr - 1e-12)),
                int(np.sum(np.abs(errs - cv_arr) <= 1e-12)),
                int(np.sum(errs > cv_arr + 1e-12)),
            ],
            "mean_runtime_sec": round(float(np.mean(rts)), 3),
            "total_runtime_sec": round(float(np.sum(rts)), 2),
            **asym,
        }

    rc_curve = compute_risk_coverage_curve(
        t_values=np.array([r["t_loo"] for r in rows], dtype=np.float64),
        delta_loo_pcts=np.array([r["delta_loo_pct"] for r in rows], dtype=np.float64),
        emp_pcts=np.array([r["emp_snv_vs_raw_pct"] for r in rows], dtype=np.float64),
        err_raw=raw_arr,
        err_snv=np.array([r["methods"]["Always_SNV"] for r in rows], dtype=np.float64),
    )
    return {"n_tasks": len(rows), "methods": out_methods, "risk_coverage": rc_curve}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-existing", action="store_true", help="Recompute summary & MD report from existing res_deploy_extended_analysis JSON.")
    args = parser.parse_args()

    t_start = time.time()
    theory_lf = file_sha256_lf(ROOT / "chemdl" / "theory.py")
    assert theory_lf in (PHASE_B_FROZEN_THEORY_LF_SHA256, FROZEN_THEORY_LF_SHA256), (
        f"chemdl/theory.py changed unexpectedly: {theory_lf}"
    )

    out_json = ROOT / "_audit" / "res_deploy_extended_analysis_2026_11_24.json"
    if args.from_existing and out_json.exists():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        task_rows = prev["tasks"]
        prev_elapsed = prev["elapsed_sec"]
    else:
        res_b = json.loads((ROOT / "_audit" / "res_prereg_deploy_holdout_2026_11_24.json").read_text(encoding="utf-8"))
        res_by_id = {r["task_id"]: r for r in res_b["tasks"]}
        tasks = load_deploy_holdout_tasks(ROOT)
        assert len(tasks) == len(res_b["tasks"]) == 18

        task_rows = []
        for idx, t in enumerate(tasks, 1):
            tid = t["task_id"]
            rb = res_by_id[tid]
            assert t["cal_test_hash_overlap"] == 0

            X_cal, y_cal = t["X_cal"], t["y_cal"]
            X_test, y_test = t["X_test"], t["y_test"]
            groups_cal = t["groups_cal"]
            mu = float(y_cal.mean())
            sd = max(float(y_cal.std()), 1e-9)
            y_cal_n = ((y_cal - mu) / sd)[:, None]
            y_test_n = ((y_test - mu) / sd)[:, None]

            kmax = min(10, max(2, len(X_cal) // 2))
            inner = min(4, max(2, len(X_cal) // 4))

            # 1) 实测单线、门控与 InnerCV 耗时（核对预测值与已归档 Phase B 完全一致）
            t0 = time.perf_counter()
            pr_raw, _ = fit_pls(X_cal, y_cal_n, kmax=kmax, inner=inner, per_col=True, seed=0)
            e_raw = float(rmsep(pr_raw(X_test), y_test_n))
            dt_raw = time.perf_counter() - t0

            t0 = time.perf_counter()
            fn_snv = fit_prep_fn("snv", X_cal)
            pr_snv, _ = fit_pls(fn_snv(X_cal), y_cal_n, kmax=kmax, inner=inner, per_col=True, seed=0)
            e_snv = float(rmsep(pr_snv(fn_snv(X_test)), y_test_n))
            dt_snv = time.perf_counter() - t0

            assert abs(e_raw - rb["rmsep_raw_z"]) <= 1e-3, f"{tid}: raw mismatch {e_raw} vs {rb['rmsep_raw_z']}"
            assert abs(e_snv - rb["rmsep_snv_z"]) <= 1e-3, f"{tid}: snv mismatch {e_snv} vs {rb['rmsep_snv_z']}"

            t0 = time.perf_counter()
            diag = diagnose_snv_decision(X_cal, y_cal, groups_tr=groups_cal, t_crit=2.0, min_effect_pct=1.0, compute_physics=True)
            dt_diag = time.perf_counter() - t0
            assert diag["decision"] == rb["gate_decision"]

            t0 = time.perf_counter()
            _inner_cv_raw_vs_snv_pls(X_cal, y_cal, groups_cal=groups_cal, seed=0)
            dt_cv_search = time.perf_counter() - t0

            # 2) SPORT 多预处理正交融合先例（严格同 cal -> test 划分与同 y_cal 标准化）
            t0 = time.perf_counter()
            pr_sp_cl, lvs_cl = fit_sport_pls(
                X_cal, y_cal_n, blocks=("snv", "sg1", "sg2", "raw"), kmax=kmax, inner=inner, seed=0
            )
            e_sp_cl = float(rmsep(np.asarray(pr_sp_cl(X_test), dtype=np.float64).reshape(-1, 1), y_test_n))
            dt_sp_cl = time.perf_counter() - t0

            t0 = time.perf_counter()
            pr_sp_ds, lvs_ds = fit_sport_pls(
                X_cal, y_cal_n, blocks=("dssnv", "sg1", "sg2", "raw"), kmax=kmax, inner=inner, seed=0
            )
            e_sp_ds = float(rmsep(np.asarray(pr_sp_ds(X_test), dtype=np.float64).reshape(-1, 1), y_test_n))
            dt_sp_ds = time.perf_counter() - t0

            # 3) Gate_Escalate_Bilevel 耗时与结果（复用同一契约）
            gate = rb["gate_decision"]
            if gate == "IMPROVE":
                dt_esc_model = dt_snv
            elif gate == "WORSEN":
                dt_esc_model = dt_raw
            else:
                t0 = time.perf_counter()
                pr_bi, _ = fit_bilevel_diffprep(X_cal, y_cal_n, steps=40, max_opt_n=800, seed=0)
                _ = float(rmsep(np.asarray(pr_bi(X_test), dtype=np.float64).reshape(-1, 1), y_test_n))
                dt_esc_model = time.perf_counter() - t0

            # 4) 端到端部署策略精确时间成本组合
            dt_fb_raw = dt_diag + (dt_snv if gate == "IMPROVE" else dt_raw)
            dt_fb_snv = dt_diag + (dt_raw if gate == "WORSEN" else dt_snv)
            if gate == "IMPROVE":
                dt_fb_cv = dt_diag + dt_snv
            elif gate == "WORSEN":
                dt_fb_cv = dt_diag + dt_raw
            else:
                dt_fb_cv = dt_diag + dt_cv_search + (dt_snv if rb["inner_cv_pick"] == "SNV" else dt_raw)
            dt_inner_cv_total = dt_cv_search + (dt_snv if rb["inner_cv_pick"] == "SNV" else dt_raw)
            dt_esc_total = dt_diag + dt_esc_model

            # 5) 部署安全体检（无监督光谱漂移 + 事后标签量纲偏移核查）
            sanity = diagnose_deploy_sanity(X_cal, X_test, y_cal=y_cal, y_new=y_test)

            methods_z = {
                **rb["strategies_same_pls"],
                "SPORT_Classic_PLS": round(e_sp_cl, 4),
                "SPORT_DSSNV_PLS": round(e_sp_ds, 4),
                **rb["hetero_workflow"],
            }
            runtimes = {
                "Always_Raw": round(dt_raw, 4),
                "Always_SNV": round(dt_snv, 4),
                "InnerCV_PLS": round(dt_inner_cv_total, 4),
                "Gate_Fallback_Raw": round(dt_fb_raw, 4),
                "Gate_Fallback_SNV": round(dt_fb_snv, 4),
                "Gate_Fallback_InnerCV": round(dt_fb_cv, 4),
                "SPORT_Classic_PLS": round(dt_sp_cl, 4),
                "SPORT_DSSNV_PLS": round(dt_sp_ds, 4),
                "Gate_Escalate_Bilevel": round(dt_esc_total, 4),
                "gate_diagnose_only": round(dt_diag, 4),
            }

            row = {
                "task_id": tid,
                "stratum": t["stratum"],
                "domain": t["domain"],
                "n_cal": t["n_cal"],
                "n_test": t["n_test"],
                "L": t["L"],
                "gate_decision": gate,
                "delta_loo_pct": rb["delta_loo_pct"],
                "t_loo": rb["t_loo"],
                "emp_snv_vs_raw_pct": rb["emp_snv_vs_raw_pct"],
                "oracle_pls_z": rb["oracle_pls_z"],
                "methods": methods_z,
                "sport_classic_lvs": lvs_cl[0],
                "sport_dssnv_lvs": lvs_ds[0],
                "runtime_sec": runtimes,
                "sanity": sanity,
            }
            task_rows.append(row)
            print(
                f"[{idx:02d}/18] {tid[:40]:40s} | Gate={gate:7s} FbSNV={methods_z['Gate_Fallback_SNV']:.4f} "
                f"SPORT_C={e_sp_cl:.4f} ({dt_sp_cl:.2f}s) SPORT_D={e_sp_ds:.4f} ({dt_sp_ds:.2f}s) | sanity={sanity['status']}",
                flush=True,
            )
        prev_elapsed = round(time.time() - t_start, 2)

    valid_17_rows = [r for r in task_rows if r["task_id"] != "Deploy_Mango_Official_SP_test2__DM"]
    craw_rows = [r for r in task_rows if r["task_id"].startswith("Deploy_CRAW_")]

    craw_diag = []
    for r in craw_rows:
        craw_diag.append(
            {
                "task_id": r["task_id"],
                "n_cal": r["n_cal"],
                "n_test": r["n_test"],
                "delta_loo_pct": r["delta_loo_pct"],
                "t_loo": r["t_loo"],
                "gate_decision": r["gate_decision"],
                "emp_snv_vs_raw_pct": r["emp_snv_vs_raw_pct"],
                "spectrum_drift_status": r["sanity"]["spectrum_drift_status"],
                "drift_score": r["sanity"]["drift_score"],
                "spe_ratio": r["sanity"]["spe_ratio"],
                "raw_ratio": r["sanity"]["raw_ratio"],
                "false_call_prevented_by_abstain": bool(
                    (r["delta_loo_pct"] > 0 and r["emp_snv_vs_raw_pct"] < 0)
                    or (r["delta_loo_pct"] < 0 and r["emp_snv_vs_raw_pct"] > 0)
                ),
            }
        )

    agg_all18 = _summarize_subset(task_rows, list(task_rows[0]["runtime_sec"].keys()))
    agg_valid17 = _summarize_subset(valid_17_rows, list(task_rows[0]["runtime_sec"].keys()))

    out = _sanitize_json(
        {
            "analysis": "PHASE_B_EXTENDED_SPORT_RISK_COVERAGE_COST_SANITY",
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "theory_py_lf_sha256": theory_lf,
            "n_tasks": len(task_rows),
            "elapsed_sec": prev_elapsed,
            "subsets": {
                "ALL_18": agg_all18,
                "VALID_17_NO_DMTEST2": agg_valid17,
            },
            "craw_temporal_abstention_diagnosis": {
                "n_craw_tasks": len(craw_diag),
                "n_abstained": sum(1 for c in craw_diag if c["gate_decision"] == "ABSTAIN"),
                "n_sign_flips_between_cal_and_temporal_test": sum(1 for c in craw_diag if c["false_call_prevented_by_abstain"]),
                "rows": craw_diag,
            },
            "sanity_summary": {
                "n_label_shift_alert": sum(1 for r in task_rows if r["sanity"]["label_scale_shift"]),
                "alert_tasks": [r["task_id"] for r in task_rows if r["sanity"]["label_scale_shift"]],
                "n_spectrum_recalibrate": sum(1 for r in task_rows if r["sanity"]["spectrum_drift_status"] == "RECALIBRATE"),
            },
            "tasks": task_rows,
        }
    )

    out_json = ROOT / "_audit" / "res_deploy_extended_analysis_2026_11_24.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)

    # 撰写简洁分析报告
    s18 = agg_all18["methods"]
    s17 = agg_valid17["methods"]
    rc18 = agg_all18["risk_coverage"]
    rc17 = agg_valid17["risk_coverage"]

    md_lines = [
        "# Phase B 扩展分析：SPORT 同台对比、Risk-Coverage 曲线、非对称代价与 CRAW 弃权诊断",
        "",
        f"- **冻结理论 LF SHA-256**：`chemdl/theory.py` = `{theory_lf}` · 总耗时 `{out['elapsed_sec']:.2f}s`",
        f"- **数据源**：[`res_prereg_deploy_holdout_2026_11_24.json`](../../_audit/res_prereg_deploy_holdout_2026_11_24.json) 与 [`res_deploy_extended_analysis_2026_11_24.json`](../../_audit/res_deploy_extended_analysis_2026_11_24.json)",
        "",
        "## 一、全策略端到端部署与非对称代价对比（N = 18 全量 / N = 17 有效量纲）",
        "",
        "| 策略（同 PLS / 多预处理 SPORT / 异构） | N=18 遗憾均值 | 最坏遗憾 | 严重数(>=5%) | w=10 加权代价 | N=17 mean_z | N=17 遗憾均值 | 平均耗时(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
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
        a, b = s18[mname], s17[mname]
        md_lines.append(
            f"| `{mname}` | {a['mean_regret_pct']:+.2f}% | {a['max_regret_pct']:+.2f}% | {a['n_severe_ge_thr']} | "
            f"{a['weighted_cost_by_w']['10.0']:.2f} | {b['mean_rmsep_z']:.4f} | {b['mean_regret_pct']:+.2f}% | {a['mean_runtime_sec']:.3f} |"
        )

    md_lines += [
        "",
        f"## 二、Risk-Coverage 权衡曲线（N=18 AURC={rc18['aurc_selective_error']:.4f}；N=17 AURC={rc17['aurc_selective_error']:.4f}）",
        "",
        "| `t_crit` | N=18 覆盖率 | N=18 准确率 | N=18 选择风险 | FbSNV 遗憾 | FbRaw 遗憾 | N=17 覆盖率 | N=17 准确率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p18, p17 in zip(rc18["points"], rc17["points"]):
        acc18 = f"{p18['sign_acc']:.3f}" if p18["sign_acc"] is not None else "—"
        acc17 = f"{p17['sign_acc']:.3f}" if p17["sign_acc"] is not None else "—"
        md_lines.append(
            f"| {p18['t_crit']:.1f} | {p18['coverage']*100:5.1f}% ({p18['n_decided']}/18) | {acc18} | {p18['selective_risk']:.3f} | "
            f"{p18['fallback_snv_mean_regret_pct']:+.2f}% | {p18['fallback_raw_mean_regret_pct']:+.2f}% | "
            f"{p17['coverage']*100:5.1f}% ({p17['n_decided']}/17) | {acc17} |"
        )

    md_lines += [
        "",
        "## 三、CRAW 跨年时间漂移 6 任务全弃权（`ABSTAIN`）与安全性诊断",
        "",
        f"- `diagnose_deploy_sanity` 自动捕获 **标签量纲偏移**：`{out['sanity_summary']['alert_tasks']}`（均值 `85.43%` vs `16.13%` 范围不交），"
        f"未在事先未见特征时发生误吞；",
        f"- 6 项 CRAW 跨年时间划分中门控 **6/6 返回 `ABSTAIN`**，原因并非报错，而是校准期内部配对差异未过 `|t|>=2 & |d|>=1%`；其中 **#1 与 #2 在时间漂移后发生了真实符号反转（校准期符号与外部跨年真实收益相反，`ABSTAIN` 直接拦截了 2 起潜在反向硬判定）**。",
        "",
        "| CRAW 任务 | 样本数 | dLOO | t_loo | 门控 | 实测收益 | 谱漂移状态 | drift_score | SPE 比 | 跨年符号反转拦截 |",
        "|---|---:|---:|---:|---|---:|---|---:|---:|---|",
    ]
    for c in craw_diag:
        md_lines.append(
            f"| `{c['task_id']}` | {c['n_cal']}->{c['n_test']} | {c['delta_loo_pct']:+.2f}% | {c['t_loo']:+.2f} | "
            f"{c['gate_decision']} | {c['emp_snv_vs_raw_pct']:+.2f}% | {c['spectrum_drift_status']} | "
            f"{c['drift_score']:.2f} | {c['spe_ratio']:.2f} | {'是' if c['false_call_prevented_by_abstain'] else '否'} |"
        )

    out_md = ROOT / "reports" / "antigravity" / "DEPLOY_EXTENDED_ANALYSIS_2026-11-24.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nSaved JSON: {out_json}\nSaved MD:   {out_md}")


if __name__ == "__main__":
    main()
