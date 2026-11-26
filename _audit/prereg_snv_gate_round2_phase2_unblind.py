"""Round-2 Phase 2：验证 Phase 1 锁定校验和并开展全盲对照揭盲（18 项新任务 / 86 折）。

头对头对比：
  1. 新版双层判据 `diagnose_snv_decision`（主规则 |t_loo|>=2.0, |Δ_LOO|>=1.0% 与次级 |t_loo|>=1.5）；
  2. 原生无弃权 LOO 符号（delta_loo < 0）；
  3. 旧版解析理论判据 `estimate_physical_params_v2`（τ=0.5% 与 τ=0.0%）。
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
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._blind_round2_loaders import load_round2_tasks
from _audit._blind_theory_loaders import (
    file_sha256,
    file_sha256_lf,
    git_blob_sha256,
    git_blob_sha256_crlf,
    split_checksum,
)
from _audit.prereg_theory_blind_phase2_unblind import _binary_metrics, _contingency_table, _sanitize_json
from chemdl.transfer import fit_pls, fit_prep_fn, rmsep


def _eval_method(calls: list[str], pred_scores: np.ndarray, emp_vals: np.ndarray) -> dict:
    total_n = len(calls)
    mask = np.array([c != "ABSTAIN" for c in calls], dtype=bool)
    y_true = (emp_vals < 0.0).astype(int)
    y_pred = np.array([1 if c == "IMPROVE" else 0 for c in calls], dtype=int)
    m_all = _binary_metrics(y_true[mask], y_pred[mask], total_n)

    dec_mask = mask & (np.abs(emp_vals) >= 5.0)
    dec_base_n = int(np.sum(np.abs(emp_vals) >= 5.0))
    m_dec = _binary_metrics(y_true[dec_mask], y_pred[dec_mask], max(dec_base_n, 1))

    sp = stats.spearmanr(pred_scores, emp_vals)
    pe = stats.pearsonr(pred_scores, emp_vals)
    return {
        "all_signs": m_all,
        "decisive_ge5pct": m_dec,
        "spearman_rho": round(float(sp.statistic), 4) if math.isfinite(float(sp.statistic)) else None,
        "spearman_p": float(sp.pvalue) if math.isfinite(float(sp.pvalue)) else None,
        "pearson_r": round(float(pe.statistic), 4) if math.isfinite(float(pe.statistic)) else None,
        "pearson_p": float(pe.pvalue) if math.isfinite(float(pe.pvalue)) else None,
        "contingency": _contingency_table(calls, emp_vals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 2 blind validation Phase 2 / verification")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify frozen Git blob and archived Round-2 split/checksum consistency without overwriting res_prereg JSON/MD.",
    )
    args = parser.parse_args()

    t0 = time.time()
    prereg_path = ROOT / "_audit" / "prereg_snv_gate_round2_2026_11_24.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    prereg_sha = file_sha256(prereg_path)

    theory_path = ROOT / "chemdl" / "theory.py"
    locked_commit = str(prereg["git_head_frozen"])
    frozen_blob_lf = git_blob_sha256(ROOT, locked_commit, "chemdl/theory.py", lf_normalize=True)
    frozen_blob_crlf = git_blob_sha256_crlf(ROOT, locked_commit, "chemdl/theory.py")
    assert prereg["theory_py_sha256"] in (frozen_blob_lf, frozen_blob_crlf), (
        f"Frozen commit {locked_commit} blob mismatch: lf={frozen_blob_lf} crlf={frozen_blob_crlf} vs prereg={prereg['theory_py_sha256']}"
    )

    execution_git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    worktree_sha_lf = file_sha256_lf(theory_path)
    theory_sha_now = file_sha256(theory_path)

    tasks = load_round2_tasks(ROOT)
    assert len(tasks) == prereg["n_tasks"]
    pre_by_id = {r["task_id"]: r for r in prereg["tasks"]}

    if args.verify_only:
        res_path = ROOT / "_audit" / "res_prereg_snv_gate_round2_2026_11_24.json"
        res_archived = json.loads(res_path.read_text(encoding="utf-8"))
        res_by_id = {r["task_id"]: r for r in res_archived["tasks"]}
        for t in tasks:
            tid = t["task_id"]
            pre = pre_by_id[tid]
            res_t = res_by_id[tid]
            assert len(t["splits"]) == pre["n_folds"] == len(res_t["folds"])
            for fi, (tr_idx, te_idx) in enumerate(t["splits"]):
                chk = split_checksum(tr_idx, te_idx)
                assert chk == pre["folds"][fi]["split_sha256"] == res_t["folds"][fi]["split_sha256"]
        print(
            f"[verify-only PASS] Round 2 locked={locked_commit[:7]} (blob_lf={frozen_blob_lf[:12]}), "
            f"exec_head={execution_git_head[:7]} (worktree_lf={worktree_sha_lf[:12]}), "
            f"18 tasks / 86 splits verified identically against archive."
        )
        return

    assert worktree_sha_lf == frozen_blob_lf, (
        f"chemdl/theory.py (LF SHA-256) modified since locked commit {locked_commit[:7]}: "
        f"expected={frozen_blob_lf} vs worktree={worktree_sha_lf}. Use --verify-only to audit archived run."
    )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "chemdl/theory.py", "_audit/prereg_snv_gate_round2_2026_11_24.json"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert not dirty, f"Frozen Round-2 prereg files dirty: {dirty}"

    unblind_tasks = []
    f_calls_pri, f_calls_sec, f_calls_loo0, f_calls_leg05, f_calls_leg0 = [], [], [], [], []
    f_scores_loo, f_scores_leg, f_emps = [], [], []

    for ti, t in enumerate(tasks, 1):
        tid = t["task_id"]
        pre = pre_by_id[tid]
        assert len(t["splits"]) == pre["n_folds"]

        X, y = t["X"], t["y"]
        sq_raw, sq_snv = [], []
        fold_recs = []

        for fi, (tr_idx, te_idx) in enumerate(t["splits"]):
            pf = pre["folds"][fi]
            chk = split_checksum(tr_idx, te_idx)
            assert chk == pf["split_sha256"], f"{tid} fold {fi} split checksum mismatch!"

            y_tr, y_te = y[tr_idx], y[te_idx]
            mu = float(y_tr.mean())
            sd = max(float(y_tr.std()), 1e-9)
            y_tr_n = ((y_tr - mu) / sd)[:, None]
            y_te_n = ((y_te - mu) / sd)[:, None]

            kmax = min(10, max(2, len(tr_idx) // 2))
            inner = min(4, max(2, len(tr_idx) // 4))

            pr_raw, _ = fit_pls(X[tr_idx], y_tr_n, kmax=kmax, inner=inner, per_col=True, seed=fi)
            fn_snv = fit_prep_fn("snv", X[tr_idx])
            pr_snv, _ = fit_pls(fn_snv(X[tr_idx]), y_tr_n, kmax=kmax, inner=inner, per_col=True, seed=fi)

            p_raw = pr_raw(X[te_idx])
            p_snv = pr_snv(fn_snv(X[te_idx]))

            e_raw = float(rmsep(p_raw, y_te_n))
            e_snv = float(rmsep(p_snv, y_te_n))
            emp_pct = (e_snv / max(e_raw, 1e-12) - 1.0) * 100.0

            sq_raw.append((p_raw - y_te_n) ** 2)
            sq_snv.append((p_snv - y_te_n) ** 2)

            d_loo = float(pf["delta_loo_pct"])
            t_loo = float(pf["t_loo"])
            leg_v2 = float(pf["pred_v2_legacy_pct"])
            c_pri = str(pf["call_primary"])
            c_sec = str(pf["call_secondary_t1_5"])
            c_loo0 = "IMPROVE" if d_loo < 0.0 else "WORSEN"
            c_leg05 = str(pf["call_legacy_v2_0_5"])
            c_leg0 = "IMPROVE" if leg_v2 < 0.0 else "WORSEN"

            f_calls_pri.append(c_pri)
            f_calls_sec.append(c_sec)
            f_calls_loo0.append(c_loo0)
            f_calls_leg05.append(c_leg05)
            f_calls_leg0.append(c_leg0)
            f_scores_loo.append(d_loo)
            f_scores_leg.append(leg_v2)
            f_emps.append(emp_pct)

            fold_recs.append(
                {
                    "fold": int(fi),
                    "split_sha256": chk,
                    "rmsep_raw_z": round(e_raw, 4),
                    "rmsep_snv_z": round(e_snv, 4),
                    "emp_snv_vs_raw_pct": round(emp_pct, 2),
                    "delta_loo_pct": d_loo,
                    "t_loo": t_loo,
                    "call_primary": c_pri,
                    "pred_v2_legacy_pct": leg_v2,
                    "call_legacy_v2_0_5": c_leg05,
                }
            )

        pool_raw = float(np.sqrt(np.mean(np.concatenate(sq_raw, axis=0))))
        pool_snv = float(np.sqrt(np.mean(np.concatenate(sq_snv, axis=0))))
        emp_pooled_pct = (pool_snv / max(pool_raw, 1e-12) - 1.0) * 100.0
        emp_f_vals = [f["emp_snv_vs_raw_pct"] for f in fold_recs]
        emp_f_mean = float(np.mean(emp_f_vals))
        emp_f_sd = float(np.std(emp_f_vals, ddof=1)) if len(emp_f_vals) > 1 else 0.0

        rec = {
            "task_id": tid,
            "domain": pre["domain"],
            "n": pre["n"],
            "L": pre["L"],
            "mode": pre["mode"],
            "delta_loo_mean_pct": float(pre["delta_loo_mean_pct"]),
            "t_loo_mean": float(pre["t_loo_mean"]),
            "call_primary": str(pre["call_primary"]),
            "call_secondary_t1_5": str(pre["call_secondary_t1_5"]),
            "pred_v2_legacy_mean_pct": float(pre["pred_v2_legacy_mean_pct"]),
            "call_legacy_v2_0_5": str(pre["call_legacy_v2_0_5"]),
            "r2_s_shape_mean": float(pre["r2_s_shape_mean"]),
            "sigma_a_true_mean": float(pre["sigma_a_true_mean"]),
            "theory_assumption_status": str(pre["theory_assumption_status"]),
            "rmsep_raw_pooled_z": round(pool_raw, 4),
            "rmsep_snv_pooled_z": round(pool_snv, 4),
            "emp_pooled_pct": round(emp_pooled_pct, 2),
            "emp_fold_mean_pct": round(emp_f_mean, 2),
            "emp_fold_sd_pct": round(emp_f_sd, 2),
            "folds": fold_recs,
        }
        unblind_tasks.append(rec)
        print(
            f"[{ti:02d}/{len(tasks)}] {tid:31s} | NEW={rec['call_primary']:7s} (dLOO={rec['delta_loo_mean_pct']:+6.2f}% t={rec['t_loo_mean']:+5.2f}) | "
            f"OLD={rec['call_legacy_v2_0_5']:7s} ({rec['pred_v2_legacy_mean_pct']:+6.2f}%) | emp={rec['emp_pooled_pct']:+7.2f}%",
            flush=True,
        )

    t_emps = np.array([r["emp_pooled_pct"] for r in unblind_tasks], dtype=np.float64)
    t_scores_loo = np.array([r["delta_loo_mean_pct"] for r in unblind_tasks], dtype=np.float64)
    t_scores_leg = np.array([r["pred_v2_legacy_mean_pct"] for r in unblind_tasks], dtype=np.float64)

    t_calls_pri = [r["call_primary"] for r in unblind_tasks]
    t_calls_sec = [r["call_secondary_t1_5"] for r in unblind_tasks]
    t_calls_loo0 = ["IMPROVE" if r["delta_loo_mean_pct"] < 0.0 else "WORSEN" for r in unblind_tasks]
    t_calls_leg05 = [r["call_legacy_v2_0_5"] for r in unblind_tasks]
    t_calls_leg0 = ["IMPROVE" if r["pred_v2_legacy_mean_pct"] < 0.0 else "WORSEN" for r in unblind_tasks]

    task_eval = {
        "new_paired_loo_primary": _eval_method(t_calls_pri, t_scores_loo, t_emps),
        "new_paired_loo_t1_5": _eval_method(t_calls_sec, t_scores_loo, t_emps),
        "new_loo_sign_zero": _eval_method(t_calls_loo0, t_scores_loo, t_emps),
        "legacy_v2_tau_0_5": _eval_method(t_calls_leg05, t_scores_leg, t_emps),
        "legacy_v2_tau_0_0": _eval_method(t_calls_leg0, t_scores_leg, t_emps),
    }

    f_emps_a = np.array(f_emps, dtype=np.float64)
    fold_eval = {
        "new_paired_loo_primary": _eval_method(f_calls_pri, np.array(f_scores_loo), f_emps_a),
        "new_paired_loo_t1_5": _eval_method(f_calls_sec, np.array(f_scores_loo), f_emps_a),
        "new_loo_sign_zero": _eval_method(f_calls_loo0, np.array(f_scores_loo), f_emps_a),
        "legacy_v2_tau_0_5": _eval_method(f_calls_leg05, np.array(f_scores_leg), f_emps_a),
        "legacy_v2_tau_0_0": _eval_method(f_calls_leg0, np.array(f_scores_leg), f_emps_a),
    }

    degradation_audit = [
        {
            "task_id": r["task_id"],
            "emp_pooled_pct": r["emp_pooled_pct"],
            "new_call": r["call_primary"],
            "new_dloo": r["delta_loo_mean_pct"],
            "new_t": r["t_loo_mean"],
            "legacy_call": r["call_legacy_v2_0_5"],
            "legacy_pred": r["pred_v2_legacy_mean_pct"],
            "r2_s_shape": r["r2_s_shape_mean"],
        }
        for r in unblind_tasks
        if r["emp_pooled_pct"] >= 2.0
    ]

    ts_local = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    out = _sanitize_json(
        {
            "phase": "ROUND2_PHASE_2_UNBLINDED_VERIFIED",
            "timestamp": ts_local,
            "prereg_commit_locked": locked_commit,
            "execution_git_head": execution_git_head,
            "prereg_json_sha256": prereg_sha,
            "theory_py_git_blob_sha256": frozen_blob_lf,
            "theory_py_worktree_sha256_lf": worktree_sha_lf,
            "theory_py_sha256_verified": theory_sha_now,
            "n_tasks": len(unblind_tasks),
            "n_folds": len(f_emps_a),
            "elapsed_sec": round(time.time() - t0, 2),
            "summary": {
                "task_level": task_eval,
                "fold_level": fold_eval,
                "degradation_ge2pct_audit": degradation_audit,
            },
            "tasks": unblind_tasks,
        }
    )

    out_json = ROOT / "_audit" / "res_prereg_snv_gate_round2_2026_11_24.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)

    # 撰写严谨报告
    lines = [
        "# 第二轮独立盲测揭盲头对头评估（Round 2 · Phase 2）",
        "",
        "## 一、冻结与防偷窥链校验",
        "",
        f"- **Phase 1 锁定提交**：`{locked_commit}`（执行时 HEAD：`{execution_git_head}`）",
        f"- **理论代码 SHA-256**：[`chemdl/theory.py`](../../chemdl/theory.py) = `{theory_sha_now}`（Git LF Blob：`{frozen_blob_lf}`）",
        f"- **预注册 JSON SHA-256**：[`_audit/prereg_snv_gate_round2_2026_11_24.json`](../../_audit/prereg_snv_gate_round2_2026_11_24.json) = `{prereg_sha}`",
        f"- **预注册文本对照**：[`PREREG_SNV_GATE_ROUND2_2026-11-24.md`](./PREREG_SNV_GATE_ROUND2_2026-11-24.md)",
        f"- **切分校验**：全部 18 项任务 / {len(f_emps_a)} 折 `split_sha256` 100% 一致，且与训练集、Round 1 零重叠。",
        "",
        "## 二、头对头总指标对照（新版 Paired-LOO 门控 vs 旧版 `v2`）",
        "",
        "### 1. 任务级（N = 18）",
        "",
        "| 方法 | 覆盖率 Cov | 准确率 Acc (命中/总数) | TPR / TNR | Balanced Acc | MCC | Spearman ρ (pred vs emp) |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for label, key in (
        ("**新版主门控 (`|t|≥2.0, |Δ|≥1%`)**", "new_paired_loo_primary"),
        ("新版次级 (`|t|≥1.5, |Δ|≥1%`)", "new_paired_loo_t1_5"),
        ("新版 LOO 零弃权 (`Δ_LOO < 0`)", "new_loo_sign_zero"),
        ("旧版 `v2` (`τ = 0.5%`)", "legacy_v2_tau_0_5"),
        ("旧版 `v2` 零弃权 (`τ = 0.0%`)", "legacy_v2_tau_0_0"),
    ):
        m = task_eval[key]["all_signs"]
        rho_s = f"{task_eval[key]['spearman_rho']:+.4f} (p={task_eval[key]['spearman_p']:.3g})"
        lines.append(
            f"| {label} | {m['coverage']:.1%} ({m['n']}/18) | {m['acc']} ({m['tp']+m['tn']}/{m['n']}) "
            f"| {m['tpr']} / {m['tnr']} | {m['balanced_acc']} | {m['mcc']} | {rho_s} |"
        )

    lines.extend([
        "",
        "### 2. 逐折级（N = 86）",
        "",
        "| 方法 | 覆盖率 Cov | 准确率 Acc (命中/总数) | TPR / TNR | Balanced Acc | MCC | Spearman ρ (pred vs emp) |",
        "|---|---|---|---:|---:|---:|---|",
    ])
    for label, key in (
        ("**新版主门控 (`|t|≥2.0, |Δ|≥1%`)**", "new_paired_loo_primary"),
        ("新版次级 (`|t|≥1.5, |Δ|≥1%`)", "new_paired_loo_t1_5"),
        ("新版 LOO 零弃权 (`Δ_LOO < 0`)", "new_loo_sign_zero"),
        ("旧版 `v2` (`τ = 0.5%`)", "legacy_v2_tau_0_5"),
        ("旧版 `v2` 零弃权 (`τ = 0.0%`)", "legacy_v2_tau_0_0"),
    ):
        m = fold_eval[key]["all_signs"]
        rho_s = f"{fold_eval[key]['spearman_rho']:+.4f} (p={fold_eval[key]['spearman_p']:.3g})"
        lines.append(
            f"| {label} | {m['coverage']:.1%} ({m['n']}/{len(f_emps_a)}) | {m['acc']} ({m['tp']+m['tn']}/{m['n']}) "
            f"| {m['tpr']} / {m['tnr']} | {m['balanced_acc']} | {m['mcc']} | {rho_s} |"
        )

    lines.extend([
        "",
        "## 三、全 18 项第二轮独立任务揭盲逐一核对表",
        "",
        "| # | Task ID | 新版判断 (`dLOO / t`) | 旧版 `v2` (`pred`) | Raw_z → SNV_z | **实测 ΔRMSEP_pool** | 折均值±SD | R²_shape |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ])
    for idx, r in enumerate(unblind_tasks, 1):
        lines.append(
            f"| {idx} | `{r['task_id']}` | **{r['call_primary']}** (`{r['delta_loo_mean_pct']:+.2f}% / {r['t_loo_mean']:+.2f}`) "
            f"| {r['call_legacy_v2_0_5']} (`{r['pred_v2_legacy_mean_pct']:+.2f}%`) "
            f"| {r['rmsep_raw_pooled_z']:.4f} → {r['rmsep_snv_pooled_z']:.4f} | **{r['emp_pooled_pct']:+.2f}%** "
            f"| {r['emp_fold_mean_pct']:+.2f}±{r['emp_fold_sd_pct']:.2f}% | {r['r2_s_shape_mean']:.3f} |"
        )

    lines.extend([
        "",
        "## 四、真实恶化任务（实测 ≥ +2.0%）防误判核查",
        "",
        "| Task ID | 实测 ΔRMSEP | 新版判断 | 新版 dLOO / t | 旧版 `v2` 判断 | 旧版 `v2` pred |",
        "|---|---:|---|---:|---|---:|",
    ])
    for d in degradation_audit:
        lines.append(
            f"| `{d['task_id']}` | **{d['emp_pooled_pct']:+.2f}%** | **{d['new_call']}** "
            f"| {d['new_dloo']:+.2f}% / {d['new_t']:+.2f} | {d['legacy_call']} | {d['legacy_pred']:+.2f}% |"
        )
    lines.append("")

    out_md = ROOT / "reports" / "antigravity" / "UNBLIND_SNV_GATE_ROUND2_2026-11-24.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRound-2 Phase 2 complete -> {out_json} & {out_md} ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
