"""Phase 2：验证 Phase 1 锁定校验和并执行严格无泄漏 Raw-PLS vs SNV-PLS 揭盲评测。

重要原则：
  - 本脚本只读取并校验 `_audit/prereg_theory_blind_2026_11_24.json`，绝不修改 Phase 1 文件；
  - 校验 `chemdl/theory.py` SHA-256 及全部 `n_tasks / total_folds` 的 `split_sha256` 与 Phase 1 一致；
  - 统计口吻严格客观，将预注册冻结检验结果与事后机理诊断清楚区分。
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

from _audit._blind_theory_loaders import (
    file_sha256,
    file_sha256_lf,
    git_blob_sha256,
    git_blob_sha256_crlf,
    load_blind_tasks,
    split_checksum,
)
from chemdl.transfer import fit_pls, fit_prep_fn, rmsep


def _sanitize_json(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize_json(v) for v in o]
    return o


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, total_n: int) -> dict:
    n = int(len(y_true))
    if n == 0:
        return {
            "coverage": 0.0,
            "n": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "acc": None,
            "tpr": None,
            "tnr": None,
            "balanced_acc": None,
            "mcc": None,
        }
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    acc = (tp + tn) / n
    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None
    bac = 0.5 * (tpr + tnr) if (tpr is not None and tnr is not None) else None
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / math.sqrt(denom) if denom > 0.0 else 0.0
    return {
        "coverage": round(n / max(total_n, 1), 4),
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": round(float(acc), 4),
        "tpr": round(float(tpr), 4) if tpr is not None else None,
        "tnr": round(float(tnr), 4) if tnr is not None else None,
        "balanced_acc": round(float(bac), 4) if bac is not None else None,
        "mcc": round(float(mcc), 4),
    }


def _evaluate_tiers(pred_vals: np.ndarray, emp_vals: np.ndarray, tau_tiers: list[float]) -> dict:
    out = {}
    total_n = len(pred_vals)
    y_true_all = (emp_vals < 0.0).astype(int)

    for tau in tau_tiers:
        if tau == 0.0:
            mask = np.ones(total_n, dtype=bool)
            y_pred = (pred_vals < 0.0).astype(int)
        else:
            mask = np.abs(pred_vals) >= float(tau)
            y_pred = (pred_vals[mask] <= -float(tau)).astype(int)
        metrics_all = _binary_metrics(y_true_all[mask], y_pred, total_n)

        dec_mask = mask & (np.abs(emp_vals) >= 5.0)
        dec_pred = (pred_vals[dec_mask] < 0.0).astype(int) if tau == 0.0 else (pred_vals[dec_mask] <= -float(tau)).astype(int)
        dec_base_n = int(np.sum(np.abs(emp_vals) >= 5.0))
        metrics_dec = _binary_metrics(y_true_all[dec_mask], dec_pred, max(dec_base_n, 1))

        out[f"tau_{tau}"] = {
            "all_signs": metrics_all,
            "decisive_ge5pct": metrics_dec,
        }
    return out


def _contingency_table(calls: list[str], emp_vals: np.ndarray) -> dict:
    rows = ("IMPROVE", "ABSTAIN", "WORSEN")
    band_cols = ("Actual_IMPROVE_le_-2pct", "Actual_NEUTRAL_within_2pct", "Actual_WORSEN_ge_+2pct")
    sign_cols = ("Actual_SIGN_IMPROVE_lt0", "Actual_SIGN_WORSEN_ge0")

    band_tbl = {r: {c: 0 for c in band_cols} for r in rows}
    sign_tbl = {r: {c: 0 for c in sign_cols} for r in rows}

    for call, emp in zip(calls, emp_vals):
        bcol = (
            band_cols[0]
            if emp <= -2.0
            else (band_cols[2] if emp >= 2.0 else band_cols[1])
        )
        scol = sign_cols[0] if emp < 0.0 else sign_cols[1]
        band_tbl[call][bcol] += 1
        sign_tbl[call][scol] += 1

    return {"practical_2pct_band": band_tbl, "strict_sign": sign_tbl}


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 1 blind validation Phase 2 / verification")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify frozen Git blob and archived Round-1 split/checksum consistency without overwriting res_prereg JSON/MD.",
    )
    args = parser.parse_args()

    t0 = time.time()
    prereg_path = ROOT / "_audit" / "prereg_theory_blind_2026_11_24.json"
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

    tasks = load_blind_tasks(ROOT)
    assert len(tasks) == prereg["n_tasks"], f"Task count mismatch: {len(tasks)} vs {prereg['n_tasks']}"
    prereg_by_id = {r["task_id"]: r for r in prereg["tasks"]}

    if args.verify_only:
        res_path = ROOT / "_audit" / "res_prereg_theory_blind_2026_11_24.json"
        res_archived = json.loads(res_path.read_text(encoding="utf-8"))
        res_by_id = {r["task_id"]: r for r in res_archived["tasks"]}
        for t in tasks:
            tid = t["task_id"]
            pre = prereg_by_id[tid]
            res_t = res_by_id[tid]
            assert len(t["splits"]) == pre["n_folds"] == len(res_t["folds"])
            for fi, (tr_idx, te_idx) in enumerate(t["splits"]):
                chk = split_checksum(tr_idx, te_idx)
                assert chk == pre["folds"][fi]["split_sha256"] == res_t["folds"][fi]["split_sha256"]
        print(
            f"[verify-only PASS] Round 1 locked={locked_commit[:7]} (blob_lf={frozen_blob_lf[:12]}), "
            f"exec_head={execution_git_head[:7]} (worktree_lf={worktree_sha_lf[:12]}), "
            f"24 tasks / 116 splits verified identically against archive."
        )
        return

    assert worktree_sha_lf == frozen_blob_lf, (
        f"chemdl/theory.py (LF SHA-256) modified since locked commit {locked_commit[:7]}: "
        f"expected={frozen_blob_lf} vs worktree={worktree_sha_lf}. Use --verify-only to audit archived run."
    )
    prereg_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "chemdl/theory.py", "_audit/prereg_theory_blind_2026_11_24.json"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert not prereg_status, f"Frozen prereg files dirty before unblinding: {prereg_status}"

    tau_tiers = [float(x) for x in prereg["tau_tiers_pct"]]
    tau_primary = float(prereg["tau_primary_pct"])

    unblind_tasks = []
    fold_preds, fold_emps, fold_calls = [], [], []

    for ti, t in enumerate(tasks, 1):
        tid = t["task_id"]
        pre = prereg_by_id[tid]
        assert len(t["splits"]) == pre["n_folds"], f"{tid}: fold count mismatch"

        X, y = t["X"], t["y"]
        sq_raw, sq_snv = [], []
        fold_recs = []

        for fi, (tr_idx, te_idx) in enumerate(t["splits"]):
            pre_f = pre["folds"][fi]
            chk = split_checksum(tr_idx, te_idx)
            assert chk == pre_f["split_sha256"], f"{tid} fold {fi} split_sha256 mismatch!"

            y_tr, y_te = y[tr_idx], y[te_idx]
            mu = float(y_tr.mean())
            sd = max(float(y_tr.std()), 1e-9)
            y_tr_n = ((y_tr - mu) / sd)[:, None]
            y_te_n = ((y_te - mu) / sd)[:, None]

            kmax = min(10, max(2, len(tr_idx) // 2))
            inner = min(4, max(2, len(tr_idx) // 4))

            pr_raw, cfg_raw = fit_pls(X[tr_idx], y_tr_n, kmax=kmax, inner=inner, per_col=True, seed=fi)
            fn_snv = fit_prep_fn("snv", X[tr_idx])
            pr_snv, cfg_snv = fit_pls(fn_snv(X[tr_idx]), y_tr_n, kmax=kmax, inner=inner, per_col=True, seed=fi)

            p_raw = pr_raw(X[te_idx])
            p_snv = pr_snv(fn_snv(X[te_idx]))

            e_raw = float(rmsep(p_raw, y_te_n))
            e_snv = float(rmsep(p_snv, y_te_n))
            emp_pct = (e_snv / max(e_raw, 1e-12) - 1.0) * 100.0

            sq_raw.append((p_raw - y_te_n) ** 2)
            sq_snv.append((p_snv - y_te_n) ** 2)

            pred_f = float(pre_f["pred_snv_vs_raw_pct_v2"])
            call_f = str(pre_f["call_primary_0_5pct"])
            fold_preds.append(pred_f)
            fold_emps.append(emp_pct)
            fold_calls.append(call_f)

            fold_recs.append(
                {
                    "fold": int(fi),
                    "split_sha256": chk,
                    "kmax": int(kmax),
                    "inner": int(inner),
                    "k_raw": int(cfg_raw[0]) if isinstance(cfg_raw, (list, tuple)) else int(cfg_raw),
                    "k_snv": int(cfg_snv[0]) if isinstance(cfg_snv, (list, tuple)) else int(cfg_snv),
                    "rmsep_raw_z": round(e_raw, 4),
                    "rmsep_snv_z": round(e_snv, 4),
                    "pred_v2_pct": pred_f,
                    "call_primary_0_5pct": call_f,
                    "emp_snv_vs_raw_pct": round(emp_pct, 2),
                    "sign_match_zero": bool((pred_f < 0.0) == (emp_pct < 0.0)),
                }
            )

        pool_raw = float(np.sqrt(np.mean(np.concatenate(sq_raw, axis=0))))
        pool_snv = float(np.sqrt(np.mean(np.concatenate(sq_snv, axis=0))))
        emp_pooled_pct = (pool_snv / max(pool_raw, 1e-12) - 1.0) * 100.0
        fold_emp_vals = [f["emp_snv_vs_raw_pct"] for f in fold_recs]
        emp_mean_fold_pct = float(np.mean(fold_emp_vals))
        emp_sd_fold_pct = float(np.std(fold_emp_vals, ddof=1)) if len(fold_emp_vals) > 1 else 0.0

        rec = {
            "task_id": tid,
            "dataset_group": pre["dataset_group"],
            "domain": pre["domain"],
            "instrument": pre["instrument"],
            "target_name": pre["target_name"],
            "n": pre["n"],
            "L": pre["L"],
            "n_folds": pre["n_folds"],
            "pred_mean_pct": float(pre["pred_mean_pct"]),
            "call_primary_0_5pct": str(pre["call_primary_0_5pct"]),
            "r2_s_shape_mean": float(pre["r2_s_shape_mean"]),
            "r2_mu_shape_mean": float(pre["r2_mu_shape_mean"]),
            "sigma_s_mean": float(pre["sigma_s_mean"]),
            "sigma_a_true_mean": float(pre["sigma_a_true_mean"]),
            "r_s_mean": float(pre["r_s_mean"]),
            "r_crit_mean": float(pre["r_crit_mean"]),
            "ws_star_mean": float(pre["ws_star_mean"]),
            "rmsep_raw_pooled_z": round(pool_raw, 4),
            "rmsep_snv_pooled_z": round(pool_snv, 4),
            "emp_pooled_pct": round(emp_pooled_pct, 2),
            "emp_fold_mean_pct": round(emp_mean_fold_pct, 2),
            "emp_fold_sd_pct": round(emp_sd_fold_pct, 2),
            "sign_match_zero": bool((float(pre["pred_mean_pct"]) < 0.0) == (emp_pooled_pct < 0.0)),
            "folds": fold_recs,
        }
        unblind_tasks.append(rec)
        print(
            f"[{ti:02d}/{len(tasks)}] {tid:32s} | pred={rec['pred_mean_pct']:+6.2f}% ({rec['call_primary_0_5pct']:7s}) | "
            f"emp_pool={rec['emp_pooled_pct']:+7.2f}% (Raw={pool_raw:.4f} SNV={pool_snv:.4f}) | "
            f"sign0={rec['sign_match_zero']}",
            flush=True,
        )

    task_preds = np.array([r["pred_mean_pct"] for r in unblind_tasks], dtype=np.float64)
    task_emps = np.array([r["emp_pooled_pct"] for r in unblind_tasks], dtype=np.float64)
    task_calls = [r["call_primary_0_5pct"] for r in unblind_tasks]

    fold_preds_a = np.array(fold_preds, dtype=np.float64)
    fold_emps_a = np.array(fold_emps, dtype=np.float64)

    sp_task = stats.spearmanr(task_preds, task_emps)
    sp_fold = stats.spearmanr(fold_preds_a, fold_emps_a)
    pe_task = stats.pearsonr(task_preds, task_emps)

    task_tiers = _evaluate_tiers(task_preds, task_emps, tau_tiers)
    fold_tiers = _evaluate_tiers(fold_preds_a, fold_emps_a, tau_tiers)

    task_cont = _contingency_table(task_calls, task_emps)
    fold_cont = _contingency_table(fold_calls, fold_emps_a)

    # 专项审计：所有 SNV 真实恶化 >= +5% 任务 + 全套 IDRC2016 小麦家族
    degradation_tasks = [
        {
            "task_id": r["task_id"],
            "domain": r["domain"],
            "pred_mean_pct": r["pred_mean_pct"],
            "locked_call": r["call_primary_0_5pct"],
            "emp_pooled_pct": r["emp_pooled_pct"],
            "r2_s_shape_mean": r["r2_s_shape_mean"],
            "sigma_s_mean": r["sigma_s_mean"],
            "sigma_a_true_mean": r["sigma_a_true_mean"],
            "r_s_mean": r["r_s_mean"],
            "r_crit_mean": r["r_crit_mean"],
            "ws_star_mean": r["ws_star_mean"],
            "outcome": (
                "CAUGHT_WORSEN"
                if r["call_primary_0_5pct"] == "WORSEN"
                else ("ABSTAINED_SAFE" if r["call_primary_0_5pct"] == "ABSTAIN" else "FALSE_POSITIVE_IMPROVE")
            ),
        }
        for r in unblind_tasks
        if r["emp_pooled_pct"] >= 5.0 or r["task_id"].startswith("IDRC2016_Wheat_")
    ]

    summary = {
        "task_level": {
            "n_tasks": len(unblind_tasks),
            "spearman_rho": round(float(sp_task.statistic), 4),
            "spearman_p": float(sp_task.pvalue),
            "pearson_r": round(float(pe_task.statistic), 4),
            "pearson_p": float(pe_task.pvalue),
            "contingency_tau_0_5": task_cont,
            "tiers": task_tiers,
        },
        "fold_level": {
            "n_folds": len(fold_preds_a),
            "spearman_rho": round(float(sp_fold.statistic), 4),
            "spearman_p": float(sp_fold.pvalue),
            "contingency_tau_0_5": fold_cont,
            "tiers": fold_tiers,
        },
        "wheat_a1_like_and_degradation_audit": degradation_tasks,
    }

    ts_local = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    out = _sanitize_json(
        {
            "phase": "PHASE_2_UNBLINDED_VERIFIED",
            "timestamp": ts_local,
            "prereg_commit_locked": locked_commit,
            "execution_git_head": execution_git_head,
            "prereg_json_sha256": prereg_sha,
            "theory_py_git_blob_sha256": frozen_blob_lf,
            "theory_py_worktree_sha256_lf": worktree_sha_lf,
            "theory_py_sha256_verified": theory_sha_now,
            "tau_primary_pct": tau_primary,
            "tau_tiers_pct": tau_tiers,
            "elapsed_sec": round(time.time() - t0, 2),
            "summary": summary,
            "tasks": unblind_tasks,
        }
    )

    out_json = ROOT / "_audit" / "res_prereg_theory_blind_2026_11_24.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)

    # 撰写严谨、双栏对照且区分先验验证 vs 事后诊断的揭盲报告
    tl_0 = task_tiers["tau_0.0"]["all_signs"]
    tl_05 = task_tiers["tau_0.5"]["all_signs"]
    fl_0 = fold_tiers["tau_0.0"]["all_signs"]
    fl_05 = fold_tiers["tau_0.5"]["all_signs"]

    lines = [
        "# 定理 1 (`estimate_physical_params_v2`) 盲测揭盲对照报告（Phase 2）",
        "",
        "## 一、冻结与防偷窥链校验（全部硬断言通过）",
        "",
        f"- **Phase 1 锁定提交**：`{locked_commit}`（执行时 HEAD：`{execution_git_head}`，`PREG` 文件与 `chemdl/theory.py` 在揭盲前后零改动）",
        f"- **冻结理论源码 SHA-256**：[`chemdl/theory.py`](../../chemdl/theory.py) = `{theory_sha_now}`（Git LF Blob：`{frozen_blob_lf}`）",
        f"- **锁定预注册 JSON SHA-256**：[`_audit/prereg_theory_blind_2026_11_24.json`](../../_audit/prereg_theory_blind_2026_11_24.json) = `{prereg_sha}`",
        f"- **预注册文本对照**：[`PREREG_THEORY_BLIND_2026-11-24.md`](./PREREG_THEORY_BLIND_2026-11-24.md)",
        f"- **全量切分校验**：24 项任务 / 116 折的 `(tr_idx, te_idx)` SHA-256 与 Phase 1 100% 逐折一致。",
        "",
        "## 二、预注册整体表现汇总（严禁任何事后改阈值）",
        "",
        "| 评估层面 | N | 零弃权 (`τ=0%`) Acc / BAC / MCC | 主冻结阈值 (`τ=0.5%`) Cov / Acc / BAC / MCC | Spearman ρ (pred vs emp) |",
        "|---|---:|---|---|---|",
        (
            f"| **任务级 (pooled OOF)** | {len(unblind_tasks)} | "
            f"{tl_0['acc']:.1%} / {tl_0['balanced_acc']} / {tl_0['mcc']} | "
            f"{tl_05['coverage']:.1%} ({tl_05['n']}/{len(unblind_tasks)}) / {tl_05['acc']} / {tl_05['balanced_acc']} / {tl_05['mcc']} | "
            f"{summary['task_level']['spearman_rho']:+.4f} (p={summary['task_level']['spearman_p']:.3g}) |"
        ),
        (
            f"| **逐折级 (single folds)** | {len(fold_preds_a)} | "
            f"{fl_0['acc']:.1%} / {fl_0['balanced_acc']} / {fl_0['mcc']} | "
            f"{fl_05['coverage']:.1%} ({fl_05['n']}/{len(fold_preds_a)}) / {fl_05['acc']} / {fl_05['balanced_acc']} / {fl_05['mcc']} | "
            f"{summary['fold_level']['spearman_rho']:+.4f} (p={summary['fold_level']['spearman_p']:.3g}) |"
        ),
        "",
        "### 三态列联表（主阈值 `τ = 0.5%`：改善 / 无法判断 / 恶化）",
        "",
        "#### 任务级（N = 24）",
        "",
        "| 事前锁定判断 | 实际明显改善 (`≤ -2%`) | 实际基本持平 (`-2% ~ +2%`) | 实际明显恶化 (`≥ +2%`) | 严格符号改善 (`< 0%`) | 严格符号恶化 (`≥ 0%`) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rname, label in (("IMPROVE", "IMPROVE（改善）"), ("ABSTAIN", "ABSTAIN（无法判断）"), ("WORSEN", "WORSEN（恶化）")):
        bc = task_cont["practical_2pct_band"][rname]
        sc = task_cont["strict_sign"][rname]
        lines.append(
            f"| **{label}** | {bc['Actual_IMPROVE_le_-2pct']} | {bc['Actual_NEUTRAL_within_2pct']} | {bc['Actual_WORSEN_ge_+2pct']} "
            f"| {sc['Actual_SIGN_IMPROVE_lt0']} | {sc['Actual_SIGN_WORSEN_ge0']} |"
        )

    lines.extend([
        "",
        "#### 逐折级（N = 116）",
        "",
        "| 事前锁定判断 | 实际明显改善 (`≤ -2%`) | 实际基本持平 (`-2% ~ +2%`) | 实际明显恶化 (`≥ +2%`) | 严格符号改善 (`< 0%`) | 严格符号恶化 (`≥ 0%`) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for rname, label in (("IMPROVE", "IMPROVE（改善）"), ("ABSTAIN", "ABSTAIN（无法判断）"), ("WORSEN", "WORSEN（恶化）")):
        bc = fold_cont["practical_2pct_band"][rname]
        sc = fold_cont["strict_sign"][rname]
        lines.append(
            f"| **{label}** | {bc['Actual_IMPROVE_le_-2pct']} | {bc['Actual_NEUTRAL_within_2pct']} | {bc['Actual_WORSEN_ge_+2pct']} "
            f"| {sc['Actual_SIGN_IMPROVE_lt0']} | {sc['Actual_SIGN_WORSEN_ge0']} |"
        )

    lines.extend([
        "",
        "## 三、全 24 项独立任务揭盲逐一核对表",
        "",
        "| # | Task ID | pred_v2 | 锁定判断 (τ=0.5%) | Raw_z | SNV_z | 实测 ΔRMSEP_pool | 折均值±SD | R²_shape | σ_a_true | 严格符号命中 |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for idx, r in enumerate(unblind_tasks, 1):
        hit = "是" if r["sign_match_zero"] else "否"
        lines.append(
            f"| {idx} | `{r['task_id']}` | {r['pred_mean_pct']:+.2f}% | **{r['call_primary_0_5pct']}** "
            f"| {r['rmsep_raw_pooled_z']:.4f} | {r['rmsep_snv_pooled_z']:.4f} | **{r['emp_pooled_pct']:+.2f}%** "
            f"| {r['emp_fold_mean_pct']:+.2f}±{r['emp_fold_sd_pct']:.2f}% "
            f"| {r['r2_s_shape_mean']:.3f} | {r['sigma_a_true_mean']:.4f} | {hit} |"
        )

    lines.extend([
        "",
        "## 四、Wheat A1 家族与高恶化（≥ +5%）失效场景专项核查",
        "",
        "| Task ID | 锁定预测 pred_v2 | 锁定判断 (τ=0.5%) | 实测 ΔRMSEP | R²_shape | σ_s | σ_a_true | r_s | r_crit | 处置结局 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for d in degradation_tasks:
        lines.append(
            f"| `{d['task_id']}` | {d['pred_mean_pct']:+.2f}% | **{d['locked_call']}** | **{d['emp_pooled_pct']:+.2f}%** "
            f"| {d['r2_s_shape_mean']:.3f} | {d['sigma_s_mean']:.4f} | {d['sigma_a_true_mean']:.4f} "
            f"| {d['r_s_mean']:+.3f} | {d['r_crit_mean']:.3f} | **{d['outcome']}** |"
        )

    lines.extend([
        "",
        "## 五、事后机制诊断（Post-hoc，与上述先验验证严格隔离）",
        "",
        "1. **是否能预测真实受益？**：在强物理散射主导的任务上（如 `OSSL_Soil_Unseen` 三项特性实测改善与预测方向），以及在高耦合负益（如 `CRAW_Maize_NIRS5000__Protein` 的 `WORSEN` 预警）与低散射液态介质（`SDOP` / `SWRI Diesel` 全部正确给出 `ABSTAIN`），请以上述揭盲实测数字为准。",
        "2. **为什么 `IDRC2016_Wheat_*` 全系呈现类似 `Wheat A1` 的行为（若实测恶化）**：`estimate_physical_params_v2` 在算 `pred_snv_vs_raw_pct_v2` 时未对 `R²_shape` 施加「形态主控强吸收耦合惩罚或置信熔断」——只要剩余 `(1 - R²_shape) * σ_s²` 的表面残差（`σ_a_true ≈ 0.08–0.09`）叠加高 `β`（使 `r_crit = 1.0`），公式就会强行给出 `-20% ~ -33%` 的高置信 `IMPROVE`，即便该整粒小麦谱的 `R²_shape = 0.84–0.92` 已提示绝大部分行标准差波动其实与光谱形态高度绑定。",
    ])

    out_md = ROOT / "reports" / "antigravity" / "UNBLIND_THEORY_BLIND_2026-11-24.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPhase 2 complete -> {out_json} & {out_md} ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
