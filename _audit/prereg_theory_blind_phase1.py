"""Phase 1：冻结判据与阈值（C-141 `estimate_physical_params_v2` + τ=0.5%）预注册脚本。

硬规则：
  - 本脚本禁止导入 PLS / SNV 建模函数或访问测试折标签 y_te；
  - 仅允许用每折训练集 (X_tr, y_tr) 计算物理判据 v2 并写出预注册预测表；
  - 必须先跑通并本地提交本阶段产物后，再编写或运行 Phase 2 揭盲脚本。
"""
from __future__ import annotations

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

from _audit._blind_theory_loaders import SPLIT_SEED, file_sha256, load_blind_tasks, split_checksum
from chemdl.theory import estimate_physical_params_v2

TAU_PRIMARY = 0.5
TAU_TIERS = (0.0, 0.5, 1.0, 2.0, 5.0)


def classify_pred(pred_pct: float, tau: float) -> str:
    if tau == 0.0:
        return "IMPROVE" if pred_pct < 0.0 else "WORSEN"
    if pred_pct <= -tau:
        return "IMPROVE"
    if pred_pct >= tau:
        return "WORSEN"
    return "ABSTAIN"


def main() -> None:
    t0 = time.time()
    theory_path = ROOT / "chemdl" / "theory.py"
    diff_out = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "chemdl/theory.py"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert not diff_out, f"chemdl/theory.py 存在未冻结改动: {diff_out}"

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    theory_sha = file_sha256(theory_path)
    ts_local = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    tasks = load_blind_tasks(ROOT)
    print(f"Loaded {len(tasks)} unseen tasks (git={git_head[:7]}, theory_sha={theory_sha[:12]})", flush=True)

    task_records = []
    total_folds = 0
    for ti, t in enumerate(tasks, 1):
        X, y = t["X"], t["y"]
        fold_recs = []
        for fi, (tr_idx, te_idx) in enumerate(t["splits"]):
            # 严格仅使用训练集 X[tr_idx], y[tr_idx]（绝不读取 y[te_idx] 或 X[te_idx]）
            y_tr = y[tr_idx]
            mu = float(y_tr.mean())
            sd = max(float(y_tr.std()), 1e-9)
            y_tr_z = ((y_tr - mu) / sd)[:, None]

            info = estimate_physical_params_v2(X[tr_idx], y_tr_z)
            pc = info["per_col"][0]
            pred_pct = float(pc["pred_snv_vs_raw_pct_v2"])
            calls = {f"tau_{tau}": classify_pred(pred_pct, tau) for tau in TAU_TIERS}

            fold_recs.append(
                {
                    "fold": int(fi),
                    "n_tr": int(len(tr_idx)),
                    "n_te": int(len(te_idx)),
                    "split_sha256": split_checksum(tr_idx, te_idx),
                    "pred_snv_vs_raw_pct_v2": pred_pct,
                    "call_primary_0_5pct": calls[f"tau_{TAU_PRIMARY}"],
                    "calls_by_tau": calls,
                    "r2_s_shape": float(pc["r2_s_shape"]),
                    "r2_mu_shape": float(pc["r2_mu_shape"]),
                    "sigma_s": float(pc["sigma_s"]),
                    "sigma_a_true": float(pc["sigma_a_true"]),
                    "sigma_non_chem": float(pc["sigma_non_chem"]),
                    "sigma_b_true": float(pc["sigma_b_true"]),
                    "r_s": float(pc["r_s"]),
                    "r_mu": float(pc["r_mu"]),
                    "r_crit": float(pc["r_crit"]),
                    "alpha": float(pc["alpha"]),
                    "beta": float(pc["beta"]),
                    "ws_star": float(pc["ws_star"]),
                    "wc_star": float(pc["wc_star"]),
                }
            )
            total_folds += 1

        preds = [f["pred_snv_vs_raw_pct_v2"] for f in fold_recs]
        mean_pred = round(float(np.mean(preds)), 2)
        sd_pred = round(float(np.std(preds, ddof=1)) if len(preds) > 1 else 0.0, 2)
        task_calls = {f"tau_{tau}": classify_pred(mean_pred, tau) for tau in TAU_TIERS}
        vote_counts = {
            "IMPROVE": sum(1 for f in fold_recs if f["call_primary_0_5pct"] == "IMPROVE"),
            "WORSEN": sum(1 for f in fold_recs if f["call_primary_0_5pct"] == "WORSEN"),
            "ABSTAIN": sum(1 for f in fold_recs if f["call_primary_0_5pct"] == "ABSTAIN"),
        }

        rec = {
            "task_id": t["task_id"],
            "dataset_group": t["dataset_group"],
            "domain": t["domain"],
            "instrument": t["instrument"],
            "target_name": t["target_name"],
            "split_type": t["split_type"],
            "n": t["n"],
            "L": t["L"],
            "n_folds": len(fold_recs),
            "pred_mean_pct": mean_pred,
            "pred_sd_pct": sd_pred,
            "call_primary_0_5pct": task_calls[f"tau_{TAU_PRIMARY}"],
            "calls_by_tau": task_calls,
            "fold_vote_counts_0_5pct": vote_counts,
            "r2_s_shape_mean": round(float(np.mean([f["r2_s_shape"] for f in fold_recs])), 4),
            "r2_mu_shape_mean": round(float(np.mean([f["r2_mu_shape"] for f in fold_recs])), 4),
            "sigma_s_mean": round(float(np.mean([f["sigma_s"] for f in fold_recs])), 4),
            "sigma_a_true_mean": round(float(np.mean([f["sigma_a_true"] for f in fold_recs])), 4),
            "r_s_mean": round(float(np.mean([f["r_s"] for f in fold_recs])), 4),
            "r_crit_mean": round(float(np.mean([f["r_crit"] for f in fold_recs])), 4),
            "ws_star_mean": round(float(np.mean([f["ws_star"] for f in fold_recs])), 4),
            "wc_star_mean": round(float(np.mean([f["wc_star"] for f in fold_recs])), 4),
            "folds": fold_recs,
        }
        task_records.append(rec)
        print(
            f"[{ti:02d}/{len(tasks)}] {t['task_id']:32s} | n={t['n']:4d} L={t['L']:4d} | "
            f"pred={mean_pred:+6.2f}% | call={rec['call_primary_0_5pct']:7s} | "
            f"R2_shape={rec['r2_s_shape_mean']:.3f} | sigma_a={rec['sigma_a_true_mean']:.4f} | "
            f"r_s={rec['r_s_mean']:+.3f} vs r_crit={rec['r_crit_mean']:.3f}",
            flush=True,
        )

    out = {
        "phase": "PHASE_1_PREREGISTRATION_LOCKED",
        "timestamp": ts_local,
        "git_head_frozen": git_head,
        "theory_py_path": "chemdl/theory.py",
        "theory_py_sha256": theory_sha,
        "split_seed": SPLIT_SEED,
        "tau_primary_pct": TAU_PRIMARY,
        "tau_tiers_pct": list(TAU_TIERS),
        "decision_rule": "pred <= -tau -> IMPROVE; pred >= +tau -> WORSEN; |pred| < tau -> ABSTAIN (tau=0: <0 IMPROVE else WORSEN)",
        "n_tasks": len(task_records),
        "total_folds": total_folds,
        "elapsed_sec": round(time.time() - t0, 2),
        "tasks": task_records,
    }

    json_path = ROOT / "_audit" / "prereg_theory_blind_2026_11_24.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)

    # 生成人类可读预注册锁定报告（全部路径为相对链接，确保 repo 内链接有效）
    lines = [
        "# 定理 1 (`estimate_physical_params_v2`) 盲测预注册锁定表（Phase 1）",
        "",
        f"- **锁定时间**：`{ts_local}`",
        f"- **基线代码提交**：`{git_head}`",
        f"- **冻结理论源码**：[`chemdl/theory.py`](../../chemdl/theory.py)（SHA-256：`{theory_sha}`）",
        f"- **切分种子**：`{SPLIT_SEED}`（共 **{len(task_records)} 项未参与开发任务 / {total_folds} 折**）",
        f"- **冻结三态规则**（主阈值 $\\tau = {TAU_PRIMARY}\\%$）：`pred <= -{TAU_PRIMARY}%` → **`IMPROVE`（改善）**；`pred >= +{TAU_PRIMARY}%` → **`WORSEN`（恶化）**；`|pred| < {TAU_PRIMARY}%` → **`ABSTAIN`（无法判断）**；次级阈值 $\\tau \\in \\{{0.0\\%, 1.0\\%, 2.0\\%, 5.0\\%\\}}$。",
        "- **盲法声明**：本阶段脚本严格仅读取各训练折 `(X_tr, y_tr)`，未调用 `fit_pls` 或 `PREPS[\"snv\"]`，未接触任何测试折性能。",
        "",
        "## 一、预注册预测汇总",
        "",
        f"- `IMPROVE`（预测改善）：**{sum(1 for r in task_records if r['call_primary_0_5pct'] == 'IMPROVE')}** / {len(task_records)}",
        f"- `WORSEN`（预测恶化）：**{sum(1 for r in task_records if r['call_primary_0_5pct'] == 'WORSEN')}** / {len(task_records)}",
        f"- `ABSTAIN`（无法判断）：**{sum(1 for r in task_records if r['call_primary_0_5pct'] == 'ABSTAIN')}** / {len(task_records)}",
        "",
        "## 二、逐任务冻结预注册预测表",
        "",
        "| # | Task ID | Domain | n | L | Split | pred_v2 (%) | 锁定判断 (τ=0.5%) | 折投票 (I/W/A) | R²_shape | σ_s | σ_a_true | r_s | r_crit | ws* |",
        "|---:|---|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, r in enumerate(task_records, 1):
        vc = r["fold_vote_counts_0_5pct"]
        lines.append(
            f"| {idx} | `{r['task_id']}` | {r['domain']} | {r['n']} | {r['L']} | {r['split_type']} "
            f"| {r['pred_mean_pct']:+.2f} | **{r['call_primary_0_5pct']}** | {vc['IMPROVE']}/{vc['WORSEN']}/{vc['ABSTAIN']} "
            f"| {r['r2_s_shape_mean']:.3f} | {r['sigma_s_mean']:.4f} | {r['sigma_a_true_mean']:.4f} "
            f"| {r['r_s_mean']:+.3f} | {r['r_crit_mean']:.3f} | {r['ws_star_mean']:.2f} |"
        )
    lines.append("")

    md_path = ROOT / "reports" / "antigravity" / "PREREG_THEORY_BLIND_2026-11-24.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nPhase 1 locked -> {json_path} & {md_path} ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
