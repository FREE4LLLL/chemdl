"""Round-2 Phase 1：冻结新版双层判据（Exact-PRESS / Group-CV Paired-LOO + 定理 1 归因）预注册锁定脚本。

硬规则：
  - 禁止导入 fit_pls 或访问任何测试折标签 y_te；
  - 验证 chemdl/theory.py 工作区无未提交改动并记录 SHA-256；
  - 跑通并本地提交锁定产物后，再编写或运行 Phase 2 揭盲脚本。
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

from _audit._blind_round2_loaders import SPLIT_SEED_R2, load_round2_tasks
from _audit._blind_theory_loaders import file_sha256, split_checksum
from chemdl.theory import diagnose_snv_decision

T_CRIT_PRIMARY = 2.0
MIN_EFFECT_PRIMARY = 1.0
T_CRIT_SEC = 1.5


def _call_from_stats(delta_pct: float, t_val: float, t_crit: float, min_eff: float) -> str:
    if t_val <= -t_crit and delta_pct <= -min_eff:
        return "IMPROVE"
    if t_val >= t_crit and delta_pct >= min_eff:
        return "WORSEN"
    return "ABSTAIN"


def _legacy_call(pred_pct: float, tau: float = 0.5) -> str:
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
    assert not diff_out, f"chemdl/theory.py 有未提交改动: {diff_out}"

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    theory_sha = file_sha256(theory_path)
    ts_local = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    tasks = load_round2_tasks(ROOT)
    print(f"Loaded {len(tasks)} Round-2 unseen tasks (git={git_head[:7]}, theory_sha={theory_sha[:12]})", flush=True)

    task_recs = []
    total_folds = 0
    for ti, t in enumerate(tasks, 1):
        X, y, groups = t["X"], t["y"], t["groups"]
        fold_recs = []
        for fi, (tr_idx, te_idx) in enumerate(t["splits"]):
            # 严格仅看训练折 X[tr_idx], y[tr_idx], groups[tr_idx]
            g_tr = groups[tr_idx] if groups is not None else None
            diag = diagnose_snv_decision(
                X[tr_idx],
                y[tr_idx],
                groups_tr=g_tr,
                t_crit=T_CRIT_PRIMARY,
                min_effect_pct=MIN_EFFECT_PRIMARY,
                compute_physics=True,
            )
            call_sec = _call_from_stats(diag["delta_loo_pct"], diag["t_loo"], T_CRIT_SEC, MIN_EFFECT_PRIMARY)
            call_legacy_05 = _legacy_call(diag["pred_v2_legacy_pct"], tau=0.5)

            fold_recs.append(
                {
                    "fold": int(fi),
                    "n_tr": int(len(tr_idx)),
                    "n_te": int(len(te_idx)),
                    "split_sha256": split_checksum(tr_idx, te_idx),
                    **diag,
                    "call_primary": diag["decision"],
                    "call_secondary_t1_5": call_sec,
                    "call_legacy_v2_0_5": call_legacy_05,
                }
            )
            total_folds += 1

        mean_delta = round(float(np.mean([f["delta_loo_pct"] for f in fold_recs])), 2)
        mean_t = round(float(np.mean([f["t_loo"] for f in fold_recs])), 3)
        mean_tabs = round(float(np.mean([f["t_loo_abs"] for f in fold_recs])), 3)
        mean_r2 = round(float(np.mean([f["r2_s_shape"] for f in fold_recs])), 4)
        mean_sa = round(float(np.mean([f["sigma_a_true"] for f in fold_recs])), 4)
        mean_leg = round(float(np.mean([f["pred_v2_legacy_pct"] for f in fold_recs])), 2)
        mean_mult = round(float(np.mean([f["pred_mult_pct"] for f in fold_recs])), 2)

        call_pri = _call_from_stats(mean_delta, mean_t, T_CRIT_PRIMARY, MIN_EFFECT_PRIMARY)
        call_sec = _call_from_stats(mean_delta, mean_t, T_CRIT_SEC, MIN_EFFECT_PRIMARY)
        call_leg = _legacy_call(mean_leg, tau=0.5)
        vc = {k: sum(1 for f in fold_recs if f["call_primary"] == k) for k in ("IMPROVE", "WORSEN", "ABSTAIN")}
        status_task = "SCATTER_DOMINATED" if mean_r2 < 0.80 else "HIGH_SHAPE_COUPLING_CAUTION"

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
            "mode": fold_recs[0]["mode"],
            "n_eff_units_mean": round(float(np.mean([f["n_eff_units"] for f in fold_recs])), 1),
            "delta_loo_mean_pct": mean_delta,
            "t_loo_mean": mean_t,
            "t_loo_abs_mean": mean_tabs,
            "call_primary": call_pri,
            "call_secondary_t1_5": call_sec,
            "fold_vote_counts_primary": vc,
            "pred_v2_legacy_mean_pct": mean_leg,
            "call_legacy_v2_0_5": call_leg,
            "pred_mult_mean_pct": mean_mult,
            "r2_s_shape_mean": mean_r2,
            "sigma_a_true_mean": mean_sa,
            "theory_assumption_status": status_task,
            "folds": fold_recs,
        }
        task_recs.append(rec)
        print(
            f"[{ti:02d}/{len(tasks)}] {t['task_id']:31s} | mode={rec['mode']:15s} | "
            f"dLOO={mean_delta:+7.2f}% t={mean_t:+6.2f} -> {call_pri:7s} | "
            f"legacy_v2={mean_leg:+6.2f}% ({call_leg:7s}) | R2_shape={mean_r2:.3f}",
            flush=True,
        )

    out = {
        "phase": "ROUND2_PHASE_1_PREREGISTRATION_LOCKED",
        "timestamp": ts_local,
        "git_head_frozen": git_head,
        "theory_py_path": "chemdl/theory.py",
        "theory_py_sha256": theory_sha,
        "split_seed": SPLIT_SEED_R2,
        "primary_rule": f"t_loo <= -{T_CRIT_PRIMARY} & delta_loo <= -{MIN_EFFECT_PRIMARY}% -> IMPROVE; >= +{T_CRIT_PRIMARY} & >= +{MIN_EFFECT_PRIMARY}% -> WORSEN; else ABSTAIN",
        "secondary_rule": f"t_crit={T_CRIT_SEC}, min_effect={MIN_EFFECT_PRIMARY}%",
        "n_tasks": len(task_recs),
        "total_folds": total_folds,
        "elapsed_sec": round(time.time() - t0, 2),
        "tasks": task_recs,
    }

    out_json = ROOT / "_audit" / "prereg_snv_gate_round2_2026_11_24.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)

    lines = [
        "# 第二轮盲测预注册锁定表：双层 Exact-PRESS Paired-LOO 门控（Round 2 · Phase 1）",
        "",
        f"- **锁定时间**：`{ts_local}`",
        f"- **基线代码提交**：`{git_head}`",
        f"- **冻结理论源码**：[`chemdl/theory.py`](../../chemdl/theory.py)（SHA-256：`{theory_sha}`）",
        f"- **切分种子**：`{SPLIT_SEED_R2}`（共 **{len(task_recs)} 项全新未触碰任务 / {total_folds} 折**，与开发集及 Round 1 零重叠）",
        f"- **主冻结判决准则**：`t_loo <= -{T_CRIT_PRIMARY} 且 Δ_LOO <= -{MIN_EFFECT_PRIMARY}%` → **`IMPROVE`**；`t_loo >= +{T_CRIT_PRIMARY} 且 Δ_LOO >= +{MIN_EFFECT_PRIMARY}%` → **`WORSEN`**；其余 → **`ABSTAIN`（无法判断）**。",
        "- **盲法声明**：本阶段脚本严格仅输入各训练折 `(X_tr, y_tr)`，未调用 `fit_pls` 或读取任何测试折目标。",
        "",
        "## 一、预注册预测分布汇总",
        "",
        (
            f"- **新版 `diagnose_snv_decision`（主规则 `|t_loo| ≥ {T_CRIT_PRIMARY}, |Δ_LOO| ≥ {MIN_EFFECT_PRIMARY}%`）**："
            f"`IMPROVE` = **{sum(1 for r in task_recs if r['call_primary'] == 'IMPROVE')}**，"
            f"`WORSEN` = **{sum(1 for r in task_recs if r['call_primary'] == 'WORSEN')}**，"
            f"`ABSTAIN` = **{sum(1 for r in task_recs if r['call_primary'] == 'ABSTAIN')}**。"
        ),
        (
            f"- **对照：旧版 `v2`（`|pred| ≥ 0.5%`）**："
            f"`IMPROVE` = **{sum(1 for r in task_recs if r['call_legacy_v2_0_5'] == 'IMPROVE')}**，"
            f"`WORSEN` = **{sum(1 for r in task_recs if r['call_legacy_v2_0_5'] == 'WORSEN')}**，"
            f"`ABSTAIN` = **{sum(1 for r in task_recs if r['call_legacy_v2_0_5'] == 'ABSTAIN')}**。"
        ),
        "",
        "## 二、全 18 项任务冻结预注册表",
        "",
        "| # | Task ID | n | L | Mode | Δ_LOO (%) | t_loo | **新版锁定判断** | 折投票 (I/W/A) | 旧 `v2` (%) | 旧 `v2` 判断 | R²_shape | σ_a_true | 物理状态 |",
        "|---:|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---|",
    ]
    for idx, r in enumerate(task_recs, 1):
        vc = r["fold_vote_counts_primary"]
        lines.append(
            f"| {idx} | `{r['task_id']}` | {r['n']} | {r['L']} | {r['mode']} "
            f"| {r['delta_loo_mean_pct']:+.2f} | {r['t_loo_mean']:+.2f} | **{r['call_primary']}** "
            f"| {vc['IMPROVE']}/{vc['WORSEN']}/{vc['ABSTAIN']} | {r['pred_v2_legacy_mean_pct']:+.2f} | {r['call_legacy_v2_0_5']} "
            f"| {r['r2_s_shape_mean']:.3f} | {r['sigma_a_true_mean']:.4f} | {r['theory_assumption_status']} |"
        )
    lines.append("")

    out_md = ROOT / "reports" / "antigravity" / "PREREG_SNV_GATE_ROUND2_2026-11-24.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRound-2 Phase 1 locked -> {out_json} & {out_md} ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
