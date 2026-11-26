"""Phase B · Phase 1：严格无测试集接触地冻结 18 项独立留出部署任务的 SNV 门控决策与 InnerCV_PLS 基线。

硬约束：
  - 本脚本严格禁止访问 `X_test` 与 `y_test`；
  - 检查 `chemdl/theory.py` 无未提交改动并同时记录 Git LF Blob SHA-256 与工作树 LF SHA-256；
  - 锁单产出提交至本地 Git 后，才允许编写/运行 Phase 2 揭盲。
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._blind_theory_loaders import file_sha256_lf, git_blob_sha256
from _audit._deploy_holdout_loaders import SPLIT_SEED_DEPLOY, load_deploy_holdout_tasks
from chemdl.theory import diagnose_snv_decision
from chemdl.transfer import fit_pls, fit_prep_fn

T_CRIT = 2.0
MIN_EFF_PCT = 1.0


def _inner_cv_raw_vs_snv_pls(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    groups_cal: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """严格仅在 cal 内按 5 折（有分组时用 GroupKFold）跑同头 PLS 对比 Raw 与 SNV。"""
    Xa = np.asarray(X_cal, dtype=np.float64)
    ya = np.asarray(y_cal, dtype=np.float64).ravel()
    mu = float(ya.mean())
    sd = max(float(ya.std()), 1e-9)
    yn = ((ya - mu) / sd)[:, None]

    n = len(Xa)
    if groups_cal is None:
        perm = np.random.default_rng(seed).permutation(n)
        folds = np.array_split(perm, 5)
        splits = [(perm[~np.isin(perm, va)], va) for va in folds]
    else:
        grp = np.asarray(groups_cal)
        uniq = np.unique(grp)
        eff = min(5, len(uniq))
        splits = list(GroupKFold(n_splits=eff).split(Xa, yn, grp))

    sq_raw, sq_snv = [], []
    for fi, (itr, iva) in enumerate(splits):
        kmax = min(10, max(2, len(itr) // 2))
        inner = min(4, max(2, len(itr) // 4))
        pr_raw, _ = fit_pls(Xa[itr], yn[itr], kmax=kmax, inner=inner, per_col=True, seed=fi)
        fn_snv = fit_prep_fn("snv", Xa[itr])
        pr_snv, _ = fit_pls(fn_snv(Xa[itr]), yn[itr], kmax=kmax, inner=inner, per_col=True, seed=fi)
        sq_raw.append((pr_raw(Xa[iva]) - yn[iva]) ** 2)
        sq_snv.append((pr_snv(fn_snv(Xa[iva])) - yn[iva]) ** 2)

    cv_raw = float(np.sqrt(np.mean(np.concatenate(sq_raw, axis=0))))
    cv_snv = float(np.sqrt(np.mean(np.concatenate(sq_snv, axis=0))))
    delta_cv_pct = (cv_snv / max(cv_raw, 1e-12) - 1.0) * 100.0
    pick = "SNV" if cv_snv < cv_raw else "RAW"
    return {
        "cv_rmsep_raw_z": round(cv_raw, 4),
        "cv_rmsep_snv_z": round(cv_snv, 4),
        "delta_cv_pct": round(delta_cv_pct, 2),
        "inner_cv_pick": pick,
    }


def _legacy_call(pred_pct: float, tau: float = 0.5) -> str:
    if pred_pct <= -tau:
        return "IMPROVE"
    if pred_pct >= tau:
        return "WORSEN"
    return "ABSTAIN"


def _lock_cal_only(
    task_meta: dict,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    groups_cal: np.ndarray | None,
) -> dict:
    # 函数签名完全不含 X_test / y_test，保证结构性零窥视
    diag = diagnose_snv_decision(
        X_cal,
        y_cal,
        groups_tr=groups_cal,
        t_crit=T_CRIT,
        min_effect_pct=MIN_EFF_PCT,
        compute_physics=True,
    )
    cv_info = _inner_cv_raw_vs_snv_pls(X_cal, y_cal, groups_cal=groups_cal, seed=0)
    return {
        **task_meta,
        "gate_decision": diag["decision"],
        "gate_mode": diag["mode"],
        "n_eff_units": diag["n_eff_units"],
        "delta_loo_pct": diag["delta_loo_pct"],
        "t_loo": diag["t_loo"],
        "t_loo_abs": diag["t_loo_abs"],
        "legacy_v2_pred_pct": diag["pred_v2_legacy_pct"],
        "legacy_v2_call_0_5": _legacy_call(diag["pred_v2_legacy_pct"], tau=0.5),
        "r2_s_shape": diag["r2_s_shape"],
        "sigma_a_true": diag["sigma_a_true"],
        "theory_assumption_status": diag["theory_assumption_status"],
        **cv_info,
    }


def main() -> None:
    t0 = time.time()
    theory_path = ROOT / "chemdl" / "theory.py"
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "chemdl/theory.py"], cwd=ROOT, text=True).strip()
    assert not dirty, f"chemdl/theory.py 有未提交改动: {dirty}"

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    blob_lf = git_blob_sha256(ROOT, git_head, "chemdl/theory.py", lf_normalize=True)
    wt_lf = file_sha256_lf(theory_path)
    assert wt_lf == blob_lf, f"Worktree LF SHA-256 mismatch vs HEAD blob: {wt_lf} vs {blob_lf}"

    tasks = load_deploy_holdout_tasks(ROOT)
    recs = []
    for i, t in enumerate(tasks, 1):
        meta = {
            "task_id": t["task_id"],
            "stratum": t["stratum"],
            "domain": t["domain"],
            "split_rule": t["split_rule"],
            "target_name": t["target_name"],
            "n_cal": t["n_cal"],
            "n_test": t["n_test"],
            "L": t["L"],
            "cal_test_hash_overlap": t["cal_test_hash_overlap"],
            "split_sha256": t["split_sha256"],
        }
        rec = _lock_cal_only(meta, t["X_cal"], t["y_cal"], t["groups_cal"])
        recs.append(rec)
        print(
            f"[{i:02d}/{len(tasks)}] {rec['task_id']:44s} | "
            f"GATE={rec['gate_decision']:7s} (dLOO={rec['delta_loo_pct']:+6.2f}% t={rec['t_loo']:+6.2f}) | "
            f"InnerCV={rec['inner_cv_pick']:3s} ({rec['delta_cv_pct']:+6.2f}%)",
            flush=True,
        )

    out = {
        "phase": "DEPLOY_PHASE_1_PREREGISTRATION_LOCKED",
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head_frozen": git_head,
        "theory_py_path": "chemdl/theory.py",
        "theory_py_git_blob_sha256_lf": blob_lf,
        "theory_py_worktree_sha256_lf": wt_lf,
        "split_seed": SPLIT_SEED_DEPLOY,
        "gate_rule": f"|t_loo|>={T_CRIT} & |delta_loo|>={MIN_EFF_PCT}%",
        "n_tasks": len(recs),
        "elapsed_sec": round(time.time() - t0, 2),
        "tasks": recs,
    }
    out_json = ROOT / "_audit" / "prereg_deploy_holdout_2026_11_24.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)

    lines = [
        "# Phase B 独立留出端到端部署对比预注册锁定表（Phase 1）",
        "",
        f"- **锁定提交**：`{git_head}`",
        f"- **冻结理论源码 LF SHA-256**：[`chemdl/theory.py`](../../chemdl/theory.py) = `{blob_lf}`",
        f"- **任务数**：**{len(recs)} 项严格独立单次留出（`cal → test`，全部强断言字节重叠 == 0）**",
        "- **盲法声明**：`_lock_cal_only` 函数签名为 `(meta, X_cal, y_cal, groups_cal)`，绝无接触 `X_test / y_test`。",
        "",
        "| # | Task ID | 分层 | cal → test | 门控决策 (`dLOO / t`) | InnerCV 选择 (`dCV`) | 旧 `v2` |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for idx, r in enumerate(recs, 1):
        lines.append(
            f"| {idx} | `{r['task_id']}` | `{r['stratum']}` | {r['n_cal']} → {r['n_test']} "
            f"| **{r['gate_decision']}** (`{r['delta_loo_pct']:+.2f}% / {r['t_loo']:+.2f}`) "
            f"| {r['inner_cv_pick']} (`{r['delta_cv_pct']:+.2f}%`) | {r['legacy_v2_call_0_5']} (`{r['legacy_v2_pred_pct']:+.2f}%`) |"
        )
    lines.append("")
    out_md = ROOT / "reports" / "antigravity" / "PREREG_DEPLOY_HOLDOUT_2026-11-24.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDeploy Phase 1 locked -> {out_json} & {out_md} ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
