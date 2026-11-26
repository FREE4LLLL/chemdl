"""Round 3 · Phase 1：严格无测试集接触地冻结 6 个全新外部 Zenodo NIR 实验室（13 项任务）的 SNV 门控决策与 InnerCV 基线。

硬约束：
  - 严格禁止访问 `X_test` 与 `y_test`；
  - 强断言 `chemdl/theory.py` 无未提交改动且 LF SHA-256 == `FROZEN_THEORY_LF_SHA256`；
  - 锁单产出提交至本地 Git 后，才允许编写/运行 Round 3 Phase 2 揭盲。
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

from _audit._blind_theory_loaders import file_sha256_lf, git_blob_sha256
from _audit._round3_unseen_loaders import load_round3_unseen_tasks
from _audit.prereg_deploy_holdout_phase1 import _inner_cv_raw_vs_snv_pls, _legacy_call
from chemdl.deploy_gate import FROZEN_THEORY_LF_SHA256
from chemdl.theory import diagnose_snv_decision

T_CRIT = 2.0
MIN_EFF_PCT = 1.0


def _lock_cal_only(
    task_meta: dict,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    groups_cal: np.ndarray | None,
) -> dict:
    # 函数签名完全不含 X_test / y_test，结构性保证零测试集窥视
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
    assert wt_lf == blob_lf == FROZEN_THEORY_LF_SHA256, (
        f"chemdl/theory.py LF SHA-256 mismatch: wt={wt_lf}, blob={blob_lf}, const={FROZEN_THEORY_LF_SHA256}"
    )

    tasks = load_round3_unseen_tasks(ROOT)
    labs = sorted({t["lab_id"] for t in tasks})
    recs = []
    for i, t in enumerate(tasks, 1):
        meta = {
            "task_id": t["task_id"],
            "lab_id": t["lab_id"],
            "license": t["license"],
            "doi": t["doi"],
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
            f"[{i:02d}/{len(tasks)}] {rec['task_id']:40s} | GATE={rec['gate_decision']:7s} "
            f"(dLOO={rec['delta_loo_pct']:+6.2f}% t={rec['t_loo']:+6.2f}) | InnerCV={rec['inner_cv_pick']:3s} ({rec['delta_cv_pct']:+6.2f}%)",
            flush=True,
        )

    out = {
        "phase": "ROUND3_PHASE_1_PREREGISTRATION_LOCKED",
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head_frozen": git_head,
        "theory_py_path": "chemdl/theory.py",
        "theory_py_git_blob_sha256_lf": blob_lf,
        "theory_py_worktree_sha256_lf": wt_lf,
        "t_crit": T_CRIT,
        "min_effect_pct": MIN_EFF_PCT,
        "n_tasks": len(recs),
        "n_distinct_labs": len(labs),
        "labs": labs,
        "excluded_before_phase1": {
            "Z-14177889_grape_ripeness": "Multispectral sensor with only 10 channels (L=10 < 16 threshold)",
            "Z-7980675_wood_rot_nir": "Only 36 spectrum columns total (70/30 test split would have n_test=11 < 15)",
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "tasks": recs,
    }
    out_json = ROOT / "_audit" / "prereg_round3_unseen_2026_11_24.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)

    md_lines = [
        "# Round 3 零接触外部 Zenodo NIR 实验室预注册锁单（Phase 1）",
        "",
        f"- **冻结理论 LF SHA-256**：`chemdl/theory.py` = `{blob_lf}`（锚定提交 `{git_head}`）",
        f"- **任务与独立实验室规模**：**{len(recs)} 项独立 `cal -> test` 任务 / {len(labs)} 个全新外部实验室**（全 `cal/test` 光谱重叠 `== 0`，总耗时 `{out['elapsed_sec']:.2f}s`）",
        f"- **Phase 1 前准入过滤（未接触测试集）**：`Z-14177889_grape`（`L=10 < 16` 多光谱）与 `Z-7980675_wood_rot`（仅 36 谱，`n_test < 15`）已于锁单前剔除。",
        "- **严格契约**：本阶段仅运行校准集 `(X_cal, y_cal)`，完全未访问 `X_test / y_test`。",
        "",
        "| # | Task ID | 实验室 | 许可 | 样本数 (`cal->test`) | L | 门控预测 | dLOO | t_loo | InnerCV | 旧版 v2 |",
        "|---:|---|---|---|---|---:|---|---:|---:|---|---|",
    ]
    for i, r in enumerate(recs, 1):
        md_lines.append(
            f"| {i} | `{r['task_id']}` | `{r['lab_id']}` | {r['license']} | {r['n_cal']}->{r['n_test']} | {r['L']} | "
            f"**{r['gate_decision']}** | {r['delta_loo_pct']:+.2f}% | {r['t_loo']:+.2f} | "
            f"{r['inner_cv_pick']} ({r['delta_cv_pct']:+.2f}%) | {r['legacy_v2_call_0_5']} ({r['legacy_v2_pred_pct']:+.2f}%) |"
        )
    out_md = ROOT / "reports" / "antigravity" / "PREREG_ROUND3_UNSEEN_2026-11-24.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nLocked Round 3 Phase 1:\n  JSON: {out_json}\n  MD:   {out_md}")


if __name__ == "__main__":
    main()
