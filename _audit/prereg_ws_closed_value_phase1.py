"""Phase 1 预注册锁单：严格仅用训练集 (X_cal, y_cal, groups_cal) 标定闭式 w_s* 的“内部最优”标签与内层 CV 配置。

硬不变量：
  1. 在进入参数估计与内层 CV 前立即剥离 X_test / y_test / groups_test，架构级保证测试集零接触；
  2. 固定 w_c = 1.0，固定 PLS（主对照）与 Ridge（伴随稳健性对照）内层选参协议；
  3. 同时锁定 v2_alg（主理论：R²_shape 分解 + σ_non=0 纯闭式）、v1_alg（σ_non=0 纯闭式）与 v1_grid（σ_non>0 1001 点网格）；
  4. 本脚本运行并提交 Git 后，才允许编写并运行 Phase 2 揭盲脚本。
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
from _audit._ws_closed_value_loaders import (
    FIXED_WC,
    INTERIOR_EPS,
    WS_GRID,
    classify_ws_regime,
    load_ws_value_tasks,
    make_inner_splits,
    select_gamma_ridge_for_ws,
    select_k_pls_for_ws,
    select_ws_inner_cv_pls,
    select_ws_inner_cv_ridge,
)
from chemdl.deploy_gate import FROZEN_THEORY_LF_SHA256
from chemdl.theory import closed_form_optimal_ws, estimate_physical_params, estimate_physical_params_v2


def _sanitize_json(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not np.isfinite(v):
            raise ValueError(f"Non-finite float encountered in JSON payload: {v}")
        return v
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _lock_cal_only_task(meta_cal: dict) -> dict:
    """Strictly receives ONLY (X_cal, y_cal, groups_cal) — zero test arrays."""
    Xc = meta_cal["X_cal"]
    yc = meta_cal["y_cal"]
    gc = meta_cal["groups_cal"]

    mu = float(np.mean(yc))
    sd = max(float(np.std(yc)), 1e-9)
    yn = (yc - mu) / sd

    # 1. Closed-form physical parameter estimation (full X_cal, no subsampling)
    t0 = time.perf_counter()
    iv2 = estimate_physical_params_v2(Xc, yc)
    dt_v2_params = time.perf_counter() - t0
    pc2 = iv2["per_col"][0]
    ws_v2_alg = float(pc2["ws_star"])

    t0 = time.perf_counter()
    iv1 = estimate_physical_params(Xc, yc)
    dt_v1_params = time.perf_counter() - t0
    pc1 = iv1["per_col"][0]
    ws_v1_grid = float(pc1["ws_star"])
    ws_v1_alg = float(
        closed_form_optimal_ws(
            pc1["alpha"], pc1["beta"], pc1["r_s"], pc1["sigma_a"], sigma_non=0.0
        )
    )

    tag_v2 = classify_ws_regime(ws_v2_alg, eps=INTERIOR_EPS)
    tag_v1_alg = classify_ws_regime(ws_v1_alg, eps=INTERIOR_EPS)
    tag_v1_grid = classify_ws_regime(ws_v1_grid, eps=INTERIOR_EPS)

    # 2. Primary PLS head: fixed inner splits, kmax=12
    splits = make_inner_splits(len(Xc), gc, n_splits=5, seed=0)

    t0 = time.perf_counter()
    k_ws0, cv_pls_ws0 = select_k_pls_for_ws(Xc, yn, 0.0, splits, kmax=12, wc=FIXED_WC)
    dt_pls_ws0 = time.perf_counter() - t0

    t0 = time.perf_counter()
    k_ws1, cv_pls_ws1 = select_k_pls_for_ws(Xc, yn, 1.0, splits, kmax=12, wc=FIXED_WC)
    dt_pls_ws1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    k_v2, cv_pls_v2 = select_k_pls_for_ws(Xc, yn, ws_v2_alg, splits, kmax=12, wc=FIXED_WC)
    dt_pls_v2_k = time.perf_counter() - t0

    t0 = time.perf_counter()
    k_v1_alg, cv_pls_v1_alg = select_k_pls_for_ws(Xc, yn, ws_v1_alg, splits, kmax=12, wc=FIXED_WC)
    dt_pls_v1_alg_k = time.perf_counter() - t0

    t0 = time.perf_counter()
    k_v1_grid, cv_pls_v1_grid = select_k_pls_for_ws(Xc, yn, ws_v1_grid, splits, kmax=12, wc=FIXED_WC)
    dt_pls_v1_grid_k = time.perf_counter() - t0

    t0 = time.perf_counter()
    ws_cv_pls, k_cv_pls, cv_pls_best, grid_pls = select_ws_inner_cv_pls(
        Xc, yn, splits, ws_grid=WS_GRID, kmax=12, wc=FIXED_WC
    )
    dt_pls_inner_cv = time.perf_counter() - t0

    # 3. Companion Ridge head: exact Allen-PRESS / Group-CV
    t0 = time.perf_counter()
    g_ws0, cv_r_ws0 = select_gamma_ridge_for_ws(Xc, yn, 0.0, groups_cal=gc, wc=FIXED_WC)
    dt_r_ws0 = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_ws1, cv_r_ws1 = select_gamma_ridge_for_ws(Xc, yn, 1.0, groups_cal=gc, wc=FIXED_WC)
    dt_r_ws1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_v2, cv_r_v2 = select_gamma_ridge_for_ws(Xc, yn, ws_v2_alg, groups_cal=gc, wc=FIXED_WC)
    dt_r_v2_g = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_v1_alg, cv_r_v1_alg = select_gamma_ridge_for_ws(Xc, yn, ws_v1_alg, groups_cal=gc, wc=FIXED_WC)
    dt_r_v1_alg_g = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_v1_grid, cv_r_v1_grid = select_gamma_ridge_for_ws(Xc, yn, ws_v1_grid, groups_cal=gc, wc=FIXED_WC)
    dt_r_v1_grid_g = time.perf_counter() - t0

    t0 = time.perf_counter()
    ws_cv_ridge, g_cv_ridge, cv_r_best, grid_ridge = select_ws_inner_cv_ridge(
        Xc, yn, groups_cal=gc, ws_grid=WS_GRID, wc=FIXED_WC
    )
    dt_r_inner_cv = time.perf_counter() - t0

    return {
        "task_id": meta_cal["task_id"],
        "suite": meta_cal["suite"],
        "domain": meta_cal["domain"],
        "split_rule": meta_cal["split_rule"],
        "target_name": meta_cal["target_name"],
        "n_cal": meta_cal["n_cal"],
        "n_test": meta_cal["n_test"],
        "L": meta_cal["L"],
        "has_groups_cal": gc is not None,
        "has_label_scale_anomaly": meta_cal["has_label_scale_anomaly"],
        "cal_test_hash_overlap": 0,
        "split_sha256": meta_cal["split_sha256"],
        "physics_v2": {
            "alpha": float(pc2["alpha"]),
            "beta": float(pc2["beta"]),
            "r_s": float(pc2["r_s"]),
            "sigma_s": float(pc2["sigma_s"]),
            "sigma_a_true": float(pc2["sigma_a_true"]),
            "sigma_non_chem": float(pc2["sigma_non_chem"]),
            "r2_s_shape": float(pc2["r2_s_shape"]),
            "r_crit": float(pc2["r_crit"]),
            "r_crit_raw": float(pc2["r_crit_raw"]),
            "can_flip": bool(pc2["can_flip"]),
            "ws_star": round(ws_v2_alg, 6),
        },
        "physics_v1": {
            "alpha": float(pc1["alpha"]),
            "beta": float(pc1["beta"]),
            "r_s": float(pc1["r_s"]),
            "sigma_s": float(pc1["sigma_s"]),
            "sigma_a": float(pc1["sigma_a"]),
            "sigma_non": float(pc1["sigma_non"]),
            "r_crit": float(pc1["r_crit"]),
            "ws_star_alg": round(ws_v1_alg, 6),
            "ws_star_grid": round(ws_v1_grid, 6),
        },
        "pre_tags": {
            "v2_alg": tag_v2,
            "v1_alg": tag_v1_alg,
            "v1_grid": tag_v1_grid,
        },
        "pls_cal_config": {
            "n_inner_folds": len(splits),
            "ws_0": {"ws": 0.0, "k": int(k_ws0), "cv_rmsep_z": round(cv_pls_ws0, 6), "cal_time_s": round(dt_pls_ws0, 4)},
            "ws_1": {"ws": 1.0, "k": int(k_ws1), "cv_rmsep_z": round(cv_pls_ws1, 6), "cal_time_s": round(dt_pls_ws1, 4)},
            "ws_closed_v2": {
                "ws": round(ws_v2_alg, 6),
                "k": int(k_v2),
                "cv_rmsep_z": round(cv_pls_v2, 6),
                "cal_time_s": round(dt_v2_params + dt_pls_v2_k, 4),
            },
            "ws_closed_v1_alg": {
                "ws": round(ws_v1_alg, 6),
                "k": int(k_v1_alg),
                "cv_rmsep_z": round(cv_pls_v1_alg, 6),
                "cal_time_s": round(dt_v1_params + dt_pls_v1_alg_k, 4),
            },
            "ws_closed_v1_grid": {
                "ws": round(ws_v1_grid, 6),
                "k": int(k_v1_grid),
                "cv_rmsep_z": round(cv_pls_v1_grid, 6),
                "cal_time_s": round(dt_v1_params + dt_pls_v1_grid_k, 4),
            },
            "ws_inner_cv": {
                "ws": float(ws_cv_pls),
                "k": int(k_cv_pls),
                "cv_rmsep_z": round(cv_pls_best, 6),
                "cal_time_s": round(dt_pls_inner_cv, 4),
            },
            "grid_pls": grid_pls,
        },
        "ridge_cal_config": {
            "ws_0": {"ws": 0.0, "gamma": float(g_ws0), "cv_rmsep_z": round(cv_r_ws0, 6), "cal_time_s": round(dt_r_ws0, 4)},
            "ws_1": {"ws": 1.0, "gamma": float(g_ws1), "cv_rmsep_z": round(cv_r_ws1, 6), "cal_time_s": round(dt_r_ws1, 4)},
            "ws_closed_v2": {
                "ws": round(ws_v2_alg, 6),
                "gamma": float(g_v2),
                "cv_rmsep_z": round(cv_r_v2, 6),
                "cal_time_s": round(dt_v2_params + dt_r_v2_g, 4),
            },
            "ws_closed_v1_alg": {
                "ws": round(ws_v1_alg, 6),
                "gamma": float(g_v1_alg),
                "cv_rmsep_z": round(cv_r_v1_alg, 6),
                "cal_time_s": round(dt_v1_params + dt_r_v1_alg_g, 4),
            },
            "ws_closed_v1_grid": {
                "ws": round(ws_v1_grid, 6),
                "gamma": float(g_v1_grid),
                "cv_rmsep_z": round(cv_r_v1_grid, 6),
                "cal_time_s": round(dt_v1_params + dt_r_v1_grid_g, 4),
            },
            "ws_inner_cv": {
                "ws": float(ws_cv_ridge),
                "gamma": float(g_cv_ridge),
                "cv_rmsep_z": round(cv_r_best, 6),
                "cal_time_s": round(dt_r_inner_cv, 4),
            },
            "grid_ridge": grid_ridge,
        },
        "timing_s": {
            "v2_params_only": round(dt_v2_params, 4),
            "v1_params_only": round(dt_v1_params, 4),
        },
    }


def _write_phase1_report(payload: dict, out_md: Path) -> None:
    lines = [
        "# Phase 1 预注册锁单：闭式 $w_s^\\star$ 训练集“内部最优”预标定（54 项独立留出任务）",
        "",
        f"- **锁单阶段**：`{payload['phase']}`",
        f"- **锁单 Git HEAD**：`{payload['git_head_frozen']}`",
        f"- **`chemdl/theory.py` LF SHA-256**：`{payload['theory_py_git_blob_sha256_lf']}`",
        f"- **固定协议**：`w_c = {payload['fixed_wc']}`；主头 `PLS (kmax=12, 5-fold / GroupKFold)`；伴随头 `Ridge (Exact Allen-PRESS / Group-CV)`；`WS_GRID = {payload['ws_grid']}`；内部区段阈值 `({payload['interior_eps']}, {1.0 - payload['interior_eps']:.2f})`。",
        "",
        "## 1. 仅用训练集预标定的激活汇总",
        "",
        "| 理论分支 | 路径 | 内部最优 (`0.01 < w_s* < 0.99`) | 严格内部 (`0 < w_s* < 1`) | 端点 `w_s*=0` | 端点 `w_s*=1` |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for vk, label in (
        ("v2_alg", "`v2_alg` (主理论: `R²_shape` + `σ_non=0` 纯闭式)"),
        ("v1_alg", "`v1_alg` (`v1` 参数 + `σ_non=0` 纯闭式)"),
        ("v1_grid", "`v1_grid` (`v1` 参数 + `σ_non>0` 1001 点网格)"),
    ):
        s = payload["summary"]["pre_tag_counts"][vk]
        lines.append(
            f"| {vk} | {label} | {s['interior_01']}/{payload['n_tasks']} | "
            f"{s['strict_interior']}/{payload['n_tasks']} | {s['boundary_0']} | {s['boundary_1']} |"
        )

    lines.extend(["", "## 2. 预标定为 `INTERIOR (0.01 < w_s* < 0.99)` 的任务清单", ""])
    for vk in ("v2_alg", "v1_alg", "v1_grid"):
        inter = [r for r in payload["tasks"] if r["pre_tags"][vk]["is_interior_01"]]
        lines.append(f"### `{vk}` ({len(inter)}/{payload['n_tasks']})")
        lines.append("")
        if not inter:
            lines.append("- 无任务落入 `(0.01, 0.99)`。")
        else:
            lines.append("| Task ID | Suite | `n_cal` | `w_s*` | `w_s_cv (PLS)` | `w_s_cv (Ridge)` |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for r in inter:
                ws_val = r["physics_v2"]["ws_star"] if vk == "v2_alg" else (
                    r["physics_v1"]["ws_star_alg"] if vk == "v1_alg" else r["physics_v1"]["ws_star_grid"]
                )
                lines.append(
                    f"| `{r['task_id']}` | {r['suite']} | {r['n_cal']} | {ws_val:.4f} | "
                    f"{r['pls_cal_config']['ws_inner_cv']['ws']:.1f} | {r['ridge_cal_config']['ws_inner_cv']['ws']:.1f} |"
                )
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t_start = time.time()
    theory_path = ROOT / "chemdl" / "theory.py"
    wt_lf = file_sha256_lf(theory_path)
    assert wt_lf == FROZEN_THEORY_LF_SHA256, f"chemdl/theory.py LF SHA-256 mismatch: {wt_lf}"
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    blob_lf = git_blob_sha256(ROOT, git_head, "chemdl/theory.py", lf_normalize=True)
    assert blob_lf == FROZEN_THEORY_LF_SHA256

    raw_tasks = load_ws_value_tasks(ROOT)
    # Strip all test arrays BEFORE entering cal-only estimation!
    cal_only_tasks = [
        {
            "task_id": t["task_id"],
            "suite": t["suite"],
            "domain": t["domain"],
            "split_rule": t["split_rule"],
            "target_name": t["target_name"],
            "n_cal": t["n_cal"],
            "n_test": t["n_test"],
            "L": t["L"],
            "split_sha256": t["split_sha256"],
            "has_label_scale_anomaly": t["has_label_scale_anomaly"],
            "X_cal": t["X_cal"],
            "y_cal": t["y_cal"],
            "groups_cal": t["groups_cal"],
        }
        for t in raw_tasks
    ]
    del raw_tasks

    locked_tasks = []
    for i, mc in enumerate(cal_only_tasks, 1):
        row = _lock_cal_only_task(mc)
        locked_tasks.append(row)
        print(
            f"[{i:02d}/54] {row['task_id']:42s} | "
            f"v2={row['physics_v2']['ws_star']:.3f} ({row['pre_tags']['v2_alg']['regime']:10s}) | "
            f"v1a={row['physics_v1']['ws_star_alg']:.3f} | "
            f"v1g={row['physics_v1']['ws_star_grid']:.3f} | "
            f"cv_pls={row['pls_cal_config']['ws_inner_cv']['ws']:.1f}"
        )

    counts = {}
    for vk in ("v2_alg", "v1_alg", "v1_grid"):
        counts[vk] = {
            "interior_01": sum(1 for r in locked_tasks if r["pre_tags"][vk]["is_interior_01"]),
            "strict_interior": sum(1 for r in locked_tasks if r["pre_tags"][vk]["is_strict_interior"]),
            "boundary_0": sum(1 for r in locked_tasks if r["pre_tags"][vk]["regime"] == "BOUNDARY_0"),
            "boundary_1": sum(1 for r in locked_tasks if r["pre_tags"][vk]["regime"] == "BOUNDARY_1"),
            "interior_task_ids": [r["task_id"] for r in locked_tasks if r["pre_tags"][vk]["is_interior_01"]],
        }

    payload = _sanitize_json(
        {
            "phase": "WS_VALUE_PHASE_1_LOCKED",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_head_frozen": git_head,
            "theory_py_git_blob_sha256_lf": blob_lf,
            "fixed_wc": FIXED_WC,
            "ws_grid": list(WS_GRID),
            "interior_eps": INTERIOR_EPS,
            "n_tasks": len(locked_tasks),
            "elapsed_s": round(time.time() - t_start, 2),
            "summary": {"pre_tag_counts": counts},
            "tasks": locked_tasks,
        }
    )

    out_json = ROOT / "_audit" / "prereg_ws_closed_value_2026_11_25.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    out_md = ROOT / "reports" / "antigravity" / "PREREG_WS_CLOSED_VALUE_2026-11-25.md"
    _write_phase1_report(payload, out_md)
    print(f"\nPhase 1 locked {len(locked_tasks)} tasks in {payload['elapsed_s']}s -> {out_json}")
    for vk, c in counts.items():
        print(f"  {vk:8s}: interior_01={c['interior_01']}/54, strict_interior={c['strict_interior']}/54")


if __name__ == "__main__":
    main()
