"""Phase 2 揭盲评估：验证 Phase 1 锁单完整性并在 54 项独立留出任务上直接检验闭式 w_s* 的实际收益。

严格控制变量：
  - 固定 w_c = 1.0；
  - 固定回归头（主对照 PLS，伴随对照 Ridge）；
  - 固定 Phase 1 锁定的训练集、测试集与内层选参；
  - 严格按 Phase 1 仅用训练集预标定的 INTERIOR / BOUNDARY 标签分组报告：
      * 相对端点 w_s=0 的误差差值 (Δ_z, Δ%)
      * 相对端点 w_s=1 的误差差值 (Δ_z, Δ%)
      * 相对更优端点 min(w_s=0, w_s=1) 的误差差值 (Δ_z, Δ%)
      * 相对内层 CV 选强度 w_s_cv 的精度与校准耗时对比。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemdl.deploy_gate import FROZEN_THEORY_LF_SHA256

FIXED_WC = 1.0
WS_GRID = tuple(round(0.1 * i, 1) for i in range(11))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _file_sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_blob_sha256_lf(root: Path, rev: str, rel_path: str) -> str:
    try:
        raw = subprocess.check_output(["git", "show", f"{rev}:{rel_path}"], cwd=root, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raw = (root / rel_path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


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


def _safe_wilcoxon_p(diffs: list[float]) -> float | None:
    arr = np.asarray(diffs, dtype=np.float64)
    if len(arr) < 5 or np.allclose(arr, 0.0, atol=1e-12):
        return None
    try:
        res = wilcoxon(arr, alternative="two-sided", zero_method="wilcox")
        p = float(res.pvalue)
        return p if np.isfinite(p) else None
    except Exception:
        return None


def _compute_method_metrics(
    err_m: float,
    err_0: float,
    err_1: float,
    err_cv: float,
) -> dict:
    best_ep = min(err_0, err_1)
    best_ep_ws = 0.0 if err_0 <= err_1 else 1.0
    dz_0 = err_m - err_0
    dz_1 = err_m - err_1
    dz_best = err_m - best_ep
    dz_cv = err_m - err_cv
    rp_0 = (err_m / max(err_0, 1e-12) - 1.0) * 100.0
    rp_1 = (err_m / max(err_1, 1e-12) - 1.0) * 100.0
    rp_best = (err_m / max(best_ep, 1e-12) - 1.0) * 100.0
    rp_cv = (err_m / max(err_cv, 1e-12) - 1.0) * 100.0
    return {
        "rmsep_z": round(float(err_m), 6),
        "best_ep_z": round(float(best_ep), 6),
        "best_ep_ws": float(best_ep_ws),
        "diff_z_vs_ws0": round(float(dz_0), 6),
        "rel_pct_vs_ws0": round(float(rp_0), 3),
        "diff_z_vs_ws1": round(float(dz_1), 6),
        "rel_pct_vs_ws1": round(float(rp_1), 3),
        "diff_z_vs_best_ep": round(float(dz_best), 6),
        "rel_pct_vs_best_ep": round(float(rp_best), 3),
        "diff_z_vs_inner_cv": round(float(dz_cv), 6),
        "rel_pct_vs_inner_cv": round(float(rp_cv), 3),
        "beats_both_strict": bool(err_m < best_ep - 1e-12),
        "beats_both_0p5pct": bool(rp_best <= -0.5),
    }


def _aggregate_subset(rows: list[dict], head_key: str) -> dict:
    if not rows:
        return {"n": 0}
    methods = ("ws_0", "ws_1", "ws_closed_v2", "ws_closed_v1_alg", "ws_closed_v1_grid", "ws_inner_cv", "ws_oracle")
    out: dict = {"n": len(rows), "methods": {}}

    for m in methods:
        ez = [r[head_key]["metrics"][m]["rmsep_z"] for r in rows]
        dz0 = [r[head_key]["metrics"][m]["diff_z_vs_ws0"] for r in rows]
        rp0 = [r[head_key]["metrics"][m]["rel_pct_vs_ws0"] for r in rows]
        dz1 = [r[head_key]["metrics"][m]["diff_z_vs_ws1"] for r in rows]
        rp1 = [r[head_key]["metrics"][m]["rel_pct_vs_ws1"] for r in rows]
        dzb = [r[head_key]["metrics"][m]["diff_z_vs_best_ep"] for r in rows]
        rpb = [r[head_key]["metrics"][m]["rel_pct_vs_best_ep"] for r in rows]
        dzc = [r[head_key]["metrics"][m]["diff_z_vs_inner_cv"] for r in rows]
        rpc = [r[head_key]["metrics"][m]["rel_pct_vs_inner_cv"] for r in rows]

        cal_t = [r[head_key]["cal_times_s"].get(m, 0.0) for r in rows]
        win_best_05 = sum(1 for x in rpb if x <= -0.5)
        tie_best_05 = sum(1 for x in rpb if -0.5 < x < 0.5)
        loss_best_05 = sum(1 for x in rpb if x >= 0.5)
        severe_regret_5 = sum(1 for x in rpb if x >= 5.0)
        win_cv_05 = sum(1 for x in rpc if x <= -0.5)
        tie_cv_05 = sum(1 for x in rpc if -0.5 < x < 0.5)
        loss_cv_05 = sum(1 for x in rpc if x >= 0.5)

        out["methods"][m] = {
            "mean_rmsep_z": round(float(np.mean(ez)), 5),
            "median_rmsep_z": round(float(np.median(ez)), 5),
            "mean_diff_z_vs_ws0": round(float(np.mean(dz0)), 5),
            "median_diff_z_vs_ws0": round(float(np.median(dz0)), 5),
            "mean_rel_pct_vs_ws0": round(float(np.mean(rp0)), 2),
            "median_rel_pct_vs_ws0": round(float(np.median(rp0)), 2),
            "mean_diff_z_vs_ws1": round(float(np.mean(dz1)), 5),
            "median_diff_z_vs_ws1": round(float(np.median(dz1)), 5),
            "mean_rel_pct_vs_ws1": round(float(np.mean(rp1)), 2),
            "median_rel_pct_vs_ws1": round(float(np.median(rp1)), 2),
            "mean_diff_z_vs_best_ep": round(float(np.mean(dzb)), 5),
            "median_diff_z_vs_best_ep": round(float(np.median(dzb)), 5),
            "mean_rel_pct_vs_best_ep": round(float(np.mean(rpb)), 2),
            "median_rel_pct_vs_best_ep": round(float(np.median(rpb)), 2),
            "worst_rel_pct_vs_best_ep": round(float(np.max(rpb)), 2),
            "mean_diff_z_vs_inner_cv": round(float(np.mean(dzc)), 5),
            "median_rel_pct_vs_inner_cv": round(float(np.median(rpc)), 2),
            "mean_rel_pct_vs_inner_cv": round(float(np.mean(rpc)), 2),
            "beats_both_strict_count": int(sum(1 for r in rows if r[head_key]["metrics"][m]["beats_both_strict"])),
            "wtl_vs_best_ep_0p5pct": f"{win_best_05}/{tie_best_05}/{loss_best_05}",
            "wtl_vs_inner_cv_0p5pct": f"{win_cv_05}/{tie_cv_05}/{loss_cv_05}",
            "severe_regret_ge5pct_count": int(severe_regret_5),
            "wilcoxon_p_vs_ws0": _safe_wilcoxon_p(dz0),
            "wilcoxon_p_vs_ws1": _safe_wilcoxon_p(dz1),
            "wilcoxon_p_vs_best_ep": _safe_wilcoxon_p(dzb),
            "wilcoxon_p_vs_inner_cv": _safe_wilcoxon_p(dzc),
            "total_cal_time_s": round(float(np.sum(cal_t)), 3),
            "mean_cal_time_s": round(float(np.mean(cal_t)), 4),
        }

    t_cv = out["methods"]["ws_inner_cv"]["total_cal_time_s"]
    for ck in ("ws_closed_v2", "ws_closed_v1_alg", "ws_closed_v1_grid"):
        t_c = out["methods"][ck]["total_cal_time_s"]
        out["methods"][ck]["speedup_vs_inner_cv"] = round(float(t_cv / max(t_c, 1e-9)), 2)

    # Oracle interior availability check
    ora_gains = [r[head_key]["metrics"]["ws_oracle"]["rel_pct_vs_best_ep"] for r in rows]
    out["oracle_strictly_interior_beats_both_0p5pct"] = int(
        sum(
            1
            for r, g in zip(rows, ora_gains)
            if (0.01 < r[head_key]["ws_oracle"] < 0.99) and (g <= -0.5)
        )
    )
    return out


def _aggregate_regimes(rows: list[dict], head_key: str) -> dict:
    res = {}
    for vk, method_key in (
        ("v2_alg", "ws_closed_v2"),
        ("v1_alg", "ws_closed_v1_alg"),
        ("v1_grid", "ws_closed_v1_grid"),
    ):
        inter_01 = [r for r in rows if r["pre_tags"][vk]["is_interior_01"]]
        strict_i = [r for r in rows if r["pre_tags"][vk]["is_strict_interior"]]
        bound_all = [r for r in rows if not r["pre_tags"][vk]["is_interior_01"]]
        b0 = [r for r in rows if r["pre_tags"][vk]["regime"] == "BOUNDARY_0"]
        b1 = [r for r in rows if r["pre_tags"][vk]["regime"] == "BOUNDARY_1"]

        b_correct = sum(
            1
            for r in bound_all
            if (r["pre_tags"][vk]["regime"] == "BOUNDARY_0" and r[head_key]["metrics"][method_key]["best_ep_ws"] == 0.0)
            or (r["pre_tags"][vk]["regime"] == "BOUNDARY_1" and r[head_key]["metrics"][method_key]["best_ep_ws"] == 1.0)
        )
        res[vk] = {
            "INTERIOR_01": _aggregate_subset(inter_01, head_key),
            "STRICT_INTERIOR": _aggregate_subset(strict_i, head_key),
            "BOUNDARY_ALL": {
                **_aggregate_subset(bound_all, head_key),
                "endpoint_choice_acc": round(float(b_correct / len(bound_all)), 4) if bound_all else None,
                "endpoint_choice_hits": f"{b_correct}/{len(bound_all)}",
            },
            "BOUNDARY_0": _aggregate_subset(b0, head_key),
            "BOUNDARY_1": _aggregate_subset(b1, head_key),
        }
    return res


def _write_unblind_report(res: dict, out_md: Path) -> None:
    pls_all = res["summary"]["ALL_54"]["pls"]["overall"]
    pls_val = res["summary"]["VALID_53"]["pls"]["overall"]
    pls_ext = res["summary"]["EXTERNAL_30"]["pls"]["overall"]
    pls_reg_val = res["summary"]["VALID_53"]["pls"]["by_regime"]
    ridge_reg_val = res["summary"]["VALID_53"]["ridge"]["by_regime"]

    lines = [
        "# Phase 2 揭盲报告：闭式 $w_s^\\star$ 在 54 项独立留出任务上给使用者带来了什么",
        "",
        f"- **状态**：`{res['phase']}`",
        f"- **Phase 1 锁单 Commit**：`{res['locked_commit']}`",
        f"- **Phase 2 执行 Git HEAD**：`{res['execution_git_head']}`",
        f"- **Phase 1 JSON SHA-256 (LF)**：`{res['prereg_json_sha256_lf']}`",
        f"- **`chemdl/theory.py` LF SHA-256**：`{res['theory_py_git_blob_sha256_lf']}`",
        f"- **受控变量**：固定 `w_c = {res['fixed_wc']}`；主头 `PLS (kmax=12)`，伴随头 `Ridge (Exact Allen-PRESS / Group-CV)`；54 项独立单次留出 `cal -> test`（光谱字节重叠 `== 0`）。",
        "",
        "## 1. 仅用训练集预标定的“内部最优”激活率（54 项真实任务）",
        "",
        "| 理论分支 | 内部最优 (`0.01 < w_s* < 0.99`) | 严格内部 (`0 < w_s* < 1`) | 端点 `w_s*=0` | 端点 `w_s*=1` |",
        "|---|---:|---:|---:|---:|",
    ]
    for vk in ("v2_alg", "v1_alg", "v1_grid"):
        c = res["pre_tag_counts"][vk]
        lines.append(
            f"| `{vk}` | {c['interior_01']}/54 ({c['interior_01']/54:.1%}) | "
            f"{c['strict_interior']}/54 ({c['strict_interior']/54:.1%}) | {c['boundary_0']} | {c['boundary_1']} |"
        )

    lines.extend(["", "## 2. 核心检验：预标定为“内部最优”的任务在独立留出集上的逐项表现（主头 PLS）", ""])
    for vk, mkey, title in (
        ("v2_alg", "ws_closed_v2", "主理论 `v2_alg` (`R²_shape` 分解 + `σ_non=0` 纯闭式)"),
        ("v1_alg", "ws_closed_v1_alg", "纯闭式 `v1_alg` (`v1` 参数 + `σ_non=0`)"),
        ("v1_grid", "ws_closed_v1_grid", "网格分支 `v1_grid` (`v1` 参数 + `σ_non>0` 1001 点)"),
    ):
        inter = [r for r in res["tasks"] if r["pre_tags"][vk]["is_interior_01"]]
        lines.append(f"### 2.{1 if vk=='v2_alg' else (2 if vk=='v1_alg' else 3)} {title}（`n = {len(inter)}`）")
        lines.append("")
        if not inter:
            lines.append("- 无任务。\n")
            continue
        lines.append(
            "| Task ID | `w_s*` | `w_s_cv` | `ws=0` | `ws=1` | `ws_closed` | `ws_inner_cv` | `Δ vs ws=0` | `Δ vs ws=1` | `Δ vs 最优端点` | `Oracle(参考)` |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in inter:
            pm = r["pls"]["metrics"]
            ws_v = r["pls"]["configs"][mkey]["ws"]
            ws_cv = r["pls"]["configs"]["ws_inner_cv"]["ws"]
            e0 = pm["ws_0"]["rmsep_z"]
            e1 = pm["ws_1"]["rmsep_z"]
            ec = pm[mkey]["rmsep_z"]
            ev = pm["ws_inner_cv"]["rmsep_z"]
            eo = pm["ws_oracle"]["rmsep_z"]
            mc = pm[mkey]
            lines.append(
                f"| `{r['task_id']}` | {ws_v:.3f} | {ws_cv:.1f} | {e0:.4f} | {e1:.4f} | {ec:.4f} | {ev:.4f} | "
                f"{mc['diff_z_vs_ws0']:+.4f} ({mc['rel_pct_vs_ws0']:+.2f}%) | "
                f"{mc['diff_z_vs_ws1']:+.4f} ({mc['rel_pct_vs_ws1']:+.2f}%) | "
                f"{mc['diff_z_vs_best_ep']:+.4f} ({mc['rel_pct_vs_best_ep']:+.2f}%) | "
                f"{eo:.4f} (`w={r['pls']['ws_oracle']:.1f}`) |"
            )
        agg_i = res["summary"]["ALL_54"]["pls"]["by_regime"][vk]["INTERIOR_01"]["methods"][mkey]
        lines.append("")
        lines.append(
            f"> **小结（`{vk}` INTERIOR `n={len(inter)}`）**：相对更优端点均值差 `{agg_i['mean_diff_z_vs_best_ep']:+.5f}` "
            f"（`{agg_i['mean_rel_pct_vs_best_ep']:+.2f}%`，中位 `{agg_i['median_rel_pct_vs_best_ep']:+.2f}%`），"
            f"严格同时击败两端点 `{agg_i['beats_both_strict_count']}/{len(inter)}` 项；"
            f"相对 `ws_inner_cv` 均值差 `{agg_i['mean_diff_z_vs_inner_cv']:+.5f}`（`{agg_i['mean_rel_pct_vs_inner_cv']:+.2f}%`）。"
        )
        lines.append("")

    lines.extend(
        [
            "## 3. 分层汇总：内部子集 vs 端点子集 vs 全量（PLS 主头，排除 `DM_test2` 量纲倒置异常的 `VALID_53`）",
            "",
            "| 子集 / 方案 | `n` | 均值 `RMSEP_z` | 均值 `Δ vs ws=0` | 均值 `Δ vs ws=1` | 均值 `Δ% vs 最优端点` | 中位 `Δ% vs 最优端点` | 最坏 `Δ%` | `≥+5%` 严重遗憾 | 校准总耗时 (加速比) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sub_name, sub_dict in (
        ("VALID_53 (全部有效任务)", pls_val),
        ("EXTERNAL_30 (PhaseB+Round3)", pls_ext),
        ("ALL_54 (含 DM_test2 契约全量)", pls_all),
    ):
        lines.append(f"| **{sub_name}** | | | | | | | | | |")
        for mk, label in (
            ("ws_0", "不做尺度校正 (`ws=0`)"),
            ("ws_1", "完整尺度校正 (`ws=1`, SNV)"),
            ("ws_closed_v2", "闭式理论 (`v2_alg`, `σ_non=0`)"),
            ("ws_closed_v1_alg", "闭式理论 (`v1_alg`, `σ_non=0`)"),
            ("ws_closed_v1_grid", "网格理论 (`v1_grid`, `σ_non>0`)"),
            ("ws_inner_cv", "内层 CV 选强度 (`11 点`)"),
            ("ws_oracle", "Oracle 网格上界 (含泄漏参考)"),
        ):
            m = sub_dict["methods"][mk]
            spd = f"{m['total_cal_time_s']:.1f}s ({m.get('speedup_vs_inner_cv', 1.0):.1f}x)" if "speedup_vs_inner_cv" in m else f"{m['total_cal_time_s']:.1f}s"
            lines.append(
                f"| {label} | {sub_dict['n']} | {m['mean_rmsep_z']:.4f} | "
                f"{m['mean_diff_z_vs_ws0']:+.4f} | {m['mean_diff_z_vs_ws1']:+.4f} | "
                f"{m['mean_rel_pct_vs_best_ep']:+.2f}% | {m['median_rel_pct_vs_best_ep']:+.2f}% | "
                f"{m['worst_rel_pct_vs_best_ep']:+.2f}% | {m['severe_regret_ge5pct_count']} | {spd} |"
            )

    lines.extend(["", "## 4. 端点子集 (`BOUNDARY`) 与伴随 Ridge 头稳健性核查 (`VALID_53`)", ""])
    for vk, mk in (("v2_alg", "ws_closed_v2"), ("v1_alg", "ws_closed_v1_alg"), ("v1_grid", "ws_closed_v1_grid")):
        bp = pls_reg_val[vk]["BOUNDARY_ALL"]
        br = ridge_reg_val[vk]["BOUNDARY_ALL"]
        ir = ridge_reg_val[vk]["INTERIOR_01"]
        lines.append(
            f"- **`{vk}`**：PLS 端点开关命中 `{bp['endpoint_choice_hits']}`（`{bp['endpoint_choice_acc']:.1%}`），"
            f"Ridge 端点开关命中 `{br['endpoint_choice_hits']}`（`{br['endpoint_choice_acc']:.1%}`）；"
            f"Ridge 在预标定 `INTERIOR_01 (n={ir['n']})` 上相对更优端点均值 `{ir['methods'][mk]['mean_rel_pct_vs_best_ep']:+.2f}%`。"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 unblinding for closed-form w_s* direct value test")
    parser.add_argument("--verify-only", action="store_true", help="Verify Phase 1 lock & archived Phase 2 JSON.")
    args = parser.parse_args()

    t_start = time.time()
    pre_path = ROOT / "_audit" / "prereg_ws_closed_value_2026_11_25.json"
    res_path = ROOT / "_audit" / "res_ws_closed_value_2026_11_25.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    pre_sha = _file_sha256(pre_path)
    pre_sha_lf = _file_sha256_lf(pre_path)

    theory_path = ROOT / "chemdl" / "theory.py"
    wt_lf = _file_sha256_lf(theory_path)
    assert wt_lf == FROZEN_THEORY_LF_SHA256 == pre["theory_py_git_blob_sha256_lf"], (
        f"chemdl/theory.py LF SHA-256 mismatch: {wt_lf}"
    )

    locked_commit = str(pre["git_head_frozen"])
    has_git = (ROOT / ".git").exists() and (
        subprocess.run(
            ["git", "cat-file", "-e", f"{locked_commit}^{{commit}}"],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0
    )
    if has_git:
        blob_lf = _git_blob_sha256_lf(ROOT, locked_commit, "chemdl/theory.py")
        assert blob_lf == FROZEN_THEORY_LF_SHA256
        phase1_lock_commit = subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%H", "--diff-filter=A", "--", "_audit/prereg_ws_closed_value_2026_11_25.json"],
            cwd=ROOT,
            text=True,
        ).strip()
        assert phase1_lock_commit, "Cannot locate commit adding _audit/prereg_ws_closed_value_2026_11_25.json"
        assert _git_blob_sha256_lf(ROOT, phase1_lock_commit, "_audit/prereg_ws_closed_value_2026_11_25.json") == pre_sha_lf
        exec_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    else:
        phase1_lock_commit = "CLEAN_CLONE_NO_GIT"
        exec_head = "CLEAN_CLONE_NO_GIT"

    if args.verify_only:
        res = json.loads(res_path.read_text(encoding="utf-8"))
        assert res.get("phase") == "WS_VALUE_PHASE_2_UNBLINDED_VERIFIED"
        assert res.get("prereg_json_sha256_lf") == pre_sha_lf
        assert res.get("theory_py_git_blob_sha256_lf") == FROZEN_THEORY_LF_SHA256
        assert res.get("n_tasks") == pre["n_tasks"] == 54 and len(res["tasks"]) == 54
        assert res.get("pre_tag_counts") == pre["summary"]["pre_tag_counts"]
        assert all(t["cal_test_hash_overlap"] == 0 for t in res["tasks"])

        data_ext_dir = ROOT / "data_ext"
        if data_ext_dir.exists():
            from _audit._ws_closed_value_loaders import load_ws_value_tasks

            live_tasks = load_ws_value_tasks(ROOT)
            pre_by_id = {r["task_id"]: r for r in pre["tasks"]}
            for lt in live_tasks:
                assert lt["split_sha256"] == pre_by_id[lt["task_id"]]["split_sha256"]
                assert lt["cal_test_hash_overlap"] == 0
        print(
            f"[verify-only PASS] ws_closed_value parent={locked_commit[:7]}, lock={phase1_lock_commit[:7]}, exec={exec_head[:7]}, "
            f"54 tasks verified (data_ext={'present' if data_ext_dir.exists() else 'absent'})."
        )
        return

    from _audit._ws_closed_value_loaders import (
        fit_eval_pls_at_ws,
        fit_eval_ridge_at_ws,
        load_ws_value_tasks,
    )

    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "chemdl/theory.py", "_audit/prereg_ws_closed_value_2026_11_25.json"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert not dirty, f"Frozen Phase 1 files dirty: {dirty}"

    tasks = load_ws_value_tasks(ROOT)
    pre_by_id = {r["task_id"]: r for r in pre["tasks"]}
    assert len(tasks) == 54

    task_rows = []
    for i, t in enumerate(tasks, 1):
        tid = t["task_id"]
        p = pre_by_id[tid]
        assert t["split_sha256"] == p["split_sha256"], f"{tid}: split_sha256 mismatch"
        assert t["cal_test_hash_overlap"] == 0

        Xc, yc = t["X_cal"], t["y_cal"]
        Xt, yt = t["X_test"], t["y_test"]

        # --- Primary Head: PLS ---
        p_cfg = p["pls_cal_config"]
        pls_evals = {}
        for mk in ("ws_0", "ws_1", "ws_closed_v2", "ws_closed_v1_alg", "ws_closed_v1_grid", "ws_inner_cv"):
            cfg = p_cfg[mk]
            pls_evals[mk] = fit_eval_pls_at_ws(Xc, yc, Xt, yt, cfg["ws"], cfg["k"], wc=FIXED_WC)

        pls_grid_evals = [
            fit_eval_pls_at_ws(Xc, yc, Xt, yt, g["ws"], g["k"], wc=FIXED_WC)
            for g in p_cfg["grid_pls"]
        ]
        best_ora_pls = min(pls_grid_evals, key=lambda d: d["rmsep_z"])
        pls_evals["ws_oracle"] = best_ora_pls

        pls_metrics = {
            mk: _compute_method_metrics(
                pls_evals[mk]["rmsep_z"],
                pls_evals["ws_0"]["rmsep_z"],
                pls_evals["ws_1"]["rmsep_z"],
                pls_evals["ws_inner_cv"]["rmsep_z"],
            )
            for mk in pls_evals
        }

        # --- Companion Head: Ridge ---
        r_cfg = p["ridge_cal_config"]
        ridge_evals = {}
        for mk in ("ws_0", "ws_1", "ws_closed_v2", "ws_closed_v1_alg", "ws_closed_v1_grid", "ws_inner_cv"):
            cfg = r_cfg[mk]
            ridge_evals[mk] = fit_eval_ridge_at_ws(Xc, yc, Xt, yt, cfg["ws"], cfg["gamma"], wc=FIXED_WC)

        ridge_grid_evals = [
            fit_eval_ridge_at_ws(Xc, yc, Xt, yt, g["ws"], g["gamma"], wc=FIXED_WC)
            for g in r_cfg["grid_ridge"]
        ]
        best_ora_ridge = min(ridge_grid_evals, key=lambda d: d["rmsep_z"])
        ridge_evals["ws_oracle"] = best_ora_ridge

        ridge_metrics = {
            mk: _compute_method_metrics(
                ridge_evals[mk]["rmsep_z"],
                ridge_evals["ws_0"]["rmsep_z"],
                ridge_evals["ws_1"]["rmsep_z"],
                ridge_evals["ws_inner_cv"]["rmsep_z"],
            )
            for mk in ridge_evals
        }

        row = {
            "task_id": tid,
            "suite": t["suite"],
            "domain": t["domain"],
            "split_rule": t["split_rule"],
            "target_name": t["target_name"],
            "n_cal": t["n_cal"],
            "n_test": t["n_test"],
            "L": t["L"],
            "has_label_scale_anomaly": t["has_label_scale_anomaly"],
            "cal_test_hash_overlap": 0,
            "split_sha256": t["split_sha256"],
            "pre_tags": p["pre_tags"],
            "pls": {
                "configs": {mk: {"ws": pls_evals[mk]["ws"], "k": pls_evals[mk]["k"]} for mk in pls_evals},
                "ws_oracle": float(best_ora_pls["ws"]),
                "cal_times_s": {
                    mk: float(p_cfg[mk]["cal_time_s"])
                    for mk in ("ws_0", "ws_1", "ws_closed_v2", "ws_closed_v1_alg", "ws_closed_v1_grid", "ws_inner_cv")
                },
                "metrics": pls_metrics,
                "grid_evals": pls_grid_evals,
            },
            "ridge": {
                "configs": {mk: {"ws": ridge_evals[mk]["ws"], "gamma": ridge_evals[mk]["gamma"]} for mk in ridge_evals},
                "ws_oracle": float(best_ora_ridge["ws"]),
                "cal_times_s": {
                    mk: float(r_cfg[mk]["cal_time_s"])
                    for mk in ("ws_0", "ws_1", "ws_closed_v2", "ws_closed_v1_alg", "ws_closed_v1_grid", "ws_inner_cv")
                },
                "metrics": ridge_metrics,
                "grid_evals": ridge_grid_evals,
            },
        }
        task_rows.append(row)
        print(
            f"[{i:02d}/54] {tid:42s} | "
            f"ws0={pls_metrics['ws_0']['rmsep_z']:.4f} ws1={pls_metrics['ws_1']['rmsep_z']:.4f} | "
            f"v2={pls_metrics['ws_closed_v2']['rmsep_z']:.4f} ({pls_metrics['ws_closed_v2']['rel_pct_vs_best_ep']:+.2f}%) | "
            f"cv={pls_metrics['ws_inner_cv']['rmsep_z']:.4f} ({pls_metrics['ws_inner_cv']['rel_pct_vs_best_ep']:+.2f}%)"
        )

    universes = {
        "ALL_54": task_rows,
        "VALID_53": [r for r in task_rows if not r["has_label_scale_anomaly"]],
        "EXTERNAL_30": [r for r in task_rows if r["suite"] in ("Round3_Unseen", "PhaseB_Deploy") and not r["has_label_scale_anomaly"]],
        "Classic_23": [r for r in task_rows if r["suite"] == "Classic_Holdout"],
        "PhaseB_Valid_17": [r for r in task_rows if r["suite"] == "PhaseB_Deploy" and not r["has_label_scale_anomaly"]],
        "Round3_13": [r for r in task_rows if r["suite"] == "Round3_Unseen"],
    }

    summary = {}
    for uname, urows in universes.items():
        summary[uname] = {
            "pls": {"overall": _aggregate_subset(urows, "pls"), "by_regime": _aggregate_regimes(urows, "pls")},
            "ridge": {"overall": _aggregate_subset(urows, "ridge"), "by_regime": _aggregate_regimes(urows, "ridge")},
        }

    payload = _sanitize_json(
        {
            "phase": "WS_VALUE_PHASE_2_UNBLINDED_VERIFIED",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "locked_commit": locked_commit,
            "execution_git_head": exec_head,
            "prereg_json_sha256": pre_sha,
            "prereg_json_sha256_lf": pre_sha_lf,
            "theory_py_git_blob_sha256_lf": FROZEN_THEORY_LF_SHA256,
            "fixed_wc": FIXED_WC,
            "ws_grid": list(WS_GRID),
            "n_tasks": len(task_rows),
            "elapsed_s": round(time.time() - t_start, 2),
            "pre_tag_counts": pre["summary"]["pre_tag_counts"],
            "summary": summary,
            "tasks": task_rows,
        }
    )

    res_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    out_md = ROOT / "reports" / "antigravity" / "UNBLIND_WS_CLOSED_VALUE_2026-11-25.md"
    _write_unblind_report(payload, out_md)
    print(f"\nPhase 2 unblinded 54 tasks in {payload['elapsed_s']}s -> {res_path}")


if __name__ == "__main__":
    main()
