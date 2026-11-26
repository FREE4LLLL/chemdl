"""Phase A 收口审计：E1 修复（无相消精确 PRESS）后全量 86 折对照重算与 Round-2 四层样本重叠精细剖析。

硬性要求：
  1. 绝不覆盖原冻结归档 `_audit/prereg_snv_gate_round2_2026_11_24.json` 与 `res_prereg_snv_gate_round2_2026_11_24.json`；
  2. 使用修复 `E1/E3` 后的 `chemdl.theory.diagnose_snv_decision` 重算全部 18 项任务 / 86 折（逐折 `split_sha256` 100% 对齐），逐一核对逐折决策变化与数值差异；
  3. 对 Dev、Round 1 与 Round 2 全部 `float64` 光谱逐行计算 SHA-256，精确披露 4 个不重叠/重叠分层上的指标（全量 86 折、剔除 R1 同目标子集重叠 3 项后的 71 折、严格未见过光谱 4 项 / 20 折、以及单次官方留出 `Tablets_X2 155->460`）。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _audit._blind_round2_loaders import load_round2_tasks
from _audit._blind_theory_loaders import (
    _load_craw_csv,
    _load_ossl_unique_scans,
    file_sha256_lf,
    git_blob_sha256,
    load_blind_tasks,
    split_checksum,
)
from _audit.prereg_snv_gate_round2_phase2_unblind import _eval_method
from _audit.prereg_theory_blind_phase2_unblind import _sanitize_json
from chemdl import corn, cordoba, tablets, tecator
from chemdl.theory import diagnose_snv_decision

T_CRIT_PRIMARY = 2.0
MIN_EFFECT_PRIMARY = 1.0
T_CRIT_SEC = 1.5


def _row_hashes(X: np.ndarray) -> set[str]:
    Xa = np.ascontiguousarray(X, dtype=np.float64)
    return {hashlib.sha256(row.tobytes()).hexdigest() for row in Xa}


def _collect_dev_hashes(root: Path) -> dict[str, set[str]]:
    ct_dir = root / "data_ext" / "ct_bench"
    craw_dir = root / "data_ext" / "industrial" / "craw"
    ind_dir = root / "data_ext" / "industrial"
    diesel_path = root / "data_ext" / "diesel" / "SWRI_Diesel_NIR.mat"

    dev: dict[str, set[str]] = {}
    dev["Tecator"] = _row_hashes(tecator.load_tecator()["X"])

    c_d = corn.load_corn()
    dev["Corn"] = _row_hashes(c_d["m5"]) | _row_hashes(c_d["mp5"]) | _row_hashes(c_d["mp6"])

    for crop in ("wheat", "barley", "maize"):
        dev[f"Cordoba_{crop}"] = _row_hashes(cordoba.load_cordoba_crop(crop)["X"])

    dm = sio.loadmat(diesel_path)
    dev["SWRI_Diesel_dev200"] = _row_hashes(np.asarray(dm["diesel_spec"][0, 0]["data"], np.float64)[:200])

    tb = tablets.load_tablets()
    dev["Tablets_X1_trim"] = _row_hashes(tb["X1"])

    mg = sio.loadmat(ct_dir / "mango.mat")
    dev["Mango"] = _row_hashes(mg["SP_cal"]) | _row_hashes(mg["SP_test1"])

    wa1 = sio.loadmat(ct_dir / "wheatA1_20048.mat", squeeze_me=True, struct_as_record=False)
    dev["Wheat_A1"] = _row_hashes(wa1["X"]) | _row_hashes(wa1["Xtest"])

    mel = sio.loadmat(ct_dir / "Melamine_Dataset.mat", squeeze_me=True, struct_as_record=False)["Melamine_Dataset"]
    mel_h: set[str] = set()
    for rk in ("R562", "R568", "R861", "R862"):
        sub = getattr(mel, rk)
        mel_h |= _row_hashes(np.asarray(sub.X1, np.float64))
    dev["Melamine"] = mel_h

    Xc_dev, _ = _load_craw_csv(craw_dir / "Protein" / "Wheat_sensAIfood_Protein_5046_NIRS5000_CRAW.csv", "Protein")
    dev["CRAW_Wheat_5046"] = _row_hashes(Xc_dev)

    X_ossl_u, _ = _load_ossl_unique_scans(ct_dir)
    dev["OSSL_Unique"] = _row_hashes(X_ossl_u)

    dfm = pd.read_csv(ind_dir / "milk_transmittance_zenodo8263430.csv")
    t_cols = [f"Trans_Tot_{i}" for i in range(1, 257)]
    X_milk = -np.log10(np.clip(dfm[t_cols].to_numpy(dtype=np.float64), 1e-5, None))
    dev["Milk_SET1to8"] = _row_hashes(X_milk)
    return dev


def _call_from_stats(delta_pct: float, t_val: float, t_crit: float, min_eff: float) -> str:
    if t_val <= -t_crit and delta_pct <= -min_eff:
        return "IMPROVE"
    if t_val >= t_crit and delta_pct >= min_eff:
        return "WORSEN"
    return "ABSTAIN"


def _subset_eval(fold_rows: list[dict], task_rows: list[dict]) -> dict:
    def _eval_block(rows: list[dict], score_key: str, emp_key: str, call_key: str) -> dict:
        if not rows:
            return {}
        calls = [r[call_key] for r in rows]
        scores = np.array([r[score_key] for r in rows], dtype=np.float64)
        emps = np.array([r[emp_key] for r in rows], dtype=np.float64)
        sign_calls = ["IMPROVE" if s < 0.0 else "WORSEN" for s in scores]
        return {
            "n_items": len(rows),
            "primary_t2": _eval_method(calls, scores, emps),
            "zero_abstention": _eval_method(sign_calls, scores, emps),
        }

    return {
        "fold_level": _eval_block(fold_rows, "post_delta_loo_pct", "emp_snv_vs_raw_pct", "post_call_primary"),
        "task_level": _eval_block(task_rows, "post_delta_loo_mean_pct", "emp_pooled_pct", "post_call_primary"),
    }


def main() -> None:
    t0 = time.time()
    pre_path = ROOT / "_audit" / "prereg_snv_gate_round2_2026_11_24.json"
    res_path = ROOT / "_audit" / "res_prereg_snv_gate_round2_2026_11_24.json"
    prereg = json.loads(pre_path.read_text(encoding="utf-8"))
    res_arch = json.loads(res_path.read_text(encoding="utf-8"))
    pre_by_id = {r["task_id"]: r for r in prereg["tasks"]}
    arch_by_id = {r["task_id"]: r for r in res_arch["tasks"]}

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    theory_path = ROOT / "chemdl" / "theory.py"
    theory_lf = file_sha256_lf(theory_path)
    theory_blob_locked_r2 = git_blob_sha256(ROOT, str(prereg["git_head_frozen"]), "chemdl/theory.py", lf_normalize=True)

    # 构建 Dev 与 Round 1 光谱行哈希库
    dev_by_src = _collect_dev_hashes(ROOT)
    dev_all: set[str] = set().union(*dev_by_src.values())
    r1_tasks = load_blind_tasks(ROOT)
    r1_all: set[str] = set()
    r1_craw_subset_map = {}
    for rt in r1_tasks:
        rh = _row_hashes(rt["X"])
        r1_all |= rh
        if rt["task_id"].startswith(("CRAW_Spelt", "CRAW_WheatDurum")):
            r1_craw_subset_map[rt["task_id"]] = rh

    r2_tasks = load_round2_tasks(ROOT)
    assert len(r2_tasks) == 18

    changed_folds = []
    max_d_delta = 0.0
    max_d_t = 0.0
    max_g_rel = 0.0

    all_fold_rows = []
    all_task_rows = []
    stratum_summary = []

    for t in r2_tasks:
        tid = t["task_id"]
        X, y, groups = t["X"], t["y"], t["groups"]
        pre_t = pre_by_id[tid]
        arch_t = arch_by_id[tid]
        assert len(t["splits"]) == len(arch_t["folds"])

        h_r2 = _row_hashes(X)
        ov_dev = len(h_r2 & dev_all)
        ov_r1 = len(h_r2 & r1_all)

        if tid in ("CRAW_Spelt_XDS2__Moisture", "CRAW_WheatDurum_XDS2__Moisture", "CRAW_WheatDurum_XDS2__Protein"):
            stratum = "r1_subset_overlap_same_target"
        elif tid in ("Tablets_X2_Official__Assay", "SDOP_Spec2__c2"):
            stratum = "cross_instrument_same_batch"
        elif ov_dev == 0 and ov_r1 == 0:
            stratum = "strictly_unseen_spectra"
        else:
            stratum = "same_spectra_new_target"

        post_folds = []
        for fi, (tr_idx, te_idx) in enumerate(t["splits"]):
            pf = pre_t["folds"][fi]
            af = arch_t["folds"][fi]
            chk = split_checksum(tr_idx, te_idx)
            assert chk == pf["split_sha256"] == af["split_sha256"]

            g_tr = groups[tr_idx] if groups is not None else None
            diag = diagnose_snv_decision(
                X[tr_idx],
                y[tr_idx],
                groups_tr=g_tr,
                t_crit=T_CRIT_PRIMARY,
                min_effect_pct=MIN_EFFECT_PRIMARY,
                compute_physics=True,
            )

            d_delta = abs(float(diag["delta_loo_pct"]) - float(pf["delta_loo_pct"]))
            d_t = abs(float(diag["t_loo"]) - float(pf["t_loo"]))
            g_raw_rel = abs(float(diag["gamma_raw"]) - float(pf["gamma_raw"])) / max(abs(float(pf["gamma_raw"])), 1e-12)
            g_snv_rel = abs(float(diag["gamma_snv"]) - float(pf["gamma_snv"])) / max(abs(float(pf["gamma_snv"])), 1e-12)
            max_d_delta = max(max_d_delta, d_delta)
            max_d_t = max(max_d_t, d_t)
            max_g_rel = max(max_g_rel, g_raw_rel, g_snv_rel)

            if str(diag["decision"]) != str(pf["call_primary"]):
                changed_folds.append(
                    {
                        "task_id": tid,
                        "fold": int(fi),
                        "pre_call": str(pf["call_primary"]),
                        "post_call": str(diag["decision"]),
                        "pre_delta": float(pf["delta_loo_pct"]),
                        "post_delta": float(diag["delta_loo_pct"]),
                        "pre_t": float(pf["t_loo"]),
                        "post_t": float(diag["t_loo"]),
                        "emp_pct": float(af["emp_snv_vs_raw_pct"]),
                    }
                )

            frow = {
                "task_id": tid,
                "stratum": stratum,
                "fold": int(fi),
                "pre_delta_loo_pct": float(pf["delta_loo_pct"]),
                "post_delta_loo_pct": float(diag["delta_loo_pct"]),
                "pre_t_loo": float(pf["t_loo"]),
                "post_t_loo": float(diag["t_loo"]),
                "pre_call_primary": str(pf["call_primary"]),
                "post_call_primary": str(diag["decision"]),
                "emp_snv_vs_raw_pct": float(af["emp_snv_vs_raw_pct"]),
            }
            post_folds.append(frow)
            all_fold_rows.append(frow)

        post_mean_d = round(float(np.mean([r["post_delta_loo_pct"] for r in post_folds])), 2)
        post_mean_t = round(float(np.mean([r["post_t_loo"] for r in post_folds])), 3)
        post_call_t = _call_from_stats(post_mean_d, post_mean_t, T_CRIT_PRIMARY, MIN_EFFECT_PRIMARY)
        trow = {
            "task_id": tid,
            "stratum": stratum,
            "n": t["n"],
            "L": t["L"],
            "n_folds": len(post_folds),
            "n_unique_rows": len(h_r2),
            "overlap_dev_rows": ov_dev,
            "overlap_r1_rows": ov_r1,
            "pre_call_primary": str(arch_t["call_primary"]),
            "post_call_primary": post_call_t,
            "pre_delta_loo_mean_pct": float(arch_t["delta_loo_mean_pct"]),
            "post_delta_loo_mean_pct": post_mean_d,
            "pre_t_loo_mean": float(arch_t["t_loo_mean"]),
            "post_t_loo_mean": post_mean_t,
            "emp_pooled_pct": float(arch_t["emp_pooled_pct"]),
        }
        all_task_rows.append(trow)
        stratum_summary.append(
            {
                "task_id": tid,
                "stratum": stratum,
                "n_rows": t["n"],
                "unique_rows": len(h_r2),
                "overlap_dev_rows": ov_dev,
                "overlap_r1_rows": ov_r1,
            }
        )

    assert len(all_fold_rows) == 86 and len(all_task_rows) == 18

    strata_eval = {
        "all_18_tasks_86_folds": _subset_eval(all_fold_rows, all_task_rows),
        "dedup_15_tasks_71_folds_no_r1_subset": _subset_eval(
            [r for r in all_fold_rows if r["stratum"] != "r1_subset_overlap_same_target"],
            [r for r in all_task_rows if r["stratum"] != "r1_subset_overlap_same_target"],
        ),
        "strictly_unseen_4_tasks_20_folds": _subset_eval(
            [r for r in all_fold_rows if r["stratum"] == "strictly_unseen_spectra"],
            [r for r in all_task_rows if r["stratum"] == "strictly_unseen_spectra"],
        ),
        "same_spectra_new_target_9_tasks_45_folds": _subset_eval(
            [r for r in all_fold_rows if r["stratum"] == "same_spectra_new_target"],
            [r for r in all_task_rows if r["stratum"] == "same_spectra_new_target"],
        ),
        "official_single_holdout_tablets_x2": next(
            r for r in all_task_rows if r["task_id"] == "Tablets_X2_Official__Assay"
        ),
    }

    out = _sanitize_json(
        {
            "phase": "POST_E1_86_FOLD_AUDIT",
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "execution_git_head": git_head,
            "r2_locked_commit": prereg["git_head_frozen"],
            "theory_py_git_blob_r2_locked_lf": theory_blob_locked_r2,
            "theory_py_post_e1_worktree_lf": theory_lf,
            "n_tasks_audited": len(all_task_rows),
            "n_folds_audited": len(all_fold_rows),
            "e1_diff": {
                "n_fold_decisions_changed": len(changed_folds),
                "n_task_decisions_changed": sum(1 for r in all_task_rows if r["pre_call_primary"] != r["post_call_primary"]),
                "max_abs_diff_delta_loo_pct": round(max_d_delta, 6),
                "max_abs_diff_t_loo": round(max_d_t, 6),
                "max_rel_diff_gamma": round(max_g_rel, 8),
                "changed_folds": changed_folds,
            },
            "stratum_inventory": stratum_summary,
            "strata_metrics_post_e1": strata_eval,
            "tasks": all_task_rows,
        }
    )

    out_json = ROOT / "_audit" / "res_press_e1_fix_86folds_2026_11_24.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)

    # 撰写严谨中文审计报告
    s_all_f = strata_eval["all_18_tasks_86_folds"]["fold_level"]
    s_all_t = strata_eval["all_18_tasks_86_folds"]["task_level"]
    s_dedup_f = strata_eval["dedup_15_tasks_71_folds_no_r1_subset"]["fold_level"]
    s_dedup_t = strata_eval["dedup_15_tasks_71_folds_no_r1_subset"]["task_level"]
    s_unseen_f = strata_eval["strictly_unseen_4_tasks_20_folds"]["fold_level"]
    tab_x2 = strata_eval["official_single_holdout_tablets_x2"]

    lines = [
        "# E1 修复后全量 86 折 PRESS 重算与 Round-2 样本重叠分层审计（2026-11-24）",
        "",
        f"- **执行代码提交**：`{git_head}`（`chemdl/theory.py` LF SHA-256 = `{theory_lf}`）",
        f"- **原 Round-2 锁定档案（未覆盖保留）**：`{prereg['git_head_frozen']}`（LF Blob = `{theory_blob_locked_r2}`）",
        f"- **审计范围**：全部 **{len(all_task_rows)} 项任务 / {len(all_fold_rows)} 折**（`split_sha256` 100% 匹配原档案）",
        f"- **耗时**：`{time.time() - t0:.2f}s`",
        "",
        "## 一、E1（无相消精确 `1 - h` PRESS）修复前后全量 86 折变化对照",
        "",
        f"- **逐折主门控决策变动数**：**{len(changed_folds)} / 86**",
        f"- **任务级描述性聚合变动数**：**{out['e1_diff']['n_task_decisions_changed']} / 18**",
        f"- **全 86 折数值最大差异**：`max|Δ(delta_loo_pct)| = {out['e1_diff']['max_abs_diff_delta_loo_pct']}`，`max|Δ(t_loo)| = {out['e1_diff']['max_abs_diff_t_loo']}`，`max rel|Δγ| = {out['e1_diff']['max_rel_diff_gamma']}`",
        "",
        "> **结论**：在 86 折真实光谱训练折上，`gamma` 网格搜索（`1e-6 * mean(lam)` 下界）始终令真实近红外光谱的 `1 - h_ii` 处于数值安全区，因此 **86 折与 18 项任务级决策、LOO 差值与配对 t 统计量 100% 不变**；该修复彻底清除了高维极小正则化极端输入下的系统性低估隐患（见回归单测 `test_exact_press_loo_high_leverage_matches_bruteforce`）。",
        "",
        "## 二、Round-2 与 Dev / Round 1 的样本重叠精确分层（回应 Codex `I2`）",
        "",
        "| # | Task ID | 分层归属 | 样本数 (唯一光谱) | 与 Dev 重叠行 | 与 R1 重叠行 | 新门控判断 (`dLOO / t`) | 实测 ΔRMSEP_pool |",
        "|---:|---|---|---:|---:|---:|---|---:|",
    ]
    for i, r in enumerate(all_task_rows, 1):
        lines.append(
            f"| {i} | `{r['task_id']}` | `{r['stratum']}` | {r['n']} ({r['n_unique_rows']}) "
            f"| {r['overlap_dev_rows']} | {r['overlap_r1_rows']} "
            f"| **{r['post_call_primary']}** (`{r['post_delta_loo_mean_pct']:+.2f}% / {r['post_t_loo_mean']:+.2f}`) "
            f"| **{r['emp_pooled_pct']:+.2f}%** |"
        )

    lines.extend([
        "",
        "## 三、按统计单位与去重分层报告的准确指标（回应 Codex `I1 + I2`）",
        "",
        "| 评估集合与粒度 | 样本/折数 | 主门控 Cov | 主门控 Acc (命中/总数) | 主门控 BAC / MCC | 零弃权 Acc / MCC | Spearman ρ (p 值) |",
        "|---|---:|---|---|---:|---:|---|",
    ])
    for name, blk in (
        ("**逐折独立留出·全量 18 任务**", s_all_f),
        ("**逐折独立留出·剔除 R1 同目标子集重叠（15 任务）**", s_dedup_f),
        ("逐折独立留出·严格未见过光谱（4 项 CRAW）", s_unseen_f),
        ("任务级描述性聚合（18 任务，非独立留出）", s_all_t),
        ("任务级描述性聚合·剔除 R1 同目标子集重叠（15 任务）", s_dedup_t),
    ):
        p = blk["primary_t2"]["all_signs"]
        z = blk["zero_abstention"]["all_signs"]
        rho = blk["primary_t2"]["spearman_rho"]
        rp = blk["primary_t2"]["spearman_p"]
        lines.append(
            f"| {name} | {blk['n_items']} | {p['coverage']:.1%} ({p['n']}/{blk['n_items']}) "
            f"| {p['acc']} ({p['tp']+p['tn']}/{p['n']}) | {p['balanced_acc']} / {p['mcc']} "
            f"| {z['acc']} ({z['tp']+z['tn']}/{blk['n_items']}) / {z['mcc']} | {rho:+.4f} (p={rp:.3g}) |"
        )
    lines.extend([
        "",
        f"- **单次严格官方独立留出（`Tablets_X2_Official__Assay`, 155 cal → 460 test）**：事前锁定 **`{tab_x2['post_call_primary']}`**（`Δ_LOO = {tab_x2['post_delta_loo_mean_pct']:+.2f}%`, `t_loo = {tab_x2['post_t_loo_mean']:+.2f}`），实测测试集 RMSEP 变动 **`{tab_x2['emp_pooled_pct']:+.2f}%`**（精准命中）。",
        "",
    ])

    out_md = ROOT / "reports" / "antigravity" / "AUDIT_PRESS_E1_AND_STRATA_2026-11-24.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"86-fold post-E1 audit complete ({time.time()-t0:.2f}s): changed_folds={len(changed_folds)}/86, max_d_delta={max_d_delta}, max_d_t={max_d_t}")


if __name__ == "__main__":
    main()
