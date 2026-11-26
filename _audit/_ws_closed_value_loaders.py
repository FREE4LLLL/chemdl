"""闭式 w_s* 独立留出直接价值检验的共享加载器与受控固定头辅助函数。

严格控制变量：
  1. 固定 w_c = 1.0（逐样本去均值固定，仅变乘性尺度收缩 w_s ∈ [0, 1]）；
  2. 固定回归头（主对照：PLS，统一内层折与 kmax=12；辅敏度对照：精确 Allen-PRESS / Group-CV Ridge）；
  3. 固定训练/独立留出划分（54 项单次 cal -> test 真实 NIR 任务：Round 3 13 + Phase B 18 + Classic 23）；
  4. 严格零泄漏：内层 CV 折内独立估计 s_bar_itr，绝不接触测试集与外层验证折。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, KFold

from _audit._deploy_holdout_loaders import _row_hashes, _split_sha256_arrays, load_deploy_holdout_tasks
from _audit._round3_unseen_loaders import _stratified_70_30_split, load_round3_unseen_tasks
from chemdl.cordoba import load_cordoba_all
from chemdl.corn import load_corn
from chemdl.diesel import load_diesel
from chemdl.tablets import load_tablets
from chemdl.tecator import load_tecator
from chemdl.theory import exact_press_loo_residuals
from chemdl.transfer import dssnv_apply, rmsep

FIXED_WC = 1.0
WS_GRID = tuple(round(0.1 * i, 1) for i in range(11))
SPLIT_SEED_CLASSIC = 20261125
INTERIOR_EPS = 0.01
ANOMALY_TASK_IDS = {"Deploy_Mango_Official_SP_test2__DM"}


def classify_ws_regime(ws: float, eps: float = INTERIOR_EPS) -> dict:
    """按给定阈值分类 w_s* 的机制区段（端点 vs 严格内部）。"""
    w = float(ws)
    if w <= eps:
        regime = "BOUNDARY_0"
    elif w >= 1.0 - eps:
        regime = "BOUNDARY_1"
    else:
        regime = "INTERIOR"
    return {
        "regime": regime,
        "is_interior_01": bool(eps < w < 1.0 - eps),
        "is_strict_interior": bool(0.0 < w < 1.0),
    }


def load_ws_value_tasks(root: Path | None = None) -> list[dict]:
    """加载全部 54 项单次独立留出任务（Round 3 13 + Phase B 18 + Classic 23）。"""
    root = Path(__file__).resolve().parent.parent if root is None else Path(root)
    tasks: list[dict] = []

    # Suite 1: Round 3 Unseen External Zenodo (13 tasks)
    for t in load_round3_unseen_tasks(root):
        Xc = np.asarray(t["X_cal"], dtype=np.float64)
        yc = np.asarray(t["y_cal"], dtype=np.float64).ravel()
        Xt = np.asarray(t["X_test"], dtype=np.float64)
        yt = np.asarray(t["y_test"], dtype=np.float64).ravel()
        gc = None if t.get("groups_cal") is None else np.asarray(t["groups_cal"])
        assert len(set(_row_hashes(Xc)) & set(_row_hashes(Xt))) == 0
        tasks.append(
            {
                "task_id": t["task_id"],
                "suite": "Round3_Unseen",
                "domain": t["lab_id"],
                "split_rule": t["split_rule"],
                "target_name": t["target_name"],
                "n_cal": int(len(Xc)),
                "n_test": int(len(Xt)),
                "L": int(Xc.shape[1]),
                "cal_test_hash_overlap": 0,
                "split_sha256": t["split_sha256"],
                "has_label_scale_anomaly": t["task_id"] in ANOMALY_TASK_IDS,
                "X_cal": Xc,
                "y_cal": yc,
                "groups_cal": gc,
                "X_test": Xt,
                "y_test": yt,
                "groups_test": None,
            }
        )

    # Suite 2: Phase B Deploy Holdout (18 tasks)
    for t in load_deploy_holdout_tasks(root):
        Xc = np.asarray(t["X_cal"], dtype=np.float64)
        yc = np.asarray(t["y_cal"], dtype=np.float64).ravel()
        Xt = np.asarray(t["X_test"], dtype=np.float64)
        yt = np.asarray(t["y_test"], dtype=np.float64).ravel()
        gc = None if t.get("groups_cal") is None else np.asarray(t["groups_cal"])
        gt = None if t.get("groups_test") is None else np.asarray(t["groups_test"])
        assert len(set(_row_hashes(Xc)) & set(_row_hashes(Xt))) == 0
        tasks.append(
            {
                "task_id": t["task_id"],
                "suite": "PhaseB_Deploy",
                "domain": t["domain"],
                "split_rule": t["split_rule"],
                "target_name": t["target_name"],
                "n_cal": int(len(Xc)),
                "n_test": int(len(Xt)),
                "L": int(Xc.shape[1]),
                "cal_test_hash_overlap": 0,
                "split_sha256": t["split_sha256"],
                "has_label_scale_anomaly": t["task_id"] in ANOMALY_TASK_IDS,
                "X_cal": Xc,
                "y_cal": yc,
                "groups_cal": gc,
                "X_test": Xt,
                "y_test": yt,
                "groups_test": gt,
            }
        )

    # Suite 3: Classic Holdout (23 tasks)
    def _add_classic(
        task_id: str,
        domain: str,
        split_rule: str,
        target_name: str,
        Xc: np.ndarray,
        yc: np.ndarray,
        Xt: np.ndarray,
        yt: np.ndarray,
        groups_cal: np.ndarray | None = None,
    ) -> None:
        Xc = np.asarray(Xc, dtype=np.float64)
        yc = np.asarray(yc, dtype=np.float64).ravel()
        Xt = np.asarray(Xt, dtype=np.float64)
        yt = np.asarray(yt, dtype=np.float64).ravel()
        assert len(Xc) >= 25 and len(Xt) >= 15
        assert Xc.ndim == 2 and Xt.ndim == 2 and Xc.shape[1] == Xt.shape[1]
        assert np.isfinite(Xc).all() and np.isfinite(yc).all()
        assert np.isfinite(Xt).all() and np.isfinite(yt).all()
        ov = len(set(_row_hashes(Xc)) & set(_row_hashes(Xt)))
        assert ov == 0, f"{task_id}: cal/test overlap={ov}"
        tasks.append(
            {
                "task_id": task_id,
                "suite": "Classic_Holdout",
                "domain": domain,
                "split_rule": split_rule,
                "target_name": target_name,
                "n_cal": int(len(Xc)),
                "n_test": int(len(Xt)),
                "L": int(Xc.shape[1]),
                "cal_test_hash_overlap": 0,
                "split_sha256": _split_sha256_arrays(Xc, yc, Xt, yt),
                "has_label_scale_anomaly": task_id in ANOMALY_TASK_IDS,
                "X_cal": Xc,
                "y_cal": yc,
                "groups_cal": groups_cal,
                "X_test": Xt,
                "y_test": yt,
                "groups_test": None,
            }
        )

    tec = load_tecator()
    for j, col in enumerate(tec["cols"]):
        tr, te = _stratified_70_30_split(tec["X"], tec["y"][:, j], seed=SPLIT_SEED_CLASSIC + j)
        gc_tec = np.asarray(_row_hashes(tec["X"][tr]))
        _add_classic(
            f"Classic_Tecator_70_30__{col}",
            "Tecator Meat NIR (100 ch)",
            "QuantileStratified_70_30_seed20261125_group_dedup",
            str(col),
            tec["X"][tr],
            tec["y"][tr, j],
            tec["X"][te],
            tec["y"][te, j],
            groups_cal=gc_tec,
        )

    tab = load_tablets(trim_saturated=True)
    m_tab_cal = tab["part"] == 0
    m_tab_te = tab["part"] == 1
    for j, col in enumerate(tab["cols"]):
        _add_classic(
            f"Classic_TabletsX1_Official__{col}",
            "IDRC 2002 Tablets X1 [35:540]",
            "Official_IDRC2002_X1_cal155_to_test460",
            str(col),
            tab["X1"][m_tab_cal],
            tab["y"][m_tab_cal, j],
            tab["X1"][m_tab_te],
            tab["y"][m_tab_te, j],
        )

    corn = load_corn()
    for j, col in enumerate(corn["cols"]):
        tr, te = _stratified_70_30_split(corn["m5"], corn["y"][:, j], seed=SPLIT_SEED_CLASSIC + j)
        _add_classic(
            f"Classic_Corn_m5_70_30__{col}",
            "Cargill Corn m5 (700 ch)",
            "QuantileStratified_70_30_seed20261125",
            str(col),
            corn["m5"][tr],
            corn["y"][tr, j],
            corn["m5"][te],
            corn["y"][te, j],
        )

    cord = load_cordoba_all()
    for crop in ("barley", "maize", "wheat"):
        d = cord[crop]
        for j, col in enumerate(d["cols"]):
            ok = np.isfinite(d["X"]).all(axis=1) & np.isfinite(d["y"][:, j])
            Xo, yo, yro = d["X"][ok], d["y"][ok, j], d["years"][ok]
            if crop == "wheat":
                mc = yro == 1998
                mt = yro == 1999
                _add_classic(
                    f"Classic_Cordoba_wheat_98to99__{col}",
                    "Cordoba Wheat NIRS5000",
                    "Natural_Temporal_1998_to_1999",
                    str(col),
                    Xo[mc],
                    yo[mc],
                    Xo[mt],
                    yo[mt],
                )
            else:
                tr, te = _stratified_70_30_split(Xo, yo, seed=SPLIT_SEED_CLASSIC + j)
                _add_classic(
                    f"Classic_Cordoba_{crop}_70_30__{col}",
                    f"Cordoba {crop.title()} NIRS5000",
                    "QuantileStratified_70_30_seed20261125",
                    str(col),
                    Xo[tr],
                    yo[tr],
                    Xo[te],
                    yo[te],
                )

    dsl = load_diesel()
    for j, col in enumerate(dsl["cols"]):
        ok = np.isfinite(dsl["X"]).all(axis=1) & np.isfinite(dsl["y"][:, j])
        Xo, yo = dsl["X"][ok], dsl["y"][ok, j]
        tr, te = _stratified_70_30_split(Xo, yo, seed=SPLIT_SEED_CLASSIC + j)
        _add_classic(
            f"Classic_Diesel_70_30__{col}",
            "SWRI Diesel NIR (401 ch)",
            "QuantileStratified_70_30_seed20261125",
            str(col),
            Xo[tr],
            yo[tr],
            Xo[te],
            yo[te],
        )

    # Cross-suite zero spectrum hash overlap checks
    cord_hashes: set[str] = set()
    craw_hashes: set[str] = set()
    tab_x1_hashes: set[str] = set()
    tab_x2_hashes: set[str] = set()
    for t in tasks:
        hs = set(_row_hashes(t["X_cal"])) | set(_row_hashes(t["X_test"]))
        if t["task_id"].startswith("Classic_Cordoba_"):
            cord_hashes |= hs
        elif t["task_id"].startswith("Deploy_CRAW_"):
            craw_hashes |= hs
        elif t["task_id"].startswith("Classic_TabletsX1_"):
            tab_x1_hashes |= hs
        elif t["task_id"].startswith("Deploy_Tablets_X2_"):
            tab_x2_hashes |= hs
    assert len(cord_hashes & craw_hashes) == 0, "Cordoba vs Deploy_CRAW spectrum overlap!"
    assert len(tab_x1_hashes & tab_x2_hashes) == 0, "Tablets X1 vs Tablets X2 spectrum overlap!"
    assert len(tasks) == 54, f"Expected 54 tasks, got {len(tasks)}"
    return tasks


def make_inner_splits(
    n_cal: int,
    groups_cal: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """构造严格只用校准集的内层 CV 划分（有分组时优先使用 GroupKFold）。"""
    if groups_cal is not None:
        grp = np.asarray(groups_cal)
        uniq = np.unique(grp)
        eff = min(int(n_splits), len(uniq))
        if eff >= 2:
            splits = [
                (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
                for tr, va in GroupKFold(n_splits=eff).split(np.zeros((n_cal, 1)), grp, grp)
            ]
            if min(len(tr) for tr, _ in splits) >= 4:
                return splits
    eff_k = min(int(n_splits), max(2, n_cal // 5))
    return [
        (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
        for tr, va in KFold(n_splits=eff_k, shuffle=True, random_state=seed).split(np.zeros((n_cal, 1)))
    ]


def pls_prefix_predictions(
    Xtr: np.ndarray,
    ytr_1d: np.ndarray,
    Xva: np.ndarray,
    kmax: int,
) -> list[np.ndarray]:
    """单次 NIPALS 拟合提取 k=1..kmax 全部前缀预测；奇异或数值异常时回退逐 k 拟合。"""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    ytr_1d = np.asarray(ytr_1d, dtype=np.float64).ravel()
    Xva = np.asarray(Xva, dtype=np.float64)
    eff_k = min(int(kmax), Xtr.shape[1], max(1, Xtr.shape[0] - 1))

    m_full = PLSRegression(n_components=eff_k, scale=False).fit(Xtr, ytr_1d[:, None])
    W = np.asarray(m_full.x_weights_, dtype=np.float64)
    P = np.asarray(m_full.x_loadings_, dtype=np.float64)
    Q = np.asarray(m_full.y_loadings_, dtype=np.float64)
    x_mean = np.asarray(m_full._x_mean, dtype=np.float64)
    y_mean = np.asarray(m_full._y_mean, dtype=np.float64)
    Xc = Xva - x_mean

    preds: list[np.ndarray] = []
    if W.shape[1] >= eff_k:
        for k in range(1, eff_k + 1):
            Wk, Pk, Qk = W[:, :k], P[:, :k], Q[:, :k]
            gram = Pk.T @ Wk
            try:
                cond = float(np.linalg.cond(gram))
                if not np.isfinite(cond) or cond > 1e10:
                    raise np.linalg.LinAlgError("ill-conditioned")
                Rk = Wk @ np.linalg.pinv(gram)
                pk = (Xc @ (Rk @ Qk.T) + y_mean).ravel()
                if not np.all(np.isfinite(pk)):
                    raise ValueError("non-finite")
                preds.append(pk)
            except Exception:
                mk = PLSRegression(n_components=k, scale=False).fit(Xtr, ytr_1d[:, None])
                preds.append(mk.predict(Xva).ravel())
    else:
        for k in range(1, eff_k + 1):
            mk = PLSRegression(n_components=k, scale=False).fit(Xtr, ytr_1d[:, None])
            preds.append(mk.predict(Xva).ravel())
    return preds


def select_k_pls_for_ws(
    X_cal: np.ndarray,
    yn_cal: np.ndarray,
    ws: float,
    splits: list[tuple[np.ndarray, np.ndarray]],
    kmax: int = 12,
    wc: float = FIXED_WC,
) -> tuple[int, float]:
    """在固定 (wc, ws) 与统一内层划分下选最优 PLS 成分数 k（折内独立 s_bar）。"""
    min_tr = min(len(itr) for itr, _ in splits)
    eff_kmax = min(int(kmax), X_cal.shape[1], max(1, min_tr - 1))
    fold_errs = np.zeros((len(splits), eff_kmax), dtype=np.float64)
    for fi, (itr, iva) in enumerate(splits):
        s_bar_itr = float(np.clip(X_cal[itr].std(axis=1), 1e-12, None).mean())
        Xtr_w = dssnv_apply(X_cal[itr], wc=wc, ws=ws, s_bar=s_bar_itr)
        Xva_w = dssnv_apply(X_cal[iva], wc=wc, ws=ws, s_bar=s_bar_itr)
        preds_k = pls_prefix_predictions(Xtr_w, yn_cal[itr], Xva_w, eff_kmax)
        for ki, pk in enumerate(preds_k):
            fold_errs[fi, ki] = rmsep(pk, yn_cal[iva])
    mean_errs = fold_errs.mean(axis=0)
    best_idx = int(np.argmin(mean_errs))
    return best_idx + 1, float(mean_errs[best_idx])


def select_ws_inner_cv_pls(
    X_cal: np.ndarray,
    yn_cal: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    ws_grid: tuple[float, ...] = WS_GRID,
    kmax: int = 12,
    wc: float = FIXED_WC,
) -> tuple[float, int, float, list[dict]]:
    """内层 CV 在 11 点 ws_grid 上联合选 (ws, k)。"""
    details = []
    best_ws, best_k, best_e = float(ws_grid[0]), 1, float("inf")
    for ws in ws_grid:
        k_opt, e_cv = select_k_pls_for_ws(X_cal, yn_cal, float(ws), splits, kmax=kmax, wc=wc)
        details.append({"ws": float(ws), "k": int(k_opt), "cv_rmsep_z": float(e_cv)})
        if e_cv < best_e - 1e-12:
            best_e = float(e_cv)
            best_ws = float(ws)
            best_k = int(k_opt)
    return best_ws, best_k, best_e, details


def fit_eval_pls_at_ws(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ws: float,
    k: int,
    wc: float = FIXED_WC,
) -> dict:
    """用训练集 s_bar 与锁定 (ws, k) 拟合 PLS 并在独立测试集上评估。"""
    mu = float(np.mean(y_cal))
    sd = max(float(np.std(y_cal)), 1e-9)
    yn_cal = (y_cal - mu) / sd
    s_bar = float(np.clip(X_cal.std(axis=1), 1e-12, None).mean())
    Xcal_w = dssnv_apply(X_cal, wc=wc, ws=ws, s_bar=s_bar)
    Xtest_w = dssnv_apply(X_test, wc=wc, ws=ws, s_bar=s_bar)
    eff_k = min(int(k), Xcal_w.shape[1], max(1, len(Xcal_w) - 1))
    m = PLSRegression(n_components=eff_k, scale=False).fit(Xcal_w, yn_cal[:, None])
    pred_z = m.predict(Xtest_w).ravel()
    pred_raw = pred_z * sd + mu
    yn_test = (y_test - mu) / sd
    return {
        "ws": float(ws),
        "k": int(eff_k),
        "rmsep_z": float(rmsep(pred_z, yn_test)),
        "rmsep_raw": float(rmsep(pred_raw, y_test)),
    }


def _fit_predict_ridge_svd(Xtr: np.ndarray, ytr_1d: np.ndarray, Xte: np.ndarray, gamma: float) -> np.ndarray:
    x_mu = Xtr.mean(axis=0, keepdims=True)
    y_mu = float(np.mean(ytr_1d))
    U, s, Vt = np.linalg.svd(Xtr - x_mu, full_matrices=False)
    lam = s ** 2
    W = Vt.T @ ((s / (lam + max(float(gamma), 1e-15))) * (U.T @ (ytr_1d - y_mu)))
    return (Xte - x_mu) @ W + y_mu


def select_gamma_ridge_for_ws(
    X_cal: np.ndarray,
    yn_cal: np.ndarray,
    ws: float,
    groups_cal: np.ndarray | None = None,
    wc: float = FIXED_WC,
) -> tuple[float, float]:
    """为伴随 Ridge 头选最优正则强度 gamma（无分组用精确 Allen-PRESS，有分组用 GroupKFold）。"""
    if groups_cal is not None:
        grp = np.asarray(groups_cal)
        uniq = np.unique(grp)
        eff = min(5, len(uniq))
        if eff >= 2:
            splits = [
                (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
                for tr, va in GroupKFold(n_splits=eff).split(X_cal, yn_cal, grp)
            ]
            if min(len(tr) for tr, _ in splits) >= 4:
                oof = np.empty_like(yn_cal, dtype=np.float64)
                gammas = []
                for itr, iva in splits:
                    s_bar_itr = float(np.clip(X_cal[itr].std(axis=1), 1e-12, None).mean())
                    Xtr_w = dssnv_apply(X_cal[itr], wc=wc, ws=ws, s_bar=s_bar_itr)
                    Xva_w = dssnv_apply(X_cal[iva], wc=wc, ws=ws, s_bar=s_bar_itr)
                    _, g_fold = exact_press_loo_residuals(Xtr_w, yn_cal[itr, None])
                    gammas.append(float(g_fold))
                    oof[iva] = _fit_predict_ridge_svd(Xtr_w, yn_cal[itr], Xva_w, g_fold)
                return float(np.median(gammas)), float(rmsep(oof, yn_cal))

    s_bar = float(np.clip(X_cal.std(axis=1), 1e-12, None).mean())
    Xw = dssnv_apply(X_cal, wc=wc, ws=ws, s_bar=s_bar)
    e_loo, best_g = exact_press_loo_residuals(Xw, yn_cal[:, None])
    return float(best_g), float(np.sqrt(np.mean(e_loo ** 2)))


def select_ws_inner_cv_ridge(
    X_cal: np.ndarray,
    yn_cal: np.ndarray,
    groups_cal: np.ndarray | None = None,
    ws_grid: tuple[float, ...] = WS_GRID,
    wc: float = FIXED_WC,
) -> tuple[float, float, float, list[dict]]:
    """内层 PRESS / Group-CV 在 11 点 ws_grid 上联合选 (ws, gamma)。"""
    details = []
    best_ws, best_g, best_e = float(ws_grid[0]), 1.0, float("inf")
    for ws in ws_grid:
        g_opt, e_cv = select_gamma_ridge_for_ws(X_cal, yn_cal, float(ws), groups_cal=groups_cal, wc=wc)
        details.append({"ws": float(ws), "gamma": float(g_opt), "cv_rmsep_z": float(e_cv)})
        if e_cv < best_e - 1e-12:
            best_e = float(e_cv)
            best_ws = float(ws)
            best_g = float(g_opt)
    return best_ws, best_g, best_e, details


def fit_eval_ridge_at_ws(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ws: float,
    gamma: float,
    wc: float = FIXED_WC,
) -> dict:
    """用训练集 s_bar 与锁定 (ws, gamma) 拟合 Ridge 并在独立测试集上评估。"""
    mu = float(np.mean(y_cal))
    sd = max(float(np.std(y_cal)), 1e-9)
    yn_cal = (y_cal - mu) / sd
    s_bar = float(np.clip(X_cal.std(axis=1), 1e-12, None).mean())
    Xcal_w = dssnv_apply(X_cal, wc=wc, ws=ws, s_bar=s_bar)
    Xtest_w = dssnv_apply(X_test, wc=wc, ws=ws, s_bar=s_bar)
    pred_z = _fit_predict_ridge_svd(Xcal_w, yn_cal, Xtest_w, gamma)
    pred_raw = pred_z * sd + mu
    yn_test = (y_test - mu) / sd
    return {
        "ws": float(ws),
        "gamma": float(gamma),
        "rmsep_z": float(rmsep(pred_z, yn_test)),
        "rmsep_raw": float(rmsep(pred_raw, y_test)),
    }
