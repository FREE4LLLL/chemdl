"""Phase B 严格独立留出（cal -> test）前瞻性部署任务加载器（18 项单次独立留出任务 / 6 大领域）。

硬不变量：
  1. 每项任务仅包含单个真实 `cal -> test` 独立划分，绝无 K 折测试标签互借；
  2. 每项任务强断言 `cal` 与 `test` 光谱字节级 SHA-256 重叠 == 0，且 `n_cal >= 30, n_test >= 30`；
  3. 存在采样分组（如 `Milk` 奶牛 ID）时，强断言 `groups_cal` 与 `groups_test` 零重叠。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from _audit._blind_theory_loaders import (
    _load_craw_csv,
    _load_ossl_unique_scans,
    load_blind_tasks,
)
from _audit._blind_round2_loaders import load_round2_tasks

SPLIT_SEED_DEPLOY = 20261124


def _row_hashes(X: np.ndarray) -> list[str]:
    Xa = np.ascontiguousarray(X, dtype=np.float64)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in Xa]


def _split_sha256_arrays(X_cal: np.ndarray, y_cal: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X_cal, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y_cal, dtype=np.float64).tobytes())
    h.update(b"|TEST|")
    h.update(np.ascontiguousarray(X_test, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y_test, dtype=np.float64).tobytes())
    return h.hexdigest()


def _load_craw_with_year(csv_path: Path, target_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    spec_cols = sorted(
        [c for c in df.columns if str(c).replace(".", "", 1).replace("-", "", 1).isdigit()],
        key=lambda c: float(c),
    )
    X = df[spec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    yr = pd.to_numeric(df["Year"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(yr)
    return X[ok], y[ok], yr[ok].astype(np.int64)


def load_deploy_holdout_tasks(root: Path | None = None) -> list[dict]:
    root = Path(__file__).resolve().parent.parent if root is None else Path(root)
    ct_dir = root / "data_ext" / "ct_bench"
    craw_dir = root / "data_ext" / "industrial" / "craw"
    ind_dir = root / "data_ext" / "industrial"

    # 构建既有光谱哈希集合，专门用于筛选 Stratum A（完全未见过光谱）
    seen_prior: set[str] = set()
    for t in load_blind_tasks(root):
        seen_prior.update(_row_hashes(t["X"]))
    for t in load_round2_tasks(root):
        seen_prior.update(_row_hashes(t["X"]))
    Xc_dev, _ = _load_craw_csv(craw_dir / "Protein" / "Wheat_sensAIfood_Protein_5046_NIRS5000_CRAW.csv", "Protein")
    seen_prior.update(_row_hashes(Xc_dev))

    tasks: list[dict] = []

    def _add(
        task_id: str,
        stratum: str,
        domain: str,
        split_rule: str,
        target_name: str,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        groups_cal: np.ndarray | None = None,
        groups_test: np.ndarray | None = None,
    ) -> None:
        X_cal = np.asarray(X_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=np.float64).ravel()
        X_test = np.asarray(X_test, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64).ravel()

        assert len(X_cal) == len(y_cal) and len(X_cal) >= 30, f"{task_id}: small cal {X_cal.shape}"
        assert len(X_test) == len(y_test) and len(X_test) >= 30, f"{task_id}: small test {X_test.shape}"
        assert X_cal.shape[1] == X_test.shape[1], f"{task_id}: channel mismatch"
        assert np.isfinite(X_cal).all() and np.isfinite(y_cal).all()
        assert np.isfinite(X_test).all() and np.isfinite(y_test).all()

        h_cal = set(_row_hashes(X_cal))
        h_test = set(_row_hashes(X_test))
        ov = len(h_cal & h_test)
        assert ov == 0, f"{task_id}: cal/test spectrum hash overlap = {ov} != 0"

        if groups_cal is not None or groups_test is not None:
            assert groups_cal is not None and groups_test is not None
            gc = np.asarray(groups_cal)
            gt = np.asarray(groups_test)
            assert len(gc) == len(X_cal) and len(gt) == len(X_test)
            assert len(set(gc.tolist()) & set(gt.tolist())) == 0, f"{task_id}: group overlap!"
        else:
            gc = None
            gt = None

        tasks.append(
            {
                "task_id": task_id,
                "stratum": stratum,
                "domain": domain,
                "split_rule": split_rule,
                "target_name": target_name,
                "n_cal": int(len(X_cal)),
                "n_test": int(len(X_test)),
                "L": int(X_cal.shape[1]),
                "cal_test_hash_overlap": 0,
                "split_sha256": _split_sha256_arrays(X_cal, y_cal, X_test, y_test),
                "X_cal": X_cal,
                "y_cal": y_cal,
                "groups_cal": gc,
                "X_test": X_test,
                "y_test": y_test,
                "groups_test": gt,
            }
        )

    # 1-2. Stratum A: 全新未见光谱前瞻时间划分 (Year <= 2017 -> >= 2018)
    for tid, rel, tcol in (
        ("Deploy_CRAW_Wheat_XDS_Moist_NewOnly__Temporal", "Moisture/Wheat_sensAIfood_Moist_500_XDS_CRAW.csv", "Moisture"),
        ("Deploy_CRAW_Wheat_XDS_Prot_NewOnly__Temporal", "Protein/Wheat_sensAIfood_Protein_500_XDS_CRAW.csv", "Protein"),
    ):
        Xa, ya, yra = _load_craw_with_year(craw_dir / rel, tcol)
        is_new = np.array([h not in seen_prior for h in _row_hashes(Xa)], dtype=bool)
        Xa, ya, yra = Xa[is_new], ya[is_new], yra[is_new]
        m_cal = yra <= 2017
        _add(tid, "A_Unseen_Spectra_Temporal", "Cereal XDS (Brand-New Spectra)", "Year <=2017 -> >=2018", tcol, Xa[m_cal], ya[m_cal], Xa[~m_cal], ya[~m_cal])

    # 3-6. Stratum B: Dev/R1 未触及的 CRA-W 谷物系列前瞻时间划分
    for tid, rel, tcol, cut in (
        ("Deploy_CRAW_Barley_NIRS5000_Moist__Temporal", "Moisture/Barley_sensAIfood_Moist_985_NIRS5000_CRAW.csv", "Moisture", 2012),
        ("Deploy_CRAW_Barley_NIRS5000_Prot__Temporal", "Protein/Barley_sensAIfood_Protein_2096_NIRS5000_CRAW.csv", "Protein", 2012),
        ("Deploy_CRAW_Wheat2_XDS_Moist__Temporal", "Moisture/Wheat2_sensAIfood_Moist_500_XDS_CRAW.csv", "Moisture", 2017),
        ("Deploy_CRAW_Wheat_XDS_Prot__Temporal", "Protein/Wheat_sensAIfood_Protein_231_XDS_CRAW.csv", "Protein", 2017),
    ):
        Xb, yb, yrb = _load_craw_with_year(craw_dir / rel, tcol)
        m_cal = yrb <= cut
        _add(tid, "B_CRAW_UnseenSeries_Temporal", "Cereal (Unseen in Dev/R1)", f"Year <={cut} -> >{cut}", tcol, Xb[m_cal], yb[m_cal], Xb[~m_cal], yb[~m_cal])

    # 7-13. Stratum C: 官方盲测 / 跨配方 / 按奶牛独立分组留出（含强受益与强失效两类典型场景）
    mg = sio.loadmat(ct_dir / "mango.mat")
    _add(
        "Deploy_Mango_Official_SP_test2__DM",
        "C_Official_CrossInst_CowGroup",
        "Anderson (2020) Mango DM",
        "Official SP_cal -> SP_test2",
        "Dry_Matter",
        np.asarray(mg["SP_cal"], np.float64),
        np.asarray(mg["DM_cal"], np.float64).ravel(),
        np.asarray(mg["SP_test2"], np.float64),
        np.asarray(mg["DM_test2"], np.float64).ravel(),
    )

    t2 = sio.loadmat(ct_dir / "nirshootout2.mat", squeeze_me=True, struct_as_record=False)
    _add(
        "Deploy_Tablets_X2_Official__Assay",
        "C_Official_CrossInst_CowGroup",
        "IDRC 2002 Tablets X2 [35:540]",
        "Official cal(155) -> test(460)",
        "Assay",
        np.asarray(t2["X"], np.float64)[:, 35:540],
        np.asarray(t2["y"], np.float64).ravel(),
        np.asarray(t2["Xtest"], np.float64)[:, 35:540],
        np.asarray(t2["ytest"], np.float64).ravel(),
    )

    wa1 = sio.loadmat(ct_dir / "wheatA1_20048.mat", squeeze_me=True, struct_as_record=False)
    _add(
        "Deploy_Wheat_A1_Official__Protein",
        "C_Official_CrossInst_CowGroup",
        "IDRC 2016 Wheat A1",
        "Official cal(200) -> test(48)",
        "Protein_%",
        np.asarray(wa1["X"], np.float64),
        np.asarray(wa1["y"], np.float64).ravel(),
        np.asarray(wa1["Xtest"], np.float64),
        np.asarray(wa1["ytest"], np.float64).ravel(),
    )

    wa3 = sio.loadmat(ct_dir / "wheatA3_48200.mat", squeeze_me=True, struct_as_record=False)
    _add(
        "Deploy_Wheat_A3_Official__Protein",
        "C_Official_CrossInst_CowGroup",
        "IDRC 2016 Wheat A3",
        "Official cal(48) -> test(200)",
        "Protein_%",
        np.asarray(wa3["X"], np.float64),
        np.asarray(wa3["y"], np.float64).ravel(),
        np.asarray(wa3["Xtest"], np.float64),
        np.asarray(wa3["ytest"], np.float64).ravel(),
    )

    mel = sio.loadmat(ct_dir / "Melamine_Dataset.mat", squeeze_me=True, struct_as_record=False)["Melamine_Dataset"]
    _add(
        "Deploy_Melamine_R568_to_R862__CloudPoint",
        "C_Official_CrossInst_CowGroup",
        "Melamine Resin FT-NIR (R568 -> R862)",
        "Cross-recipe R568(733) -> R862(472)",
        "Cloud_Point",
        np.asarray(mel.R568.X1, np.float64),
        np.asarray(mel.R568.Y, np.float64).ravel(),
        np.asarray(mel.R862.X1, np.float64),
        np.asarray(mel.R862.Y, np.float64).ravel(),
    )

    dfm = pd.read_csv(ind_dir / "milk_transmittance_zenodo8263430.csv")
    t_cols = [f"Trans_Tot_{i}" for i in range(1, 257)]
    X_milk = -np.log10(np.clip(dfm[t_cols].to_numpy(dtype=np.float64), 1e-5, None))
    cows_all = dfm["Cow_ID"].astype(str).to_numpy()
    for tcol in ("SCC", "Urea"):
        ym = pd.to_numeric(dfm[tcol], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(X_milk).all(axis=1) & np.isfinite(ym)
        Xm_ok, ym_ok, gc_ok = X_milk[ok], ym[ok], cows_all[ok]
        uniq_c = np.unique(gc_ok)
        perm_c = np.random.default_rng(SPLIT_SEED_DEPLOY).permutation(uniq_c)
        cal_cows = set(perm_c[: int(0.7 * len(uniq_c))].tolist())
        m_cal = np.array([c in cal_cows for c in gc_ok], dtype=bool)
        _add(
            f"Deploy_Milk_CowGroupHoldout__{tcol}",
            "C_Official_CrossInst_CowGroup",
            f"On-farm Milk NIR ({len(uniq_c)} cows)",
            "Cow_ID deterministic 70/30 holdout",
            tcol,
            Xm_ok[m_cal],
            ym_ok[m_cal],
            Xm_ok[~m_cal],
            ym_ok[~m_cal],
            groups_cal=gc_ok[m_cal],
            groups_test=gc_ok[~m_cal],
        )

    # 14-18. Stratum D: OSSL 未参与主基准的 5 项全新土壤理化性质（唯一扫描 70/30 单次留出）
    X_ossl_u, mg_ossl = _load_ossl_unique_scans(ct_dir)
    for tlabel, col in (
        ("Al_dith", "al.dith_usda.a65_w.pct"),
        ("Mg_ext", "mg.ext_usda.a724_cmolc.kg"),
        ("WR1500", "wr.1500kPa_usda.a417_w.pct"),
        ("P_ext", "p.ext_usda.a652_mg.kg"),
        ("EC", "ec_usda.a364_ds.m"),
    ):
        yo = pd.to_numeric(mg_ossl[col], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(X_ossl_u).all(axis=1) & np.isfinite(yo)
        Xo, yo_ok = X_ossl_u[ok], yo[ok]
        perm = np.random.default_rng(SPLIT_SEED_DEPLOY).permutation(len(yo_ok))
        n_cal = int(0.7 * len(perm))
        cal_idx, te_idx = perm[:n_cal], perm[n_cal:]
        _add(
            f"Deploy_OSSL_Holdout__{tlabel}",
            "D_OSSL_UnseenProps_Holdout",
            "OSSL Soil MEMS (Unique Scans)",
            "Unique-soil deterministic 70/30 holdout",
            col,
            Xo[cal_idx],
            yo_ok[cal_idx],
            Xo[te_idx],
            yo_ok[te_idx],
        )

    assert len(tasks) == 18, f"Expected 18 deploy holdout tasks, got {len(tasks)}"
    return tasks
