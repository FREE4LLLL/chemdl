"""第二轮独立盲测加载器（18 项全新未参与任务 / 86 折，SPLIT_SEED = 20261125）。

与训练集（Tecator, Corn m5, Tablets X1, Cordoba v2 grains, SWRI Diesel 0:200 CN）、
已有观测集（Mango SP_test1, CRAW Wheat Protein 5046, OSSL OC/Clay/pH, Milk Fat/Prot/Lact,
Melamine, Wheat A1 200->48）及第一轮 24 项任务（Round 1）完全互不相交。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import GroupKFold

from _audit._blind_theory_loaders import (
    _kfold_splits,
    _load_craw_csv,
    _load_ossl_unique_scans,
)

SPLIT_SEED_R2 = 20261125


def _group_kfold_splits(groups: np.ndarray, n_splits: int = 5, seed: int = SPLIT_SEED_R2) -> list[tuple[np.ndarray, np.ndarray]]:
    grp = np.asarray(groups)
    uniq = np.unique(grp)
    perm_g = np.random.default_rng(seed).permutation(uniq)
    remap = {g: i for i, g in enumerate(perm_g)}
    mapped = np.array([remap[g] for g in grp], dtype=np.int64)
    dummy_X = np.zeros((len(grp), 1), dtype=np.float64)
    return [
        (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
        for tr, va in GroupKFold(n_splits=n_splits).split(dummy_X, mapped, mapped)
    ]


def load_round2_tasks(root: Path | None = None) -> list[dict]:
    root = Path(__file__).resolve().parent.parent if root is None else Path(root)
    ct_dir = root / "data_ext" / "ct_bench"
    craw_dir = root / "data_ext" / "industrial" / "craw"
    ind_dir = root / "data_ext" / "industrial"
    diesel_path = root / "data_ext" / "diesel" / "SWRI_Diesel_NIR.mat"

    tasks: list[dict] = []

    def _add(
        task_id: str,
        dataset_group: str,
        domain: str,
        instrument: str,
        target_name: str,
        X: np.ndarray,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        split_type: str,
        groups: np.ndarray | None = None,
    ) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        assert len(X) == len(y) and len(X) >= 20, f"{task_id}: invalid shape {X.shape}, {y.shape}"
        assert np.isfinite(X).all() and np.isfinite(y).all(), f"{task_id}: non-finite values"
        tasks.append(
            {
                "task_id": task_id,
                "dataset_group": dataset_group,
                "domain": domain,
                "instrument": instrument,
                "target_name": target_name,
                "split_type": split_type,
                "n": int(len(X)),
                "L": int(X.shape[1]),
                "X": X,
                "y": y,
                "groups": None if groups is None else np.asarray(groups),
                "splits": splits,
            }
        )

    # 1-8. CRAW 第二批未触碰谷物子库
    craw_specs = [
        ("CRAW_Barley_NIRS5000__Moisture", "CRAW_Barley_NIRS5000", "Cereal (Barley NIRS5000)", "NIRS5000 (700 ch)", craw_dir / "Moisture" / "Barley_sensAIfood_Moist_985_NIRS5000_CRAW.csv", "Moisture"),
        ("CRAW_Barley_NIRS5000__Protein", "CRAW_Barley_NIRS5000", "Cereal (Barley NIRS5000)", "NIRS5000 (700 ch)", craw_dir / "Protein" / "Barley_sensAIfood_Protein_2096_NIRS5000_CRAW.csv", "Protein"),
        ("CRAW_Wheat_NIRS5000__Moisture", "CRAW_Wheat_NIRS5000", "Cereal (Wheat NIRS5000)", "NIRS5000 (700 ch)", craw_dir / "Moisture" / "Wheat_sensAIfood_Moist_3305_NIRS5000_CRAW.csv", "Moisture"),
        ("CRAW_Wheat2_XDS__Moisture", "CRAW_Wheat2_XDS", "Cereal (Wheat2 XDS)", "XDS (1050 ch)", craw_dir / "Moisture" / "Wheat2_sensAIfood_Moist_500_XDS_CRAW.csv", "Moisture"),
        ("CRAW_Wheat_XDS__Protein", "CRAW_Wheat_XDS", "Cereal (Wheat XDS)", "XDS (1050 ch)", craw_dir / "Protein" / "Wheat_sensAIfood_Protein_231_XDS_CRAW.csv", "Protein"),
        ("CRAW_Spelt_XDS2__Moisture", "CRAW_Spelt_XDS2", "Cereal (Spelt XDS subset2)", "XDS (1050 ch)", craw_dir / "Moisture" / "Spelt_sensAIfood_Moist_198_XDS_CRAW.csv", "Moisture"),
        ("CRAW_WheatDurum_XDS2__Moisture", "CRAW_WheatDurum_XDS2", "Cereal (Durum Moisture subset)", "XDS (1050 ch)", craw_dir / "Moisture" / "WheatDurum_sensAIfood_Moist_170_XDS_CRAW.csv", "Moisture"),
        ("CRAW_WheatDurum_XDS2__Protein", "CRAW_WheatDurum_XDS2", "Cereal (Durum Protein subset)", "XDS (1050 ch)", craw_dir / "Protein" / "WheatDurum_sensAIfood_Protein_125_XDS_CRAW.csv", "Protein"),
    ]
    for tid, grp, dom, inst, p, tcol in craw_specs:
        Xc, yc = _load_craw_csv(p, tcol)
        _add(tid, grp, dom, inst, tcol, Xc, yc, _kfold_splits(len(yc), seed=SPLIT_SEED_R2), "5fold")

    # 9-10. On-Farm Raw Milk unseen target columns (Urea, Milk_yield) with strict Cow_ID GroupKFold
    dfm = pd.read_csv(ind_dir / "milk_transmittance_zenodo8263430.csv")
    t_cols = [f"Trans_Tot_{i}" for i in range(1, 257)]
    X_milk = -np.log10(np.clip(dfm[t_cols].to_numpy(dtype=np.float64), 1e-5, None))
    cows_all = dfm["Cow_ID"].astype(str).to_numpy()
    for tcol in ("Urea", "Milk_yield"):
        ym = pd.to_numeric(dfm[tcol], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(X_milk).all(axis=1) & np.isfinite(ym)
        Xm_ok, ym_ok, g_ok = X_milk[ok], ym[ok], cows_all[ok]
        _add(
            f"Milk_Raw__{tcol}",
            "Milk_Raw_Unseen",
            f"On-farm dairy transmittance NIR (GroupKFold by Cow_ID, {len(np.unique(g_ok))} cows)",
            "On-farm Vis-NIR (256 ch)",
            tcol,
            Xm_ok,
            ym_ok,
            _group_kfold_splits(g_ok, n_splits=5, seed=SPLIT_SEED_R2),
            "5fold_group_cow",
            groups=g_ok,
        )

    # 11-14. OSSL Soil Round-2 unseen properties (1 unique scan per sample ID + dedup)
    X_ossl_u, mg_ossl = _load_ossl_unique_scans(ct_dir)
    ossl_specs = [
        ("OSSL_Soil_R2__Silt", "Silt_pct", "silt.tot_usda.c62_w.pct"),
        ("OSSL_Soil_R2__BulkDensity", "BulkDensity_g_cm3", "bd_usda.a4_g.cm3"),
        ("OSSL_Soil_R2__CaCO3", "CaCO3_pct", "caco3_usda.a54_w.pct"),
        ("OSSL_Soil_R2__K_ext", "K_ext_cmolc_kg", "k.ext_usda.a725_cmolc.kg"),
    ]
    for tid, tlabel, col in ossl_specs:
        yv = pd.to_numeric(mg_ossl[col], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(X_ossl_u).all(axis=1) & np.isfinite(yv)
        Xo, yo = X_ossl_u[ok], yv[ok]
        _add(tid, "OSSL_Soil_R2", "Handheld soil MEMS-NIR (unique soils, R2 targets)", "NeoSpectra (601 ch)", tlabel, Xo, yo, _kfold_splits(len(yo), seed=SPLIT_SEED_R2), "5fold")

    # 15-16. SWRI Diesel Holdout rows >= 200 unseen targets FLASH, TOTAL
    dm = sio.loadmat(diesel_path)
    Xd_hold = np.asarray(dm["diesel_spec"][0, 0]["data"], np.float64)[200:]
    Yd_hold = np.asarray(dm["diesel_prop"][0, 0]["data"], np.float64)[200:]
    d_names = [str(s).strip() for s in dm["diesel_prop"][0, 0]["label"][1, 0]]
    for col_name in ("FLASH", "TOTAL"):
        j = d_names.index(col_name)
        yv = Yd_hold[:, j]
        ok = np.isfinite(Xd_hold).all(axis=1) & np.isfinite(yv)
        Xs, ys = Xd_hold[ok], yv[ok]
        _add(f"SWRI_Diesel_R2__{col_name}", "SWRI_Diesel_R2", "Liquid diesel (SWRI rows >= 200, R2 targets)", "SWRI NIR (401 ch)", col_name, Xs, ys, _kfold_splits(len(ys), seed=SPLIT_SEED_R2), "5fold")

    # 17. Tablets X2 official split (trimmed [35:540], L=505)
    t2 = sio.loadmat(ct_dir / "nirshootout2.mat", squeeze_me=True, struct_as_record=False)
    X_t2 = np.vstack([np.asarray(t2["X"], np.float64)[:, 35:540], np.asarray(t2["Xtest"], np.float64)[:, 35:540]])
    y_t2 = np.concatenate([np.asarray(t2["y"], np.float64).ravel(), np.asarray(t2["ytest"], np.float64).ravel()])
    n_tr_t2 = int(len(t2["X"]))
    splits_t2 = [(np.arange(n_tr_t2, dtype=np.int64), np.arange(n_tr_t2, len(X_t2), dtype=np.int64))]
    _add("Tablets_X2_Official__Assay", "Tablets_X2_Official", "IDRC 2002 Tablets inst X2 trimmed [35:540]", "NIR_X2 (505 ch)", "Assay", X_t2, y_t2, splits_t2, "official")

    # 18. SDOP Liquid inst 2 (spec2, c2)
    sdop = sio.loadmat(ct_dir / "sdop_nir_data.mat", squeeze_me=True, struct_as_record=False)
    X_sdop2 = np.asarray(sdop["spec2"], np.float64)
    y_sdop2 = np.asarray(sdop["conc"], np.float64)[:, 2]
    _add("SDOP_Spec2__c2", "SDOP_Spec2", "5-component liquid NIR (spec2, c2)", "spec2 (401 ch)", "Component_2", X_sdop2, y_sdop2, _kfold_splits(len(y_sdop2), seed=SPLIT_SEED_R2), "5fold")

    return tasks
