"""未参与准则开发的独立盲测数据集加载器（Phase 1 与 Phase 2 共用确定性切分）。

严格排除所有开发集（Tecator, Corn m5, Tablets X1, Cordoba 全麦/大麦/玉米 Grain grains grain_v2,
SWRI_Diesel 前 200 行 CN）及既往冻结后观察集（Mango SP_test1, CRAW Wheat Protein 5046,
OSSL OC/Clay/pH, Milk, Melamine, Wheat A1 200->48）。
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

SPLIT_SEED = 20261124


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def file_sha256_lf(path: Path) -> str:
    """Line-ending-normalized LF SHA-256 (identical across Windows CRLF / Linux LF checkouts)."""
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha256(root: Path, rev: str, rel_path: str, lf_normalize: bool = True) -> str:
    """Compute SHA-256 of `<rev>:<rel_path>` from Git object database, with working-tree fallback for clean clones."""
    try:
        raw = subprocess.check_output(["git", "show", f"{rev}:{rel_path}"], cwd=root, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raw = (root / rel_path).read_bytes()
    if lf_normalize:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha256_crlf(root: Path, rev: str, rel_path: str) -> str:
    """Compute Windows-CRLF equivalent SHA-256 of an LF Git blob for legacy archive verification."""
    try:
        raw = subprocess.check_output(["git", "show", f"{rev}:{rel_path}"], cwd=root, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raw = (root / rel_path).read_bytes()
    raw_crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(raw_crlf).hexdigest()


def split_checksum(tr_idx: np.ndarray, te_idx: np.ndarray) -> str:
    payload = (
        np.asarray(tr_idx, dtype=np.int64).tobytes()
        + b"|"
        + np.asarray(te_idx, dtype=np.int64).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _kfold_splits(n: int, n_splits: int = 5, seed: int = SPLIT_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, n_splits)
    out = []
    for va in folds:
        tr = perm[~np.isin(perm, va)]
        out.append((np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64)))
    return out


def _load_craw_csv(csv_path: Path, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    spec_cols = sorted(
        [c for c in df.columns if str(c).replace(".", "", 1).replace("-", "", 1).isdigit()],
        key=lambda c: float(c),
    )
    X = df[spec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[ok], y[ok]


def _load_ossl_unique_scans(ct_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """每个样本 ID 仅保留首条扫描，并剔除跨样本的逐位/6 位小数近似重复光谱。"""
    dfn = pd.read_csv(ct_dir / "neospectra_nir_v1.2.csv.gz")
    dfl = pd.read_csv(ct_dir / "neospectra_soillab_v1.2.csv.gz")
    spec_cols = [c for c in dfn.columns if c.startswith("scan_nir.")]
    dfn_first = dfn.drop_duplicates(subset=["id.sample_local_c"], keep="first").copy()
    mg = dfn_first.merge(dfl, on="id.sample_local_c", how="inner")
    mg = mg.sort_values("id.sample_local_c", kind="mergesort").reset_index(drop=True)
    X_all = mg[spec_cols].to_numpy(dtype=np.float64)
    fin_x = np.isfinite(X_all).all(axis=1)
    X_all = X_all[fin_x]
    mg = mg.loc[fin_x].reset_index(drop=True)

    # 剔除跨 sample ID 重复光谱（四舍五入 6 位小数精确去重）
    seen: set[bytes] = set()
    keep_idx = []
    for i, row in enumerate(np.round(X_all, 6)):
        key = row.tobytes()
        if key not in seen:
            seen.add(key)
            keep_idx.append(i)
    keep_arr = np.asarray(keep_idx, dtype=np.int64)
    return X_all[keep_arr], mg.iloc[keep_arr].reset_index(drop=True)


def load_blind_tasks(root: Path | None = None) -> list[dict]:
    root = Path(__file__).resolve().parent.parent if root is None else Path(root)
    ct_dir = root / "data_ext" / "ct_bench"
    craw_dir = root / "data_ext" / "industrial" / "craw"
    diesel_path = root / "data_ext" / "diesel" / "SWRI_Diesel_NIR.mat"

    tasks: list[dict] = []

    def _add_task(
        task_id: str,
        dataset_group: str,
        domain: str,
        instrument: str,
        target_name: str,
        X: np.ndarray,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        split_type: str,
    ) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        assert len(X) == len(y) and len(X) >= 20, f"{task_id}: invalid shape {X.shape}, {y.shape}"
        assert np.isfinite(X).all() and np.isfinite(y).all(), f"{task_id}: non-finite entries"
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
                "splits": splits,
            }
        )

    # 1. IDRC-2016 Wheat A3 official split (48 -> 200)
    wa3 = sio.loadmat(ct_dir / "wheatA3_48200.mat", squeeze_me=True, struct_as_record=False)
    X_a3 = np.vstack([np.asarray(wa3["X"], np.float64), np.asarray(wa3["Xtest"], np.float64)])
    y_a3 = np.concatenate([np.asarray(wa3["y"], np.float64).ravel(), np.asarray(wa3["ytest"], np.float64).ravel()])
    n_tr_a3 = int(len(wa3["X"]))
    splits_a3 = [(np.arange(n_tr_a3, dtype=np.int64), np.arange(n_tr_a3, len(X_a3), dtype=np.int64))]
    _add_task(
        "IDRC2016_Wheat_A3__Protein",
        "IDRC2016_Wheat_A3",
        "Whole-kernel wheat protein (IDRC 2016 A3)",
        "NIR_A3 (741 ch)",
        "Protein_%",
        X_a3,
        y_a3,
        splits_a3,
        "official",
    )

    # 2-4. IDRC-2016 Cross-manufacturer Wheat Inst A / B / C (100 ch, 248 scans)
    w_abc = np.load(ct_dir / "idrc2016_wheat_parsed.npz", allow_pickle=True)
    y_abc = np.asarray(w_abc["y"], np.float64).ravel()
    for inst_key, label in (("X_A", "InstA"), ("X_B", "InstB"), ("X_C", "InstC")):
        X_i = np.asarray(w_abc[inst_key], np.float64)
        _add_task(
            f"IDRC2016_Wheat_{label}__Protein",
            "IDRC2016_Wheat_ABC",
            f"Whole-kernel wheat protein (IDRC 2016 {label})",
            f"Manufacturer_{label} (100 ch)",
            "Protein_%",
            X_i,
            y_abc,
            _kfold_splits(len(y_abc)),
            "5fold",
        )

    # 5-9. SDOP 5-component Liquid NIR (spec1: 30 x 401, conc: 30 x 5)
    sdop = sio.loadmat(ct_dir / "sdop_nir_data.mat", squeeze_me=True, struct_as_record=False)
    X_sdop = np.asarray(sdop["spec1"], np.float64)
    Y_sdop = np.asarray(sdop["conc"], np.float64)
    for j in range(5):
        _add_task(
            f"SDOP_Liquid__c{j}",
            "SDOP_Liquid",
            "5-component liquid ternary/quinary NIR",
            "spec1 (401 ch)",
            f"Component_{j}",
            X_sdop,
            Y_sdop[:, j],
            _kfold_splits(len(X_sdop)),
            "5fold",
        )

    # 10-17. CRA-W unseen cereal subsets (XDS 1050 ch & Maize NIRS5000 700 ch)
    craw_specs = [
        ("CRAW_Spelt_XDS__Moisture", "CRAW_Spelt_XDS", "Cereal (Spelt)", "XDS (1050 ch)", craw_dir / "Moisture" / "Spelt_sensAIfood_Moist_411_XDS_CRAW.csv", "Moisture"),
        ("CRAW_Spelt_XDS__Protein", "CRAW_Spelt_XDS", "Cereal (Spelt)", "XDS (1050 ch)", craw_dir / "Protein" / "Spelt_sensAIfood_Protein_411_XDS_CRAW.csv", "Protein"),
        ("CRAW_WheatDurum_XDS__Moisture", "CRAW_WheatDurum_XDS", "Cereal (Durum Wheat)", "XDS (1050 ch)", craw_dir / "Moisture_and_Protein" / "WheatDurum_sensAIfood_Prot_Moist_500_XDS_CRAW.csv", "Moisture"),
        ("CRAW_WheatDurum_XDS__Protein", "CRAW_WheatDurum_XDS", "Cereal (Durum Wheat)", "XDS (1050 ch)", craw_dir / "Moisture_and_Protein" / "WheatDurum_sensAIfood_Prot_Moist_500_XDS_CRAW.csv", "Protein"),
        ("CRAW_Barley_XDS__Moisture", "CRAW_Barley_XDS", "Cereal (Barley)", "XDS (1050 ch)", craw_dir / "Moisture_and_Protein" / "Barley_sensAIfood_Prot_Moist_636_XDS_CRAW.csv", "Moisture"),
        ("CRAW_Barley_XDS__Protein", "CRAW_Barley_XDS", "Cereal (Barley)", "XDS (1050 ch)", craw_dir / "Moisture_and_Protein" / "Barley_sensAIfood_Prot_Moist_636_XDS_CRAW.csv", "Protein"),
        ("CRAW_Maize_NIRS5000__Moisture", "CRAW_Maize_NIRS5000", "Cereal (Maize)", "NIRS5000 (700 ch)", craw_dir / "Moisture" / "Maize_sensAIfood_Moist_585_NIRS5000_CRAW.csv", "Moisture"),
        ("CRAW_Maize_NIRS5000__Protein", "CRAW_Maize_NIRS5000", "Cereal (Maize)", "NIRS5000 (700 ch)", craw_dir / "Protein" / "Maize_sensAIfood_Protein_549_NIRS5000_CRAW.csv", "Protein"),
    ]
    for tid, grp, dom, inst, p, tcol in craw_specs:
        Xc, yc = _load_craw_csv(p, tcol)
        _add_task(tid, grp, dom, inst, tcol, Xc, yc, _kfold_splits(len(yc)), "5fold")

    # 18-21. SWRI Diesel strictly holdout rows 200: & unseen targets
    dm = sio.loadmat(diesel_path)
    Xd_hold = np.asarray(dm["diesel_spec"][0, 0]["data"], np.float64)[200:]
    Yd_hold = np.asarray(dm["diesel_prop"][0, 0]["data"], np.float64)[200:]
    d_names = [str(s).strip() for s in dm["diesel_prop"][0, 0]["label"][1, 0]]
    for col_name in ("BP50", "D4052", "FREEZE", "VISC"):
        j = d_names.index(col_name)
        yv = Yd_hold[:, j]
        ok = np.isfinite(Xd_hold).all(axis=1) & np.isfinite(yv)
        Xs, ys = Xd_hold[ok], yv[ok]
        _add_task(
            f"SWRI_Diesel_Holdout__{col_name}",
            "SWRI_Diesel_Holdout",
            "Liquid diesel fuel (SWRI rows >= 200)",
            "SWRI NIR (401 ch)",
            col_name,
            Xs,
            ys,
            _kfold_splits(len(ys)),
            "5fold",
        )

    # 22-24. OSSL Soil unseen properties (1 unique scan per sample ID, deduped)
    X_ossl_u, mg_ossl = _load_ossl_unique_scans(ct_dir)
    ossl_targets = [
        ("OSSL_Soil_Unseen__Sand", "Sand_pct", "sand.tot_usda.c60_w.pct"),
        ("OSSL_Soil_Unseen__N_tot", "N_tot_pct", "n.tot_usda.a623_w.pct"),
        ("OSSL_Soil_Unseen__CEC", "CEC_cmolc_kg", "cec_usda.a723_cmolc.kg"),
    ]
    for tid, tlabel, raw_col in ossl_targets:
        yv = pd.to_numeric(mg_ossl[raw_col], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(X_ossl_u).all(axis=1) & np.isfinite(yv)
        Xo, yo = X_ossl_u[ok], yv[ok]
        _add_task(
            tid,
            "OSSL_Soil_Unseen",
            "Commercial handheld MEMS soil NIR (unique soils, unseen targets)",
            "NeoSpectra MEMS (601 ch)",
            tlabel,
            Xo,
            yo,
            _kfold_splits(len(yo)),
            "5fold",
        )

    return tasks
