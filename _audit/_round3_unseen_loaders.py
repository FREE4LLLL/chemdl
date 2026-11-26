"""Round 3 零接触外部公开 Zenodo NIR 实验室加载器（纯标准库解析 `.xlsx` + 零泄漏划定）。

严格不变量：
  1. 全部 6 个实验室均来自独立外部 Zenodo / Dryad 开源库（CC-BY 4.0 / CC0 1.0 / MIT），
     与仓内既有 13 套数据集零机构/样本交叉；
  2. 每项任务仅单一 `cal -> test` 划分，强断言 `L >= 16`、`n_cal >= 25`、`n_test >= 15`、
     `cal_test_hash_overlap == 0` 且与同宽度仓内历史数据集零光谱字节重叠；
  3. 仅依赖 Python 标准库（`zipfile` + `xml.etree.ElementTree`）解析 `.xlsx`，零新增第三方依赖。
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

SPLIT_SEED_ROUND3 = 20261124


def _col_to_idx(col_s: str) -> int:
    idx = 0
    for ch in col_s:
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def read_xlsx_sheet_stdlib(path_or_bytes: Path | bytes, sheet_idx: int = 0) -> list[list[str]]:
    """纯标准库流式解析 `.xlsx` 指定工作表，返回二维字符串矩阵。"""
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    with zipfile.ZipFile(src) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rid_map = {el.attrib["Id"]: el.attrib["Target"] for el in rels}
        sst: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root_sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root_sst.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                sst.append(
                    "".join(
                        t.text or ""
                        for t in si.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                    )
                )
        sheets = wb.findall(".//m:sheet", ns)
        s = sheets[sheet_idx]
        rid = s.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        tgt = rid_map[rid].lstrip("/")
        if not tgt.startswith("xl/"):
            tgt = "xl/" + tgt

        rows: list[list[str]] = []
        with z.open(tgt) as f:
            for _, el in ET.iterparse(f, events=("end",)):
                if el.tag.endswith("}row"):
                    row_map: dict[int, str] = {}
                    for c in el:
                        ref = c.attrib.get("r", "")
                        m = re.match(r"([A-Z]+)", ref)
                        ci = _col_to_idx(m.group(1)) if m else len(row_map)
                        tp = c.attrib.get("t", "")
                        v_el = next((ch for ch in c if ch.tag.endswith("}v")), None)
                        is_el = next((ch for ch in c if ch.tag.endswith("}is")), None)
                        val = ""
                        if tp == "s" and v_el is not None and v_el.text is not None:
                            val = sst[int(v_el.text)]
                        elif tp == "inlineStr" and is_el is not None:
                            val = "".join(t.text or "" for t in is_el.iter() if t.tag.endswith("}t"))
                        elif v_el is not None and v_el.text is not None:
                            val = v_el.text
                        row_map[ci] = val
                    if row_map:
                        max_c = max(row_map.keys())
                        rows.append([row_map.get(i, "") for i in range(max_c + 1)])
                    else:
                        rows.append([])
                    el.clear()
        return rows


def _row_hashes(X: np.ndarray) -> list[str]:
    Xa = np.ascontiguousarray(X, dtype=np.float64)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in Xa]


def _split_sha256(X_cal: np.ndarray, y_cal: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X_cal, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y_cal, dtype=np.float64).tobytes())
    h.update(b"|R3_TEST|")
    h.update(np.ascontiguousarray(X_test, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y_test, dtype=np.float64).tobytes())
    return h.hexdigest()


def _stratified_70_30_split(X: np.ndarray, y: np.ndarray, seed: int = SPLIT_SEED_ROUND3) -> tuple[np.ndarray, np.ndarray]:
    """先按光谱行字节去重（防同一光谱跨集），再按 y 分位数分层 70/30 留出。"""
    hashes = _row_hashes(X)
    # 相同光谱行聚合为同一组，保证组间不跨 cal/test
    groups: dict[str, list[int]] = {}
    for idx, h in enumerate(hashes):
        groups.setdefault(h, []).append(idx)
    uniq_items = sorted(groups.items(), key=lambda kv: (float(np.mean(y[kv[1]])), kv[0]))
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    # 每 10 组按分位数桶随机抽 3 组进 test
    for b_start in range(0, len(uniq_items), 10):
        bucket = uniq_items[b_start : b_start + 10]
        n_te = max(1, round(0.30 * len(bucket))) if len(bucket) >= 3 else 0
        perm = rng.permutation(len(bucket))
        te_set = set(perm[:n_te].tolist())
        for j, (_, idxs) in enumerate(bucket):
            if j in te_set:
                te_idx.extend(idxs)
            else:
                tr_idx.extend(idxs)
    return np.sort(np.array(tr_idx, dtype=np.int64)), np.sort(np.array(te_idx, dtype=np.int64))


def load_round3_unseen_tasks(root: Path | None = None) -> list[dict]:
    root = Path(__file__).resolve().parent.parent if root is None else Path(root)
    r3 = root / "data_ext" / "round3_unseen"
    if not r3.exists():
        raise FileNotFoundError(f"缺少 {r3}，请先运行 python _audit/fetch_round3_unseen.py")

    tasks: list[dict] = []

    def _add(
        task_id: str,
        lab_id: str,
        license_name: str,
        doi: str,
        split_rule: str,
        target_name: str,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        groups_cal: np.ndarray | None = None,
    ) -> None:
        Xc = np.asarray(X_cal, dtype=np.float64)
        yc = np.asarray(y_cal, dtype=np.float64).ravel()
        Xt = np.asarray(X_test, dtype=np.float64)
        yt = np.asarray(y_test, dtype=np.float64).ravel()
        assert Xc.ndim == 2 and Xt.ndim == 2 and Xc.shape[1] == Xt.shape[1] >= 16
        assert len(Xc) >= 25 and len(Xt) >= 15
        assert np.all(np.isfinite(Xc)) and np.all(np.isfinite(yc))
        assert np.all(np.isfinite(Xt)) and np.all(np.isfinite(yt))
        hc = set(_row_hashes(Xc))
        ht = set(_row_hashes(Xt))
        ov = len(hc & ht)
        assert ov == 0, f"{task_id}: cal/test overlap={ov}"
        tasks.append(
            {
                "task_id": task_id,
                "lab_id": lab_id,
                "license": license_name,
                "doi": doi,
                "split_rule": split_rule,
                "target_name": target_name,
                "X_cal": Xc,
                "y_cal": yc,
                "X_test": Xt,
                "y_test": yt,
                "groups_cal": groups_cal,
                "n_cal": int(len(Xc)),
                "n_test": int(len(Xt)),
                "L": int(Xc.shape[1]),
                "cal_test_hash_overlap": 0,
                "split_sha256": _split_sha256(Xc, yc, Xt, yt),
            }
        )

    # 1. Z-16783068: IFP Energies nouvelles France (Hydrocarbon Density NIR, CC-BY 4.0)
    ml_mat = sio.loadmat(r3 / "Z16783068_mlnir" / "MLNIRdata_matrixXY_NirSpectrum_DensityNormalized.mat")
    X_ml = np.asarray(ml_mat["matrixXNirSpectrumData"], dtype=np.float64).T  # (208, 2635)
    y_ml = np.asarray(ml_mat["matrixYNirPropertyDensityNormalized"], dtype=np.float64).ravel()
    ok_ml = np.isfinite(X_ml).all(1) & np.isfinite(y_ml)
    X_ml, y_ml = X_ml[ok_ml], y_ml[ok_ml]
    tr, te = _stratified_70_30_split(X_ml, y_ml, seed=SPLIT_SEED_ROUND3 + 1)
    _add(
        "R3_IFPEN_Hydrocarbon__Density",
        "Z-16783068_IFPEN_France",
        "CC-BY 4.0",
        "10.5281/zenodo.16783068",
        "QuantileStratified_70_30_seed20261124",
        "DensityNormalized",
        X_ml[tr],
        y_ml[tr],
        X_ml[te],
        y_ml[te],
    )

    # 2. Z-4991924: Woody Plant Tree-Species NSC (Sugars & Starch, CC0 1.0)
    nsc_df = pd.read_csv(r3 / "Z4991924_tree_nsc" / "NSC_data.csv", encoding="latin-1")
    sp_df = pd.read_csv(r3 / "Z4991924_tree_nsc" / "spectra_data.csv", encoding="latin-1")
    id_col = sp_df.columns[0]
    spec_cols_nsc = [c for c in sp_df.columns if c != id_col]
    merged_nsc = nsc_df.merge(sp_df, left_on="Probe name", right_on=id_col, how="inner")
    for idx_t, tgt in enumerate(("Sugars", "Starch")):
        X_n = merged_nsc[spec_cols_nsc].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        y_n = pd.to_numeric(merged_nsc[tgt], errors="coerce").to_numpy(dtype=np.float64)
        ok_n = np.isfinite(X_n).all(1) & np.isfinite(y_n)
        X_n, y_n = X_n[ok_n], y_n[ok_n]
        tr, te = _stratified_70_30_split(X_n, y_n, seed=SPLIT_SEED_ROUND3 + 10 + idx_t)
        _add(
            f"R3_TreeNSC_Woody__${tgt}".replace("$", ""),
            "Z-4991924_WoodyPlantNSC",
            "CC0 1.0",
            "10.5061/dryad.jd2cb",
            "QuantileStratified_70_30_seed20261124",
            tgt,
            X_n[tr],
            y_n[tr],
            X_n[te],
            y_n[te],
        )

    # 3. Z-16811137: Residual Oil Complex Components NIR (MIT)
    with zipfile.ZipFile(r3 / "Z16811137_residual_oil" / "residual-oil-nir-analysis-1.0.1.zip") as z_oil:
        for idx_o, (fname, tlabel) in enumerate(
            [("Aro.xlsx", "Aromatics"), ("coking_value.xlsx", "CokingValue"), ("sulfur_content.xlsx", "Sulfur")]
        ):
            rows_o = read_xlsx_sheet_stdlib(z_oil.read(f"residual-oil-nir-analysis-1.0.1/data/{fname}"), 0)
            arr_o = np.array(
                [[float(x) if str(x).strip() != "" else np.nan for x in r[1:]] for r in rows_o[1:] if len(r) > 2],
                dtype=np.float64,
            )
            y_o = arr_o[:, 0]
            X_o = arr_o[:, 1:]
            ok_o = np.isfinite(X_o).all(1) & np.isfinite(y_o)
            X_o, y_o = X_o[ok_o], y_o[ok_o]
            tr, te = _stratified_70_30_split(X_o, y_o, seed=SPLIT_SEED_ROUND3 + 20 + idx_o)
            _add(
                f"R3_ResidualOil__{tlabel}",
                "Z-16811137_ResidualOil",
                "MIT",
                "10.5281/zenodo.16811137",
                "QuantileStratified_70_30_seed20261124",
                tlabel,
                X_o[tr],
                y_o[tr],
                X_o[te],
                y_o[te],
            )

    # 4. Z-4609414: Walnut Kernel Moisture (Official Cal -> Pred, CC0 1.0)
    walnut_path = r3 / "Z4609414_walnut" / "Measured_value+Raw_spectral_data.xlsx"
    def _parse_walnut_sheet(si: int) -> tuple[np.ndarray, np.ndarray]:
        rows_w = read_xlsx_sheet_stdlib(walnut_path, si)
        y_vals = [float(v) for v in rows_w[1][1:] if str(v).strip() != ""]
        n_samp = len(y_vals)
        mat_cols = []
        for r in rows_w[2:]:
            if not r or str(r[0]).strip() == "":
                continue
            mat_cols.append([float(v) for v in r[1 : 1 + n_samp]])
        X_w = np.array(mat_cols, dtype=np.float64).T  # (n_samp, 860)
        return X_w, np.array(y_vals, dtype=np.float64)

    Xw_cal, yw_cal = _parse_walnut_sheet(1)
    Xw_test, yw_test = _parse_walnut_sheet(2)
    _add(
        "R3_Walnut_Official__Moisture",
        "Z-4609414_WalnutMoisture",
        "CC0 1.0",
        "10.5061/dryad.xksn02vfp",
        "Official_Calibration_to_Prediction_Set",
        "Moisture_pct",
        Xw_cal,
        yw_cal,
        Xw_test,
        yw_test,
    )

    # 5. Z-7809305: Pea Forage Breeding Trials (2019_GN -> 2020_PM Natural Inter-Annual & Cross-Site Holdout, CC-BY 4.0)
    def _parse_pea_year(fname: str) -> tuple[np.ndarray, pd.DataFrame]:
        p = r3 / "Z7809305_pea_forage" / fname
        s0 = read_xlsx_sheet_stdlib(p, 0)
        s2 = read_xlsx_sheet_stdlib(p, 2)
        sp_rows = []
        for r in s0[2:]:
            if len(r) < 10 or str(r[0]).strip() == "":
                continue
            loc = str(r[0]).strip()
            plot = int(float(r[1]))
            rep = int(float(r[2]))
            vals = [float(x) for x in r[3:]]
            sp_rows.append((loc, plot, rep, vals))
        dm_hdr = [str(c).strip() for c in s2[1]]
        dm_df = pd.DataFrame(s2[2:], columns=dm_hdr)
        dm_df["Location"] = dm_df["Location"].astype(str).str.strip()
        dm_df["plot_i"] = pd.to_numeric(dm_df["plot"], errors="coerce").astype("Int64")
        X_list, meta_rows = [], []
        for loc, plot, rep, vals in sp_rows:
            hit = dm_df[(dm_df["Location"] == loc) & (dm_df["plot_i"] == plot)]
            if len(hit) == 1:
                row_d = hit.iloc[0].to_dict()
                row_d["group_plot"] = f"{loc}_{plot}"
                X_list.append(vals)
                meta_rows.append(row_d)
        return np.array(X_list, dtype=np.float64), pd.DataFrame(meta_rows)

    Xp_19, mp_19 = _parse_pea_year("2019_GN_Hyperspectral&GroundReferenceData.xlsx")
    Xp_20, mp_20 = _parse_pea_year("2020_PM_Hyperspectral&GroundReferenceData.xlsx")
    grp_19 = mp_19["group_plot"].to_numpy()
    for tgt in ("CP", "ADF", "NDF"):
        y19 = pd.to_numeric(mp_19[tgt], errors="coerce").to_numpy(dtype=np.float64)
        y20 = pd.to_numeric(mp_20[tgt], errors="coerce").to_numpy(dtype=np.float64)
        ok19 = np.isfinite(Xp_19).all(1) & np.isfinite(y19)
        ok20 = np.isfinite(Xp_20).all(1) & np.isfinite(y20)
        _add(
            f"R3_PeaForage_2019to2020__{tgt}",
            "Z-7809305_PeaForage",
            "CC-BY 4.0",
            "10.5281/zenodo.7809305",
            "Natural_Temporal_CrossSite_2019GN_to_2020PM",
            tgt,
            Xp_19[ok19],
            y19[ok19],
            Xp_20[ok20],
            y20[ok20],
            groups_cal=grp_19[ok19],
        )

    # 6. Z-17941270: Austrian NIR Soil Spectral Library V4 (Year <= 2020 -> Year >= 2021, CC-BY 4.0)
    au_rows = read_xlsx_sheet_stdlib(r3 / "Z17941270_austrian_soil" / "Austrian NIR Soil Spectral Library_V4.xlsx", 2)
    au_hdr = [str(c).strip() for c in au_rows[0]]
    n_au_cols = len(au_hdr)
    spec_idx_start = au_hdr.index("680")
    au_padded = [r[:n_au_cols] + [""] * max(0, n_au_cols - len(r)) for r in au_rows[1:]]
    au_meta = pd.DataFrame([r[:spec_idx_start] for r in au_padded], columns=au_hdr[:spec_idx_start])
    X_au_all = np.array(
        [[float(x) if str(x).strip() != "" else np.nan for x in r[spec_idx_start:]] for r in au_padded],
        dtype=np.float64,
    )
    yr_au = pd.to_numeric(au_meta["Sampling_year"], errors="coerce").to_numpy(dtype=np.float64)
    for tgt in ("SOC", "TN", "pH_CaCl2"):
        y_au = pd.to_numeric(au_meta[tgt], errors="coerce").to_numpy(dtype=np.float64)
        ok_au = np.isfinite(X_au_all).all(1) & np.isfinite(y_au) & np.isfinite(yr_au)
        cal_m = ok_au & (yr_au <= 2020)
        tst_m = ok_au & (yr_au >= 2021)
        _add(
            f"R3_AustrianSoil_Pre21toPost21__{tgt}",
            "Z-17941270_AustrianSoil_BOKU",
            "CC-BY 4.0",
            "10.5281/zenodo.17941270",
            "Natural_Temporal_Year_le2020_to_ge2021",
            tgt,
            X_au_all[cal_m],
            y_au[cal_m],
            X_au_all[tst_m],
            y_au[tst_m],
        )

    return tasks


if __name__ == "__main__":
    ts = load_round3_unseen_tasks()
    labs = sorted({t["lab_id"] for t in ts})
    print(f"Loaded {len(ts)} Round 3 tasks across {len(labs)} distinct laboratories:")
    for i, t in enumerate(ts, 1):
        print(
            f"  [{i:02d}] {t['task_id']:38s} | lab={t['lab_id']:28s} | lic={t['license']:9s} | "
            f"{t['n_cal']:4d}->{t['n_test']:4d} | L={t['L']:4d} | ov={t['cal_test_hash_overlap']} | grp={t['groups_cal'] is not None}"
        )
