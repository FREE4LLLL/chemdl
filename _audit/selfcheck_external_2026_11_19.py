"""外部开源新数据集全量自检 + 统计卫生修复（消除 5×5 重复 CV 伪重复）+ 固定方差 SNV 耦合复现。

新增外部公开跨仪器配对库（此前 dsh 与我方均未测试过）：
1. OSSL v1.2 NeoSpectra Handheld MEMS Soil NIR (CC-BY 4.0)：NEO2->NEO3 (1104 对) & NEO8->NEO9 (986 对)，严格不相交切分；
2. IDRC-2016 Cross-Manufacturer Wheat Shootout (CNIRS)：3 家不同厂商 A/B/C (248 对 × 100 波段)；
3. 现有三库（Corn / Tablets 官方 460 独立测试集 / SDOP）补齐逐样本 OOF 配对 Wilcoxon + Nadeau–Bengio 校正 t 检验；
4. 固定方差 std(a)=0.35 耦合注入复现：Cordoba Wheat + Cordoba Barley + SWRI Diesel 液态柴油库。
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.io
from scipy import stats

from chemdl.cordoba import load_cordoba_all
from chemdl.corn import load_corn
from chemdl.select import d_opt_select, estimate_spectral_temper_alpha, t_opt_select, v_opt_select
from chemdl.tablets import load_tablets
from chemdl.transfer import (
    PREPS,
    active_transfer_fit,
    ds_affine,
    ds_apply_affine,
    fit_pls,
    rmsep,
    select_active_transfer,
    select_kennard_stone,
    subspace_lot_apply,
    subspace_lot_fit,
)

CT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_ext", "ct_bench")


def _read_xlsx_numeric(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """纯标准库解析 IDRC-2016 Wheat Test_*.xlsx（零 openpyxl 依赖）。"""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as xz:
        root = ET.fromstring(xz.read("xl/worksheets/sheet1.xml"))
    rows = root.findall(".//m:sheetData/m:row", ns)
    parsed = []
    for r in rows:
        vals = []
        for c in r.findall("m:c", ns):
            v = c.find("m:v", ns)
            vals.append(v.text if v is not None else None)
        if vals and any(v is not None for v in vals):
            parsed.append(vals)
    wl = np.array([float(x) for x in parsed[0][2:]], dtype=np.float64)
    ids, ys, xs = [], [], []
    for row in parsed[1:]:
        if len(row) < len(wl) + 2 or row[0] is None or row[1] is None:
            continue
        try:
            sid = int(float(row[0]))
            yv = float(row[1])
            xv = [float(v) for v in row[2 : 2 + len(wl)]]
        except (ValueError, TypeError):
            continue
        ids.append(sid)
        ys.append(yv)
        xs.append(xv)
    return np.array(ids, int), np.array(ys, np.float64), wl, np.array(xs, np.float64)


def load_idrc2016_wheat() -> dict:
    cache = os.path.join(CT_DIR, "idrc2016_wheat_parsed.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    frames = {}
    for man in ("A", "B", "C"):
        zpath = os.path.join(CT_DIR, f"idrc2016_wheat_{man}.zip")
        with zipfile.ZipFile(zpath) as z:
            tname = next(n for n in z.namelist() if n.lower().startswith("test") and n.endswith(".xlsx"))
            ids, y, wl, X = _read_xlsx_numeric(z.read(tname))
        frames[man] = {"ids": ids, "y": y, "wl": wl, "X": X}
    grid = frames["C"]["wl"]
    ids_ref = frames["A"]["ids"]
    out = {"ids": ids_ref, "wl": grid, "y": frames["A"]["y"][:, None]}
    for man in ("A", "B", "C"):
        pos = {int(s): i for i, s in enumerate(frames[man]["ids"])}
        take = np.array([pos[int(s)] for s in ids_ref])
        Xm = frames[man]["X"][take]
        ym = frames[man]["y"][take]
        assert np.allclose(ym, frames["A"]["y"], atol=1e-6), f"IDRC2016 y mismatch on {man}"
        cols = [int(np.flatnonzero(np.isclose(frames[man]["wl"], w, atol=1e-6))[0]) for w in grid]
        out[f"X_{man}"] = np.ascontiguousarray(Xm[:, cols])
    np.savez_compressed(cache, **out)
    return out


def load_neospectra_pairs() -> dict:
    cache = os.path.join(CT_DIR, "neospectra_pairs_parsed.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    lab_path = os.path.join(CT_DIR, "neospectra_soillab_v1.2.csv.gz")
    nir_path = os.path.join(CT_DIR, "neospectra_nir_v1.2.csv.gz")
    tcols = ["oc_usda.c729_w.pct", "clay.tot_usda.a334_w.pct", "ph.h2o_usda.a268_index"]
    y_map: dict[str, np.ndarray] = {}
    with gzip.open(lab_path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        i_id = hdr.index("id.sample_local_c")
        i_t = [hdr.index(c) for c in tcols]
        for row in rd:
            vals = []
            ok = True
            for j in i_t:
                s = row[j].strip()
                if s in ("", "NA", "NaN", "nan"):
                    ok = False
                    break
                vals.append(float(s))
            if ok:
                y_map[row[i_id].strip()] = np.array(vals, dtype=np.float64)
    # Average replicate scans per (device, sample_id)
    dev_scans: dict[str, dict[str, list[np.ndarray]]] = {}
    with gzip.open(nir_path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        i_id = hdr.index("id.sample_local_c")
        i_dev = hdr.index("scan.nir.model.name_utf8_txt")
        i_spec = [i for i, h in enumerate(hdr) if h.startswith("scan_nir.")]
        for row in rd:
            sid = row[i_id].strip()
            if sid not in y_map:
                continue
            dev = row[i_dev].strip()
            if dev not in ("NEO2", "NEO3", "NEO8", "NEO9"):
                continue
            try:
                sp = np.array([float(row[j]) for j in i_spec], dtype=np.float64)
            except ValueError:
                continue
            if not np.all(np.isfinite(sp)):
                continue
            dev_scans.setdefault(dev, {}).setdefault(sid, []).append(sp)
    out = {}
    for s_dev, t_dev in (("NEO2", "NEO3"), ("NEO8", "NEO9")):
        shared = sorted(set(dev_scans.get(s_dev, {})) & set(dev_scans.get(t_dev, {})))
        Xs = np.stack([np.mean(dev_scans[s_dev][sid], axis=0) for sid in shared])
        Xt = np.stack([np.mean(dev_scans[t_dev][sid], axis=0) for sid in shared])
        Y = np.stack([y_map[sid] for sid in shared])
        tag = f"{s_dev}_{t_dev}"
        out[f"{tag}_Xs"] = Xs
        out[f"{tag}_Xt"] = Xt
        out[f"{tag}_Y"] = Y
    np.savez_compressed(cache, **out)
    return out


def _nb_t(diffs: np.ndarray, n_te_over_tr: float) -> tuple[float, float]:
    """Nadeau & Bengio (2003) 校正重复 CV 配对 t 检验。"""
    m = len(diffs)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    se = math.sqrt((1.0 / m + n_te_over_tr) * max(var_d, 1e-18))
    t_val = mean_d / se
    p_val = float(2.0 * stats.t.sf(abs(t_val), df=m - 1))
    return round(t_val, 3), round(p_val, 5)


def eval_paired_cv_rigorous(Xs, Xt, Y, ks, kmax=10, reps=5, folds=5) -> dict:
    """5×5 CV 下同时汇报：① 均值；② 逐样本 OOF 配对 Wilcoxon；③ Nadeau-Bengio 校正 t 检验（ds_affine）。"""
    fn = PREPS["snv"]
    N = len(Y)
    yn = (Y - Y.mean(0)) / np.clip(Y.std(0), 1e-9, None)
    out = {}
    for k in ks:
        fold_rmsep = {h: [] for h in ("rnd", "ks", "dopt", "vopt", "t_auto")}
        oof_sq = {h: np.zeros((reps, N), dtype=np.float64) for h in fold_rmsep}
        alphas = []
        for rep in range(reps):
            perm = np.random.default_rng(rep).permutation(N)
            for va in np.array_split(perm, folds):
                tr = np.setdiff1d(perm, va)
                Xtr, Xte = fn(Xs[tr]), fn(Xt[va])
                pred, _ = fit_pls(Xtr, yn[tr], kmax=kmax, inner=4, per_col=False)
                sel_t, av = t_opt_select(Xtr, k, alpha="auto", lam=1e-3, return_alpha=True)
                alphas.append(av)
                sels = {
                    "rnd": tr[np.random.default_rng(2000 + rep).choice(len(tr), k, replace=False)],
                    "ks": tr[select_kennard_stone(Xtr, k)],
                    "dopt": tr[d_opt_select(Xtr, k)],
                    "vopt": tr[v_opt_select(Xtr, Xtr, k, lam=1e-3)],
                    "t_auto": tr[sel_t],
                }
                for h, sel in sels.items():
                    F, ms, mt = ds_affine(fn(Xs[sel]), fn(Xt[sel]))
                    yh = pred(ds_apply_affine(Xte, F, ms, mt))
                    err2 = np.mean((yh - yn[va]) ** 2, axis=1)
                    oof_sq[h][rep, va] = err2
                    fold_rmsep[h].append(float(np.sqrt(np.mean(err2))))
        # 逐样本均方误差（每个物理样本取 5 次重复均值 -> N 个独立物理样本）
        samp_err = {h: np.sqrt(oof_sq[h].mean(axis=0)) for h in fold_rmsep}
        row = {"alpha_auto": round(float(np.mean(alphas)), 3), "n_samples": N}
        for h in ("rnd", "ks", "dopt", "vopt", "t_auto"):
            row[h] = round(float(np.mean(fold_rmsep[h])), 4)
        for base in ("rnd", "ks", "dopt"):
            t_nb, p_nb = _nb_t(np.array(fold_rmsep["t_auto"]) - np.array(fold_rmsep[base]), 1.0 / (folds - 1))
            _, wp_samp = stats.wilcoxon(samp_err["t_auto"], samp_err[base])
            row[f"vs_{base}_pct"] = round((row["t_auto"] / row[base] - 1.0) * 100.0, 2)
            row[f"t_NB_vs_{base}"] = t_nb
            row[f"p_NB_vs_{base}"] = p_nb
            row[f"p_sample_wilcoxon_vs_{base}"] = round(float(wp_samp), 5)
        out[f"k={k}"] = row
    return out


def eval_disjoint_splits(Xs, Xt, Y, n_cal, n_unlab, n_test, ks=(10, 20, 50), n_splits=10, kmax=10) -> dict:
    """严格不相交外部切分（cal ∩ unlab ∩ test = ∅）：同时测配对 T-Opt 与无配对 Active(K)。"""
    fn = PREPS["snv"]
    yn = (Y - Y.mean(0)) / np.clip(Y.std(0), 1e-9, None)
    paired_res = {k: {h: [] for h in ("rnd", "ks", "dopt", "vopt", "t_auto")} for k in ks}
    unpaired_res = {h: [] for h in (
        "Bare", "MeanShift", "GatedMean", "SLOT_PLS",
        "MeanShift_K5", "SLOT_K5", "MeanShift_K10", "SLOT_K10", "MeanShift_K20", "SLOT_K20",
    )}
    alphas, rshifts, gates = [], [], []
    N = len(Y)
    for sd in range(n_splits):
        pm = np.random.default_rng(500 + sd).permutation(N)
        ical = pm[:n_cal]
        iun = pm[n_cal : n_cal + n_unlab]
        ite = pm[n_cal + n_unlab : n_cal + n_unlab + n_test]
        Xcal, Ycal = fn(Xs[ical]), yn[ical]
        Xun, Yun = fn(Xt[iun]), yn[iun]
        Xte, Yte = fn(Xt[ite]), yn[ite]
        pred, _ = fit_pls(Xcal, Ycal, kmax=kmax, inner=4, per_col=False)

        # 配对选样（在源池 ical 中选 k 条配对标准品）
        for k in ks:
            sel_t, av = t_opt_select(Xcal, k, alpha="auto", lam=1e-3, return_alpha=True)
            if k == ks[0]:
                alphas.append(av)
            for h, sel in [
                ("rnd", ical[np.random.default_rng(7000 + sd * 100 + k).choice(len(ical), k, replace=False)]),
                ("ks", ical[select_kennard_stone(Xcal, k)]),
                ("dopt", ical[d_opt_select(Xcal, k)]),
                ("vopt", ical[v_opt_select(Xcal, Xcal, k, lam=1e-3)]),
                ("t_auto", ical[sel_t]),
            ]:
                F, ms, mt = ds_affine(fn(Xs[sel]), fn(Xt[sel]))
                paired_res[k][h].append(rmsep(pred(ds_apply_affine(Xte, F, ms, mt)), Yte))

        # 无配对预对齐 + 目标侧 Active(K)
        ms_s, mt_u = Xcal.mean(0), Xun.mean(0)
        st_slot = subspace_lot_fit(Xcal, Xun, Ycal, rank=8, basis="pls")
        rshifts.append(st_slot["r_shift"])
        gates.append(st_slot["gate"])
        st_ms = dict(st_slot, A_r=np.eye(st_slot["rank"]), gate=1.0, tau=0.0)
        unpaired_res["Bare"].append(rmsep(pred(Xte), Yte))
        unpaired_res["MeanShift"].append(rmsep(pred(Xte + ms_s - mt_u), Yte))
        unpaired_res["GatedMean"].append(rmsep(pred(Xte + st_slot["gate"] * (ms_s - mt_u)), Yte))
        unpaired_res["SLOT_PLS"].append(rmsep(pred(subspace_lot_apply(Xte, st_slot)), Yte))
        for k_act in (5, 10, 20):
            if len(Xun) > k_act + 1:
                im = select_active_transfer(Xcal, Xun, k=k_act, st_lot=st_ms)
                unpaired_res[f"MeanShift_K{k_act}"].append(
                    rmsep(active_transfer_fit(Xun, im, Yun[im], pred, st_ms)(Xte), Yte)
                )
                is_ = select_active_transfer(Xcal, Xun, k=k_act, st_lot=st_slot)
                unpaired_res[f"SLOT_K{k_act}"].append(
                    rmsep(active_transfer_fit(Xun, is_, Yun[is_], pred, st_slot)(Xte), Yte)
                )

    out = {"alpha_auto": round(float(np.mean(alphas)), 3), "n_splits": n_splits, "paired_ds_affine": {}, "unpaired": {}}
    for k in ks:
        r = {h: round(float(np.mean(paired_res[k][h])), 4) for h in ("rnd", "ks", "dopt", "vopt", "t_auto")}
        for base in ("rnd", "ks", "dopt"):
            _, wp = stats.wilcoxon(paired_res[k]["t_auto"], paired_res[k][base])
            r[f"vs_{base}_pct"] = round((r["t_auto"] / r[base] - 1.0) * 100.0, 2)
            r[f"p_wilcoxon_vs_{base}"] = round(float(wp), 5)
        out["paired_ds_affine"][f"k={k}"] = r
    out["unpaired"] = {h: round(float(np.mean(v)), 4) for h, v in unpaired_res.items() if v}
    out["unpaired"]["r_shift_mean"] = round(float(np.mean(rshifts)), 3)
    out["unpaired"]["gate_mean"] = round(float(np.mean(gates)), 3)
    return out


def eval_tablets_official_460() -> dict:
    """Tablets 官方完全独立测试集（cal=155 -> test=460，零重叠），在 460 个物理测试药片上做逐样本配对 Wilcoxon。"""
    fn = PREPS["snv"]
    dt = load_tablets()
    cal = np.where(dt["part"] == 0)[0]
    te = np.where(dt["part"] == 1)[0]
    X1c, X2c = fn(dt["X1"][cal]), fn(dt["X2"][cal])
    X2te = fn(dt["X2"][te])
    ymu, ysd = dt["y"][cal].mean(0), dt["y"][cal].std(0)
    yc = (dt["y"][cal] - ymu) / ysd
    yte = (dt["y"][te] - ymu) / ysd
    pred, _ = fit_pls(X1c, yc, kmax=10, inner=4, per_col=False)
    out = {"n_cal": int(len(cal)), "n_test": int(len(te)), "alpha_auto": round(estimate_spectral_temper_alpha(X1c), 3)}
    for k in (10, 20, 50):
        # Random 取 20 次独立抽样的逐样本均方误差均值作为稳定随机基线
        rnd_sq = np.zeros(len(te), dtype=np.float64)
        for sd in range(20):
            s_r = np.random.default_rng(9000 + sd).choice(len(cal), k, replace=False)
            F, ms, mt = ds_affine(X1c[s_r], X2c[s_r])
            rnd_sq += np.mean((pred(ds_apply_affine(X2te, F, ms, mt)) - yte) ** 2, axis=1) / 20.0
        samp_e = {"rnd": np.sqrt(rnd_sq)}
        rmsep_v = {"rnd": float(np.sqrt(np.mean(rnd_sq)))}
        for h, sel in [
            ("ks", select_kennard_stone(X1c, k)),
            ("dopt", d_opt_select(X1c, k)),
            ("vopt", v_opt_select(X1c, X1c, k, lam=1e-3)),
            ("t_auto", t_opt_select(X1c, k, alpha="auto", lam=1e-3)),
        ]:
            F, ms, mt = ds_affine(X1c[sel], X2c[sel])
            yh = pred(ds_apply_affine(X2te, F, ms, mt))
            e2 = np.mean((yh - yte) ** 2, axis=1)
            samp_e[h] = np.sqrt(e2)
            rmsep_v[h] = float(np.sqrt(np.mean(e2)))
        row = {h: round(rmsep_v[h], 4) for h in ("rnd", "ks", "dopt", "vopt", "t_auto")}
        for base in ("rnd", "ks", "dopt"):
            _, wp = stats.wilcoxon(samp_e["t_auto"], samp_e[base])
            row[f"vs_{base}_pct"] = round((row["t_auto"] / row[base] - 1.0) * 100.0, 2)
            row[f"p_460sample_wilcoxon_vs_{base}"] = round(float(wp), 5)
        out[f"k={k}"] = row
    return out


def eval_fixed_var_snv_coupling() -> dict:
    """严格固定 std(a)=0.35 的 SNV 散射-浓度耦合对照实验（3 个独立低散射库：Cordoba Wheat/Barley + SWRI Diesel）。"""
    cb = load_cordoba_all()
    swri = scipy.io.loadmat(os.path.join(os.path.dirname(CT_DIR), "diesel", "SWRI_Diesel_NIR.mat"))
    X_dies = np.asarray(swri["diesel_spec"][0, 0]["data"], np.float64)
    y_dies = np.asarray(swri["diesel_prop"][0, 0]["data"], np.float64)[:, [1]]  # CN (Cetane Number)
    ok_d = np.isfinite(X_dies).all(1) & np.isfinite(y_dies).all(1)
    X_dies, y_dies = X_dies[ok_d], y_dies[ok_d]
    # 子抽 200 条 SWRI Diesel 以快速 5 折评估
    idx_d = np.random.default_rng(7).choice(len(y_dies), min(200, len(y_dies)), replace=False)

    datasets = [
        ("cordoba_wheat", np.asarray(cb["wheat"]["X"], np.float64), np.asarray(cb["wheat"]["y"], np.float64)),
        ("cordoba_barley", np.asarray(cb["barley"]["X"], np.float64), np.asarray(cb["barley"]["y"], np.float64)),
        ("swri_diesel_CN", X_dies[idx_d], y_dies[idx_d]),
    ]
    out = {}
    sigma_a = 0.35
    for name, X0, y0 in datasets:
        if y0.ndim == 1:
            y0 = y0[:, None]
        ok = np.isfinite(X0).all(1) & np.isfinite(y0).all(1)
        X0, y0 = X0[ok], y0[ok]
        N = len(y0)
        ys = (y0[:, 0] - y0[:, 0].mean()) / y0[:, 0].std()

        def _cv(Xin, prep):
            fn = PREPS[prep]
            vals = []
            for rep in range(3):
                perm = np.random.default_rng(900 + rep).permutation(N)
                for va in np.array_split(perm, 5):
                    tr = np.setdiff1d(perm, va)
                    pred, _ = fit_pls(fn(Xin[tr]), ys[tr][:, None], kmax=10, inner=4, per_col=False)
                    vals.append(rmsep(pred(fn(Xin[va])), ys[va][:, None]))
            return float(np.mean(vals))

        base_raw, base_snv = _cv(X0, "raw"), _cv(X0, "snv")
        rows = {"baseline_no_inject": {"raw": round(base_raw, 4), "snv": round(base_snv, 4),
                                       "gain_pct": round((base_snv / base_raw - 1.0) * 100.0, 2)}}
        for r_tgt in (0.0, 0.2, 0.4, 0.6, 0.8):
            rng = np.random.default_rng(51)
            eps = rng.normal(0.0, 1.0, size=N)
            eps = eps - (np.dot(eps, ys) / np.dot(ys, ys)) * ys
            eps = eps / eps.std()
            z = r_tgt * ys + math.sqrt(1.0 - r_tgt ** 2) * eps
            a = np.clip(1.0 + sigma_a * z, 0.2, 5.0)
            Xi = a[:, None] * X0
            p_raw, p_snv = _cv(Xi, "raw"), _cv(Xi, "snv")
            rows[f"r={r_tgt:.1f}"] = {
                "actual_corr": round(float(abs(np.corrcoef(a, ys)[0, 1])), 3),
                "std_a": round(float(a.std()), 3),
                "raw": round(p_raw, 4),
                "snv": round(p_snv, 4),
                "gain_pct": round((p_snv / p_raw - 1.0) * 100.0, 2),
            }
        out[name] = rows
        print(f"SNV_fixed_var [{name}]:", {k: v["gain_pct"] for k, v in rows.items()}, flush=True)
    return out


def main():
    report = {}

    print("== [1] OSSL v1.2 NeoSpectra Handheld Soil NIR (Blind External, Strictly Disjoint) ==", flush=True)
    neo = load_neospectra_pairs()
    report["external_neospectra"] = {}
    for pair in ("NEO2_NEO3", "NEO8_NEO9"):
        Xs, Xt, Y = neo[f"{pair}_Xs"], neo[f"{pair}_Xt"], neo[f"{pair}_Y"]
        print(f"  {pair}: N={len(Y)}, L={Xs.shape[1]}, p={Y.shape[1]}", flush=True)
        r = eval_disjoint_splits(Xs, Xt, Y, n_cal=250, n_unlab=200, n_test=450, ks=(10, 20, 50), n_splits=10, kmax=10)
        report["external_neospectra"][pair] = r
        print("   paired:", r["paired_ds_affine"], flush=True)
        print("   unpaired:", r["unpaired"], flush=True)

    print("\n== [2] IDRC-2016 Cross-Manufacturer Wheat Shootout (Blind External, 3 Manufacturers) ==", flush=True)
    wh = load_idrc2016_wheat()
    report["external_idrc2016_wheat"] = {}
    for src, tgt in (("A", "B"), ("A", "C"), ("B", "C"), ("C", "A")):
        Xs, Xt, Y = wh[f"X_{src}"], wh[f"X_{tgt}"], wh["y"]
        r_cv = eval_paired_cv_rigorous(Xs, Xt, Y, ks=(10, 20, 40), kmax=10)
        r_dis = eval_disjoint_splits(Xs, Xt, Y, n_cal=110, n_unlab=60, n_test=78, ks=(10, 20), n_splits=10, kmax=10)
        report["external_idrc2016_wheat"][f"{src}->{tgt}"] = {"paired_cv_rigorous": r_cv, "disjoint_10split": r_dis}
        print(f"  Wheat {src}->{tgt} (a={r_dis['alpha_auto']}):", r_cv, flush=True)
        print(f"    unpaired:", r_dis["unpaired"], flush=True)

    print("\n== [3] Existing Benchmarks Statistical Hygiene Fix (ds_affine, OOF Sample Wilcoxon & NB-t & Disjoint 460) ==", flush=True)
    dc = load_corn()
    sd_mat = scipy.io.loadmat(os.path.join(CT_DIR, "sdop_nir_data.mat"))
    report["existing_rigorous_stats_ds_affine"] = {
        "corn_m5_to_mp6": eval_paired_cv_rigorous(dc["m5"], dc["mp6"], dc["y"], ks=(10, 20), kmax=10),
        "tablets_official_disjoint_460": eval_tablets_official_460(),
        "sdop_1_to_2": eval_paired_cv_rigorous(sd_mat["spec1"], sd_mat["spec2"], sd_mat["conc"], ks=(8, 12, 16), kmax=8),
    }
    print("  Corn rigorous:", report["existing_rigorous_stats_ds_affine"]["corn_m5_to_mp6"], flush=True)
    print("  Tablets 460 disjoint:", report["existing_rigorous_stats_ds_affine"]["tablets_official_disjoint_460"], flush=True)
    print("  SDOP rigorous:", report["existing_rigorous_stats_ds_affine"]["sdop_1_to_2"], flush=True)

    print("\n== [4] Fixed-Variance SNV Coupling Mechanism (Cordoba Wheat / Barley + SWRI Diesel) ==", flush=True)
    report["fixed_var_snv_coupling"] = eval_fixed_var_snv_coupling()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res_selfcheck_external_2026_11_19.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\nSaved:", out_path, flush=True)


if __name__ == "__main__":
    main()
