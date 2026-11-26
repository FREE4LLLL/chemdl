"""Round 3 零接触外部公开 Zenodo NIR 实验室数据集拉取与 SHA-256 清单生成脚本。

全部候选项均经许可证核验（仅允许 CC-BY 4.0 / CC0 1.0 / MIT），下载至未入库的
`data_ext/round3_unseen/` 并生成可校验的 `data/MANIFEST_ROUND3.yaml`。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_ext" / "round3_unseen"
MANIFEST_PATH = ROOT / "data" / "MANIFEST_ROUND3.yaml"

ROUND3_DATASETS = [
    {
        "id": 1,
        "key": "Z-16783068_mlnir_hydrocarbon",
        "tier": "primary",
        "name": "MLNIRdata Hydrocarbon Mixtures Density NIR (208 samples)",
        "lab": "IFP Energies nouvelles (France)",
        "license": "CC-BY 4.0",
        "doi": "10.5281/zenodo.16783068",
        "record_url": "https://zenodo.org/records/16783068",
        "subdir": "Z16783068_mlnir",
        "files": [
            (
                "MLNIRdata_matrixXY_NirSpectrum_DensityNormalized.mat",
                "https://zenodo.org/api/records/16783068/files/MLNIRdata_matrixXY_NirSpectrum_DensityNormalized.mat/content",
            ),
            (
                "MLNIR_matrixX_NirSpectrumDataAxis.csv",
                "https://zenodo.org/api/records/16783068/files/MLNIR_matrixX_NirSpectrumDataAxis.csv/content",
            ),
        ],
    },
    {
        "id": 2,
        "key": "Z-4991924_woody_plant_nsc",
        "tier": "primary",
        "name": "Tree Species Woody Tissue Non-Structural Carbohydrates NIRS",
        "lab": "Dryad / Zenodo Woody Plant Physiology Lab",
        "license": "CC0 1.0",
        "doi": "10.5061/dryad.jd2cb",
        "record_url": "https://zenodo.org/records/4991924",
        "subdir": "Z4991924_tree_nsc",
        "files": [
            ("NSC_data.csv", "https://zenodo.org/api/records/4991924/files/NSC_data.csv/content"),
            ("info_samples.csv", "https://zenodo.org/api/records/4991924/files/info_samples.csv/content"),
            ("spectra_data.csv", "https://zenodo.org/api/records/4991924/files/spectra_data.csv/content"),
        ],
    },
    {
        "id": 3,
        "key": "Z-7980675_wood_rot_nir",
        "tier": "primary",
        "name": "Heat-Treated Wood Fungal Degradation NIR Spectra",
        "lab": "Aalto University (Finland)",
        "license": "CC-BY 4.0",
        "doi": "10.5281/zenodo.7980675",
        "record_url": "https://zenodo.org/records/7980675",
        "subdir": "Z7980675_wood_rot",
        "files": [
            ("NIR spectra.csv", "https://zenodo.org/api/records/7980675/files/NIR%20spectra.csv/content"),
            (
                "Sample IDs and mass data.csv",
                "https://zenodo.org/api/records/7980675/files/Sample%20IDs%20and%20mass%20data.csv/content",
            ),
        ],
    },
    {
        "id": 4,
        "key": "Z-16811137_residual_oil",
        "tier": "primary",
        "name": "Residual Oil Complex Components NIR Analysis",
        "lab": "Residual Oil NIR Analysis Lab",
        "license": "MIT",
        "doi": "10.5281/zenodo.16811137",
        "record_url": "https://zenodo.org/records/16811137",
        "subdir": "Z16811137_residual_oil",
        "files": [
            (
                "residual-oil-nir-analysis-1.0.1.zip",
                "https://zenodo.org/api/records/16811137/files/residual-oil-nir-analysis-1.0.1.zip/content",
            ),
        ],
    },
    {
        "id": 5,
        "key": "Z-4609414_walnut_moisture",
        "tier": "primary",
        "name": "Walnut Kernel Moisture NIR Spectra",
        "lab": "Dryad / Zenodo Walnut Kernel Lab",
        "license": "CC0 1.0",
        "doi": "10.5061/dryad.xksn02vfp",
        "record_url": "https://zenodo.org/records/4609414",
        "subdir": "Z4609414_walnut",
        "files": [
            (
                "Measured_value+Raw_spectral_data.xlsx",
                "https://zenodo.org/api/records/4609414/files/Measured_value+Raw_spectral_data.xlsx/content",
            ),
        ],
    },
    {
        "id": 6,
        "key": "Z-7809305_pea_forage",
        "tier": "primary",
        "name": "Pea Breeding Program Forage Quality Spectra (2019 -> 2020)",
        "lab": "Pea Breeding Hyperspectral / NIRS Lab",
        "license": "CC-BY 4.0",
        "doi": "10.5281/zenodo.7809305",
        "record_url": "https://zenodo.org/records/7809305",
        "subdir": "Z7809305_pea_forage",
        "files": [
            (
                "2019_GN_Hyperspectral&GroundReferenceData.xlsx",
                "https://zenodo.org/api/records/7809305/files/"
                + urllib.parse.quote("2019_GN_Hyperspectral&GroundReferenceData.xlsx", safe="")
                + "/content",
            ),
            (
                "2020_PM_Hyperspectral&GroundReferenceData.xlsx",
                "https://zenodo.org/api/records/7809305/files/"
                + urllib.parse.quote("2020_PM_Hyperspectral&GroundReferenceData.xlsx", safe="")
                + "/content",
            ),
        ],
    },
    {
        "id": 7,
        "key": "Z-17941270_austrian_soil",
        "tier": "primary",
        "name": "Austrian NIR Soil Spectral Library (680-2500 nm)",
        "lab": "BOKU / Austrian Soil Health Lab",
        "license": "CC-BY 4.0",
        "doi": "10.5281/zenodo.17941270",
        "record_url": "https://zenodo.org/records/17941270",
        "subdir": "Z17941270_austrian_soil",
        "files": [
            (
                "Austrian NIR Soil Spectral Library_V4.xlsx",
                "https://zenodo.org/api/records/17941270/files/Austrian%20NIR%20Soil%20Spectral%20Library_V4.xlsx/content",
            ),
        ],
    },
    {
        "id": 8,
        "key": "Z-14177889_grape_ripeness",
        "tier": "primary",
        "name": "Tempranillo Grapes Multispectral Ripeness Sensor",
        "lab": "Rioja / Spain Viticulture Sensor Lab",
        "license": "CC-BY 4.0",
        "doi": "10.5281/zenodo.14177889",
        "record_url": "https://zenodo.org/records/14177889",
        "subdir": "Z14177889_grape",
        "files": [
            ("Dataset_sensor.csv", "https://zenodo.org/api/records/14177889/files/Dataset_sensor.csv/content"),
        ],
    },
    {
        "id": 9,
        "key": "Z-20702109_helsinki_soil",
        "tier": "backup",
        "name": "Field & Imaging Soil Organic Carbon and Nitrogen Spectroscopy",
        "lab": "University of Helsinki",
        "license": "CC-BY 4.0",
        "doi": "10.5281/zenodo.20702109",
        "record_url": "https://zenodo.org/records/20702109",
        "subdir": "Z20702109_helsinki_soil",
        "files": [
            ("Soil spectra.zip", "https://zenodo.org/api/records/20702109/files/Soil%20spectra.zip/content"),
            ("Soil_reference_data.csv", "https://zenodo.org/api/records/20702109/files/Soil_reference_data.csv/content"),
            ("Metadata.txt", "https://zenodo.org/api/records/20702109/files/Metadata.txt/content"),
        ],
    },
    {
        "id": 10,
        "key": "Z-8182624_italy_poplar_soil",
        "tier": "backup",
        "name": "Northern Italy Poplar Plains Soil NIR Dataset",
        "lab": "Northern Italy Poplar Forestry Lab",
        "license": "CC-BY 4.0",
        "doi": "10.5281/zenodo.8182624",
        "record_url": "https://zenodo.org/records/8182624",
        "subdir": "Z8182624_italy_soil",
        "files": [
            ("Soil Dataset.xlsx", "https://zenodo.org/api/records/8182624/files/Soil%20Dataset.xlsx/content"),
        ],
    },
    {
        "id": 11,
        "key": "Z-14868248_field_visnir_soil",
        "tier": "backup",
        "name": "Field Vis-NIR Soil SOM and Total Nitrogen Dataset",
        "lab": "Field Vis-NIR Soil & RS Fusion Lab",
        "license": "CC-BY 4.0",
        "doi": "10.5281/zenodo.14868248",
        "record_url": "https://zenodo.org/records/14868248",
        "subdir": "Z14868248_field_soil",
        "files": [
            (
                "soil propertie and spectral data.xlsx",
                "https://zenodo.org/api/records/14868248/files/soil%20propertie%20and%20spectral%20data.xlsx/content",
            ),
        ],
    },
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _download_with_retry(url: str, dst: Path, retries: int = 2) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ChemDL-Round3Fetcher/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = getattr(resp, "status", 200)
                if status != 200:
                    raise RuntimeError(f"HTTP {status} for {url}")
                with open(tmp, "wb") as out:
                    while chunk := resp.read(1 << 18):
                        out.write(chunk)
            os.replace(tmp, dst)
            return
        except Exception as exc:
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            print(f"  [retry {attempt}/{retries}] {url}: {exc}", flush=True)
            time.sleep(1.5)
    raise RuntimeError(f"Failed downloading {url}: {last_err}")


def _write_manifest(active_entries: list[dict]) -> None:
    lines = [
        "# ChemDL Round 3 Zero-Contact Unseen External NIR Dataset Manifest (full 64-char SHA-256)",
        "# Generated by: python _audit/fetch_round3_unseen.py",
        "datasets:",
    ]
    for ds in active_entries:
        ds_dir = DATA_DIR / ds["subdir"]
        if not ds_dir.exists():
            continue
        lines += [
            f"  - id: {ds['id']}",
            f"    key: \"{ds['key']}\"",
            f"    tier: \"{ds['tier']}\"",
            f"    name: \"{ds['name']}\"",
            f"    lab: \"{ds['lab']}\"",
            f"    license: \"{ds['license']}\"",
            f"    doi: \"{ds['doi']}\"",
            f"    url: \"{ds['record_url']}\"",
            f"    files:",
        ]
        for fname, url in ds["files"]:
            fp = ds_dir / fname
            if not fp.exists():
                raise FileNotFoundError(f"Missing downloaded Round 3 file: {fp}")
            lines += [
                f"      - name: \"{fname}\"",
                f"        rel_path: \"data_ext/round3_unseen/{ds['subdir']}/{fname}\"",
                f"        size_bytes: {fp.stat().st_size}",
                f"        sha256: \"{_sha256_file(fp)}\"",
            ]
    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-backups", action="store_true", help="Also download backup CC-BY 4.0 datasets.")
    parser.add_argument("--verify-only", action="store_true", help="Verify downloaded files match MANIFEST_ROUND3.yaml.")
    args = parser.parse_args()

    if args.verify_only:
        import re

        txt = MANIFEST_PATH.read_text(encoding="utf-8")
        rels = re.findall(r'rel_path:\s*"([^"]+)"', txt)
        shas = re.findall(r'sha256:\s*"([0-9a-f]{64})"', txt)
        assert len(rels) == len(shas) and len(rels) > 0
        for r, s in zip(rels, shas):
            p = ROOT / r
            assert p.exists(), f"Missing {p}"
            got = _sha256_file(p)
            assert got == s, f"SHA-256 mismatch on {r}: {got} vs {s}"
        print(f"Round 3 MANIFEST verified: {len(rels)} files match 64-hex SHA-256.")
        return

    targets = [ds for ds in ROUND3_DATASETS if args.include_backups or ds["tier"] == "primary"]
    active = []
    for ds in targets:
        print(f"[{ds['id']}] {ds['key']} ({ds['license']}) ...", flush=True)
        for fname, url in ds["files"]:
            dst = DATA_DIR / ds["subdir"] / fname
            if dst.exists() and dst.stat().st_size > 0:
                print(f"  cached: {dst.name} ({dst.stat().st_size / 1024:.1f} KB)", flush=True)
            else:
                print(f"  downloading: {fname} <- {url}", flush=True)
                _download_with_retry(url, dst)
                print(f"  saved: {dst.name} ({dst.stat().st_size / 1024:.1f} KB)", flush=True)
        active.append(ds)

    _write_manifest(active)
    print(f"Wrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
