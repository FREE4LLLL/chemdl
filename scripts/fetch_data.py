"""公开 NIR 基准数据集清单、全长 SHA-256 校验与自动下载脚本（闭环 REVIEW_2026-11-23 P2-3 & P0-3）。

覆盖 `NUMBER_BASELINE.md` 登记的全部 13 套独立公开数据集：
  - `--verify`：核对本地 `data_ext/` 与 `data_raw/` 中全部 13 套数据集存在性及 `data/MANIFEST.yaml` 中的全长 64 位 SHA-256；
  - `--write-manifest`：从当前已校验本地数据生成 `data/MANIFEST.yaml`（纯标准库生成/解析，零额外依赖）；
  - `--download`：自动拉取支持官方直链的 Zenodo CC-BY 4.0 / 开放基准文件（仅缺项）；
  - `--manifest`：输出含官方 URL/DOI 与许可证的 Markdown 溯源表。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "data", "MANIFEST.yaml")

DATASETS = [
    {
        "id": 1,
        "name": "Tecator Meat NIR (240x100)",
        "category": "classic",
        "license": "Public Domain Academic Archive (StatLib / OpenML:505)",
        "rel_path": "data_ext/tecator.txt",
        "url": "https://lib.stat.cmu.edu/datasets/tecator",
        "direct_url": None,
    },
    {
        "id": 2,
        "name": "Eigenvector Corn (m5/mp5/mp6, 80x700)",
        "category": "classic",
        "license": "Public Academic Benchmark (Eigenvector Research)",
        "rel_path": "data_raw/corn.mat",
        "url": "https://eigenvector.com/resources/data-sets/",
        "direct_url": None,
    },
    {
        "id": 3,
        "name": "IDRC 2002 Shootout Tablets (X1/X2, 655x650)",
        "category": "classic",
        "license": "Public Academic Shootout Benchmark (Eigenvector / IDRC)",
        "rel_path": "data_ext/tablets.zip",
        "url": "https://eigenvector.com/resources/data-sets/",
        "direct_url": None,
    },
    {
        "id": 4,
        "name": "Cordoba Grain NIR — Wheat (149x700)",
        "category": "classic",
        "license": "CC-BY 4.0 (sensAIfood / Univ. of Cordoba, Zenodo 15838136)",
        "rel_path": "data_ext/cordoba",
        "url": "https://doi.org/10.5281/zenodo.15838136",
        "direct_url": None,
    },
    {
        "id": 5,
        "name": "Cordoba Grain NIR — Barley (155x700)",
        "category": "classic",
        "license": "CC-BY 4.0 (sensAIfood / Univ. of Cordoba, Zenodo 15838136)",
        "rel_path": "data_ext/cordoba",
        "url": "https://doi.org/10.5281/zenodo.15838136",
        "direct_url": None,
    },
    {
        "id": 6,
        "name": "Cordoba Grain NIR — Maize (164x700)",
        "category": "classic",
        "license": "CC-BY 4.0 (sensAIfood / Univ. of Cordoba, Zenodo 15838136)",
        "rel_path": "data_ext/cordoba",
        "url": "https://doi.org/10.5281/zenodo.15838136",
        "direct_url": None,
    },
    {
        "id": 7,
        "name": "SWRI Diesel Fuels NIR (200x401, CN)",
        "category": "classic",
        "license": "Public Academic Benchmark (Eigenvector Research / SWRI)",
        "rel_path": "data_ext/diesel/SWRI_Diesel_NIR.mat",
        "url": "https://eigenvector.com/resources/data-sets/",
        "direct_url": None,
    },
    {
        "id": 8,
        "name": "Anderson (2020) Mango Dry Matter (11,863x103)",
        "category": "industrial",
        "license": "CC-BY 4.0 (CQUniversity / Mendeley Data)",
        "rel_path": "data_ext/ct_bench/mango.mat",
        "url": "https://doi.org/10.17632/b9d6s334s5.1",
        "direct_url": None,
    },
    {
        "id": 9,
        "name": "IDRC 2016 Wheat (A/B/C & A1, 248x741)",
        "category": "industrial",
        "license": "Public Academic Competition Dataset (CNIRS / IDRC 2016)",
        "rel_path": "data_ext/ct_bench/wheatA1_20048.mat",
        "url": "https://www.cnirs.org/content.aspx?page_id=22&club_id=409746&module_id=239453",
        "direct_url": None,
    },
    {
        "id": 10,
        "name": "Melamine-Formaldehyde Resin FT-NIR (8,127x225, 4 recipes)",
        "category": "industrial",
        "license": "MIT License (RNL1/Melamine-Dataset; Metadynea GmbH; Anal. Chem. 2018)",
        "rel_path": "data_ext/ct_bench/Melamine_Dataset.mat",
        "url": "https://github.com/RNL1/Melamine-Dataset",
        "direct_url": None,
    },
    {
        "id": 11,
        "name": "OSSL NeoSpectra Soil MEMS (8,151x601)",
        "category": "industrial",
        "license": "CC-BY 4.0 (Open Soil Spectral Library)",
        "rel_path": "data_ext/ct_bench/neospectra_nir_v1.2.csv.gz",
        "url": "https://explorer.soilspectroscopy.org",
        "direct_url": None,
    },
    {
        "id": 12,
        "name": "CRA-W Industrial Cereal NIR 27-Year Archive (5,046x700)",
        "category": "industrial",
        "license": "CC-BY 4.0 (Zenodo 16108496)",
        "rel_path": "data_ext/industrial/sensAIfood_CRAW_zenodo16108496.zip",
        "url": "https://doi.org/10.5281/zenodo.16108496",
        "direct_url": "https://zenodo.org/records/16108496/files/sensAIfood.zip?download=1",
    },
    {
        "id": 13,
        "name": "On-Farm Raw Milk Transmittance NIR (1,224x256)",
        "category": "industrial",
        "license": "CC-BY 4.0 (Zenodo 8263430 / Data in Brief 2023)",
        "rel_path": "data_ext/industrial/milk_transmittance_zenodo8263430.csv",
        "url": "https://doi.org/10.5281/zenodo.8263430",
        "direct_url": "https://zenodo.org/records/8263430/files/data_milk_NIR.csv?download=1",
    },
    {
        # codex audit 09: the OSSL soil task merges reference labels from this
        # companion file on id.sample_local_c. Without it the task cannot be
        # rebuilt, so it must be inside the integrity gate.
        "id": 14,
        "name": "OSSL NeoSpectra Soil Lab Labels (companion to id 11)",
        "category": "industrial",
        "license": "CC-BY 4.0 (Zenodo 4719801 / OSSL)",
        "rel_path": "data_ext/ct_bench/neospectra_soillab_v1.2.csv.gz",
        "url": "https://doi.org/10.5281/zenodo.4719801",
        "direct_url": "",
    },
    {
        # codex full-audit E30: 跨仪器小麦迁移会读 A3，此前不在门禁内。
        "id": 15,
        "name": "IDRC 2016 Wheat A3 (48->200 companion set)",
        "category": "public",
        "license": "Open benchmark (IDRC 2016 shootout)",
        "rel_path": "data_ext/ct_bench/wheatA3_48200.mat",
        "url": "https://doi.org/10.1016/j.chemolab.2007.03.003",
        "direct_url": "",
    },
]


def _sha256_full(path: str, chunk_size: int = 65536) -> str:
    """Compute the full 64-char SHA-256 hex digest over the entire file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hash_entry(full: str) -> tuple[int, str, int]:
    """Return (total_bytes, full_sha256_hex, n_files) for a local dataset file or dir."""
    if os.path.isfile(full):
        return os.path.getsize(full), _sha256_full(full), 1
    if os.path.isdir(full):
        files = sorted(f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f)))
        if not files:
            return 0, "", 0
        total_b = 0
        h_dir = hashlib.sha256()
        for fn in files:
            fp = os.path.join(full, fn)
            total_b += os.path.getsize(fp)
            h_dir.update(f"{fn}:{_sha256_full(fp)}\n".encode("utf-8"))
        return total_b, h_dir.hexdigest(), len(files)
    return 0, "", 0


def _load_manifest_hashes() -> dict[int, dict]:
    """Parse data/MANIFEST.yaml using stdlib only (zero PyYAML dependency)."""
    if not os.path.exists(MANIFEST_PATH):
        return {}
    entries: dict[int, dict] = {}
    cur: dict | None = None
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("  - id:"):
                cur = {"id": int(line.split(":", 1)[1].strip())}
                entries[cur["id"]] = cur
            elif cur is not None and line.startswith("    ") and ":" in line:
                k, v = line.strip().split(":", 1)
                v = v.strip()
                if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                    v = v[1:-1]
                cur[k] = int(v) if k in ("size_bytes", "n_files") else v
    return entries


def write_manifest() -> int:
    """Compute full SHA-256 digests of all local datasets and write data/MANIFEST.yaml.

    codex full-audit E15: 早先边算边写，缺数据时会把清单里的 sha256 覆盖成空串，
    破坏后续用于校验的基线。现在先在内存完成全部摘要，有缺失就**不动**原文件。
    """
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    out = [
        "# ChemDL Dataset Integrity & License Manifest (full 64-char SHA-256)",
        "# Generated by: python scripts/fetch_data.py --write-manifest",
        "datasets:",
    ]
    missing = 0
    for ds in DATASETS:
        full = os.path.join(ROOT, ds["rel_path"])
        sz, digest, n_files = _hash_entry(full)
        if not digest:
            missing += 1
        out += [
            f"  - id: {ds['id']}",
            f"    name: \"{ds['name']}\"",
            f"    category: \"{ds['category']}\"",
            f"    rel_path: \"{ds['rel_path']}\"",
            f"    url: \"{ds['url']}\"",
            f"    license: \"{ds['license']}\"",
            f"    n_files: {n_files}",
            f"    size_bytes: {sz}",
            f"    sha256: \"{digest}\"",
        ]
    if missing:
        # 有缺失就整体放弃，保留磁盘上的旧清单（它仍是可用的校验基线）
        print(f"ABORT: {missing}/{len(DATASETS)} datasets missing; "
              f"{MANIFEST_PATH} left unchanged.")
        return 1
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, MANIFEST_PATH)
    print(f"Wrote {MANIFEST_PATH} ({len(DATASETS)}/{len(DATASETS)} populated).")
    return 0


def verify_local() -> int:
    manifest = _load_manifest_hashes()
    print(f"=== Verifying all {len(DATASETS)} dataset entries (13 independent public datasets) under {ROOT} ===")
    failed = 0
    for ds in DATASETS:
        full = os.path.join(ROOT, ds["rel_path"])
        sz, digest, n_files = _hash_entry(full)
        if not digest:
            status = "MISSING"
            failed += 1
        else:
            exp = manifest.get(ds["id"], {})
            exp_hash = exp.get("sha256", "")
            size_mb = sz / (1024 * 1024)
            tag = f"{n_files} files, " if os.path.isdir(full) else ""
            if exp_hash and digest != exp_hash:
                status = f"HASH MISMATCH ({tag}{size_mb:.2f} MB, got={digest[:12]}, want={exp_hash[:12]})"
                failed += 1
            elif exp_hash:
                status = f"OK ({tag}{size_mb:.2f} MB, sha256={digest[:16]}... [full 64-hex verified])"
            else:
                # codex 审计 09：清单缺预期哈希时不能报 OK——那等于「无法校验」被
                # 当成「校验通过」。缺条目即判失败，避免完整性门禁形同虚设。
                status = f"FAIL [manifest 缺该数据集条目，无法校验] ({tag}{size_mb:.2f} MB, sha256={digest[:16]}...)"
                failed += 1
        print(f"  [{ds['id']:2d}] {ds['name']:58s} | {status}")
    print(f"\nSummary: {len(DATASETS) - failed}/{len(DATASETS)} datasets verified locally.")
    return 0 if failed == 0 else 1


def download_direct(timeout: int = 60):
    manifest = _load_manifest_hashes()
    for ds in DATASETS:
        if not ds["direct_url"]:
            continue
        dst = os.path.join(ROOT, ds["rel_path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        exp_hash = str(manifest.get(ds["id"], {}).get("sha256", ""))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            if not exp_hash or _sha256_full(dst) == exp_hash:
                print(f"  [SKIP] {ds['rel_path']} already exists" + (" (SHA-256 verified)." if exp_hash else "."))
                continue
            print(f"  [WARN] {ds['rel_path']} SHA-256 mismatch; re-downloading safely.")
        part = dst + ".part"
        print(f"  [GET ] {ds['direct_url']} -> {ds['rel_path']}")
        try:
            with urllib.request.urlopen(ds["direct_url"], timeout=timeout) as resp, open(part, "wb") as out_f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out_f.write(chunk)
            if exp_hash:
                got = _sha256_full(part)
                if got != exp_hash:
                    raise RuntimeError(
                        f"SHA-256 mismatch for {ds['rel_path']}: got {got}, expected {exp_hash}"
                    )
            os.replace(part, dst)
        except Exception:
            if os.path.exists(part):
                os.remove(part)
            raise
    print("Direct downloads complete. For portal-hosted datasets, run --manifest for official DOIs.")


def print_manifest():
    print("| # | Dataset | Category | License | Official URL / DOI | Local Path |")
    print("|---|---|---|---|---|---|")
    for ds in DATASETS:
        print(
            f"| {ds['id']} | {ds['name']} | {ds['category']} | {ds['license']} | "
            f"{ds['url']} | `{ds['rel_path']}` |"
        )


def main():
    ap = argparse.ArgumentParser(description="ChemDL 13-dataset manifest, verifier, and downloader")
    ap.add_argument("--verify", action="store_true", help="Verify local dataset population & SHA-256")
    ap.add_argument("--write-manifest", action="store_true", help="Write full SHA-256 digests to data/MANIFEST.yaml")
    ap.add_argument("--download", action="store_true", help="Download direct-URL Zenodo/open datasets")
    ap.add_argument("--manifest", action="store_true", help="Print markdown manifest table")
    args = ap.parse_args()
    if not (args.verify or args.download or args.manifest or args.write_manifest):
        args.verify = True
    if args.manifest:
        print_manifest()
    if args.download:
        download_direct()
    if args.write_manifest:
        rc = write_manifest()
        if not args.verify:
            sys.exit(rc)
    if args.verify:
        sys.exit(verify_local())


if __name__ == "__main__":
    main()
