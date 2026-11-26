"""开源农业谷物 NIR 数据库：Zenodo `sensAIfood` 科尔多瓦大学（Univ. Córdoba）多谷物数据库。

包含三种核心谷物在 FOSS NIRSystem 5000 上的 700 通道吸收光谱（1100–2498 nm，步长 2 nm）
与水分（Moisture）/ 蛋白质（Protein）参考值，以及真实品种、国家与收获年份标签：
  · Barley（大麦）：178 样本（1998 年）
  · Maize（玉米）：141 样本（1998 年，与 Eigenvector Cargill Corn 完全同波段轴，可独立跨库验证）
  · Wheat（小麦）：149 样本（含 1998 年 100 条 + 1999 年 49 条跨年份自然漂移 OOD 测试集）
"""
from __future__ import annotations
from pathlib import Path

import csv
import os

import numpy as np

# 相对仓库根目录解析数据路径，而不是依赖当前工作目录（CWD）。
# 这样在 notebooks/ 或 tests/ 下调用 load_*() 也不会 FileNotFoundError。
_ROOT = Path(__file__).resolve().parents[1]

def _p(rel: str) -> str:
    """把仓库相对路径转成绝对路径；允许 CHEMDL_DATA_DIR 覆盖数据根。"""
    import os
    base = os.environ.get("CHEMDL_DATA_DIR")
    return str(Path(base) / rel if base else _ROOT / rel)


DIR = _p("data_ext/cordoba")
FILES = {
    "barley": "Barely_sensAIfood_UnivCordoba_v2.csv",
    "maize": "Maize_sensAIfood_UnivCordoba_v2.csv",
    "wheat": "Wheat_sensAIfood_UnivCordoba_v2.csv",
}


def load_cordoba_crop(crop: str = "wheat", root: str = DIR) -> dict:
    key = crop.lower()
    if key not in FILES:
        raise ValueError(f"未知谷物 {crop!r}，可选 {list(FILES)}")
    path = os.path.join(root, FILES[key])
    with open(path, encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd)
        spec_start = header.index("1100")
        axis = np.asarray([float(h) for h in header[spec_start:]], dtype=np.float64)
        idx_m = header.index("Moisture")
        idx_p = header.index("Protein")
        idx_yr = header.index("Year")
        idx_sp = header.index("Spectrometer")
        X, y, years, spectrometers = [], [], [], []
        for row in rd:
            # 行尾多一个逗号会让 len(row) 多出一列空字符串；这里统一裁掉尾部空列
            while row and row[-1].strip() == "":
                row.pop()
            if not row or len(row) < spec_start + 10:
                continue
            # CSV 中常见 ""/"NA"/"NaN" 缺失值：原实现直接 float(v) 会 ValueError 崩掉整个文件。
            def _f(v, default=np.nan):
                s = str(v).strip()
                if s == "" or s.upper() in ("NA", "NAN", "NULL", "-"):
                    return default
                try:
                    return float(s)
                except ValueError:
                    return default

            X.append([_f(v) for v in row[spec_start:]])
            y.append([_f(row[idx_m]), _f(row[idx_p])])
            years.append(int(_f(row[idx_yr], 0.0)))
            spectrometers.append(row[idx_sp])
    return {
        "crop": key,
        "X": np.asarray(X, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "cols": ["Moisture", "Protein"],
        "axis": axis,
        "years": np.asarray(years, dtype=int),
        "spectrometers": np.asarray(spectrometers),
    }


def load_cordoba_all(root: str = DIR) -> dict[str, dict]:
    return {c: load_cordoba_crop(c, root=root) for c in FILES}


if __name__ == "__main__":
    for c, d in load_cordoba_all().items():
        yrs = {int(y): int((d["years"] == y).sum()) for y in np.unique(d["years"])}
        print(f"Cordoba {c:<7}: X={d['X'].shape}, years={yrs}, "
              f"Moisture=[{d['y'][:,0].min():.1f},{d['y'][:,0].max():.1f}], "
              f"Protein=[{d['y'][:,1].min():.1f},{d['y'][:,1].max():.1f}]")
