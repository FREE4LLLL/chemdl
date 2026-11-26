"""Eigenvector 药片射击赛 2002（IDRC shootout 2002）：655 片药片 x 2 台光谱仪。

数据来自 https://eigenvector.com/resources/data-sets/ 的 nir_shootout_2002.mat，
原始页 http://www.idrc-chambersburg.org/shootout_2002.htm。
同一批药片在两台仪器上各测一遍（配对），参考值 3 个：weight / hardness / assay。
官方划分 calibrate 155 / test 460 / validate 40。

用途：作为 Corn（80 样本 x 3 仪器）之外的**独立复现集**——样本量 8 倍，
且配对结构干净，适合检验「PDS 应在 raw 空间」「k=1 就够」「10 条配对谱够用」
这些只在 Corn 上得到的结论。
"""
from __future__ import annotations
from pathlib import Path

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


NPZ = _p("data_raw/tablets.npz")
SRC = _p("data_ext/tab/nir_shootout_2002.mat")


def load_tablets(cache=NPZ, trim_saturated: bool = True):
    """加载 IDRC 2002 Tablets 数据集。

    trim_saturated=True（默认，审阅 P3-03）裁掉首端低信噪比边带与尾端 540:650 探测器饱和盲区
    （吸光度高达 5.1~6.59 AU），保留线性吸光度区间 [35:540]（< 4.0 AU）。
    """
    if os.path.exists(cache):
        # cols 是字符串数组，必须 allow_pickle；用 with 保证句柄关闭
        with np.load(cache, allow_pickle=True) as _f:
            d = {k: _f[k] for k in _f.files}
        out = dict(d)
    else:
        import scipy.io as sio

        m = sio.loadmat(SRC, squeeze_me=True, struct_as_record=False)
        X1, X2, Y, part = [], [], [], []
        for name, tag in (("calibrate", "calibrate"), ("test", "test"), ("validate", "validate")):
            X1.append(np.asarray(m[f"{tag}_1"].data, dtype=np.float64))
            X2.append(np.asarray(m[f"{tag}_2"].data, dtype=np.float64))
            Y.append(np.asarray(m[f"{tag}_Y"].data, dtype=np.float64))
            part.append(np.full(len(X1[-1]), len(part)))
        # 列名优先从 .mat 的 Y.label 读（不同镜像来源的子集顺序可能不同），读不到才回退
        try:
            _lab = getattr(m["calibrate_Y"], "label", None)
            _cols = np.asarray(_lab, dtype=str).ravel() if _lab is not None else None
        except Exception:
            _cols = None
        if _cols is None or _cols.size != np.vstack(Y).shape[1]:
            _cols = np.array(["weight", "hardness", "assay"])
        out = {"X1": np.vstack(X1), "X2": np.vstack(X2), "y": np.vstack(Y),
               "part": np.concatenate(part),          # 0=calibrate 1=test 2=validate
               "cols": _cols,
               # IDRC 2002 Tablets 真实波长轴：600-1898 nm、2 nm 步长（650 通道）。
               # 原实现直接用通道整数索引 0..649 冒充波长，导致下游 match_nir_bands 按
               # 890-2380 nm 匹配时全部落入 Unassigned，跨库 resample_spectrum 也会彻底错配。
               "axis": 600.0 + 2.0 * np.arange(np.vstack(X1).shape[1], dtype=float)}
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        out["cols"] = np.asarray(out["cols"], dtype=str)
        np.savez_compressed(cache, **out)

    if trim_saturated:
        sl = slice(35, 540)
        out["X1"] = out["X1"][:, sl]
        out["X2"] = out["X2"][:, sl]
        out["axis"] = out["axis"][sl]
    return out


def summarize(d):
    print("Tablets:", d["X1"].shape, "x 2 instruments;", d["y"].shape[1], "refs")
    for i, c in enumerate(d["cols"]):
        v = d["y"][:, i]
        print(f"  {c:<10} {v.min():8.2f} – {v.max():8.2f}  mean={v.mean():7.2f} std={v.std():6.2f}")
    for p, nm in enumerate(("calibrate", "test", "validate")):
        print(f"  {nm:<10} n={int((d['part']==p).sum())}")
    print("  axis:", d["axis"].min(), "-", d["axis"].max(), "|", len(d["axis"]), "channels")


if __name__ == "__main__":
    summarize(load_tablets())
