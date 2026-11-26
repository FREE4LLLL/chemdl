"""真实 NIR 数据：Eigenvector 的 Corn 数据集。

80 个玉米样本，1100–2498 nm（2 nm 间隔，700 通道），测水分/油/蛋白/淀粉。
同一批样本在**三台仪器**上各测一遍：

  m5   仪器 1
  mp5  仪器 2（与 m5 同型号，不同台）
  mp6  仪器 3（不同型号）

这组数据的价值在于它天然提供**真实的跨仪器迁移**测试：
用 m5 训练、测 mp5 / mp6，正是化学计量学最关心的场景，而且没有合成成分。

来源：https://www.eigenvector.com/data/Corn/index.html（Cargill 提供）
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


NPZ = _p("data_raw/corn.npz")


def load_corn(cache=NPZ):
    """返回 dict：spec[m]= (80,700)，y=(80,4)，cols=[Moisture,Oil,Protein,Starch]。"""
    if os.path.exists(cache):
        with np.load(cache) as _f:            # with 保证 NpzFile 句柄关闭
            d = {k: _f[k] for k in _f.files}
        return {"m5": d["m5"], "mp5": d["mp5"], "mp6": d["mp6"],
                "y": d["y"], "cols": np.asarray([str(c) for c in d["cols"]]),
                "axis": 1100.0 + 2.0 * np.arange(d["m5"].shape[1], dtype=float)}

    import scipy.io as sio

    m = sio.loadmat(_p("data_raw/corn.mat"))
    out = {k: np.asarray(m[k][0, 0]["data"], dtype=np.float64)
           for k in ("m5spec", "mp5spec", "mp6spec")}
    y = np.asarray(m["propvals"][0, 0]["data"], dtype=np.float64)
    cols = None
    for item in np.asarray(m["propvals"][0, 0]["label"]).ravel():
        a = np.asarray(item).ravel()
        if a.size >= 4 and all(isinstance(x, str) for x in a):
            cols = [str(x).strip() for x in a]
    cols = cols or ["Moisture", "Oil", "Protein", "Starch"]

    d = {"m5": out["m5spec"], "mp5": out["mp5spec"], "mp6": out["mp6spec"],
         "y": y, "cols": np.array(cols)}
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez_compressed(cache, **d)
    return {"m5": d["m5"], "mp5": d["mp5"], "mp6": d["mp6"], "y": y,
            "cols": np.asarray([str(c) for c in d["cols"]]),
            # 与其余 4 个加载器统一返回波长轴（1100-2498 nm、2 nm 步长、700 通道）
            "axis": 1100.0 + 2.0 * np.arange(d["m5"].shape[1], dtype=float)}


def summarize(d):
    """打印各仪器的谱统计，确认数据合理（吸光度应在 0–1.5 量级）。"""
    for k in ("m5", "mp5", "mp6"):
        x = d[k]
        print(f"{k:<5} shape={x.shape}  min={x.min():.3f} max={x.max():.3f} "
              f"mean={x.mean():.3f}  nan={int(np.isnan(x).sum())}")
    print(f"\n参考值（{len(d['y'])} 样本 × {d['y'].shape[1]} 组分）：")
    for i, c in enumerate(d["cols"]):
        v = d["y"][:, i]
        print(f"  {c:<10} {v.min():6.2f} – {v.max():6.2f}  mean={v.mean():6.2f} std={v.std():5.2f}")


if __name__ == "__main__":
    summarize(load_corn())
