"""开源石化 NIR 数据集：SWRI Diesel Fuels（Eigenvector 公开基准）。

包含 784 条真实柴油近红外透射/吸收光谱 × 401 通道（750–1550 nm，步长 2 nm）与 7 项理化指标：
  BP50（50% 沸点）、CN（十六烷值）、D4052（密度）、FLASH（闪点）、
  FREEZE（凝点）、TOTAL（总芳烃含量）、VISC（粘度）。

数据集天然包含约 395 条有标签样品与 ~389 条无标签样品，非常适合同时测试：
  1) Masked Spectral Pretraining（在全量/无标签光谱上自监督预训练）
  2) `SpecHybridNet` vs `PLS` 在高频石油 C-H 倍频谱上的定量精度与非线性增益
  3) Kennard-Stone 边界样本（高杠杆 OOD 外推）与保形区间覆盖率。
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import scipy.io as sio

# 相对仓库根目录解析数据路径，而不是依赖当前工作目录（CWD）。
# 这样在 notebooks/ 或 tests/ 下调用 load_*() 也不会 FileNotFoundError。
_ROOT = Path(__file__).resolve().parents[1]

def _p(rel: str) -> str:
    """把仓库相对路径转成绝对路径；允许 CHEMDL_DATA_DIR 覆盖数据根。"""
    import os
    base = os.environ.get("CHEMDL_DATA_DIR")
    return str(Path(base) / rel if base else _ROOT / rel)


SRC = _p("data_ext/diesel/SWRI_Diesel_NIR.mat")


def load_diesel(path: str = SRC) -> dict:
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    ds = m["diesel_spec"]
    dp = m["diesel_prop"]
    X = np.asarray(ds.data, dtype=np.float64)
    y = np.asarray(dp.data, dtype=np.float64)
    cols = [str(c).strip() for c in dp.label[1][0]]
    axis = np.asarray(ds.axisscale[1][0], dtype=np.float64)
    # 附上有效标签掩码：7 项理化指标里约一半样本的标签是 NaN，
    # 下游（fit_pls / fit_bilevel_diffprep / rmsep）都不过滤 NaN，
    # 直接把整列送进去会让该列统计量全变 NaN。valid_mask[i, j] 表示第 i 个样本第 j 列有有效标签。
    # 注意：valid_mask 才是「该位置有有效标签」（True = 非 NaN）。
    # 历史上叫 nan_mask 但语义是反的，极易被调用方误当缺失掩码取反，故两者都提供。
    valid_mask = ~np.isnan(y)          # (n_samples, n_targets)
    nan_mask = valid_mask              # 向后兼容别名，新代码请用 valid_mask
    return {"X": X, "y": y, "cols": cols, "axis": axis,
            "valid_mask": valid_mask, "nan_mask": nan_mask}


def summarize(d: dict | None = None):
    if d is None:
        d = load_diesel()
    print(f"SWRI Diesel: X={d['X'].shape}, axis=[{d['axis'][0]:.0f}, {d['axis'][-1]:.0f}] nm")
    for j, c in enumerate(d["cols"]):
        ok = np.isfinite(d["y"][:, j])
        v = d["y"][ok, j]
        print(f"  {c:<8} valid={int(ok.sum()):<4} range=[{v.min():7.3f}, {v.max():7.3f}] std={v.std():6.3f}")


if __name__ == "__main__":
    summarize()
