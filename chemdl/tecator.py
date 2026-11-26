"""Tecator 肉类 NIR 数据集（240 样本 × 100 通道 × 3 组分，官方 StatLib 5 子集划分）。

来源：http://lib.stat.cmu.edu/datasets/tecator（公有领域）
文件格式：前段为说明文本，末尾每样本 25 行×5 列 = 125 个浮点数：
  100 个吸光度通道（850–1050 nm） + 22 个主成分得分 + 3 个化学参考值（moisture, fat, protein）。
官方划分（按顺序）：C(tr, 0:129)、M(va, 129:172)、T(te_id, 172:215)、E1+E2(te_ood, 215:240)。
"""
from __future__ import annotations
from pathlib import Path

import os
import re

import numpy as np

# 相对仓库根目录解析数据路径，而不是依赖当前工作目录（CWD）。
# 这样在 notebooks/ 或 tests/ 下调用 load_*() 也不会 FileNotFoundError。
_ROOT = Path(__file__).resolve().parents[1]

def _p(rel: str) -> str:
    """把仓库相对路径转成绝对路径；允许 CHEMDL_DATA_DIR 覆盖数据根。"""
    import os
    base = os.environ.get("CHEMDL_DATA_DIR")
    return str(Path(base) / rel if base else _ROOT / rel)


NPZ = _p("data_raw/tecator.npz")
SRC = _p("data_ext/tecator.txt")
N_CH, N_PC, N_REF, N_SAMPLE = 100, 22, 3, 240


def _parse(path: str = SRC) -> tuple[np.ndarray, np.ndarray]:
    """从文件末尾解析完整 240×125 数值矩阵，拆分 100 通道光谱与 3 组分真值。"""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    block = []
    for line in reversed(lines):
        s = line.strip()
        if not s:
            continue
        toks = re.split(r"\s+", s)
        try:
            vals = [float(t) for t in toks]
        except ValueError:
            break
        if len(vals) != 5:
            break
        block.append(vals)
    block.reverse()
    arr = np.asarray([v for row in block for v in row], dtype=np.float64)
    per = N_CH + N_PC + N_REF  # 125
    n = min(N_SAMPLE, len(arr) // per)
    # 必须断言：解析器从末尾反向读，遇异常行直接 break，原实现会静默返回残缺样本数，
    # 之后 tr=0:129 / te_ood=215:240 的官方索引切片会全部越界而不报错。
    if n != N_SAMPLE:
        raise ValueError(
            f"tecator 解析失败：期望 {N_SAMPLE} 条样本，实际解析到 {n} 条"
            f"（文件可能被截断、末尾有额外注释行，或格式不符）。")
    mat = arr[len(arr) - n * per :].reshape(n, per)
    return mat[:, :N_CH], mat[:, N_CH + N_PC : N_CH + N_PC + N_REF]


def load_tecator(cache: str = NPZ, force_rebuild: bool = False) -> dict:
    if (not force_rebuild) and os.path.exists(cache):
        with np.load(cache) as _f:            # 不再需要 allow_pickle（纯数值数组）
            d = {k: _f[k] for k in _f.files}
        out = dict(d)
        # 防护旧缓存缺失官方划分键或错按 103 列解析导致真值错位
        if "tr" in out and "te_id" in out and "te_ood" in out and float(out["y"][:, 0].mean()) > 20.0:
            return out

    X, y = _parse()
    d = {
        "X": X,
        "y": y,
        "cols": np.array(["moisture", "fat", "protein"]),
        # 官方 850-1050 nm、2 nm 步长：linspace(850,1050,100) 的步长是 2.0202，
        # 末端会漂到 1050 而不是 1048，跨库按步长换算时会累积误差。
        "axis": 850.0 + 2.0 * np.arange(N_CH, dtype=float),
        "tr": np.arange(0, 129, dtype=np.int64),
        "va": np.arange(129, 172, dtype=np.int64),
        "te_id": np.arange(172, 215, dtype=np.int64),
        "te_ood": np.arange(215, 240, dtype=np.int64),
    }
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez_compressed(cache, **d)
    return d


if __name__ == "__main__":
    d = load_tecator(force_rebuild=True)
    print("X", d["X"].shape, "y", d["y"].shape, "tr/va/id/ood:",
          len(d["tr"]), len(d["va"]), len(d["te_id"]), len(d["te_ood"]))
    for i, c in enumerate(d["cols"]):
        v = d["y"][:, i]
        print(f"  {c:<10} {v.min():7.2f} – {v.max():7.2f}  mean={v.mean():6.2f} std={v.std():5.2f}")
