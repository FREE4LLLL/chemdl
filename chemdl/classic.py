"""五步预处理的可解释基线实现（numpy / scipy / sklearn）。

严格按课件流程：01 噪声去除 → 02 基线校正 → 03 标准化 → 04 峰校准对齐 → 05 异常剔除。
每一步都保留中间结果，便于和可微版本逐步对齐比较。

用法::

    cp = ClassicPipeline().fit(X_train)          # 只在训练集上学参数
    Z_train = cp.transform(X_train)
    Z_test  = cp.transform(X_test)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.linalg import solveh_banded

LN = lambda xs: np.log(np.clip(xs, 1e-12, None))


# --------------------------------------------------------------------------- #
# 01 噪声去除：Savitzky-Golay 平滑（局域多项式最小二乘）
# --------------------------------------------------------------------------- #
def denoise_sg(x, win=11, order=3):
    if win % 2 == 0:
        win += 1
    win = min(win, x.shape[-1] - (1 - x.shape[-1] % 2))
    return savgol_filter(x, win, order, axis=-1)


# --------------------------------------------------------------------------- #
# 02 基线校正：非对称最小二乘（Eilers & Boelens）
# --------------------------------------------------------------------------- #
_DTD_CACHE: dict[tuple[int, int], sparse.csc_matrix] = {}


def _diff_matrix(n, order=2):
    D = sparse.eye(n, format="csc")
    for _ in range(order):
        D = D[1:] - D[:-1]
    return D


def _get_dtd(n, order=2):
    key = (int(n), int(order))
    dtd = _DTD_CACHE.get(key)
    if dtd is None:
        D = _diff_matrix(n, order)
        dtd = (D.T @ D).tocsc()
        _DTD_CACHE[key] = dtd
    return dtd


def als_baseline(y, lam=1e5, p=0.01, n_iter=10, _lam_dtd=None):
    """单条谱的 ALS 基线。返回 (baseline, corrected)。"""
    L = y.shape[-1]
    w = np.ones(L)
    z = y.copy()
    # W + lam*D'D 是带宽仅为 2 的对称正定五对角矩阵：用带状 Cholesky（solveh_banded）
    # 比每次迭代构造 csc 并做通用稀疏 LU 快 10-30 倍，且避免重复建 sparse.diags/.tocsc。
    # 注意必须把调用方的 lam 传进去：_dtd_banded 的默认值 1e5 会在省略时静默覆盖 lam。
    _band = (_dtd_banded(L, lam=lam) if _lam_dtd is None
             else _lam_dtd_banded(_lam_dtd, L))
    for _ in range(n_iter):
        ab = _band.copy()
        ab[2] += w                              # 主对角在最后一行（upper 约定）
        z = solveh_banded(ab, w * y, lower=False)
        w = np.where(y > z, p, 1.0 - p)
    return z, y - z


_DTD_BAND_CACHE: dict = {}


def _dtd_banded(L: int, lam: float = 1e5) -> np.ndarray:
    """lam * D'D 的带状表示（upper 形式，形如 (3, L)，主对角在第 1 行）。

    D 为二阶差分算子，D'D 是对称五对角、半带宽 2。这里按 solveh_banded 的
    upper 约定返回上三角带。
    """
    # 直接由差分系数构造 D'D 的对角带（避免建稀疏矩阵）
    # D 为二阶差分（内部行 [-1,2,-1]），D'D 的对角：
    #   主对角 [1, 5, 6, 6, ..., 6, 5, 1]
    #   偏移1  [-2, -4, -4, ..., -4, -2]
    #   偏移2  [1, 1, ..., 1]
    # 直接从 _get_dtd 提取三条对角带，保证与稀疏求解数值完全一致
    # （_diff_matrix 的边界处理不是朴素的内部差分行，手推系数会差 2）。
    # D'D 半带宽为 2，toarray() 只在首次调用时发生且结果被 _DTD_BAND_CACHE 缓存。
    key = (int(L), float(lam))
    cached = _DTD_BAND_CACHE.get(key)
    if cached is not None:
        return cached
    M = (lam * _get_dtd(L)).toarray()
    ab = np.zeros((3, L), dtype=np.float64)
    ab[2, :] = np.diag(M)                 # solveh_banded upper: 主对角在最后一行
    if L > 1:
        ab[1, 1:] = np.diag(M, 1)         # 偏移 1
    if L > 2:
        ab[0, 2:] = np.diag(M, 2)         # 偏移 2
    _DTD_BAND_CACHE[key] = ab
    return ab


def _lam_dtd_banded(lam_dtd, L: int) -> np.ndarray:
    """把 lam*D'D 的系数（标量或带）转成 solveh_banded 的 upper 带。"""
    if isinstance(lam_dtd, np.ndarray) and lam_dtd.shape == (3, L):
        return lam_dtd
    # 也接受 scipy 稀疏矩阵 / 稠密 ndarray（调用方常直接传 lam*_get_dtd(L)）
    if hasattr(lam_dtd, "toarray") or (isinstance(lam_dtd, np.ndarray) and lam_dtd.shape == (L, L)):
        M = lam_dtd.toarray() if hasattr(lam_dtd, "toarray") else lam_dtd
        ab = np.zeros((3, L), dtype=np.float64)
        ab[2, :] = np.diag(M)
        if L > 1:
            ab[1, 1:] = np.diag(M, 1)
        if L > 2:
            ab[0, 2:] = np.diag(M, 2)
        return ab
    raise TypeError(
        f"_lam_dtd 需为 (3,{L}) 带状矩阵、({L},{L}) 稠密矩阵或 scipy 稀疏矩阵，"
        f"收到 {type(lam_dtd).__name__} shape={getattr(lam_dtd, 'shape', None)}")


def baseline_batch(X, lam=1e5):
    L = X.shape[-1]
    lam_dtd = lam * _get_dtd(L)
    # 必须显式 float64：empty_like 在 float32 输入下会把 spsolve 的 float64 基线静默截断
    B = np.empty(X.shape, dtype=np.float64)
    for i in range(X.shape[0]):
        B[i], _ = als_baseline(X[i], lam=lam, _lam_dtd=lam_dtd)
    return B


# --------------------------------------------------------------------------- #
# 03 标准化：SNV（逐样本）→ 逐通道 Z-score（跨样本）
# --------------------------------------------------------------------------- #
def snv(X):
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-9
    return (X - mu) / sd


def channel_zscore(X, mu=None, sd=None):
    if mu is None:
        mu = X.mean(0, keepdims=True)
        _raw_sd = X.std(0, keepdims=True)
        # 死通道（训练方差为 0）不能给 1e-9 作分母：测试期 1e-5 的读数噪声会被放大到 1e4，
        # 直接击穿后续 PCAGate 的 Q 残差与 T^2。零方差通道置 1.0（即原样保留，不做缩放）。
        _floor = max(float(np.median(_raw_sd)) * 1e-3, 1e-8)
        sd = np.where(_raw_sd < _floor, 1.0, _raw_sd)
    return (X - mu) / sd, mu, sd


# --------------------------------------------------------------------------- #
# 04 峰校准对齐：分段相关对齐（icoshift 简化版）
# --------------------------------------------------------------------------- #
def _shift_segment(y, ref, lo, hi, smax):
    """在**固定**参考窗 [lo,hi) 上，找样本 y 的整点平移 s 使与 ref 相关最大。

    参考窗必须固定：早先版本让样本窗与参考窗同步平移（比较 y[lo+s:hi+s] 与
    ref[lo+s:hi+s]），测到的是同一窗口的自相似性而不是"样本相对参考的错位"，
    选出的平移量由窗口边缘截断的浮点噪声决定（恒等输入都会被挪 1–2 点）。
    这里只移样本窗，参考侧取与之重叠的固定区间，按实际重叠长度归一化。
    """
    L = y.shape[-1]
    best, best_s = -np.inf, 0
    for s in range(-smax, smax + 1):
        ylo, yhi = max(0, lo + s), min(L, hi + s)   # 平移后的样本窗
        rlo, rhi = ylo - s, yhi - s                 # 与之重叠的固定参考区间
        if yhi - ylo < (hi - lo) * 0.5:
            continue
        seg, r = y[ylo:yhi], ref[rlo:rhi]
        a, b = seg - seg.mean(), r - r.mean()
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        # 平坦/零方差段（基线区、饱和平顶）相关系数无定义：必须跳过，
        # 否则所有 s 都得 c=0.0，而 0.0 > -inf 会让首轮 s=-smax 被选中锁死。
        if na < 1e-9 or nb < 1e-9:
            continue
        c = float((a @ b) / (na * nb))
        # 平局时优先取 |s| 更小者（无偏移优先），避免边缘截断的浮点噪声决定平移量
        if c > best + 1e-12 or (abs(c - best) <= 1e-12 and abs(s) < abs(best_s)):
            best, best_s = c, s
    return best_s


def align_segments(X, ref, n_seg=8, smax=6):
    """对每条谱估计分段平移量，插值成整条 warp 曲线后重采样。"""
    L = X.shape[-1]
    edges = np.linspace(0, L, n_seg + 1).astype(int)
    centers = (edges[:-1] + edges[1:]) // 2
    out = np.empty(X.shape, dtype=np.float64)
    deltas = np.empty((X.shape[0], L), dtype=np.float32)
    grid = np.arange(L)
    for i, y in enumerate(X):
        sh = np.array([_shift_segment(y, ref, edges[j], edges[j + 1], smax)
                       for j in range(n_seg)], dtype=np.float32)
        d = np.interp(grid, centers, sh).astype(np.float32)
        d -= d.mean()
        deltas[i] = d
        out[i] = np.interp(grid + d, grid, y, left=y[0], right=y[-1])
    return out, deltas


# --------------------------------------------------------------------------- #
# 05 异常剔除：PCA 得分图 + Q 残差 / T² 双指标
# --------------------------------------------------------------------------- #
class PCAGate:
    def __init__(self, n_comp=5, q=0.97):
        self.n_comp, self.q = n_comp, q

    def fit(self, Z):
        from sklearn.decomposition import PCA
        # n_comp 必须夹到 min(n-1, L)：小样本/短谱时 PCA 会直接报错
        n_ok = int(min(self.n_comp, max(1, len(Z) - 1), Z.shape[1]))
        self.n_comp = n_ok
        self.pca = PCA(n_components=n_ok, random_state=0).fit(Z)
        stat = self.score(Z)
        self.thr_q = np.quantile(stat["q"], self.q)
        self.thr_t = np.quantile(stat["t2"], self.q)
        return self

    def score(self, Z):
        rec = self.pca.inverse_transform(self.pca.transform(Z))
        q = ((Z - rec) ** 2).sum(1)
        t = self.pca.transform(Z)
        t2 = (t ** 2 / (self.pca.explained_variance_ + 1e-12)).sum(1)
        return {"q": q, "t2": t2}

    def flag(self, Z):
        s = self.score(Z)
        return (s["q"] > self.thr_q) | (s["t2"] > self.thr_t), s


# --------------------------------------------------------------------------- #
# 管线
# --------------------------------------------------------------------------- #
class ClassicPipeline:
    """五步顺序流水线。fit 只在训练集上估计参数，transform 可复用到测试集。"""

    def __init__(self, sg=(11, 3), als_lam=1e5, n_seg=8, smax=6, pca_comp=5, pca_q=0.97):
        self.sg, self.als_lam = sg, als_lam
        self.n_seg, self.smax = n_seg, smax
        self.pca_comp, self.pca_q = pca_comp, pca_q

    def fit(self, X):
        """与 transform 走同一条路径：01 → 02 → 03 → 04 → 03' → 05。

        审计（2026-11-16）修正：早先 fit 在对齐**前**不做 SNV，参考谱也存成
        未 SNV 的均值；transform 却在对齐前 SNV，并把参考谱再 SNV 一次。
        于是 PCA 门控阈值与 mu/sd 是在一种表示上标定、在另一种表示上应用。
        """
        Z1 = denoise_sg(np.asarray(X, np.float64), *self.sg)          # 01
        Z2 = Z1 - baseline_batch(Z1, self.als_lam)                    # 02
        Z3 = snv(Z2)                                                  # 03
        self.ref = Z3.mean(0)                                         # 参考谱落在 SNV 空间
        Z4, self.deltas = align_segments(Z3, self.ref, self.n_seg, self.smax)   # 04
        Z5, self.mu, self.sd = channel_zscore(snv(Z4))                # 03'
        self.gate = PCAGate(self.pca_comp, self.pca_q).fit(Z5)        # 05
        self._fitted = Z5
        return self

    # ---- 状态序列化（修复 codex 审计 01：classic.pt 推理缺预处理）------- #
    def state_dict(self):
        """导出可 pickle 的训练态，供 train.py 存进 checkpoint。

        只含 transform 需要的量；``_fitted`` 是缓存，不入盘。
        """
        if not hasattr(self, "gate"):
            raise RuntimeError("ClassicPipeline 未 fit，不能导出状态")
        g = self.gate
        # 全部用 torch.Tensor + Python 基础类型，保证 weights_only=True 可直接加载
        # （codex full-audit E2：numpy 数组会让受限加载抛 UnpicklingError）。
        import torch as _t
        def _tt(a):
            return _t.as_tensor(np.asarray(a, np.float64))
        return {
            "sg": tuple(int(v) for v in self.sg), "als_lam": float(self.als_lam),
            "n_seg": int(self.n_seg), "smax": int(self.smax),
            "pca_comp": int(self.pca_comp), "pca_q": float(self.pca_q),
            "ref": _tt(self.ref),
            "deltas": _tt(self.deltas),
            "mu": _tt(self.mu),
            "sd": _tt(self.sd),
            "gate_n_comp": int(g.n_comp), "gate_q": float(g.q),
            "gate_thr_q": float(g.thr_q), "gate_thr_t": float(g.thr_t),
            "gate_components": _tt(g.pca.components_),
            "gate_mean": _tt(g.pca.mean_),
            "gate_explained_variance": _tt(g.pca.explained_variance_),
        }

    @classmethod
    def from_state_dict(cls, st):
        """从 checkpoint 里的状态重建一个可用于 transform 的 pipeline。"""
        obj = cls(sg=tuple(st["sg"]), als_lam=st["als_lam"],
                  n_seg=st["n_seg"], smax=st["smax"],
                  pca_comp=st["pca_comp"], pca_q=st["pca_q"])
        obj.ref = np.asarray(np.asarray(st["ref"]), np.float64)
        obj.deltas = np.asarray(np.asarray(st["deltas"]), np.float64)
        obj.mu = np.asarray(np.asarray(st["mu"]), np.float64)
        obj.sd = np.asarray(np.asarray(st["sd"]), np.float64)
        from sklearn.decomposition import PCA
        g = PCAGate(int(st["gate_n_comp"]), float(st["gate_q"]))
        pca = PCA(n_components=int(st["gate_n_comp"]), random_state=0)
        pca.components_ = np.asarray(np.asarray(st["gate_components"]), np.float64)
        pca.mean_ = np.asarray(np.asarray(st["gate_mean"]), np.float64)
        pca.explained_variance_ = np.asarray(np.asarray(st["gate_explained_variance"]), np.float64)
        pca.n_components_ = int(st["gate_n_comp"])
        pca.n_features_in_ = pca.mean_.shape[0]
        g.pca = pca
        g.thr_q = float(st["gate_thr_q"])
        g.thr_t = float(st["gate_thr_t"])
        obj.gate = g
        return obj

    def transform(self, X, return_stages=False):
        # 01
        Z1 = denoise_sg(np.asarray(X, np.float64), *self.sg)
        # 02
        B = baseline_batch(Z1, self.als_lam)
        Z2 = Z1 - B
        # 03（先用 SNV 消除散射，对齐更稳）
        Z3 = snv(Z2)
        # 04
        Z4, deltas = align_segments(Z3, self.ref, self.n_seg, self.smax)   # ref 已在 SNV 空间
        # 03'（对齐后再做一次 SNV + 跨样本标准化）
        # !! 口径提示 !! 这一步使 classic 的实际链路为
        #   01 SG -> 02 ALS -> 03 SNV -> 04 对齐 -> 03' SNV -> zscore -> 05 门控，
        # 而可微版 DiffPreprocess 是 x3(SNV) -> x_out(对齐) 后直接输出，没有这第二次 SNV。
        # 两者步骤数不同，论文中凡是「classic vs diff 逐步对照」的表述必须注明这一点，
        # 否则读者会误以为两条链完全同构。
        Z5, _, _ = channel_zscore(snv(Z4), self.mu, self.sd)
        # 05
        bad, stat = self.gate.flag(Z5)
        if return_stages:
            return Z5, dict(denoised=Z1, baseline=B, corrected=Z2, snv=Z3,
                            aligned=Z4, standardized=Z5, flagged=bad, score=stat, deltas=deltas)
        return Z5
