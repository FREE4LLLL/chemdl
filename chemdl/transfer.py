"""跨仪器迁移校正的单一实现：DS / PDS，以及共用的 PLS 基线。

项目里原来散着三份 DS、两份 PDS、三份 SNV、六份 rmsep，口径各不相同。
其中 train_pds.py 的 PDS 只取了 PLSRegression.coef_，丢掉了中心化项：
sklearn 的预测是 (X - x_mean) @ coef_.T + y_mean，只乘 coef_ 等于给
每一列加了一个不同的常数偏移。实测（snv / mp6 / 5 折留出）：

    只取 coef_（原实现）   w=5,k=3 → 8.97   w=11,k=3 → 8.93
    m.predict(...)         w=5,k=3 → 0.27   w=11,k=3 → 0.28
    DS（全局 ridge）                       → 0.25

8.97 被当成「PDS 本身不行」写进了 HANDOFF 的已知局限。它其实是实现错误。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def rmsep(pred, true) -> float:
    """pooled RMSEP：(样本 × 组分) 全部平方误差的均方根，原单位。

    !! 必须做形状校验 !!
    历史代码里 pred/true 混用了 (n,) 与 (n,1) 两种约定；若不校验，numpy 会把
    (n,1) - (n,) 广播成 (n,n)，静默算出完全错误的数字（实测同一组数据
    1D 版 0.10、误广播版 1.10，差 11 倍）。
    本函数把单列 2D 自动 squeeze 到 1D（兼容两种约定），其余形状不一致直接报错。
    """
    pred = np.asarray(pred, np.float64)
    true = np.asarray(true, np.float64)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if true.ndim == 2 and true.shape[1] == 1:
        true = true[:, 0]
    if pred.shape != true.shape:
        raise ValueError(
            "rmsep: pred shape %s != true shape %s（拒绝静默广播；"
            "两参数需同为 (n,) 或同为 (n,m)）" % (pred.shape, true.shape))
    return float(np.sqrt(((pred - true) ** 2).mean()))


# --------------------------------------------------------------------------- #
# 预处理（真实数据侧的最小集合）
# --------------------------------------------------------------------------- #
def prep_raw(X):
    return np.asarray(X, np.float64)


def prep_snv(X):
    """逐样本行标准化。真实 Corn 上唯一稳定有效的单步（REPORT §8）。"""
    X = np.asarray(X, np.float64)
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)


def prep_sg1(X):
    return savgol_filter(np.asarray(X, np.float64), 11, 3, deriv=1, axis=1)


def prep_sg2(X):
    return savgol_filter(np.asarray(X, np.float64), 15, 3, deriv=2, axis=1)


def prep_classic(X):
    """01 SG 平滑 → 02 ALS 去基线（03 SNV 不在这里，留给下游按需叠加）。"""
    from .classic import als_baseline, denoise_sg

    X = denoise_sg(np.asarray(X, np.float64), 11, 3)
    return np.stack([als_baseline(x, 1e5)[1] for x in X])


_DCT_ALS_CACHE: dict[int, object] = {}


def prep_dct_als_snv(X):
    """02 UnrolledALS（48 正交 DCT 基）→ 03 逐样本 SNV（原始谱输入）。

    审阅 P3-01 实测：在 Corn 5×5 CV 上单视图+DS 达 0.2026±0.0277，
    显著优于 SNV+DS 的 0.2298（配对 t=-5.07），接入 DVTF-B 达 0.2001。
    """
    import torch
    from .diffprep import LearnableSNV, UnrolledALS

    X_arr = np.asarray(X, np.float32)
    L = X_arr.shape[1]
    mods = _DCT_ALS_CACHE.get(L)
    if mods is None:
        als = UnrolledALS(L, n_basis=48, iters=8, identity=False).eval()
        snv = LearnableSNV().eval()
        mods = (als, snv)
        _DCT_ALS_CACHE[L] = mods
    als, snv = mods
    with torch.no_grad():
        x = torch.from_numpy(X_arr).unsqueeze(1)
        _, corr = als(x)
        return snv(corr).squeeze(1).numpy().astype(np.float64)


def emsc_apply(X, ref=None, degree: int = 2, ridge: float = 1e-4, poly_ridge: float = 0.0):
    """扩展乘性信号校正（EMSC）：逐样本闭式消除乘性缩放与多阶波长漂移基线。

    当 ``poly_ridge > 0`` 时启用 Afseth & Kohler (2012, Chemom. Intell. Lab. Syst.
    117:92-99, DOI: 10.1016/j.chemolab.2012.03.004) 正则化 EMSC：仅对高阶多项式项
    （d >= 1）施加 Tikhonov 惩罚 ``poly_ridge * L``，消除平滑宽带谱（如 Tecator）上
    参考谱与多项式基共线性导致的化学方差抹除爆炸（CLAIMS C-080/C-083）。
    """
    X = np.asarray(X, np.float64)
    if ref is None:
        if len(X) < 2:
            raise ValueError("单样本调用 emsc_apply 时必须显式传入训练集参考谱 ref，否则会退化为恒等映射")
        ref = X.mean(0)
    else:
        ref = np.asarray(ref, np.float64).ravel()
    L = X.shape[1]
    lam = np.linspace(-1.0, 1.0, L)
    polys = [np.ones_like(lam)]
    for d in range(1, degree + 1):
        if d == 1:
            polys.append(lam)
        elif d == 2:
            polys.append(0.5 * (3.0 * lam ** 2 - 1.0))
        else:
            polys.append(lam ** d)
    P = np.stack(polys, axis=1)
    M = np.concatenate([ref[:, None], P], axis=1)                  # (L, D)
    pen = np.full(M.shape[1], float(ridge), dtype=np.float64)
    if poly_ridge > 0.0 and M.shape[1] > 2:
        pen[2:] = float(poly_ridge) * float(L)
    gram = M.T @ M + np.diag(pen)
    proj = np.linalg.solve(gram, M.T)                             # (D, L)
    coef = X @ proj.T                                             # (N, D)
    b = coef[:, :1]
    b_safe = np.where(np.abs(b) < 0.2, np.where(b >= 0, 0.2, -0.2), b)
    return (X - coef[:, 1:] @ P.T) / b_safe


def emsc_reg_apply(X, ref=None, degree: int = 2, ridge: float = 1e-4, poly_ridge: float = 1.0):
    """正则化多项式 EMSC（Afseth & Kohler 2012），默认 poly_ridge=1.0 避免宽带共线性爆炸。"""
    return emsc_apply(X, ref=ref, degree=degree, ridge=ridge, poly_ridge=poly_ridge)


def emsc_iter_apply(X, iters: int = 4, degree: int = 2, ridge: float = 1e-4, ref=None):
    """迭代版 EMSC：每次用校正后的谱更新参考谱。

    动因：既有 emsc_apply 用数据集均值谱当参考，而均值谱含化学变异，
    校正后所有样本被拉向它 -> 化学信息被当散射抹掉（CLAIMS C-080，
    Tecator 上 RMSEP 从 0.244 涨到 6.64）。

    本函数按 Helland 1995 的 ITMSC 思路（Sec.V）迭代更新参考：
    参考谱收敛到「共同形状」后，样本间的差异才主要是化学差异。
    与 emsc_apply 并存，不替换既有行为。
    """
    X = np.asarray(X, np.float64)
    L = X.shape[1]
    lam = np.linspace(-1.0, 1.0, L)
    # 与 emsc_apply 一致的任意阶 Legendre 多项式（原实现硬编码 3 项，
    # degree>=3 时会静默退化为 2 阶而不报错）
    polys = [np.polynomial.legendre.Legendre.basis(d)(lam) for d in range(int(degree) + 1)]
    P = np.stack(polys, axis=1)
    ref = X.mean(0) if ref is None else np.asarray(ref, np.float64).ravel()
    cur = X.copy()
    s0 = float(np.linalg.norm(ref))
    for _ in range(max(1, int(iters))):
        M = np.concatenate([ref[:, None], P], axis=1)
        gram = M.T @ M + ridge * np.eye(M.shape[1])
        proj = np.linalg.solve(gram, M.T)
        coef = X @ proj.T
        b = coef[:, 0]
        # 只在 b 确实靠近 0 时才截断，且保留符号（原实现用 +-0.2 硬夹，会翻号）
        b_safe = np.where(np.abs(b) < 1e-3, 1.0, b)
        cur = (X - coef[:, 1:] @ P.T) / b_safe[:, None]
        ref_new = cur.mean(0)
        s_now = float(np.linalg.norm(ref_new))
        if s_now > 1e-12 and s0 > 1e-12:
            cur = cur * (s0 / s_now)
            ref_new = cur.mean(0)
        ref = ref_new
    return cur

def prep_emsc(X):
    return emsc_apply(X, ref=None, degree=2)


def _msc_pair(xi, xr, ridge: float = 1e-8):
    """单个样本对参考谱的 MSC 回归 x_i = c + b*x_ref + e，返回 (c, b)。"""
    xr_c = xr - xr.mean()
    xi_c = xi - xi.mean()
    denom = float(xr_c @ xr_c) + ridge
    b = float(xi_c @ xr_c) / denom
    if abs(b) < 1e-12:
        b = 1.0
    c = float(xi.mean() - b * xr.mean())
    return c, b


def rnv_apply(X, lo: float = 5.0, hi: float = 95.0):
    """Robust Normal Variate (RNV) — 稳健标准正态变量变换。

    文献出处：Guo, Q., Wu, W., & Massart, D. L. (1999). *The robust normal variate
    transform for pattern recognition with near-infrared data*. Analytica Chimica
    Acta, 382(1–2), 87–103（作为 Barnes, Dhanoa & Lister 1989 SNV 的鲁棒化改进）。

    关键实现要点：用 ``[q_lo, q_hi]`` 内层分位区间估计稳健中心 ``mu_rob`` 与尺度
    ``sd_rob``，但对**未截断的原始光谱 ``x``** 做标准化 ``(x - mu_rob) / sd_rob``，
    避免把最高 5% 的真实分析吸收峰平头削掉（绝不对输出光谱本身做 clip）。
    """
    X = np.asarray(X, dtype=np.float64)
    was_1d = X.ndim == 1
    X2 = X[None, :] if was_1d else X
    out = np.empty_like(X2)
    for i, x in enumerate(X2):
        a, b = np.percentile(x, lo), np.percentile(x, hi)
        inner = x[(x >= a) & (x <= b)]
        if inner.size < 2:
            inner = x
        mu_r = float(inner.mean())
        sd_r = float(inner.std()) + 1e-9
        out[i] = (x - mu_r) / sd_r
    return out[0] if was_1d else out


def msc_apply(X, ref=None):
    """乘法散射校正（MSC）。

    出自 Martens et al. (1983)，本文实现参照 Helland, Naes & Isaksson (1995)
    Chemom. Intell. Lab. Syst. 29:233-241 的 Eq.(4)：
        x_ik = c_i + b_i * x_ref,k + e_ik
        x*_ik = (x_ik - c_i) / b_i
    ref 缺省时用 X 的均值谱（与 ISC/ITMSC 共用同一参考，便于对照）。
    """
    X = np.asarray(X, np.float64)
    if ref is None:
        if len(X) < 2:
            raise ValueError("单样本调用 msc_apply 时必须显式传入训练集参考谱 ref，否则会退化为恒等映射")
        xr = X.mean(0)
    else:
        xr = np.asarray(ref, np.float64).reshape(-1)
    out = np.empty_like(X)
    for i in range(len(X)):
        c, b = _msc_pair(X[i], xr)
        out[i] = (X[i] - c) / b
    return out


def rsc_apply(X, ref=None):
    """倒数散射校正（RSC，reciprocal scatter correction）。

    命名说明：本算子原名 isc_apply，但与 Helland et al. (1995) Eq.(10) 方向相反
    （文献是乘 r^2，此处是除 r^2），继续叫 ISC 会误导复现者。
    已改名为独立的 RSC，并在模块末尾保留 isc_apply = rsc_apply 作为兼容别名。


    文献出处：Helland, Naes & Isaksson (1995) Chemom. Intell. Lab. Syst. 29:233-241,
    Section IV 判据 (d) 与 Eq.(10)，原文给出 (m_i)_ISC = r_i^2 * (m_i)_MSC。

    !! 本实现与上述公式的方向相反，这是刻意保留的 !!
    把 MSC 的回归方向反过来（参考谱当因变量）得到反向斜率
        b_inv = cov(x, ref) / var(x)
    由于 b * b_inv = r^2（已数值验证逐位吻合），等价除数为
        m_impl = 1 / b_inv = b / r^2 = m_MSC / r^2
    即『除以 r^2』，而非文献的『乘以 r^2』。

    实测（_audit/verify_isc_fix.py，5x5 严格口径）三个库上反文献方向都更好：
      Tecator       isc_now 0.1754 (-27.2%, 全场最好) vs isc_lit 0.2632 (+9.3%)
      Corn m5       isc_now 1.3783 (+433.7%)           vs isc_lit 1.7200 (+566.0%)
      Cordoba maize isc_now 1.4248 (+75.7%)            vs isc_lit 1.7222 (+112.4%)
    物理上『除 r^2』放大低相关样本的除数、从而抑制其偏离参考谱，符合散射校正的目标；
    文献的『乘 r^2』方向相反。

    处置：保留本实现作为独立算子，文档如实描述其实际行为，不再声称等同 Eq.(10)。
    台账：CLAIMS.md C-103。
    """
    X = np.asarray(X, np.float64)
    if ref is None:
        if len(X) < 2:
            raise ValueError("单样本调用 rsc_apply 时必须显式传入训练集参考谱 ref，否则会退化为恒等映射")
        xr = X.mean(0)
    else:
        xr = np.asarray(ref, np.float64).reshape(-1)
    out = np.empty_like(X)
    for i in range(len(X)):
        # ISC 把回归方向反过来：x_ref = c' + b'*x_i（参考谱当因变量）
        xi_c = X[i] - X[i].mean()
        xr_c = xr - xr.mean()
        den = float(xi_c @ xi_c)
        b_inv = float(xr_c @ xi_c) / den if den > 1e-12 else 1.0
        if abs(b_inv) < 1e-12:
            b_inv = 1.0
        c_inv = float(xr.mean() - b_inv * X[i].mean())
        # 反解 x_i = (x_ref - c')/b'，要让校正谱贴近 x_ref 即 x* = b'*x_i + c'
        out[i] = b_inv * X[i] + c_inv
    return out


def itmsc_apply(X, iters: int = 3):
    """迭代 MSC（ITMSC）。

    出自 Helland, Naes & Isaksson (1995) Chemom. Intell. Lab. Syst. 29:233-241，
    Section V：判据的"final version leads to an iterated MSC"。
    做法：用当前参考谱做 MSC，再用校正后谱的均值更新参考谱，重复 iters 次。
    """
    X = np.asarray(X, np.float64)
    cur = X.copy()
    ref = X.mean(0)
    s_ref = float(np.linalg.norm(ref))
    for _ in range(max(1, int(iters))):
        cur = msc_apply(X, ref=ref)
        ref = cur.mean(0)
        # 防止正反馈发散：否则 (x-c)/b 把谱放大 -> 新参考更大 -> b 更小 -> 再放大
        s_now = float(np.linalg.norm(ref))
        if s_now > 1e-12:
            cur = cur * (s_ref / s_now)
            ref = cur.mean(0)
    return cur


def dssnv_apply(X, wc: float = 1.0, ws: float = 0.75, s_bar: float | None = None):
    """解耦收缩标准正态变量变换（Decoupled Shrinkage-SNV, DSSNV）。

    将经典 Barnes (1989) SNV 解耦为两个独立物理自由度：
        x_out = (x - wc * row_mean(x)) / ((1 - ws) * s_bar + ws * row_std(x))
      - (wc=0, ws=0): 恒等缩放原谱（保留全部散射代理信号）
      - (wc=1, ws=0): 仅逐样本去加性基线（如 Corn：消除行均值漂移同时避免乘性除法放大噪声）
      - (wc=0, ws=1): 仅逐样本乘性校正（保留加性吸收基线）
      - (wc=1, ws∈(0,1)): 部分收缩 SNV（如 Tecator：压缩大部分乘性散射畸变，同时保留与浓度耦合的单调散射代理）
      - (wc=1, ws=1): 退化为经典 SNV

    归纳式推理时 ``s_bar`` 必须由训练折 ``X_tr.std(1).mean()`` 传入，测试折零泄漏。
    """
    X = np.asarray(X, np.float64)
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    sb = float(sd.mean()) if s_bar is None else float(s_bar)
    denom = np.clip((1.0 - float(ws)) * sb + float(ws) * sd, 1e-9, None)
    return (X - float(wc) * mu) / denom


def fit_dssnv_pls(Xtr, ytr, kmax: int = 12, inner: int = 4, seed: int = 0,
                  grid=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.5), (1.0, 0.75), (1.0, 1.0))):
    """纯归纳式嵌套 CV 联合选择单组分最优 ``(wc, ws, n_LV)`` 的 DSSNV-PLS 回归器。

    所有统计量（``s_bar``、``(wc, ws)``、``n_LV``、``ymu/ysd``）严格仅在训练折内确定。
    返回 ``(predict_fn, chosen_cfgs)``，其中 ``chosen_cfgs[j] = (wc, ws, n_LV)``。

    内层交叉验证的 ``s_bar_f`` 严格仅由各内折训练子集 ``Xtr[itr]`` 独立估计，
    全折零直推泄露（E41）；最终模型在完整训练折上重算 ``s_bar`` 与 ``ymu/ysd`` 拟合。
    返回 ``(predict_fn, chosen_cfgs)``，其中 ``chosen_cfgs[j] = (wc, ws, n_LV)``。
    """
    Xtr = np.asarray(Xtr, np.float64)
    ytr = np.asarray(ytr, np.float64)
    if ytr.ndim == 1:
        ytr = ytr[:, None]
    if len(Xtr) < 2:
        raise ValueError(f"fit_dssnv_pls 至少需要 2 条训练样本，实际收到 {len(Xtr)} 条")
    s_bar = float(Xtr.std(1).mean())
    ymu, ysd = ytr.mean(0), np.clip(ytr.std(0), 1e-9, None)
    yn = (ytr - ymu) / ysd

    eff_inner = min(int(inner), len(Xtr))
    splits = list(KFold(eff_inner, shuffle=True, random_state=seed).split(Xtr))
    max_k = min(int(kmax), max(1, len(Xtr) - max(len(va) for _, va in splits) - 1), Xtr.shape[1])
    s_bar_folds = [float(Xtr[itr].std(1).mean()) for itr, _ in splits]
    fold_grids = [
        [
            (dssnv_apply(Xtr[itr], wc, ws, s_f), dssnv_apply(Xtr[iva], wc, ws, s_f))
            for (itr, iva), s_f in zip(splits, s_bar_folds)
        ]
        for wc, ws in grid
    ]

    models, chosen = [], []
    for j in range(ytr.shape[1]):
        best_e, best_cfg = np.inf, (0, 1)
        for gi in range(len(grid)):
            for k in range(1, max_k + 1):
                errs = []
                for (itr, iva), (Xitr, Xiva) in zip(splits, fold_grids[gi]):
                    m = PLSRegression(n_components=min(k, max(1, len(itr) - 1)), scale=False).fit(Xitr, yn[itr, [j]])
                    errs.append(rmsep(m.predict(Xiva), yn[iva, [j]]))
                e = float(np.mean(errs))
                if e < best_e:
                    best_e, best_cfg = e, (gi, k)
        gi, k = best_cfg
        wc, ws = grid[gi]
        X_full = dssnv_apply(Xtr, wc, ws, s_bar)
        m = PLSRegression(n_components=min(k, max(1, len(Xtr) - 1)), scale=False).fit(X_full, yn[:, [j]])
        models.append((wc, ws, m))
        chosen.append((float(wc), float(ws), int(k)))

    def pred(Xte):
        preds = [m.predict(dssnv_apply(Xte, wc, ws, s_bar)) for wc, ws, m in models]
        return np.concatenate(preds, axis=1) * ysd + ymu

    return pred, chosen


def fit_sport_pls(
    Xtr,
    ytr,
    blocks: tuple[str, ...] = ("dssnv", "sg1", "sg2", "raw"),
    kmax: int = 10,
    inner: int = 4,
    tol: float = 0.995,
    seed: int = 0,
):
    """SPORT（Sequential Preprocessing through Orthogonalization）多预处理正交融合 PLS。

    出处：
      - Roger, Biancolillo & Marini (2020), *Chemom. Intell. Lab. Syst.* 199:103975,
        DOI: 10.1016/j.chemolab.2020.103975 (SPORT / SO-PLS)
      - Mishra et al. (2020), *Trends Anal. Chem.* 132:116045,
        DOI: 10.1016/j.trac.2020.116045 (多预处理融合综述，替代朴素逆误差加权 DVTF)

    原理：按序对每个预处理块（如 ``dssnv/snv → sg1 → sg2 → raw``）提取与前序潜变量得分
    ``T_prev`` 严格正交的新信息 ``X_b^⊥ = X_b - T_prev P_{orth,b}``，并在内层零泄漏 CV 上
    选择 ``k_b ∈ {0, 1, ..., kmax}``（无净增益时取 ``k_b = 0`` 跳过该块，防止过拟合）。
    全部参考统计量（``s_bar``、``ref``、``mu_b``、``P_{orth,b}``、``R_b``、``Q_b``）严格仅在
    训练折内估计，测试折单样本归纳前传。
    """
    Xtr = np.asarray(Xtr, np.float64)
    ytr = np.asarray(ytr, np.float64)
    was_1d = ytr.ndim == 1
    if was_1d:
        ytr = ytr[:, None]
    n = len(Xtr)
    if n < 3:
        raise ValueError(f"fit_sport_pls 至少需要 3 条训练样本，实际收到 {n} 条")

    ymu, ysd = ytr.mean(0), np.clip(ytr.std(0), 1e-9, None)
    yn = (ytr - ymu) / ysd
    ref_tr = Xtr.mean(0)
    s_bar_tr = float(Xtr.std(1).mean())

    def _apply_block(name: str, X_in: np.ndarray, wc_ws=(1.0, 0.75)) -> np.ndarray:
        if name == "raw":
            return prep_raw(X_in)
        if name == "snv":
            return prep_snv(X_in)
        if name == "dssnv":
            return dssnv_apply(X_in, wc_ws[0], wc_ws[1], s_bar=s_bar_tr)
        if name == "sg1":
            return prep_sg1(X_in)
        if name == "sg2":
            return prep_sg2(X_in)
        if name == "msc":
            return msc_apply(X_in, ref=ref_tr)
        if name == "emsc_reg":
            return emsc_reg_apply(X_in, ref=ref_tr)
        if name in PREPS:
            return PREPS[name](X_in)
        raise ValueError(f"未知 SPORT 预处理块: {name}")

    eff_inner = min(int(inner), max(2, n // 2))
    splits = list(KFold(eff_inner, shuffle=True, random_state=seed).split(Xtr))
    max_k = min(int(kmax), max(1, n - max(len(va) for _, va in splits) - 2))

    # 预先缓存各块训练特征（含 dssnv 时用固定物理默认 (1.0, 0.75) 或每个目标快速选）
    block_tr = {b: _apply_block(b, Xtr) for b in blocks}

    col_models = []
    chosen_lvs = []
    for j in range(ytr.shape[1]):
        yj = yn[:, j]
        y_res = yj - float(yj.mean())
        T_prev = None
        stages = []
        lvs_j = {}

        # 为每个内层 CV 折维护严格独立的 (T_tr, T_va, y_res_tr, y_res_va) 状态，
        # 杜绝把全训练折拟合出的 T_prev / y_res 复用进内层验证折造成目标泄漏。
        fold_states = [
            {
                "T_tr": None,
                "T_va": None,
                "y_res_tr": yj[itr] - float(yj[itr].mean()),
                "y_res_va": yj[iva] - float(yj[itr].mean()),
            }
            for itr, iva in splits
        ]

        for b_idx, b_name in enumerate(blocks):
            Xb_raw = block_tr[b_name]
            mu_b = Xb_raw.mean(0)
            Xc = Xb_raw - mu_b

            # 预先计算当前块在每个内层折内与前序折内得分正交化后的 (X_itr, X_iva)
            fold_X = []
            for f_idx, (itr, iva) in enumerate(splits):
                fs = fold_states[f_idx]
                mu_f = Xb_raw[itr].mean(0)
                X_itr = Xb_raw[itr] - mu_f
                X_iva = Xb_raw[iva] - mu_f
                if fs["T_tr"] is not None:
                    Tp_tr = fs["T_tr"]
                    Tp_va = fs["T_va"]
                    gram = Tp_tr.T @ Tp_tr + 1e-10 * np.eye(Tp_tr.shape[1])
                    P_f = np.linalg.solve(gram, Tp_tr.T @ X_itr)
                    X_itr = X_itr - Tp_tr @ P_f
                    X_iva = X_iva - Tp_va @ P_f
                fold_X.append((X_itr, X_iva))

            # 内层 CV 选 k_b ∈ {0, 1, ..., max_k}（零验证折目标泄漏）
            null_errs = [float(np.sqrt(np.mean(fs["y_res_va"] ** 2))) for fs in fold_states]
            best_e = float(np.mean(null_errs))
            best_k = 0
            k_candidates = range(1, max_k + 1)
            thresh = float("inf") if (T_prev is None and b_idx == 0) else (best_e * float(tol))

            for k in k_candidates:
                fold_errs = []
                for f_idx, (itr, iva) in enumerate(splits):
                    X_itr, X_iva = fold_X[f_idx]
                    fs = fold_states[f_idx]
                    k_use = min(k, max(1, len(itr) - 2), X_itr.shape[1])
                    m = PLSRegression(n_components=k_use, scale=False).fit(X_itr, fs["y_res_tr"])
                    pred_va = m.predict(X_iva).ravel()
                    fold_errs.append(float(np.sqrt(np.mean((pred_va - fs["y_res_va"]) ** 2))))
                e_k = float(np.mean(fold_errs))
                if e_k < (best_e if best_k > 0 else thresh):
                    best_e, best_k = e_k, k

            lvs_j[b_name] = int(best_k)
            if best_k <= 0:
                continue

            # 用选出的 best_k 严格在各内层折内推进 fold_states（供后续块使用）
            for f_idx, (itr, iva) in enumerate(splits):
                X_itr, X_iva = fold_X[f_idx]
                fs = fold_states[f_idx]
                k_f = min(best_k, max(1, len(itr) - 2), X_itr.shape[1])
                pls_f = PLSRegression(n_components=k_f, scale=False).fit(X_itr, fs["y_res_tr"])
                R_f = np.asarray(pls_f.x_rotations_, dtype=np.float64)
                T_tr_b = X_itr @ R_f
                T_va_b = X_iva @ R_f
                Q_f = np.linalg.solve(T_tr_b.T @ T_tr_b + 1e-10 * np.eye(k_f), T_tr_b.T @ fs["y_res_tr"])
                fs["y_res_tr"] = fs["y_res_tr"] - T_tr_b @ Q_f
                fs["y_res_va"] = fs["y_res_va"] - T_va_b @ Q_f
                fs["T_tr"] = T_tr_b if fs["T_tr"] is None else np.hstack([fs["T_tr"], T_tr_b])
                fs["T_va"] = T_va_b if fs["T_va"] is None else np.hstack([fs["T_va"], T_va_b])

            if T_prev is not None:
                gram = T_prev.T @ T_prev + 1e-10 * np.eye(T_prev.shape[1])
                P_orth = np.linalg.solve(gram, T_prev.T @ Xc)
                X_orth = Xc - T_prev @ P_orth
            else:
                P_orth = None
                X_orth = Xc

            k_fit = min(best_k, max(1, n - 2), X_orth.shape[1])
            pls = PLSRegression(n_components=k_fit, scale=False).fit(X_orth, y_res)
            R_b = np.asarray(pls.x_rotations_, dtype=np.float64)        # (L, k_fit)
            T_b = X_orth @ R_b                                          # (n, k_fit)
            Q_b = np.linalg.solve(T_b.T @ T_b + 1e-10 * np.eye(k_fit), T_b.T @ y_res)
            y_res = y_res - T_b @ Q_b
            T_prev = T_b if T_prev is None else np.hstack([T_prev, T_b])
            stages.append((b_name, mu_b, P_orth, R_b, Q_b))

        col_models.append((float(yj.mean()), stages))
        chosen_lvs.append(lvs_j)

    def pred(Xte):
        Xa = np.asarray(Xte, np.float64)
        block_te = {b: _apply_block(b, Xa) for b in blocks}
        cols = []
        for y0, stages in col_models:
            yh = np.full(len(Xa), y0, dtype=np.float64)
            T_prev_te = None
            for b_name, mu_b, P_orth, R_b, Q_b in stages:
                Xc = block_te[b_name] - mu_b
                X_orth = Xc if P_orth is None else (Xc - T_prev_te @ P_orth)
                T_b = X_orth @ R_b
                yh = yh + T_b @ Q_b
                T_prev_te = T_b if T_prev_te is None else np.hstack([T_prev_te, T_b])
            cols.append(yh)
        out = np.stack(cols, axis=1) * ysd + ymu
        return out[:, 0] if was_1d else out

    return pred, chosen_lvs



# 向后兼容别名：旧名 isc_apply 保留，新代码请用 rsc_apply；
# 同步提供 SNV-opt（Helland 1995 m_i 族 SNR 最优收缩）别名，兼容原 dssnv_apply / fit_dssnv_pls
isc_apply = rsc_apply
snv_opt_apply = dssnv_apply
fit_snv_opt_pls = fit_dssnv_pls

PREPS = {"raw": prep_raw, "snv": prep_snv, "sg1": prep_sg1,
         "msc": msc_apply, "rnv": rnv_apply, "isc": isc_apply, "rsc": rsc_apply,
         "itmsc": itmsc_apply, "emsc_it": emsc_iter_apply,
         "emsc_reg": emsc_reg_apply,
         "sg2": prep_sg2, "classic": prep_classic, "dct_als_snv": prep_dct_als_snv,
         "emsc": prep_emsc}


def fit_prep_fn(name: str, X_tr, wc: float = 1.0, ws: float = 0.75):
    """严格归纳式预处理打包器：在训练折 ``X_tr`` 上冻结全部有状态参考统计量并返回单样本可调用闭包。

    解决直接对测试集调用 ``PREPS['msc' | 'isc' | 'emsc' | 'emsc_reg'](X_te)`` 时的测试集均值泄漏
    与单样本退化恒等映射风险。无状态算子（``raw / snv / sg1 / sg2 / classic / dct_als_snv``）直接返回对应函数。
    """
    X_arr = np.asarray(X_tr, dtype=np.float64)
    if X_arr.ndim != 2 or len(X_arr) < 1:
        raise ValueError(f"X_tr 须为非空二维数组 (N, L)，实际 shape={X_arr.shape}")
    ref_tr = X_arr.mean(0)
    s_bar_tr = float(X_arr.std(1).mean())
    if name == "msc":
        return lambda X: msc_apply(X, ref=ref_tr)
    if name in ("isc", "rsc"):
        return lambda X: isc_apply(X, ref=ref_tr)
    if name == "emsc":
        return lambda X: emsc_apply(X, ref=ref_tr, degree=2)
    if name == "emsc_reg":
        return lambda X: emsc_reg_apply(X, ref=ref_tr, degree=2)
    if name == "itmsc":
        ref_it = itmsc_apply(X_arr, iters=3).mean(0)
        return lambda X: msc_apply(X, ref=ref_it)
    if name == "emsc_it":
        ref_it = emsc_iter_apply(X_arr, iters=4, degree=2).mean(0)
        return lambda X: emsc_apply(X, ref=ref_it, degree=2)
    if name in ("dssnv", "snv_opt"):
        return lambda X: dssnv_apply(X, wc=wc, ws=ws, s_bar=s_bar_tr)
    if name in PREPS:
        return PREPS[name]
    raise ValueError(f"未知预处理算子: {name}")


# --------------------------------------------------------------------------- #
# 主动转移集选样（Kennard-Stone / PCA Leverage）
# --------------------------------------------------------------------------- #
def select_kennard_stone(X, k: int) -> np.ndarray:
    """Kennard-Stone 贪心极大极小距离选样，在源谱空间均匀覆盖样品分布与边界。"""
    X = np.asarray(X, np.float64)
    n = len(X)
    if k >= n:
        return np.arange(n, dtype=int)
    if k <= 0:
        return np.empty(0, dtype=int)
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
    d2 = np.maximum(sq_norms + sq_norms.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(d2, -1.0)
    i0, i1 = np.unravel_index(int(np.argmax(d2)), d2.shape)
    if k == 1:
        return np.array([int(i0)], dtype=int)
    selected = [int(i0), int(i1)]
    chosen = np.zeros(n, dtype=bool)
    chosen[selected] = True
    np.fill_diagonal(d2, 0.0)
    min_d2 = np.minimum(d2[i0], d2[i1])
    while len(selected) < k:
        cand = np.where(~chosen, min_d2, -1.0)
        nxt = int(np.argmax(cand))
        selected.append(nxt)
        chosen[nxt] = True
        min_d2 = np.minimum(min_d2, d2[nxt])
    return np.asarray(selected, dtype=int)


def select_leverage(X, k: int, ncomp: int = 6) -> np.ndarray:
    """基于主成分白化子空间杠杆值 + 距离多样的代表性样本选择。"""
    X = np.asarray(X, np.float64)
    n = len(X)
    if k >= n:
        return np.arange(n, dtype=int)
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    r = int(min(ncomp, max(1, (S > 1e-8).sum()), n - 1))
    scores = U[:, :r]
    return select_kennard_stone(scores, k)


# --------------------------------------------------------------------------- #
# PLS 基线：成分数在内层 CV 上选
# --------------------------------------------------------------------------- #
def _inner_k(X, yn, kmax, inner, seed):
    n = len(X)
    # codex full-audit E29: 早先只按样本量限 kmax，窄波段（如只有 2 列）时
    # PLSRegression 尝试第 3 个成分会抛 ValueError。有效维度也要参与限制。
    n_feat = int(X.shape[1]) if hasattr(X, "shape") and X.ndim == 2 else n
    eff_inner = min(int(inner), n)
    max_k = min(int(kmax), max(1, n - max(1, n // eff_inner) - 1), max(1, n_feat))
    if max_k <= 1 or eff_inner < 2:
        return 1
    best_k, best_e = 1, np.inf
    for k in range(1, max_k + 1):
        errs = []
        for itr, iva in KFold(eff_inner, shuffle=True, random_state=seed).split(X):
            m = PLSRegression(n_components=min(k, max(1, len(itr) - 1)), scale=False)
            m.fit(X[itr], yn[itr])
            errs.append(rmsep(m.predict(X[iva]), yn[iva]))
        e = float(np.mean(errs))
        if e < best_e:
            best_k, best_e = k, e
    return best_k


def fit_pls(Xtr, ytr, kmax=10, inner=4, per_col=False, seed=0):
    """返回 (predict_fn, ks)。predict_fn(X) -> (n, m) 原单位；y 内部标准化。

    per_col=True 时每个组分各选自己的成分数（化学计量学常规做法）。
    """
    Xtr = np.asarray(Xtr, np.float64)
    ytr = np.asarray(ytr, np.float64)
    if ytr.ndim == 1:
        ytr = ytr[:, None]
    if len(Xtr) < 2:
        raise ValueError(f"fit_pls 至少需要 2 条训练样本，实际收到 {len(Xtr)} 条")
    ymu, ysd = ytr.mean(0), np.clip(ytr.std(0), 1e-9, None)
    yn = (ytr - ymu) / ysd

    if not per_col:
        k = _inner_k(Xtr, yn, kmax, inner, seed)
        m = PLSRegression(n_components=min(k, len(Xtr) - 1), scale=False).fit(Xtr, yn)
        return (lambda X: m.predict(X) * ysd + ymu), [k] * ytr.shape[1]

    models, ks = [], []
    for j in range(ytr.shape[1]):
        k = _inner_k(Xtr, yn[:, [j]], kmax, inner, seed)
        models.append(PLSRegression(n_components=min(k, len(Xtr) - 1),
                                    scale=False).fit(Xtr, yn[:, [j]]))
        ks.append(k)

    def pred(X):
        return np.concatenate([m.predict(X) for m in models], 1) * ysd + ymu

    return pred, ks


# --------------------------------------------------------------------------- #
# DS：全局线性映射
# --------------------------------------------------------------------------- #
def ds_matrix(X_src, X_tgt, ridge=1e-3):
    """F = argmin ||X_tgt @ F - X_src||² + ridge·||F||²

    N ≪ L 时走 Woodbury 对偶：`(XtᵀXt + λI)⁻¹Xtᵀ = Xtᵀ(XtXtᵀ + λI)⁻¹`，
    把 700×700 的求解换成 N×N（Corn 上 N=64）。数值等价，实测提速约 150 倍。
    """
    Xs = np.asarray(X_src, np.float64)
    Xt = np.asarray(X_tgt, np.float64)
    if Xs.shape != Xt.shape:
        raise ValueError(f"ds_matrix 需要配对光谱 (Xs.shape == Xt.shape)，实际 {Xs.shape} vs {Xt.shape}")
    N, L = Xt.shape
    if N < L:
        K = Xt @ Xt.T + ridge * np.eye(N)
        return Xt.T @ np.linalg.solve(K, Xs)
    A = Xt.T @ Xt + ridge * np.eye(L)
    return np.linalg.solve(A, Xt.T @ Xs)


def ds_apply(X, F):
    return np.asarray(X, np.float64) @ F


def ds_affine(X_src, X_tgt, ridge=1e-3):
    """带中心化的仿射 DS：x~ = (x - mean_t) @ F + mean_s。

    无中心化版（ds_matrix）在 L=700 >> N=64 时，均值外积占据最大奇异方向，
    且 ridge 把均值映射压向 0，在 raw 空间引入系统偏移。化学成分学界标准
    Direct Standardization 就是仿射形式。返回 (F, mean_src, mean_tgt)。
    """
    Xs, Xt = np.asarray(X_src, np.float64), np.asarray(X_tgt, np.float64)
    if Xs.shape != Xt.shape:
        raise ValueError(f"ds_affine 需要配对光谱 (Xs.shape == Xt.shape)，实际 {Xs.shape} vs {Xt.shape}")
    ms, mt = Xs.mean(0), Xt.mean(0)
    Xc, Xtc = Xs - ms, Xt - mt
    N, L = Xtc.shape
    if N < L:                                    # Woodbury 对偶，同 ds_matrix
        K = Xtc @ Xtc.T + ridge * np.eye(N)
        F = Xtc.T @ np.linalg.solve(K, Xc)
    else:
        A = Xtc.T @ Xtc + ridge * np.eye(L)
        F = np.linalg.solve(A, Xtc.T @ Xc)
    return F, ms, mt


def ds_apply_affine(X, F, mean_src, mean_tgt):
    return (np.asarray(X, np.float64) - mean_tgt) @ F + mean_src


# --------------------------------------------------------------------------- #
# PDS：逐波长局部映射
# --------------------------------------------------------------------------- #
def _nipals1_path(Xc: np.ndarray, yc: np.ndarray, ncomp: int):
    """NIPALS 1D PLS1 载荷路径（唯一的实现，_pls1_fit 与 hybrid._pls1_cv_best 共用）。

    输入已是中心化数据。返回 (P, W, Q, used)：载荷矩阵、权重矩阵、y 载荷、实际成分数。
    """
    n_feat = Xc.shape[1]
    P = np.empty((n_feat, ncomp), dtype=np.float64)
    W = np.empty((n_feat, ncomp), dtype=np.float64)
    Q = np.empty(ncomp, dtype=np.float64)
    Xk = Xc.copy()
    yk = yc.copy()
    used = 0
    for a in range(ncomp):
        w = Xk.T @ yk
        wn = float(np.linalg.norm(w))
        if wn < 1e-12:
            break
        w /= wn
        t = Xk @ w
        tt = float(t @ t)
        if tt < 1e-12:
            break
        p = (Xk.T @ t) / tt
        q = float(yk @ t) / tt
        P[:, a], W[:, a], Q[a] = p, w, q
        Xk -= np.outer(t, p)
        yk -= t * q
        used += 1
    return P, W, Q, used


def _pls1_fit(X: np.ndarray, y: np.ndarray, ncomp: int):
    """纯 NumPy 中心化 1D PLS1（与 sklearn PLSRegression(scale=False) 数值等价，提速约 8.6 倍）。"""
    xm = X.mean(0)
    ym = float(y.mean())
    Xc = X - xm
    yc = y - ym
    n_feat = X.shape[1]
    P, W, Q, used = _nipals1_path(Xc, yc, ncomp)
    if used == 0:
        return np.zeros(n_feat, dtype=np.float64), ym
    R = W[:, :used] @ np.linalg.solve(P[:, :used].T @ W[:, :used], Q[:used])
    coef = R
    intercept = ym - float(xm @ coef)
    return coef, intercept


def pds_predict(X_src, X_tgt, X_new, window=11, ncomp=3):
    """Piecewise Direct Standardization：把目标仪器谱映回源仪器谱。

    对每个波长 j，只用目标谱 [j-window/2, j+window/2] 的通道回归源谱的第 j 列。
    window 是**直径**（通道数），局部回归用纯 NumPy 1D PLS(ncomp)。
    """
    X_src = np.asarray(X_src, np.float64)
    X_tgt = np.asarray(X_tgt, np.float64)
    if X_src.shape != X_tgt.shape:
        raise ValueError(f"pds_predict 需要配对光谱 (X_src.shape == X_tgt.shape)，实际 {X_src.shape} vs {X_tgt.shape}")
    X_new = np.asarray(X_new, np.float64)
    L = X_src.shape[1]
    out = np.empty((X_new.shape[0], L))
    for j in range(L):
        lo, hi = max(0, j - window // 2), min(L, j + window // 2 + 1)
        k = int(min(ncomp, hi - lo, len(X_src) - 1))
        coef, b = _pls1_fit(X_tgt[:, lo:hi], X_src[:, j], k)
        out[:, j] = X_new[:, lo:hi] @ coef + b
    return out


def pds_matrix(X_src, X_tgt, window=11, ncomp=3):
    """PDS 的矩阵形式 (F, b)，满足 X @ F + b == pds_predict(X)。"""
    X_src = np.asarray(X_src, np.float64)
    X_tgt = np.asarray(X_tgt, np.float64)
    if X_src.shape != X_tgt.shape:
        raise ValueError(f"pds_matrix 需要配对光谱 (X_src.shape == X_tgt.shape)，实际 {X_src.shape} vs {X_tgt.shape}")
    L = X_src.shape[1]
    F = np.zeros((L, L))
    b = np.zeros(L)
    for j in range(L):
        lo, hi = max(0, j - window // 2), min(L, j + window // 2 + 1)
        k = int(min(ncomp, hi - lo, len(X_src) - 1))
        coef, bj = _pls1_fit(X_tgt[:, lo:hi], X_src[:, j], k)
        F[lo:hi, j] = coef
        b[j] = bj
    return F, b


# --------------------------------------------------------------------------- #
# 公平评测：只在留出折上算
# --------------------------------------------------------------------------- #
def cv_transfer(prep, d, y, target, method, folds, window=11, ncomp=3,
                ridge=1e-3, per_col=False):
    """跨仪器评测，**只评估留出折**。

    历史坑：train_pds.py 曾在全部 80 条上算 RMSEP，而 PLS 与变换矩阵都只用
    其中 64 条拟合过；train_real.py 的 te / te_ood 也是全部 80 条。两个口径
    混在一起，跨表比较会把 24% 的样本级泄漏算成方法差异。
    per_col=False 与 train_ds.py / results_ds.json / OPTIMIZATION §4.1 的
    0.2298 基线同口径（四个组分共享一个成分数）；改成逐组分选参会让 DS 列
    变成 0.2423，凭空多出第四套数字。

    返回 (mean, per_fold)。
    """
    perm = np.concatenate([np.asarray(f, int) for f in folds])
    vals = []
    for va in folds:
        tr = np.setdiff1d(perm, va)
        # 有状态算子（msc/isc/emsc/itmsc/emsc_it/dssnv）必须在训练折上冻结参考谱，
        # 否则验证折会用自己的均值当参考——既泄漏又退化为恒等（见 fit_prep_fn 文档）。
        fn = (fit_prep_fn(prep, d["m5"][tr]) if isinstance(prep, str)
              else prep)
        fn_t = (fit_prep_fn(prep, d[target][tr]) if isinstance(prep, str)
                else prep)
        Xs_tr, Xt_tr = fn(d["m5"][tr]), fn_t(d[target][tr])
        Xv, yv = fn_t(d[target][va]), y[va]
        pred, _ = fit_pls(Xs_tr, y[tr], per_col=per_col)
        if method == "bare":
            corr = Xv
        elif method == "ds":
            corr = ds_apply(Xv, ds_matrix(Xs_tr, Xt_tr, ridge))
        elif method == "pds":
            corr = pds_predict(Xs_tr, Xt_tr, Xv, window, ncomp)
        else:
            raise ValueError("unknown method: " + str(method))
        vals.append(rmsep(pred(corr), yv))
    return float(np.mean(vals)), [float(v) for v in vals]


# --------------------------------------------------------------------------- #
# 无标准样品（非配对）迁移：判别子空间 Bures-Wasserstein 最优传输 & di-PLS
# --------------------------------------------------------------------------- #
def _spd_pow(M: np.ndarray, power: float, eps: float = 1e-8) -> np.ndarray:
    """对称正定矩阵幂运算 M^power（特征分解实现，数值稳定）。"""
    Ms = 0.5 * (M + M.T)
    evals, evecs = np.linalg.eigh(Ms)
    evals = np.maximum(evals, eps)
    return (evecs * (evals ** power)) @ evecs.T


def _bw_subspace_map(Zs: np.ndarray, Zt: np.ndarray, sh: float) -> np.ndarray:
    """在 r 维子空间内计算对角目标收缩正则化的 Bures-Wasserstein Monge 线性映射 A_r。"""
    r = Zs.shape[1]
    cs = (Zs.T @ Zs) / max(len(Zs) - 1, 1)
    ct = (Zt.T @ Zt) / max(len(Zt) - 1, 1)
    floor_s = 1e-5 * float(np.trace(cs)) / max(r, 1) + 1e-8
    floor_t = 1e-5 * float(np.trace(ct)) / max(r, 1) + 1e-8
    Ss = (1.0 - sh) * cs + sh * np.diag(np.diag(cs)) + floor_s * np.eye(r)
    St = (1.0 - sh) * ct + sh * np.diag(np.diag(ct)) + floor_t * np.eye(r)
    Sth = _spd_pow(St, 0.5)
    Stih = _spd_pow(St, -0.5)
    # 只算一次矩阵开方（原实现把同一项 eigh 分解写了两遍，白烧一半算力）
    M_raw = Stih @ _spd_pow(Sth @ Ss @ Sth, 0.5) @ Stih
    M = 0.5 * (M_raw + M_raw.T)
    # 信赖域谱截断：防止秩亏配方子空间在极弱特征方向发生百倍方差放大爆炸
    evals, evecs = np.linalg.eigh(M)
    evals = np.clip(evals, 0.5, 2.0)
    return (evecs * evals) @ evecs.T


def subspace_lot_fit(X_src, X_tgt, y_src=None, rank: int = 8,
                     basis: str = "auto", shrink: float | None = None,
                     mean_match: bool = True,
                     geodesic: str | float = "auto",
                     gate_label_shift: bool = True,
                     r_pca_gate: int | str = 8,
                     gate_low: float = 2.2,
                     gate_high: float = 5.0,
                     n_boot: int = 16, seed: int = 0) -> dict:
    """判别子空间 Bures-Wasserstein 线性最优传输（Subspace-LOT）。

    针对无配对标准样品的跨仪器/跨年份迁移：在 L 维全谱做 CORAL/OT 会因 N_t << L 奇异崩溃，
    因此把最优传输严格限制在源域前 r 维 PLS 判别子空间（或 PCA 主子空间）U_r ∈ R^{L×r} 内，
    先对子空间协方差做保留各主分量的对角目标收缩（Schäfer-Strimmer 风格，避免微量成分失真），
    再求解唯一对称正定 Monge 传输映射：
        A_r = Σ_t^{-1/2} (Σ_t^{1/2} Σ_s Σ_t^{1/2})^{1/2} Σ_t^{-1/2}
    正交补空间严格保留恒等映射（仅可选平移中心），参数自由度从 O(L²) 降至 O(r²)。
    当 ``geodesic="auto"`` 时，在 r 维极低开销下做 B 次非参数自助采样估计位移方差 σ_boot²，
    按 James-Stein / McCann 测地线比例 τ = max(0, 1 - σ_boot² / ||A_r - I||_F²) 压缩：
        A_r^geo = (1 - τ)·I + τ·A_r
    当 ``gate_label_shift=True`` 时，利用源域前 PCA 补空间上的未中心化/中心化 Q 残差比
    ``r_shift = Q_t(m_s) / Q_t(m_t)`` 辨识真实硬件光学偏移 vs 纯样本先验分布偏移（Label Shift）：
    同仪器跨年份/高低浓度域的纯先验偏移满足 ``r_shift ≈ 1–2``，自动门控退回源域恒等预测而不误伤真值。
    """
    Xs = np.asarray(X_src, dtype=np.float64)
    Xt = np.asarray(X_tgt, dtype=np.float64)
    ms = Xs.mean(0)
    mt = Xt.mean(0)
    Xsc = Xs - ms
    Xtc = Xt - mt
    L = Xs.shape[1]
    r = int(min(rank, L, max(1, len(Xs) - 2), max(1, len(Xt) - 1)))

    _, _, Vt = np.linalg.svd(Xsc, full_matrices=False)
    use_pls = (y_src is not None) and (basis in ("auto", "pls"))
    if use_pls:
        ya = np.asarray(y_src, dtype=np.float64)
        if ya.ndim == 1:
            ya = ya[:, None]
        yn = (ya - ya.mean(0)) / np.clip(ya.std(0), 1e-9, None)
        pls = PLSRegression(n_components=r, scale=False).fit(Xsc, yn)
        Q, _ = np.linalg.qr(pls.x_weights_)
        U_r = Q[:, :r]
    else:
        U_r = Vt[:r].T

    r_shift = np.inf
    gate = 1.0
    if gate_label_shift:
        # 自由度安全上限：防止小目标样本集（如 Cordoba n_t=25）因 k_g 逼近 n_t 导致正交残差自由度耗尽而虚高 r_shift
        k_cap = int(min(L, max(1, len(Xs) - 2), max(1, len(Xt) // 3 if len(Xt) >= 12 else len(Xt) - 1)))
        Xtc_s = Xt - ms
        if r_pca_gate == "auto":
            cand_k = sorted({int(min(k0, k_cap)) for k0 in (6, 8, 10)})
            rs = []
            for kg in cand_k:
                Up = Vt[:kg].T
                q_u = float(np.mean(np.sum((Xtc_s - (Xtc_s @ Up) @ Up.T) ** 2, axis=1)))
                q_c = float(np.mean(np.sum((Xtc - (Xtc @ Up) @ Up.T) ** 2, axis=1)))
                rs.append(q_u / max(q_c, 1e-12))
            r_shift = float(np.median(rs))
        else:
            k_g = int(min(int(r_pca_gate), k_cap))
            Up = Vt[:k_g].T
            q_uncent = float(np.mean(np.sum((Xtc_s - (Xtc_s @ Up) @ Up.T) ** 2, axis=1)))
            q_cent = float(np.mean(np.sum((Xtc - (Xtc @ Up) @ Up.T) ** 2, axis=1)))
            r_shift = q_uncent / max(q_cent, 1e-12)
        if float(np.linalg.norm(ms - mt)) > 1e-6:
            gate = float(np.clip((r_shift - gate_low) / max(gate_high - gate_low, 1e-6), 0.0, 1.0))

    Zs = Xsc @ U_r                                      # (N_s, r)
    Zt = Xtc @ U_r                                      # (N_t, r)
    sh = min(0.50, max(0.10, float(r) / max(min(len(Zs), len(Zt)), 2))) if shrink is None else float(shrink)
    A_raw = _bw_subspace_map(Zs, Zt, sh)

    if geodesic == "auto":
        rng = np.random.default_rng(seed)
        ns, nt = len(Zs), len(Zt)
        eye_r = np.eye(r)
        deltas = np.empty((n_boot, r, r), dtype=np.float64)
        for b in range(n_boot):
            is_ = rng.integers(0, ns, ns)
            it_ = rng.integers(0, nt, nt)
            Zsb = Zs[is_] - Zs[is_].mean(0)
            Ztb = Zt[it_] - Zt[it_].mean(0)
            deltas[b] = _bw_subspace_map(Zsb, Ztb, sh) - eye_r
        sig2 = float(np.sum((deltas - deltas.mean(0)) ** 2) / max(n_boot - 1, 1))
        d2 = float(np.sum((A_raw - eye_r) ** 2))
        tau = min(1.0, max(0.0, 1.0 - sig2 / max(d2, 1e-12)))
    elif geodesic in ("none", None):
        tau = 1.0
    else:
        tau = float(np.clip(geodesic, 0.0, 1.0))

    eff_tau = tau * gate
    A_r = (1.0 - eff_tau) * np.eye(r) + eff_tau * A_raw
    A_r = 0.5 * (A_r + A_r.T)

    return {
        "U_r": U_r,
        "A_r": A_r,
        "ms": ms,
        "mt": mt,
        "mean_match": bool(mean_match),
        "rank": r,
        "tau": float(eff_tau),
        "gate": float(gate),
        "r_shift": float(r_shift),
    }


def subspace_lot_apply(X, state: dict) -> np.ndarray:
    """应用 Subspace-LOT 映射：仅在 U_r 子空间做 Monge 传输，正交补空间严格恒等。"""
    X = np.asarray(X, dtype=np.float64)
    U_r, A_r, ms, mt = state["U_r"], state["A_r"], state["ms"], state["mt"]
    gate = float(state.get("gate", 1.0))
    Xc = X - mt
    delta_sub = (Xc @ U_r) @ (A_r - np.eye(A_r.shape[0])) @ U_r.T
    if state["mean_match"] and gate > 0.0:
        return X + gate * (ms - mt) + delta_sub
    return X + delta_sub


def select_active_transfer(X_src, X_tgt_pool, k: int = 5,
                           st_lot: dict | None = None, y_src=None,
                           rank: int = 8, basis: str = "auto",
                           unc_weight: float = 0.35) -> np.ndarray:
    """在 Subspace-LOT 对齐后的判别子空间内为主动少量标样迁移挑选目标域样本索引。

    当 ``k <= 4`` 时，采用中位点触发的归一化判别空间 Kennard-Stone 覆盖（规避极少标样时的边界杠杆方差）；
    当 ``k >= 5`` 时，结合子空间马氏杠杆不确定度 ``u`` 与多样性距离贪心挑选最互补的 ``k`` 个标样。
    """
    Xs = np.asarray(X_src, dtype=np.float64)
    Xp = np.asarray(X_tgt_pool, dtype=np.float64)
    if st_lot is None:
        st_lot = subspace_lot_fit(Xs, Xp, y_src=y_src, rank=rank, basis=basis)
    Xp_al = subspace_lot_apply(Xp, st_lot)
    U_r, ms = st_lot["U_r"], st_lot["ms"]
    Zs = (Xs - ms) @ U_r
    Zt = (Xp_al - ms) @ U_r
    Zw = Zt / np.clip(np.std(Zs, axis=0), 1e-6, None)
    n = len(Zw)
    if k >= n:
        return np.arange(n, dtype=int)
    if k <= 0:
        return np.empty(0, dtype=int)

    sel = [int(np.argmin(np.linalg.norm(Zw - np.median(Zw, axis=0), axis=1)))]
    min_d = np.linalg.norm(Zw - Zw[sel[0]], axis=1)
    if k <= 4 or unc_weight <= 0.0:
        for _ in range(1, k):
            min_d[sel] = -1.0
            nxt = int(np.argmax(min_d))
            sel.append(nxt)
            min_d = np.minimum(min_d, np.linalg.norm(Zw - Zw[nxt], axis=1))
        return np.asarray(sel, dtype=int)

    cov_s = (Zs.T @ Zs) / max(len(Zs) - 1, 1) + 1e-4 * np.eye(U_r.shape[1])
    h_t = np.sum(np.linalg.solve(cov_s, Zt.T).T * Zt, axis=1)
    q10, q90, q95 = (float(np.percentile(h_t, p)) for p in (10, 90, 95))
    # 对超出 95% 分位的极端重尾离群点施加杠杆软压尾，防止少量离群样劫持小预算，主分布内权重恒为 1.0
    inlier_w = np.where(h_t > q95, np.sqrt(q95 / np.maximum(h_t, 1e-12)), 1.0)
    u = np.clip((h_t - q10) / max(q90 - q10, 1e-9), 0.0, 1.0)
    for _ in range(1, k):
        crit = min_d * inlier_w * (1.0 + unc_weight * u)
        crit[sel] = -1.0
        nxt = int(np.argmax(crit))
        sel.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(Zw - Zw[nxt], axis=1))
    return np.asarray(sel, dtype=int)


def fit_active_residual_corrector(yhat_k, Z_k, y_k, robust_huber: bool = True,
                                  return_loo: bool = False):
    """基于精确含截距杠杆 LOO-CV 与单离群防火墙门控的极少样本残差精修器（Zero -> 经验贝叶斯压缩Bias -> 低秩 Ridge / Huber-IRLS Ridge）。

    当 ``robust_huber=True``（默认）时，在标准 OLS Ridge 之外并行评估 4 步 Huber IRLS M-估计
    岭回归（引用 Huber 1964, Ann. Math. Statist. 35:73-101；重尾光谱稳健校正背景参见
    RESPIRE, arXiv:2511.19810, 2025），以同一精确加权帽子矩阵 LOO 与去最差单点防火墙择优；
    常数偏置采用经验贝叶斯 James-Stein 最优收缩 ``w_EB = t² / (t² + 1)``，在纯噪声时强收缩、
    在真实大偏移时近无偏校正。
    """
    yh = np.asarray(yhat_k, dtype=np.float64)
    Zk = np.asarray(Z_k, dtype=np.float64)
    yk = np.asarray(y_k, dtype=np.float64)
    if yh.ndim == 1:
        yh = yh[:, None]
    if yk.ndim == 1:
        yk = yk[:, None]

    K = len(yk)
    d = min(3 if K >= 10 else (2 if K >= 8 else 1), Zk.shape[1])
    models = []
    col_loo_mses = []
    alphas = (2.0, 8.0, 32.0, 128.0) if K < 15 else (1.0, 4.0, 16.0, 64.0)

    for j in range(yk.shape[1]):
        res = yk[:, j] - yh[:, j]
        e0_sq = res ** 2
        loo_zero_mse = float(np.mean(e0_sq))
        loo_zero_mae = float(np.mean(np.abs(res)))
        if K <= 1:
            models.append(("bias", j, 0.5 * float(res[0]) if K == 1 else 0.0))
            col_loo_mses.append(loo_zero_mse)
            continue

        e_b = (res - res.mean()) * (K / (K - 1.0))
        loo_bias_mse = float(np.mean(e_b ** 2))
        loo_bias_mae = float(np.mean(np.abs(e_b)))
        # K>=5 时叠加去最差单点截尾 MSE 与 MAE 校验，阻止单个重尾离群残差虚造全局偏置
        if K >= 5:
            t0_mse = float(np.mean(np.sort(e0_sq)[:-1]))
            tb_mse = float(np.mean(np.sort(e_b ** 2)[:-1]))
            bias_ok = (loo_bias_mse < loo_zero_mse) and (tb_mse < t0_mse) and (loo_bias_mae < loo_zero_mae)
        else:
            t0_mse, tb_mse = loo_zero_mse, loo_bias_mse
            bias_ok = loo_bias_mse < loo_zero_mse

        # 经验贝叶斯 James-Stein 偏置收缩：噪声主导 (|t|小) 时强收缩，真实系统偏移 (|t|大) 时近无偏
        rm = float(res.mean())
        rv = float(np.var(res, ddof=1)) if K > 1 else 0.0
        t2 = (rm * rm) / (rv / float(K) + 1e-12)
        w_eb = min(float(K) / (float(K) + 0.25), t2 / (t2 + 1.0))
        bias_use = w_eb * rm if bias_ok else 0.0
        base_mse = loo_bias_mse if bias_ok else loo_zero_mse
        base_mae = loo_bias_mae if bias_ok else loo_zero_mae
        base_tmse = tb_mse if bias_ok else t0_mse
        if K <= 5:
            models.append(("bias", j, bias_use))
            col_loo_mses.append(base_mse)
            continue

        rmu = float(res.mean())
        rc = res - rmu
        F_k = np.column_stack([yh[:, j], Zk[:, :d]])
        fmu, fsd = F_k.mean(0), np.clip(F_k.std(0), 1e-8, None)
        Fn = (F_k - fmu) / fsd
        best_loo, best_rmu, best_w = 0.78 * base_mse, rmu, None
        U, S, _ = np.linalg.svd(Fn, full_matrices=False)
        for a in alphas:
            # 含无惩罚截距项的精确块对角帽子矩阵对角线 h_ii = 1/K + Σ U_ir^2 s_r^2/(s_r^2 + a)
            diag_h = (1.0 / K) + np.sum((U ** 2) * (S ** 2 / (S ** 2 + a)), axis=1)
            if float(np.max(diag_h)) >= 0.88:
                continue
            w_a = np.linalg.solve(Fn.T @ Fn + a * np.eye(Fn.shape[1]), Fn.T @ rc)
            e_loo = (rc - Fn @ w_a) / (1.0 - diag_h)
            loo_mse = float(np.mean(e_loo ** 2))
            loo_mae = float(np.mean(np.abs(e_loo)))
            loo_tmse = float(np.mean(np.sort(e_loo ** 2)[:-1]))
            if loo_mse < best_loo and loo_mae < 0.95 * base_mae and loo_tmse < 0.90 * base_tmse:
                best_loo, best_rmu, best_w = loo_mse, rmu, w_a

            # Huber IRLS 稳健候选（应对重尾分布如土壤 SOC）
            if robust_huber and K >= 8:
                X_aug = np.column_stack([np.ones(K, dtype=np.float64), Fn])
                pen = np.diag([0.0] + [float(a)] * Fn.shape[1])
                theta = np.concatenate([[rmu], w_a])
                mad0 = float(np.median(np.abs(res - np.median(res)))) * 1.4826 + 1e-9
                delta = 1.345 * mad0
                wt = np.ones(K, dtype=np.float64)
                for _ in range(4):
                    r_cur = res - X_aug @ theta
                    abs_r = np.abs(r_cur)
                    wt = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-12))
                    Xw = X_aug * wt[:, None]
                    gram_w = Xw.T @ X_aug + pen
                    theta = np.linalg.solve(gram_w, Xw.T @ res)
                Xw = X_aug * wt[:, None]
                # 用 solve 代替显式求逆：杠杆对角线只需要 X_aug @ A^{-1}，
                # 而 X @ A^{-1} = (A^{-T} @ X^T)^T，无需构造 p×p 的完整逆矩阵。
                A_w = Xw.T @ X_aug + pen
                lhs = np.linalg.solve(A_w.T, X_aug.T).T      # (n, p) = X_aug @ A^{-1}
                diag_hw = wt * np.sum(lhs * X_aug, axis=1)
                if float(np.max(diag_hw)) < 0.88:
                    e_hloo = (res - X_aug @ theta) / (1.0 - diag_hw)
                    h_mse = float(np.mean(e_hloo ** 2))
                    h_mae = float(np.mean(np.abs(e_hloo)))
                    h_tmse = float(np.mean(np.sort(e_hloo ** 2)[:-1]))
                    if h_mse < best_loo and h_mae < 0.95 * base_mae and h_tmse < 0.90 * base_tmse:
                        best_loo = h_mse
                        best_rmu = float(theta[0])
                        best_w = theta[1:].copy()

        if best_w is None:
            models.append(("bias", j, bias_use))
            col_loo_mses.append(base_mse)
        else:
            mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826 + 1e-9
            max_step = max(3.5 * float(np.std(res)), 3.5 * mad)
            models.append(("ridge", j, F_k.shape[1], fmu, fsd, best_rmu, best_w, max_step))
            col_loo_mses.append(best_loo)

    def predict(yhat_q, Z_q):
        ya = np.asarray(yhat_q, dtype=np.float64)
        Zq = np.asarray(Z_q, dtype=np.float64)
        was_1d = ya.ndim == 1
        out = ya[:, None].copy() if was_1d else ya.copy()
        for item in models:
            if item[0] == "bias":
                _, j, b = item
                out[:, j] += b
            else:
                _, j, ncol, fmu, fsd, rmu, w, max_step = item
                F = out[:, j : j + 1] if ncol == 1 else np.column_stack([out[:, j], Zq[:, : ncol - 1]])
                Fn_q = np.clip((F - fmu) / fsd, -4.0, 4.0)
                out[:, j] += np.clip(rmu + Fn_q @ w, -max_step, max_step)
        return out[:, 0] if was_1d else out

    if return_loo:
        # 返回逐列 LOO MSE 数组（不是均值）：多目标量纲差异极大（如 Tablets 的
        # weight ~1e4 vs hardness ~1e1），下游必须按各列自身的 y 尺度标准化后再比较，
        # 否则大数值量纲的目标会主导熔断判据。
        return predict, np.asarray(col_loo_mses, dtype=np.float64)
    return predict


def active_transfer_fit(X_src_k_pool, idx_sel, y_sel, pred_fn, st_lot: dict,
                        auto_gate: bool = True):
    """组合 ``Subspace-LOT`` 与主动挑出的 K 条目标标样，返回端到端精修预测器。

    当 ``auto_gate=True`` 且 ``st_lot`` 启用了均值/子空间对齐（``gate > 0.05``）时，
    利用 K 条目标标样的精确留一法（LOO）误差在「SLOT 对齐分支」与「零均值减法 Bare 分支」间
    自动择优——既在常规跨仪器偏移下保留 SLOT 全部收益，又在仪器偏移与强先验偏移
    （P_S(y) != P_T(y)）混叠时自动熔断错误均值减法、防止灾难性偏移。
    """
    X_sel = np.asarray(X_src_k_pool, dtype=np.float64)[idx_sel]
    X_sel_al = subspace_lot_apply(X_sel, st_lot)
    Z_k = (X_sel_al - st_lot["ms"]) @ st_lot["U_r"]
    yhat_k = pred_fn(X_sel_al)
    refiner, loo_lot = fit_active_residual_corrector(yhat_k, Z_k, y_sel, return_loo=True)
    chosen_st = st_lot

    if auto_gate and float(st_lot.get("gate", 0.0)) > 0.05 and len(idx_sel) >= 3:
        st_bare = dict(st_lot)
        st_bare["gate"] = 0.0
        st_bare["tau"] = 0.0
        st_bare["A_r"] = np.eye(st_lot["U_r"].shape[1])
        Z_k_b = (X_sel - st_lot["ms"]) @ st_lot["U_r"]
        yhat_k_b = pred_fn(X_sel)
        ya_s = np.asarray(y_sel, dtype=np.float64)
        yh_lot = np.asarray(yhat_k, dtype=np.float64)
        yh_bar = np.asarray(yhat_k_b, dtype=np.float64)
        if ya_s.ndim == 1:
            ya_s = ya_s[:, None]
        if yh_lot.ndim == 1:
            yh_lot = yh_lot[:, None]
        if yh_bar.ndim == 1:
            yh_bar = yh_bar[:, None]
        # 逐目标无量纲标准化，避免多目标物理量纲（如 weight vs hardness）互相淹没
        y_scale = np.clip(ya_s.std(0, keepdims=True), 1e-6, None)
        raw_mse_lot = float(np.mean(((ya_s - yh_lot) / y_scale) ** 2))
        raw_mse_bare = float(np.mean(((ya_s - yh_bar) / y_scale) ** 2))
        # loo_lot / loo_bare 是逐列原始量纲 MSE，必须按同样的 y 尺度标准化后再比较
        loo_lot_n = np.asarray(loo_lot, dtype=np.float64).ravel() / (y_scale.ravel() ** 2)
        loo_lot_s = float(np.mean(loo_lot_n))
        # 双重熔断判据：仅当无监督 SLOT 确实恶化了原始标样预测（raw_mse_bare < 0.70 * raw_mse_lot）
        # 且精修后 LOO 亦显著更优时，才熔断错误均值减法（确保常规跨仪器零误触发、先验偏移精准抢救）
        if raw_mse_bare < 0.70 * raw_mse_lot:
            ref_b, loo_bare = fit_active_residual_corrector(yhat_k_b, Z_k_b, y_sel, return_loo=True)
            loo_bare_n = np.asarray(loo_bare, dtype=np.float64).ravel() / (y_scale.ravel() ** 2)
            loo_bare_s = float(np.mean(loo_bare_n))
            if loo_bare_s < 0.90 * loo_lot_s:
                refiner = ref_b
                chosen_st = st_bare

    def predict(X_tgt):
        X_al = subspace_lot_apply(X_tgt, chosen_st)
        Z_t = (X_al - chosen_st["ms"]) @ chosen_st["U_r"]
        return refiner(pred_fn(X_al), Z_t)

    return predict


def fit_dipls(X_src, y_src, X_tgt, ncomp: int = 8, gamma: float = 1.0,
              per_col: bool = False):
    """Domain-Invariant PLS（di-PLS, Nikzad-Langerodi et al.）：无配对域不变潜变量回归基线。"""
    Xs = np.asarray(X_src, dtype=np.float64)
    Xt = np.asarray(X_tgt, dtype=np.float64)
    ya = np.asarray(y_src, dtype=np.float64)
    if ya.ndim == 1:
        ya = ya[:, None]
    ymu, ysd = ya.mean(0), np.clip(ya.std(0), 1e-9, None)
    yn = (ya - ymu) / ysd

    ms, mt = Xs.mean(0), Xt.mean(0)
    Xs0, Xt0 = Xs - ms, Xt - mt
    L = Xs.shape[1]
    # 当 N_s + N_t < L 时所有残差、均值差与协方差算子严格张成于统一样本子空间，SVD 降维无损加速
    if len(Xs0) + len(Xt0) < L:
        U_sub, _, _ = np.linalg.svd(np.vstack([Xs0, Xt0]).T, full_matrices=False)
        Xs_red, Xt_red = Xs0 @ U_sub, Xt0 @ U_sub
    else:
        U_sub = None
        Xs_red, Xt_red = Xs0, Xt0

    def _select_ncomp_cv(Xc, y_col, kmax: int, inner: int = 4) -> int:
        """用列内 K 折 CV 为单列选成分数（per_col=True 的语义）。

        与 per_col=False 的唯一差别就在这里：成分数是否逐列自适应。
        """
        n = len(y_col)
        kmax_eff = int(min(kmax, Xc.shape[1], max(1, n - 1)))
        if kmax_eff <= 1 or n < 2 * inner:
            return max(1, kmax_eff)
        splits = list(KFold(inner, shuffle=True, random_state=0).split(Xc))
        # 只在源域内部做 CV：绝不能借用 _fit_one 返回的 mt（那是目标域均值）。
        best_k, best_mse = 1, float("inf")
        for k in range(1, kmax_eff + 1):
            sq, cnt = 0.0, 0
            ok = True
            for itr, iva in splits:
                try:
                    mu_i = Xc[itr].mean(0)
                    m = PLSRegression(
                        n_components=max(1, min(k, len(itr) - 1, Xc.shape[1])),
                        scale=False,
                    ).fit(Xc[itr] - mu_i, y_col[itr])
                    p = m.predict(Xc[iva] - mu_i).ravel()
                    sq += float(np.sum((p - y_col[iva]) ** 2))
                    cnt += len(iva)
                except Exception:
                    ok = False
                    break
            if not ok or cnt == 0:
                continue
            mse = sq / cnt
            if mse < best_mse:
                best_mse, best_k = mse, k
        return best_k

    def _fit_one(y_col, ncomp_override=None):
        Xsk, Xtk = Xs_red.copy(), Xt_red.copy()
        yk = (y_col - y_col.mean()).copy()
        D = Xsk.shape[1]
        nc_use = int(ncomp if ncomp_override is None else ncomp_override)
        k_use = int(min(nc_use, D, len(Xs) - 1))
        W = np.empty((D, k_use))
        P = np.empty((D, k_use))
        Q = np.empty(k_use)
        used = 0
        for a in range(k_use):
            yy = float(yk @ yk)
            if yy < 1e-12:
                break
            b = Xsk.T @ yk / yy
            cov_diff = (Xsk.T @ Xsk) / max(len(Xsk) - 1, 1) - (Xtk.T @ Xtk) / max(len(Xtk) - 1, 1)
            dm = Xsk.mean(0) - Xtk.mean(0)
            M_pen = np.eye(D) + gamma * (cov_diff @ cov_diff + np.outer(dm, dm))
            w = np.linalg.solve(M_pen, b)
            wn = float(np.linalg.norm(w))
            if wn < 1e-12:
                break
            w /= wn
            t = Xsk @ w
            tt = float(t @ t)
            if tt < 1e-12:
                break
            p = (Xsk.T @ t) / tt
            t_t = Xtk @ w
            pt = (Xtk.T @ t_t) / max(float(t_t @ t_t), 1e-12)
            q = float(yk @ t) / tt
            W[:, a], P[:, a], Q[a] = w, p, q
            Xsk -= np.outer(t, p)
            Xtk -= np.outer(t_t, pt)
            yk -= t * q
            used += 1
        if used == 0:
            # 与正常分支返回同一语义的中心（目标域均值 mt），保证 pred 的 (Xa - center) 一致
            return np.zeros(L), float(y_col.mean()), mt
        R = W[:, :used] @ np.linalg.solve(P[:, :used].T @ W[:, :used], Q[:used])
        coef = R
        if U_sub is not None:
            coef = U_sub @ coef
        return coef, float(y_col.mean()), mt

    # 两种模式都逐列独立拟合（绝不能把第 0 列系数复制给其他列）。
    # per_col 的真正差异是「成分数是否逐列选」：
    #   per_col=False -> 所有列共用调用方传入的 ncomp
    #   per_col=True  -> 每列用列内 CV 选 ncomp
    #
    # !! 曾用 X 的 rank99 做 per_col=True 的上界，但 NIR 上散射/基线占 >99% 方差，
    #    rank99 恒为 1，会把所有组分静默压到 1 个成分（Corn 上 RMSEP 0.48 -> 0.99，恶化 105%）。
    #    这是典型的「用 X 的谱性质决定 y 的成分数」的错误，已废弃。
    models = []
    for j in range(yn.shape[1]):
        yj = yn[:, j]
        if per_col:
            n_j = _select_ncomp_cv(Xs_red if U_sub is not None else Xs0, yj,
                                   kmax=ncomp, inner=min(4, max(2, len(yj) // 8)))
            models.append(_fit_one(yj, ncomp_override=n_j))
        else:
            models.append(_fit_one(yj))

    def pred(X):
        Xa = np.asarray(X, dtype=np.float64)
        out = np.stack([(Xa - mt_j) @ c_j + ym_j for c_j, ym_j, mt_j in models], axis=1)
        return out * ysd + ymu

    return pred
