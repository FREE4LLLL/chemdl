"""迁移集选样准则：D-optimal / V-optimal / T-optimal (Tempered Spectral Trace) / P-optimal。

动机与实证发现：
  1. D-optimal：最大化 logdet(X_S X_S^T + lam I_k)（k×k），小预算在均匀谱集上强于随机与 KS；
     但在有效物理秩 r_eff << k 的高维共线光谱上，大 k 时因对数门槛追逐尾部特征噪声而退化。
  2. V-optimal（已修复 Woodbury 对偶求逆顺序）：
     最小化目标分布 M = U^T U / m 上的平均预测方差 tr(M (X_S^T X_S + lam I)^{-1})，
     等价于最大化 tr((X_S X_S^T + lam I_k)^{-1} X_S M X_S^T) = tr(solve(B, A))。
     （旧版误写成 solve(A, B)，导致小预算下反选出在主分布投影最小的外围噪声点。）
  3. T-optimal（Affine Scale-Invariant Tempered Spectral Trace / Effective-DoF Selection）：
     对仿射标准化（ds_affine）构造尺度不变截距增广矩阵 Z = [1_n, (X - x̄) / sqrt(tr(Cov(X)))]，
     最大化 Φ_α(S; λ) = tr((Z_S Z_S^T + λ I_k)^{-1} P_S P_S^T)。
     默认 affine=True, alpha="auto"（α=0, λ=1e-2）精确对应仿射岭回归有效自由度
     d_eff(S; λ) = Σ_j σ̃_j^2 / (σ̃_j^2 + λ)：每个有效化学方向单调饱和于 1（消除 V-Opt
     在单主成分主导谱如 IDRC-2016 Wheat 上的 PC1 垄断），低于 λ 的尾部噪声方向以线性
     斜率平滑衰减至 0（消除经典 D-Opt 大 k 追逐噪声退化），且对全局乘性缩放 X -> c·X (c>0)
     与列平移 X -> X + 1·bᵀ 严格不变。注意：逐样本行标准化 SNV 改变了样本间相对行范数，
     不属于全局缩放，因此本准则并非「SNV 不变」——早期文档在此处的表述过强，已修正。
     设置 affine=False 时回退至未中心化旧版谱幂次迹族（α=1 时严格等价于 v_opt_select）。
"""
from __future__ import annotations

import numpy as np


def _schur_trace_greedy(X: np.ndarray, P: np.ndarray, k: int, lam: float) -> np.ndarray:
    """用 Schur 补秩-1 更新精确贪心最大化 tr((X_S X_S^T + lam I)^{-1} P_S P_S^T)。

    设当前选中集为 S，增量分母 d_i = k_{ii} + lam - ||v_i||^2（精确 Schur 补），
    分子增量为 ||p_i - V^T W 的正交投影||^2，与每步重算 trace(solve(B, A)) 数学等价。
    """
    n = len(X)
    if k >= n:
        return np.arange(n, dtype=int)
    if k <= 0:
        return np.empty(0, dtype=int)
    K = X @ X.T
    chosen: list[int] = []
    mask = np.zeros(n, dtype=bool)
    V = np.zeros((k, n), dtype=np.float64)
    W = np.zeros((k, P.shape[1]), dtype=np.float64)
    gains_denom = np.diag(K).astype(np.float64, copy=True) + lam
    for step in range(k):
        P_res = P if step == 0 else P - V[:step].T @ W[:step]
        num = np.sum(P_res ** 2, axis=1)
        score = np.where(mask, -np.inf, num / np.maximum(gains_denom, 1e-12))
        bi = int(np.argmax(score))
        chosen.append(bi)
        mask[bi] = True
        if step + 1 < k:
            v = K[bi] - V[:step].T @ V[:step, bi]
            norm_v = np.sqrt(max(float(gains_denom[bi]), 1e-15))
            v = v / norm_v
            w_vec = (P[bi] - V[:step, bi] @ W[:step]) / norm_v
            V[step] = v
            W[step] = w_vec
            gains_denom = gains_denom - v ** 2
    return np.asarray(chosen, dtype=int)


def d_opt_select(X_src, k: int, lam: float = 1e-6) -> np.ndarray:
    """经典 D-optimal：精确贪心最大化 logdet(X_S X_S^T + lam I_k)（Cholesky Schur 补加速）。"""
    X = np.asarray(X_src, np.float64)
    n = len(X)
    if k >= n:
        return np.arange(n, dtype=int)
    if k <= 0:
        return np.empty(0, dtype=int)
    K = X @ X.T
    chosen: list[int] = []
    mask = np.zeros(n, dtype=bool)
    V = np.zeros((k, n), dtype=np.float64)
    gains = np.diag(K).astype(np.float64, copy=True) + lam
    for step in range(k):
        bi = int(np.argmax(np.where(mask, -np.inf, gains)))
        chosen.append(bi)
        mask[bi] = True
        if step + 1 < k:
            v = K[bi] - V[:step].T @ V[:step, bi]
            v = v / np.sqrt(max(float(gains[bi]), 1e-15))
            V[step] = v
            gains = gains - v ** 2
    return np.asarray(chosen, dtype=int)


def v_opt_select(X_src, X_unlab=None, k: int = 10, lam: float = 1e-3) -> np.ndarray:
    """V-optimal：最小化 DS 在目标（或源）分布 M = U^T U / m 上的平均预测方差。

    目标等价于最大化 tr((X_S X_S^T + lam I_k)^{-1} (X_S U^T U X_S^T / m))。
    若 X_unlab 为 None，则默认使用源域 X_src 自身，无需任何目标域样本。
    """
    X = np.asarray(X_src, np.float64)
    U = X if X_unlab is None else np.asarray(X_unlab, np.float64)
    m = max(len(U), 1)
    P = (X @ U.T) / np.sqrt(m)
    return _schur_trace_greedy(X, P, k, lam)


def estimate_spectral_temper_alpha(X_src) -> float:
    """根据源域化学方差占比 ρ = tr(Cov(X)) / tr(X^T X / n) 自适应计算幂次 α∈[0,1]。"""
    X = np.asarray(X_src, np.float64)
    total_energy = float(np.mean(np.sum(X ** 2, axis=1)))
    Xc = X - X.mean(0, keepdims=True)
    var_energy = float(np.mean(np.sum(Xc ** 2, axis=1)))
    rho = var_energy / max(total_energy, 1e-12)
    return float(np.clip((np.log10(max(rho, 1e-12)) + 3.0) / 1.3, 0.0, 1.0))


def t_opt_select(
    X_src,
    k: int,
    alpha: float | str = "auto",
    lam: float | None = None,
    affine: bool = True,
    return_alpha: bool = False,
):
    """仿射尺度不变 Tempered Spectral Trace / 有效自由度选样（T-Opt）。

    默认 ``affine=True`` 时构造与 ``ds_affine`` 严格对齐的尺度不变截距增广设计矩阵
    ``Z = [1_n, (X - mean) / sqrt(tr(Cov(X)))]``，``alpha="auto"`` 取 ``α=0``、``lam=1e-2``，
    精确最大化仿射岭估计有效自由度 ``Σ_j σ̃_j^2 / (σ̃_j^2 + lam)``（消除 SNV 下 ``α=1`` 饱和
    引起的 PC1 垄断及经典 D-Opt 的大预算尾噪退化，且对全局乘性缩放 ``X → c·X (c > 0)`` 与
    列平移 ``X → X + 1·bᵀ`` 严格不变；注意逐样本行标准化 SNV 改变了样本间相对行范数，不属于全局缩放）。
    设置 ``affine=False`` 时回退至未中心化旧版谱幂次迹族（``α=1`` 严格等价于 ``v_opt_select``）。
    """
    X = np.asarray(X_src, np.float64)
    n = len(X)
    lam_val = (1e-2 if affine else 1e-3) if lam is None else float(lam)
    if affine:
        alpha_val = 0.0 if alpha == "auto" else float(alpha)
        if k >= n:
            sel = np.arange(n, dtype=int)
            return (sel, alpha_val) if return_alpha else sel
        if k <= 0:
            sel = np.empty(0, dtype=int)
            return (sel, alpha_val) if return_alpha else sel
        Xc = X - X.mean(0, keepdims=True)
        scale = np.sqrt(max(float(np.mean(np.sum(Xc ** 2, axis=1))), 1e-12))
        Xn = Xc / scale
        ones = np.ones((n, 1), dtype=np.float64)
        Z = np.column_stack([ones, Xn])
        if abs(alpha_val) <= 1e-12:
            P = Z
        else:
            U, s, _ = np.linalg.svd(Xn, full_matrices=False)
            eig = (s ** 2) / max(n, 1)
            eig_n = eig / max(float(eig.sum()), 1e-12)
            P_x = U * (s * (eig_n ** (alpha_val / 2.0)))[None, :]
            P = np.column_stack([ones, P_x])
        sel = _schur_trace_greedy(Z, P, k, lam_val)
        return (sel, alpha_val) if return_alpha else sel

    alpha_val = estimate_spectral_temper_alpha(X) if alpha == "auto" else float(alpha)
    if k >= n:
        sel = np.arange(n, dtype=int)
        return (sel, alpha_val) if return_alpha else sel
    if abs(alpha_val) <= 1e-12:
        P = X
    else:
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        w = s * ((s ** 2 / max(n, 1)) ** (alpha_val / 2.0))
        P = U * w[None, :]
    sel = _schur_trace_greedy(X, P, k, lam_val)
    return (sel, alpha_val) if return_alpha else sel


def p_opt_select(X_src, X_unlab, k: int, gamma: float = 1.0, lam: float = 1e-6) -> np.ndarray:
    """P-optimal：logdet(X_S (I + gamma·M) X_S^T + lam·I)，兼顾分散与预测分布覆盖。

    预计算增广 Gram 矩阵 ``G = X X^T + (gamma/m) (X U^T)(X U^T)^T + lam I`` 后以 Schur 补
    增量更新对数行列式，与暴力逐候选重算 ``slogdet`` 数学全等，复杂度由 O(k n (k m L + k^3))
    降至 O(n (n + m) L + k^2 n)。
    """
    X = np.asarray(X_src, np.float64)
    U = np.asarray(X_unlab, np.float64)
    n = len(X)
    if k >= n:
        return np.arange(n, dtype=int)
    if k <= 0:
        return np.empty(0, dtype=int)
    m = max(len(U), 1)
    G = X @ X.T
    if gamma > 0.0 and len(U) > 0:
        P = X @ U.T
        G = G + (float(gamma) / float(m)) * (P @ P.T)
    diag_res = np.diag(G).astype(np.float64) + float(lam)

    chosen = []
    mask = np.zeros(n, dtype=bool)
    # 维护增广 Gram 矩阵的增量 Cholesky 因子（逐列正交投影残差）
    L_cols = np.empty((n, k), dtype=np.float64)
    for step in range(k):
        cand = np.where(~mask, diag_res, -np.inf)
        bi = int(np.argmax(cand))
        chosen.append(bi)
        mask[bi] = True
        piv = float(diag_res[bi])
        if piv <= 1e-15:
            L_cols[:, step] = 0.0
            continue
        inv_sqrt = 1.0 / np.sqrt(piv)
        col = G[:, bi].copy()
        col[bi] += float(lam)
        if step > 0:
            col = col - L_cols[:, :step] @ L_cols[bi, :step]
        col *= inv_sqrt
        L_cols[:, step] = col
        diag_res = np.maximum(diag_res - col * col, 0.0)
    return np.asarray(chosen, dtype=int)


# --------------------------------------------------------------------------- #
# 成熟文献方法引入：FCAM-SIG 变量选择 & OPCS 有监督样本选择
# 出处：Andries & Vander Heyden (2024), Anal. Bioanal. Chem.
#       DOI: 10.1007/s00216-024-05468-6
# --------------------------------------------------------------------------- #
def _pls_jackknife_sig(X: np.ndarray, y: np.ndarray, ncomp: int = 5,
                       folds: int = 8, seed: int = 0) -> np.ndarray:
    """用 n-fold leave-more-out jack-knifing 估计 PLS 回归系数显著性 SIG_j = |b_j|/se(b_j)。"""
    from sklearn.cross_decomposition import PLSRegression

    X = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if ya.ndim == 1:
        ya = ya[:, None]
    n, L = X.shape
    eff_folds = min(int(folds), max(2, n // 2))
    idx = np.random.default_rng(seed).permutation(n)
    sigs = []
    for col in range(ya.shape[1]):
        yc = ya[:, col]
        coefs = []
        for f in range(eff_folds):
            hold = idx[f::eff_folds]
            keep = np.setdiff1d(idx, hold)
            if len(keep) < 3:
                continue
            k_use = min(int(ncomp), len(keep) - 1, L)
            m = PLSRegression(n_components=max(1, k_use), scale=False)
            m.fit(X[keep], yc[keep])
            coefs.append(np.asarray(m.coef_, dtype=np.float64).ravel())
        if len(coefs) < 2:
            sigs.append(np.ones(L, dtype=np.float64))
        else:
            C = np.asarray(coefs, dtype=np.float64)
            b = C.mean(0)
            # Jack-knife 方差：Var_JK = (F-1)/F * sum_f (b_f - b_bar)^2
            # 而 C.std(ddof=1) = sqrt(sum_f (b_f - b_bar)^2 / (F-1))
            # => se_JK = std * (F-1)/sqrt(F)。原实现漏了这个因子，
            # F=8 时低估标准误 2.47 倍（SIG 虚高同倍数；仅排序时无影响，有阈值判定时假阳性）。
            F = len(coefs)
            se = C.std(0, ddof=1) * ((F - 1) / np.sqrt(F)) + 1e-12
            sigs.append(np.abs(b) / se)
    return np.max(np.stack(sigs, axis=0), axis=0)


def fcam_sig_select(
    X,
    y,
    keep_frac: float = 0.3,
    ncomp: int = 5,
    rounds: int = 6,
    folds: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """FCAM-SIG 向后变量消除（Andries & Vander Heyden 2024, Sec. 2.1）。

    通过 n-fold jack-knife 估计 PLS 回归系数显著性 ``SIG_j = |b_j| / se(b_j)``，
    逐轮剔除最不显著波长直至保留 ``keep_frac`` 比例的变量。返回升序波长索引。
    """
    X = np.asarray(X, dtype=np.float64)
    L = X.shape[1]
    alive = np.arange(L, dtype=int)
    target = max(1, min(L, int(round(L * float(keep_frac)))))
    for _ in range(max(1, int(rounds))):
        if len(alive) <= target:
            break
        sig = _pls_jackknife_sig(X[:, alive], y, ncomp=ncomp, folds=folds, seed=seed)
        order = np.argsort(-sig)
        n_keep = max(target, int(len(alive) * 0.7))
        alive = alive[order[:n_keep]]
    return np.sort(alive)


def opcs_select(
    X_src,
    y_src,
    k: int,
    ncomp: int | None = None,
    strategy: str = "ks",
    lam: float = 1e-2,
) -> np.ndarray:
    """OPCS 有监督属性空间选样（借鉴 Andries & Vander Heyden 2024 模型导向选样思想）。

    作为无监督设计导向 ``t_opt_select`` 的有监督对照：先在源域 ``(X_src, y_src)`` 上
    拟合 PLS 提取与目标化学属性直接相关的潜变量得分空间 ``T``（按各成分对 y 的解释方差加权），
    再在该低维属性空间内用 Kennard-Stone（``strategy='ks'``）或仿射有效自由度
    ``t_opt_select``（``strategy='t_opt'``）挑选 ``k`` 个迁移标准样。
    """
    from sklearn.cross_decomposition import PLSRegression

    X = np.asarray(X_src, dtype=np.float64)
    ya = np.asarray(y_src, dtype=np.float64)
    if ya.ndim == 1:
        ya = ya[:, None]
    n, L = X.shape
    if k >= n:
        return np.arange(n, dtype=int)
    if k <= 0:
        return np.empty(0, dtype=int)

    ymu, ysd = ya.mean(0), np.clip(ya.std(0), 1e-9, None)
    yn = (ya - ymu) / ysd
    r = int(min(8 if ncomp is None else ncomp, n - 1, L))
    r = max(1, r)
    pls = PLSRegression(n_components=r, scale=False).fit(X, yn)
    T = np.asarray(pls.x_scores_, dtype=np.float64)
    # 按每个潜变量成分对归一化 y 的解释方差开方加权，抑制尾部噪声方向
    Q = np.asarray(pls.y_loadings_, dtype=np.float64)  # (n_targets, r)
    # 每个成分 a 对归一化 Y 的实际贡献幅度是 ||t_a|| * ||q_a|| / sqrt(n)，
    # 开方即 sqrt(mean(t^2)) * ||q|| = t_norm * q_w。
    # 原实现写成 (T / t_norm) * q_w：由于 sklearn 的 q_a = Y^T t_a / (t_a^T t_a) 与 1/||t_a|| 成正比，
    # 除以 t_norm 会把尾部弱得分成分放大 1/||t_a|| 倍，与「抑制尾部噪声」的设计初衷相反。
    t_norm = np.sqrt(np.clip(np.mean(T ** 2, axis=0), 1e-12, None))
    q_w = np.sqrt(np.clip(np.sum(Q ** 2, axis=0), 1e-12, None))
    T_w = T * q_w[None, :]
    if strategy == "t_opt":
        return t_opt_select(T_w, k, affine=True, lam=lam)
    # 默认 Kennard-Stone 覆盖属性潜变量空间
    sq = np.sum(T_w ** 2, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (T_w @ T_w.T), 0.0)
    np.fill_diagonal(d2, -1.0)
    i0, i1 = np.unravel_index(int(np.argmax(d2)), d2.shape)
    if k == 1:
        return np.array([int(i0)], dtype=int)
    sel = [int(i0), int(i1)]
    chosen = np.zeros(n, dtype=bool)
    chosen[sel] = True
    np.fill_diagonal(d2, 0.0)
    min_d2 = np.minimum(d2[i0], d2[i1])
    while len(sel) < k:
        nxt = int(np.argmax(np.where(~chosen, min_d2, -1.0)))
        sel.append(nxt)
        chosen[nxt] = True
        min_d2 = np.minimum(min_d2, d2[nxt])
    return np.asarray(sel, dtype=int)
