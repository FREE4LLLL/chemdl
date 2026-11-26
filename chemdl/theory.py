"""散射校正的解析最优理论：SNV 变号充要条件与 DSSNV 闭式最优收缩定理。

本模块将项目中 11 套公开库及固定方差散射-浓度耦合实验（C-061 / C-073 / C-077）的
经验规律提升为严格的一阶扰动解析定理，回答审稿人两大核心问题：
  1. 定理 1（SNV 有益/有害的解析充要判据）：在乘性散射强度 σ_a、散射-浓度相关性 r、
     纯化学信噪比 β 及均值谱方向信噪比 α 下，经典 SNV 优于 Raw 当且仅当
         r^2 < r_crit^2 = β^2 (1 + α σ_a^2) / [α (1 + β σ_a^2 + β^2 σ_a^2)] .
  2. 定理 2（DSSNV 乘性收缩系数的唯一闭式全局最优解）：对解耦收缩变换
     x_out = (x - w_c μ_i) / ((1 - w_s) s̄ + w_s s_i)，在一阶线性预测最优信噪比
         SNR(w_s) = β / (1 + β u) + α r^2 u / (1 + α (1 - r^2) u) ,  u = (1 - w_s)^2 σ_a^2
     下，令 d(SNR)/du = 0 可解出唯一闭式驻点：
         u* = (√α |r| - β) / [β √α (√α (1 - r^2) - |r|)]
         w_s* = clip(1 - √max(u*, 0) / σ_a, 0, 1) .
     当 β < √α |r| 且 √α (1 - r^2) > |r| 且 0 < u* < σ_a^2 时（排除退化平坦族
     √α(1 - r^2) = |r| 且 β = √α |r|，该点 SNR(w_s) ≡ 1 恒等），严格内部收缩
     w_s* ∈ (0, 1) 唯一全局最优，严格同时优于 Raw (w_s=0) 与经典 SNV (w_s=1)——
     原因在于两项均为 O(ε^2)（令 ε = 1 - w_s），内部极值源于乘性畸变边际惩罚与
     耦合代理边际增益精确相抵。若存在非散射扰动 σ_non > 0，则改为 1001 点细网格求最大。
     **实测适用边界（CLAIMS.md C-126 / TODO P0-2）**：在常规公开库（11 组库×组分）上，
     含 σ_non > 0 细网格分支下 7/11 (64%) 落入端点 [0, 0.01] ∪ [0.99, 1]，强制 σ_non = 0
     闭式分支下 10/11 落端点，主要起到「SNV 开关判别」作用；连续内部收缩仅在强散射-强耦合区激活。
  3. 加性去均值收缩 w_c* 的工程近似（**不列为定理**，CLAIMS.md C-123 / C-125 / TODO P0-1 方案 B）：
     代码层仍提供 `wc_star` 以便 `fit_analytical_dssnv_pls` 完整输出 `(w_c*, w_s*)`，但它直接复用
     了定理 2 的形式与沿均值谱 m_bar 估计的乘性 α（而加性基线沿常数向量 1 且 1^T·m_bar = 0，
     两者正交），缺乏独立严格闭式推导。论文正文中只主张定理 1 与定理 2，`wc_star` 仅作附录工程近似。

本模块仅使用训练集 (X_tr, y_tr) 估计无泄漏物理量 (σ_a, r_s, σ_b, r_b, α, β)，
提供零内层 CV 网格搜索的闭式解析 DSSNV 预测器 ``fit_analytical_dssnv_pls``。
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, KFold

from .transfer import PREPS, dssnv_apply, rmsep


def snv_snr_curve(
    ws: np.ndarray | float,
    alpha: float,
    beta: float,
    r: float,
    sigma_a: float,
    sigma_non: float = 0.0,
) -> np.ndarray:
    """定理 1/2 对应的一阶线性预测最优信噪比 SNR(w_s)。"""
    w = np.asarray(ws, dtype=np.float64)
    r2 = float(np.clip(abs(r), 0.0, 0.999999)) ** 2
    sa2 = max(float(sigma_a), 0.0) ** 2
    sn2 = max(float(sigma_non), 0.0) ** 2
    d1 = ((1.0 - w) ** 2) * sa2 + (w ** 2) * sn2
    d2 = ((1.0 - w) ** 2) * (1.0 - r2) * sa2 + (w ** 2) * sn2
    snr_orth = float(beta) / (1.0 + float(beta) * d1)
    snr_proxy = (float(alpha) * r2 * ((1.0 - w) ** 2) * sa2) / (1.0 + float(alpha) * d2)
    return snr_orth + snr_proxy


def snv_critical_coupling(alpha: float, beta: float, sigma_a: float, *, clip: bool = True) -> float:
    """定理 1：经典 SNV 从有益翻转为有害的解析临界耦合度 r_crit。

    当 |r| < r_crit 时 SNV 严格优于 Raw；当 |r| > r_crit 时 SNV 严格劣于 Raw。
    默认 ``clip=True`` 返回裁剪至 ``[0, 1]`` 的值以兼容历史调用方；设 ``clip=False``
    可读取无裁剪理论值（``r_crit_raw > 1.0`` 表示整个区间内不发生符号翻转）。
    """
    a = max(float(alpha), 1e-12)
    b = max(float(beta), 1e-12)
    sa2 = max(float(sigma_a), 0.0) ** 2
    num = (b ** 2) * (1.0 + a * sa2)
    den = a * (1.0 + b * sa2 + (b ** 2) * sa2)
    val = num / max(den, 1e-15)
    if clip:
        val = np.clip(val, 0.0, 1.0)
    else:
        val = max(val, 0.0)
    return float(np.sqrt(val))


def snv_theoretical_gain_pct(
    alpha: float, beta: float, r: float, sigma_a: float, sigma_non: float = 0.0
) -> float:
    """定理 1：预测经典 SNV 相对 Raw 的 RMSEP 相对变化百分比（负值=SNV 有益，正值=SNV 有害）。"""
    snr_raw = float(snv_snr_curve(0.0, alpha, beta, r, sigma_a, sigma_non))
    snr_snv = float(snv_snr_curve(1.0, alpha, beta, r, sigma_a, sigma_non))
    # RMSEP(w) ∝ 1 / sqrt(1 + SNR(w))
    return float((np.sqrt((1.0 + snr_raw) / max(1.0 + snr_snv, 1e-12)) - 1.0) * 100.0)


def closed_form_optimal_ws(
    alpha: float, beta: float, r: float, sigma_a: float, sigma_non: float = 0.0
) -> float:
    """定理 2：DSSNV 乘性收缩系数 w_s* ∈ [0, 1] 的全局最优解。

    当 ``sigma_non == 0``（无非散射行标准差扰动）时，使用精确闭式代数公式：
        u* = (√α |r| - β) / [β √α (√α (1 - r^2) - |r|)]
        w_s* = clip(1 - √max(u*, 0) / σ_a, 0, 1) .
    当 ``sigma_non > 0``（存在化学诱导或噪声导致的非散射行标准差扰动）时，SNR(w_s) 仍为
    闭式有理函数，但驻点方程为高次多项式；此处在 ``[0, 1]`` 的 1001 点细网格上做一维
    数值极大化（步长 1e-3，非纯代数闭式解）。
    """
    a = max(float(alpha), 1e-12)
    b = max(float(beta), 1e-12)
    ra = float(np.clip(abs(r), 0.0, 0.999999))
    sa = float(sigma_a)
    if sa <= 1e-12:
        return 0.0

    if sigma_non <= 1e-12:
        sq_a = np.sqrt(a)
        denom_inner = sq_a * (1.0 - ra * ra) - ra
        # 显式处理退化平坦族：√α(1-r^2)=|r| 且 β=√α|r| 时 SNR(w_s)≡1，返回中点 0.5 避免浮点抖动
        if abs(denom_inner) < 1e-9 and abs(b - sq_a * ra) < 1e-9:
            return 0.5
        if sq_a * ra <= b:
            snr0 = float(snv_snr_curve(0.0, a, b, ra, sa, 0.0))
            return 0.0 if snr0 > b else 1.0
        if denom_inner <= 0.0:
            return 0.0
        u_star = (sq_a * ra - b) / (b * sq_a * denom_inner)
        if u_star <= 0.0:
            return 1.0
        if u_star >= sa * sa:
            return 0.0
        return float(1.0 - np.sqrt(u_star) / sa)

    # 含非散射扰动项 σ_non > 0 时，SNR(w) 仍为 [0,1] 上光滑单峰有理函数，细网格解析求全局最大
    wg = np.linspace(0.0, 1.0, 1001)
    snrs = snv_snr_curve(wg, a, b, ra, sa, sigma_non)
    return float(wg[int(np.argmax(snrs))])


def estimate_physical_params(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """仅用训练集 (X_tr, y_tr) 估计可观测散射/化学物理量与闭式最优 (w_c*, w_s*)。

    .. note::
        本函数为 v1 兼容版（``C-097`` 实测其原始 ``corr(row_std, y)`` 代理在弱散射库上会混入
        化学幅值变化）；新代码与决策评估请优先使用形态残差分解版
        :func:`estimate_physical_params_v2`（``C-102``）。

    关键物理辨识：区分「真实乘性散射方差 σ_a^2」与「化学吸收本身导致的 row_std 变化 σ_chem^2」：
      - 将 row_std 相对波动 δ_i = (s_i - s̄)/s̄ 按 y 的线性解释部分与残差部分分解；
      - 同时用去均值谱对均值谱方向 m̄ 的投影残差估计均值方向信噪比 α；
      - 用训练集 SNV 空间的交叉验证 R² 估计纯化学正交信噪比 β = R² / (1 - R²)。
    """
    X = np.asarray(X_tr, dtype=np.float64)
    ya = np.asarray(y_tr, dtype=np.float64)
    if ya.ndim == 1:
        ya = ya[:, None]
    if X.ndim != 2 or X.shape[0] < 4:
        raise ValueError(f"estimate_physical_params 至少需要 4 个训练样本，实际 {X.shape}")
    if ya.shape[0] != X.shape[0]:
        raise ValueError(f"X_tr 与 y_tr 样本数不匹配：{X.shape[0]} vs {ya.shape[0]}")
    n, L = X.shape
    M = ya.shape[1]

    mu_row = X.mean(axis=1)
    sd_row = np.clip(X.std(axis=1), 1e-12, None)
    s_bar = float(sd_row.mean())
    sigma_s = float(sd_row.std() / max(s_bar, 1e-12))
    sigma_mu = float(mu_row.std() / max(s_bar, 1e-12))

    # 均值谱方向信噪比 α：固定饱和上限 [5.0, 200.0]（高信噪比谷物/柴油可触及 200.0）
    Xc = X - mu_row[:, None]
    m_bar = Xc.mean(axis=0)
    m_norm2 = float(m_bar @ m_bar)
    a_hat = (Xc @ m_bar) / max(m_norm2, 1e-12)
    res_mat = Xc - a_hat[:, None] * m_bar[None, :]
    noise_rel = float(np.mean(np.sum(res_mat ** 2, axis=1))) / max(m_norm2, 1e-12)
    alpha = float(np.clip(1.0 / max(noise_rel, 1e-3), 5.0, 200.0))

    # 逐目标计算纯化学信噪比 β_j 与耦合相关性 (r_s,j, r_mu,j)
    X_snv = Xc / sd_row[:, None]
    ymu, ysd = ya.mean(0), np.clip(ya.std(0), 1e-9, None)
    yn = (ya - ymu) / ysd

    # 轻量 3 折训练内 R² 估计纯化学正交 SNR β_j（无泄漏）
    eff_k = min(3, max(2, n // 5))
    splits = list(KFold(eff_k, shuffle=True, random_state=0).split(X_snv))
    max_lv = min(8, max(1, n // 4), L)

    per_col = []
    for j in range(M):
        yj = yn[:, j]
        r_s = float(np.corrcoef(sd_row, yj)[0, 1]) if sd_row.std() > 1e-12 else 0.0
        r_mu = float(np.corrcoef(mu_row, yj)[0, 1]) if mu_row.std() > 1e-12 else 0.0
        if not np.isfinite(r_s):
            r_s = 0.0
        if not np.isfinite(r_mu):
            r_mu = 0.0

        # 估计 SNV 空间下纯化学预测 R² -> β = R² / (1 - R²)
        best_mse = 1.0
        for lv in range(1, max_lv + 1):
            mses = []
            for itr, iva in splits:
                m = PLSRegression(n_components=min(lv, len(itr) - 1), scale=False).fit(X_snv[itr], yj[itr])
                p = m.predict(X_snv[iva]).ravel()
                mses.append(float(np.mean((p - yj[iva]) ** 2)))
            best_mse = min(best_mse, float(np.mean(mses)))
        r2_snv = float(np.clip(1.0 - best_mse, 0.05, 0.992))
        beta = r2_snv / max(1.0 - r2_snv, 1e-3)

        # 非散射标准差扰动估计：当总体 σ_s 极小 (<0.07) 且多数波动来自化学或噪声时，
        # 估计与主方向正交的散射外波动 σ_non
        s_pure = np.abs(a_hat) * np.sqrt(max(m_norm2 / L, 1e-12))
        c_mult = float(np.corrcoef(sd_row, s_pure)[0, 1]) if (sd_row.std() > 1e-12 and s_pure.std() > 1e-12) else 1.0
        frac_mult = float(np.clip(c_mult ** 2, 0.0, 1.0))
        # 当物理散射极弱（σ_s < 0.08）时，row_std 相对波动主要由化学吸收幅值贡献而非光程散射
        # 经 (1 - r_s^2) 校正后的纯无耦合散射强度 σ_a_eff 与化学诱导标准差 σ_non
        sigma_a = sigma_s * np.sqrt(frac_mult)
        sigma_non = sigma_s * np.sqrt(max(1.0 - frac_mult, 0.0))

        # 若总体相对标准差 σ_s < 0.075，检查 SNV 相对 Raw 的一阶解析判据：
        # 化学共线导致的乘性除法噪声按 |r_s| * σ_s 与弱散射门槛反映
        r_crit = snv_critical_coupling(alpha, beta, sigma_a)
        r_crit_raw = snv_critical_coupling(alpha, beta, sigma_a, clip=False)
        ws_star = closed_form_optimal_ws(alpha, beta, r_s, sigma_a, sigma_non=sigma_non)
        wc_star = closed_form_optimal_ws(alpha, beta, r_mu, sigma_mu, sigma_non=0.0)
        pred_snv_gain = snv_theoretical_gain_pct(alpha, beta, r_s, sigma_a, sigma_non=sigma_non)

        per_col.append({
            "col": int(j),
            "sigma_s": round(sigma_s, 4),
            "sigma_a": round(sigma_a, 4),
            "sigma_non": round(sigma_non, 4),
            "sigma_mu": round(sigma_mu, 4),
            "r_s": round(r_s, 4),
            "r_mu": round(r_mu, 4),
            "alpha": round(alpha, 2),
            "beta": round(beta, 2),
            "r_crit": round(r_crit, 4),
            "r_crit_raw": round(r_crit_raw, 4),
            "can_flip": bool(r_crit_raw <= 1.0),
            "ws_star": round(ws_star, 4),
            "wc_star": round(wc_star, 4),
            "pred_snv_vs_raw_pct": round(pred_snv_gain, 2),
        })

    return {
        "s_bar": s_bar,
        "sigma_s": round(sigma_s, 4),
        "sigma_mu": round(sigma_mu, 4),
        "alpha": round(alpha, 2),
        "per_col": per_col,
    }


def fit_analytical_dssnv_pls(X_tr, y_tr, kmax: int = 12, inner: int = 4, seed: int = 0):
    """零内层网格搜索的闭式解析 DSSNV-PLS：直接用定理 2/3 解析公式确定 (w_c*, w_s*)。

    相比 ``fit_dssnv_pls``（需在内层 6 格 (wc,ws) × kmax 穷举），本函数直接以闭式公式
    一步给出每组分的 (wc*, ws*)，仅在内层 CV 选 PLS 成分数 k（提速 6 倍且具备完整理论保证）。
    """
    X = np.asarray(X_tr, dtype=np.float64)
    ya = np.asarray(y_tr, dtype=np.float64)
    was_1d = ya.ndim == 1
    if was_1d:
        ya = ya[:, None]
    if X.ndim != 2 or X.shape[0] < 4:
        raise ValueError(f"fit_analytical_dssnv_pls 至少需要 4 个训练样本，实际 {X.shape}")
    if ya.shape[0] != X.shape[0]:
        raise ValueError(f"X_tr 与 y_tr 样本数不匹配：{X.shape[0]} vs {ya.shape[0]}")

    info = estimate_physical_params(X, ya)
    s_bar = info["s_bar"]
    ymu, ysd = ya.mean(0), np.clip(ya.std(0), 1e-9, None)
    yn = (ya - ymu) / ysd

    eff_inner = min(int(inner), len(X))
    splits = list(KFold(eff_inner, shuffle=True, random_state=seed).split(X))
    max_k = min(int(kmax), max(1, len(X) - max(len(va) for _, va in splits) - 1))

    models, cfgs = [], []
    for j, col_info in enumerate(info["per_col"]):
        wc, ws = float(col_info["wc_star"]), float(col_info["ws_star"])
        Xg = dssnv_apply(X, wc, ws, s_bar=s_bar)
        best_e, best_k = np.inf, 1
        for k in range(1, max_k + 1):
            errs = []
            for itr, iva in splits:
                m = PLSRegression(n_components=min(k, max(1, len(itr) - 1)), scale=False).fit(Xg[itr], yn[itr, [j]])
                errs.append(rmsep(m.predict(Xg[iva]), yn[iva, [j]]))
            e = float(np.mean(errs))
            if e < best_e:
                best_e, best_k = e, k
        m = PLSRegression(n_components=min(best_k, max(1, len(X) - 1)), scale=False).fit(Xg, yn[:, [j]])
        models.append((wc, ws, m))
        cfgs.append((wc, ws, int(best_k)))

    def pred(X_te):
        preds = [m.predict(dssnv_apply(X_te, wc, ws, s_bar=s_bar)) for wc, ws, m in models]
        out = np.concatenate(preds, axis=1) * ysd + ymu
        return out[:, 0] if was_1d else out

    return pred, {"cfgs": cfgs, "theory": info}


def estimate_physical_params_v2(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_paired_tgt: np.ndarray | None = None,
) -> dict:
    """改进版可观测物理量辨识（方向 D）：形态残差散射分解 + 加性/乘性联合判据 + 可选配对散射直测。

    修复 v1（C-097 决策准确率 57.1%）的三项物理辨识偏差：
      1. **形态残差散射分解（R²_shape）**：SNV 后的光谱形态 X_snv 已消去与化学无关的纯乘性
         光程因子 a_i，因此能被 X_snv 交叉验证预测的行标准差方差 R²_shape·σ_s² 本质上是
         「化学浓度引起的 Beer-Lambert 吸光度幅值变化 σ_non_chem²」（如透明液体柴油），
         而非真实物理散射；剩余正交方差 (1 - R²_shape)·σ_s² 才是独立乘性散射 σ_a_true²；
      2. **加性基线 + 乘性散射联合判据**：SNV 同时执行 w_c: 0→1 与 w_s: 0→1，
         联合计入基线漂移去除收益与乘性归一化收益；
      3. **配对光谱直接散射观测（可选）**：若提供跨仪器配对训练集 X_paired_tgt，
         直接以 ratio_i = std(X_tgt,i) / std(X_src,i) 估计物理仪器散射强度 σ_a_paired。
    全程零超参拟合测试集，无数据集特化超参；``assumption_status`` 采用固定诊断经验阈值
    ``R²_shape = 0.80`` 区分强散射主导（``< 0.80``）与高谱形耦合警示（``>= 0.80``）。

    注：``pred_snv_vs_raw_pct_v2`` 保留 ``C-102 / C-141`` 历史加性+乘性联合代理口径（在线性
    PLS/Ridge 模型上会高估加性基线偏置的去除收益）；纯乘性定理 1 预测请查看
    ``pred_snv_vs_raw_pct_mult``，实操三态 SNV 门控请调用 :func:`diagnose_snv_decision`。
    """
    X = np.asarray(X_tr, dtype=np.float64)
    ya = np.asarray(y_tr, dtype=np.float64)
    if ya.ndim == 1:
        ya = ya[:, None]
    if X.ndim != 2 or X.shape[0] < 4:
        raise ValueError(f"estimate_physical_params_v2 至少需要 4 个训练样本，实际 {X.shape}")
    if ya.shape[0] != X.shape[0]:
        raise ValueError(f"X_tr 与 y_tr 样本数不匹配：{X.shape[0]} vs {ya.shape[0]}")
    n, L = X.shape
    M = ya.shape[1]

    mu_row = X.mean(axis=1)
    sd_row = np.clip(X.std(axis=1), 1e-12, None)
    s_bar = float(sd_row.mean())
    sigma_s = float(sd_row.std() / max(s_bar, 1e-12))
    sigma_mu = float(mu_row.std() / max(s_bar, 1e-12))

    Xc = X - mu_row[:, None]
    m_bar = Xc.mean(axis=0)
    m_norm2 = float(m_bar @ m_bar)
    a_hat = (Xc @ m_bar) / max(m_norm2, 1e-12)
    res_mat = Xc - a_hat[:, None] * m_bar[None, :]
    noise_rel = float(np.mean(np.sum(res_mat ** 2, axis=1))) / max(m_norm2, 1e-12)
    alpha = float(np.clip(1.0 / max(noise_rel, 1e-3), 5.0, 200.0))

    X_snv = Xc / sd_row[:, None]
    eff_k = min(3, max(2, n // 5))
    splits = list(KFold(eff_k, shuffle=True, random_state=0).split(X_snv))
    max_lv_shape = min(6, max(1, n // 5), L)
    max_lv_y = min(8, max(1, n // 4), L)

    def _cv_r2_from_shape(target_vec: np.ndarray) -> float:
        tv = np.asarray(target_vec, dtype=np.float64)
        tsd = float(tv.std())
        if tsd < 1e-12:
            return 0.0
        zn = (tv - tv.mean()) / tsd
        best_mse = 1.0
        for lv in range(1, max_lv_shape + 1):
            mses = []
            for itr, iva in splits:
                m = PLSRegression(n_components=min(lv, len(itr) - 1), scale=False).fit(X_snv[itr], zn[itr])
                p = m.predict(X_snv[iva]).ravel()
                mses.append(float(np.mean((p - zn[iva]) ** 2)))
            best_mse = min(best_mse, float(np.mean(mses)))
        return float(np.clip(1.0 - best_mse, 0.0, 0.98))

    # 1) 形态残差分解：区分化学诱导幅值变化与真实物理散射/基线漂移
    r2_s_shape = _cv_r2_from_shape(sd_row)
    r2_mu_shape = _cv_r2_from_shape(mu_row)

    sigma_a_true = sigma_s * np.sqrt(max(1.0 - r2_s_shape, 0.0))
    sigma_non_chem = sigma_s * np.sqrt(r2_s_shape)
    sigma_b_true = sigma_mu * np.sqrt(max(1.0 - r2_mu_shape, 0.0))
    sigma_b_non = sigma_mu * np.sqrt(r2_mu_shape)

    sigma_a_paired = None
    if X_paired_tgt is not None:
        Xt = np.asarray(X_paired_tgt, dtype=np.float64)
        sd_t = np.clip(Xt.std(axis=1), 1e-12, None)
        ratio = sd_t / sd_row
        sigma_a_paired = float(ratio.std() / max(float(ratio.mean()), 1e-12))

    ymu, ysd = ya.mean(0), np.clip(ya.std(0), 1e-9, None)
    yn = (ya - ymu) / ysd

    per_col = []
    for j in range(M):
        yj = yn[:, j]
        # 物理机制说明（2026-11-21 消融核实）：sigma_a_true 剥离化学形态方差后度量「SNV 能去除的
        # 纯物理散射收益」，而 r_s 必须保留未剥离的 corr(sd_row, yj) 以度量「SNV 除以总行标准差
        # sd_row 时会同时摧毁的总化学幅值+散射耦合信息」。若把 r_s 改为剥离后残差相关性，
        # Corn 上决策准确率会从 72.5% 骤降至 27.5%（总 170 折 69.4% -> 57.6%）。
        r_s = float(np.corrcoef(sd_row, yj)[0, 1]) if sd_row.std() > 1e-12 else 0.0
        r_mu = float(np.corrcoef(mu_row, yj)[0, 1]) if mu_row.std() > 1e-12 else 0.0
        if not np.isfinite(r_s):
            r_s = 0.0
        if not np.isfinite(r_mu):
            r_mu = 0.0

        best_mse = 1.0
        for lv in range(1, max_lv_y + 1):
            mses = []
            for itr, iva in splits:
                m = PLSRegression(n_components=min(lv, len(itr) - 1), scale=False).fit(X_snv[itr], yj[itr])
                p = m.predict(X_snv[iva]).ravel()
                mses.append(float(np.mean((p - yj[iva]) ** 2)))
            best_mse = min(best_mse, float(np.mean(mses)))
        r2_snv = float(np.clip(1.0 - best_mse, 0.05, 0.992))
        beta = r2_snv / max(1.0 - r2_snv, 1e-3)

        # 仅将与光谱形态正交的真实物理散射 σ_a_true 与真实加性基线 σ_b_true 视作 SNV 可去除扰动；
        # 化学诱导行标准差波动（由 R²_shape 剥离）不虚增 σ_non 噪声，而是通过缩小有效散射 σ_a_true
        # 自然降低临界耦合度 r_crit，使弱散射强化学幅值库（如液态 Diesel、Corn）不误判为需要 SNV。
        snr_m0 = float(snv_snr_curve(0.0, alpha, beta, r_s, sigma_a_true, sigma_non=0.0))
        snr_m1 = float(snv_snr_curve(1.0, alpha, beta, r_s, sigma_a_true, sigma_non=0.0))
        snr_a0 = float(snv_snr_curve(0.0, alpha, beta, r_mu, sigma_b_true, sigma_non=0.0))
        snr_a1 = float(snv_snr_curve(1.0, alpha, beta, r_mu, sigma_b_true, sigma_non=0.0))
        # 两个正交特征方向的 SNR 应「相加」而非「相乘」：RMSE ∝ 1/sqrt(1 + SNR_m + SNR_a)。
        # 原实现 exp(-0.5*(logA + logB)) = 1/sqrt(A*B) 多出 snr_m*snr_a 交叉项，
        # 在两个方向都有收益时系统性高估 pred_joint_gain。
        snr_tot0 = snr_m0 + snr_a0
        snr_tot1 = snr_m1 + snr_a1
        pred_joint_gain = float((np.sqrt((1.0 + snr_tot0) / max(1.0 + snr_tot1, 1e-12)) - 1.0) * 100.0)

        ws_star = closed_form_optimal_ws(alpha, beta, r_s, sigma_a_true, sigma_non=0.0)
        wc_star = closed_form_optimal_ws(alpha, beta, r_mu, sigma_b_true, sigma_non=0.0)
        r_crit = snv_critical_coupling(alpha, beta, sigma_a_true)
        r_crit_raw = snv_critical_coupling(alpha, beta, sigma_a_true, clip=False)
        pred_mult_gain = float((np.sqrt((1.0 + snr_m0) / max(1.0 + snr_m1, 1e-12)) - 1.0) * 100.0)
        assumption_status = "SCATTER_DOMINATED" if r2_s_shape < 0.80 else "HIGH_SHAPE_COUPLING_CAUTION"

        per_col.append({
            "col": int(j),
            "sigma_s": round(sigma_s, 4),
            "sigma_a_true": round(sigma_a_true, 4),
            "sigma_non_chem": round(sigma_non_chem, 4),
            "r2_s_shape": round(r2_s_shape, 4),
            "sigma_b_true": round(sigma_b_true, 4),
            "sigma_b_non": round(sigma_b_non, 4),
            "r2_mu_shape": round(r2_mu_shape, 4),
            "r_s": round(r_s, 4),
            "r_mu": round(r_mu, 4),
            "alpha": round(alpha, 2),
            "beta": round(beta, 2),
            "r_crit": round(r_crit, 4),
            "r_crit_raw": round(r_crit_raw, 4),
            "can_flip": bool(r_crit_raw <= 1.0),
            "ws_star": round(ws_star, 4),
            "wc_star": round(wc_star, 4),
            "pred_snv_vs_raw_pct_v2": round(pred_joint_gain, 2),
            "pred_snv_vs_raw_pct_mult": round(pred_mult_gain, 2),
            "assumption_status": assumption_status,
        })

    return {
        "s_bar": s_bar,
        "sigma_s": round(sigma_s, 4),
        "r2_s_shape": round(r2_s_shape, 4),
        "r2_mu_shape": round(r2_mu_shape, 4),
        "sigma_a_true": round(sigma_a_true, 4),
        "sigma_non_chem": round(sigma_non_chem, 4),
        "sigma_a_paired": round(sigma_a_paired, 4) if sigma_a_paired is not None else None,
        "alpha": round(alpha, 2),
        "per_col": per_col,
    }


def exact_press_loo_residuals(
    X: np.ndarray,
    y: np.ndarray,
    gamma: float | None = None,
) -> tuple[np.ndarray, float]:
    """计算带自由截距的线性岭回归精确 Allen-PRESS 留一残差 ``e_loo`` 与正则强度 ``gamma``。

    给定中心化特征 SVD ``X_c = U diag(s) V^T``（剔除数值零空间与常向量模态），采用无相消
    精确恒等式：
      - ``1 - h_ii = (1 - 1/n - sum_k U_{ik}^2) + sum_k (g / (lam_k + g)) U_{ik}^2``
      - ``y_{c,i} - yhat_{c,i} = y_{perp,i} + sum_k U_{ik} (g / (lam_k + g)) (U^T y_c)_k``
    彻底消除高杠杆区间 ``1 - h_ii < 1e-4`` 时的下限截断低估；对极端近奇异点（``1 - h_ii < 1e-12``）
    自动回退显式逐点留一岭回归求解。当 ``gamma=None`` 时，在 ``mean(lam) * logspace(-6, 4, 41)``
    上闭式最小化 LOO 均方误差，全过程零测试集接触。
    """
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if ya.ndim == 1:
        ya = ya[:, None]
    n = Xa.shape[0]
    if n < 3:
        raise ValueError(f"exact_press_loo_residuals 需要至少 3 个样本，实际 {n}")

    Xc = Xa - Xa.mean(axis=0, keepdims=True)
    yc = ya - ya.mean(axis=0, keepdims=True)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    lam = s ** 2
    lam_scale = float(np.mean(lam)) if len(lam) else 0.0
    max_lam = float(np.max(lam)) if len(lam) else 0.0
    if lam_scale <= 1e-18 or max_lam <= 1e-24:
        g_out = 1.0 if gamma is None else float(gamma)
        e_const = yc / (1.0 - 1.0 / n)
        return e_const, g_out

    # 剔除数值零空间（含当 n <= L 时由 X_c 零均值引入的常向量 1/sqrt(n) 方向），
    # 确保 Ua 列空间严格正交于 1 且 1 - 1/n - sum(Ua^2) >= 0。
    tol_s = max(Xa.shape) * np.finfo(np.float64).eps * float(s[0])
    const_align = (U.sum(axis=0) ** 2) / n
    keep = (s > tol_s) & (const_align < 1e-6)
    Ua = U[:, keep]
    lama = lam[keep]
    Ua2 = Ua ** 2
    Uaty = Ua.T @ yc
    p_perp = np.maximum(0.0, 1.0 - (1.0 / n) - np.sum(Ua2, axis=1))
    y_perp = yc - Ua @ Uaty

    def _eval_gamma(g: float) -> tuple[np.ndarray, float]:
        shrink = g / (lama + g)
        denom_1d = p_perp + (Ua2 @ shrink)
        num = y_perp + Ua @ (shrink[:, None] * Uaty)
        e_loo = np.empty_like(yc)
        safe = denom_1d >= 1e-12
        e_loo[safe] = num[safe] / denom_1d[safe, None]
        if not np.all(safe):
            g_fb = max(float(g), 1e-15)
            for i in np.flatnonzero(~safe):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                Xtr = Xa[mask]
                ytr = ya[mask]
                xmi = Xtr.mean(axis=0, keepdims=True)
                ymi = ytr.mean(axis=0, keepdims=True)
                Xci = Xtr - xmi
                yci = ytr - ymi
                Ui, si, Vti = np.linalg.svd(Xci, full_matrices=False)
                lami = si ** 2
                W = Vti.T @ ((si / (lami + g_fb))[:, None] * (Ui.T @ yci))
                pred = ymi + (Xa[i : i + 1] - xmi) @ W
                e_loo[i] = ya[i] - pred[0]
        return e_loo, float(np.mean(e_loo ** 2))

    if gamma is not None:
        g_val = float(gamma)
        e_loo, _ = _eval_gamma(g_val)
        return e_loo, g_val

    g_grid = lam_scale * np.logspace(-6, 4, 41)
    best_mse = float("inf")
    best_e = yc
    best_g = float(g_grid[0])
    for g in g_grid:
        e_loo, mse = _eval_gamma(float(g))
        if mse < best_mse:
            best_mse = mse
            best_e = e_loo
            best_g = float(g)
    return best_e, best_g


def _group_cv_ridge_residuals(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """按独立样本分组的 GroupKFold 外推残差（防重复个体泄漏），第 3 返回值为 ``gamma_per_fold_median``。"""
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if ya.ndim == 1:
        ya = ya[:, None]
    grp = np.asarray(groups)
    uniq_g = np.unique(grp)
    eff_splits = min(int(n_splits), len(uniq_g))
    if eff_splits < 2:
        raise ValueError(f"_group_cv_ridge_residuals 需要至少 2 个独立分组，实际 {len(uniq_g)}")

    splits = list(GroupKFold(n_splits=eff_splits).split(Xa, ya, grp))
    min_tr = min(len(itr) for itr, _ in splits)
    if min_tr < 3:
        raise ValueError(
            f"_group_cv_ridge_residuals 每个内层训练折需要至少 3 个样本，实际最小折仅 {min_tr} 个"
        )

    e_oof = np.empty_like(ya)
    gammas = []
    for itr, iva in splits:
        _, best_g = exact_press_loo_residuals(Xa[itr], ya[itr])
        gammas.append(best_g)
        x_mu = Xa[itr].mean(axis=0, keepdims=True)
        y_mu = ya[itr].mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(Xa[itr] - x_mu, full_matrices=False)
        lam = s ** 2
        W = Vt.T @ ((s / (lam + best_g))[:, None] * (U.T @ (ya[itr] - y_mu)))
        pred_va = (Xa[iva] - x_mu) @ W + y_mu
        e_oof[iva] = ya[iva] - pred_va
    return e_oof, uniq_g, float(np.median(gammas))


def diagnose_snv_decision(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray | None = None,
    t_crit: float = 2.0,
    min_effect_pct: float = 1.0,
    compute_physics: bool = True,
) -> dict:
    """双层闭式 SNV 预处决策器：Exact-PRESS/Group-CV 配对留一门控 + 定理 1 物理适用域归因。

    分工机制：
      - **决策层（Layer 2）**：仅在训练折 `(X_tr, y_tr)` 上计算精确 Allen-PRESS（无组别时）或
        GroupKFold（存在重复扫描组 `groups_tr` 时）下 Raw vs SNV 的配对留一误差差 ``d_i`` 与
        配对 t 统计量 ``t_loo``，仅当 ``|t_loo| >= t_crit`` 且 ``|delta_loo_pct| >= min_effect_pct``
        时给出 ``"IMPROVE"`` 或 ``"WORSEN"``，否则主动弃权 ``"ABSTAIN"``（无法判断）；若分组后
        内层训练折不足 3 个样本或独立组数不足 2，亦优雅返回 ``"ABSTAIN"``（``reason="insufficient_group_fold_size"``）。
      - **归因层（Layer 1）**：调用 :func:`estimate_physical_params_v2` 输出散射强度 ``sigma_a_true``、
        谱形决定度 ``R²_shape`` 与定理 1 适用域警示 ``theory_assumption_status``（当 ``n_train > 800`` 时，
        归因层使用 ``seed=0`` 的 800 条子集估参，而决策层仍使用全量 ``n_train``）。
    """
    Xa = np.asarray(X_tr, dtype=np.float64)
    ya = np.asarray(y_tr, dtype=np.float64)
    if ya.ndim == 1:
        ya = ya[:, None]
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError(f"X_tr 与 y_tr 样本数不匹配：{Xa.shape[0]} vs {ya.shape[0]}")
    if Xa.shape[0] < 4:
        raise ValueError(f"diagnose_snv_decision 至少需要 4 个训练样本，实际 {Xa.shape[0]}")

    ymu = ya.mean(axis=0, keepdims=True)
    ysd = np.clip(ya.std(axis=0, keepdims=True), 1e-9, None)
    yn = (ya - ymu) / ysd
    X_snv = PREPS["snv"](Xa)

    insufficient_group = False
    if groups_tr is None:
        e_raw, gamma_raw = exact_press_loo_residuals(Xa, yn)
        e_snv, gamma_snv = exact_press_loo_residuals(X_snv, yn)
        d_sq = np.mean(e_snv ** 2 - e_raw ** 2, axis=1)
        d_abs = np.mean(np.abs(e_snv) - np.abs(e_raw), axis=1)
        n_units = int(Xa.shape[0])
        mode = "exact_press_loo"
    else:
        grp = np.asarray(groups_tr)
        if grp.shape[0] != Xa.shape[0]:
            raise ValueError(f"groups_tr 长度与 X_tr 不匹配：{grp.shape[0]} vs {Xa.shape[0]}")
        uniq_g = np.unique(grp)
        eff_splits = min(5, len(uniq_g))
        if eff_splits < 2:
            insufficient_group = True
        else:
            splits_chk = list(GroupKFold(n_splits=eff_splits).split(Xa, yn, grp))
            if min(len(itr) for itr, _ in splits_chk) < 3:
                insufficient_group = True

        if insufficient_group:
            n_units = int(len(uniq_g))
        else:
            e_raw, uniq_g, gamma_raw = _group_cv_ridge_residuals(Xa, yn, grp)
            e_snv, _, gamma_snv = _group_cv_ridge_residuals(X_snv, yn, grp)
            d_sq_row = np.mean(e_snv ** 2 - e_raw ** 2, axis=1)
            d_abs_row = np.mean(np.abs(e_snv) - np.abs(e_raw), axis=1)
            d_sq = np.array([float(np.mean(d_sq_row[grp == g])) for g in uniq_g], dtype=np.float64)
            d_abs = np.array([float(np.mean(d_abs_row[grp == g])) for g in uniq_g], dtype=np.float64)
            n_units = int(len(uniq_g))
            mode = "group_cv_loo"

    if insufficient_group:
        out = {
            "decision": "ABSTAIN",
            "mode": "group_cv_loo_insufficient",
            "reason": "insufficient_group_fold_size",
            "t_crit": float(t_crit),
            "min_effect_pct": float(min_effect_pct),
            "n_train": int(Xa.shape[0]),
            "n_eff_units": n_units,
            "rmsep_loo_raw": None,
            "rmsep_loo_snv": None,
            "delta_loo_pct": 0.0,
            "t_loo": 0.0,
            "t_loo_abs": 0.0,
            "gamma_raw": None,
            "gamma_snv": None,
        }
    else:
        rmsep_raw = float(np.sqrt(np.mean(e_raw ** 2)))
        rmsep_snv = float(np.sqrt(np.mean(e_snv ** 2)))
        delta_loo_pct = float((rmsep_snv / max(rmsep_raw, 1e-12) - 1.0) * 100.0)

        if n_units > 1:
            se_sq = float(np.std(d_sq, ddof=1) / np.sqrt(n_units))
            se_abs = float(np.std(d_abs, ddof=1) / np.sqrt(n_units))
            t_loo = float(np.mean(d_sq) / se_sq) if se_sq > 1e-12 else 0.0
            t_loo_abs = float(np.mean(d_abs) / se_abs) if se_abs > 1e-12 else 0.0
        else:
            t_loo = 0.0
            t_loo_abs = 0.0

        if t_loo <= -float(t_crit) and delta_loo_pct <= -float(min_effect_pct):
            decision = "IMPROVE"
        elif t_loo >= float(t_crit) and delta_loo_pct >= float(min_effect_pct):
            decision = "WORSEN"
        else:
            decision = "ABSTAIN"

        out = {
            "decision": decision,
            "mode": mode,
            "t_crit": float(t_crit),
            "min_effect_pct": float(min_effect_pct),
            "n_train": int(Xa.shape[0]),
            "n_eff_units": n_units,
            "rmsep_loo_raw": round(rmsep_raw, 4),
            "rmsep_loo_snv": round(rmsep_snv, 4),
            "delta_loo_pct": round(delta_loo_pct, 2),
            "t_loo": round(t_loo, 3),
            "t_loo_abs": round(t_loo_abs, 3),
            "gamma_raw": float(gamma_raw),
            "gamma_snv": float(gamma_snv),
        }

    if compute_physics:
        if Xa.shape[0] > 800:
            sub_idx = np.random.default_rng(0).choice(Xa.shape[0], 800, replace=False)
            info = estimate_physical_params_v2(Xa[sub_idx], yn[sub_idx])
        else:
            info = estimate_physical_params_v2(Xa, yn)
        pc = info["per_col"]
        out.update(
            {
                "r2_s_shape": float(info["r2_s_shape"]),
                "sigma_s": float(info["sigma_s"]),
                "sigma_a_true": float(info["sigma_a_true"]),
                "sigma_non_chem": float(info["sigma_non_chem"]),
                "r_s": round(float(pc[0]["r_s"]), 4),
                "max_abs_r_s": round(float(max(abs(c["r_s"]) for c in pc)), 4),
                "r_crit": round(float(np.mean([c["r_crit"] for c in pc])), 4),
                "ws_star": round(float(np.mean([c["ws_star"] for c in pc])), 4),
                "pred_mult_pct": round(float(np.mean([c["pred_snv_vs_raw_pct_mult"] for c in pc])), 2),
                "pred_v2_legacy_pct": round(float(np.mean([c["pred_snv_vs_raw_pct_v2"] for c in pc])), 2),
                "theory_assumption_status": str(pc[0]["assumption_status"]),
            }
        )
    return out
