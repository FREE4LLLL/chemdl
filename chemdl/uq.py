"""ChemDL 2.0 不确定性量化（Conformal UQ）与批次级漂移监控器。

1. AdaptiveConformalCalibrator：
   基于难度加权的分裂保形预测（Split / Locally-Weighted Conformal Prediction），
   融合 `SoftOutlierGate` 的质量分数 w 与多视图/集成模型分歧度 `disagreement`，
   使 OOD/高噪声样本自动获得更宽的安全区间：
     - 当难度缩放器 ``sigma(x)`` 独立于保形校准集拟合（如设置 ``scale_split > 0`` 将校准集
       一分为二，或使用固定常数缩放）且校准/测试样本可交换时，满足严格有限样本分布无关覆盖率保证
       ``P(y ∈ [y_lo, y_hi]) >= 1 - α``（含小样本边界 ``ceil((n+1)(1-α)) > n`` 时输出 ``+∞``）；
     - 当 ``scale_split == 0.0``（默认，便于小校准集保留全部样本数）在同一校准集上估计
       ``base_scale_ / dis_scale_ / pca`` 时，提供经验/渐近自适应覆盖率近似而非严格有限样本保证。

2. BatchDriftMonitor：
   针对样本级分歧无法识别共享仪器偏移的问题（RESEARCH_GAPS §3.15），
   在整批（Batch / Instrument）层面融合 PCA 子空间马氏距离、能量距离、
   门控退化度与批次平均分歧，输出 OK / WARN_ADAPT / RECALIBRATE 三级触发决策。
"""
from __future__ import annotations

import math

import numpy as np


class AdaptiveConformalCalibrator:
    """自适应局部加权分裂保形校准器（支持多目标回归与协变量偏移距离缩放）。"""

    def __init__(self, alpha: float = 0.10, c_gate: float = 2.5,
                 c_disagreement: float = 1.2, c_md: float = 1.0, min_scale: float = 1e-3):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha 必须在 (0, 1) 内，实际 {alpha}")
        self.alpha = float(alpha)
        self.c_gate = float(c_gate)
        self.c_disagreement = float(c_disagreement)
        self.c_md = float(c_md)
        self.min_scale = float(min_scale)
        self.quantiles_: np.ndarray | None = None
        self.base_scale_: np.ndarray | None = None
        self.dis_scale_: np.ndarray | None = None
        self.pca_mu_: np.ndarray | None = None
        self.pca_proj_: np.ndarray | None = None
        self.pca_sd_: np.ndarray | None = None
        self.md_med_: float = 1.0
        self.d2_med_: float = 1.0

    @staticmethod
    def _to_snv(X: np.ndarray) -> np.ndarray:
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)

    def _fit_spectral_dist(self, X_cal: np.ndarray):
        X = self._to_snv(np.asarray(X_cal, dtype=np.float64))
        self.pca_mu_ = X.mean(0, keepdims=True)
        Xc = X - self.pca_mu_
        # 单样本校准（codex 审计 07）：早先 len(X)-1 == 0 会让 PCA 秩为 0，
        # scores 全零 -> sd 被夹到 1e-4 -> 距离比炸成 +inf -> 分位数为 inf，
        # 最终区间上下界都是 NaN。现在秩为 0 时明确停用谱距离缩放。
        n_eff, L_eff = len(X), X.shape[1]
        r_max = min(12, n_eff - 1, L_eff)
        if n_eff < 2 or r_max < 1:
            self.pca_proj_ = None
            self.pca_sd_ = None
            self.md_med_ = 1.0
            self.d2_med_ = 1.0
            return
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        r = int(min(r_max, max(1, (S > 1e-8).sum())))
        self.pca_proj_ = Vt[:r].T
        scores = Xc @ self.pca_proj_
        sd = scores.std(0, keepdims=True)
        # 全部退化为中心点时同样放弃缩放，避免 1e-4 夹断把比值放大到 inf
        if not np.all(np.isfinite(sd)) or float(sd.max()) < 1e-8:
            self.pca_proj_ = None
            self.pca_sd_ = None
            self.md_med_ = 1.0
            self.d2_med_ = 1.0
            return
        self.pca_sd_ = np.clip(sd, 1e-4, None)
        md = np.sqrt(np.mean((scores / self.pca_sd_) ** 2, axis=1))
        self.md_med_ = max(float(np.median(md)), 1e-4)
        if X.shape[1] >= 4:
            d2 = np.abs(X[:, 2:] - 2.0 * X[:, 1:-1] + X[:, :-2])
            d2_mad = np.median(d2, axis=1)
            self.d2_med_ = max(float(np.median(d2_mad)), 1e-5)

    def _spectral_dist_ratio(self, X_in: np.ndarray) -> np.ndarray:
        X = self._to_snv(np.asarray(X_in, dtype=np.float64))
        # 校准集样本不足（或 PCA 退化）时不做谱距离缩放，返回全 1。
        # 这使区间退回「仅由残差分位数决定」，仍是有限闭区间而非 NaN
        # （codex 审计 07）。
        if self.pca_proj_ is None or self.pca_sd_ is None or self.pca_mu_ is None:
            return np.ones(len(X), dtype=np.float64)
        z = ((X - self.pca_mu_) @ self.pca_proj_) / self.pca_sd_
        md_ratio = np.sqrt(np.mean(z ** 2, axis=1)) / self.md_med_
        if X.shape[1] >= 4:
            d2 = np.abs(X[:, 2:] - 2.0 * X[:, 1:-1] + X[:, :-2])
            d2_ratio = np.median(d2, axis=1) / self.d2_med_
            return np.maximum(md_ratio, d2_ratio)
        return md_ratio

    def _difficulty(self, pred: np.ndarray, w_gate: np.ndarray | None,
                    disagreement: np.ndarray | None, X_in: np.ndarray | None = None,
                    fit_mode: bool = False) -> np.ndarray:
        n, m = pred.shape
        base = np.ones((1, m), dtype=np.float64) if self.base_scale_ is None else self.base_scale_[None, :]
        scale = np.repeat(base, n, axis=0)
        if w_gate is not None:
            w = np.asarray(w_gate, dtype=np.float64).reshape(n, 1)
            pen_w = np.where(np.isfinite(w), np.clip(1.0 - w, 0.0, 1.0), 1.0)
            scale = scale * (1.0 + self.c_gate * pen_w)
        if disagreement is not None:
            dis = np.asarray(disagreement, dtype=np.float64).reshape(n, m)
            if fit_mode:
                med = np.empty(m, dtype=np.float64)
                for j in range(m):
                    fin = dis[np.isfinite(dis[:, j]), j]
                    med[j] = float(np.median(fin)) if fin.size > 0 else 1.0
                self.dis_scale_ = np.clip(med, 1e-6, None)
            elif self.dis_scale_ is None:
                # 推理期绝不能用测试批次的中位数初始化/覆写：那是测试集泄漏，
                # 且会让校准期与测试期的 sigma(x) 不再是同一个函数，破坏保形覆盖保证。
                raise ValueError(
                    "推理期首次遇到 disagreement，但 fit() 时未提供 disagreement，"
                    "dis_scale_ 未标定。请在 fit() 时传入 disagreement_cal，"
                    "或推理时不要传 disagreement。")
            dis_norm = dis / self.dis_scale_[None, :]
            pen_dis = np.where(np.isfinite(dis_norm), np.clip(dis_norm - 1.0, 0.0, 5.0), 5.0)
            scale = scale * (1.0 + self.c_disagreement * pen_dis)
        if (X_in is not None) and (self.pca_proj_ is not None):
            r = self._spectral_dist_ratio(X_in).reshape(n, 1)
            scale = scale * (1.0 + self.c_md * np.log1p(np.clip(r - 1.0, 0.0, 12.0)))
        return np.clip(scale, self.min_scale, None)

    def fit(self, y_cal, pred_cal, w_gate=None, disagreement=None, X_cal=None,
            scale_split: float = 0.0, seed: int = 0):
        """在校准集（验证折/独立标定集）上拟合保形分位数。

        参数 ``scale_split ∈ [0, 0.5]``：
          - ``scale_split == 0.0``（默认）：在全校准集上同时拟合难度缩放与分位数（小样本高效近似）；
          - ``scale_split > 0.0``：先切出 ``scale_split`` 比例样本拟合难度缩放器，再在完全独立的
            剩余校准子集上计算精确保形秩统计量 ``ceil((n_q + 1)(1 - α))``，满足严格有限样本覆盖保证。
        当 ``ceil((n_q + 1)(1 - α)) > n_q`` 时按分裂保形理论返回 ``+∞``（不截断为最大残差）。
        """
        y = np.asarray(y_cal, dtype=np.float64)
        p = np.asarray(pred_cal, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if p.ndim == 1:
            p = p[:, None]
        n = len(y)
        if n == 0:
            raise ValueError("校准集不能为空")
        # 重复 fit 必须清空上一次的状态：否则第一次带 X_cal/disagreement 拟合后，
        # 第二次不带它们重新 fit，旧的 PCA 投影与分歧尺度会继续残留并生效。
        self.dis_scale_ = None
        self.pca_proj_ = None
        self.pca_mu_ = None
        if y.shape != p.shape:
            raise ValueError(
                "y_cal 与 pred_cal 形状不一致：%s vs %s（拒绝静默广播）"
                % (y.shape, p.shape))
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
            raise ValueError("y_cal 与 pred_cal 必须全部有限")

        w_arr = None if w_gate is None else np.asarray(w_gate, dtype=np.float64)
        d_arr = None if disagreement is None else np.asarray(disagreement, dtype=np.float64)
        X_arr = None if X_cal is None else np.asarray(X_cal, dtype=np.float64)

        # codex full-audit E19: n<4 时早先静默退回「同一批样本既拟合缩放又
        # 估计分位数」，与文档声明的独立分割承诺不符。现在明确拒绝。
        if 0.0 < float(scale_split) < 1.0 and n < 4:
            raise ValueError(
                "scale_split>0 需要至少 4 个校准样本才能切出互不相交的缩放集与"
                "分位集；当前 n=%d。请改用 scale_split=0.0，或提供更多校准谱。" % n)
        if 0.0 < float(scale_split) < 1.0 and n >= 4:
            rng = np.random.default_rng(int(seed))
            perm = rng.permutation(n)
            n_sc = max(2, min(n - 2, int(round(n * float(scale_split)))))
            idx_sc, idx_q = perm[:n_sc], perm[n_sc:]
        else:
            idx_sc = np.arange(n)
            idx_q = np.arange(n)

        if X_arr is not None:
            self._fit_spectral_dist(X_arr[idx_sc])
        res_sc = np.abs(y[idx_sc] - p[idx_sc])
        self.base_scale_ = np.clip(np.median(res_sc, axis=0), self.min_scale, None)
        # 在难度拟合子集上初始化 dis_scale_
        _ = self._difficulty(
            p[idx_sc],
            None if w_arr is None else w_arr[idx_sc],
            None if d_arr is None else d_arr[idx_sc],
            X_in=None if X_arr is None else X_arr[idx_sc],
            fit_mode=True,
        )

        # 在校准分位数子集上严格冻结缩放参数并计算精确保形阶统计量
        res_q = np.abs(y[idx_q] - p[idx_q])
        sigma_q = self._difficulty(
            p[idx_q],
            None if w_arr is None else w_arr[idx_q],
            None if d_arr is None else d_arr[idx_q],
            X_in=None if X_arr is None else X_arr[idx_q],
            fit_mode=False,
        )
        scores = res_q / sigma_q
        n_q = len(idx_q)
        k_rank = int(math.ceil((n_q + 1) * (1.0 - self.alpha)))
        if k_rank > n_q:
            self.quantiles_ = np.full(scores.shape[1], np.inf, dtype=np.float64)
        elif k_rank <= 1:
            self.quantiles_ = np.min(scores, axis=0)
        else:
            self.quantiles_ = np.sort(scores, axis=0)[k_rank - 1]
        return self

    def predict_interval(self, pred, w_gate=None, disagreement=None, X_in=None):
        """返回 (y_lo, y_hi, half_width)。"""
        if self.quantiles_ is None:
            raise RuntimeError("请先调用 fit() 完成保形校准")
        p = np.asarray(pred, dtype=np.float64)
        was_1d = p.ndim == 1
        if was_1d:
            p = p[:, None]
        # codex full-audit E25: 一维预测在校准了两列时早先只返回第一列区间，
        # 第二列被静默丢弃；列数不一致也可能广播出看似完整的区间。
        _n_q = int(np.asarray(self.quantiles_).size)
        if p.shape[1] != _n_q:
            raise ValueError(
                "predict_interval: 预测列数 %d 与校准目标数 %d 不一致"
                "（拒绝静默丢弃或广播）" % (p.shape[1], _n_q))
        sigma = self._difficulty(p, w_gate, disagreement, X_in=X_in, fit_mode=False)
        if np.asarray(sigma).shape[1] != _n_q:
            raise ValueError(
                "predict_interval: 难度缩放列数 %d 与校准目标数 %d 不一致"
                % (np.asarray(sigma).shape[1], _n_q))
        half = self.quantiles_[None, :] * sigma
        lo, hi = p - half, p + half
        if was_1d:
            return lo[:, 0], hi[:, 0], half[:, 0]
        return lo, hi, half

    def evaluate(self, y_true, pred, w_gate=None, disagreement=None, X_in=None) -> dict:
        # codex full-audit E25: 早先对 y_true / pred 各自 reshape，形状不一致时
        # numpy 会广播出「看起来完整」的覆盖率。现在要求二者形状严格一致。
        _yt = np.asarray(y_true, dtype=np.float64)
        _pd = np.asarray(pred, dtype=np.float64)
        if _yt.shape != _pd.shape:
            raise ValueError(
                "evaluate 要求 y_true 与 pred 形状一致（拒绝广播）：%s vs %s"
                % (_yt.shape, _pd.shape))
        y = _yt
        if y.ndim == 1:
            y = y[:, None]
        lo, hi, half = self.predict_interval(pred, w_gate=w_gate, disagreement=disagreement, X_in=X_in)
        if lo.ndim == 1:
            lo, hi, half = lo[:, None], hi[:, None], half[:, None]
        covered = (y >= lo) & (y <= hi)
        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(covered.mean()),
            "per_col_coverage": [float(c) for c in covered.mean(0)],
            "mean_half_width": float(half.mean()),
            "per_col_half_width": [float(w) for w in half.mean(0)],
        }


class BatchDriftMonitor:
    """批次级无监督分布漂移监控与重标定触发器。"""

    def __init__(self, n_components: int = 8, warn_ratio: float = 1.45, recal_ratio: float = 2.4):
        self.n_components = int(n_components)
        self.warn_ratio = float(warn_ratio)
        self.recal_ratio = float(recal_ratio)
        self.mu_: np.ndarray | None = None
        self.proj_: np.ndarray | None = None
        self.pc_sd_: np.ndarray | None = None
        self.cal_scores_: np.ndarray | None = None
        self.cal_md_q75_: float = 1.0
        self.cal_raw_rms_: float = 1.0
        self.cal_energy_: float = 1.0
        self.cal_gate_bad_: float = 0.1
        self.cal_dis_med_: float = 1.0

    @staticmethod
    def _pair_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        a2 = np.sum(A ** 2, axis=1, keepdims=True)
        b2 = np.sum(B ** 2, axis=1, keepdims=True).T
        return np.sqrt(np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0))

    @classmethod
    def _mean_dist_unbiased(cls, A: np.ndarray, B: np.ndarray | None = None) -> float:
        """无偏 U-统计量的平均距离（同集时排除 i==j 对角零）。

        直接对含对角零的距离矩阵取 mean 是有偏 V-统计量：
        E[d] = (m-1)/m * E||X-X'||。m=1 时 d 恒为 0，会让能量距离退化成
        「2*d_xy - cal_energy ≈ cal_energy」，ratio ≈ 1.0，越过 warn_ratio 误报。
        """
        if B is None:
            m = len(A)
            if m < 2:
                return float("nan")          # 单样本无定义，由调用方跳过该项
            D = cls._pair_dist(A, A)
            return float(D.sum() / (m * (m - 1)))
        if len(B) < 2:
            return float("nan")
        return float(cls._pair_dist(A, B).mean())

    def fit(self, X_cal, w_cal=None, disagreement_cal=None):
        X = np.asarray(X_cal, dtype=np.float64)
        self.cal_raw_rms_ = max(float(np.sqrt(np.mean(X ** 2))), 1e-6)
        # 在逐样本 SNV 空间计算形态与位置偏移
        X_snv = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
        self.mu_ = X_snv.mean(0, keepdims=True)
        Xc = X_snv - self.mu_
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        r = int(min(self.n_components, len(X) - 1, X.shape[1], max(1, (S > 1e-8).sum())))
        self.proj_ = Vt[:r].T
        scores = Xc @ self.proj_
        # Q 残差 / SPE 基线：波长漂移、新干扰组分、检测器坏点首先出现在与主成分正交的
        # 残差子空间，只看得分空间会漏报大量真实光学漂移。
        _resid = Xc - scores @ self.proj_.T
        self.cal_spe_q95_ = max(float(np.quantile(np.sum(_resid ** 2, axis=1), 0.95)), 1e-8)
        self.pc_sd_ = np.clip(scores.std(0, keepdims=True), 1e-4, None)
        z_scores = scores / self.pc_sd_
        self.cal_scores_ = z_scores
        md = np.sqrt(np.mean(z_scores ** 2, axis=1))
        self.cal_md_q75_ = max(float(np.quantile(md, 0.75)), 1e-4)
        _ce = self._mean_dist_unbiased(z_scores)
        self.cal_energy_ = max(float(_ce) if np.isfinite(_ce) else 1e-4, 1e-4)
        if w_cal is not None:
            w_arr = np.asarray(w_cal, dtype=np.float64)
            w_fin = w_arr[np.isfinite(w_arr)]
            if w_fin.size > 0:
                self.cal_gate_bad_ = max(float(np.mean(1.0 - w_fin)), 1e-3)
        if disagreement_cal is not None:
            d_arr = np.asarray(disagreement_cal, dtype=np.float64)
            d_fin = d_arr[np.isfinite(d_arr)]
            if d_fin.size > 0:
                self.cal_dis_med_ = max(float(np.mean(d_fin)), 1e-6)
        return self

    def assess_batch(self, X_batch, w_batch=None, disagreement_batch=None) -> dict:
        if self.proj_ is None:
            raise RuntimeError("请先调用 fit() 建立基准参考分布")
        X = np.asarray(X_batch, dtype=np.float64)
        X_snv = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
        _Xc_b = X_snv - self.mu_
        _sc_b = _Xc_b @ self.proj_
        z_b = _sc_b / self.pc_sd_
        # SPE 比：正交残差空间的能量相对基线 95 分位
        _spe = np.sum((_Xc_b - _sc_b @ self.proj_.T) ** 2, axis=1)
        spe_ratio = float(np.quantile(_spe, 0.75) / self.cal_spe_q95_)
        md_b = np.sqrt(np.mean(z_b ** 2, axis=1))
        md_ratio = float(np.quantile(md_b, 0.75) / self.cal_md_q75_)
        # 原始幅值比：SNV 会抹掉加性基线与乘性光源衰减，必须在原始空间另设一项，
        # 否则整批标度漂移（最常见的硬件故障）完全失明。
        raw_rms_b = max(float(np.sqrt(np.mean(X ** 2))), 1e-6)
        raw_ratio = float(max(raw_rms_b / self.cal_raw_rms_, self.cal_raw_rms_ / raw_rms_b))

        d_xy = self._mean_dist_unbiased(z_b, self.cal_scores_)
        d_xx = self._mean_dist_unbiased(z_b)          # 单样本时返回 nan
        if not np.isfinite(d_xx):
            # 单样本批次：能量距离无定义，只用样本级指标（md / gate / disagreement）
            energy, energy_ratio = 0.0, 0.0
        else:
            energy = max(2.0 * d_xy - d_xx - self.cal_energy_, 0.0)
            energy_ratio = float(energy / self.cal_energy_)

        gate_ratio = 1.0
        if w_batch is not None:
            w_b = np.asarray(w_batch, dtype=np.float64)
            w_fin = w_b[np.isfinite(w_b)]
            gate_ratio = (float(np.mean(1.0 - w_fin)) / self.cal_gate_bad_) if w_fin.size > 0 else float("nan")

        dis_ratio = 1.0
        if disagreement_batch is not None:
            d_b = np.asarray(disagreement_batch, dtype=np.float64)
            d_fin = d_b[np.isfinite(d_b)]
            dis_ratio = (float(np.mean(d_fin)) / self.cal_dis_med_) if d_fin.size > 0 else float("nan")

        # codex full-audit E13: 零秩 PCA 会让 md_ratio 变 NaN，而 Python 的
        # max(nan, x) 返回 nan —— 后续所有阈值比较都是 False，最终落到
        # else 分支报 status="OK"。先在候选集合里剔除无效项。
        _cands = [v for v in (md_ratio, 1.0 + energy_ratio, gate_ratio,
                              dis_ratio, raw_ratio, spe_ratio)
                  if np.isfinite(v)]
        if not _cands:
            drift_score = float("nan")
        else:
            drift_score = max(_cands)

        if not np.isfinite(drift_score):
            # 所有分量都无定义时明确报「校准不足」，不伪装成稳定
            status = "UNDEFINED"
            action = "校准样本不足或谱方差退化，漂移指标无法定义；请增加校准谱后重标定"
        elif drift_score >= self.recal_ratio:
            status = "RECALIBRATE"
            action = "检测到严重分布/跨仪器偏移，建议使用 Kennard-Stone 精选 10~20 条配对谱重标定"
        elif drift_score >= self.warn_ratio:
            status = "WARN_ADAPT"
            action = "检测到中度漂移，建议执行轻量仿射适应并加宽保形区间"
        else:
            status = "OK"
            action = "分布稳定，模型正常服役"

        return {
            "status": status,
            "action": action,
            "drift_score": round(drift_score, 4),
            "mahalanobis_ratio": round(md_ratio, 4),
            "energy_ratio": round(energy_ratio, 4),
            "raw_ratio": round(raw_ratio, 4),
            "spe_ratio": round(spe_ratio, 4),
            "gate_ratio": round(gate_ratio, 4),
            "disagreement_ratio": round(dis_ratio, 4),
        }
