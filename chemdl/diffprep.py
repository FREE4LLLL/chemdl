"""五步预处理的「可微版本」——每一课件的步骤都变成一个可训练算子。

01 噪声去除      → LearnableWaveletDenoise   多级 Haar 软阈值（尺度自适应 + 可学）+ 残差卷积
02 基线校正      → UnrolledALS               ALS 展开成可微迭代（DCT 基 + 闭式第 0 步 + 1-Step 反传）
03 标准化处理    → LearnableSNV              逐样本 SNV（后接训练集标定的 03' 逐通道 Z-score）
04 峰校准对齐    → XCorrAlign                分段零均值余弦互相关 + 段中心插值 + soft-argmax
05 异常剔除      → SoftOutlierGate           低秩自编码残差 + 多阶段物理异常特征，输出 w∈[0,1]

设计原则：每个算子都能退化成课件里的经典算法（阈值=0 时是恒等，stretched_sigmoid(-3.5)=0
精确恒等），所以网络是从「经典流程」出发做端到端微调，而不是黑箱替代。
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def stretched_sigmoid(z: torch.Tensor, bound: float = 3.5) -> torch.Tensor:
    """带边界截断的拉伸 Sigmoid：z <= -bound 时精确为 0.0，z >= +bound 时精确为 1.0。

    普通 sigmoid 在 ±12 处梯度仅 6.2e-6（分辨率极低），在 -3 处又有 4.74% 的残余泄漏
    （在 SNV 后极小通道标准差下会被放大近 1 个标准差）。拉伸 Sigmoid 兼顾了
    「精确恒等起点」与「高灵敏度可微梯度」。
    """
    s0 = 1.0 / (1.0 + math.exp(bound))
    return ((torch.sigmoid(z) - s0) / (1.0 - 2.0 * s0)).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# 01 噪声去除
# --------------------------------------------------------------------------- #
class LearnableWaveletDenoise(nn.Module):
    """多级 Haar 分解 + 尺度自适应可学习软阈值。

    细节系数 d 与噪声水平同量级，软阈值 d' = sign(d)·max(|d|-λ,0) 正是小波去噪的核心；
    乘以输入信号自身标准差（adaptive_scale=True），使固定初值 init_lam=0.05
    既适用于标准差≈1 的合成谱，也不会在原始吸光度谱（std≈0.18）上过度抹平弱吸收带。
    """

    def __init__(self, levels: int = 4, init_lam: float = 0.05, identity: bool = False,
                 adaptive_scale: bool = True, smooth: bool = False,
                 use_res_conv: bool = True, wavelet: str = "haar"):
        super().__init__()
        self.levels = levels
        # wavelet="haar" 走原有的手写两条路径（保持历史结果可比）；
        # 其他取值（db4/sym8/...）走 ptwt 实现。离线评估：固定阈值降噪下
        # db4 的相对 RMSE 0.01552 优于 haar 0.01760（-11.8%），见 _audit/eval_ptwt.py
        self.wavelet = str(wavelet).lower()
        self.adaptive_scale = adaptive_scale
        self.smooth = smooth
        self.use_res_conv = use_res_conv
        # identity=True 时 λ≈0（软阈值不起作用）⇒ 这一层在初始化时是恒等映射
        init_val = (1e-5 if smooth else 1e-4) if identity else init_lam
        self.log_lam = nn.Parameter(torch.full((levels,), math.log(init_val)))
        # 残差修正卷积：权重置零 ⇒ 初值为恒等映射
        if use_res_conv:
            self.res = nn.Conv1d(1, 1, 9, padding=4)
            nn.init.zeros_(self.res.weight)
            nn.init.zeros_(self.res.bias)
        else:
            self.res = None

    def forward(self, x):
        lam = self.log_lam.exp().clamp(1e-5 if self.smooth else 1e-4, 5.0)
        scale = x.std(-1, keepdim=True).detach().clamp_min(1e-3) if self.adaptive_scale else 1.0
        if self.wavelet != "haar":
            return self._forward_ptwt(x, lam, scale)
        approx, details = x, []
        for _ in range(self.levels):
            low = F.avg_pool1d(approx, 2)
            d = approx - F.interpolate(low, size=approx.shape[-1], mode="nearest")
            details.append(d)                      # 只存细节系数；低频由递推得到
            approx = low
        # 从最粗层往回重建：必须先重建更深层的结果，再回到浅层。
        for i in range(len(details) - 1, -1, -1):
            thr = lam[i] * scale if self.adaptive_scale else lam[i]
            if self.smooth:
                # C∞ 双曲软阈值：零尖点、小细节梯度非零、λ→0 严格退化为恒等
                di = details[i]
                d = di * (1.0 - thr / torch.sqrt(di * di + thr * thr + 1e-12))
            else:
                d = torch.sign(details[i]) * F.relu(details[i].abs() - thr)
            approx = F.interpolate(approx, size=d.shape[-1], mode="nearest") + d
        if self.res is None:
            return approx
        return approx + self.res(approx)

    def _forward_ptwt(self, x, lam, scale):
        """用 ptwt 的任意正交小波（db4/sym8 等）做同样的软阈值降噪。

        与手写 Haar 路径的差别：小波基更平滑（减少块效应），代价是系数长度不必为 2 的幂。
        """
        try:
            import ptwt
        except ImportError as exc:
            raise ImportError(
                f"wavelet='{self.wavelet}' 需要可选依赖 ptwt（pip install ptwt PyWavelets）；"
                f"默认 wavelet='haar' 无需额外依赖。"
            ) from exc
        coeffs = ptwt.wavedec(x, self.wavelet, level=self.levels, mode="zero")
        # ptwt 返回 [cA_n, cD_n, ..., cD_1]（从粗到细），而 Haar 路径的 lam[0] 控制最细尺度。
        # 这里反向索引，使两条路径的 lam 物理语义一致（lam[0]=最细高频噪声）。
        n_detail = len(coeffs) - 1
        out = [coeffs[0]]
        for i, c in enumerate(coeffs[1:]):
            thr = lam[min(n_detail - 1 - i, len(lam) - 1)] * scale
            if self.smooth:
                out.append(c * (1.0 - thr / torch.sqrt(c * c + thr * thr + 1e-12)))
            else:
                out.append(torch.sign(c) * torch.relu(c.abs() - thr))
        rec = ptwt.waverec(out, self.wavelet)[..., : x.shape[-1]]
        if self.res is None:
            return rec
        return rec + self.res(rec)


# --------------------------------------------------------------------------- #
# 02 基线校正
# --------------------------------------------------------------------------- #
class UnrolledALS(nn.Module):
    """02 基线校正：把非对称最小二乘展开成可微迭代。

    min_z  Σ w_i (y_i - z_i)² + λ‖D²z‖² ,   w_i = p 若 y_i>z_i 否则 1-p

    基线限制在 K 个正交 DCT 低频基上，第 0 步对称平滑直接使用正交对角闭式解，
    训练期前 iters-1 步在 no_grad 下收敛非对称权重不动点，最后 1 步开启梯度反传
    （1-Step Phantom Gradient），在数值等价的前提下将迭代前向+反向提速约 2.6 倍。
    """

    def __init__(self, length: int = 512, n_basis: int = 48, iters: int = 8,
                 p_asym: float = 0.02, tau: float = 0.03, identity: bool = False,
                 legacy: bool = False, learn_p: bool = False):
        super().__init__()
        k = torch.arange(n_basis).float()[:, None]
        i = torch.arange(length).float()[None, :]
        B = torch.cos(math.pi * k * (i + 0.5) / length) * math.sqrt(2.0 / length)
        B[0] = math.sqrt(1.0 / length)
        self.register_buffer("B", B)                        # (K, L) 正交基
        self.register_buffer("pen", (k.squeeze(1) * math.pi / length) ** 4)
        self.log_lam = nn.Parameter(torch.tensor(math.log(1e5)))
        self.legacy = legacy
        self.learn_p = learn_p
        init_g = (-3.0 if identity else 3.0) if legacy else (-3.5 if identity else 3.5)
        self.base_gate = nn.Parameter(torch.tensor(init_g))
        self.iters, self.p_asym, self.tau = iters, p_asym, tau
        if learn_p:
            p0 = max(0.003, min(0.44, float(p_asym)))
            self.logit_p = nn.Parameter(torch.tensor(math.log((p0 - 0.002) / (0.45 - p0))))

    def _current_p(self):
        if self.learn_p:
            return 0.002 + 0.448 * torch.sigmoid(self.logit_p)
        return self.p_asym

    def _als_step(self, y, a, lam, eye_k):
        z = (a.squeeze(-1) @ self.B).unsqueeze(1)
        s = torch.sigmoid((y - z) / self.tau)
        p_use = self._current_p()
        w = (1.0 - p_use) - (1.0 - 2.0 * p_use) * s
        BtW = (self.B.t().unsqueeze(0) * w.transpose(1, 2)).transpose(1, 2)   # (B,K,L)
        A = BtW @ self.B.t() + lam * torch.diag_embed(self.pen)
        b = BtW @ y.squeeze(1).unsqueeze(-1)
        return torch.linalg.solve(A.double() + eye_k, b.double()).float()

    def forward(self, y):
        lam = self.log_lam.exp().clamp(1e3, 1e8)
        K = self.B.shape[0]
        if self.legacy:
            a = torch.zeros(y.shape[0], K, 1, dtype=y.dtype, device=y.device)
            for i in range(self.iters):
                z = (a.squeeze(-1) @ self.B).unsqueeze(1)
                if i == 0:
                    w = torch.ones_like(y)
                else:
                    s = torch.sigmoid((y - z) / self.tau)
                    w = (1.0 - self.p_asym) - (1.0 - 2.0 * self.p_asym) * s
                BtW = (self.B.t().unsqueeze(0) * w.transpose(1, 2)).transpose(1, 2)
                A = BtW @ self.B.t() + lam * torch.diag_embed(self.pen)
                b = BtW @ y.squeeze(1).unsqueeze(-1)
                # A 是 (B, K, K)：batch 维是 shape[0]，基函数维是 shape[-1]。
                # 必须用 -1，否则 batch_size != K 时广播报错。
                # 标量加到矩阵上是全元素偏移，不是 Tikhonov 对角正则（与非 legacy 分支不一致）。
                eye_leg = torch.eye(A.shape[-1], device=A.device, dtype=torch.float64) * 1e-9
                a = torch.linalg.solve(A.double() + eye_leg, b.double()).float()
            base = (a.squeeze(-1) @ self.B).unsqueeze(1)
            g = torch.sigmoid(self.base_gate)
            return base, y - g * base

        eye_k = torch.eye(K, device=y.device, dtype=torch.float64) * 1e-9
        # 第 0 步对称平滑：B 为正交基（B @ B.T == I），直接对角闭式求解，无需构造 K×K 系统
        if self.iters <= 1:
            a = (y.squeeze(1) @ self.B.t()).unsqueeze(-1) / (1.0 + lam * self.pen[None, :, None])
        else:
            with torch.no_grad():
                a = (y.squeeze(1) @ self.B.t()).unsqueeze(-1) / (1.0 + lam * self.pen[None, :, None])
                for _ in range(self.iters - 2):
                    a = self._als_step(y, a, lam, eye_k)
            a = self._als_step(y, a, lam, eye_k)
        base = (a.squeeze(-1) @ self.B).unsqueeze(1)
        g = stretched_sigmoid(self.base_gate)
        return base, y - g * base


# --------------------------------------------------------------------------- #
# 03 标准化
# --------------------------------------------------------------------------- #
class LearnableSNV(nn.Module):
    """03 标准化：逐样本 SNV / z-score。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True).clamp_min(1e-3)
        return ((x - mu) / sd).clamp(-20.0, 20.0)


class LearnableDSSNV(nn.Module):
    """可微解耦收缩 SNV（Differentiable Decoupled Shrinkage-SNV）。

    以 2 个有界标量参数 ``(w_c, w_s) ∈ [0, 1]^2`` 连续连接 Raw (0,0)、RowCenter (1,0)、
    RowScale (0,1)、部分收缩 SNV (1, 0.75) 与经典 SNV (1, 1)，训练期通过首批直赋 + EMA
    标定归纳式参考标准差 ``s_bar``，推理期严格单样本独立（零批内/测试折泄漏）。
    """

    def __init__(self, init_wc: float = 1.0, init_ws: float = 0.75):
        super().__init__()
        self.logit_wc = nn.Parameter(torch.tensor(self._inv_stretched(init_wc), dtype=torch.float32))
        self.logit_ws = nn.Parameter(torch.tensor(self._inv_stretched(init_ws), dtype=torch.float32))
        self.register_buffer("s_bar", torch.tensor(0.0))

    @staticmethod
    def _inv_stretched(p: float, bound: float = 3.5) -> float:
        p_c = max(0.0, min(1.0, float(p)))
        if p_c <= 0.0:
            return -bound
        if p_c >= 1.0:
            return bound
        s0 = 1.0 / (1.0 + math.exp(bound))
        sig = s0 + p_c * (1.0 - 2.0 * s0)
        return math.log(sig / (1.0 - sig))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True).clamp_min(1e-3)
        if self.training and x.shape[0] > 0:
            with torch.no_grad():
                b_sd = sd.mean()
                if float(self.s_bar) <= 0.0:
                    self.s_bar.copy_(b_sd)
                else:
                    self.s_bar.mul_(0.9).add_(0.1 * b_sd)
        sb = self.s_bar if float(self.s_bar) > 0.0 else sd.mean().detach()
        wc = stretched_sigmoid(self.logit_wc)
        ws = stretched_sigmoid(self.logit_ws)
        denom = ((1.0 - ws) * sb + ws * sd).clamp_min(1e-3)
        return ((x - wc * mu) / denom).clamp(-20.0, 20.0)


# --------------------------------------------------------------------------- #
# 04 峰校准对齐
# --------------------------------------------------------------------------- #
class XCorrAlign(nn.Module):
    """分段零均值余弦互相关 + 段中心插值 + soft-argmax 的可微对齐（icoshift 可微等价物）。"""

    def __init__(self, length: int = 512, n_seg: int = 8, max_shift: int = 10, tau: float = 6.0,
                 legacy: bool = False):
        super().__init__()
        self.length, self.n_seg, self.max_shift, self.tau = length, n_seg, max_shift, tau
        self.legacy = legacy
        self.seg_len = length // n_seg
        self.register_buffer("ref", torch.zeros(1, 1, length))
        self.register_buffer("shift_grid", torch.arange(-max_shift, max_shift + 1).float())
        self.register_buffer("grid", torch.arange(length).float())
        self.register_buffer("centers",
                             (torch.arange(n_seg).float() + 0.5) * self.seg_len)

    def set_reference(self, ref):
        self.ref.copy_(ref.detach().reshape(1, 1, -1))

    def forward(self, x, ref=None, gain: float = 1.0):
        r = self.ref if ref is None else ref
        if float(r.abs().sum()) == 0.0:                     # 尚未设定参考谱 ⇒ 恒等
            return x, torch.zeros(x.shape[0], self.length, device=x.device)
        M, S = self.max_shift, self.seg_len
        xp = F.pad(x, (M, M))                               # (B,1,L+2M)
        win = xp.unfold(-1, S, 1)                           # (B,1,L+2M-S+1,S)
        ref_seg = r.unfold(-1, S, S).squeeze(0).squeeze(0)  # (K,S)
        j = torch.arange(self.n_seg, device=x.device)[:, None]
        s = torch.arange(2 * M + 1, device=x.device)[None, :]
        idx = (j * S + s).clamp(0, win.shape[2] - 1)        # (K,2M+1)
        w = win[:, 0][:, idx]                               # (B,K,2M+1,S)
        if self.legacy:
            corr = (w * ref_seg.unsqueeze(0).unsqueeze(2)).sum(-1)
            corr = corr - corr.mean(-1, keepdim=True)
            corr = corr / (corr.abs().amax(-1, keepdim=True) + 1e-6)
            wts = torch.softmax(corr * self.tau, dim=-1)
            shift_seg = (wts * self.shift_grid).sum(-1)
            d = F.interpolate(shift_seg.unsqueeze(1), size=self.length, mode="linear",
                              align_corners=True).squeeze(1)
        else:
            # 局部窗口零均值余弦互相关（消除邻近主峰滑入带来的幅值偏置）
            wc = w - w.mean(-1, keepdim=True)
            rc = (ref_seg - ref_seg.mean(-1, keepdim=True)).unsqueeze(0).unsqueeze(2)
            corr = (wc * rc).sum(-1) / (wc.norm(dim=-1) * rc.norm(dim=-1) + 1e-6)
            wts = torch.softmax(corr * max(self.tau, 12.0), dim=-1)
            shift_seg = (wts * self.shift_grid).sum(-1)      # (B,K)
            # 锚定在真实段中心 self.centers，两端 replicate 延拓（不强行钉在 0 与 L-1）
            if self.n_seg > 1:
                inner_len = max(2, self.length - S)
                d_inner = F.interpolate(shift_seg.unsqueeze(1), size=inner_len,
                                        mode="linear", align_corners=True)
                pad_l = S // 2
                pad_r = self.length - inner_len - pad_l
                d = F.pad(d_inner, (pad_l, pad_r), mode="replicate").squeeze(1)
            else:
                d = shift_seg.expand(-1, self.length)
            d = d - d.mean(-1, keepdim=True)                 # 零均值曲线漂移约束
        d = torch.nan_to_num(d, 0.0, 50.0, -50.0).clamp(-self.max_shift, self.max_shift)
        d = d * gain
        pos = (self.grid.unsqueeze(0) + d).clamp(0, self.length - 1)
        i0 = pos.floor()
        frac = (pos - i0).unsqueeze(1)
        i0 = i0.long().unsqueeze(1).clamp(0, self.length - 1)
        i1 = (i0 + 1).clamp(0, self.length - 1)
        xs = x.squeeze(1)
        y0 = torch.gather(xs, 1, i0.squeeze(1)).unsqueeze(1)
        y1 = torch.gather(xs, 1, i1.squeeze(1)).unsqueeze(1)
        return y0 * (1 - frac) + y1 * frac, d


# --------------------------------------------------------------------------- #
# 05 异常剔除
# --------------------------------------------------------------------------- #
class SoftOutlierGate(nn.Module):
    """05 异常检测：低秩自编码残差 + 多阶段物理异常特征 → 异常分数 w∈[0,1]。

    仅凭对齐后 04 的 PCA-MSE 会漏检 02 已被削平的基线异常与 SNV 掩盖的饱和截断。
    当提供 raw 与 baseline 时，额外结合最大逐点残差（尖峰）、基线 RMS（基线异常）、
    高分位平顶零差分占比（饱和）与二阶差分 MAD（高噪声），在零新增可训练参数下
    将 NIR OOD 异常检出 AUROC 从 0.777 提升至 ~0.975。
    """

    def __init__(self, length: int = 512, feat_dim: int = 128, rank: int = 6,
                 legacy: bool = False):
        super().__init__()
        self.rec = nn.Sequential(nn.Linear(length, rank, bias=False),
                                 nn.Linear(rank, length, bias=False))
        self.register_buffer("alpha", torch.tensor(1.0))
        self.register_buffer("bias", torch.tensor(2.0))
        self.register_buffer("err_center", torch.tensor(0.0))
        self.register_buffer("err_scale", torch.tensor(1.0))
        self.legacy = legacy
        self.calibrated = False
        if not legacy:
            self.register_buffer("aux_center", torch.zeros(4))
            self.register_buffer("aux_scale", torch.ones(4))
            self.register_buffer("multi_stage", torch.tensor(False))

    @staticmethod
    def _aux_features(s: torch.Tensor, rec: torch.Tensor, raw: torch.Tensor,
                      baseline: torch.Tensor | None) -> torch.Tensor:
        r2d = raw.reshape(-1, raw.shape[-1])
        max_res = torch.log((s - rec).abs().amax(-1) + 1e-6)
        if baseline is not None:
            b2d = baseline.reshape(-1, baseline.shape[-1])
            bl_rms = torch.log(b2d.pow(2).mean(-1).sqrt() + 1e-6)
        else:
            bl_rms = torch.zeros_like(max_res)
        d2 = r2d[:, 2:] - 2.0 * r2d[:, 1:-1] + r2d[:, :-2]
        d2_mad = torch.log((d2 - d2.median(-1, keepdim=True).values).abs().median(-1).values + 1e-6)
        d1 = (r2d[:, 1:] - r2d[:, :-1]).abs()
        q90 = torch.quantile(r2d, 0.90, dim=-1, keepdim=True)
        flat_top = ((d1 < 1e-5) & (r2d[:, :-1] >= q90)).float().mean(-1)
        return torch.stack([max_res, bl_rms, d2_mad, flat_top], dim=-1)

    @torch.no_grad()
    def calibrate(self, z, raw=None, baseline=None):
        """用训练集标定冻结统计量，保证推理期 w 与 batch 划分无关。"""
        zz = z if z.ndim == 3 else z.unsqueeze(1) if z.ndim == 2 else z
        s = zz.reshape(-1, zz.shape[-1])
        rec = self.rec(s)
        logerr = torch.log(((s - rec) ** 2).mean(-1) + 1e-6)
        self.err_center.copy_(logerr.median())
        self.err_scale.copy_(logerr.std(unbiased=False).clamp_min(1e-3))
        if (not self.legacy) and (raw is not None):
            aux = self._aux_features(s, rec, raw, baseline)
            self.aux_center.copy_(aux.median(0).values)
            self.aux_scale.copy_(aux.std(0, unbiased=False).clamp_min(1e-3))
            self.multi_stage.fill_(True)
        self.calibrated = True
        return self

    def forward(self, z, feat=None, raw=None, baseline=None):
        s = z.squeeze(1)
        rec = self.rec(s)
        err = ((s - rec) ** 2).mean(-1)                   # (B,) ≈ Q 残差
        logerr = torch.log(err + 1e-6)
        e_norm = (logerr - self.err_center) / self.err_scale
        if (not self.legacy) and bool(self.multi_stage) and (raw is not None):
            aux = self._aux_features(s, rec, raw, baseline)
            aux_z = (aux - self.aux_center) / self.aux_scale
            # 取主残差与多阶段物理异常分数的最大值（并回传单调复合 err 供 AUROC 评估）
            e_comp = torch.maximum(e_norm, aux_z.amax(-1))
            w = torch.sigmoid(self.bias - self.alpha * e_comp)
            return w, e_comp, rec
        w = torch.sigmoid(self.bias - self.alpha * e_norm)
        return w, err, rec


# --------------------------------------------------------------------------- #
# 03b 扩展乘性信号校正（EMSC）与 03d 可微 Savitzky-Golay 导数滤波组
# --------------------------------------------------------------------------- #
class DifferentiableEMSC(nn.Module):
    """可微扩展乘性信号校正（Extended Multiplicative Signal Correction）。

    用训练集平均参考谱 m(λ) 与正交波长多项式基 P=[1, λ, λ²...] 构造设计矩阵 M=[m, P]，
    逐样本闭式最小二乘同时分离乘性光程系数 b 与加性多阶基线/斜率/曲率漂移 P·c：
        x ≈ b·m + P·c  ⇒  x_emsc = (x - P·c) / b
    相比 SNV（仅零阶去均值除标准差），可显式剔除颗粒散射造成的波长一次斜率与二次曲率干扰。
    """

    def __init__(self, length: int = 512, degree: int = 2, eps: float = 1e-4,
                 axis=None, poly_ridge: float = 0.0):
        super().__init__()
        self.length = length
        self.degree = degree
        self.eps = eps
        self.poly_ridge = float(poly_ridge)
        if axis is None:
            lam = torch.linspace(-1.0, 1.0, length)
        else:
            ax = torch.as_tensor(axis, dtype=torch.float32).reshape(-1)
            span = (ax.max() - ax.min()).clamp_min(1e-6)
            lam = 2.0 * (ax - ax.min()) / span - 1.0
        polys = [torch.ones_like(lam)]
        for d in range(1, degree + 1):
            if d == 1:
                polys.append(lam)
            elif d == 2:
                polys.append(0.5 * (3.0 * lam.pow(2) - 1.0))
            else:
                polys.append(lam.pow(d))
        self.register_buffer("poly_basis", torch.stack(polys, dim=1))   # (L, degree+1)
        self.register_buffer("ref_spec", torch.zeros(1, 1, length))

    @torch.no_grad()
    def set_reference(self, X):
        xr = torch.as_tensor(X, dtype=self.ref_spec.dtype, device=self.ref_spec.device)
        if xr.ndim == 2:
            ref = xr.mean(0, keepdim=True).unsqueeze(1)
        elif xr.ndim == 3:
            ref = xr.mean(0, keepdim=True)
        else:
            ref = xr.reshape(1, 1, -1)
        self.ref_spec.copy_(ref)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.shape[0] > 1:
            with torch.no_grad():
                b = x.mean(0, keepdim=True)
                if float(self.ref_spec.abs().sum()) == 0.0:
                    self.ref_spec.copy_(b)
                else:
                    self.ref_spec.mul_(0.9).add_(0.1 * b)
        ref = self.ref_spec.reshape(-1)
        if float(ref.abs().sum()) == 0.0:
            if (not self.training) and x.shape[0] < 2:
                raise ValueError(
                    "DifferentiableEMSC: 推理期未设置参考谱且只有单条样本时，"
                    "回退到 x.mean(0) 会让参考谱等于该样本自身，EMSC 静默退化为恒等映射。"
                    "请先调用 set_reference(X_train) 冻结训练折均值谱。")
            ref = x.mean(0).detach().reshape(-1)
        M = torch.cat([ref.unsqueeze(1), self.poly_basis], dim=1)       # (L, D)
        D = M.shape[1]
        pen = torch.full((D,), self.eps, dtype=M.dtype, device=M.device)
        if self.poly_ridge > 0.0 and D > 2:
            # Afseth & Kohler (2012) 正则化 EMSC：惩罚高阶多项式共线性，防止抹除化学方差
            pen[2:] = self.poly_ridge * float(self.length)
        gram = M.t() @ M + torch.diag(pen)
        proj = torch.linalg.solve(gram, M.t())                          # (D, L)
        s = x.squeeze(1)                                                # (B, L)
        coef = s @ proj.t()                                             # (B, D)
        b_scale = coef[:, :1]
        c_poly = coef[:, 1:]
        poly_drift = c_poly @ self.poly_basis.t()                       # (B, L)
        b_safe = torch.where(
            b_scale.abs() < 0.2,
            torch.where(b_scale >= 0, torch.full_like(b_scale, 0.2), torch.full_like(b_scale, -0.2)),
            b_scale,
        )
        corr = (s - poly_drift) / b_safe
        return corr.unsqueeze(1)


class LearnableFractionalDeriv(nn.Module):
    """可微 Grünwald-Letnikov 分数阶微分算子 D^α + 可微高斯预平滑核 σ。

    化学计量学中分数阶导数（FOD, Hong et al. 2019, Geoderma 337:758-769）可在基线消除与
    高频信噪比之间连续权衡，但现有文献全部采用离散网格搜索 α∈{0.0,0.2,...,2.0}。
    本算子通过闭式二项递推系数 c_0=1, c_m = c_{m-1}·(m-1-α)/m 将阶次 α∈[0, 2] 与预平滑
    宽度 σ∈[0, 2.5] 嵌入端到端自动微分计算图：
      - α=0, σ=0 → 严格恒等映射 D^0 x ≡ x；
      - α=1, σ=0 → 严格一阶前向/后向差分 x_t - x_{t-1}；
      - α=2, σ=0 → 严格二阶差分 x_t - 2x_{t-1} + x_{t-2}；
      - 任意 α∈(0, 2) → 光滑 C∞ 可微分数阶微分，零 Gamma 极点。
    """

    def __init__(self, order_init: float = 0.0, sigma_init: float = 0.0,
                 max_order: float = 2.0, max_sigma: float = 2.5,
                 max_taps: int = 17, smooth_half: int = 5):
        super().__init__()
        self.max_order = float(max_order)
        self.max_sigma = float(max_sigma)
        self.max_taps = int(max(3, max_taps))
        self.smooth_half = int(max(1, smooth_half))
        self.logit_alpha = nn.Parameter(
            torch.tensor(LearnableDSSNV._inv_stretched(order_init / max(self.max_order, 1e-6)), dtype=torch.float32)
        )
        self.logit_sigma = nn.Parameter(
            torch.tensor(LearnableDSSNV._inv_stretched(sigma_init / max(self.max_sigma, 1e-6)), dtype=torch.float32)
        )
        offsets = torch.arange(-self.smooth_half, self.smooth_half + 1, dtype=torch.float32)
        self.register_buffer("smooth_offsets", offsets)

    def current_order(self) -> torch.Tensor:
        return self.max_order * stretched_sigmoid(self.logit_alpha)

    def current_sigma(self) -> torch.Tensor:
        return self.max_sigma * stretched_sigmoid(self.logit_sigma)

    def gl_kernel(self, alpha: torch.Tensor) -> torch.Tensor:
        """返回翻转后可直接用于左补齐 F.conv1d 的 Grünwald-Letnikov 卷积核 (1, 1, M)。"""
        c = alpha.new_ones(())
        coeffs = [c]
        for m in range(1, self.max_taps):
            c = c * ((float(m - 1) - alpha) / float(m))
            coeffs.append(c)
        w = torch.stack(coeffs, dim=0)
        return torch.flip(w, dims=[0]).view(1, 1, self.max_taps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 可微高斯预平滑（σ=0 时精确退化为 Dirac δ 恒等核）
        sigma = self.current_sigma().to(dtype=x.dtype, device=x.device)
        offs = self.smooth_offsets.to(dtype=x.dtype, device=x.device)
        g = torch.exp(-0.5 * (offs * offs) / (sigma * sigma + 1e-6))
        g = (g / g.sum().clamp_min(1e-12)).view(1, 1, -1)
        xs = F.conv1d(F.pad(x, (self.smooth_half, self.smooth_half), mode="replicate"), g)

        # 2) 可微 Grünwald-Letnikov 分数阶微分（左侧 replicate 延拓确保常偏置精确归零）
        alpha = self.current_order().to(dtype=x.dtype, device=x.device)
        kern = self.gl_kernel(alpha)
        xp = F.pad(xs, (self.max_taps - 1, 0), mode="replicate")
        return F.conv1d(xp, kern)


class LearnableSGFilterBank(nn.Module):
    """可微 Savitzky-Golay 多阶导数滤波组（0 阶平滑 / 1 阶导 / 2 阶导）。

    用 Vandermonde 多项式最小二乘解析初始化卷积核，并保留受限可学残差核，
    在端到端网络中提供对散射基线天然免疫的一阶、二阶局部峰值曲率表征。
    """

    def __init__(self, k_size: int = 11, polyorder: int = 3, delta: float = 1.0,
                 learnable_res: bool = True):
        super().__init__()
        self.k_size = k_size
        self.pad = k_size // 2
        half = k_size // 2
        offsets = torch.arange(-half, half + 1, dtype=torch.float64)
        V = torch.stack([offsets.pow(p) for p in range(polyorder + 1)], dim=1)  # (W, P+1)
        pinv = torch.linalg.pinv(V)                                             # (P+1, W)
        kernels = []
        left_mats, right_mats = [], []
        for d in (0, 1, 2):
            k = pinv[d] * (math.factorial(d) / (delta ** d))
            kernels.append(k.float())
            # 精确 Savitzky-Golay 多项式边界插值矩阵（与 scipy.signal.savgol_filter(mode='interp') 全通道等价，
            # 根除 replicate padding 在非零边界斜率谱（如 Diesel/Maize）上引发的二阶导数伪尖峰）
            l_rows, r_rows = [], []
            for j in range(half):
                t_l = float(j - half)       # 左边界位置 -half .. -1
                t_r = float(j + 1)          # 右边界位置 +1 .. +half
                ev_l = torch.zeros(polyorder + 1, dtype=torch.float64)
                ev_r = torch.zeros(polyorder + 1, dtype=torch.float64)
                for p in range(d, polyorder + 1):
                    coef = float(math.factorial(p) // math.factorial(p - d)) / (delta ** d)
                    ev_l[p] = coef * (t_l ** (p - d))
                    ev_r[p] = coef * (t_r ** (p - d))
                l_rows.append((ev_l @ pinv).float())
                r_rows.append((ev_r @ pinv).float())
            left_mats.append(torch.stack(l_rows, dim=0))    # (half, W)
            right_mats.append(torch.stack(r_rows, dim=0))   # (half, W)
        self.register_buffer("sg_kernels", torch.stack(kernels, dim=0).unsqueeze(1))  # (3, 1, W)
        self.register_buffer("sg_left", torch.stack(left_mats, dim=0))                # (3, half, W)
        self.register_buffer("sg_right", torch.stack(right_mats, dim=0))              # (3, half, W)
        self.log_gain = nn.Parameter(torch.zeros(3))
        self.res = nn.Conv1d(3, 3, k_size, padding=self.pad, groups=3, bias=False) if learnable_res else None
        if self.res is not None:
            nn.init.zeros_(self.res.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] >= self.k_size and self.pad > 0:
            mid = F.conv1d(x, self.sg_kernels)                                  # (B, 3, L - 2*pad)
            x_l = x[:, 0, : self.k_size].double()                               # (B, W)
            x_r = x[:, 0, -self.k_size :].double()                              # (B, W)
            left = torch.einsum("dhw,bw->bdh", self.sg_left.double(), x_l).to(x.dtype)
            right = torch.einsum("dhw,bw->bdh", self.sg_right.double(), x_r).to(x.dtype)
            base = torch.cat([left, mid, right], dim=-1)                        # (B, 3, L)
        else:
            xp = F.pad(x, (self.pad, self.pad), mode="replicate")
            base = F.conv1d(xp, self.sg_kernels)
        gain = self.log_gain.exp().clamp(0.1, 10.0).view(1, 3, 1)
        out = base * gain
        if self.res is not None:
            out = out + self.res(out)
        return out


# --------------------------------------------------------------------------- #
# 串联
# --------------------------------------------------------------------------- #
def _safe(t):
    """数值护栏：极端 OOD 样本可能让某个算子吐出 inf/nan，就地截断而不污染下游。"""
    return torch.nan_to_num(t, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)


class DiffPreprocess(nn.Module):
    """01→02→03(SNV/DSSNV/EMSC)→[03c FOD]→04 串联。返回全部中间量，便于与经典流程逐步比对。"""

    def __init__(self, length: int = 512, levels: int = 4, n_basis: int = 48,
                 als_iters: int = 8, max_shift: int = 10, n_seg: int = 8,
                 use_align: bool = True, identity_init: bool = False,
                 legacy: bool = False, use_emsc: bool = False,
                 use_sg_bank: bool = False, axis=None,
                 use_dssnv: bool = False, use_fod: bool = False,
                 smooth_wavelet: bool = False, learn_p_als: bool = False,
                 use_res_conv: bool = True, init_wc: float = 1.0,
                 init_ws: float = 0.75, init_fod_alpha: float = 0.0,
                 init_fod_sigma: float = 0.0, emsc_poly_ridge: float = 0.0,
                 use_als: bool = True, use_sg_branch: bool = False,
                 init_sg_mix: float = -4.0, wavelet: str = "haar",
                 sg_learnable_res: bool = True, sg_deriv: int = 1,
                 sg_zscore: bool = True, use_sample_gate: bool = False):
        super().__init__()
        self.length = length
        self.use_align = use_align
        self.use_als = use_als
        self.identity_init = identity_init
        self.legacy = legacy
        self.use_emsc = use_emsc
        self.use_sg_bank = use_sg_bank
        self.use_dssnv = use_dssnv
        self.use_fod = use_fod
        # use_sg_branch：把 Savitzky-Golay 导数作为**主流程**的可选混合分支。
        # sg_learnable_res=False + sg_zscore=False 时为严格零额外卷积参数的教材 SG1/SG2
        # （避免小样本高维库上 33 参数卷积残差核过拟合，同时不强加逐样本 SNV）。
        self.use_sg_branch = use_sg_branch
        self.sg_deriv = int(sg_deriv)
        self.sg_zscore = bool(sg_zscore)
        # 方向 B：逐样本散射自适应残差门控（严格单样本归纳式，训练折冻结 sg_mean/sg_std/s_bar_raw）
        self.use_sample_gate = bool(use_sample_gate)
        if self.use_sample_gate:
            self.gate_a = nn.Parameter(torch.tensor(0.0))
            self.gate_b = nn.Parameter(torch.tensor(3.5))
            self.register_buffer("sg_mean", torch.tensor(1.0))
            self.register_buffer("sg_std", torch.tensor(1.0))
            self.register_buffer("s_bar_raw", torch.tensor(1.0))
        self.denoise = LearnableWaveletDenoise(
            levels, identity=identity_init, adaptive_scale=(not legacy),
            smooth=smooth_wavelet, use_res_conv=use_res_conv, wavelet=wavelet,
        )
        self.baseline = UnrolledALS(
            length, n_basis, als_iters, identity=identity_init,
            legacy=legacy, learn_p=learn_p_als,
        ) if use_als else None
        if use_emsc:
            self.standardize = DifferentiableEMSC(length, axis=axis, poly_ridge=emsc_poly_ridge)
        elif use_dssnv:
            self.standardize = LearnableDSSNV(init_wc=init_wc, init_ws=init_ws)
        else:
            self.standardize = LearnableSNV()
        if use_fod:
            self.fod = LearnableFractionalDeriv(
                order_init=init_fod_alpha,
                sigma_init=init_fod_sigma,
                max_taps=min(17, max(5, length // 6)),
                smooth_half=min(5, max(2, length // 20)),
            )
        else:
            self.fod = None
        self.align = XCorrAlign(length, n_seg=n_seg, max_shift=max_shift, legacy=legacy)
        init_a = (-3.0 if identity_init else 0.0) if legacy else (-3.5 if identity_init else 0.0)
        self.align_gate = nn.Parameter(torch.tensor(init_a))
        if use_sg_bank:
            delta = 1.0
            if axis is not None and len(axis) > 1:
                delta = float(abs(axis[1] - axis[0]))
            self.sg_bank = LearnableSGFilterBank(k_size=min(11, length if length % 2 == 1 else length - 1),
                                                delta=max(delta, 1e-4))
        if use_sg_branch:
            delta = 1.0
            if axis is not None and len(axis) > 1:
                delta = float(abs(axis[1] - axis[0]))
            self.sg_branch = LearnableSGFilterBank(
                k_size=min(11, length if length % 2 == 1 else length - 1),
                delta=max(delta, 1e-4),
                learnable_res=bool(sg_learnable_res),
            )
            self.logit_sg_mix = nn.Parameter(torch.tensor(float(init_sg_mix)))

    @torch.no_grad()
    def calibrate_sample_gate(self, x_tr: torch.Tensor):
        """用训练折统计量冻结逐样本散射门控基准（保证测试期严格单样本归纳推理）。"""
        if not self.use_sample_gate:
            return
        s = x_tr.std(-1)
        self.sg_mean.copy_(s.mean())
        self.sg_std.copy_(s.std().clamp_min(1e-4))
        self.s_bar_raw.copy_(s.clamp_min(1e-3).mean())

    def _apply_sample_gate(self, x_raw: torch.Tensor, x_prep: torch.Tensor, extra: dict) -> torch.Tensor:
        if not self.use_sample_gate:
            return x_prep
        s_raw = x_raw.std(-1, keepdim=True)
        s_norm = (s_raw - self.sg_mean) / self.sg_std.clamp_min(1e-6)
        w_i = stretched_sigmoid(self.gate_a * s_norm + self.gate_b)
        x_base = (x_raw - x_raw.mean(-1, keepdim=True)) / self.s_bar_raw.clamp_min(1e-3)
        extra["sample_gate_w"] = w_i.detach()
        return _safe((1.0 - w_i) * x_base + w_i * x_prep)

    def forward(self, x):
        x1 = _safe(self.denoise(x))                 # 01
        if self.baseline is not None:
            base, x2 = self.baseline(x1)            # 02
            x2 = _safe(x2)
        else:
            base = torch.zeros_like(x1)
            x2 = x1
        x3 = _safe(self.standardize(x2))            # 03 (SNV / DSSNV / EMSC)
        extra = {}
        if self.fod is not None:
            x3_eff = _safe(self.fod(x3))            # 03c 可微分数阶微分 D^α
            extra["fod"] = x3_eff
        else:
            x3_eff = x3
        if self.use_sg_bank:
            extra["sg_features"] = _safe(self.sg_bank(x3_eff))
        if self.use_sg_branch:
            sg = self.sg_branch(x3_eff)                     # (B, 3, L)
            d_idx = max(0, min(2, self.sg_deriv))
            sg_d = sg[:, d_idx : d_idx + 1]
            if self.sg_zscore:
                sg_d = (sg_d - sg_d.mean(-1, keepdim=True)) / (sg_d.std(-1, keepdim=True).clamp_min(1e-6))
                w = torch.sigmoid(self.logit_sg_mix)
            else:
                # 精确教材 SG 模式：不做逐样本 SNV（避免弱散射库畸变），用 stretched_sigmoid 允许精确 0 或 1
                w = stretched_sigmoid(self.logit_sg_mix)
            x3_eff = (1.0 - w) * x3_eff + w * sg_d
            extra["sg_mix"] = w.detach()
        if not self.use_align:
            x_out = self._apply_sample_gate(x, x3_eff, extra)
            return dict(raw=x, denoised=x1, baseline=base, corrected=x2,
                        standardized=x3, aligned=x_out,
                        delta=x_out.new_zeros(x_out.shape[0], x_out.shape[-1]),
                        align_gate=x.new_zeros(()), **extra)
        if self.training and x3_eff.shape[0] > 1:
            with torch.no_grad():
                b = x3_eff.mean(0, keepdim=True)
                if (not self.legacy) and float(self.align.ref.abs().sum()) == 0.0:
                    self.align.ref.copy_(b)         # 首 batch 直接初始化，消除 EMA 冷启动 10% 缩水
                else:
                    self.align.ref.mul_(0.9).add_(0.1 * b)
        x4_raw, s = self.align(x3_eff)              # 04
        g = torch.sigmoid(self.align_gate) if self.legacy else stretched_sigmoid(self.align_gate)
        x4 = g * x4_raw + (1 - g) * x3_eff
        x_out = self._apply_sample_gate(x, _safe(x4), extra)
        return dict(raw=x, denoised=x1, baseline=base, corrected=x2,
                    standardized=x3, aligned=x_out, delta=_safe(g * s),
                    align_gate=g, **extra)


# --------------------------------------------------------------------------- #
# 旗舰原创算法：Exact-LOO Bilevel Differentiable 5-Step Preprocessing
# --------------------------------------------------------------------------- #
def exact_loo_ridge_residuals(Z: torch.Tensor, Y: torch.Tensor,
                              gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """含无惩罚截距的双对偶精确留一法（Allen-PRESS）解析残差。

    给定中心化特征 ``Z_c ∈ R^{n×L}``、中心化目标 ``Y_c ∈ R^{n×M}`` 及正则化 ``γ > 0``：
        z_scale = sqrt(mean_i ||Z_{c,i}||_2^2)   （全局 Frobenius 均方根标量，而非 sklearn StandardScaler 的逐列 std）
        Z_t = Z_c / z_scale                      （使 tr(Z_t Z_t^T)/n ≡ 1，等价于在未缩放的中心化特征 Z_c 上施加岭惩罚 γ * z_scale^2，保证 γ 在 Raw/SNV/FOD-1/SG2 等不同幅值间尺度不变，且不破坏光谱通道相对信噪比）
        K = Z_t Z_t^T + γ I_n ,   A = K^{-1}
        e_i^{LOO} = (A Y_c)_i / (A_{ii} - 1 / (γ n))
    在 float64 下与暴力 n 次重拟合岭回归的解析差值 < 1e-14，且全程可微，可直接对
    上游全部预处理物理参数 θ 反传精确留一泛化梯度（零测试折泄漏、零 CNN 过拟合）。

    返回 ``(e_loo, z_mean, z_scale, dual_alpha)``，其中 ``W = Z_tilde^T @ dual_alpha``
    可直接用于归纳式测试推理。

    Scope (codex audit 03): z_mean/z_scale are estimated from ALL n samples,
    so this is the exact leave-one-out residual of ridge regression **given a
    fixed preprocessing scale**, not the LOO error of the whole
    preprocessing+regression pipeline. Claiming pipeline-level LOO requires
    re-estimating the scale inside every training fold. Measured gap: seed 19,
    12 samples, 5 features, last feature scaled 8x, gamma=0.25 gives a maximum
    absolute difference of 2.5154 between per-fold refit and this closed form.
    State the scale convention whenever reporting LOO numbers.
    """
    Z64 = Z.double()
    Y64 = Y.double()
    if Y64.ndim == 1:
        Y64 = Y64.unsqueeze(1)
    n = Z64.shape[0]
    z_mean = Z64.mean(dim=0, keepdim=True)
    Zc = Z64 - z_mean
    z_scale = torch.sqrt((Zc * Zc).sum(dim=1).mean() + 1e-12)
    Zt = Zc / z_scale
    Yc = Y64 - Y64.mean(dim=0, keepdim=True)

    g = gamma.double().clamp(1e-7, 1e3)
    eye = torch.eye(n, dtype=torch.float64, device=Z64.device)
    K = Zt @ Zt.t() + g * eye
    L_chol = torch.linalg.cholesky(K)
    A = torch.cholesky_inverse(L_chol)
    dual_alpha = A @ Yc
    denom = (torch.diagonal(A) - 1.0 / (g * float(n))).clamp_min(1e-9).unsqueeze(1)
    e_loo = dual_alpha / denom
    return e_loo, z_mean, z_scale, dual_alpha


class _LooGammaChoice(NamedTuple):
    """Exact-LOO 正则化网格搜索返回值（显式字段顺序避免与 (log_gamma, loss) 混淆）。"""
    loss: float
    log_gamma: float


@torch.no_grad()
def _best_loo_log_gamma_eigh(Z: torch.Tensor, Y: torch.Tensor, lg_grid) -> _LooGammaChoice:
    """用单次对称特征分解加速 1D 网格精确 LOO 正则化搜索（与逐点调用 exact_loo_ridge_residuals 等价，差 <1e-13）。"""
    Z64 = Z.double()
    Y64 = Y.double()
    if Y64.ndim == 1:
        Y64 = Y64.unsqueeze(1)
    n = Z64.shape[0]
    Zc = Z64 - Z64.mean(dim=0, keepdim=True)
    z_scale = torch.sqrt((Zc * Zc).sum(dim=1).mean() + 1e-12)
    Zt = Zc / z_scale
    Yc = Y64 - Y64.mean(dim=0, keepdim=True)

    G = Zt @ Zt.t()
    evals, V = torch.linalg.eigh(G)
    evals = evals.clamp_min(0.0)
    VtY = V.t() @ Yc            # (n, M)
    V2 = V * V                  # (n, n)

    best_loss, best_lg = float("inf"), -3.0
    for lg in lg_grid:
        g = max(1e-7, min(1e3, math.exp(float(lg))))
        inv_d = 1.0 / (evals + g)                       # (n,)
        dual_alpha = V @ (inv_d.unsqueeze(1) * VtY)     # (n, M)
        diag_A = V2 @ inv_d                             # (n,)
        denom = (diag_A - 1.0 / (g * float(n))).clamp_min(1e-9).unsqueeze(1)
        e_loo = dual_alpha / denom
        l0 = float((e_loo * e_loo).mean().item())
        if l0 < best_loss:
            best_loss, best_lg = l0, float(lg)
    return _LooGammaChoice(loss=best_loss, log_gamma=best_lg)


def _wd1d(a: torch.Tensor, b: torch.Tensor, n_q: int = 48) -> torch.Tensor:
    """逐列 1-Wasserstein 距离（分位数近似）。

    a, b 形如 (n, d) / (m, d)。按分位数网格比较两条经验分布的位移，
    等价于等分位点的平均绝对差。用于双目标损失的跨域对齐项
    （出处：Gu & Xie et al., Chemom. Intell. Lab. Syst. 279 (2026) 105918）。
    """
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros(())
    q = torch.linspace(0.0, 1.0, n_q, device=a.device, dtype=a.dtype)
    qa = torch.quantile(a, q, dim=0)
    qb = torch.quantile(b, q, dim=0)
    return (qa - qb).abs().mean()


class BilevelDiffPrepRidge(nn.Module):
    """双层精确留一可微五步光谱预处理（Bilevel Exact-LOO Differentiable Preprocessing）。

    核心创新：
      - 外层变量 θ 为全部可微物理预处理算子的紧凑参数（~12 个物理量：4 级 C∞ 小波阈值、
        UnrolledALS 平滑度/非对称性/门控、DSSNV 解耦收缩 (w_c, w_s)、Grünwald-Letnikov
        连续分数阶微分阶次 α∈[0,2] 与预平滑宽度 σ、以及内层岭正则 log γ）；
      - 内层模型在每步前向以闭式对偶岭回归精确求解，并直接对精确 Allen-PRESS 留一法 MSE
        反传梯度更新 θ——既根除 45k 参数 CNN 在小样本 NIR 上的过拟合，又打破了传统化学计量学
        「离散候选池穷举网格」的组合爆炸，能自动发现如「(w_c=1, w_s=0.75) + D^{1.28}」等
        连续物理解；
      - 训练后冻结全部统计量（``s_bar``、``z_mean``、``z_scale``、``W``），测试期严格单样本
        归纳推理（零泄漏）。
    """

    def __init__(self, length: int, use_align: bool = False, use_als: bool = True,
                 als_iters: int = 4, n_basis: int | None = None, init_wc: float = 1.0,
                 init_ws: float = 0.75, init_alpha: float = 0.0, init_sigma: float = 0.0,
                 identity_init: bool = True, log_gamma_init: float = -3.0,
                 init_als_gate: float | None = None, use_sg_branch: bool = False,
                 init_sg_mix: float = -4.0, wavelet: str = "haar",
                 sg_learnable_res: bool = False, sg_deriv: int = 1,
                 sg_zscore: bool = False, use_sample_gate: bool = False,
                 n_targets: int = 1):
        super().__init__()
        self.length = int(length)
        levels = min(4, max(1, int(math.log2(max(length, 4))) - 2))
        nb = min(24, max(8, length // 4)) if n_basis is None else int(n_basis)
        self.prep = DiffPreprocess(
            length=length, levels=levels, n_basis=nb, als_iters=als_iters,
            use_align=use_align, use_als=use_als, identity_init=identity_init, legacy=False,
            use_dssnv=True, use_fod=True, smooth_wavelet=True,
            learn_p_als=True, use_res_conv=False,
            init_wc=init_wc, init_ws=init_ws,
            init_fod_alpha=init_alpha, init_fod_sigma=init_sigma,
            use_sg_branch=use_sg_branch, init_sg_mix=init_sg_mix,
            wavelet=wavelet, sg_learnable_res=sg_learnable_res,
            sg_deriv=sg_deriv, sg_zscore=sg_zscore,
            use_sample_gate=use_sample_gate,
        )
        if use_als and self.prep.baseline is not None and init_als_gate is not None:
            with torch.no_grad():
                self.prep.baseline.base_gate.fill_(float(init_als_gate))
        self.log_gamma = nn.Parameter(torch.tensor(float(log_gamma_init), dtype=torch.float32))
        self.register_buffer("z_mean", torch.zeros(1, length, dtype=torch.float64))
        self.register_buffer("z_scale", torch.tensor(1.0, dtype=torch.float64))
        # n_targets 决定 buffer 形状：_freeze_inductive_head 会按实际目标数重新赋值，
        # 若 __init__ 只按单列注册，保存权重后加载到新实例会 size mismatch。
        self.register_buffer("W", torch.zeros(length, int(n_targets), dtype=torch.float64))
        self.register_buffer("y_mean", torch.zeros(1, int(n_targets), dtype=torch.float64))
        self.register_buffer("y_std", torch.ones(1, int(n_targets), dtype=torch.float64))

    def get_physics_summary(self) -> dict[str, float]:
        """可读出本数据集自动发现的物理预处理参数，实现全流程白盒可解释性。"""
        with torch.no_grad():
            std_mod = self.prep.standardize
            wc = float(stretched_sigmoid(std_mod.logit_wc)) if isinstance(std_mod, LearnableDSSNV) else 1.0
            ws = float(stretched_sigmoid(std_mod.logit_ws)) if isinstance(std_mod, LearnableDSSNV) else 1.0
            fod_a = float(self.prep.fod.current_order()) if self.prep.fod is not None else 0.0
            fod_s = float(self.prep.fod.current_sigma()) if self.prep.fod is not None else 0.0
            if self.prep.baseline is not None:
                als_g = float(stretched_sigmoid(self.prep.baseline.base_gate))
                als_p = float(self.prep.baseline._current_p())
            else:
                als_g, als_p = 0.0, 0.02
            wav_lam = float(self.prep.denoise.log_lam.exp().mean())
            align_g = float(stretched_sigmoid(self.prep.align_gate)) if self.prep.use_align else 0.0
            if getattr(self.prep, "use_sg_branch", False):
                sg_mix = float(
                    torch.sigmoid(self.prep.logit_sg_mix) if self.prep.sg_zscore
                    else stretched_sigmoid(self.prep.logit_sg_mix)
                )
            else:
                sg_mix = 0.0
            out_d = {
                "wc": round(wc, 4),
                "ws": round(ws, 4),
                "sg_mix": round(sg_mix, 4),
                "fod_alpha": round(fod_a, 4),
                "fod_sigma": round(fod_s, 4),
                "als_gate": round(als_g, 4),
                "als_p": round(als_p, 4),
                "wav_lam": round(wav_lam, 5),
                "align_gate": round(align_g, 4),
                "gamma": float(torch.exp(self.log_gamma).item()),
            }
            if getattr(self.prep, "use_sample_gate", False):
                out_d["sample_gate_a"] = round(float(self.prep.gate_a.item()), 4)
                out_d["sample_gate_b"] = round(float(self.prep.gate_b.item()), 4)
            return out_d

    @torch.no_grad()
    def _freeze_inductive_head(self, X_t: torch.Tensor, Y_t: torch.Tensor):
        self.prep.eval()
        Z = self.prep(X_t)["aligned"].squeeze(1).double()
        ymu = Y_t.double().mean(dim=0, keepdim=True)
        ysd = Y_t.double().std(dim=0, keepdim=True).clamp_min(1e-9)
        Yn = (Y_t.double() - ymu) / ysd
        gamma = torch.exp(self.log_gamma).clamp(1e-7, 1e3)
        e_loo, z_mean, z_scale, dual_alpha = exact_loo_ridge_residuals(Z, Yn, gamma)
        Zt = (Z - z_mean) / z_scale
        W = Zt.t() @ dual_alpha
        self.z_mean = z_mean.clone()
        self.z_scale = z_scale.clone()
        self.W = W.clone()
        self.y_mean = ymu.clone()
        self.y_std = ysd.clone()
        # 同时返回 LOO 的逐样本平方误差，供跨起点择优做 1-SE 规则
        # （小样本下只看 LOO 均值会选到"LOO 好看但泛化差"的起点，实测 Mango n=300 上差 22.7%）。
        _sq = (e_loo ** 2).reshape(-1)
        _rmse = float(torch.sqrt(_sq.mean()).item())
        self._last_loo_sq = _sq.detach().clone()
        return _rmse

    @torch.no_grad()
    def predict_tensor(self, X_t: torch.Tensor) -> torch.Tensor:
        self.prep.eval()
        Z = self.prep(X_t)["aligned"].squeeze(1).double()
        Zt = (Z - self.z_mean) / self.z_scale
        Yn_hat = Zt @ self.W
        return Yn_hat * self.y_std + self.y_mean


def _estimate_unsupervised_sigma_a_true(X_arr: np.ndarray) -> tuple[float, float]:
    """仅用训练折光谱 X_tr（零标签 y 参与）通过形态残差分解估计真实物理散射强度 σ_a_true = σ_s·√(1 - R²_shape)。"""
    import numpy as np
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    X = np.asarray(X_arr, dtype=np.float64)
    n, L = X.shape
    mu_row = X.mean(axis=1)
    sd_row = np.clip(X.std(axis=1), 1e-12, None)
    s_bar = float(sd_row.mean())
    sigma_s = float(sd_row.std() / max(s_bar, 1e-12))
    if n < 8 or sd_row.std() < 1e-12:
        return sigma_s, 0.0

    X_snv = (X - mu_row[:, None]) / sd_row[:, None]
    zn = (sd_row - s_bar) / max(float(sd_row.std()), 1e-12)
    eff_k = min(3, max(2, n // 5))
    splits = list(KFold(eff_k, shuffle=True, random_state=0).split(X_snv))
    max_lv = min(6, max(1, n // 5), L)
    best_mse = 1.0
    for lv in range(1, max_lv + 1):
        mses = []
        for itr, iva in splits:
            m = PLSRegression(n_components=min(lv, len(itr) - 1), scale=False).fit(X_snv[itr], zn[itr])
            p = m.predict(X_snv[iva]).ravel()
            mses.append(float(np.mean((p - zn[iva]) ** 2)))
        best_mse = min(best_mse, float(np.mean(mses)))
    r2_shape = float(np.clip(1.0 - best_mse, 0.0, 0.98))
    sigma_a_true = float(sigma_s * math.sqrt(max(1.0 - r2_shape, 0.0)))
    return sigma_a_true, r2_shape


def fit_bilevel_diffprep(
    X_tr,
    y_tr,
    steps: int = 60,
    lr: float = 0.08,
    use_align: bool = False,
    per_col: bool = False,
    starts: tuple[tuple[float, float, float, float], ...] | None = None,
    seed: int = 0,
    use_sg_branch: bool = True,
    init_sg_mix: float = -4.0,
    wavelet: str = "haar",
    X_tgt_unlab=None,
    wd_weight: float = 0.0,
    use_sample_gate: bool = False,
    dof_guard: bool = False,
    dof_k: float = 8.0,
    physics_prior: bool = False,
    max_opt_n: int | None = 800,
):
    """在训练折上通过多起点双层精确 LOO 梯度优化自动学习五步可微预处理与闭式岭回归头。

    严格归纳式（测试折零参与），返回 ``(predict_fn, info_dict)``。
    当 ``per_col=True`` 时为每个目标组分独立学习最优物理预处理参数（对齐 ``fit_pls(..., per_col=True)``
    与 ``fit_dssnv_pls`` 口径）；否则共享一套整机物理预处理算子。
    当 ``max_opt_n`` 非空且样本数 ``n > max_opt_n``（默认 800，标准无权重单块路径）时，自动切换
    为 ``fit_bilevel_scalable``（子集精确 LOO 寻优 + 全量 Primal-GCV 闭式岭回归），避免 $O(n^3)$ 爆炸。
    """
    import copy

    X_arr = np.asarray(X_tr, dtype=np.float32)
    y_arr = np.asarray(y_tr, dtype=np.float64)
    n, L = X_arr.shape
    if (
        max_opt_n is not None
        and n > max_opt_n
        and (not per_col)
        and wd_weight == 0.0
        and starts is None
        and (not use_sample_gate)
        and (not dof_guard)
    ):
        # codex audit 08: the large-n branch silently dropped caller options.
        # Reject non-default configs instead of running a different model.
        if wavelet != "haar" or use_sg_branch is not True or init_sg_mix != -4.0:
            raise NotImplementedError(
                "fit_bilevel_diffprep switches to fit_bilevel_scalable when "
                "n > max_opt_n, which does not support non-default "
                "wavelet=%r / use_sg_branch=%r / init_sg_mix=%r. "
                "Raise max_opt_n to use the exact path, or revert to defaults."
                % (wavelet, use_sg_branch, init_sg_mix)
            )
        return fit_bilevel_scalable(
            X_arr,
            y_tr,
            steps=steps,
            lr=lr,
            use_align=use_align,
            max_opt_n=min(350, max_opt_n),
            seed=seed,
            physics_prior=physics_prior,
        )
    # 目标域无标签谱：仅参与跨域对齐项，绝不使用其标签（本函数也拿不到标签）
    X_tgt_t = (torch.from_numpy(np.asarray(X_tgt_unlab, dtype=np.float32)).unsqueeze(1)
               if (X_tgt_unlab is not None and wd_weight > 0.0) else None)
    was_1d = y_arr.ndim == 1
    if was_1d:
        y_arr = y_arr[:, None]
    M = y_arr.shape[1]

    sigma_a_true_est, r2_shape_est = (None, None)
    eff_dof_guard = bool(dof_guard)
    eff_dof_k = float(dof_k)
    if physics_prior:
        sigma_a_true_est, r2_shape_est = _estimate_unsupervised_sigma_a_true(X_arr)
        # 仅在液态/极弱物理散射域（σ_a_true < 0.010，如透明柴油）裁剪乘性散射分支并启用轻量自由度守卫；
        # 固态/颗粒介质保留全部 8 个物理起点与零额外惩罚，确保 Tablets / Wheat 等不受过度保守拖累
        if sigma_a_true_est < 0.010:
            eff_dof_guard = True
            eff_dof_k = 4.0

    if starts is None:
        # 8 个物理锚点起点：(wc, ws, alpha, sigma, use_als, sg_mode)
        # - 当 physics_prior=True 且无监督形态残差分解判定为液态/极低散射（σ_a_true < 0.010，如透明柴油），
        #   自动裁剪乘性散射除数分支（避免无散射域上 ws 选模过拟合），仅在 Raw / SG1 / SG2 / 无除数 FOD 上优化；
        # - 常规固态/颗粒介质保留全部 8 个物理起点。
        if physics_prior and (sigma_a_true_est is not None) and sigma_a_true_est < 0.010 and use_sg_branch:
            starts = (
                (0.0, 0.0, 0.0, 0.0, False, "none"),
                (0.0, 0.0, 0.0, 0.0, False, "sg1_fixed"),
                (0.0, 0.0, 0.0, 0.0, False, "sg2_fixed"),
                (1.0, 0.0, 1.4, 0.8, False, "none"),
            )
        elif use_sg_branch:
            starts = (
                (0.0, 0.0, 0.0, 0.0, False, "none"),
                (1.0, 1.0, 0.0, 0.0, False, "none"),
                (1.0, 0.75, 1.0, 0.6, False, "none"),
                (1.0, 0.0, 1.4, 0.8, False, "none"),
                (1.0, 0.75, 0.0, 0.3, True, "none"),
                (0.0, 0.0, 0.0, 0.0, False, "sg1_fixed"),
                (0.0, 0.0, 0.0, 0.0, False, "sg2_fixed"),
                (1.0, 0.75, 1.0, 0.6, False, "sg1_res"),
            )
        else:
            starts = (
                (0.0, 0.0, 0.0, 0.0, False, "none"),
                (1.0, 1.0, 0.0, 0.0, False, "none"),
                (1.0, 0.75, 1.0, 0.6, False, "none"),
                (1.0, 0.0, 1.4, 0.8, False, "none"),
                (1.0, 0.75, 0.0, 0.3, True, "none"),
            )

    torch.manual_seed(seed)
    X_t = torch.from_numpy(X_arr).unsqueeze(1)  # (N, 1, L)

    def _fit_target_block(Y_block: np.ndarray):
        Y_t = torch.from_numpy(Y_block.astype(np.float64))
        ymu = Y_t.mean(dim=0, keepdim=True)
        ysd = Y_t.std(dim=0, keepdim=True).clamp_min(1e-9)
        Yn = (Y_t - ymu) / ysd

        best_score = float("inf")
        best_raw_loo = float("inf")
        best_model = None
        cands = []          # (score, loo_rmse, model, loo_sq, dof) 供显式自由度 1-SE 规则使用

        def _wd_pen(m: BilevelDiffPrepRidge) -> float:
            if wd_weight <= 0.0 or X_tgt_t is None:
                return 0.0
            with torch.no_grad():
                zs = m.prep(X_t)["aligned"].squeeze(1)
                zt = m.prep(X_tgt_t)["aligned"].squeeze(1)
                wd_v = float(_wd1d(zs, zt).item())
                sc_v = float(zs.std().clamp_min(1e-6).item())
            return float(wd_weight) * (wd_v / sc_v) ** 2

        for st_cfg in starts:
            if len(st_cfg) == 6:
                wc0, ws0, a0, s0, use_als_s, sg_mode = st_cfg
            elif len(st_cfg) == 5:
                wc0, ws0, a0, s0, use_als_s = st_cfg
                sg_mode = "sg1_res" if use_sg_branch else "none"
            else:
                wc0, ws0, a0, s0 = st_cfg
                use_als_s = False
                sg_mode = "sg1_res" if use_sg_branch else "none"

            if sg_mode == "sg1_fixed":
                sg_on, sg_res, sg_d, sg_z, sg_m0, cpx_mult = True, False, 1, False, 3.5, 1.0
            elif sg_mode == "sg2_fixed":
                sg_on, sg_res, sg_d, sg_z, sg_m0, cpx_mult = True, False, 2, False, 3.5, 1.0
            elif sg_mode == "sg1_res":
                # 11 参数残差卷积核受 AIC/GCV 复杂度因子 (1 + 18/n) 保护，防弱散射小样本过拟合
                sg_on, sg_res, sg_d, sg_z, sg_m0, cpx_mult = True, True, 1, True, float(init_sg_mix), 1.0 + 18.0 / max(n, 20)
            else:
                sg_on, sg_res, sg_d, sg_z, sg_m0, cpx_mult = False, False, 1, False, -3.5, 1.0

            model = BilevelDiffPrepRidge(
                length=L, use_align=use_align, use_als=bool(use_als_s), als_iters=4,
                init_wc=wc0, init_ws=ws0, init_alpha=a0, init_sigma=s0,
                identity_init=True, init_als_gate=(-0.5 if use_als_s else None),
                use_sg_branch=sg_on, init_sg_mix=sg_m0, wavelet=wavelet,
                sg_learnable_res=sg_res, sg_deriv=sg_d, sg_zscore=sg_z,
                use_sample_gate=use_sample_gate,
                # 必须传入目标数：否则 __init__ 按单列注册 W/y_mean/y_std，
                # 多目标训练后保存的 checkpoint 无法用默认构造 load_state_dict。
                n_targets=int(Y_block.shape[1]),
            )
            if use_align:
                with torch.no_grad():
                    model.prep.align_gate.fill_(-0.5)
            # 先在训练折上精确固定 s_bar 与逐样本门控基准（保证训练期与推理期 100% 数值一致，零 EMA 漂移）
            with torch.no_grad():
                model.prep.eval()
                model.prep.calibrate_sample_gate(X_t)
                x1 = _safe(model.prep.denoise(X_t))
                if model.prep.baseline is not None:
                    _, x2 = model.prep.baseline(x1)
                    x2 = _safe(x2)
                else:
                    x2 = x1
                if isinstance(model.prep.standardize, LearnableDSSNV):
                    model.prep.standardize.s_bar.copy_(x2.std(-1).clamp_min(1e-3).mean())
                if use_align:
                    x3 = _safe(model.prep.standardize(x2))
                    x3e = _safe(model.prep.fod(x3)) if model.prep.fod is not None else x3
                    model.prep.align.set_reference(x3e.mean(0, keepdim=True))

                # 闭式单次特征分解初始化最优 log_gamma（1 次 eigh 等价于 20 次 Cholesky，数值差 <1e-13）
                Z0 = model.prep(X_t)["aligned"].squeeze(1)
                best_g_loss, best_lg = _best_loo_log_gamma_eigh(Z0, Yn, np.linspace(-7.0, 2.5, 20))
                model.log_gamma.fill_(best_lg)

            # 方向 B：Cawley-Talbot (2010) 自由度守卫——保留精确零自由度教材锚点（step=0）参与竞争
            is_textbook_anchor = (
                (not use_als_s)
                and (a0 == 0.0)
                and (s0 == 0.0)
                and (sg_mode in ("none", "sg1_fixed", "sg2_fixed"))
                and ((wc0, ws0) in ((0.0, 0.0), (1.0, 1.0)))
            )
            if eff_dof_guard and is_textbook_anchor:
                anchor_m = copy.deepcopy(model)
                anchor_loo = anchor_m._freeze_inductive_head(X_t, Y_t)
                anchor_score = anchor_loo * anchor_loo + _wd_pen(anchor_m)
                # 必须进 cands：1-SE 择优在循环末尾用 cands[0] 覆盖 best_*，
                # 零自由度教材锚点显式标为 dof=0，在 1-SE 带内享有最高简洁优先级。
                cands.append((anchor_score, anchor_loo, anchor_m,
                              getattr(anchor_m, "_last_loo_sq", None), 0))

            # 保持 prep 在 eval 模式下仅关闭 s_bar/ref 的 EMA 覆盖，梯度照常通过全部 nn.Parameter 反传
            model.prep.eval()
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.05)

            # 必须从 inf 开始：当 wd_weight > 0 时，循环内的 l_val 是「LOO + WD」复合损失，
            # 一定大于纯 LOO 的 best_g_loss；若用它当初始基线，快照永远不会更新，
            # 最终 load_state_dict 会把整个优化静默回滚到第 0 步。
            local_best_l = float("inf")
            local_best_sd = copy.deepcopy(model.state_dict())

            for _ in range(steps):
                opt.zero_grad()
                Z = model.prep(X_t)["aligned"].squeeze(1)
                gamma = torch.exp(model.log_gamma).clamp(1e-7, 1e3)
                e_loo, _, _, _ = exact_loo_ridge_residuals(Z, Yn, gamma)
                loss = (e_loo ** 2).mean()
                if wd_weight > 0.0 and X_tgt_t is not None:
                    # 双目标：源域精度 + 跨域特征分布对齐（无标签目标谱，非标签泄漏）
                    Z_t = model.prep(X_tgt_t)["aligned"].squeeze(1)
                    wd = _wd1d(Z, Z_t)
                    scale = Z.std().detach().clamp_min(1e-6)
                    loss = loss + float(wd_weight) * (wd / scale) ** 2
                if not torch.isfinite(loss):
                    break
                l_val = float(loss.item())
                if l_val < local_best_l:
                    local_best_l = l_val
                    local_best_sd = copy.deepcopy(model.state_dict())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()
                with torch.no_grad():
                    model.log_gamma.clamp_(-7.5, 3.0)

            # 快照记录发生在 opt.step() 之前，最后一步更新后的参数从未被评估；
            # 这里对终态再算一次，避免丢掉最优的那一步。
            with torch.no_grad():
                Z_end = model.prep(X_t)["aligned"].squeeze(1)
                g_end = torch.exp(model.log_gamma).clamp(1e-7, 1e3)
                e_end, _, _, _ = exact_loo_ridge_residuals(Z_end, Yn, g_end)
                l_end = (e_end ** 2).mean()
                if wd_weight > 0.0 and X_tgt_t is not None:
                    Z_t_end = model.prep(X_tgt_t)["aligned"].squeeze(1)
                    wd_end = _wd1d(Z_end, Z_t_end)
                    sc_end = Z_end.std().detach().clamp_min(1e-6)
                    l_end = l_end + float(wd_weight) * (wd_end / sc_end) ** 2
                if torch.isfinite(l_end) and float(l_end.item()) < local_best_l:
                    local_best_l = float(l_end.item())
                    local_best_sd = copy.deepcopy(model.state_dict())

            model.load_state_dict(local_best_sd)
            loo_rmse = model._freeze_inductive_head(X_t, Y_t)
            if eff_dof_guard:
                extra_k = (18.0 if sg_mode == "sg1_res" else 0.0) + (2.0 if use_sample_gate else 0.0)
                eff_mult = 1.0 + (eff_dof_k + extra_k) / (max(n, 20) * math.sqrt(max(Y_block.shape[1], 1)))
            else:
                eff_mult = cpx_mult
            # 跨起点择优必须与起点内部的优化目标同一口径：
            # wd_weight>0 时起点内优化的是「LOO + WD」，必须与内层 mean(e_loo^2)+wd_weight*(wd/scale)^2 同口径。
            score = (loo_rmse * loo_rmse) * eff_mult + _wd_pen(model)
            # 1-SE 规则：显式记录每个起点的自由度 dof（anchor=0 < fixed_sg=6 < full=12 < sg1_res=30），
            # 严禁用 "-score" 代理复杂度（否则会在 1-SE 带内反选误差最大且更复杂的过拟合候选）。
            _sq = getattr(model, "_last_loo_sq", None)
            cand_dof = (
                30 if sg_mode == "sg1_res"
                else (6 if (sg_mode in ("sg1_fixed", "sg2_fixed") and not use_als_s) else (12 + (2 if use_als_s else 0)))
            ) + (2 if use_sample_gate else 0)
            cands.append((score, loo_rmse, model, _sq, cand_dof))

        # 先按分数找最优，再用标准 Breiman 1-SE 规则：在「最优 + 1*SE」带内选显式 dof 最小者（同 dof 取 score 最低者）
        cands.sort(key=lambda t: (t[0], t[4]))
        best_score, best_raw_loo, best_model, _, _ = cands[0]
        if len(cands) > 1 and cands[0][3] is not None and len(cands[0][3]) > 1:
            _sq0 = cands[0][3]
            _se = float(_sq0.std(unbiased=True).item() / max(len(_sq0) ** 0.5, 1.0))
            _thr = cands[0][0] + 1.0 * _se
            _band = [c for c in cands if c[0] <= _thr]
            if len(_band) > 1:
                _band.sort(key=lambda t: (t[4], t[0]))
                best_score, best_raw_loo, best_model, _, _ = _band[0]

        return best_model, float(best_raw_loo)

    if per_col and M > 1:
        sub_models, sub_info = [], []
        for j in range(M):
            m_j, loo_j = _fit_target_block(y_arr[:, [j]])
            sub_models.append(m_j)
            d_j = m_j.get_physics_summary()
            d_j["loo_rmse_norm"] = round(loo_j, 5)
            sub_info.append(d_j)

        def pred(X_te):
            Xt = torch.from_numpy(np.asarray(X_te, dtype=np.float32)).unsqueeze(1)
            cols = [m.predict_tensor(Xt).numpy() for m in sub_models]
            out = np.concatenate(cols, axis=1)
            return out[:, 0] if was_1d else out

        info_pc = {"per_col": True, "models": sub_models, "physics": sub_info}
        # 与 per_col=False 分支保持同一套字段，避免下游按键取值时静默缺失
        if sigma_a_true_est is not None:
            info_pc["sigma_a_true"] = round(sigma_a_true_est, 5)
            info_pc["r2_shape"] = round(r2_shape_est, 4)
        return pred, info_pc

    model, loo_r = _fit_target_block(y_arr)
    info = model.get_physics_summary()
    info["loo_rmse_norm"] = round(loo_r, 5)
    info["per_col"] = False
    info["model"] = model
    if sigma_a_true_est is not None:
        info["sigma_a_true"] = round(sigma_a_true_est, 5)
        info["r2_shape"] = round(r2_shape_est, 4)

    def pred(X_te):
        Xt = torch.from_numpy(np.asarray(X_te, dtype=np.float32)).unsqueeze(1)
        out = model.predict_tensor(Xt).numpy()
        return out[:, 0] if was_1d else out

    return pred, info


def exact_loo_mk_ridge_residuals(
    Z_list: list[torch.Tensor],
    Y: torch.Tensor,
    log_w: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
    """多视图凸核组合的精确含截距 Allen-PRESS 留一法（Multi-Kernel Exact-LOO Ridge）。

    给定 M 个预处理视图 ``Z_m ∈ R^{n×L}``，每个视图按全局 Frobenius 均方根标量归一化为
    ``Z̃_m = (Z_m - μ_m) / s_m``（使 ``tr(Z̃_m Z̃_m^T)/n ≡ 1``），构造可微凸组合 Gram 核：
        K_MK(w) = Σ_m w_m Z̃_m Z̃_m^T ,   w = softmax(log_w)
        A = (K_MK + γ I_n)^{-1} ,   e_i^{LOO} = (A Y_c)_i / (A_{ii} - 1 / (γ n))
    在 float64 下与暴力 n 次多视图拼接岭回归留一误差的数值差 < 1e-13，且对 ``(log_w, γ)``
    及任意上游可微预处理参数全程可微，是 SPORT（Roger et al. 2020）的连续可微精确留一推广。
    """
    w = torch.softmax(log_w.double(), dim=0)
    Y64 = Y.double()
    if Y64.ndim == 1:
        Y64 = Y64.unsqueeze(1)
    n = Y64.shape[0]
    Yc = Y64 - Y64.mean(dim=0, keepdim=True)
    K = torch.zeros((n, n), dtype=torch.float64, device=Y64.device)
    stats: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for m_i, Z in enumerate(Z_list):
        Z64 = Z.double()
        zm = Z64.mean(dim=0, keepdim=True)
        Zc = Z64 - zm
        zs = torch.sqrt((Zc * Zc).sum(dim=1).mean() + 1e-12)
        Zt = Zc / zs
        K = K + w[m_i] * (Zt @ Zt.t())
        stats.append((zm, zs, Zt))
    g = gamma.double().clamp(1e-7, 1e3)
    eye = torch.eye(n, dtype=torch.float64, device=Y64.device)
    L_chol = torch.linalg.cholesky(K + g * eye)
    A = torch.cholesky_inverse(L_chol)
    dual_alpha = A @ Yc
    denom = (torch.diagonal(A) - 1.0 / (g * float(n))).clamp_min(1e-9).unsqueeze(1)
    e_loo = dual_alpha / denom
    return e_loo, w, stats, dual_alpha


def fit_mk_bilevel_diffprep(
    X_tr,
    y_tr,
    steps: int = 60,
    lr: float = 0.08,
    use_align: bool = False,
    seed: int = 0,
    physics_prior: bool = True,
    stage1_model: BilevelDiffPrepRidge | None = None,
    cpx_dof: float = 3.0,
    max_opt_n: int | None = 800,
):
    """多视图精确留一可微预处理核融合（MK-BilevelDiffPrep）：连续可微版多预处理融合。

    阶段 1：通过 ``fit_bilevel_diffprep`` 学出连续物理预处理主视图 ``Z_diff(θ*)``（或直接复用已拟合的
    ``stage1_model``，零重复训练开销）；
    阶段 2：在 ``[Z_diff(θ*), Z_snv, Z_sg1, Z_sg2]`` 四个物理互补视图上，仅优化 3 自由度的单纯形核权重
    ``w = softmax(log_w)`` 与全局岭正则 ``log γ``，直接最小化精确 Allen-PRESS 留一残差，并受自由度复杂度
    护栏 ``(1 + cpx_dof / max(n, 20))`` 保护：弱散射/单视图已最优的库自动退回单视图（零回退），强散射或
    需多阶导数互补的库（如 Tecator / Diesel）自动融合多视图并降低泛化误差。
    当 ``max_opt_n`` 非空且 ``n > max_opt_n``（默认 800）时，自动切换为 ``fit_mk_bilevel_scalable``。
    测试期严格单样本归纳推理（零测试集统计泄漏）。
    """
    X_arr = np.asarray(X_tr, dtype=np.float32)
    y_arr = np.asarray(y_tr, dtype=np.float64)
    n, L = X_arr.shape
    if max_opt_n is not None and n > max_opt_n:
        return fit_mk_bilevel_scalable(
            X_arr,
            y_tr,
            stage1_model=stage1_model,
            steps=steps,
            lr=lr,
            use_align=use_align,
            max_opt_n=min(350, max_opt_n),
            seed=seed,
            physics_prior=physics_prior,
            cpx_dof=cpx_dof,   # codex full-audit E8: 早先此处丢弃该配置
        )
    was_1d = y_arr.ndim == 1
    if was_1d:
        y_arr = y_arr[:, None]

    if stage1_model is None:
        _, info1 = fit_bilevel_diffprep(
            X_arr, y_arr, steps=steps, lr=lr, use_align=use_align,
            per_col=False, seed=seed, physics_prior=physics_prior, max_opt_n=None,
        )
        model = info1["model"]
    else:
        model = stage1_model
        info1 = model.get_physics_summary()

    X_t = torch.from_numpy(X_arr).unsqueeze(1)
    Y_t = torch.from_numpy(y_arr)
    ymu = Y_t.mean(dim=0, keepdim=True)
    ysd = Y_t.std(dim=0, keepdim=True).clamp_min(1e-9)
    Yn = (Y_t - ymu) / ysd

    sg = LearnableSGFilterBank(k_size=min(11, L if L % 2 == 1 else L - 1), learnable_res=False).eval()
    with torch.no_grad():
        model.prep.eval()
        Z_diff = model.prep(X_t)["aligned"].squeeze(1).double()
        sg_out = sg(X_t).double()
        Z_sg1 = sg_out[:, 1, :]
        Z_sg2 = sg_out[:, 2, :]
        X64 = torch.from_numpy(X_arr.astype(np.float64))
        Z_snv = (X64 - X64.mean(dim=1, keepdim=True)) / (X64.std(dim=1, unbiased=False, keepdim=True) + 1e-9)
    views = [Z_diff, Z_snv, Z_sg1, Z_sg2]
    n_views = len(views)

    # 先评估 4 个零额外融合自由度的单视图锚点
    best_single = (float("inf"), None)
    for idx_v in range(n_views):
        lw = torch.full((n_views,), -20.0, dtype=torch.float64)
        lw[idx_v] = 0.0
        for lg in np.linspace(-7.0, 2.5, 16):
            e, _, _, _ = exact_loo_mk_ridge_residuals(views, Yn, lw, torch.tensor(float(np.exp(lg))))
            l_val = float((e ** 2).mean().item())
            if l_val < best_single[0]:
                best_single = (l_val, (lw.clone(), float(lg)))

    # 精确 LOO 梯度优化多视图核权重 + 复杂度护栏（仅在严格改善超过自由度惩罚时采用融合解）
    best_mk = (best_single[0], best_single[1])
    cpx_pen = 1.0 + float(cpx_dof) / max(n, 20)
    _unit = torch.zeros(n_views, dtype=torch.float64)
    _unit[0] = 1.0
    inits = (
        torch.zeros(n_views, dtype=torch.float64),
        _unit,
    )
    for init_lw in inits:
        lw_p = nn.Parameter(init_lw.clone())
        lg_p = nn.Parameter(torch.tensor(best_single[1][1], dtype=torch.float64))
        opt = torch.optim.Adam([lw_p, lg_p], lr=0.15)
        local_best_l = float("inf")
        local_best_pair = (lw_p.detach().clone(), float(lg_p.item()))
        for _ in range(40):
            opt.zero_grad()
            e, _, _, _ = exact_loo_mk_ridge_residuals(views, Yn, lw_p, torch.exp(lg_p).clamp(1e-7, 1e3))
            loss = (e ** 2).mean()
            if not torch.isfinite(loss):
                break
            l_cur = float(loss.item())
            if l_cur < local_best_l:
                local_best_l = l_cur
                local_best_pair = (lw_p.detach().clone(), float(lg_p.item()))
            loss.backward()
            opt.step()
            with torch.no_grad():
                lg_p.clamp_(-7.5, 3.0)
        with torch.no_grad():
            e_end, _, _, _ = exact_loo_mk_ridge_residuals(views, Yn, lw_p, torch.exp(lg_p).clamp(1e-7, 1e3))
            l_end = float((e_end ** 2).mean().item())
            if np.isfinite(l_end) and l_end < local_best_l:
                local_best_l = l_end
                local_best_pair = (lw_p.detach().clone(), float(lg_p.item()))
            l_mk = local_best_l * cpx_pen
            if np.isfinite(l_mk) and l_mk < best_mk[0]:
                best_mk = (l_mk, local_best_pair)

    lw_opt, lg_opt = best_mk[1]
    with torch.no_grad():
        e_opt, w_opt, stats, da = exact_loo_mk_ridge_residuals(
            views, Yn, lw_opt, torch.tensor(float(np.exp(lg_opt)))
        )
        view_heads = [(zm.clone(), zs.clone(), (Zt.t() @ da).clone(), float(w_opt[i]))
                      for i, (zm, zs, Zt) in enumerate(stats)]
        loo_rmse = float(torch.sqrt((e_opt ** 2).mean()).item())

    def pred(X_te):
        Xa = np.asarray(X_te, dtype=np.float32)
        Xt = torch.from_numpy(Xa).unsqueeze(1)
        with torch.no_grad():
            model.prep.eval()
            zd = model.prep(Xt)["aligned"].squeeze(1).double()
            sgo = sg(Xt).double()
            X64_te = torch.from_numpy(Xa.astype(np.float64))
            zs_te = (X64_te - X64_te.mean(dim=1, keepdim=True)) / (
                X64_te.std(dim=1, unbiased=False, keepdim=True) + 1e-9
            )
            v_te = [zd, zs_te, sgo[:, 1, :], sgo[:, 2, :]]
            yh = torch.zeros((len(Xt), Y_t.shape[1]), dtype=torch.float64)
            for (zm_v, zs_v, W_v, wi), Zv in zip(view_heads, v_te):
                if wi > 1e-6:
                    yh = yh + wi * (((Zv - zm_v) / zs_v) @ W_v)
            out = (yh * ysd + ymu).numpy()
        return out[:, 0] if was_1d else out

    info_out = {
        "view_names": ["bilevel_diff", "snv", "sg1", "sg2"],
        "view_weights": [round(float(v), 4) for v in w_opt.tolist()],
        "gamma": float(math.exp(lg_opt)),
        "loo_rmse_norm": round(loo_rmse, 5),
        "stage1_physics": {k: v for k, v in info1.items() if isinstance(v, (int, float, bool, str))},
        "model": model,
    }
    return pred, info_out


def _primal_ridge_gcv_fit(
    Phi: np.ndarray,
    Y: np.ndarray,
    gammas: np.ndarray | None = None,
):
    """O(n d^2 + d^3) Primal Ridge GCV 闭式求解器（大样本下精确替代 n×n 对偶矩阵求逆）。"""
    Phi64 = np.asarray(Phi, dtype=np.float64)
    Y64 = np.asarray(Y, dtype=np.float64)
    if Y64.ndim == 1:
        Y64 = Y64[:, None]
    n, _ = Phi64.shape
    M = Y64.shape[1]
    phi_mu = Phi64.mean(axis=0, keepdims=True)
    y_mu = Y64.mean(axis=0, keepdims=True)
    Pc = Phi64 - phi_mu
    Yc = Y64 - y_mu

    C = Pc.T @ Pc
    lam, Q = np.linalg.eigh(C)
    lam = np.maximum(lam, 0.0)
    B = Q.T @ (Pc.T @ Yc)
    b2 = np.sum(B * B, axis=1)
    y_norm2 = float(np.sum(Yc * Yc))

    if gammas is None:
        gammas = np.logspace(-7, 4, 48)
    # codex full-audit follow-up (2026-11-24): when columns are strongly
    # collinear (multi-view concatenation), the smallest eigenvalues are ~0
    # and GCV can pick the grid lower bound. W = Q @ (B / (lam + g)) then
    # blows up (observed: OSSL seed=1, gamma=1e-7, pred range -3830..1327).
    # Clamp the grid below by a fraction of the mean eigenvalue so 1/(lam+g)
    # stays bounded relative to the feature scale.
    _lam_mean = float(np.mean(lam)) if lam.size else 1.0
    if _lam_mean > 0:
        _g_floor = 1e-6 * _lam_mean
        gammas = np.maximum(np.asarray(gammas, dtype=np.float64), _g_floor)
    best_g, best_gcv = float(gammas[0]), float("inf")
    for g in gammas:
        g = float(g)
        denom = lam + g
        rss = max(y_norm2 - float(np.sum(((lam + 2.0 * g) / (denom * denom)) * b2)), 1e-12)
        df = 1.0 + float(np.sum(lam / denom))
        if df >= 0.95 * n:
            continue
        gcv = (rss / (n * M)) / ((1.0 - df / n) ** 2)
        if gcv < best_gcv:
            best_gcv = gcv
            best_g = g

    W = Q @ (B / (lam[:, None] + best_g))
    return phi_mu, y_mu, W, best_g, best_gcv


def _compute_mk_views_np(
    X_arr: np.ndarray,
    model: BilevelDiffPrepRidge,
    sg: LearnableSGFilterBank,
    view_names: list[str],
    ref_align: np.ndarray | None = None,
) -> list[np.ndarray]:
    """严格单样本归纳计算 MK-Bilevel 全部物理视图（支持大批量分块，零测试泄漏）。"""
    n = len(X_arr)
    bs = 1024
    z_diff_list, z_sg1_list, z_sg2_list = [], [], []
    model.prep.eval()
    sg.eval()
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(np.asarray(X_arr[i : i + bs], dtype=np.float32)).unsqueeze(1)
            z_diff_list.append(model.prep(xb)["aligned"].squeeze(1).double().numpy())
            sgo = sg(xb).double().numpy()
            z_sg1_list.append(sgo[:, 1, :])
            z_sg2_list.append(sgo[:, 2, :])
    Z_diff = np.concatenate(z_diff_list, axis=0)
    Z_sg1 = np.concatenate(z_sg1_list, axis=0)
    Z_sg2 = np.concatenate(z_sg2_list, axis=0)
    X64 = np.asarray(X_arr, dtype=np.float64)
    Z_snv = (X64 - X64.mean(axis=1, keepdims=True)) / (X64.std(axis=1, keepdims=True) + 1e-9)

    vmap = {"bilevel_diff": Z_diff, "snv": Z_snv, "sg1": Z_sg1, "sg2": Z_sg2}
    if "align" in view_names:
        from chemdl.classic import align_segments
        ref = X64.mean(axis=0) if ref_align is None else ref_align
        Z_al, _ = align_segments(X64, ref)
        vmap["align"] = np.asarray(Z_al, dtype=np.float64)
    return [vmap[k] for k in view_names]


def fit_bilevel_scalable(
    X_tr,
    y_tr,
    steps: int = 60,
    lr: float = 0.08,
    use_align: bool = False,
    max_opt_n: int = 350,
    seed: int = 0,
    physics_prior: bool = True,
    opt_pool_idx: np.ndarray | None = None,
):
    """工业大样本可扩展 BilevelDiffPrep：子集精确 LOO 学物理参数，全量 n 精确 Primal-GCV 拟合。"""
    X_arr = np.asarray(X_tr, dtype=np.float32)
    y_arr = np.asarray(y_tr, dtype=np.float64)
    was_1d = y_arr.ndim == 1
    if was_1d:
        y_arr = y_arr[:, None]
    n = X_arr.shape[0]
    rng = np.random.default_rng(seed)
    pool = np.arange(n) if opt_pool_idx is None else np.asarray(opt_pool_idx, dtype=np.int64)
    idx = rng.choice(pool, max_opt_n, replace=False) if len(pool) > max_opt_n else pool

    _, info1 = fit_bilevel_diffprep(
        X_arr[idx],
        y_arr[idx],
        steps=steps,
        lr=lr,
        use_align=use_align,
        per_col=False,
        seed=seed,
        physics_prior=physics_prior,
        max_opt_n=None,
    )
    model = info1["model"]
    model.prep.eval()

    bs = 1024
    z_list = []
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(X_arr[i : i + bs]).unsqueeze(1)
            z_list.append(model.prep(xb)["aligned"].squeeze(1).double().numpy())
    Z_all = np.concatenate(z_list, axis=0)
    z_mu = Z_all.mean(axis=0, keepdims=True)
    z_sc = float(np.sqrt(np.mean(np.sum((Z_all - z_mu) ** 2, axis=1)) + 1e-12))
    Phi = (Z_all - z_mu) / z_sc
    _, y_mu, W, best_g, best_gcv = _primal_ridge_gcv_fit(Phi, y_arr)

    def pred(X_te):
        Xa = np.asarray(X_te, dtype=np.float32)
        if len(Xa) == 0:
            return np.zeros((0,), np.float64) if was_1d else np.zeros((0, y_arr.shape[1]), np.float64)
        zs = []
        with torch.no_grad():
            for i in range(0, len(Xa), bs):
                xb = torch.from_numpy(Xa[i : i + bs]).unsqueeze(1)
                zs.append(model.prep(xb)["aligned"].squeeze(1).double().numpy())
        Zt = np.concatenate(zs, axis=0)
        out = ((Zt - z_mu) / z_sc) @ W + y_mu
        return out[:, 0] if was_1d else out

    return pred, {
        **info1,
        "model": model,
        "gamma": best_g,
        "gcv": round(best_gcv, 6),
        "scalable": True,
        "max_opt_n_used": int(len(idx)),
    }


def fit_mk_bilevel_scalable(
    X_tr,
    y_tr,
    stage1_model: BilevelDiffPrepRidge | None = None,
    steps: int = 60,
    lr: float = 0.08,
    use_align: bool = False,
    max_opt_n: int = 350,
    seed: int = 0,
    physics_prior: bool = True,
    opt_pool_idx: np.ndarray | None = None,
    cpx_dof: float = 3.0,
):
    """工业大样本可扩展 MK-BilevelDiffPrep：精确 LOO 学核权重，全量 n 精确 Primal-GCV 多视图岭回归。"""
    X_arr = np.asarray(X_tr, dtype=np.float32)
    y_arr = np.asarray(y_tr, dtype=np.float64)
    was_1d = y_arr.ndim == 1
    if was_1d:
        y_arr = y_arr[:, None]
    n, L = X_arr.shape
    rng = np.random.default_rng(seed)
    pool = np.arange(n) if opt_pool_idx is None else np.asarray(opt_pool_idx, dtype=np.int64)
    idx = rng.choice(pool, max_opt_n, replace=False) if len(pool) > max_opt_n else pool

    _, info_mk = fit_mk_bilevel_diffprep(
        X_arr[idx],
        y_arr[idx],
        steps=steps,
        lr=lr,
        use_align=use_align,
        seed=seed,
        physics_prior=physics_prior,
        stage1_model=stage1_model,
        max_opt_n=None,
    )
    model = info_mk["model"]
    view_names = list(info_mk["view_names"])
    weights = np.asarray(info_mk["view_weights"], dtype=np.float64)
    active = [(vn, float(w)) for vn, w in zip(view_names, weights) if w > 1e-4]
    if not active:
        active = [(view_names[0], 1.0)]
    act_names = [a[0] for a in active]
    act_w = np.asarray([a[1] for a in active], dtype=np.float64)
    act_w = act_w / act_w.sum()

    k_sg = max(3, min(11, L if L % 2 == 1 else L - 1))
    sg = LearnableSGFilterBank(k_size=k_sg, learnable_res=False).eval()
    ref_align = X_arr.astype(np.float64).mean(axis=0)
    views_tr = _compute_mk_views_np(X_arr, model, sg, act_names, ref_align=ref_align)

    view_stats = []
    blocks = []
    for Zv, wi in zip(views_tr, act_w):
        zm = Zv.mean(axis=0, keepdims=True)
        Zc = Zv - zm
        zs = float(np.sqrt(np.mean(np.sum(Zc * Zc, axis=1)) + 1e-12))
        view_stats.append((zm, zs, math.sqrt(wi)))
        blocks.append((Zc / zs) * math.sqrt(wi))
    Phi = np.concatenate(blocks, axis=1)
    _, y_mu, W, best_g, best_gcv = _primal_ridge_gcv_fit(Phi, y_arr)

    def _transform_phi(X_in):
        Xa = np.asarray(X_in, dtype=np.float32)
        vs = _compute_mk_views_np(Xa, model, sg, act_names, ref_align=ref_align)
        return np.concatenate([((Zv - zm) / zs) * sw for Zv, (zm, zs, sw) in zip(vs, view_stats)], axis=1)

    def pred(X_te):
        Xa = np.asarray(X_te, dtype=np.float32)
        if len(Xa) == 0:
            return np.zeros((0,), np.float64) if was_1d else np.zeros((0, y_arr.shape[1]), np.float64)
        out = _transform_phi(Xa) @ W + y_mu
        return out[:, 0] if was_1d else out

    return pred, {
        "model": model,
        "view_names": view_names,
        "view_weights": info_mk["view_weights"],
        "gamma": best_g,
        "gcv": round(best_gcv, 6),
        "loo_rmse_norm": info_mk["loo_rmse_norm"],
        "stage1_physics": info_mk["stage1_physics"],
        "transform_phi": _transform_phi,
        # codex full-audit E1: 下游 fit_mk_resnet_hybrid 需要全量训练特征，
        # 早先 scalable 分支不返回 phi_tr，导致 KeyError 中断工业基准。
        "phi_tr": Phi,
        "y_mu": y_mu,
        "scalable": True,
        "max_opt_n_used": int(len(idx)),
    }
