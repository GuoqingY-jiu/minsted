from collections.abc import Callable
from typing import Dict

import numpy as np

try:
    import torch
except ImportError:
    torch = None


class MinstedSimulator:
    def __init__(
        self,
        wl_ex: float = 640.0,
        wl_sted: float = 775.0,
        na: float = 1.4,
        n_media: float = 1.518,
        d0: float = 250.0,
        w_ex: float = 120.0,
        zernike_coeffs: Dict[str, float] | None = None,
        pupil_res: int = 128,
        f_scan: float = 125.0,
        alpha: float = 0.15,
        gamma: float = 0.97,
        r_d: float = 0.5,
        r_min: float = 20.0,
        d_min: float = 40.0,
        n_on: int = 10,
        bg_global: float = 3.0,
        bg_map_func: Callable[[float, float], float] | None = None,
    ):
        self.wl_ex = wl_ex
        self.w_ex = w_ex
        self.wl_sted = wl_sted
        self.na = na
        self.n_media = n_media
        self.d0 = d0
        self.pupil_res = pupil_res
        self.zernike_coeffs = zernike_coeffs if zernike_coeffs is not None else {}

        self.alpha = alpha
        self.gamma = gamma
        self.R_d = r_d
        self.r_min = r_min
        self.d_min = d_min
        self.n_on = n_on
        self.f_scan = f_scan
        self.bg_global = bg_global
        self.bg_map_func = bg_map_func

        if torch is not None:
            self._gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._gpu_device = None

        self.C_i = np.array([0.0, 0.0])
        self.photon_count = 0
        self.trajectory = []
        self.estimate_trajectory = []
        self.d_i = self.d0
        self.R_i = self.d0 * self.R_d
        self.I_i = 0.0
        self.imax = (self.d0 / self.d_min) ** 2 - 1.0
        self.Nc_reached = False
        self.Nc = None
        self.history_after_Nc = []
        self.localization_estimate = self.C_i.copy()

        self.reset_tracker()

    def reset_tracker(self, r0_est=None):
        self.C_i = np.array([0.0, 0.0]) if r0_est is None else np.array(r0_est, dtype=float)
        self.photon_count = 0
        self.trajectory = [self.C_i.copy()]
        self.estimate_trajectory = [self.C_i.copy()]

        self.d_i = self.d0
        self.R_i = self.d0 * self.R_d
        self.I_i = 0.0

        self.Nc_reached = False
        self.Nc = None
        self.history_after_Nc = []
        self.localization_estimate = self.C_i.copy()

    def get_current_eod_position(self, time_ms: float) -> np.ndarray:
        omega = 2.0 * np.pi * self.f_scan
        offset = np.array(
            [
                self.R_i * np.cos(-omega * time_ms),
                self.R_i * np.sin(-omega * time_ms),
            ]
        )
        return self.C_i + offset

    def epsf_from_distance_sq(self, r_sq):
        """Simplified MINSTED E-PSF. d_i is the current FWHM in nm."""
        return np.exp(-4.0 * np.log(2.0) * r_sq / (self.d_i**2 + 1e-12))

    def detection_rate(self, mol_pos: np.ndarray, s_pos: np.ndarray, brightness: float):
        """Total detected photon rate in kHz: signal plus background."""
        return self.signal_rate(mol_pos, s_pos, brightness) + self.background_rate(s_pos)

    def signal_rate(self, mol_pos: np.ndarray, s_pos: np.ndarray, brightness: float):
        """Detected signal photon rate in kHz."""
        r_sq = np.sum((mol_pos - s_pos) ** 2, axis=0)
        return brightness * self.epsf_from_distance_sq(r_sq)

    def background_rate(self, s_pos: np.ndarray):
        """Background photon rate in kHz."""
        if self.bg_map_func is None:
            if np.ndim(s_pos) == 1:
                return float(self.bg_global)
            return np.full(np.shape(s_pos)[1], float(self.bg_global))
        if np.ndim(s_pos) == 1:
            return float(self.bg_map_func(s_pos[0], s_pos[1]))
        return np.array([self.bg_map_func(x, y) for x, y in s_pos.T], dtype=float)

    def register_one_photon(self, s_i: np.ndarray, origin="SIGNAL"):
        self.photon_count += 1

        self.C_i = (1.0 - self.alpha) * self.C_i + self.alpha * s_i
        self.trajectory.append(self.C_i.copy())

        if self.d_i > self.d_min:
            self.d_i = max(self.d_i * self.gamma, self.d_min)
            self.R_i = max(self.R_i * self.gamma, self.r_min)
            self.I_i = min((self.d0 / self.d_i) ** 2 - 1.0, self.imax)

            if self.d_i <= self.d_min and not self.Nc_reached:
                self.Nc_reached = True
                self.Nc = self.photon_count
                self.history_after_Nc.append(self.C_i.copy())
        elif self.Nc_reached:
            self.history_after_Nc.append(self.C_i.copy())
        else:
            self.Nc_reached = True
            self.Nc = self.photon_count
            self.history_after_Nc.append(self.C_i.copy())

        if self.Nc_reached:
            self.localization_estimate = np.mean(self.history_after_Nc, axis=0)
        else:
            self.localization_estimate = self.C_i.copy()
        self.estimate_trajectory.append(self.localization_estimate.copy())

    def get_localization_estimate(self) -> np.ndarray:
        """Return the current paper-style localization estimate r_hat(N)."""
        return self.localization_estimate.copy()

    def get_center_history(self) -> np.ndarray:
        """Return the raw circular-scan center trajectory C_i."""
        return np.array(self.trajectory)

    def get_history(self) -> np.ndarray:
        """Return the paper-style localization estimate trajectory r_hat(N)."""
        return np.array(self.estimate_trajectory)

    def get_effective_psf_coefficient(self, mol_pos: np.ndarray, time_ms: float) -> float:
        """
        【严格物理叠加】计算分子在当前位置由 激发光高斯轮廓 与 STED Doughnut 轮廓叠加后的有效发射概率（E-PSF）
        mol_pos: 荧光分子绝对坐标 [x, y]
        time_ms: 当前仿真物理时间戳
        返回：有效激发发射效率因子（对应图c中的黄线 E-PSF，范围在 0 ~ 1.0 之间）
        """
        # 1. 获取当前微秒 EOD 转到的 Doughnut 绝对中心 s_i
        s_i = self.get_current_eod_position(time_ms)

        # 2. 计算分子到当前 Doughnut 暗核中心的距离平方
        r_sq = np.sum((mol_pos - s_i) ** 2)

        # 3. 模拟【绿线】：激发光强分布 (假设激发光轴心与当前扫描位置 s_i 同步)
        # w_ex 是标准共聚焦激发光的束腰半径，通常对应衍射极限 (如 w_ex = 200 nm)
        w_ex = self.d0 / (2 * np.sqrt(2 * np.log(2)))  # 从 FWHM 换算为高斯标准差
        I_ex = np.exp(-(r_sq / (2 * w_ex ** 2)))

        # 4. 模拟【红线】：STED 损耗光分布
        # 理想情况下，Doughnut 零点附近的空心光强分布可以用抛物线（二次方）近似
        # 这里的自适应形状因子随着有效直径 d_i 的收缩变得越来越陡峭
        shape_factor = r_sq / (self.d_i ** 2 + 1e-12)
        I_sted_normalized = self.I_i * shape_factor  # 此时单位已经是 Is

        # 5. 【黄线叠加】：根据图 c 公式组合出真正的有效点扩散函数产额
        # E-PSF = Excitation / (1 + I_sted / Is)
        e_psf = I_ex / (1.0 + I_sted_normalized)

        return float(e_psf)