import numpy as np


class FluorescentMolecule:
    def __init__(self,
                 x_init: float = 0.0,  # 初始位置 X (nm)
                 y_init: float = 0.0,  # 初始位置 Y (nm)
                 diff_coef: float = 0.0,  # 扩散系数 D (nm^2/ms), 0 代表固定分子
                 k_on: float = 2.0,  # 从暗态回到亮态的速率常数 (ms^-1)
                 k_off: float = 0.5,  # 从亮态进入暗态的速率常数 (ms^-1)
                 k_bleach_base: float = 0.005  # 基础光漂白速率常数 (ms^-1)
                 ):
        """
        荧光分子动力学状态机类
        """
        # ---- 空间坐标 ----
        self.pos = np.array([x_init, y_init], dtype=float)
        self.diff_coef = diff_coef

        # ---- 动力学速率常数 ----
        self.k_on = k_on
        self.k_off = k_off
        self.k_bleach_base = k_bleach_base

        # ---- 分子当前状态 ----
        # 状态总共有三种: "ON" (亮态), "OFF" (闪烁暗态/不发光), "BLEACHED" (彻底淬灭)
        self.state = "ON"

        # 用于记录该分子自身的历史轨迹，方便后续画图对比“真实轨迹 vs 追踪轨迹”
        self.pos_history = [self.pos.copy()]
        self.state_history = [self.state]

    def update_physics(self, dt: float, local_sted_intensity: float = 0.0):
        """
        在时间步长 dt (单位: ms) 内，更新分子的物理位置和化学状态
        local_sted_intensity: 分子当前所处位置的 STED 光强（用于动态加速光漂白）
        """
        if self.state == "BLEACHED":
            # 已经熄灭的分子，位置不再改变，状态不再转移
            self.pos_history.append(self.pos.copy())
            self.state_history.append(self.state)
            return self.state

        # 1. 仿真分子热运动（二维布朗运动：标准差 σ = sqrt(2 * D * dt)）
        if self.diff_coef > 0:
            step_sigma = np.sqrt(2 * self.diff_coef * dt)
            self.pos += np.random.normal(0, step_sigma, size=2)

        # 2. 基于马尔可夫状态转移概率更新分子状态
        r = np.random.rand()

        if self.state == "ON":
            # 物理现实：由于 STED 光强极大，分子在高光强下发生光化学破坏（漂白）的概率会暴涨
            # 这里的动态漂白速率 = 基础速率 + 系数 * 局部 STED 光强
            current_k_bleach = self.k_bleach_base * (1.0 + 0.5 * local_sted_intensity)

            p_to_off = self.k_off * dt
            p_to_bleach = current_k_bleach * dt

            if r < p_to_bleach:
                self.state = "BLEACHED"
            elif r < (p_to_bleach + p_to_off):
                self.state = "OFF"

        elif self.state == "OFF":
            # 处于暗态时，不承受激发和淬灭，通常认为不会发生光漂白
            p_to_on = self.k_on * dt
            if r < p_to_on:
                self.state = "ON"

        # 3. 记录历史，用于后续定位误差分析
        self.pos_history.append(self.pos.copy())
        self.state_history.append(self.state)

        return self.state

    def reset_molecule(self, x_init: float, y_init: float):
        """重置分子到初始状态，以便开始新一轮蒙特卡洛独立重复实验"""
        self.pos = np.array([x_init, y_init], dtype=float)
        self.state = "ON"
        self.pos_history = [self.pos.copy()]
        self.state_history = [self.state]