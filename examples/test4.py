import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
OUTPUT_DIR = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))

from minsted.simulators import MinstedSimulator


PANEL_DPI = 300
PANEL_FIGSIZE = (5.2, 3.9)
TRUE_POS = np.array([0.0, 0.0])
BRIGHTNESS = 30.0
DEFAULT_D0 = 250.0
CAMERA_SIGMA_PSF = 117.0
CAMERA_REFERENCE_SBR = np.inf

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
    }
)


def save_panel(fig, output_stem):
    """保存 PNG 预览图。"""
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    fig.savefig(png_path, dpi=PANEL_DPI)
    print(f"Saved figure: {png_path}")


def background_from_sbr(sbr, brightness=BRIGHTNESS):
    """把 SBR 转成均匀背景光子率；SBR=inf 表示无背景。"""
    if np.isinf(sbr):
        return 0.0
    return brightness / float(sbr)


def make_simulator(r_d=0.5, sbr=10, alpha=0.15, gamma=0.97, d_min=40.0, d0=DEFAULT_D0):
    """创建一套 MINSTED 动态追踪参数。"""
    return MinstedSimulator(
        d0=d0,
        alpha=alpha,
        gamma=gamma,
        r_d=r_d,
        r_min=5.0,
        d_min=d_min,
        bg_global=background_from_sbr(sbr),
        f_scan=125.0,
    )


def gaussian_epsf_rate(mol_pos, scan_pos, d, sbr, brightness=BRIGHTNESS):
    """高斯 E-PSF 加均匀背景的探测率模型，供固定 d 的图 a/b 使用。"""
    mol_pos = np.asarray(mol_pos, dtype=float)
    scan_pos = np.asarray(scan_pos, dtype=float)
    r_sq = np.sum((mol_pos[:, None] - scan_pos) ** 2, axis=0)
    signal = brightness * np.exp(-4.0 * np.log(2.0) * r_sq / (d**2 + 1e-12))
    return signal + background_from_sbr(sbr, brightness=brightness)


def circular_positions(center, radius, phase):
    """根据圆心、半径和相位生成 EOD 扫描位置。"""
    return np.vstack(
        [
            center[0] + radius * np.cos(phase),
            center[1] + radius * np.sin(phase),
        ]
    )


def sigma_loc_from_estimates(estimates, true_pos=TRUE_POS):
    """由重复定位的二维估计位置计算等效单轴定位标准差。"""
    valid = np.all(np.isfinite(estimates), axis=1)
    estimates = estimates[valid]
    if len(estimates) < 2:
        return np.nan

    errors = estimates - true_pos[None, :]
    radial_error = np.linalg.norm(errors, axis=1)
    median_error = np.median(radial_error)
    keep = radial_error <= 5.0 * median_error if median_error > 0 else radial_error == 0
    errors = errors[keep]
    if len(errors) < 2:
        return np.nan

    std_x = np.std(errors[:, 0], ddof=1)
    std_y = np.std(errors[:, 1], ddof=1)
    return np.sqrt((std_x**2 + std_y**2) / 2.0)


def fisher_precision_fixed_scan(n_photons, r_over_d, sbr, d=DEFAULT_D0, n_phase=4096):
    """由完整探测率模型数值积分 Fisher 信息，得到固定扫描圆的理论 sigma/d。"""
    radius = r_over_d * d
    phase = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    scan_positions = circular_positions(TRUE_POS, radius, phase)
    signal = gaussian_epsf_rate(TRUE_POS, scan_positions, d, np.inf)
    background = background_from_sbr(sbr)
    total_rate = signal + background

    derivative_x = signal * (8.0 * np.log(2.0) * radius * np.cos(phase) / (d**2))
    mean_total_rate = np.mean(total_rate)
    fisher_x_per_photon = np.mean((derivative_x**2) / (total_rate + 1e-300)) / (
        mean_total_rate + 1e-300
    )
    if fisher_x_per_photon <= 0.0:
        return np.nan
    return 1.0 / (np.sqrt(n_photons * fisher_x_per_photon) * d)


def fisher_information_coefficient(r_over_d=0.5, sbr=10, n_phase=4096):
    """返回单个 photon 的无量纲 Fisher 信息系数 k，使 I_x = k / d_i^2。"""
    sigma_over_d = fisher_precision_fixed_scan(
        n_photons=1,
        r_over_d=r_over_d,
        sbr=sbr,
        d=DEFAULT_D0,
        n_phase=n_phase,
    )
    return 1.0 / (sigma_over_d**2)


def zooming_fisher_precision_curve(
    n_values,
    d_min,
    d0=210.0,
    gamma=0.97,
    r_over_d=0.5,
    sbr=10,
):
    """Fig. 2c 风格快速计算：逐 photon 累加随 d_i 缩小而增加的 Fisher 信息。"""
    n_values = np.asarray(n_values, dtype=int)
    max_n = int(np.max(n_values))
    info_coefficient = fisher_information_coefficient(r_over_d=r_over_d, sbr=sbr)

    photon_indices = np.arange(1, max_n + 1)
    d_i = np.maximum(d0 * (gamma**photon_indices), d_min)
    cumulative_information = np.cumsum(info_coefficient / (d_i**2))
    return 1.0 / np.sqrt(cumulative_information[n_values - 1])


def simulate_zooming_localization_curve(
    n_values,
    d_min,
    repeats=500,
    max_sim_photons=1200,
    d0=210.0,
    gamma=0.97,
    alpha=0.15,
    r_over_d=0.5,
    sbr=10,
    initial_sigma=90.0,
    n_phase=90,
):
    """按论文的 r_hat(N) 规则模拟 Fig. 2c 曲线，并对长 N 做 1/sqrt 外推。"""
    n_values = np.asarray(n_values, dtype=int)
    max_requested = int(np.max(n_values))
    max_sim = min(max_sim_photons, max_requested)
    simulated_n_values = n_values[n_values <= max_sim]

    phase = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    unit_vectors = np.column_stack([np.cos(phase), np.sin(phase)])
    background = background_from_sbr(sbr)
    centers = np.random.normal(loc=0.0, scale=initial_sigma, size=(repeats, 2))

    d_i = d0
    r_i = r_over_d * d_i
    d_min_reached = False
    nc = None
    post_nc_sum = np.zeros_like(centers)
    post_nc_count = 0
    estimates_at_n = {}
    requested_set = set(int(n) for n in simulated_n_values)

    for photon_idx in range(1, max_sim + 1):
        center_sq = np.sum(centers**2, axis=1, keepdims=True)
        center_dot_scan = centers @ unit_vectors.T
        distance_sq = center_sq + r_i**2 + 2.0 * r_i * center_dot_scan
        signal = BRIGHTNESS * np.exp(-4.0 * np.log(2.0) * distance_sq / (d_i**2))
        rates = signal + background

        cumulative = np.cumsum(rates, axis=1)
        random_values = np.random.rand(repeats) * cumulative[:, -1]
        phase_indices = np.sum(cumulative < random_values[:, None], axis=1)
        scan_positions = centers + r_i * unit_vectors[phase_indices]
        centers = (1.0 - alpha) * centers + alpha * scan_positions

        if not d_min_reached:
            d_i = max(d_i * gamma, d_min)
            r_i = max(r_i * gamma, r_over_d * d_min)
            if d_i <= d_min:
                d_min_reached = True
                nc = photon_idx

        if d_min_reached:
            post_nc_sum += centers
            post_nc_count += 1
            current_estimates = post_nc_sum / post_nc_count
        else:
            current_estimates = centers

        if photon_idx in requested_set:
            estimates_at_n[photon_idx] = current_estimates.copy()

    sigma_curve = np.full(len(n_values), np.nan)
    for idx, n in enumerate(n_values):
        if n <= max_sim and n in estimates_at_n:
            sigma_curve[idx] = sigma_loc_from_estimates(estimates_at_n[n])

    if max_requested > max_sim:
        anchor_sigma = sigma_loc_from_estimates(current_estimates)
        if nc is None:
            nc = max_sim
        anchor_count = max(max_sim - nc + 1, 1)
        for idx, n in enumerate(n_values):
            if n > max_sim:
                count_n = max(n - nc + 1, 1)
                sigma_curve[idx] = anchor_sigma * np.sqrt(anchor_count / count_n)

    return sigma_curve


def sample_next_photon(sim, current_time, mol_pos=TRUE_POS, brightness=BRIGHTNESS, n_phase=80):
    """按当前周期内 signal+background 的累计强度抽样下一颗光子。"""
    t_scan = 1.0 / sim.f_scan
    phase_ticks = np.linspace(0.0, t_scan, n_phase + 1)
    dt_phase = phase_ticks[1] - phase_ticks[0]
    omega = 2.0 * np.pi * sim.f_scan

    theta = -omega * (current_time + phase_ticks)
    s_mesh = np.vstack(
        [
            sim.C_i[0] + sim.R_i * np.cos(theta),
            sim.C_i[1] + sim.R_i * np.sin(theta),
        ]
    )

    total_mesh = sim.detection_rate(mol_pos[:, None], s_mesh, brightness)
    cumulative = np.concatenate(
        [[0.0], np.cumsum((total_mesh[:-1] + total_mesh[1:]) * 0.5 * dt_phase)]
    )
    photons_per_turn = cumulative[-1]
    if photons_per_turn <= 0.0:
        raise RuntimeError("Total detection rate is zero; cannot sample photons.")

    optical_depth = np.random.exponential(scale=1.0)
    full_turns = int(optical_depth // photons_per_turn)
    residual = optical_depth - full_turns * photons_per_turn
    t_within_turn = np.interp(residual, cumulative, phase_ticks)
    photon_time = current_time + full_turns * t_scan + t_within_turn

    s_i = sim.get_current_eod_position(photon_time)
    signal_i = float(sim.signal_rate(mol_pos, s_i, brightness))
    background_i = float(sim.background_rate(s_i))
    total_i = signal_i + background_i
    origin = "SIGNAL" if np.random.rand() < signal_i / total_i else "BACKGROUND"

    return photon_time, s_i, origin


def run_one_localization(max_photons, sim, initial_center=None, n_phase=80):
    """运行一次逐光子动态定位，返回估计轨迹、中心轨迹和 d_i 轨迹。"""
    if initial_center is None:
        initial_center = np.random.normal(loc=0.0, scale=60.0, size=2)
    sim.reset_tracker(r0_est=initial_center)

    estimates = np.full((max_photons + 1, 2), np.nan)
    d_history = np.full(max_photons + 1, np.nan)
    estimates[0] = sim.get_localization_estimate()
    d_history[0] = sim.d_i
    current_time = 0.0

    for photon_idx in range(1, max_photons + 1):
        current_time, s_i, origin = sample_next_photon(sim, current_time, n_phase=n_phase)
        sim.register_one_photon(s_i=s_i, origin=origin)
        estimates[photon_idx] = sim.get_localization_estimate()
        d_history[photon_idx] = sim.d_i

    return {
        "estimates": estimates,
        "centers": sim.get_center_history(),
        "d_history": d_history,
        "nc_shrink": sim.Nc,
    }


def simulate_dynamic_precision(
    max_photons,
    repeats,
    r_d=0.5,
    sbr=10,
    alpha=0.15,
    gamma=0.97,
    d_min=40.0,
    d0=DEFAULT_D0,
    initial_scale=60.0,
    n_phase=80,
):
    """重复运行动态 MINSTED 定位，供图 c 统计不同 N 下的位置精度。"""
    all_estimates = np.full((repeats, max_photons + 1, 2), np.nan)

    for repeat_idx in range(repeats):
        sim = make_simulator(r_d=r_d, sbr=sbr, alpha=alpha, gamma=gamma, d_min=d_min, d0=d0)
        initial_center = np.random.normal(loc=0.0, scale=initial_scale, size=2)
        result = run_one_localization(
            max_photons,
            sim,
            initial_center=initial_center,
            n_phase=n_phase,
        )
        all_estimates[repeat_idx] = result["estimates"]

    return all_estimates


def simulate_static_center_process_mc(
    sbr,
    alpha,
    repeats,
    max_photons,
    r_d=0.5,
    d_static=DEFAULT_D0,
    n_phase=90,
):
    """固定 d 和 R，逐光子 Monte Carlo 模拟扫描中心 C_i 的分布演化。"""
    radius = r_d * d_static
    phase = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)
    unit_vectors = np.column_stack([np.cos(phase), np.sin(phase)])
    background = background_from_sbr(sbr)

    centers = np.zeros((max_photons + 1, repeats, 2), dtype=float)
    # 论文补充视频/图 2b 的设定：初始中心均匀分布在以荧光分子为中心、半径为 d 的圆内。
    initial_radius = d_static * np.sqrt(np.random.rand(repeats))
    initial_angle = 2.0 * np.pi * np.random.rand(repeats)
    current_centers = np.column_stack(
        [
            initial_radius * np.cos(initial_angle),
            initial_radius * np.sin(initial_angle),
        ]
    )
    centers[0] = current_centers
    lost = np.zeros(repeats, dtype=bool)

    for photon_idx in range(1, max_photons + 1):
        center_sq = np.sum(current_centers**2, axis=1, keepdims=True)
        center_dot_scan = current_centers @ unit_vectors.T
        distance_sq = center_sq + radius**2 + 2.0 * radius * center_dot_scan
        signal = BRIGHTNESS * np.exp(-4.0 * np.log(2.0) * distance_sq / (d_static**2))
        rates = signal + background

        cumulative = np.cumsum(rates, axis=1)
        random_values = np.random.rand(repeats) * cumulative[:, -1]
        phase_indices = np.sum(cumulative < random_values[:, None], axis=1)

        current_centers += alpha * radius * unit_vectors[phase_indices]
        centers[photon_idx] = current_centers
        lost |= np.linalg.norm(current_centers - TRUE_POS[None, :], axis=1) > d_static

    return centers, lost


def distribution_width_vs_photon(centers, keep_mask):
    """计算每个 photon index 下 C_i 分布的二维等效宽度。"""
    kept_centers = centers[:, keep_mask, :]
    std_x = np.std(kept_centers[:, :, 0], axis=1, ddof=1)
    std_y = np.std(kept_centers[:, :, 1], axis=1, ddof=1)
    return np.sqrt((std_x**2 + std_y**2) / 2.0)


def moving_average(values, window):
    """对 Monte Carlo 宽度曲线做轻微平滑，降低有限重复次数带来的判据抖动。"""
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def convergence_nc_from_static_mc(
    centers,
    lost,
    final_window=40,
    smooth_window=5,
    hold=8,
    width_tol_fraction=0.08,
):
    """按论文图 b 的定义，从 Monte Carlo 的 C_i 分布收敛得到 Nc。"""
    keep_mask = ~lost
    if np.count_nonzero(keep_mask) < 20:
        return np.nan

    widths = distribution_width_vs_photon(centers, keep_mask)
    widths = moving_average(widths, smooth_window)
    final_width = np.mean(widths[-final_window:])
    if not np.isfinite(final_width) or final_width <= 0.0:
        return np.nan

    tolerance = width_tol_fraction * final_width
    last_candidate = len(widths) - hold
    for photon_idx in range(1, last_candidate):
        stable = np.all(np.abs(widths[photon_idx : photon_idx + hold] - final_width) <= tolerance)
        if stable:
            return float(photon_idx)
    return np.nan


def precision_vs_n_from_repeats(all_estimates, n_values):
    """从重复定位轨迹中提取每个总光子数 N 下的 Monte Carlo 精度。"""
    return np.array([sigma_loc_from_estimates(all_estimates[:, n]) for n in n_values])


def camera_precision_curve(n_values, sbr=CAMERA_REFERENCE_SBR):
    """按补充材料 Eq. S10 计算 camera-based localization uncertainty。"""
    n_values = np.asarray(n_values, dtype=float)
    if np.isinf(sbr):
        return 121.0 / np.sqrt(n_values)

    sbr = float(sbr)
    signal_fraction = 1.0 + 2.91 / sbr
    background_factor = 1.0 + 4.24 / sbr + np.sqrt(2.12 / sbr + 4.24)
    return 121.0 / np.sqrt(n_values) * np.sqrt(signal_fraction * background_factor)


def make_panel_a():
    """图 a：固定 R/d 和 SBR，从探测模型数值积分 Fisher 信息得到 sigma/d。"""
    n_detections = int(os.environ.get("MINSTED_TEST4_A_PHOTONS", 100))

    r_over_d = np.linspace(0.20, 2.0, 180)
    sbr_values = [np.inf, 50, 20, 10, 5, 2]
    colors = ["black", "#1f77b4", "#ff7f0e", "#f2b000", "#7b3294", "#6ba43a"]
    labels = [r"$\infty$", "50", "20", "10", "5", "2"]

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    print("Calculating panel a from numerical Fisher information...")

    for sbr, color, label in zip(sbr_values, colors, labels):
        sigma_over_d = []
        for r_d in r_over_d:
            sigma_over_d.append(
                fisher_precision_fixed_scan(
                    n_detections,
                    r_over_d=r_d,
                    sbr=sbr,
                    d=DEFAULT_D0,
                )
            )
        ax.plot(r_over_d, sigma_over_d, color=color, linewidth=1.8, label=label)

    fig.text(0.015, 0.93, "a", fontsize=8, fontweight="bold")
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 0.25)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["", "0.5", "1.0", "1.5", "2.0"])
    ax.set_xlabel(r"$R/d$")
    ax.set_ylabel(r"$\sigma/d$")
    ax.grid(True, color="#dddddd", linewidth=0.55, alpha=0.8)
    ax.tick_params(direction="in", top=True, right=True)

    legend = ax.legend(
        title="SBR",
        loc="center right",
        bbox_to_anchor=(0.98, 0.42),
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        borderpad=0.1,
        labelspacing=0.2,
    )
    legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.96)
    save_panel(fig, "test4_fig2a_localization_precision")
    return fig


def make_panel_b():
    """图 b：固定 d，用 Monte Carlo 中心轨迹统计 Nc 和 |C_i-r|>d 的丢失率。"""
    repeats = int(os.environ.get("MINSTED_TEST4_B_REPEATS", 2500))
    max_photons = int(os.environ.get("MINSTED_TEST4_B_PHOTONS", 180))
    n_phase = int(os.environ.get("MINSTED_TEST4_B_N_PHASE", 90))

    alphas = np.linspace(0.05, 0.25, 9)
    sbr_values = [50, 20, 10, 5, 2]
    colors = ["#1f77b4", "#ff7f0e", "#f2b000", "#7b3294", "#6ba43a"]

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax_right = ax.twinx()
    print("Simulating panel b with static-d Monte Carlo centre distributions...")

    for sbr, color in zip(sbr_values, colors):
        nc_curve = []
        lost_curve = []
        for alpha in alphas:
            centers, lost = simulate_static_center_process_mc(
                sbr=sbr,
                alpha=alpha,
                repeats=repeats,
                max_photons=max_photons,
                r_d=0.5,
                d_static=DEFAULT_D0,
                n_phase=n_phase,
            )
            nc_curve.append(convergence_nc_from_static_mc(centers, lost))
            lost_curve.append(100.0 * np.mean(lost))

        nc_curve = np.asarray(nc_curve, dtype=float)
        lost_curve = np.asarray(lost_curve, dtype=float)

        ax.plot(alphas, nc_curve, color=color, linewidth=1.8, marker=".", label=str(sbr))
        ax_right.plot(
            alphas,
            lost_curve,
            color=color,
            linewidth=1.6,
            marker=".",
            linestyle=(0, (3.0, 2.0)),
        )

    fig.text(0.015, 0.93, "b", fontsize=8, fontweight="bold")
    ax.set_xlim(0.05, 0.25)
    ax.set_ylim(0, 80)
    ax_right.set_ylim(0, 20)
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$N_c$")
    ax_right.set_ylabel("Lost fraction (%)", rotation=270, labelpad=18)
    ax.grid(True, color="#dddddd", linewidth=0.55, alpha=0.8)
    ax.tick_params(direction="in", top=True, right=False)
    ax_right.tick_params(direction="in", top=True, right=True)

    legend = ax.legend(
        title="SBR",
        loc="upper center",
        bbox_to_anchor=(0.63, 0.96),
        frameon=False,
        handlelength=1.4,
        handletextpad=0.35,
        borderpad=0.1,
        labelspacing=0.2,
    )
    legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.16, right=0.84, bottom=0.18, top=0.96)
    save_panel(fig, "test4_fig2b_photon_budget")
    return fig


def make_panel_c():
    """图 c：按论文 r_hat(N) 规则 Monte Carlo 模拟 sigma 随 N 的变化。"""
    max_photons = int(os.environ.get("MINSTED_TEST4_C_PHOTONS", 10000))
    repeats = int(os.environ.get("MINSTED_TEST4_C_REPEATS", 500))
    max_sim_photons = int(os.environ.get("MINSTED_TEST4_C_SIM_PHOTONS", 1200))
    n_phase = int(os.environ.get("MINSTED_TEST4_C_N_PHASE", 90))
    camera_sbr_text = os.environ.get("MINSTED_TEST4_CAMERA_SBR", str(CAMERA_REFERENCE_SBR))
    camera_sbr = np.inf if camera_sbr_text.lower() in {"inf", "infinity"} else float(camera_sbr_text)

    n_values = np.unique(np.round(np.logspace(1, np.log10(max_photons), 260)).astype(int))
    d_min_values = [210, 160, 80, 40, 20, 10]
    colors = ["#1f77b4", "#ff7f0e", "#f2b000", "#7b3294", "#6ba43a", "#4cb7e8"]

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    print("Simulating panel c with paper-style localization estimates...")

    for d_min, color in zip(d_min_values, colors):
        sigma_curve = simulate_zooming_localization_curve(
            n_values,
            d_min=d_min,
            repeats=repeats,
            max_sim_photons=max_sim_photons,
            d0=210.0,
            gamma=0.97,
            alpha=0.15,
            r_over_d=0.5,
            sbr=10,
            initial_sigma=90.0,
            n_phase=n_phase,
        )
        ax.plot(n_values, sigma_curve, color=color, linewidth=1.8, label=f"{d_min}")

    ax.plot(
        n_values,
        camera_precision_curve(n_values, sbr=camera_sbr),
        color="black",
        linewidth=1.6,
        label=rf"Camera, SBR={camera_sbr:g}" if np.isfinite(camera_sbr) else r"Camera, SBR=$\infty$",
    )

    fig.text(0.015, 0.93, "c", fontsize=8, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, max_photons)
    ax.set_ylim(0.05, 120)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\sigma$ (nm)")
    ax.grid(True, color="#dddddd", linewidth=0.55, alpha=0.8)
    ax.tick_params(direction="in", top=True, right=True)

    legend = ax.legend(
        title=r"$d_{\min}$ (nm)",
        loc="lower left",
        frameon=False,
        handlelength=1.6,
        handletextpad=0.35,
        borderpad=0.1,
        labelspacing=0.2,
        fontsize=8,
        title_fontsize=8,
    )
    legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.15, right=0.94, bottom=0.18, top=0.95)
    save_panel(fig, "test4_fig2c_precision_vs_photons")
    return fig


def show_or_close(figures):
    if os.environ.get("MINSTED_NO_PLOT") == "1":
        for fig in figures:
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    seed = int(os.environ.get("MINSTED_TEST4_SEED", 1))
    np.random.seed(seed)
    figs = [make_panel_a(), make_panel_b(), make_panel_c()]
    show_or_close(figs)
