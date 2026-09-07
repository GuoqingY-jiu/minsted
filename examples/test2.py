import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]

# 允许脚本直接从 examples 目录外运行时仍能导入本项目包。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))

from minsted.fluorophores import FlStatic
from minsted.simulators import MinstedSimulator


def sample_next_photon(sim, fl, current_time, n_phase=400):
    """Sample one photon from the periodic total detection rate."""
    # 取当前分子二维位置；当前示例是静止分子，但保留时间参数以兼容运动分子。
    mol_pos_3d, _ = fl.position(current_time)
    mol_pos_2d = mol_pos_3d[:2]

    # 在当前扫描周期内离散采样相位，保持绝对时间相位连续。
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

    # 计算一圈扫描上每个相位的信号率、背景率和总探测率。
    signal_mesh = sim.signal_rate(mol_pos_2d[:, None], s_mesh, fl.brightness)
    background_mesh = sim.background_rate(s_mesh)
    total_mesh = signal_mesh + background_mesh

    # 对总探测率做梯形积分，得到一圈内累计期望光子数。
    cumulative = np.concatenate(
        [[0.0], np.cumsum((total_mesh[:-1] + total_mesh[1:]) * 0.5 * dt_phase)]
    )
    photons_per_turn = cumulative[-1]
    if photons_per_turn <= 0.0:
        raise RuntimeError("Total detection rate is zero; cannot sample photons.")

    # 非均匀泊松过程采样：抽取指数随机阈值，再用累计积分反查光子时间。
    optical_depth = np.random.exponential(scale=1.0)
    full_turns = int(optical_depth // photons_per_turn)
    residual = optical_depth - full_turns * photons_per_turn
    t_within_turn = np.interp(residual, cumulative, phase_ticks)
    photon_time = current_time + full_turns * t_scan + t_within_turn

    # 在光子实际到达时刻计算扫描位置，并按信号/背景占比抽样光子来源。
    s_i = sim.get_current_eod_position(photon_time)
    mol_pos_at_photon = fl.position(photon_time)[0][:2]
    signal_i = float(sim.signal_rate(mol_pos_at_photon, s_i, fl.brightness))
    background_i = float(sim.background_rate(s_i))
    total_i = signal_i + background_i
    origin = "SIGNAL" if np.random.rand() < signal_i / total_i else "BACKGROUND"

    return photon_time, s_i, origin, signal_i, background_i, total_i


def make_fluorophore():
    # 每次重复定位都使用同一个真实静止分子模型。
    fl = FlStatic()
    fl.pos = [10.0, 0.0]
    fl.brightness = 30.0
    return fl


def make_simulator():
    # 创建一次定位使用的 MINSTED 简化模拟器。
    return MinstedSimulator(
        d0=250.0,
        alpha=0.15,
        gamma=0.97,
        r_min=5.0,
        d_min=40.0,
        bg_global=3.0,
        f_scan=125.0,
    )


def run_one_localization(max_photons, n_phase=400):
    # 单次 localization：随机初始中心，逐个采样光子并更新定位估计。
    fl = make_fluorophore()
    sim = make_simulator()
    sim.reset_tracker(r0_est=np.random.normal(loc=0.0, scale=60.0, size=2))

    estimates = np.full((max_photons + 1, 2), np.nan)
    estimates[0] = sim.C_i.copy()
    current_time = 0.0
    signal_rates = []
    background_rates = []
    photon_origins = []

    for photon_idx in range(1, max_photons + 1):
        # 采样下一个光子，并用该光子对应的扫描位置更新 MINSTED 追踪器。
        (
            current_time,
            s_i_actual,
            origin,
            signal_i,
            background_i,
            _,
        ) = sample_next_photon(sim, fl, current_time, n_phase=n_phase)

        sim.register_one_photon(s_i=s_i_actual, origin=origin)
        estimates[photon_idx] = sim.get_localization_estimate()
        signal_rates.append(signal_i)
        background_rates.append(background_i)
        photon_origins.append(origin)

        if hasattr(fl, "remainingphotons") and fl.remainingphotons <= 0:
            break

    # 统计 Nc 后圆心位置的波动，作为单次定位稳定阶段的中心轨迹宽度。
    center_after_nc = np.array(sim.history_after_Nc, dtype=float)
    if len(center_after_nc) > 1:
        std_x = np.std(center_after_nc[:, 0], ddof=1)
        std_y = np.std(center_after_nc[:, 1], ddof=1)
        sigma_c = np.sqrt((std_x**2 + std_y**2) / 2.0)
    else:
        sigma_c = np.nan
    photons_after_nc = len(center_after_nc)
    return {
        "estimates": estimates,
        "nc": sim.Nc,
        "signal_rates": np.array(signal_rates),
        "background_rates": np.array(background_rates),
        "background_hits": photon_origins.count("BACKGROUND"),
        "sigma_c": sigma_c,
        "photons_for_estimate": photons_after_nc,
        "center_history": sim.get_center_history(),
    }


def rms_uncertainty_bias_vs_n(all_estimates, true_pos):
    """Evaluate RMSE, random spread, and bias at each photon count."""
    # 跨重复实验统计每个光子数 N 下的误差、随机散布和系统偏差。
    errors = all_estimates - true_pos[None, None, :]
    radial_error = np.linalg.norm(errors, axis=2)

    bias_x = np.full(all_estimates.shape[1], np.nan)
    bias_y = np.full(all_estimates.shape[1], np.nan)
    std_x = np.full(all_estimates.shape[1], np.nan)
    std_y = np.full(all_estimates.shape[1], np.nan)
    rmse_x = np.full(all_estimates.shape[1], np.nan)
    rmse_y = np.full(all_estimates.shape[1], np.nan)
    rmse_loc = np.full(all_estimates.shape[1], np.nan)
    std_loc = np.full(all_estimates.shape[1], np.nan)
    bias_loc = np.full(all_estimates.shape[1], np.nan)
    success_experiment_count = np.zeros(all_estimates.shape[1], dtype=int)

    for n in range(1, all_estimates.shape[1]):
        completed = np.isfinite(radial_error[:, n])
        if not np.any(completed):
            continue

        completed_errors = radial_error[completed, n]
        median_error = np.median(completed_errors)
        if median_error <= 0.0:
            valid_completed = completed_errors == 0.0
        else:
            valid_completed = completed_errors <= 5.0 * median_error

        run_indices = np.flatnonzero(completed)[valid_completed]
        success_experiment_count[n] = len(run_indices)
        if len(run_indices) == 0:
            continue

        dx = errors[run_indices, n, 0]
        dy = errors[run_indices, n, 1]
        bias_x[n] = np.mean(dx)
        bias_y[n] = np.mean(dy)
        if len(run_indices) > 1:
            std_x[n] = np.std(dx, ddof=1)
            std_y[n] = np.std(dy, ddof=1)
        else:
            std_x[n] = 0.0
            std_y[n] = 0.0
        rmse_x[n] = np.sqrt(np.mean(dx**2))
        rmse_y[n] = np.sqrt(np.mean(dy**2))
        rmse_loc[n] = np.sqrt((rmse_x[n] ** 2 + rmse_y[n] ** 2) / 2.0)
        std_loc[n] = np.sqrt((std_x[n] ** 2 + std_y[n] ** 2) / 2.0)
        bias_loc[n] = np.sqrt((bias_x[n] ** 2 + bias_y[n] ** 2) / 2.0)

    return {
        "bias_x": bias_x,
        "bias_y": bias_y,
        "bias_loc": bias_loc,
        "std_x": std_x,
        "std_y": std_y,
        "std_loc": std_loc,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_loc": rmse_loc,
        "success_count": success_experiment_count,
    }

def estimate_precision_by_subsampling(center_trace, min_centers=200):
    """Estimate one localization precision from one post-Nc center trace."""
    # 对一条 Nc 后中心轨迹做分组外推，估计单次 localization 的精度。
    center_trace = np.asarray(center_trace, dtype=float)
    n_centers = len(center_trace)
    if n_centers < min_centers:
        return np.nan

    max_m = n_centers // 5
    if max_m < np.floor(10**1.5):
        return np.nan

    m_values = np.floor(
        10 ** np.arange(
            1.5,
            np.log10(max_m) + 0.01,
            0.1,
        )
    ).astype(int)
    m_values = np.unique(m_values[m_values > 1])

    # 对每个组大小 M，连续分组后计算组均值位置的二维等效标准差 sigma_M。
    sigma_m = []
    used_m = []
    for group_size in m_values:
        group_count = n_centers // group_size
        if group_count < 5:
            continue

        grouped = center_trace[: group_count * group_size].reshape(
            group_count,
            group_size,
            2,
        )
        group_means = np.mean(grouped, axis=1)
        std_x = np.std(group_means[:, 0], ddof=1)
        std_y = np.std(group_means[:, 1], ddof=1)
        sigma_m.append(np.sqrt((std_x**2 + std_y**2) / 2.0))
        used_m.append(group_size)

    if len(sigma_m) < 2:
        return np.nan

    used_m = np.asarray(used_m, dtype=float)
    sigma_m = np.asarray(sigma_m, dtype=float)
    # 拟合 sigma_M(M) ~= sigma_1 / sqrt(M) + sigma_inf。
    design = np.column_stack([1.0 / np.sqrt(used_m), np.ones_like(used_m)])
    sigma_1, sigma_inf = np.linalg.lstsq(design, sigma_m, rcond=None)[0]
    sigma_1 = max(float(sigma_1), 0.0)
    sigma_inf = max(float(sigma_inf), 0.0)

    return sigma_1 / np.sqrt(n_centers) + sigma_inf


def uncertainty_vs_n_one_localization(center_history, n_c, min_centers=200):
    """Estimate individual-localization precision for each total photon count N."""
    # 对同一条中心轨迹不断截断到不同 N，得到 precision(N) 曲线。
    center_history = np.asarray(center_history, dtype=float)
    precision_vs_n = np.full(len(center_history), np.nan)
    if n_c is None or not np.isfinite(n_c):
        return precision_vs_n

    n_c = int(n_c)
    first_valid_n = n_c + min_centers - 1
    for n in range(first_valid_n, len(center_history)):
        center_trace = center_history[n_c: n + 1]
        precision_vs_n[n] = estimate_precision_by_subsampling(
            center_trace,
            min_centers=min_centers,
        )

    return precision_vs_n

N_EXPERIMENTS = int(os.environ.get("MINSTED_N_EXPERIMENTS", 1000))
MAX_PHOTONS = int(os.environ.get("MINSTED_MAX_PHOTONS", 700))
N_PHASE = int(os.environ.get("MINSTED_N_PHASE", 120))

# 真实位置只在模拟里已知，用于计算 Monte Carlo 误差统计。
fl_ref = make_fluorophore()
true_pos = fl_ref.pos[:2].astype(float)

# 预分配所有重复实验的统计数组；无数据的位置用 NaN 占位。
all_estimates = np.full((N_EXPERIMENTS, MAX_PHOTONS + 1, 2), np.nan)  # 每次实验在每个光子数 N 下的二维定位估计 [x, y]。
final_errors = np.full(N_EXPERIMENTS, np.nan)  # 每次实验最终定位点到真实位置的径向误差。
mean_signal_rates = np.full(N_EXPERIMENTS, np.nan)  # 每次实验所有探测光子处的平均信号光子率。
mean_background_rates = np.full(N_EXPERIMENTS, np.nan)  # 每次实验所有探测光子处的平均背景光子率。
background_hits = np.zeros(N_EXPERIMENTS, dtype=int)  # 每次实验中被判定为背景来源的光子数量。
nc_values = np.full(N_EXPERIMENTS, np.nan)  # 每次实验达到稳定阶段或 d_min 时的光子编号 Nc。
target_photons = np.full(N_EXPERIMENTS, np.nan)  # 每次实验随机抽取的目标总探测光子数 N。
sigma_c_values = np.full(N_EXPERIMENTS, np.nan)  # 每次实验 Nc 后中心轨迹的二维等效标准差 sigma_c。
photons_for_estimate = np.full(N_EXPERIMENTS, np.nan)  # 每次实验 Nc 后参与最终均值估计的中心点数量。
subsampled_precision = np.full((N_EXPERIMENTS, MAX_PHOTONS + 1), np.nan)  # 每次实验在每个总光子数 N 下的子采样精度估计。

print(f"Running {N_EXPERIMENTS} repeated MINSTED localizations...")

# 重复运行多次独立 localization，积累统计分布。
for exp_idx in range(N_EXPERIMENTS):
    if (exp_idx + 1) % 100 == 0:
        print(f"  completed {exp_idx + 1}/{N_EXPERIMENTS}")

    target_n = int(np.clip(np.random.lognormal(mean=np.log(360), sigma=0.55), 90, MAX_PHOTONS))
    target_photons[exp_idx] = target_n

    # 执行一次定位并保存每个 N 下的定位估计。
    result = run_one_localization(target_n, n_phase=N_PHASE)
    estimates = result["estimates"]
    all_estimates[exp_idx, : estimates.shape[0]] = estimates
    nc_values[exp_idx] = np.nan if result["nc"] is None else result["nc"]
    sigma_c_values[exp_idx] = result["sigma_c"]
    photons_for_estimate[exp_idx] = result["photons_for_estimate"]
    precision_trace = uncertainty_vs_n_one_localization(
        result["center_history"],
        result["nc"],
    )
    subsampled_precision[exp_idx, : len(precision_trace)] = precision_trace

    # 保存最终定位误差以及本次定位的平均信号/背景信息。
    final_estimate = estimates[target_n]
    if np.all(np.isfinite(final_estimate)):
        final_errors[exp_idx] = np.linalg.norm(final_estimate - true_pos)

    mean_signal_rates[exp_idx] = np.mean(result["signal_rates"])
    mean_background_rates[exp_idx] = np.mean(result["background_rates"])


    background_hits[exp_idx] = result["background_hits"]

# 跨所有重复实验，按总光子数 N 统计 Monte Carlo RMSE/std/bias。
uncertainty_stats = rms_uncertainty_bias_vs_n(all_estimates, true_pos)
precision_count_vs_n = np.sum(np.isfinite(subsampled_precision), axis=0)
median_subsampled_precision = np.full(MAX_PHOTONS + 1, np.nan)
q25_subsampled_precision = np.full(MAX_PHOTONS + 1, np.nan)
q75_subsampled_precision = np.full(MAX_PHOTONS + 1, np.nan)
for n in range(MAX_PHOTONS + 1):
    values_n = subsampled_precision[:, n]
    values_n = values_n[np.isfinite(values_n)]
    if len(values_n) == 0:
        continue
    median_subsampled_precision[n] = np.median(values_n)
    q25_subsampled_precision[n] = np.percentile(values_n, 25)
    q75_subsampled_precision[n] = np.percentile(values_n, 75)

# 最终误差成功率采用 5 倍中位径向误差作为离群过滤阈值。
valid_final_errors = np.isfinite(final_errors)
final_median_error = np.median(final_errors[valid_final_errors])
final_success = valid_final_errors & (final_errors <= 5.0 * final_median_error)

print()
print("=" * 72)
print(f"True position:              {true_pos}")
print(f"Repeated runs:              {N_EXPERIMENTS}")
print(f"Target photons per run:     variable, median = {np.nanmedian(target_photons):.0f}")
print(f"Mean Nc:                    {np.nanmean(nc_values):.1f}")
print(f"Mean signal rate:           {np.nanmean(mean_signal_rates):.3f} kHz")
print(f"Mean background rate:       {np.nanmean(mean_background_rates):.3f} kHz")
print(f"Mean background hits:       {np.mean(background_hits):.1f}")
print(f"Final median radial error:  {final_median_error:.3f} nm")
print(f"Final success rate:         {100.0 * np.mean(final_success):.2f}%")
print(f"Median sigma_c (center fluctuation): {np.nanmedian(sigma_c_values):.3f} nm")
print(f"Median photons N-Nc:        {np.nanmedian(photons_for_estimate):.0f}")
print(f'N = 200 RMSE :              {uncertainty_stats["rmse_loc"][200]:.3f} nm')
print(f'N = 200 precision :         {uncertainty_stats["std_loc"][200]:.3f} nm')
print(f'N = 200 bias :              {uncertainty_stats["bias_loc"][200]:.3f} nm')
if np.isfinite(median_subsampled_precision[200]):
    print(f'N = 200 subsampled precision: {median_subsampled_precision[200]:.3f} nm')
print("=" * 72)

# 图 1：重复 localization 的整体分布统计。
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6))
hist_style = dict(color="#6BAED6", edgecolor="#4F7FA2", linewidth=0.9)

valid_sigma_c = sigma_c_values[np.isfinite(sigma_c_values)]
axes[0].hist(valid_sigma_c, bins=np.linspace(0, 30, 46), **hist_style)
axes[0].axvline(12, color="tomato", linewidth=2)
axes[0].text(12.3, axes[0].get_ylim()[1] * 0.92, "<12 nm", fontsize=11)
axes[0].text(-0.14, 1.04, "a", transform=axes[0].transAxes, fontsize=13, fontweight="bold")
axes[0].set_xlabel(r"$\sigma_c$ [nm]")
axes[0].set_ylabel("Frequency")
axes[0].set_xlim(0, 30)

valid_photons = photons_for_estimate[
    np.isfinite(photons_for_estimate) & (photons_for_estimate > 0)
]
log_bins = np.logspace(0, np.log10(MAX_PHOTONS), 38)
axes[1].hist(valid_photons, bins=log_bins, **hist_style)
axes[1].set_xscale("log")
axes[1].axvline(250, color="tomato", linewidth=2)
axes[1].text(270, axes[1].get_ylim()[1] * 0.92, ">250", fontsize=11)
axes[1].text(-0.14, 1.04, "b", transform=axes[1].transAxes, fontsize=13, fontweight="bold")
axes[1].set_xlabel(r"$N-N_c$")
axes[1].set_ylabel("Frequency")
axes[1].set_xlim(1, MAX_PHOTONS)


axes[2].hist(final_errors[valid_final_errors], bins=np.linspace(0, 12, 49), **hist_style)
axes[2].text(
    0.45,
    0.58,
    f"median = {final_median_error:.1f} nm",
    transform=axes[2].transAxes,
    fontsize=12,
)
axes[2].text(-0.14, 1.04, "c", transform=axes[2].transAxes, fontsize=13, fontweight="bold")
axes[2].set_xlabel(r"Errors [nm]")
axes[2].set_ylabel("Frequency")
axes[2].set_xlim(0, 12)

for ax in axes:
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)

fig.suptitle(
    "Characteristics of simulated MINSTED localizations",
    fontsize=14,
    fontweight="bold",
    y=0.96,
)
fig.text(
    0.01,
    0.055,
    "Suppl. Fig. S6 style. a, Distribution of the standard deviation of centre positions. "
    "b, Distribution of detected photons used after Nc.",
    fontsize=10.5,
)
fig.text(
    0.01,
    0.025,
    "c, Distribution of estimated localization errors, approximated here by final radial localization error.",
    fontsize=10.5,
)

fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.24, wspace=0.18)
output_path = PROJECT_ROOT / "examples" / "test2_supp_fig_s6_style.png"
plt.savefig(output_path, dpi=300)
print(f"Saved figure: {output_path}")

# 图 2：Monte Carlo RMSE、std、bias 随总光子数 N 的变化。
n_values = np.arange(MAX_PHOTONS + 1)
valid_n = uncertainty_stats["success_count"] > 0
line_style = dict(linewidth=1.8)
component_style = dict(linewidth=1.0, alpha=0.55, linestyle="--")

fig_metrics, metric_axes = plt.subplots(1, 3, figsize=(14.0, 4.6), sharex=True)

metric_axes[0].plot(
    n_values[valid_n],
    uncertainty_stats["rmse_loc"][valid_n],
    color="#1f77b4",
    label="combined",
    **line_style,
)

metric_axes[0].set_title("RMSE")
metric_axes[0].set_ylabel("Error [nm]")

metric_axes[1].plot(
    n_values[valid_n],
    uncertainty_stats["std_loc"][valid_n],
    color="#2ca02c",
    label="combined",
    **line_style,
)

metric_axes[1].set_title("Standard deviation")
metric_axes[1].set_ylabel("Spread [nm]")

metric_axes[2].plot(
    n_values[valid_n],
    uncertainty_stats["bias_loc"][valid_n],
    color="#d62728",
    label="combined",
    **line_style,
)
metric_axes[2].plot(
    n_values[valid_n],
    uncertainty_stats["bias_x"][valid_n],
    color="#d62728",
    label="x",
    **component_style,
)
metric_axes[2].plot(
    n_values[valid_n],
    uncertainty_stats["bias_y"][valid_n],
    color="#8c564b",
    label="y",
    **component_style,
)
metric_axes[2].set_title("Bias")
metric_axes[2].set_ylabel("Bias [nm]")

for ax in metric_axes:
    ax.set_xlabel("Detected photon number N")
    ax.set_xlim(1, MAX_PHOTONS)
    ax.set_xscale("log")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=9)

fig_metrics.suptitle("Localization error statistics versus photon number", fontsize=14, fontweight="bold")
fig_metrics.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
metrics_output_path = PROJECT_ROOT / "examples" / "test2_rmse_std_bias_vs_n.png"
fig_metrics.savefig(metrics_output_path, dpi=300)
print(f"Saved figure: {metrics_output_path}")

# 导出图 3 使用的数据，便于后续复查和重绘。
precision_output = np.column_stack(
    [
        n_values,
        median_subsampled_precision,
        q25_subsampled_precision,
        q75_subsampled_precision,
        precision_count_vs_n,
        uncertainty_stats["std_loc"],
    ]
)
precision_csv_path = PROJECT_ROOT / "examples" / "test2_subsampled_precision_vs_n.csv"
np.savetxt(
    precision_csv_path,
    precision_output,
    delimiter=",",
    header=(
        "N,median_subsampled_precision_nm,q25_subsampled_precision_nm,"
        "q75_subsampled_precision_nm,valid_subsampled_localizations,monte_carlo_std_nm"
    ),
    comments="",
)
print(f"Saved data: {precision_csv_path}")

# 图 3：验证单次轨迹 sub-sampling 精度估计；横轴从其第一个有效 N 开始。
valid_subsampled_n = np.isfinite(median_subsampled_precision)
first_precision_n = n_values[valid_subsampled_n][0] if np.any(valid_subsampled_n) else 1
valid_monte_carlo_n = valid_n & (n_values >= first_precision_n)
valid_subsampled_n = valid_subsampled_n & (n_values >= first_precision_n)
fig_precision, ax_precision = plt.subplots(figsize=(7.0, 4.6))
ax_precision.plot(
    n_values[valid_monte_carlo_n],
    uncertainty_stats["std_loc"][valid_monte_carlo_n],
    color="black",
    linewidth=1.8,
    label="Monte Carlo std",
)
ax_precision.plot(
    n_values[valid_subsampled_n],
    median_subsampled_precision[valid_subsampled_n],
    color="#1f77b4",
    linewidth=2.0,
    label="median sub-sampling precision",
)
ax_precision.fill_between(
    n_values[valid_subsampled_n],
    q25_subsampled_precision[valid_subsampled_n],
    q75_subsampled_precision[valid_subsampled_n],
    color="#6BAED6",
    alpha=0.22,
    linewidth=0.0,
    label="25-75% sub-sampling range",
)
ax_precision.set_xlim(first_precision_n, MAX_PHOTONS)
ax_precision.set_xlabel("Detected photon number N")
ax_precision.set_ylabel(r"$\sigma$ [nm]")
ax_precision.set_title("Sub-sampling precision validation")
ax_precision.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
ax_precision.tick_params(direction="in", top=True, right=True)
ax_precision.legend(frameon=False)
fig_precision.tight_layout()
precision_fig_path = PROJECT_ROOT / "examples" / "test2_subsampled_precision_vs_n.png"
fig_precision.savefig(precision_fig_path, dpi=300)
print(f"Saved figure: {precision_fig_path}")

if os.environ.get("MINSTED_NO_PLOT") == "1":
    plt.close(fig)
    plt.close(fig_metrics)
    plt.close(fig_precision)
else:
    plt.show()
