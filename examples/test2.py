import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))

from minsted.fluorophores import FlStatic
from minsted.simulators import MinstedSimulator


def sample_next_photon(sim, fl, current_time, n_phase=400):
    """Sample one photon from the periodic total detection rate."""
    mol_pos_3d, _ = fl.position(current_time)
    mol_pos_2d = mol_pos_3d[:2]

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

    signal_mesh = sim.signal_rate(mol_pos_2d[:, None], s_mesh, fl.brightness)
    background_mesh = sim.background_rate(s_mesh)
    total_mesh = signal_mesh + background_mesh

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
    mol_pos_at_photon = fl.position(photon_time)[0][:2]
    signal_i = float(sim.signal_rate(mol_pos_at_photon, s_i, fl.brightness))
    background_i = float(sim.background_rate(s_i))
    total_i = signal_i + background_i
    origin = "SIGNAL" if np.random.rand() < signal_i / total_i else "BACKGROUND"

    return photon_time, s_i, origin, signal_i, background_i, total_i


def make_fluorophore():
    fl = FlStatic()
    fl.pos = [10.0, 0.0]
    fl.brightness = 30.0
    return fl


def make_simulator():
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

    center_after_nc = np.array(sim.history_after_Nc, dtype=float)
    if len(center_after_nc) > 1:
        std_x = np.std(center_after_nc[:, 0], ddof=1)
        std_y = np.std(center_after_nc[:, 1], ddof=1)
        sigma_c = np.sqrt((std_x**2 + std_y**2) / 2.0)
    else:
        sigma_c = np.nan

    photons_for_estimate = len(center_after_nc)
    estimated_precision = (
        sigma_c / np.sqrt(photons_for_estimate)
        if photons_for_estimate > 0 and np.isfinite(sigma_c)
        else np.nan
    )

    return {
        "estimates": estimates,
        "detections": photon_idx,
        "nc": sim.Nc,
        "signal_rates": np.array(signal_rates),
        "background_rates": np.array(background_rates),
        "background_hits": photon_origins.count("BACKGROUND"),
        "sigma_c": sigma_c,
        "photons_for_estimate": photons_for_estimate,
        "estimated_precision": estimated_precision,
    }


def rms_uncertainty_vs_n(all_estimates, true_pos):
    """Paper-style RMS localization uncertainty evaluated at each photon count."""
    errors = all_estimates - true_pos[None, None, :]
    radial_error = np.linalg.norm(errors, axis=2)

    rmse_x = np.full(all_estimates.shape[1], np.nan)
    rmse_y = np.full(all_estimates.shape[1], np.nan)
    sigma_loc = np.full(all_estimates.shape[1], np.nan)
    success_count = np.zeros(all_estimates.shape[1], dtype=int)

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
        success_count[n] = len(run_indices)
        if len(run_indices) == 0:
            continue

        dx = errors[run_indices, n, 0]
        dy = errors[run_indices, n, 1]
        rmse_x[n] = np.sqrt(np.mean(dx**2))
        rmse_y[n] = np.sqrt(np.mean(dy**2))
        sigma_loc[n] = np.sqrt((rmse_x[n] ** 2 + rmse_y[n] ** 2) / 2.0)

    return rmse_x, rmse_y, sigma_loc, success_count


np.random.seed(1)

N_EXPERIMENTS = int(os.environ.get("MINSTED_N_EXPERIMENTS", 1000))
MAX_PHOTONS = int(os.environ.get("MINSTED_MAX_PHOTONS", 700))
N_PHASE = int(os.environ.get("MINSTED_N_PHASE", 120))

fl_ref = make_fluorophore()
true_pos = fl_ref.pos[:2].astype(float)

all_estimates = np.full((N_EXPERIMENTS, MAX_PHOTONS + 1, 2), np.nan)
final_errors = np.full(N_EXPERIMENTS, np.nan)
mean_signal_rates = np.full(N_EXPERIMENTS, np.nan)
mean_background_rates = np.full(N_EXPERIMENTS, np.nan)
background_hits = np.zeros(N_EXPERIMENTS, dtype=int)
nc_values = np.full(N_EXPERIMENTS, np.nan)
target_photons = np.full(N_EXPERIMENTS, np.nan)
sigma_c_values = np.full(N_EXPERIMENTS, np.nan)
photons_for_estimate = np.full(N_EXPERIMENTS, np.nan)
estimated_precision = np.full(N_EXPERIMENTS, np.nan)

print(f"Running {N_EXPERIMENTS} repeated MINSTED localizations...")

for exp_idx in range(N_EXPERIMENTS):
    if (exp_idx + 1) % 100 == 0:
        print(f"  completed {exp_idx + 1}/{N_EXPERIMENTS}")

    target_n = int(np.clip(np.random.lognormal(mean=np.log(360), sigma=0.55), 90, MAX_PHOTONS))
    target_photons[exp_idx] = target_n

    result = run_one_localization(target_n, n_phase=N_PHASE)
    estimates = result["estimates"]
    all_estimates[exp_idx, : estimates.shape[0]] = estimates
    nc_values[exp_idx] = np.nan if result["nc"] is None else result["nc"]
    sigma_c_values[exp_idx] = result["sigma_c"]
    photons_for_estimate[exp_idx] = result["photons_for_estimate"]

    final_estimate = estimates[target_n]
    if np.all(np.isfinite(final_estimate)):
        final_errors[exp_idx] = np.linalg.norm(final_estimate - true_pos)
        estimated_precision[exp_idx] = final_errors[exp_idx]

    mean_signal_rates[exp_idx] = np.mean(result["signal_rates"])
    mean_background_rates[exp_idx] = np.mean(result["background_rates"])
    background_hits[exp_idx] = result["background_hits"]

rmse_x, rmse_y, sigma_loc, success_count = rms_uncertainty_vs_n(all_estimates, true_pos)

valid_final = np.isfinite(final_errors)
final_median_error = np.median(final_errors[valid_final])
final_success = valid_final & (final_errors <= 5.0 * final_median_error)

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
print(f"Median sigma_c:             {np.nanmedian(sigma_c_values):.3f} nm")
print(f"Median photons N-Nc:        {np.nanmedian(photons_for_estimate):.0f}")
print(f"Median estimated precision: {np.nanmedian(estimated_precision):.3f} nm")
print("=" * 72)

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

valid_precision = estimated_precision[np.isfinite(estimated_precision)]
median_precision = np.nanmedian(valid_precision)
axes[2].hist(valid_precision, bins=np.linspace(0, 12, 49), **hist_style)
axes[2].text(
    0.45,
    0.58,
    f"median = {median_precision:.1f} nm",
    transform=axes[2].transAxes,
    fontsize=12,
)
axes[2].text(-0.14, 1.04, "c", transform=axes[2].transAxes, fontsize=13, fontweight="bold")
axes[2].set_xlabel(r"Estimated $\sigma$ [nm]")
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
    "c, Distribution of estimated localization precision, approximated here by final radial localization error.",
    fontsize=10.5,
)

fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.24, wspace=0.18)
output_path = PROJECT_ROOT / "examples" / "test2_supp_fig_s6_style.png"
plt.savefig(output_path, dpi=300)
print(f"Saved figure: {output_path}")
if os.environ.get("MINSTED_NO_PLOT") == "1":
    plt.close()
else:
    plt.show()