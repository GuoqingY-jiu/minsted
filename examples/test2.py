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

    return {
        "estimates": estimates,
        "detections": photon_idx,
        "nc": sim.Nc,
        "signal_rates": np.array(signal_rates),
        "background_rates": np.array(background_rates),
        "background_hits": photon_origins.count("BACKGROUND"),
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

N_EXPERIMENTS = 1000
MAX_PHOTONS = 300
N_PHASE = 400

fl_ref = make_fluorophore()
true_pos = fl_ref.pos[:2].astype(float)

all_estimates = np.full((N_EXPERIMENTS, MAX_PHOTONS + 1, 2), np.nan)
final_errors = np.full(N_EXPERIMENTS, np.nan)
mean_signal_rates = np.full(N_EXPERIMENTS, np.nan)
mean_background_rates = np.full(N_EXPERIMENTS, np.nan)
background_hits = np.zeros(N_EXPERIMENTS, dtype=int)
nc_values = np.full(N_EXPERIMENTS, np.nan)

print(f"Running {N_EXPERIMENTS} repeated MINSTED localizations...")

for exp_idx in range(N_EXPERIMENTS):
    if (exp_idx + 1) % 100 == 0:
        print(f"  completed {exp_idx + 1}/{N_EXPERIMENTS}")

    result = run_one_localization(MAX_PHOTONS, n_phase=N_PHASE)
    estimates = result["estimates"]
    all_estimates[exp_idx] = estimates
    nc_values[exp_idx] = np.nan if result["nc"] is None else result["nc"]

    final_estimate = estimates[MAX_PHOTONS]
    if np.all(np.isfinite(final_estimate)):
        final_errors[exp_idx] = np.linalg.norm(final_estimate - true_pos)

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
print(f"Detected photons per run:   {MAX_PHOTONS}")
print(f"Mean Nc:                    {np.nanmean(nc_values):.1f}")
print(f"Mean signal rate:           {np.nanmean(mean_signal_rates):.3f} kHz")
print(f"Mean background rate:       {np.nanmean(mean_background_rates):.3f} kHz")
print(f"Mean background hits:       {np.mean(background_hits):.1f}/{MAX_PHOTONS}")
print(f"Final median radial error:  {final_median_error:.3f} nm")
print(f"Final success rate:         {100.0 * np.mean(final_success):.2f}%")
print(f"Final RMSE x/y:             {rmse_x[MAX_PHOTONS]:.3f} nm / {rmse_y[MAX_PHOTONS]:.3f} nm")
print(f"Final sigma_loc:            {sigma_loc[MAX_PHOTONS]:.3f} nm")
print("=" * 72)

photon_numbers = np.arange(MAX_PHOTONS + 1)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

ax1.plot(photon_numbers[1:], sigma_loc[1:], color="black", linewidth=2, label=r"$\sigma_{loc}$")
ax1.plot(photon_numbers[1:], rmse_x[1:], color="tab:blue", alpha=0.8, label="RMSE X")
ax1.plot(photon_numbers[1:], rmse_y[1:], color="tab:orange", alpha=0.8, label="RMSE Y")
ax1.set_xlabel("Detected photon number N")
ax1.set_ylabel("Localization uncertainty (nm)")
ax1.set_title("RMS localization error versus detected photons")
ax1.grid(True, linestyle=":")
ax1.legend()

ax2.plot(photon_numbers[1:], success_count[1:], color="tab:green", linewidth=2)
ax2.set_xlabel("Detected photon number N")
ax2.set_ylabel("Successful localizations")
ax2.set_ylim(0, N_EXPERIMENTS * 1.05)
ax2.set_title("Successful runs after outlier filtering")
ax2.grid(True, linestyle=":")

ax3.hist(final_errors[final_success], bins=np.arange(0, 30, 0.5), color="tab:purple", alpha=0.8)
ax3.axvline(final_median_error, color="black", linestyle="--", label=f"median = {final_median_error:.2f} nm")
ax3.set_xlabel(f"Radial localization error at N={MAX_PHOTONS} (nm)")
ax3.set_ylabel("Frequency")
ax3.set_title("Final localization error distribution")
ax3.grid(True, linestyle=":")
ax3.legend()

plt.tight_layout()
plt.show()
