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

    # Trapezoidal cumulative integral. Rates are kHz and time is ms, so the
    # integral is expected detected photons.
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


# np.random.seed(1)

fl = FlStatic()
fl.pos = [10.0, 0.0]
fl.brightness = 30.0  # kHz, detected signal brightness scale

sim = MinstedSimulator(
    d0=250.0,
    alpha=0.15,
    gamma=0.97,
    r_min=5.0,
    d_min=40.0,
    bg_global=3.0,
    f_scan=125.0,
)

max_photons = 300
current_time = 0.0
initial_c0 = np.random.normal(loc=0.0, scale=60.0, size=2)
sim.reset_tracker(r0_est=initial_c0)

photon_arrival_times = []
photon_origins = []
signal_rates = []
background_rates = []
total_rates = []

print("Running one MINSTED trace...")

for _ in range(max_photons):
    (
        current_time,
        s_i_actual,
        origin,
        signal_i,
        background_i,
        total_i,
    ) = sample_next_photon(sim, fl, current_time)

    sim.register_one_photon(s_i=s_i_actual, origin=origin)

    photon_arrival_times.append(current_time)
    photon_origins.append(origin)
    signal_rates.append(signal_i)
    background_rates.append(background_i)
    total_rates.append(total_i)

    if hasattr(fl, "remainingphotons") and fl.remainingphotons <= 0:
        break

# est_pos = sim.get_history()
est_pos = sim.get_center_history()

if sim.Nc_reached and len(sim.history_after_Nc) > 1:
    stable_trace = np.array(sim.history_after_Nc)
    start_stable_idx = len(est_pos) - len(stable_trace)
else:
    start_stable_idx = int(len(est_pos) * 0.5)
    stable_trace = est_pos[start_stable_idx:]

total_std_x = np.std(stable_trace[:, 0], ddof=1)
total_std_y = np.std(stable_trace[:, 1], ddof=1)
final_estimated_pos = sim.get_localization_estimate()
background_count = photon_origins.count("BACKGROUND")

print()
print("=" * 64)
print(f"Initial estimate: {initial_c0}")
print(f"True position:    {fl.pos[:2]}")
print(f"Final pos:   {final_estimated_pos}")
print(f"Stable photons:   {len(stable_trace)} from photon index {start_stable_idx}")
print(f"Stable STD x/y:   {total_std_x:.3f} nm / {total_std_y:.3f} nm")
print(f"Mean signal rate: {np.mean(signal_rates):.3f} kHz")
print(f"Mean bg rate:     {np.mean(background_rates):.3f} kHz")
print(f"Mean total rate:  {np.mean(total_rates):.3f} kHz")
print(f"Background hits:  {background_count}/{len(photon_origins)}")
print("=" * 64)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(est_pos[:, 0], label="Localization Center X", color="blue", alpha=0.8)
ax1.axhline(y=fl.pos[0], color="red", linestyle="--", label="True X")
ax1.plot(est_pos[:, 1], label="Localization Center Y", color="teal", alpha=0.8)
ax1.axhline(y=fl.pos[1], color="green", linestyle="--", label="True Y")
ax1.axvspan(
    start_stable_idx,
    len(est_pos),
    color="khaki",
    alpha=0.3,
    label=f"Stable phase, std=({total_std_x:.1f}, {total_std_y:.1f}) nm",
)
ax1.set_xlabel("Detected photon number")
ax1.set_ylabel("Position (nm)")
ax1.set_title("MINSTED localization Center trace")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle=":")

ax2.set_xlabel("Detected photon number")
ax2.set_ylabel("Arrival time (ms)", color="tab:orange")
ax2.plot(photon_arrival_times, color="tab:orange", linewidth=2)
ax2.tick_params(axis="y", labelcolor="tab:orange")

ax2_twin = ax2.twinx()
time_intervals_us = np.diff(np.insert(photon_arrival_times, 0, 0.0)) * 1000.0
ax2_twin.set_ylabel("Inter-photon interval (us)", color="tab:blue")
ax2_twin.plot(time_intervals_us, ".", color="tab:blue", alpha=0.4)
ax2_twin.tick_params(axis="y", labelcolor="tab:blue")
ax2.grid(True, linestyle=":")

plt.tight_layout()
plt.show()
