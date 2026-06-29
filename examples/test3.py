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


# 1. Create a static fluorophore at the true position [10 nm, 0 nm].
fl = FlStatic()
fl.pos = [10, 0]
fl.brightness = 1000  # kHz

# 2. Initialize the MINSTED tracker.
sim = MinstedSimulator(
    d0=250.0,
    alpha=0.15,
    gamma=0.97,
    r_min=5.0,
    d_min=40.0,
    f_scan=125.0,
)

# 3. Simulation settings.
dt = 0.001  # ms per simulation step
total_steps = 3000
current_time = 0.0

estimated_positions = [sim.get_localization_estimate()]
eod_positions = [sim.C_i.copy()]
true_positions = [fl.pos[:2].copy()]
total_detected_photons = 0

print("开始 MINSTED 静止分子定位追踪...")

for step in range(total_steps):
    mol_pos_3d, _ = fl.position(current_time)
    mol_pos_2d = mol_pos_3d[:2]

    # Effective emission coefficient under the current excitation + STED pattern.
    eff_psf_coeff = sim.get_effective_psf_coefficient(mol_pos_2d, time_ms=current_time)

    # Expected signal photons in this time bin. kHz * ms = photons.
    emitted_intensity = fl.intensity(
        I0=eff_psf_coeff,
        dwelltime=dt,
        time=current_time,
        phfac=1.0,
    )
    expected_photons = emitted_intensity + sim.bg_global * dt
    detected_photons = np.random.poisson(expected_photons)
    total_detected_photons += detected_photons

    if detected_photons > 0:
        s_i_snap = sim.get_current_eod_position(current_time)
        for _ in range(detected_photons):
            sim.register_one_photon(s_i=s_i_snap, origin="SIGNAL")
            eod_positions.append(sim.C_i.copy())
            true_positions.append(fl.pos[:2].copy())
            estimated_positions.append(sim.get_localization_estimate())

    # Important: one record and one time advance per simulation step, not per photon.
    current_time += dt
    # eod_positions.append(sim.C_i.copy())
    # true_positions.append(fl.pos[:2].copy())
    # estimated_positions.append(sim.get_localization_estimate())

print(f"定位结束！分子真实位置: {fl.pos[:2]}, 算法最终估计位置: {sim.get_localization_estimate()}")
print(
    f"Simulation steps: {total_steps}, "
    f"recorded points: {len(estimated_positions)}, "
    f"detected photons: {total_detected_photons}, "
    f"simulated time: {current_time:.3f} ms"
)

# 4. Plot localization convergence.
est_pos = np.array(eod_positions)
true_pos = np.array(true_positions)

plt.figure(figsize=(10, 4))
plt.plot(est_pos[:, 0], label="Estimated X")
plt.plot(true_pos[:, 0], color="r", linestyle="--", label="True X")
plt.plot(est_pos[:, 1], label="Estimated Y")
plt.plot(true_pos[:, 1], color="g", linestyle="--", label="True Y")
plt.xlabel("Simulation Steps")
plt.ylabel("Position (nm)")
plt.title("MINSTED Localization Convergence for Static Fluorophore")
plt.legend()
plt.grid(True)
if os.environ.get("MINSTED_NO_PLOT") == "1":
    plt.close()
else:
    plt.show()