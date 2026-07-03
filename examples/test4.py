import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
    }
)


def localization_precision_over_d(r_over_d, sbr, n_detections=100):
    """Approximate MINSTED Fig. 2a localization precision.

    The scan positions lie on a circle with radius R around the current
    molecular estimate. For a Gaussian E-PSF with FWHM d, the per-photon
    Fisher information scales with R/d in the no-background limit. Finite SBR
    reduces that information because only the signal-dependent part of the
    detection rate carries position information; background detections are
    phase-uniform.
    """
    r_over_d = np.asarray(r_over_d, dtype=float)
    gaussian_fwhm_factor = 4.0 * np.log(2.0)

    infinite_sbr_sigma_over_d = 1.0 / (
        np.sqrt(2.0 * n_detections) * gaussian_fwhm_factor * r_over_d
    )

    if np.isinf(sbr):
        return infinite_sbr_sigma_over_d

    signal_at_scan_radius = np.exp(-gaussian_fwhm_factor * r_over_d**2)
    information_fraction = (sbr * signal_at_scan_radius) / (
        1.0 + sbr * signal_at_scan_radius
    )

    return infinite_sbr_sigma_over_d / np.maximum(information_fraction, 1e-12)


def make_panel_a():
    n_detections = 100
    r_over_d = np.linspace(0.25, 2.0, 700)
    sbr_values = [np.inf, 50, 20, 10, 5, 2]
    colors = ["black", "#1f77b4", "#ff7f0e", "#f2b000", "#7b3294", "#6ba43a"]
    labels = [r"$\infty$", "50", "20", "10", "5", "2"]

    fig, ax = plt.subplots(figsize=(2.9, 2.25))

    for sbr, color, label in zip(sbr_values, colors, labels):
        sigma_over_d = localization_precision_over_d(
            r_over_d,
            sbr,
            n_detections=n_detections,
        )
        ax.plot(r_over_d, sigma_over_d, color=color, linewidth=1.0, label=label)

    fig.text(0.035, 0.93, "a", fontsize=8, fontweight="bold")
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 0.25)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["", "0.5", "1.0", "1.5", "2.0"])
    ax.set_yticks(np.arange(0.0, 0.251, 0.05))
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
        fontsize=7,
        title_fontsize=7,
    )
    legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.96)
    output_path = OUTPUT_DIR / "test4_fig2a_localization_precision.png"
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure: {output_path}")

    if os.environ.get("MINSTED_NO_PLOT") == "1":
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    make_panel_a()
