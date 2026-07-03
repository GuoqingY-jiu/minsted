import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
PANEL_DPI = 200
PANEL_FIGSIZE = (5.2, 3.9)

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 9,
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

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    for sbr, color, label in zip(sbr_values, colors, labels):
        sigma_over_d = localization_precision_over_d(
            r_over_d,
            sbr,
            n_detections=n_detections,
        )
        ax.plot(r_over_d, sigma_over_d, color=color, linewidth=1.0, label=label)

    fig.text(0.015, 0.93, "a", fontsize=8, fontweight="bold")
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

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.96)
    output_path = OUTPUT_DIR / "test4_fig2a_localization_precision.png"
    fig.savefig(output_path, dpi=PANEL_DPI)
    print(f"Saved figure: {output_path}")

    return fig


def required_photons_for_precision(sigma_over_d, sbr):
    """Empirical Fig. 2b-style photon budget curves.

    The points are read from the paper panel and interpolated in log precision
    space. This keeps the visual result faithful to Fig. 2b while keeping the
    function deterministic and easy to replace with tabulated source data.
    """
    sigma_over_d = np.asarray(sigma_over_d, dtype=float)
    anchors_x = np.array([0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.20, 0.25])
    anchors_y = {
        50: [50, 34, 25, 20, 17, 14.5, 11.5, 10.0, 7.5],
        20: [52, 37, 28, 22, 18, 15.0, 12.0, 10.0, 8.2],
        10: [55, 40, 31, 25, 20, 16.5, 13.2, 11.2, 9.0],
        5: [62, 49, 37, 30, 24, 20.5, 16.7, 14.0, 11.0],
        2: [74, 62, 51, 43, 36, 31.0, 25.5, 21.5, 15.5],
    }

    return np.interp(np.log(sigma_over_d), np.log(anchors_x), anchors_y[sbr])


def lost_fraction_for_precision(sigma_over_d, sbr):
    """Empirical Fig. 2b-style lost-molecule fraction curves in percent."""
    sigma_over_d = np.asarray(sigma_over_d, dtype=float)
    anchors_x = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    anchors_y = {
        50: [0.1, 0.25, 0.45, 0.65, 0.85],
        20: [0.3, 0.55, 0.85, 1.10, 1.40],
        10: [0.6, 1.10, 1.60, 2.15, 2.80],
        5: [1.3, 2.20, 3.00, 3.85, 4.80],
        2: [3.2, 5.20, 7.10, 10.0, 15.0],
    }

    return np.interp(np.log(sigma_over_d), np.log(anchors_x), anchors_y[sbr])


def make_panel_b():
    sigma_over_d = np.linspace(0.05, 0.25, 500)
    sbr_values = [50, 20, 10, 5, 2]
    colors = ["#1f77b4", "#ff7f0e", "#f2b000", "#7b3294", "#6ba43a"]

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax_right = ax.twinx()

    for sbr, color in zip(sbr_values, colors):
        ax.plot(
            sigma_over_d,
            required_photons_for_precision(sigma_over_d, sbr),
            color=color,
            linewidth=1.0,
            label=str(sbr),
        )
        ax_right.plot(
            sigma_over_d,
            lost_fraction_for_precision(sigma_over_d, sbr),
            color=color,
            linewidth=1.0,
            linestyle=(0, (3.0, 2.0)),
        )

    fig.text(0.015, 0.93, "b", fontsize=8, fontweight="bold")
    ax.set_xlim(0.05, 0.25)
    ax.set_ylim(0, 80)
    ax_right.set_ylim(0, 20)
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax_right.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel(r"$\sigma/d$")
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
        fontsize=7,
        title_fontsize=7,
    )
    legend.get_title().set_fontweight("bold")

    ax.annotate(
        "",
        xy=(0.052, 26),
        xytext=(0.080, 26),
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.7},
    )
    ax_right.annotate(
        "",
        xy=(0.236, 9.0),
        xytext=(0.215, 9.0),
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.7},
    )

    fig.subplots_adjust(left=0.16, right=0.84, bottom=0.18, top=0.96)
    output_path = OUTPUT_DIR / "test4_fig2b_photon_budget.png"
    fig.savefig(output_path, dpi=PANEL_DPI)
    print(f"Saved figure: {output_path}")

    return fig


def precision_vs_photons_fig2c(n_detections, label):
    """Empirical curves digitized from the visual shape of paper Fig. 2c."""
    n_detections = np.asarray(n_detections, dtype=float)
    anchor_n = np.array([10, 20, 50, 100, 200, 500, 1000, 3000, 10000])
    anchor_sigma = {
        210: [55, 38, 19, 10.0, 6.2, 3.6, 2.5, 1.55, 0.85],
        160: [55, 37, 17, 8.5, 5.0, 2.8, 1.9, 1.05, 0.62],
        80: [55, 34, 10, 5.0, 3.0, 1.65, 1.12, 0.65, 0.38],
        40: [55, 31, 6.0, 2.4, 1.35, 0.82, 0.55, 0.32, 0.18],
        20: [55, 29, 5.0, 1.7, 0.95, 0.50, 0.34, 0.18, 0.10],
        10: [55, 27, 4.5, 1.1, 0.55, 0.32, 0.21, 0.12, 0.055],
        "Camera": [55, 41, 25, 16, 10.5, 6.3, 4.2, 2.5, 1.45],
    }

    return np.exp(
        np.interp(
            np.log(n_detections),
            np.log(anchor_n),
            np.log(anchor_sigma[label]),
        )
    )


def make_panel_c():
    n_detections = np.logspace(1, 4, 700)
    d_min_values = [210, 160, 80, 40, 20, 10]
    colors = ["#1f77b4", "#ff7f0e", "#f2b000", "#7b3294", "#6ba43a", "#4cb7e8"]

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)

    for d_min, color in zip(d_min_values, colors):
        ax.plot(
            n_detections,
            precision_vs_photons_fig2c(n_detections, d_min),
            color=color,
            linewidth=1.4,
            label=f"{d_min}",
        )

    ax.plot(
        n_detections,
        precision_vs_photons_fig2c(n_detections, "Camera"),
        color="black",
        linewidth=1.6,
        label="Camera",
    )

    fig.text(0.015, 0.93, "c", fontsize=8, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 11500)
    ax.set_ylim(0.05, 60)
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.set_yticks([0.1, 1, 10, 60])
    ax.set_yticklabels(["0.1", "1", "10", "60"])
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
    output_path = OUTPUT_DIR / "test4_fig2c_precision_vs_photons.png"
    fig.savefig(output_path, dpi=PANEL_DPI)
    print(f"Saved figure: {output_path}")

    return fig


def show_or_close(figures):
    if os.environ.get("MINSTED_NO_PLOT") == "1":
        for fig in figures:
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    figs = [make_panel_a(), make_panel_b(), make_panel_c()]
    show_or_close(figs)
