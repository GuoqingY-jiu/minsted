"""Compare Bayesian point exposures with the original circular MINSTED tracker.

Both use the same Gaussian E-PSF, brightness, background and acquisition time.
Photon counts and acquisition protocols differ; this is not a photon-matched
benchmark. Run with --help for settings. MINSTED_NO_PLOT=1 disables the window.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from minsted.simulators import BayesianMinstedSimulator, MinstedSimulator


def sample_circular_photon(sim, true_position, current_time, brightness, rng, n_phase=256):
    """Same integrated-hazard sampling as test1, for a static emitter."""
    period = 1.0 / sim.f_scan
    ticks = np.linspace(0., period, n_phase + 1)
    phase = -2. * np.pi * sim.f_scan * (current_time + ticks)
    positions = sim.C_i[:, None] + sim.R_i * np.vstack((np.cos(phase), np.sin(phase)))
    rates = sim.detection_rate(true_position[:, None], positions, brightness)
    cumulative = np.r_[0., np.cumsum((rates[:-1] + rates[1:]) * .5 * np.diff(ticks))]
    if cumulative[-1] <= 0:
        return np.inf, None
    threshold = rng.exponential()
    turns = int(threshold // cumulative[-1])
    arrival = current_time + turns * period + np.interp(threshold - turns * cumulative[-1], cumulative, ticks)
    return arrival, sim.get_current_eod_position(arrival)


def row(method, run, time_ms, photon_count, estimate, truth, **extra):
    result = dict(
        run=run,
        method=method,
        time_ms=time_ms,
        cumulative_photons=photon_count,
        estimate_x_nm=estimate[0],
        estimate_y_nm=estimate[1],
        true_x_nm=truth[0],
        true_y_nm=truth[1],
        radial_error_nm=float(np.linalg.norm(estimate - truth)),
        posterior_sigma_nm=np.nan,
        fwhm_nm=np.nan,
        offset_radius_nm=np.nan,
        update_length_nm=np.nan,
        planned_eig_nats=np.nan,
    )
    result.update(extra)
    return result


def run_comparison(args, run):
    truth = np.array(args.true_position, dtype=float)
    initial = np.array(args.initial_center, dtype=float)
    total_time = args.exposures * args.dwell_ms
    random_streams = np.random.SeedSequence([args.seed, run]).spawn(2)
    old_rng, new_rng = [np.random.default_rng(stream) for stream in random_streams]
    baseline = MinstedSimulator(d0=250., d_min=40., r_min=5., bg_global=args.background)
    baseline.reset_tracker(initial)
    baseline_rows = [row("original", run, 0., 0, initial, truth, fwhm_nm=baseline.d_i)]
    current_time = 0.
    start = perf_counter()
    while True:
        arrival, scan = sample_circular_photon(baseline, truth, current_time, args.brightness, old_rng)
        if arrival > total_time:
            break
        before = baseline.get_localization_estimate()
        width, radius = baseline.d_i, baseline.R_i
        baseline.register_one_photon(scan)
        estimate = baseline.get_localization_estimate()
        baseline_rows.append(row(
            "original", run, arrival, baseline.photon_count, estimate, truth,
            fwhm_nm=width, offset_radius_nm=radius,
            update_length_nm=float(np.linalg.norm(estimate - before)),
        ))
        current_time = arrival
    old_compute = perf_counter() - start
    baseline_rows.append(row(
        "original", run, total_time, baseline.photon_count,
        baseline.get_localization_estimate(), truth, fwhm_nm=baseline.d_i,
        offset_radius_nm=baseline.R_i,
    ))

    bayes = BayesianMinstedSimulator(
        prior_mean=initial, prior_sigma=args.prior_sigma,
        brightness=args.brightness, bg_global=args.background,
        dwell_ms=args.dwell_ms, grid_size=args.grid_size,
        width_levels=args.width_levels, n_angles=args.angles,
    )
    bayes_rows = [row("bayesian", run, 0., 0, initial, truth, posterior_sigma_nm=bayes.get_uncertainty())]
    start = perf_counter()
    for step in range(args.exposures):
        action = bayes.select_next_exposure()
        eig = float(bayes.last_eig.max())
        # Synthetic detector only: the estimator never receives truth or origins.
        count = int(new_rng.poisson(bayes.expected_counts(truth, action)))
        estimate = bayes.observe(count, action)
        record = bayes.history[-1]
        bayes_rows.append(row(
            "bayesian", run, bayes.time_ms, bayes.photon_count, estimate, truth,
            posterior_sigma_nm=record["posterior_sigma_nm"],
            fwhm_nm=record["fwhm_nm"], offset_radius_nm=record["offset_radius_nm"],
            update_length_nm=record["update_length_nm"], planned_eig_nats=eig,
        ))
        if (step + 1) % 40 == 0:
            print(f"  run {run + 1}: {step + 1}/{args.exposures} Bayesian exposures", flush=True)
    new_compute = perf_counter() - start
    print(
        f"Run {run + 1}, acquisition {total_time:g} ms | "
        f"original: N={baseline.photon_count}, error={baseline_rows[-1]['radial_error_nm']:.3f} nm, "
        f"compute={old_compute:.3f} s | "
        f"Bayesian: N={bayes.photon_count}, error={bayes_rows[-1]['radial_error_nm']:.3f} nm, "
        f"posterior sigma={bayes.get_uncertainty():.3f} nm, compute={new_compute:.3f} s",
        flush=True,
    )
    return baseline_rows, bayes_rows, bayes


def plot_comparison(original, bayesian, bayes, output_dir, no_plot):
    def values(records, key):
        return np.array([r[key] for r in records])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for records, label, color in [(original, "Original circular tracker", "tab:orange"), (bayesian, "Bayesian point exposures", "tab:blue")]:
        axes[0, 0].plot(values(records, "estimate_x_nm"), values(records, "estimate_y_nm"), color=color, label=label)
        axes[0, 1].plot(values(records, "time_ms"), values(records, "radial_error_nm"), color=color, label=label)
        axes[0, 2].plot(values(records, "cumulative_photons"), values(records, "radial_error_nm"), color=color, label=label)
        axes[1, 0].step(values(records, "time_ms"), values(records, "fwhm_nm"), where="pre", color=color, label=label)
    axes[0, 0].scatter(original[0]["true_x_nm"], original[0]["true_y_nm"], marker="*", s=100, color="black", label="True position")
    axes[0, 0].set(xlabel="x (nm)", ylabel="y (nm)", title="Position estimates")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 1].set(xlabel="Acquisition time (ms)", ylabel="Radial error (nm)", title="Equal acquisition duration")
    axes[0, 2].set(xlabel="Detected photons (including background)", ylabel="Radial error (nm)", title="Photon counts are not matched")
    axes[1, 0].set(xlabel="Acquisition time (ms)", ylabel="E-PSF FWHM (nm)", title="Selected width; no forced monotonic shrink")
    axes[1, 1].plot(values(bayesian, "time_ms"), values(bayesian, "offset_radius_nm"), label="Exposure offset from prior mean")
    axes[1, 1].plot(values(bayesian, "time_ms"), values(bayesian, "update_length_nm"), label="Posterior mean update length")
    axes[1, 1].set(xlabel="Acquisition time (ms)", ylabel="Distance (nm)", title="Bayesian scan and estimate updates")
    extent = [bayes.x_axis[0], bayes.x_axis[-1], bayes.y_axis[0], bayes.y_axis[-1]]
    image = axes[1, 2].imshow(bayes.posterior, origin="lower", extent=extent, aspect="equal", cmap="Blues")
    axes[1, 2].scatter(original[0]["true_x_nm"], original[0]["true_y_nm"], marker="*", color="black", label="True position")
    axes[1, 2].set(xlabel="x (nm)", ylabel="y (nm)", title=f"Final posterior; sigma={bayes.get_uncertainty():.2f} nm")
    fig.colorbar(image, ax=axes[1, 2], label="Probability per grid point")
    for ax in axes.ravel():
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle("Bayesian MINSTED prototype: first repeat, same Gaussian forward model")
    figure_path = output_dir / "bayesian_minsted_comparison.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Saved figure: {figure_path}")
    if no_plot:
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposures", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dwell-ms", type=float, default=.05)
    parser.add_argument("--grid-size", type=int, default=65)
    parser.add_argument("--angles", type=int, default=8)
    parser.add_argument("--width-levels", type=int, default=5)
    parser.add_argument("--brightness", type=float, default=30.)
    parser.add_argument("--background", type=float, default=3.)
    parser.add_argument("--prior-sigma", type=float, default=60.)
    parser.add_argument("--true-position", nargs=2, type=float, default=[10., -5.])
    parser.add_argument("--initial-center", nargs=2, type=float, default=[-50., 40.])
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.exposures < 1 or args.repeats < 1:
        parser.error("exposures and repeats must be positive")
    # Validate before running either simulation, including the zero-rate case.
    BayesianMinstedSimulator(
        prior_mean=args.initial_center, prior_sigma=args.prior_sigma,
        brightness=args.brightness, bg_global=args.background, dwell_ms=args.dwell_ms,
        grid_size=args.grid_size, n_angles=args.angles, width_levels=args.width_levels,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    final_errors = {"original": [], "bayesian": []}
    first = None
    for run in range(args.repeats):
        original, bayesian, bayes = run_comparison(args, run)
        all_rows.extend(original + bayesian)
        for records in [original, bayesian]:
            final_errors[records[-1]["method"]].append(records[-1]["radial_error_nm"])
        if first is None:
            first = original, bayesian, bayes
    for method, errors in final_errors.items():
        errors = np.array(errors)
        print(f"{method}: all {len(errors)} repeats retained; equivalent single-axis RMSE={np.sqrt(np.mean(errors**2)/2):.3f} nm; median error/sqrt(2)={np.median(errors)/np.sqrt(2):.3f} nm")
    csv_path = args.output_dir / "bayesian_minsted_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved data: {csv_path}")
    plot_comparison(*first, args.output_dir, args.no_plot or os.environ.get("MINSTED_NO_PLOT") == "1")


if __name__ == "__main__":
    main()
