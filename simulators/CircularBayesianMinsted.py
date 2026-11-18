"""Bayesian inference and first-arrival EIG for continuous circular MINSTED.

The likelihood uses photon times AND survival over the full preceding interval.
Planning uses a phase-binned, one-revolution censored first-arrival observation.
That finite-horizon, finite-action approximation is distinct from full-path EIG.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import i0e, logsumexp, xlogy

from .BayesianMinstedSimulator import BayesianMinstedSimulator
from .gaussian_epsf import gaussian_epsf


@dataclass(frozen=True)
class CircularScan:
    center_nm: tuple[float, float]
    radius_nm: float
    fwhm_nm: float
    frequency_khz: float = 125.0

    def __post_init__(self):
        values = np.asarray([*self.center_nm, self.radius_nm, self.fwhm_nm, self.frequency_khz], dtype=float)
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError("Circular scan parameters must be finite, with a 2D center.")
        if self.radius_nm < 0 or self.fwhm_nm <= 0 or self.frequency_khz <= 0:
            raise ValueError("Require radius >= 0 and positive FWHM/frequency.")

    @property
    def period_ms(self):
        return 1.0 / self.frequency_khz

    def positions(self, times_ms):
        phase = -2.0 * np.pi * self.frequency_khz * np.asarray(times_ms)
        return np.asarray(self.center_nm) + self.radius_nm * np.stack((np.cos(phase), np.sin(phase)), axis=-1)


def circular_rate(points, scan, times_ms, brightness=30.0, background=3.0):
    """Rates for points (..., 2) and scan times; use explicit broadcast axes."""
    distance_sq = np.sum((np.asarray(points) - scan.positions(times_ms)) ** 2, axis=-1)
    return brightness * gaussian_epsf(distance_sq, scan.fwhm_nm) + background


def next_circular_photon(scan, true_position, start_ms, rng, brightness=30.0, background=3.0):
    """Exact thinning for a static emitter with a uniform background.

    Rejected proposals are unobserved simulator internals. No phase grid or
    fixed time step is used to generate arrival times.
    """
    bound = brightness + background
    if not np.isfinite(bound) or brightness < 0 or background < 0:
        raise ValueError("Rates must be finite and nonnegative.")
    if bound == 0:
        return np.inf
    time = float(start_ms)
    while True:
        time += rng.exponential(1.0 / bound)
        rate = circular_rate(true_position, scan, time, brightness, background)
        if rng.random() * bound < rate:
            return time


class CircularBayesianMinsted(BayesianMinstedSimulator):
    """Continuous-scan posterior, plus a restricted circular-action planner.

    The emitter/brightness/background are static and known except for position.
    Controls stay fixed between photons; scan phase follows absolute time.
    Uniform background is required for analytic full-revolution integration.
    """

    def __init__(
        self, *args, quadrature_order=24, phase_bins=16, planning_grid=17,
        radius_factors=(.25, .5, .75), center_offset_sigma=.5,
        frequency_khz=125.0, max_radius_nm=125.0, min_radius_nm=5.0,
        refinement_candidates=12, refinement_grid=33, refinement_phase_bins=32,
        **kwargs,
    ):
        if kwargs.get("bg_map_func") is not None:
            raise ValueError("Circular inference currently requires a uniform background.")
        if quadrature_order < 8 or phase_bins < 4 or planning_grid < 3:
            raise ValueError("Require quadrature_order >= 8, phase_bins >= 4, planning_grid >= 3.")
        if not 0 < min_radius_nm <= max_radius_nm or frequency_khz <= 0:
            raise ValueError("Invalid circular scan radius/frequency limits.")
        if not np.isfinite(center_offset_sigma) or center_offset_sigma < 0:
            raise ValueError("center_offset_sigma must be finite and nonnegative.")
        super().__init__(*args, radius_factors=radius_factors, **kwargs)
        self.quadrature_order = int(quadrature_order)
        self.phase_bins = int(phase_bins)
        self.planning_grid = int(planning_grid)
        self.center_offset_sigma = float(center_offset_sigma)
        self.frequency_khz = float(frequency_khz)
        self.max_radius_nm = float(max_radius_nm)
        self.min_radius_nm = float(min_radius_nm)
        self.refinement_candidates = int(refinement_candidates)
        self.refinement_grid = int(refinement_grid)
        self.refinement_phase_bins = int(refinement_phase_bins)
        if self.refinement_candidates < 1 or self.refinement_grid < 3 or self.refinement_phase_bins < 4:
            raise ValueError("Invalid EIG refinement settings.")
        self._quad_nodes, self._quad_weights = leggauss(self.quadrature_order)

    def integrated_rate(self, points, scan, start_ms, end_ms):
        """Hazard: analytic full turns plus Gauss-Legendre residual arc."""
        if end_ms < start_ms or not np.all(np.isfinite([start_ms, end_ms])):
            raise ValueError("Require finite end_ms >= start_ms.")
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != 2:
            raise ValueError("Points must have final dimension 2.")
        duration = end_ms - start_ms
        if duration == 0:
            return np.zeros(points.shape[:-1])
        period = scan.period_ms
        full_turns = int(np.floor(duration / period))
        remainder = duration - full_turns * period
        distance = np.linalg.norm(points - np.asarray(scan.center_nm), axis=-1)
        a = 4.0 * np.log(2.0) / (scan.fwhm_nm**2 + 1e-12)
        signal_integral = np.zeros_like(distance)
        if full_turns:
            # exp(-a*(r^2+R^2))*I0(2*a*r*R), evaluated without overflow.
            mean_signal = self.brightness * np.exp(-a * (distance - scan.radius_nm)**2)
            mean_signal *= i0e(2.0 * a * distance * scan.radius_nm)
            signal_integral += full_turns * period * mean_signal
        if remainder > 1e-15:
            times = start_ms + .5 * remainder * (self._quad_nodes + 1.0)
            positions = scan.positions(times)
            dist_sq = np.sum((points[..., None, :] - positions) ** 2, axis=-1)
            signal_integral += .5 * remainder * self.brightness * np.sum(
                gaussian_epsf(dist_sq, scan.fwhm_nm) * self._quad_weights, axis=-1,
            )
        return signal_integral + self.bg_global * duration

    def _updated_weights(self, scan, end_ms, photon):
        if end_ms < self.time_ms:
            raise ValueError("Observations must be chronological.")
        log_likelihood = -self.integrated_rate(self._points, scan, self.time_ms, end_ms)
        if photon:
            rate = circular_rate(self._points, scan, end_ms, self.brightness, self.bg_global)
            with np.errstate(divide="ignore"):
                log_likelihood += np.log(rate)
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.posterior.ravel()) + log_likelihood
        log_z = logsumexp(log_weights)
        if not np.isfinite(log_z):
            raise ValueError("Observation has zero probability under the model/prior.")
        return np.exp(log_weights - log_z)

    def preview_no_photon(self, scan, end_ms):
        """Nonmutating fixed-time readout, including the final silent interval."""
        weights = self._updated_weights(scan, end_ms, False)
        mean = np.sum(weights[:, None] * self._points, axis=0)
        delta = self._points - mean
        covariance = (delta * weights[:, None]).T @ delta
        return mean, covariance

    def observe_scan_interval(self, scan, end_ms, photon=True):
        """Consume a no-photon interval, optionally ending in one photon."""
        old_center = self.get_localization_estimate()
        weights = self._updated_weights(scan, end_ms, photon)
        self.posterior = weights.reshape(self.posterior.shape)
        if self.adaptive_grid:
            self._refine_grid()
        self.time_ms = float(end_ms)
        self.photon_count += int(photon)
        mean = self.get_localization_estimate()
        self.estimate_trajectory.append(mean.copy())
        self.history.append({
            "time_ms": self.time_ms, "cumulative_photons": self.photon_count,
            "center_nm": scan.center_nm, "radius_nm": scan.radius_nm,
            "fwhm_nm": scan.fwhm_nm, "photon": bool(photon),
            "estimate_nm": tuple(mean), "update_nm": tuple(mean - old_center),
            "posterior_sigma_nm": self.get_uncertainty(),
        })
        return mean.copy()

    def _planning_distribution(self):
        """Probability-weighted cell centroids, preserving the posterior mean."""
        groups = min(self.planning_grid, self.grid_size)
        bounds = np.linspace(0, self.grid_size, groups + 1).astype(int)
        labels = np.empty(self.grid_size, dtype=int)
        for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            labels[lo:hi] = i
        ids = (labels[:, None] * groups + labels[None, :]).ravel()
        weights = self.posterior.ravel()
        mass = np.bincount(ids, weights=weights, minlength=groups**2)
        occupied = mass > 0
        centroids = np.column_stack([
            np.bincount(ids, weights=weights * self._points[:, dim], minlength=groups**2)[occupied]
            / mass[occupied] for dim in range(2)
        ])
        return centroids, mass[occupied]

    def circular_candidates(self, current_scan=None):
        mean = self.get_localization_estimate()
        eigenvalues, vectors = np.linalg.eigh(self.get_covariance())
        offsets = self.center_offset_sigma * np.sqrt(np.maximum(eigenvalues, 0))[:, None] * vectors.T
        centers = [mean]
        if self.center_offset_sigma:
            centers.extend(mean + offsets)
            centers.extend(mean - offsets)
        if current_scan is not None:
            centers.append(np.asarray(current_scan.center_nm))
        candidates = []
        seen = set()
        for center in centers:
            for width in self.widths:
                radii = np.unique(np.clip(self.radius_factors * width, self.min_radius_nm, self.max_radius_nm))
                for radius in radii:
                    key = (*np.round(center, 10), float(radius), float(width))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(CircularScan(tuple(float(v) for v in center), float(radius), float(width), self.frequency_khz))
        return tuple(candidates)

    def first_arrival_eig(self, scans, start_ms=None):
        """EIG of (no event, first-event phase bin) over ONE revolution.

        Within each bin the rate uses midpoint quadrature. For that piecewise
        constant rate, outcome probabilities sum to one including survival.
        The actual data likelihood uses continuous event times, not these bins.
        Grid compression and phase bins affect planning only.
        """
        scans = tuple(scans)
        if not scans:
            raise ValueError("At least one circular scan is required.")
        if any(scan.frequency_khz != self.frequency_khz for scan in scans):
            raise ValueError("All candidate scans must use the same frequency.")
        time = self.time_ms if start_ms is None else float(start_ms)
        points, weights = self._planning_distribution()
        period = 1.0 / self.frequency_khz
        bin_duration = period / self.phase_bins
        phase_times = time + (np.arange(self.phase_bins) + .5) * bin_duration
        scores = np.empty(len(scans))
        for start in range(0, len(scans), 24):
            batch = scans[start:start + 24]
            positions = np.array([scan.positions(phase_times) for scan in batch])
            dist_sq = np.sum((points[None, :, None, :] - positions[:, None, :, :])**2, axis=-1)
            widths = np.array([scan.fwhm_nm for scan in batch])[:, None, None]
            hazards = bin_duration * (self.brightness * gaussian_epsf(dist_sq, widths) + self.bg_global)
            cumulative = np.cumsum(hazards, axis=-1)
            first = np.exp(-(cumulative - hazards)) * -np.expm1(-hazards)
            none = np.exp(-cumulative[..., -1:])
            outcomes = np.concatenate((none, first), axis=-1)
            predictive = np.sum(outcomes * weights[None, :, None], axis=1)
            score = np.sum(xlogy(outcomes, outcomes) * weights[None, :, None], axis=(1, 2))
            score -= np.sum(xlogy(predictive, predictive), axis=1)
            scores[start:start + len(batch)] = np.maximum(score, 0.0)
        return scores

    def select_next_scan(self, current_scan=None):
        candidates = self.circular_candidates(current_scan)
        scores = self.first_arrival_eig(candidates)
        # Cell centroids can overvalue narrow beams for a broad posterior.
        # Refine leading candidates and the leading candidate for each width.
        if self.get_uncertainty() > self.d_min / 4.0:
            leading = set(np.argsort(scores)[-self.refinement_candidates:].tolist())
            for width in self.widths:
                indices = [i for i, candidate in enumerate(candidates) if candidate.fwhm_nm == width]
                leading.add(indices[int(np.argmax(scores[indices]))])
            candidates = tuple(candidates[i] for i in sorted(leading))
            old_grid, old_bins = self.planning_grid, self.phase_bins
            try:
                self.planning_grid = max(old_grid, self.refinement_grid)
                self.phase_bins = max(old_bins, self.refinement_phase_bins)
                scores = self.first_arrival_eig(candidates)
            finally:
                self.planning_grid, self.phase_bins = old_grid, old_bins
        self.last_candidates = candidates
        self.last_eig = scores
        return self.last_candidates[int(np.argmax(self.last_eig))]
