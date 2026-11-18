"""Sequential Bayesian experimental design for discrete MINSTED exposures.

Each exposure holds a Gaussian E-PSF at one position and width. This is a
different acquisition protocol from the continuous circular, per-photon tracker.
Positions are in nm, durations in ms, rates in kHz, and information in nats.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import poisson

from .gaussian_epsf import gaussian_epsf


@dataclass(frozen=True)
class MinstedExposure:
    position_nm: tuple[float, float]
    fwhm_nm: float
    dwell_ms: float


class BayesianMinstedSimulator:
    """Grid posterior and one-step EIG selection of position and E-PSF width.

    Brightness/background are assumed known; the emitter is static. The
    optimizer is restricted to a configurable finite set of candidate actions.
    The objective is information per fixed-duration exposure, not information
    per photon or per unit STED energy. No alpha/gamma feedback rule is used.
    """

    def __init__(
        self,
        prior_mean=(0.0, 0.0),
        prior_sigma=60.0,
        brightness: float = 30.0,
        bg_global: float = 3.0,
        bg_map_func: Callable[[float, float], float] | None = None,
        d0: float = 250.0,
        d_min: float = 40.0,
        dwell_ms: float = 0.05,
        grid_size: int = 65,
        prior_extent_sigma: float = 5.0,
        width_levels: int = 5,
        radius_factors: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
        n_angles: int = 8,
        poisson_tail: float = 1e-10,
        adaptive_grid: bool = True,
        grid_tail_mass: float = 1e-8,
    ):
        values = [brightness, bg_global, d0, d_min, dwell_ms, prior_extent_sigma]
        if not np.all(np.isfinite(values)):
            raise ValueError("Rates, widths and durations must be finite.")
        if brightness < 0 or bg_global < 0:
            raise ValueError("Photon rates must be nonnegative.")
        if not 0 < d_min <= d0 or dwell_ms <= 0 or prior_extent_sigma <= 0:
            raise ValueError("Require 0 < d_min <= d0 and positive durations/extents.")
        for name, value, minimum in [
            ("grid_size", grid_size, 17),
            ("width_levels", width_levels, 2),
            ("n_angles", n_angles, 4),
        ]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        factors = np.asarray(radius_factors, dtype=float)
        if factors.ndim != 1 or factors.size == 0 or not np.all(np.isfinite(factors)) or np.any(factors < 0):
            raise ValueError("radius_factors must be a nonempty nonnegative sequence.")
        if not 0 < poisson_tail < 1e-2 or not 0 < grid_tail_mass < 1e-2:
            raise ValueError("Tail tolerances must lie strictly between 0 and .01.")
        self.brightness = float(brightness)
        self.bg_global = float(bg_global)
        self.bg_map_func = bg_map_func
        self.d0 = float(d0)
        self.d_min = float(d_min)
        self.dwell_ms = float(dwell_ms)
        self.grid_size = int(grid_size)
        self.prior_extent_sigma = float(prior_extent_sigma)
        self.widths = np.unique(np.geomspace(d_min, d0, width_levels))
        self.radius_factors = np.unique(factors)
        angles = np.arange(n_angles) * (2.0 * np.pi / n_angles)
        self.directions = np.column_stack((np.cos(angles), np.sin(angles)))
        self.poisson_tail = float(poisson_tail)
        self.adaptive_grid = bool(adaptive_grid)
        self.grid_tail_mass = float(grid_tail_mass)
        self._initial_mean = self._position(prior_mean)
        sigma = np.broadcast_to(np.asarray(prior_sigma, dtype=float), (2,)).copy()
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("prior_sigma must contain positive finite standard deviations.")
        self._initial_sigma = sigma
        self.reset_tracker()

    @staticmethod
    def _position(position):
        point = np.asarray(position, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("A position must contain two finite coordinates.")
        return point.copy()

    def _refresh_points(self):
        xx, yy = np.meshgrid(self.x_axis, self.y_axis)
        self._points = np.column_stack((xx.ravel(), yy.ravel()))

    def reset_tracker(self, r0_est=None):
        """Reset the Gaussian prior and all acquisition records."""
        mean = self._initial_mean if r0_est is None else self._position(r0_est)
        extent = self.prior_extent_sigma * self._initial_sigma
        self.x_axis = np.linspace(mean[0] - extent[0], mean[0] + extent[0], self.grid_size)
        self.y_axis = np.linspace(mean[1] - extent[1], mean[1] + extent[1], self.grid_size)
        self._refresh_points()
        log_prior = -0.5 * np.sum(((self._points - mean) / self._initial_sigma) ** 2, axis=1)
        self.posterior = np.exp(log_prior - logsumexp(log_prior)).reshape(self.grid_size, self.grid_size)
        self.time_ms = 0.0
        self.photon_count = 0
        self.history = []
        self.estimate_trajectory = [self.get_localization_estimate()]
        self.last_candidates = ()
        self.last_eig = np.empty(0)
        self.total_pruned_mass = 0.0

    def get_localization_estimate(self):
        """Posterior mean, the Bayes estimator for squared position error."""
        return np.sum(self._points * self.posterior.ravel()[:, None], axis=0)

    def get_map_estimate(self):
        return self._points[np.argmax(self.posterior)].copy()

    def get_covariance(self):
        delta = self._points - self.get_localization_estimate()
        return (delta * self.posterior.ravel()[:, None]).T @ delta

    def get_uncertainty(self):
        """Equivalent single-axis posterior standard deviation in nm."""
        return float(np.sqrt(np.trace(self.get_covariance()) / 2.0))

    def get_history(self):
        return np.array(self.estimate_trajectory)

    def _validate_action(self, action):
        if not isinstance(action, MinstedExposure):
            raise TypeError("An exposure must be a MinstedExposure.")
        self._position(action.position_nm)
        if not np.isfinite(action.fwhm_nm) or not self.d_min <= action.fwhm_nm <= self.d0:
            raise ValueError("Exposure FWHM must be within [d_min, d0].")
        if not np.isfinite(action.dwell_ms) or action.dwell_ms <= 0:
            raise ValueError("Exposure duration must be positive and finite.")

    def _background(self, position):
        rate = self.bg_global if self.bg_map_func is None else float(self.bg_map_func(*position))
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("The background map must return finite nonnegative rates.")
        return rate

    def expected_counts(self, positions, action):
        """Poisson mean for candidate emitter coordinates (..., 2).

        This also supplies the forward model to an external synthetic detector;
        observe() and select_next_exposure() do not receive the true position.
        """
        self._validate_action(action)
        positions = np.asarray(positions, dtype=float)
        if positions.ndim < 1 or positions.shape[-1] != 2 or not np.all(np.isfinite(positions)):
            raise ValueError("Emitter coordinates must be finite and have final dimension 2.")
        distance_sq = np.sum((positions - np.asarray(action.position_nm)) ** 2, axis=-1)
        rate = self.brightness * gaussian_epsf(distance_sq, action.fwhm_nm)
        return action.dwell_ms * (rate + self._background(action.position_nm))

    def candidate_exposures(self, dwell_ms=None):
        """Positions on candidate rings about the posterior mean, for each width.

        The ring radius is an offset for one stationary exposure, not the radius
        of a continuously executed circular scan. It is chosen jointly with its
        direction and FWHM. Width is allowed to increase if evidence favors it.
        """
        dwell = self.dwell_ms if dwell_ms is None else float(dwell_ms)
        center = self.get_localization_estimate()
        actions = []
        for width in self.widths:
            for factor in self.radius_factors:
                offsets = np.zeros((1, 2)) if factor == 0 else factor * width * self.directions
                for position in center + offsets:
                    action = MinstedExposure(tuple(float(v) for v in position), float(width), dwell)
                    self._validate_action(action)
                    actions.append(action)
        return tuple(actions)

    def expected_information_gain(self, actions):
        """Mutual information I(emitter position; photon count), in nats.

        Includes n=0. A Poisson tail bound is applied using the largest mean
        over ALL grid positions/actions, rather than only a predictive mean.
        Actions must have equal durations, so their scores compare time costs.
        """
        actions = tuple(actions)
        if not actions:
            raise ValueError("At least one candidate exposure is required.")
        for action in actions:
            self._validate_action(action)
        dwell = actions[0].dwell_ms
        if any(action.dwell_ms != dwell for action in actions):
            raise ValueError("EIG comparisons require equal exposure durations.")
        eig = np.empty(len(actions))
        weights = self.posterior.ravel()
        # Bound working memory when the user supplies many candidate actions.
        for start in range(0, len(actions), 32):
            batch = actions[start:start + 32]
            means = np.array([self.expected_counts(self._points, action) for action in batch])
            max_mean = float(np.max(means))
            max_count = int(poisson.isf(self.poisson_tail, max_mean)) if max_mean > 0 else 0
            score = np.zeros(len(batch))
            # Log evaluation remains stable when exp(-mean) would underflow.
            with np.errstate(divide="ignore"):
                log_means = np.log(means)
            for count in range(max_count + 1):
                log_likelihood = -means - gammaln(count + 1)
                if count:
                    log_likelihood = log_likelihood + count * log_means
                likelihood = np.exp(log_likelihood)
                predictive = np.sum(likelihood * weights, axis=1)
                score += np.sum(xlogy(likelihood, likelihood) * weights, axis=1)
                score -= xlogy(predictive, predictive)
            eig[start:start + len(batch)] = np.maximum(score, 0.0)
        return eig

    def select_next_exposure(self, dwell_ms=None):
        """Choose the highest-EIG action in the finite candidate set."""
        self.last_candidates = self.candidate_exposures(dwell_ms)
        self.last_eig = self.expected_information_gain(self.last_candidates)
        return self.last_candidates[int(np.argmax(self.last_eig))]

    def observe(self, photon_count, action):
        """Update from one complete exposure, including a zero-photon exposure."""
        self._validate_action(action)
        if isinstance(photon_count, bool) or not isinstance(photon_count, (int, np.integer)) or photon_count < 0:
            raise ValueError("photon_count must be a nonnegative integer.")
        old_center = self.get_localization_estimate()
        means = self.expected_counts(self._points, action)
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.posterior.ravel())
        log_weights += xlogy(photon_count, means) - means - gammaln(photon_count + 1)
        normalizer = logsumexp(log_weights)
        if not np.isfinite(normalizer):
            raise ValueError("The observation has zero probability under the model/prior.")
        self.posterior = np.exp(log_weights - normalizer).reshape(self.posterior.shape)
        if self.adaptive_grid:
            self._refine_grid()
        center = self.get_localization_estimate()
        self.photon_count += int(photon_count)
        self.time_ms += action.dwell_ms
        self.estimate_trajectory.append(center.copy())
        self.history.append({
            "time_ms": self.time_ms,
            "photon_count": int(photon_count),
            "cumulative_photons": self.photon_count,
            "position_nm": action.position_nm,
            "fwhm_nm": action.fwhm_nm,
            "dwell_ms": action.dwell_ms,
            "offset_radius_nm": float(np.linalg.norm(np.asarray(action.position_nm) - old_center)),
            "estimate_nm": tuple(center),
            "update_nm": tuple(center - old_center),
            "update_length_nm": float(np.linalg.norm(center - old_center)),
            "posterior_sigma_nm": self.get_uncertainty(),
        })
        return center.copy()

    def _refine_grid(self):
        """Crop negligible marginal tails and interpolate on a finer grid.

        Refinement only occurs when an axis can shrink by at least 20%, avoiding
        repeated interpolation at every count. Tails outside the retained box
        are bounded by the configured probability mass on the current grid.
        This cannot recover probability outside the initial finite prior domain.
        """
        old_x, old_y = self.x_axis, self.y_axis
        axes = []
        masks = []
        for axis, marginal in [(old_x, self.posterior.sum(axis=0)), (old_y, self.posterior.sum(axis=1))]:
            cumulative = np.cumsum(marginal)
            tail = self.grid_tail_mass / 4.0
            lo = max(0, int(np.searchsorted(cumulative, tail)) - 1)
            hi = min(len(axis) - 1, int(np.searchsorted(cumulative, 1.0 - tail)) + 1)
            if hi > lo and axis[hi] - axis[lo] <= 0.8 * (axis[-1] - axis[0]):
                new_axis = np.linspace(axis[lo], axis[hi], self.grid_size)
                mask = (axis >= axis[lo]) & (axis <= axis[hi])
            else:
                new_axis, mask = axis, np.ones(len(axis), dtype=bool)
            axes.append(new_axis)
            masks.append(mask)
        if axes[0] is old_x and axes[1] is old_y:
            return
        retained_mass = self.posterior[np.ix_(masks[1], masks[0])].sum()
        interpolation = RegularGridInterpolator((old_y, old_x), self.posterior, bounds_error=True)
        xx, yy = np.meshgrid(axes[0], axes[1])
        refined = interpolation(np.column_stack((yy.ravel(), xx.ravel()))).reshape(self.posterior.shape)
        mass = refined.sum()
        if not np.isfinite(mass) or mass <= 0:
            raise RuntimeError("Posterior regridding failed to preserve positive probability mass.")
        self.total_pruned_mass += max(0.0, float(1.0 - retained_mass))
        self.x_axis, self.y_axis = axes
        self.posterior = refined / mass
        self._refresh_points()
