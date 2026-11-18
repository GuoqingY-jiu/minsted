import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import xlogy

from simulators import CircularBayesianMinsted, CircularScan
from simulators.CircularBayesianMinsted import circular_rate, next_circular_photon


class CircularBayesianTests(unittest.TestCase):
    def make_sim(self, **kwargs):
        options = dict(grid_size=33, adaptive_grid=False, planning_grid=9, width_levels=3)
        options.update(kwargs)
        return CircularBayesianMinsted(**options)

    def test_integral_matches_independent_continuous_quadrature(self):
        sim = self.make_sim(quadrature_order=32)
        scan = CircularScan((10., -8.), 100., 40.)
        for point in [np.array([10., 0.]), np.array([100., -50.]), np.array([-200., 80.])]:
            for start, end in [(.0013, .0041), (.0013, .0837), (0., .08)]:
                reference = quad(lambda t: circular_rate(point, scan, t), start, end, epsabs=1e-10, limit=200)[0]
                self.assertAlmostEqual(float(sim.integrated_rate(point, scan, start, end)), reference, places=8)

    def test_integral_partition_and_long_wait(self):
        sim = self.make_sim(quadrature_order=48)
        scan = CircularScan((12., 20.), 125., 40.)
        points = np.array([[0., 0.], [100., 30.], [-12., 250.]])
        whole = sim.integrated_rate(points, scan, .001, 3.199)
        pieces = sim.integrated_rate(points, scan, .001, .004) + sim.integrated_rate(points, scan, .004, 3.199)
        np.testing.assert_allclose(whole, pieces, rtol=1e-10, atol=1e-10)

    def test_event_and_survival_likelihood_match_direct_product(self):
        sim = self.make_sim()
        initial = sim.posterior.copy().ravel()
        expected = initial.copy()
        start = 0.
        observations = [(.023, CircularScan((20., 10.), 80., 160.), True),
                        (.052, CircularScan((-8., 7.), 20., 40.), True),
                        (.06, CircularScan((1., 4.), 20., 40.), False)]
        for end, scan, event in observations:
            expected *= np.exp(-sim.integrated_rate(sim._points, scan, start, end))
            if event:
                expected *= circular_rate(sim._points, scan, end)
            expected /= expected.sum()
            sim.observe_scan_interval(scan, end, event)
            np.testing.assert_allclose(sim.posterior.ravel(), expected, atol=1e-14)
            start = end
        self.assertEqual(sim.photon_count, 2)
        self.assertEqual(sim.time_ms, .06)

    def test_final_silent_interval_is_included_without_mutation(self):
        sim = self.make_sim()
        scan = CircularScan((40., 0.), 10., 100.)
        sim.observe_scan_interval(scan, .01)
        before = sim.posterior.copy()
        old_mean = sim.get_localization_estimate()
        preview, covariance = sim.preview_no_photon(scan, .2)
        np.testing.assert_array_equal(sim.posterior, before)
        self.assertEqual(sim.time_ms, .01)
        self.assertGreater(np.linalg.norm(preview - old_mean), .1)
        sim.observe_scan_interval(scan, .2, photon=False)
        np.testing.assert_allclose(preview, sim.get_localization_estimate(), atol=1e-12)
        np.testing.assert_allclose(covariance, sim.get_covariance(), atol=1e-12)

    def test_phase_binned_eig_matches_explicit_outcome_entropies(self):
        sim = self.make_sim(grid_size=17, planning_grid=17, phase_bins=8)
        scan = CircularScan((30., 0.), 40., 80.)
        p = sim.posterior.ravel()
        survival = np.ones_like(p)
        outcomes = []
        for i in range(sim.phase_bins):
            dt = scan.period_ms / sim.phase_bins
            rate = circular_rate(sim._points, scan, (i + .5) * dt)
            event = survival * (1 - np.exp(-rate * dt))
            outcomes.append(event)
            survival *= np.exp(-rate * dt)
        outcomes.append(survival)
        np.testing.assert_allclose(np.sum(outcomes, axis=0), 1., atol=1e-14)
        entropy_before = -np.sum(xlogy(p, p))
        entropy_after = 0.
        for likelihood in outcomes:
            joint = p * likelihood
            probability = joint.sum()
            posterior = joint / probability
            entropy_after += probability * -np.sum(xlogy(posterior, posterior))
        self.assertAlmostEqual(sim.first_arrival_eig([scan])[0], entropy_before - entropy_after, places=12)

    def test_no_signal_has_no_location_information(self):
        sim = self.make_sim(brightness=0.)
        scans = sim.circular_candidates()
        np.testing.assert_allclose(sim.first_arrival_eig(scans), 0., atol=1e-12)

    def test_compressed_planning_measure_preserves_mean(self):
        sim = self.make_sim()
        sim.observe_scan_interval(CircularScan((30., 0.), 50., 100.), .05)
        points, weights = sim._planning_distribution()
        np.testing.assert_allclose(np.sum(points * weights[:, None], axis=0), sim.get_localization_estimate(), atol=1e-12)
        self.assertAlmostEqual(weights.sum(), 1., places=13)

    def test_selected_circle_respects_hardware_limits(self):
        sim = self.make_sim()
        chosen = sim.select_next_scan()
        self.assertEqual(chosen, sim.last_candidates[int(np.argmax(sim.last_eig))])
        for scan in sim.last_candidates:
            self.assertTrue(5. <= scan.radius_nm <= 125.)
            self.assertTrue(40. <= scan.fwhm_nm <= 250.)
            self.assertEqual(scan.frequency_khz, 125.)

    def test_exact_thinning_has_exponential_integrated_hazard(self):
        sim = self.make_sim(quadrature_order=48)
        scan = CircularScan((0., 0.), 60., 100.)
        point = np.array([25., -10.])
        rng = np.random.default_rng(4817)
        hazards = []
        for _ in range(2000):
            time = next_circular_photon(scan, point, .0013, rng)
            hazards.append(float(sim.integrated_rate(point, scan, .0013, time)))
        self.assertLess(abs(np.mean(hazards) - 1.), .06)
        self.assertLess(abs(np.median(hazards) - np.log(2)), .06)


if __name__ == '__main__':
    unittest.main()
