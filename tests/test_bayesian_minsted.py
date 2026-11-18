import unittest

import numpy as np
from scipy.special import gammaln, xlogy
from scipy.stats import poisson

from simulators import BayesianMinstedSimulator, MinstedExposure, MinstedSimulator


class BayesianMinstedTests(unittest.TestCase):
    def make_sim(self, **kwargs):
        options = dict(grid_size=33, adaptive_grid=False, width_levels=3, n_angles=4)
        options.update(kwargs)
        return BayesianMinstedSimulator(**options)

    def test_forward_model_matches_original_at_multiple_widths(self):
        bayes = self.make_sim(bg_map_func=lambda x, y: 3 + .001 * (x*x + y*y))
        original = MinstedSimulator(bg_map_func=bayes.bg_map_func)
        points = np.array([[10., -5.], [20., 30.], [-100., 80.]])
        for width in [40., 100., 250.]:
            original.d_i = width
            action = MinstedExposure((12., -8.), width, .05)
            expected = original.detection_rate(points.T, np.array(action.position_nm)[:, None], 30.)
            np.testing.assert_allclose(bayes.expected_counts(points, action), expected * action.dwell_ms)
            at_half = np.array(action.position_nm) + [width / 2, 0]
            background = bayes._background(action.position_nm)
            self.assertAlmostEqual(float(bayes.expected_counts(at_half, action)) / action.dwell_ms - background, 15.)

    def test_zero_and_positive_counts_match_direct_bayes_updates(self):
        sim = self.make_sim()
        action = MinstedExposure((40., -20.), 100., .1)
        initial = sim.posterior.copy()
        for count in [0, 3, 1, 0]:
            mean = sim.expected_counts(sim._points, action).reshape(sim.posterior.shape)
            expected = sim.posterior * np.exp(xlogy(count, mean) - mean - gammaln(count + 1))
            expected /= expected.sum()
            sim.observe(count, action)
            np.testing.assert_allclose(sim.posterior, expected, atol=1e-15)
        self.assertFalse(np.allclose(initial, sim.posterior))
        self.assertEqual(sim.photon_count, 4)
        self.assertAlmostEqual(sim.time_ms, .4)

    def test_eig_matches_explicit_expected_posterior_entropy(self):
        sim = self.make_sim(grid_size=17, poisson_tail=1e-12)
        action = MinstedExposure((30., 20.), 100., .1)
        p = sim.posterior.ravel()
        mean = sim.expected_counts(sim._points, action)
        prior_entropy = -np.sum(xlogy(p, p))
        expected_entropy = 0.
        predictive_mass = 0.
        for count in range(int(poisson.isf(1e-14, mean.max())) + 1):
            joint = p * poisson.pmf(count, mean)
            probability = joint.sum()
            if probability:
                posterior = joint / probability
                expected_entropy += probability * -np.sum(xlogy(posterior, posterior))
            predictive_mass += probability
        self.assertAlmostEqual(predictive_mass, 1., places=12)
        expected = prior_entropy - expected_entropy
        self.assertAlmostEqual(sim.expected_information_gain([action])[0], expected, places=10)

    def test_position_independent_detector_has_zero_information(self):
        for background in [0., 3.]:
            sim = self.make_sim(brightness=0., bg_global=background)
            actions = sim.candidate_exposures()
            np.testing.assert_allclose(sim.expected_information_gain(actions), 0., atol=1e-12)
            old = sim.posterior.copy()
            sim.observe(0, actions[0])
            np.testing.assert_allclose(sim.posterior, old, atol=1e-15)

    def test_planner_selects_candidate_maximum_without_true_position(self):
        sim = self.make_sim()
        sim.observe(2, MinstedExposure((35., 0.), 100., .1))
        action = sim.select_next_exposure()
        self.assertEqual(action, sim.last_candidates[int(np.argmax(sim.last_eig))])
        self.assertTrue(sim.d_min <= action.fwhm_nm <= sim.d0)
        self.assertGreater(sim.last_eig.max(), 0)
        self.assertTrue(np.all(sim.last_eig >= 0))
        self.assertEqual(len({a.dwell_ms for a in sim.last_candidates}), 1)

    def test_large_means_do_not_underflow_information_calculation(self):
        sim = self.make_sim(brightness=1000., bg_global=0.)
        score = sim.expected_information_gain([MinstedExposure((0., 0.), 100., 1.)])
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertGreater(score[0], 0)

    def test_adaptive_grid_agrees_with_dense_fixed_grid(self):
        adaptive = self.make_sim(grid_size=65, adaptive_grid=True)
        dense = self.make_sim(grid_size=401)
        initial_spacing = np.diff(adaptive.x_axis)[0]
        truth = np.array([10., -5.])
        # Deterministic count data, avoiding a seed-dependent accuracy assertion.
        for _ in range(8):
            for position in [(30., 0.), (-30., 0.), (0., 30.), (0., -30.)]:
                action = MinstedExposure(position, 80., 1.)
                count = int(np.rint(dense.expected_counts(truth, action)))
                adaptive.observe(count, action)
                dense.observe(count, action)
        self.assertLess(np.diff(adaptive.x_axis)[0], initial_spacing / 4)
        self.assertAlmostEqual(adaptive.posterior.sum(), 1., places=13)
        self.assertLessEqual(adaptive.total_pruned_mass, len(adaptive.history) * adaptive.grid_tail_mass)
        np.testing.assert_allclose(adaptive.get_localization_estimate(), dense.get_localization_estimate(), atol=.5)
        self.assertLess(abs(adaptive.get_uncertainty() / dense.get_uncertainty() - 1), .1)

    def test_update_records_data_driven_step_and_zero_counts(self):
        sim = self.make_sim()
        before = sim.get_localization_estimate()
        action = MinstedExposure((40., 0.), 100., .1)
        after = sim.observe(0, action)
        self.assertLess(after[0], before[0])
        np.testing.assert_allclose(sim.history[-1]['update_nm'], after - before)
        self.assertEqual(sim.history[-1]['photon_count'], 0)
        self.assertEqual(sim.get_history().shape, (2, 2))

    def test_impossible_observation_does_not_mutate_state(self):
        sim = self.make_sim(brightness=0., bg_global=0.)
        old = sim.posterior.copy()
        with self.assertRaises(ValueError):
            sim.observe(1, MinstedExposure((0., 0.), 100., .1))
        np.testing.assert_array_equal(sim.posterior, old)
        self.assertEqual(sim.time_ms, 0)
        self.assertEqual(sim.history, [])

    def test_invalid_observations_actions_and_maps(self):
        sim = self.make_sim()
        good = MinstedExposure((0., 0.), 100., .1)
        for count in [-1, .5, True]:
            with self.assertRaises(ValueError):
                sim.observe(count, good)
        for action in [MinstedExposure((0., 0.), 10., .1), MinstedExposure((0., 0.), 100., 0.)]:
            with self.assertRaises(ValueError):
                sim.observe(0, action)
        with self.assertRaises(ValueError):
            sim.expected_information_gain([good, MinstedExposure((0., 0.), 100., .2)])
        with self.assertRaises(ValueError):
            self.make_sim(bg_map_func=lambda x, y: -1).expected_counts([0., 0.], good)
        for options in [dict(prior_sigma=0.), dict(grid_size=4), dict(d_min=300.), dict(brightness=-1.), dict(radius_factors=[-1.])]:
            with self.assertRaises(ValueError):
                self.make_sim(**options)

    def test_reset_restores_prior_and_clears_diagnostics(self):
        sim = self.make_sim()
        action = sim.select_next_exposure()
        sim.observe(1, action)
        sim.reset_tracker([20., -10.])
        np.testing.assert_allclose(sim.get_localization_estimate(), [20., -10.], atol=1e-10)
        self.assertEqual(sim.photon_count, 0)
        self.assertEqual(sim.history, [])
        self.assertEqual(sim.last_candidates, ())
        self.assertEqual(len(sim.last_eig), 0)


if __name__ == '__main__':
    unittest.main()
