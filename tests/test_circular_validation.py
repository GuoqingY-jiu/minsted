import unittest

import numpy as np

from examples import validate_circular_bayesian as validation


class CircularValidationTests(unittest.TestCase):
    def test_shared_record_budgets_and_exact_replay(self):
        config = dict(
            seed=87, prior_sigma=60., brightness=30., background=3.,
            grid_size=33, planning_grid=7, phase_bins=16, width_levels=3,
            quadrature_order=24, time_ms=.2, photons=4, max_time_ms=10.,
        )
        trial = validation.run_trial((config, 'fixed_offset', 0))
        rows = validation.collect_rows([trial])
        self.assertEqual(len(rows), 6)
        audit = validation.resolution_audit([trial], config, 1, planning_only=True)
        self.assertEqual(audit['posterior_replays'], [])
        self.assertEqual(len(audit['planning_comparisons']), 1)
        self.assertGreaterEqual(audit['planning_comparisons'][0]['relative_eig_regret'], 0)
        for budget in ['time', 'photons']:
            original = next(r for r in rows if r['method'] == 'original' and r['budget'] == budget)
            replay = next(r for r in rows if r['method'] == 'bayes_replay' and r['budget'] == budget)
            self.assertEqual(original['photons'], replay['photons'])
            self.assertEqual(original['time_ms'], replay['time_ms'])
        for path in trial['paths']:
            for checkpoint in path['checkpoints']:
                self.assertTrue(checkpoint['budget_reached'])
                if checkpoint['budget'] == 'photons':
                    self.assertEqual(checkpoint['photons'], 4)
                else:
                    self.assertEqual(checkpoint['time_ms'], .2)
                if checkpoint['method'] != 'original':
                    estimate = validation.replay_path(trial, path, checkpoint, config)
                    np.testing.assert_allclose(estimate, checkpoint['estimate_nm'], atol=1e-12)

    def test_summary_keeps_large_errors_and_preserves_rmse_decomposition(self):
        rows = [dict(
            scenario='fixed_offset', budget='time', method='original',
            error_x_nm=error, error_y_nm=0., budget_reached=True,
            photons=100, time_ms=6.,
        ) for error in [1., 100.]]
        summary = validation.summarize(rows, {'failure_nm': 20.})[0]
        self.assertEqual(summary['repeats'], 2)
        self.assertEqual(summary['failure_rate'], .5)
        self.assertAlmostEqual(summary['rmse_single_axis_nm'], np.sqrt(10001. / 4))
        self.assertAlmostEqual(summary['rmse_single_axis_nm']**2,
                               summary['spread_single_axis_nm']**2 + summary['bias_single_axis_nm']**2)

    def test_paired_bootstrap_retains_known_error_ratio(self):
        rows = []
        for scenario in ['fixed_offset', 'prior_matched']:
            for budget in ['time', 'photons']:
                for method, factor in [('original', 1.), ('bayes_replay', .5), ('bayes_eig_circle', .25)]:
                    for run in range(4):
                        rows.append(dict(scenario=scenario, budget=budget, method=method,
                                         run=run, radial_error_nm=(run + 1) * factor))
        intervals = validation.paired_intervals(rows, 42)
        self.assertEqual(len(intervals), 12)
        for comparison in intervals:
            expected = .25 if comparison['reference'] == 'original' and comparison['comparison'] == 'bayes_eig_circle' else .5
            self.assertAlmostEqual(comparison['rmse_ratio'], expected)
            self.assertAlmostEqual(comparison['bootstrap_95_low'], expected)
            self.assertAlmostEqual(comparison['bootstrap_95_high'], expected)


if __name__ == '__main__':
    unittest.main()
