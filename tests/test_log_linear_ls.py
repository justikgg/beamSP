import unittest

import numpy as np

from core.formula.power_law import PowerLawForm
from core.models.baseline.log_linear_ls import LogLinearLSModel
from power_law_helpers import fit_excluding


class LogLinearLSChecks(unittest.TestCase):
    def test_recovers_power_laws_with_arbitrary_feature_counts(self):
        rng = np.random.default_rng(73)
        for width in (1, 3, 7):
            with self.subTest(width=width):
                logX = rng.uniform(-1, 1, size=(40, width))
                positive = np.exp(logX)
                # The binary column must be excluded without shifting exponents.
                X = np.column_stack((positive[:, 0], np.zeros(40), positive[:, 1:]))
                powers = np.linspace(-0.8, 1.2, width)
                y = np.exp(0.7 + logX @ powers)
                model = fit_excluding(LogLinearLSModel(), X, y, [1])
                np.testing.assert_allclose(model.predict(X), y, rtol=1e-10)
                np.testing.assert_allclose(model._params, np.r_[0.7, powers], atol=1e-10)
                self.assertNotIn(1, model._used_idx)
                self.assertTrue(model._result.success)

    def test_active_power_bound_reoptimizes_intercept(self):
        logX = np.linspace(0, 2, 21)
        X = np.exp(logX[:, None])
        y = np.exp(7 * logX)
        model = LogLinearLSModel().fit(X, y)
        # With p constrained to 5, the optimal intercept is 2, not 0.
        np.testing.assert_allclose(model._params, [2, 5], atol=1e-10)
        self.assertEqual(model._result.active_mask[1], 1)
        self.assertAlmostEqual(model._log_mse, np.mean((2 - 2 * logX) ** 2))

    def test_fixed_powers_use_original_column_indices(self):
        rng = np.random.default_rng(17)
        logX = rng.uniform(-1, 1, size=(30, 3))
        X = np.column_stack((np.exp(logX[:, 0]), np.zeros(30), np.exp(logX[:, 1:])))
        y = np.exp(0.4 + 1.25 * logX[:, 0] - 0.6 * logX[:, 1])
        model = fit_excluding(LogLinearLSModel(fixed_exponents={0: 1.25, 3: 0.0}), X, y, [1])
        np.testing.assert_allclose(model._params, [0.4, -0.6], atol=1e-10)
        np.testing.assert_allclose(model.predict(X), y, rtol=1e-10)
        self.assertIn("x0^1.250", model.formula())
        self.assertIn("x3^0.000", model.formula())

    def test_all_fixed_powers_leave_only_intercept(self):
        X = np.exp(np.arange(12).reshape(6, 2) / 10)
        y = np.exp(0.3 + np.log(X) @ np.array([1.5, -0.4]))
        model = LogLinearLSModel(form=PowerLawForm({0: 1.5, 1: -0.4})).fit(X, y)
        np.testing.assert_allclose(model._params, [0.3], atol=1e-10)
        np.testing.assert_allclose(model.predict(X), y, rtol=1e-10)

    def test_intercept_only_and_intercept_bounds(self):
        X = np.column_stack((np.zeros(5), -np.ones(5)))
        model = fit_excluding(LogLinearLSModel(), X, np.full(5, np.exp(25)), [0, 1])
        self.assertEqual(model._used_idx.size, 0)
        np.testing.assert_allclose(model._params, [20], atol=1e-10)
        np.testing.assert_allclose(model.predict(X), np.exp(20), rtol=1e-10)
        self.assertNotIn("*", model.formula())

    def test_rank_deficient_design_and_strict_target_domain(self):
        x = np.exp(np.linspace(-1, 1, 20))
        X = np.column_stack((x, x**2))
        y = np.exp(0.5) * x**1.8
        model = LogLinearLSModel().fit(X, y)
        np.testing.assert_allclose(model.predict(X), y, rtol=1e-10)
        self.assertFalse(model.diagnostics_['identifiable'])
        self.assertEqual(model.diagnostics_['design_rank'], 2)
        for targets in ([-1, 1, 2], [0, 1, 2]):
            with self.assertRaisesRegex(ValueError, 'положительные.*target'):
                LogLinearLSModel().fit(np.ones((3, 1)), targets)
        tiny = fit_excluding(LogLinearLSModel(), np.zeros((3, 1)), [1e-12] * 3, [0])
        np.testing.assert_allclose(tiny._params, [-20], atol=1e-10)
        self.assertAlmostEqual(tiny._log_mse, (np.log(1e-12) + 20) ** 2)

    def test_invalid_inputs_and_prediction_domain(self):
        model = LogLinearLSModel()
        with self.assertRaises(RuntimeError):
            model.predict([[1]])
        for X, y in (([], []), ([[1], [2]], [1]), ([[np.nan]], [1]), ([[1]], [np.inf])):
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                model.fit(X, y)
        model.fit([[1], [2], [3]], [2, 4, 6])
        with self.assertRaises(ValueError):
            model.predict([[1, 2]])
        with self.assertRaises(ValueError):
            model.predict([[0]])


if __name__ == "__main__":
    unittest.main()
