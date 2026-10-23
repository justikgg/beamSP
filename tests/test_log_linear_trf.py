"""Contract and numerical checks for the two-stage bounded power-law fit."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares as scipy_least_squares

from core.formula.power_law import PowerLawForm
from core.models.baseline.log_linear_ls import LogLinearLSModel
from core.models.baseline.log_linear_trf import LogLinearTRFModel
from core.models.bioinspired.formula_search import FormulaSearchModel
from core.models.bioinspired.optimizers.de import de
from power_law_helpers import fit_excluding


TRF_MODULE = "core.models.baseline.log_linear_trf"


class LogLinearTRFChecks(unittest.TestCase):
    def test_passes_bounded_ls_coefficients_to_trf_without_changing_them(self):
        X = np.exp(np.linspace(-0.8, 0.9, 30)[:, None])
        y = np.exp(0.4) * X[:, 0] ** 1.2 * (1 + 0.08 * np.sin(np.arange(30)))
        expected = LogLinearLSModel().fit(X, y)
        with patch(f"{TRF_MODULE}.least_squares", wraps=scipy_least_squares) as solver:
            model = LogLinearTRFModel().fit(X, y)

        solver.assert_called_once()
        call = solver.call_args
        x0 = call.kwargs.get("x0", call.args[1] if len(call.args) > 1 else None)
        np.testing.assert_array_equal(x0, expected._params)
        np.testing.assert_array_equal(model._initial_params, expected._params)
        np.testing.assert_array_equal(model._initializer._params, expected._params)
        self.assertIsInstance(model._initializer, LogLinearLSModel)
        self.assertFalse(np.shares_memory(model._initial_params, model._initializer._params))
        np.testing.assert_array_equal(model._params, model._result.x)
        np.testing.assert_array_equal(model._used_idx, expected._used_idx)
        self.assertEqual(model.n_features_in_, X.shape[1])

    def test_explicit_trf_settings_bounds_and_diagnostics(self):
        X = np.exp(np.arange(12).reshape(6, 2) / 12)
        y = np.exp(0.3 + np.log(X) @ [0.7, -0.1])
        with patch(f"{TRF_MODULE}.least_squares", wraps=scipy_least_squares) as solver:
            model = LogLinearTRFModel().fit(X, y)
        settings = {
            "method": "trf", "ftol": 1e-10, "xtol": 1e-10,
            "gtol": 1e-10, "max_nfev": 2000, "x_scale": "jac",
            "loss": "linear", "tr_solver": "exact",
        }
        for name, value in settings.items():
            with self.subTest(setting=name):
                self.assertEqual(solver.call_args.kwargs[name], value)
                self.assertEqual(model.diagnostics_["settings"][name], value)
        self.assertTrue(callable(solver.call_args.kwargs["jac"]))
        lower, upper = solver.call_args.kwargs["bounds"]
        np.testing.assert_array_equal(lower, [-20, -5, -5])
        np.testing.assert_array_equal(upper, [20, 5, 5])
        for key in ("success", "status", "message", "nfev", "njev", "cost", "optimality"):
            self.assertEqual(model.diagnostics_[key], model._result[key])
        np.testing.assert_array_equal(model.diagnostics_["active_mask"], model._result.active_mask)
        self.assertGreaterEqual(model.diagnostics_["residual_calls"], model._result.nfev)
        self.assertGreaterEqual(model.diagnostics_["jacobian_calls"], model._result.njev)
        self.assertGreaterEqual(model.diagnostics_["trf_seconds"], 0)
        self.assertIn("initializer", model.diagnostics_)

    def test_raw_unit_residuals_and_objective_improve_over_log_fit(self):
        X = np.zeros((3, 2))
        y = np.array([1.0, 4.0, 16.0])
        model = fit_excluding(LogLinearTRFModel(), X, y, [0, 1])
        np.testing.assert_allclose(np.exp(model._initial_params), [4.0], rtol=1e-12)
        np.testing.assert_allclose(model.predict(X), [7.0] * 3, rtol=1e-7)
        np.testing.assert_allclose(model._result.fun, model.predict(X) - y, atol=1e-12)
        initial_sse = np.sum((model._initializer.predict(X) - y) ** 2)
        final_sse = np.sum((model.predict(X) - y) ** 2)
        self.assertAlmostEqual(model.diagnostics_["initial_sse"], initial_sse)
        self.assertAlmostEqual(model.diagnostics_["final_sse"], final_sse)
        self.assertAlmostEqual(model.diagnostics_["improvement"], initial_sse - final_sse)
        self.assertLess(final_sse, initial_sse - 1)
        self.assertAlmostEqual(model._result.cost, final_sse / 2)

    def test_same_training_sample_never_worsens_raw_sse(self):
        rng = np.random.default_rng(29)
        for width in (1, 4, 8):
            with self.subTest(width=width):
                X = np.exp(rng.uniform(-1, 1, size=(70, width)))
                y = np.exp(1.2 + np.log(X) @ np.linspace(-0.4, 0.7, width))
                y *= np.exp(rng.normal(0, 0.2, y.size))
                model = LogLinearTRFModel().fit(X, y)
                initial = np.sum((model._initializer.predict(X) - y) ** 2)
                final = np.sum((model.predict(X) - y) ** 2)
                self.assertLessEqual(final, initial + 1e-12 * max(1, initial))
                self.assertTrue(model._result.success)

    def test_recovers_power_laws_with_arbitrary_feature_counts_and_shared_selection(self):
        rng = np.random.default_rng(73)
        for width in (1, 3, 7):
            with self.subTest(width=width):
                logX = rng.uniform(-1, 1, size=(45, width))
                X = np.column_stack((np.exp(logX[:, 0]), np.zeros(45), np.exp(logX[:, 1:]), -np.ones(45)))
                powers = np.linspace(-0.8, 1.2, width)
                y = np.exp(0.7 + logX @ powers)
                model = fit_excluding(LogLinearTRFModel(), X, y, [1, X.shape[1] - 1])
                np.testing.assert_allclose(model.predict(X), y, rtol=1e-9)
                np.testing.assert_allclose(model._params, np.r_[0.7, powers], atol=1e-8)
                np.testing.assert_array_equal(model._used_idx, np.r_[0, np.arange(2, width + 1)])
                np.testing.assert_array_equal(model._used_idx, PowerLawForm().usable_features(
                    X, allowed=[0, *range(2, width + 1)]))

    def test_free_power_and_intercept_bounds_are_preserved(self):
        logX = np.linspace(0, 2, 25)
        X = np.exp(logX[:, None])
        model = LogLinearTRFModel().fit(X, np.exp(7 * logX))
        self.assertTrue(np.all(model._params >= [-20, -5]))
        self.assertTrue(np.all(model._params <= [20, 5]))
        self.assertAlmostEqual(model._params[1], 5, places=6)
        intercept = fit_excluding(LogLinearTRFModel(), np.zeros((5, 2)), np.full(5, np.exp(25)), [0, 1])
        self.assertLessEqual(intercept._params[0], 20)
        self.assertAlmostEqual(intercept._params[0], 20, places=6)

    def test_fixed_exponents_use_original_column_indices(self):
        rng = np.random.default_rng(17)
        logX = rng.uniform(-0.5, 0.5, size=(30, 3))
        X = np.column_stack((np.exp(logX[:, 0]), np.zeros(30), np.exp(logX[:, 1:])))
        y = np.exp(0.4 + 1.25 * logX[:, 0] - 0.6 * logX[:, 1])
        model = fit_excluding(LogLinearTRFModel(fixed_exponents={0: 1.25, 3: 0.0}), X, y, [1])
        np.testing.assert_allclose(model._params, [0.4, -0.6], atol=1e-8)
        np.testing.assert_allclose(model.predict(X), y, rtol=1e-9)
        self.assertIn("x0^1.250", model.formula())
        self.assertIn("x3^0.000", model.formula())
        self.assertNotIn("x1^", model.formula())

    def test_all_fixed_powers_allow_only_intercept_and_do_not_bound_fixed_values(self):
        X = np.exp(np.arange(12).reshape(6, 2) / 20)
        fixed = {0: 6.25, 1: -0.4}
        y = np.exp(0.3 + np.log(X) @ [6.25, -0.4])
        model = LogLinearTRFModel(form=PowerLawForm(fixed)).fit(X, y)
        np.testing.assert_allclose(model._params, [0.3], atol=1e-8)
        np.testing.assert_allclose(model.predict(X), y, rtol=1e-9)
        self.assertIn("x0^6.250", model.formula())

    def test_no_usable_features_and_zero_feature_matrix(self):
        for X in (np.column_stack((np.zeros(4), -np.ones(4))), np.empty((4, 0))):
            with self.subTest(shape=X.shape):
                model = fit_excluding(LogLinearTRFModel(), X, np.full(4, 3.0), range(X.shape[1]))
                self.assertEqual(model._used_idx.size, 0)
                np.testing.assert_allclose(model._params, [np.log(3)], atol=1e-9)
                np.testing.assert_allclose(model.predict(X), 3, rtol=1e-9)
                self.assertNotIn("*", model.formula())

    def test_predictions_and_engineering_formula_use_existing_power_law(self):
        X = np.array([[1, 0, 2], [2, 0, 3], [4, 0, 2], [3, 0, 5]], dtype=float)
        y = 2.5 * X[:, 0] ** 1.2 * X[:, 2] ** -0.7
        model = fit_excluding(LogLinearTRFModel(), X, y, [1])
        expected = model._form.predict(model._params, X[:, model._used_idx])
        np.testing.assert_array_equal(model.predict(X), expected)
        names = ["height", "binary", "width"]
        self.assertEqual(model.formula(names), model._form.render(model._params, ["height", "width"]))
        self.assertIn("height^1.200", model.formula(names))
        self.assertIn("width^-0.700", model.formula(names))
        with self.assertRaises(ValueError):
            model.formula(["wrong_width"])

    def test_clipped_predictions_and_analytic_jacobian_match_shared_form(self):
        X = np.exp(np.array([-20.0, 0.0, 20.0])[:, None])
        with patch(f"{TRF_MODULE}.least_squares", wraps=scipy_least_squares) as solver:
            model = LogLinearTRFModel().fit(X, np.ones(3))
        fun = solver.call_args.args[0] if solver.call_args.args else solver.call_args.kwargs["fun"]
        jac = solver.call_args.kwargs["jac"]
        saturated_params = np.array([0.0, 5.0])
        expected_predictions = np.exp([-50.0, 0.0, 50.0])
        np.testing.assert_array_equal(fun(saturated_params), expected_predictions - 1)
        np.testing.assert_array_equal(jac(saturated_params), [[0, 0], [1, 0], [0, 0]])
        params = np.array([0.2, 0.3])
        step = 1e-6
        numeric = np.column_stack([
            (fun(params + step * np.eye(2)[i]) - fun(params - step * np.eye(2)[i])) / (2 * step)
            for i in range(2)
        ])
        np.testing.assert_allclose(jac(params), numeric, rtol=1e-8, atol=1e-7)
        model._params = saturated_params
        np.testing.assert_array_equal(model.predict(X), expected_predictions)

    def test_jacobian_has_only_free_columns_after_fixed_power_selection(self):
        X = np.array([[1, 0, 2], [2, 0, 3], [3, 0, 5]], dtype=float)
        y = np.exp(0.3) * X[:, 0] ** 1.25 * X[:, 2] ** -0.6
        with patch(f"{TRF_MODULE}.least_squares", wraps=scipy_least_squares) as solver:
            model = fit_excluding(LogLinearTRFModel(fixed_exponents={0: 1.25}), X, y, [1])
        params = np.array([0.2, -0.4])
        pred = model._form.predict(params, X[:, model._used_idx])
        expected = pred[:, None] * np.column_stack((np.ones(3), np.log(X[:, 2])))
        np.testing.assert_allclose(solver.call_args.kwargs["jac"](params), expected)

    def test_rank_deficient_features_still_produce_valid_predictions(self):
        x = np.exp(np.linspace(-1, 1, 20))
        X = np.column_stack((x, x ** 2))
        y = np.exp(0.5) * x ** 1.8
        model = LogLinearTRFModel().fit(X, y)
        np.testing.assert_allclose(model.predict(X), y, rtol=1e-9)
        self.assertTrue(np.all(np.isfinite(model._params)))

    def test_nonpositive_targets_are_rejected_without_clipping(self):
        for y in ([-1.0, 1.0, 2.0], [0.0, 1.0, 2.0]):
            with self.subTest(y=y), self.assertRaisesRegex(ValueError, 'положительные.*target'):
                LogLinearTRFModel().fit(np.ones((3, 1)), y)

    def test_invalid_inputs_and_prediction_domain(self):
        self.assertEqual(LogLinearTRFModel().formula(), "модель не обучена")
        with self.assertRaises(RuntimeError):
            LogLinearTRFModel().predict([[1]])
        invalid = (
            ([], []), ([[1], [2]], [1]), ([[1]], [[1]]),
            ([[np.nan]], [1]), ([[np.inf]], [1]), ([[1]], [np.inf]),
            ([[1]], [np.nan]), ([1, 2], [1, 2]), (np.empty((0, 2)), []),
        )
        for X, y in invalid:
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                LogLinearTRFModel().fit(X, y)
        model = LogLinearTRFModel().fit([[1], [2], [3]], [2, 4, 6])
        for X in ([[1, 2]], [[0]], [[-1]], [[np.nan]], [[np.inf]]):
            with self.subTest(prediction=X), self.assertRaises(ValueError):
                model.predict(X)

    def test_invalid_refit_clears_previous_fit(self):
        model = LogLinearTRFModel().fit([[1], [2], [3]], [2, 4, 6])
        with self.assertRaises(ValueError):
            model.fit([[1], [np.nan]], [2, 4])
        with self.assertRaises(RuntimeError):
            model.predict([[1]])

    def test_nonfinite_fixed_powers_and_objective_overflow_are_rejected(self):
        for fixed in (np.nan, np.inf, -np.inf):
            with self.subTest(fixed=fixed), self.assertRaises(ValueError):
                LogLinearTRFModel(fixed_exponents={0: fixed}).fit([[1], [2]], [1, 2])
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaises(ValueError):
                LogLinearTRFModel().fit([[1], [2]], [1e308, 1e308])
            with self.assertRaises(ValueError):
                LogLinearTRFModel(fixed_exponents={0: 1e308}).fit([[10], [20]], [1, 2])

    def test_max_nfev_nonconvergence_is_recorded_when_solution_is_usable(self):
        def limited_solver(*args, **kwargs):
            kwargs["max_nfev"] = 1
            return scipy_least_squares(*args, **kwargs)

        with patch(f"{TRF_MODULE}.least_squares", side_effect=limited_solver):
            model = fit_excluding(LogLinearTRFModel(), np.zeros((3, 1)), [1, 4, 16], [0])
        self.assertFalse(model.diagnostics_["success"])
        self.assertEqual(model.diagnostics_["status"], 0)
        self.assertEqual(model.diagnostics_["nfev"], 1)
        self.assertTrue(model.diagnostics_["message"])
        self.assertTrue(any('не достиг сходимости' in s for s in model.diagnostics_['warnings']))
        np.testing.assert_allclose(model._params, model._initial_params)

    def test_unusable_or_worsening_solver_result_is_rejected(self):
        for bad_params in (np.array([np.nan]), np.array([21.0]), np.array([0.0])):
            def bad_solver(*args, **kwargs):
                result = scipy_least_squares(*args, **kwargs)
                result.x = bad_params.copy()
                return result

            with self.subTest(params=bad_params), patch(f"{TRF_MODULE}.least_squares", side_effect=bad_solver):
                with self.assertRaises(RuntimeError):
                    fit_excluding(LogLinearTRFModel(), np.zeros((3, 1)), [1, 4, 16], [0])

    def test_constructor_conflicting_power_law_configuration(self):
        with self.assertRaises(ValueError):
            LogLinearTRFModel(fixed_exponents={0: 1}, form=PowerLawForm())

    def test_old_log_linear_fit_keeps_log_objective_and_bvls_settings(self):
        from scipy.optimize import lsq_linear

        with patch("core.models.baseline.log_linear_ls.lsq_linear", wraps=lsq_linear) as solver:
            model = fit_excluding(LogLinearLSModel(), np.zeros((3, 1)), [1, 4, 16], [0])
        np.testing.assert_allclose(model.predict(np.zeros((3, 1))), 4, rtol=1e-12)
        self.assertEqual(solver.call_args.kwargs["method"], "bvls")
        self.assertEqual(solver.call_args.kwargs["lsq_solver"], "exact")
        self.assertEqual(solver.call_args.kwargs["tol"], 1e-10)
        self.assertAlmostEqual(model._log_mse, np.mean((np.log(4) - np.log([1, 4, 16])) ** 2))

    def test_old_de_wrapper_and_formula_search_keep_original_contract(self):
        objective = lambda params: float(np.sum(np.asarray(params) ** 2))
        sentinel = np.array([0.1, 0.2])
        with patch("core.models.bioinspired.optimizers.de.differential_evolution") as solver:
            solver.return_value.x = sentinel
            self.assertIs(de(objective, [(-20, 20), (-5, 5)], seed=73), sentinel)
            solver.assert_called_once_with(
                objective, [(-20, 20), (-5, 5)], seed=73, maxiter=200,
                popsize=20, tol=1e-8, mutation=(0.5, 1.0), recombination=0.7,
                polish=True,
            )

        captured = {}

        def optimizer(objective, bounds, seed):
            captured.update(objective=objective, bounds=bounds, seed=seed)
            return np.array([np.log(4), 0.0])

        X = np.array([[1, 0], [2, 0], [4, 0]], dtype=float)
        y = np.array([1, 4, 16])
        model = fit_excluding(FormulaSearchModel(optimizer, seed=19), X, y, [1])
        self.assertEqual(captured["seed"], 19)
        self.assertEqual(captured["bounds"], [(-20.0, 20.0), (-5.0, 5.0)])
        self.assertAlmostEqual(captured["objective"](model._params), np.mean((np.log(4) - np.log(y)) ** 2))
        np.testing.assert_array_equal(model._used_idx, [0])
        np.testing.assert_allclose(model.predict(X), 4)


if __name__ == "__main__":
    unittest.main()
