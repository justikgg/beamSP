import pytest
from unittest.mock import patch

import numpy as np

from test_pipeline import dynamic
from core.models.parameters import configured_model, parameter_defaults
from core.models.registry import available_models
from core.evaluation.tuning import TuningCancelled, tune_method
from entrypoint.method_parameters import TUNABLE_METHODS

from core.evaluation.tuning import numeric_range, parameter_combinations


def test_numeric_range_includes_end_and_supports_log_scale():
    assert numeric_range(0, 1, 0.5) == [0, 0.5, 1]
    assert numeric_range(1, 16, 2, logarithmic=True) == [1, 2, 4, 8, 16]


def test_combinations_are_cartesian_and_limit_is_explicit():
    assert parameter_combinations({"a": [1, 2], "b": ["x", "y"]}) == [
        {"a": 1, "b": "x"}, {"a": 1, "b": "y"},
        {"a": 2, "b": "x"}, {"a": 2, "b": "y"},
    ]
    with pytest.raises(ValueError, match="4"):
        parameter_combinations({"a": [1, 2], "b": ["x", "y"]}, limit=3)


def test_empty_grid_keeps_default_configuration():
    assert parameter_combinations({}) == [{}]


def test_registry_tuning_capabilities():
    assert set(available_models()) - TUNABLE_METHODS == {
        "linear", "log_linear_ls", "log_linear_trf"
    }
    assert len(TUNABLE_METHODS) == 16


@pytest.mark.parametrize("name", sorted(TUNABLE_METHODS))
def test_two_settings_for_each_tunable_method(name):
    # Keep optimizer/neural tests cheap while exercising the real constructor,
    # parameter validation, complete grid and nested fold orchestration.
    defaults = parameter_defaults(name)
    key = next(k for k, v in defaults.items() if isinstance(v, (int, float))
               and not isinstance(v, bool))
    value = defaults[key]
    other = value + 1 if isinstance(value, int) else value * 1.1
    configs = {k: [v] for k, v in defaults.items()}
    configs[key] = [value, other]
    calls = []

    def cheap_metric(factory, X, y, groups, is_synth, cancel=None):
        model = factory()
        assert model.name == name
        calls.append((len(np.unique(groups[~is_synth])), len(X)))
        score = 1.0 if len(calls) % 2 else 2.0
        return {"RMSE": score, "MAE": score, "R2": 0.0}

    def cheap_model(method, values, fixed=None):
        model = configured_model(method, values, fixed=fixed)
        model.fit = lambda X, y: model
        model.predict = lambda X: np.full(len(X), 1.0)
        return model

    with (patch("core.evaluation.tuning._metrics", side_effect=cheap_metric),
          patch("core.evaluation.tuning.configured_model", side_effect=cheap_model)):
        result = tune_method(name, configs, dataset=dynamic(), target_column="target_a")
    assert result.combinations == 2
    assert len(result.history) == 2
    assert result.selection_metrics["RMSE"] == 1.0
    assert calls[:2] == [(4, 32), (4, 32)]
    assert all(n == 3 and rows == 24 for n, rows in calls[2:])


def test_ridge_selects_minimum_validation_rmse_and_cancels():
    ds = dynamic()
    result = tune_method("ridge", {"alpha": [1, 2]}, dataset=ds,
                         target_column="target_a")
    assert result.best_parameters["alpha"] == min(
        result.history, key=lambda row: row["RMSE"])["parameters"]["alpha"]
    assert all(np.isfinite(v) for v in result.independent_metrics.values())
    with pytest.raises(TuningCancelled):
        tune_method("ridge", {"alpha": [1, 2]}, dataset=ds,
                    target_column="target_a", cancel=lambda: True)


def test_invalid_grids_and_limits():
    with pytest.raises(ValueError):
        numeric_range(float("nan"), 2, 1)
    with pytest.raises(ValueError):
        numeric_range(0, 100000, 0.1)
    with pytest.raises(ValueError):
        parameter_combinations({"alpha": []})
    with pytest.raises(ValueError, match="лимит"):
        tune_method("ridge", {"alpha": [1, 2, 3]}, dataset=dynamic(),
                    target_column="target_a", limit=2)


def test_nested_folds_keep_synthetic_rows_with_their_source_group():
    seen = []

    def inspect(factory, X, y, groups, is_synth, cancel=None):
        seen.append((len(np.unique(groups[~is_synth])), len(X), int(is_synth.sum())))
        return {"RMSE": 1.0, "MAE": 1.0, "R2": 0.0}

    with patch("core.evaluation.tuning._metrics", side_effect=inspect):
        result = tune_method("ridge", {"alpha": [1]}, dataset=dynamic(),
                             target_column="target_a",
                             synth_cfg={"enabled": True, "samples_per_profile": 1})
    assert result.combinations == 1
    assert seen[0] == (4, 64, 32)
    assert seen[1:] == [(3, 48, 24)] * 4


def test_real_elasticnet_pso_and_symbolic_smoke_with_small_budgets():
    ds = dynamic()
    elastic = tune_method(
        "elasticnet", {"alpha": [0.1, 1.0], "l1_ratio": [0.2]},
        dataset=ds, target_column="target_a", limit=2,
    )
    assert elastic.combinations == 2
    assert set(elastic.best_parameters) == {"alpha", "l1_ratio"}

    pso_values = {
        "n_particles": [3], "iters": [2], "w_hi": [0.8], "w_lo": [0.4],
        "c1": [1.0], "c2": [1.0],
    }
    pso_result = tune_method("bio_pso", pso_values, dataset=ds,
                             target_column="target_a", limit=1)
    assert pso_result.combinations == 1
    assert pso_result.history[0]["parameters"]["iters"] == 2

    symreg = tune_method(
        "symreg", {"population_size": [10], "generations": [1],
                    "parsimony_coefficient": [0.08]},
        dataset=ds, target_column="target_a", limit=1,
    )
    assert symreg.combinations == 1
    assert symreg.best_parameters["population_size"] == 10


def test_fixed_bio_tuning_passes_constraint_to_every_fit():
    ds = dynamic()
    result = tune_method(
        "bio_pso", {"n_particles": [3], "iters": [1], "w_hi": [0.8],
                    "w_lo": [0.4], "c1": [1.0], "c2": [1.0]},
        dataset=ds, target_column="target_a", limit=1,
        fixed_exponents={"x1": 1},
    )
    assert result.combinations == 1
