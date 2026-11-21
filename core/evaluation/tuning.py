"""Leakage-safe grid search shared by the GUI and small research scripts."""

from __future__ import annotations

import itertools
import time
from copy import deepcopy
from dataclasses import dataclass
from math import prod
from typing import Any, Callable

import numpy as np

from core.evaluation.metrics import compute_metrics
from core.evaluation.validation import leave_one_group_out
from core.models.parameters import configured_model, validate_parameters


class TuningCancelled(Exception):
    """Raised when a user requests a safe stop between model fits."""


@dataclass
class TuningResult:
    best_parameters: dict[str, Any]
    selection_metrics: dict[str, float]
    independent_metrics: dict[str, float]
    combinations: int
    elapsed_seconds: float
    history: list[dict[str, Any]]


def numeric_range(start, stop, step, *, logarithmic=False):
    """Return an inclusive, deterministic range for a GUI range editor."""
    start, stop, step = float(start), float(stop), float(step)
    if not all(np.isfinite(v) for v in (start, stop, step)) or step <= 0 or stop < start:
        raise ValueError("Требуются: шаг > 0 и максимум не меньше минимума")
    if logarithmic:
        if start <= 0 or stop <= 0 or step <= 1:
            raise ValueError("Для логарифмической шкалы нужны положительные границы и множитель > 1")
        values = []
        value = start
        while value <= stop * (1 + 1e-12):
            values.append(int(value) if value.is_integer() else value)
            value *= step
            if len(values) > 10000:
                raise ValueError("Слишком много значений диапазона")
        return values
    count = int(np.floor((stop - start) / step + 1e-12))
    if count >= 10000:
        raise ValueError("Слишком много значений диапазона")
    values = [start + i * step for i in range(count + 1)]
    if values and abs(values[-1] - stop) < 1e-9:
        values[-1] = stop
    if all(float(v).is_integer() for v in values):
        return [int(v) for v in values]
    return values


def parameter_combinations(parameter_values, *, limit=500):
    """Build the Cartesian product and fail loudly instead of shrinking it."""
    values = {k: list(v) for k, v in parameter_values.items()}
    if any(not v for v in values.values()):
        raise ValueError("Каждый параметр должен содержать хотя бы одно значение")
    total = prod((len(v) for v in values.values()), start=1) if values else 1
    if total > limit:
        raise ValueError(f"Диапазон содержит {total} комбинаций; лимит {limit}")
    keys = list(values)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(values[k] for k in keys))] or [{}]


def _fit_logo(factory: Callable[[], Any], X, y, groups, is_synth, cancel=None):
    if cancel and cancel():
        raise TuningCancelled
    try:
        return leave_one_group_out(factory, X, y, groups, is_synth, cancel=cancel)
    except InterruptedError as exc:
        raise TuningCancelled from exc


def _metrics(factory, X, y, groups, is_synth, cancel=None):
    preds = _fit_logo(factory, X, y, groups, is_synth, cancel)
    real = ~is_synth & np.isfinite(preds)
    return compute_metrics(y[real], preds[real])


def tune_method(name, parameter_values, *, dataset, target_column, synth_cfg=None,
                seed=1337, features=None, schema=None, limit=500,
                progress=None, cancel=None, fixed_exponents=None):
    """Tune one target with inner LOGO and report an outer nested estimate.

    Synthetic rows, when enabled, are generated exactly once before the grid.
    The inner score is used for selection; ``independent_metrics`` is evaluated
    by outer leave-one-profile-out folds and is never used to select a grid row.
    """
    from core.evaluation.runner import prepare

    started = time.monotonic()
    schema = schema or dataset.schema
    df, groups = prepare(dataset.frame, synth_cfg=synth_cfg or {}, seed=seed, schema=schema)
    features = list(features or schema.features)
    X = df[features].to_numpy(float)
    y = df[target_column].to_numpy(float)
    is_synth = np.asarray(df.attrs["beamsp_synthetic"], bool)
    configs = parameter_combinations(parameter_values, limit=limit)
    real_groups = [g for g in np.unique(groups[~is_synth])]
    if len(real_groups) < 3:
        raise ValueError("Для вложенной LOGO-валидации подбора нужны минимум три реальные группы")
    # Reject invalid values before starting any potentially expensive fit.
    valid_configs = []
    for params in configs:
        try:
            valid_configs.append(validate_parameters(name, params))
        except ValueError:
            # Cross-parameter constraints (for example PSO w_lo <= w_hi)
            # can make only part of a Cartesian grid invalid.
            continue
    if not valid_configs:
        raise ValueError(f"{name}: диапазон не содержит допустимых комбинаций")
    configs = valid_configs
    work_total = len(configs) * (1 + len(real_groups)) + len(real_groups)
    completed = 0
    history = []

    def factory(params):
        return configured_model(name, params, fixed=fixed_exponents).set_seed(seed).configure_features(features, schema)

    for index, params in enumerate(configs, 1):
        if cancel and cancel():
            raise TuningCancelled
        metrics = _metrics(lambda: factory(params), X, y, groups, is_synth, cancel)
        history.append({"parameters": deepcopy(params), **metrics})
        completed += 1
        if progress:
            progress(completed, work_total, f"{name}: {params}", time.monotonic() - started)
    best = min(history, key=lambda row: row["RMSE"])

    # Nested estimate: for every outer held-out group, select on the remaining groups.
    outer_true, outer_pred = [], []
    for outer_index, held_out in enumerate(real_groups, 1):
        if cancel and cancel():
            raise TuningCancelled
        train = groups != held_out
        inner_values = []
        for params in configs:
            inner_metrics = _metrics(lambda: factory(params), X[train], y[train],
                                     groups[train], is_synth[train], cancel)
            inner_values.append((inner_metrics["RMSE"], params))
            completed += 1
            if progress:
                progress(completed, work_total, f"{name}: внутренняя проверка профиля {held_out}", time.monotonic() - started)
        selected = min(inner_values, key=lambda row: row[0])[1]
        test = (groups == held_out) & ~is_synth
        if cancel and cancel():
            raise TuningCancelled
        model = factory(selected)
        model.fit(X[train], y[train])
        outer_true.extend(y[test])
        outer_pred.extend(np.asarray(model.predict(X[test]), float))
        completed += 1
        if progress:
            progress(completed, work_total,
                     f"{name}: независимая оценка, профиль {held_out}", time.monotonic() - started)
    independent = compute_metrics(outer_true, outer_pred)
    return TuningResult(best["parameters"], {k: best[k] for k in ("RMSE", "MAE", "R2")},
                        {k: independent[k] for k in ("RMSE", "MAE", "R2")},
                        len(configs), time.monotonic() - started, history)
