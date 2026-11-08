"""Typed run settings shared by the GUI and evaluation pipeline."""

from copy import deepcopy
from inspect import signature
from math import isfinite
from typing import Any

from core.models.bioinspired.optimizers import OPTIMIZERS
from core.models.registry import model_class

LABELS = {
    "alpha": "Регуляризация",
    "l1_ratio": "Доля L1",
    "C": "Штраф C",
    "epsilon": "Нечувствительность ε",
    "gamma": "Коэффициент ядра γ",
    "n_neighbors": "Количество соседей",
    "weights": "Веса соседей",
    "kernel_type": "Ядро",
    "n_restarts_optimizer": "Перезапуски оптимизатора",
    "n_estimators": "Количество деревьев",
    "max_depth": "Глубина деревьев",
    "learning_rate": "Скорость обучения",
    "hidden": "Размеры скрытых слоёв",
    "epochs": "Эпохи",
    "lr": "Скорость обучения",
    "phys_weight": "Вес физического штрафа",
    "entropy_weight": "Вес энтропии",
    "threshold_lambda": "Порог отбора",
    "population_size": "Размер популяции",
    "generations": "Поколения",
    "parsimony_coefficient": "Штраф за сложность",
    "pop_size": "Размер популяции",
    "elite_frac": "Доля элиты",
    "tournament": "Размер турнира",
    "blx_alpha": "Расширение скрещивания",
    "mutation_rate": "Вероятность мутации",
    "maxiter": "Максимум итераций",
    "popsize": "Множитель популяции",
    "tol": "Допуск",
    "mutation": "Диапазон мутации",
    "recombination": "Рекомбинация",
    "polish": "Финальная доводка",
    "n_particles": "Количество частиц",
    "iters": "Итерации",
    "w_hi": "Начальная инерция",
    "w_lo": "Конечная инерция",
    "c1": "Личный коэффициент",
    "c2": "Социальный коэффициент",
}
CHOICES = {
    "weights": ("uniform", "distance"),
    "kernel_type": (
        "rbf",
        "rbf_ard",
        "matern1.5",
        "matern1.5_ard",
        "matern2.5",
        "matern2.5_ard",
        "rq",
    ),
}


def parameter_defaults(name: str) -> dict[str, Any]:
    """Expose existing constructor defaults, excluding infrastructure arguments."""
    target = OPTIMIZERS[name[4:]] if name.startswith("bio_") else model_class(name)
    return {
        key: deepcopy(p.default)
        for key, p in signature(target).parameters.items()
        if p.default is not p.empty and key not in {"seed", "form", "fixed_exponents"}
    }


def validate_parameters(name: str, values: dict[str, Any]) -> dict[str, Any]:
    defaults = parameter_defaults(name)
    unknown = values.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"{name}: неизвестные параметры: {', '.join(unknown)}")
    result = {**defaults, **deepcopy(values)}
    zero_allowed = {
        "alpha",
        "l1_ratio",
        "epsilon",
        "n_restarts_optimizer",
        "phys_weight",
        "entropy_weight",
        "parsimony_coefficient",
        "blx_alpha",
        "tol",
        "mutation_rate",
        "recombination",
        "w_hi",
        "w_lo",
        "c1",
        "c2",
    }
    for key, value in result.items():
        default = defaults[key]
        label = LABELS.get(key, key)
        valid = True
        if key in CHOICES:
            valid = value in CHOICES[key]
        elif isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, tuple):
            valid = isinstance(value, (tuple, list)) and bool(value)
            if valid and key == "hidden":
                valid = all(type(v) is int and v > 0 for v in value)
            elif valid:
                valid = (
                    len(value) == 2
                    and all(isinstance(v, (int, float)) and isfinite(v) for v in value)
                    and 0 <= value[0] <= value[1] < 2
                )
            if valid:
                result[key] = tuple(value)
        else:
            valid = (
                type(value) is int
                if type(default) is int
                else type(value) in (int, float)
            )
            if valid:
                valid = isfinite(value) and (
                    value >= 0 if key in zero_allowed else value > 0
                )
                if key in {"l1_ratio", "mutation_rate", "recombination"}:
                    valid = valid and value <= 1
                if key == "elite_frac":
                    valid = valid and value < 1
                if key in {"population_size", "pop_size"}:
                    valid = valid and value >= 2
        if not valid:
            raise ValueError(f"{name}: недопустимое значение «{label}»")
    if "w_hi" in result and result["w_lo"] > result["w_hi"]:
        raise ValueError(f"{name}: конечная инерция больше начальной")
    return result


def configured_model(name: str, values: dict[str, Any], fixed: dict | None = None):
    """Build a fresh model for one fold without modifying the global registry."""
    params = validate_parameters(name, values)
    if name.startswith("bio_"):
        from functools import partial

        from core.formula.power_law import PowerLawForm
        from core.models.bioinspired.formula_search import FormulaSearchModel

        model = FormulaSearchModel(
            partial(OPTIMIZERS[name[4:]], **params),
            form=PowerLawForm(fixed_exponents=fixed),
        )
        model.name = name
        return model
    if name in {"log_linear_ls", "log_linear_trf"}:
        params["fixed_exponents"] = fixed
    return model_class(name)(**params)
