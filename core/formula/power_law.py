import numpy as np
from core.data.schema import DatasetError


class PowerLawForm:
    log_a_bounds = (-20.0, 20.0)
    power_bounds = (-5.0, 5.0)

    def __init__(self, fixed_exponents=None):
        self._fixed_exponents = dict(fixed_exponents or {})
        self._used_idx = None
        self._free_mask = None
        self._fixed_vals = None

    def usable_features(self, X, names=None, allowed=None):
        names = names or [f"x{i}" for i in range(X.shape[1])]
        fixed = {}
        for key, value in self._fixed_exponents.items():
            if isinstance(key, str):
                if key not in names:
                    raise DatasetError(f"Фиксация степени: feature {key} отсутствует среди выбранных признаков.")
                key = names.index(key)
            if key < 0 or key >= len(names) or not np.isfinite(value):
                raise DatasetError("Некорректный индекс или значение фиксированной степени.")
            fixed[key] = value
        used_idx = np.asarray(allowed if allowed is not None else range(X.shape[1]), dtype=int)
        for i in used_idx:
            if np.any(X[:, i] <= 0):
                raise DatasetError(f"Степенной метод не может использовать feature {names[i]}: присутствуют неположительные значения.")
        for i, value in fixed.items():
            if i not in used_idx and value != 0:
                raise DatasetError(f"Feature {names[i]} исключён из степенной формулы; нельзя фиксировать ненулевую степень.")
        self._used_idx = used_idx
        self._free_mask = np.array([i not in fixed for i in used_idx], dtype=bool)
        self._fixed_vals = np.array(
            [fixed.get(i, 0.0) for i in used_idx], dtype=float)
        return used_idx

    def _expand(self, free_params):
        powers = self._fixed_vals.copy()
        powers[self._free_mask] = free_params
        return powers

    def param_bounds(self, n_features):
        n_free = int(self._free_mask.sum())
        return [self.log_a_bounds] + [self.power_bounds] * n_free

    def objective(self, Xu, y):
        logX = np.log(Xu)
        log_y = np.log(y)

        def mse(params):
            powers = self._expand(params[1:])
            log_pred = params[0] + logX @ powers
            return float(np.mean((log_pred - log_y) ** 2))

        return mse

    def predict(self, params, Xu):
        powers = self._expand(params[1:])
        log_q = params[0] + np.log(Xu) @ powers
        return np.exp(np.clip(log_q, -50, 50))

    def render(self, params, names, target="Qдв"):
        a = np.exp(params[0])
        powers = self._expand(params[1:])
        terms = [f"{name}^{p:.3f}" for name, p in zip(names, powers)]
        return f"{target} = {a:.4g}" + (" * " + " * ".join(terms) if terms else "")
