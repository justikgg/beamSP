"""Bounded log-linear initialization followed by least squares in Q units."""

from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from core.models.baseline.log_linear_ls import LogLinearLSModel


class LogLinearTRFModel(LogLinearLSModel):
    """Refine the existing power law without changing its feature/domain rules.

    Both stages require positive targets, as do the runtime power-law models.
    The second stage uses the original targets and the shared prediction clipping
    exp(clip(log(Q), -50, 50)). Fixed powers refer to original column indices.
    A finite solution at the evaluation limit remains usable; inspect
    diagnostics_["success"] to distinguish it from a converged solution.
    """

    name = "log_linear_trf"

    TRF_SETTINGS = {
        "method": "trf",
        "ftol": 1e-10,
        "xtol": 1e-10,
        "gtol": 1e-10,
        "max_nfev": 2000,
        "x_scale": "jac",
        "loss": "linear",
        "tr_solver": "exact",
    }

    def __init__(self, fixed_exponents=None, form=None):
        super().__init__(fixed_exponents=fixed_exponents, form=form)
        self._initializer = None
        self._initial_params = None
        self.diagnostics_ = None

    def fit(self, X, y):
        # Invalidate a previous fit even when validation of the new data fails.
        self._params = self._result = self._initial_params = None
        self._used_idx = self.n_features_in_ = None
        self._objective = self._log_mse = self.diagnostics_ = None
        self._initializer = LogLinearLSModel(form=self._form)
        if getattr(self, "feature_names", None) is not None:
            self._initializer.configure_features(self.feature_names, self.dataset_schema)
        started = perf_counter()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                self._initializer.fit(X, y)
        except FloatingPointError as exc:
            raise ValueError("Nonfinite log-linear computation; rescale the inputs") from exc
        except RuntimeError as exc:
            # BLAS overflow need not obey NumPy's errstate. If BVLS failed,
            # distinguish an overflowing fixed contribution from nonconvergence.
            selected = np.asarray(X, dtype=float)[:, self._initializer._used_idx]
            with np.errstate(over="ignore", invalid="ignore"):
                fixed_log_q = np.log(selected) @ self._form._fixed_vals
            if not np.all(np.isfinite(fixed_log_q)):
                raise ValueError("Nonfinite fixed-power contribution; rescale the inputs") from exc
            raise
        initializer_seconds = perf_counter() - started

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._used_idx = self._initializer._used_idx.copy()
        self._initial_params = self._initializer._params.copy()
        Xu = X[:, self._used_idx]
        logX = np.log(Xu)
        design = np.column_stack((np.ones(y.size), logX[:, self._form._free_mask]))
        bounds = np.asarray(self._form.param_bounds(Xu.shape[1]), dtype=float)
        counts = {"residual_calls": 0, "jacobian_calls": 0}

        def prediction(params):
            # Same order of operations and saturation as PowerLawForm.predict.
            with np.errstate(over="ignore", invalid="ignore"):
                log_q = params[0] + logX @ self._form._expand(params[1:])
            if not np.all(np.isfinite(log_q)):
                raise ValueError("Nonfinite power-law log prediction; rescale the inputs")
            return np.exp(np.clip(log_q, -50, 50)), log_q

        def squared_norm(values):
            with np.errstate(over="ignore", invalid="ignore"):
                value = float(values @ values)
            if not np.isfinite(value):
                raise ValueError("Squared residuals overflow; rescale the target values")
            return value

        def residual(params):
            counts["residual_calls"] += 1
            with np.errstate(over="ignore", invalid="ignore"):
                values = prediction(params)[0] - y
            squared_norm(values)
            return values

        def jacobian(params):
            counts["jacobian_calls"] += 1
            pred, log_q = prediction(params)
            # Clipping is locally constant outside the shared prediction range.
            derivative = pred * ((log_q > -50) & (log_q < 50))
            return derivative[:, None] * design

        initial_sse = squared_norm(residual(self._initial_params))
        settings = dict(self.TRF_SETTINGS)
        started = perf_counter()
        self._result = least_squares(
            residual,
            self._initial_params.copy(),
            jac=jacobian,
            bounds=(bounds[:, 0], bounds[:, 1]),
            **settings,
        )
        trf_seconds = perf_counter() - started
        params = self._result.x.copy()
        if (not np.all(np.isfinite(params)) or np.any(params < bounds[:, 0])
                or np.any(params > bounds[:, 1])):
            raise RuntimeError("TRF returned nonfinite or out-of-bounds coefficients")
        final_sse = squared_norm(residual(params))
        if final_sse > initial_sse + 1e-12 * max(1.0, initial_sse):
            raise RuntimeError("TRF increased the original-unit sum of squared residuals")

        self._objective = self._initializer._objective
        self._log_mse = self._objective(params)
        initial_result = self._initializer._result
        self.diagnostics_ = {
            "design_rank": self._initializer.diagnostics_["design_rank"],
            "n_parameters": self._initializer.diagnostics_["n_parameters"],
            "identifiable": self._initializer.diagnostics_["identifiable"],
            "condition_number": self._initializer.diagnostics_["condition_number"],
            "warnings": list(self._initializer.diagnostics_["warnings"]),
            "objective": "sum((Q_pred - Q_real)**2)",
            "prediction_log_clip": [-50.0, 50.0],
            "settings": {**settings, "jac": "analytic (including prediction clipping)"},
            "bounds": bounds.tolist(),
            "initial_sse": initial_sse,
            "final_sse": final_sse,
            "improvement": initial_sse - final_sse,
            "initial_log_mse": float(self._initializer._log_mse),
            "final_log_mse": float(self._log_mse),
            "success": bool(self._result.success),
            "status": int(self._result.status),
            "message": str(self._result.message),
            "nfev": int(self._result.nfev),
            "njev": int(self._result.njev),
            "cost": float(self._result.cost),
            "optimality": float(self._result.optimality),
            "active_mask": self._result.active_mask.tolist(),
            **counts,
            "trf_seconds": trf_seconds,
            "initializer": {
                **self._initializer.diagnostics_,
                "method": "lsq_linear/bvls",
                "success": bool(initial_result.success),
                "status": int(initial_result.status),
                "message": str(initial_result.message),
                "nit": int(initial_result.nit),
                "optimality": float(initial_result.optimality),
                "fit_seconds": initializer_seconds,
            },
        }
        if not self._result.success:
            self.diagnostics_["warnings"].append(
                f"TRF не достиг сходимости: {self._result.message}. "
                "Сохранено допустимое решение с невозросшей обучающей ошибкой.")
        self._params = params
        self.n_features_in_ = X.shape[1]
        return self
