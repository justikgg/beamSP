import numpy as np
from scipy.optimize import lsq_linear

from core.formula.power_law import PowerLawForm
from core.models.base import BaseModel
from core.data.schema import DatasetError


class LogLinearLSModel(BaseModel):
    """Fit the existing bounded power law by least squares in log space."""

    name = "log_linear_ls"
    group = "formula"
    stochastic = False

    def __init__(self, fixed_exponents=None, form=None):
        if form is not None and fixed_exponents is not None:
            raise ValueError("Pass either form or fixed_exponents, not both")
        self._form = form if form is not None else PowerLawForm(fixed_exponents)
        self._params = None
        self._used_idx = None
        self._result = None
        self._objective = None
        self._log_mse = None
        self.n_features_in_ = None
        self.diagnostics_ = None

    def _allowed_indices(self, X):
        schema = getattr(self, "dataset_schema", None)
        return [i for i, name in enumerate(self.names_for(X))
                if schema is None or schema.spec(name).formula_allowed]

    def validate_input(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise DatasetError("X must be a nonempty two-dimensional array")
        if not np.isfinite(X).all():
            raise DatasetError("X must contain only finite values")
        names = self.names_for(X)
        for i in self._allowed_indices(X):
            if np.any(X[:, i] <= 0):
                raise DatasetError(f"Метод {self.name} не может использовать feature {names[i]}: "
                                   "присутствуют неположительные значения. "
                                   "Отключите Formula для этого столбца в схеме датасета.")
        if y is not None:
            y = np.asarray(y, dtype=float)
            if y.ndim != 1 or len(y) != len(X):
                raise DatasetError("y must be one-dimensional with one value per row of X")
            if not np.isfinite(y).all():
                raise DatasetError("y must contain only finite values")
            if np.any(y <= 0):
                raise DatasetError(f"Метод {self.name} требует положительные значения target.")

    def fit(self, X, y):
        # A failed refit must never leave the previous coefficients usable.
        self._params = self._result = self._objective = self._log_mse = None
        self._used_idx = self.n_features_in_ = self.diagnostics_ = None
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.validate_input(X, y)
        self._used_idx = self._form.usable_features(
            X, self.names_for(X), self._allowed_indices(X))
        Xu = X[:, self._used_idx]
        logX = np.log(Xu)
        if not np.all(np.isfinite(self._form._fixed_vals)):
            raise ValueError("Fixed exponents must be finite")

        design = np.column_stack((np.ones(y.size), logX[:, self._form._free_mask]))
        response = np.log(y) - logX @ self._form._fixed_vals
        if not np.isfinite(response).all():
            raise ValueError("Nonfinite fixed-power contribution; rescale the inputs")
        bounds = np.asarray(self._form.param_bounds(Xu.shape[1]), dtype=float)
        self._objective = self._form.objective(Xu, y)
        self._result = lsq_linear(
            design,
            response,
            bounds=(bounds[:, 0], bounds[:, 1]),
            method="bvls",
            lsq_solver="exact",
            tol=1e-10,
        )
        if not self._result.success:
            raise RuntimeError(f"Bounded log-linear fit failed: {self._result.message}")
        self._params = self._result.x.copy()
        self._log_mse = self._objective(self._params)
        self.n_features_in_ = X.shape[1]
        rank = int(np.linalg.matrix_rank(design))
        n_params = int(design.shape[1])
        condition = float(np.linalg.cond(design))
        self.diagnostics_ = {
            "objective": "mean((log_a + log(X) @ powers - log(Q_real))**2)",
            "method": "lsq_linear/bvls", "lsq_solver": "exact", "tol": 1e-10,
            "success": bool(self._result.success), "status": int(self._result.status),
            "message": str(self._result.message), "nit": int(self._result.nit),
            "optimality": float(self._result.optimality), "log_mse": float(self._log_mse),
            "design_rank": rank, "n_parameters": n_params,
            "identifiable": rank == n_params,
            "condition_number": condition if np.isfinite(condition) else None,
            "warnings": [] if rank == n_params else [
                f"Ранг [1, ln(X_free)] = {rank} < {n_params}: "
                "свободные показатели не определяются однозначно."],
        }
        return self

    def predict(self, X):
        if self._params is None:
            raise RuntimeError("Model is not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} feature columns")
        self.validate_input(X)
        Xu = X[:, self._used_idx]
        return self._form.predict(self._params, Xu)

    def formula(self, feature_names=None):
        if self._params is None:
            return "модель не обучена"
        if feature_names is None:
            configured = getattr(self, "feature_names", None)
            names = [configured[i] if configured is not None else f"x{i}"
                     for i in self._used_idx]
        else:
            if len(feature_names) != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} feature names")
            names = [feature_names[i] for i in self._used_idx]
        if not names:
            return f"Qдв = {np.exp(self._params[0]):.4g}"
        return self._form.render(self._params, names)

    def fitted_metadata(self):
        """Portable exact coefficients and their input order for exported results."""
        if self._params is None:
            raise RuntimeError("Model is not fitted")
        names = getattr(self, "feature_names", None)
        if names is None:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        schema = getattr(self, "dataset_schema", None)
        excluded = [{"name": name, "reason": "formula_allowed=false",
                     "unit": schema.spec(name).unit if schema else ""}
                    for i, name in enumerate(names) if i not in self._used_idx]
        return {
            "feature_columns": list(names),
            "used_feature_columns": [names[i] for i in self._used_idx],
            "excluded_features": excluded,
            "log_a": float(self._params[0]), "A": float(np.exp(self._params[0])),
            "exponents": {names[i]: float(p) for i, p in
                          zip(self._used_idx, self._form._expand(self._params[1:]))},
            "free_parameters": self._params.tolist(),
            "bounds": np.asarray(self._form.param_bounds(len(self._used_idx))).tolist(),
            "prediction_log_clip": [-50.0, 50.0],
            "diagnostics": self.diagnostics_,
        }
