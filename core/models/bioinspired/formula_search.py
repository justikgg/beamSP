import numpy as np

from core.models.base import BaseModel
from core.formula.power_law import PowerLawForm
from core.models.bioinspired.optimizers import OPTIMIZERS
from core.data.schema import DatasetError


class FormulaSearchModel(BaseModel):
    group = "formula"
    stochastic = True

    def __init__(self, optimizer, form=None, seed=1337):
        self._optimizer = optimizer
        self._form = form or PowerLawForm()
        self._seed = seed
        self._params = None
        self._used_idx = None

    def set_seed(self, seed):
        self._seed = seed
        return self

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.validate_input(X, y)
        self._used_idx = self._form.usable_features(X, self.names_for(X), self._allowed_indices(X))
        Xu = X[:, self._used_idx]

        objective = self._form.objective(Xu, y)
        bounds = self._form.param_bounds(Xu.shape[1])
        self._params = self._optimizer(objective, bounds, self._seed)
        return self

    def predict(self, X):
        self.validate_input(np.asarray(X, float))
        Xu = np.asarray(X, float)[:, self._used_idx]
        return self._form.predict(self._params, Xu)

    def formula(self, feature_names=None):
        if self._params is None:
            return "модель не обучена"
        if feature_names is not None:
            names = [feature_names[i] for i in self._used_idx]
        else:
            names = [f"x{i}" for i in self._used_idx]
        return self._form.render(self._params, names)

    def _allowed_indices(self, X):
        schema = getattr(self, "dataset_schema", None)
        return [i for i, name in enumerate(self.names_for(X))
                if schema is None or schema.spec(name).formula_allowed]

    def validate_input(self, X, y=None):
        names = self.names_for(X)
        for i in self._allowed_indices(X):
            if np.any(X[:, i] <= 0):
                raise DatasetError(f"Метод {self.name} не может использовать feature {names[i]}: присутствуют неположительные значения.")
        if y is not None and np.any(y <= 0):
            raise DatasetError(f"Метод {self.name} требует положительные значения target.")


def power_law_model(opt_name):
    optimizer = OPTIMIZERS[opt_name]

    class _PowerLaw(FormulaSearchModel):
        name = f"bio_{opt_name}"

        def __init__(self, seed=1337):
            super().__init__(optimizer, form=PowerLawForm(), seed=seed)

    _PowerLaw.__name__ = _PowerLaw.__qualname__ = f"PowerLaw_{opt_name.upper()}"
    return _PowerLaw


def bio_models():
    models = {}
    for opt in OPTIMIZERS:
        cls = power_law_model(opt)
        models[cls.name] = cls
    return models
