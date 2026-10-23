import numpy as np
from sklearn.linear_model import ARDRegression

from core.models.base import BaseModel

_REL_THRESHOLD = 0.02


def _build_terms(feature_names, binary_features=()):
    idx = {name: i for i, name in enumerate(feature_names)}
    positive = [n for n in feature_names if n not in binary_features]
    terms = []

    for n in feature_names:                       
        i = idx[n]
        terms.append((n, lambda X, i=i: X[:, i]))
    for n in positive:                            
        i = idx[n]
        terms.append((f"{n}^2", lambda X, i=i: X[:, i] ** 2))
    for n in positive:                            
        i = idx[n]
        terms.append((f"sqrt({n})", lambda X, i=i: np.sqrt(np.abs(X[:, i]))))
    for n in positive:                            
        i = idx[n]
        terms.append((f"ln({n})", lambda X, i=i: np.log(np.abs(X[:, i]) + 1e-9)))
    for a in range(len(feature_names)):           
        for b in range(a + 1, len(feature_names)):
            na, nb = feature_names[a], feature_names[b]
            terms.append((f"{na}*{nb}", lambda X, a=a, b=b: X[:, a] * X[:, b]))
    return terms


class _BayesianFormula:
    def __init__(self, terms, model, scaler_mean, scaler_std):
        self._terms = terms
        self._model = model
        self._mean = scaler_mean
        self._std = scaler_std

    def _design(self, X):
        cols = [f(X) for _, f in self._terms]
        M = np.column_stack(cols)
        return (M - self._mean) / self._std

    def predict(self, X):
        return self._model.predict(self._design(np.asarray(X, float)))


def fit_bayesian_symbolic(X, y, feature_names=None, seed=42, threshold_lambda=10000.0,
                         binary_features=()):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    terms = _build_terms(feature_names, binary_features)
    M = np.column_stack([f(X) for _, f in terms])


    mean = M.mean(axis=0)
    std = M.std(axis=0)
    std[std == 0] = 1.0
    Ms = (M - mean) / std

    model = ARDRegression(threshold_lambda=threshold_lambda)
    model.fit(Ms, y)

    
    
    
    
    cutoff = _REL_THRESHOLD * np.max(np.abs(model.coef_)) if model.coef_.size else 0
    parts = []  
    for (tname, _), coef, m, s in zip(terms, model.coef_, mean, std):
        if abs(coef) < cutoff:
            continue
        parts.append((abs(coef), coef / s, tname, m))
    parts.sort(reverse=True)  
    intercept = float(model.intercept_) - sum(w * m for _, w, _, m in parts)

    terms_str = " ".join(
        f"{'+' if w >= 0 else '-'} {abs(w):.4g}*{name}"
        for _, w, name, _ in parts)
    expr = f"Qдв = {intercept:.4g} {terms_str}".strip()
    if not parts:
        expr += "  (все термы отсеяны как незначимые)"

    wrapper = _BayesianFormula(terms, model, mean, std)
    return wrapper, expr


class BayesianSymbolicRegressionModel(BaseModel):
    name = "bayes_symreg"
    group = "formula"

    def __init__(self, threshold_lambda=10000.0, seed=1337):
        self.threshold_lambda = threshold_lambda
        self._seed = seed
        self._wrapper = None
        self._formula = None

    def set_seed(self, seed):
        self._seed = seed
        return self

    def fit(self, X, y):
        schema = getattr(self, "dataset_schema", None)
        binary = [n for n in self.names_for(X) if schema is not None and schema.spec(n).dtype == "binary"]
        self._wrapper, self._formula = fit_bayesian_symbolic(
            X, y, feature_names=self.names_for(X), seed=self._seed,
            threshold_lambda=self.threshold_lambda, binary_features=binary)
        return self

    def predict(self, X):
        return self._wrapper.predict(X)

    def formula(self, feature_names=None):
        return self._formula or "модель не обучена"
