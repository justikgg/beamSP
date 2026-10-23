from gplearn.genetic import SymbolicRegressor

from core.models.base import BaseModel


def fit_symbolic(X, y, feature_names=None, seed=1337,
                  population_size=1000, generations=20, parsimony_coefficient=0.08):
    sr = SymbolicRegressor(
        population_size=population_size, generations=generations,
        function_set=("add", "sub", "mul", "div", "sqrt", "log"),
        parsimony_coefficient=parsimony_coefficient, random_state=seed, verbose=0,
        feature_names=feature_names,
    )
    sr.fit(X, y)
    raw = str(sr._program)
    try:
        expr = humanize_expression(raw)
    except Exception:
        expr = raw
    return sr, f"Qдв = {expr}"


class SymbolicRegressionModel(BaseModel):
    name = "symreg"
    group = "formula"
    stochastic = True

    def __init__(self, population_size=1000, generations=20, parsimony_coefficient=0.08, seed=1337):
        self.population_size = population_size
        self.generations = generations
        self.parsimony_coefficient = parsimony_coefficient
        self._seed = seed
        self._sr = None
        self._formula = None

    def set_seed(self, seed):
        self._seed = seed
        return self

    def fit(self, X, y):
        self._sr, self._formula = fit_symbolic(
            X, y, feature_names=self.names_for(X), seed=self._seed,
            population_size=self.population_size, generations=self.generations,
            parsimony_coefficient=self.parsimony_coefficient)
        return self

    def predict(self, X):
        return self._sr.predict(X)

    def formula(self, feature_names=None):
        if self._sr is not None and feature_names is not None:
            self._sr._program.feature_names = list(feature_names)
            return "Qдв = " + humanize_expression(str(self._sr._program))
        return self._formula or "модель не обучена"


_BINARY_OPS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_UNARY_OPS = {"sqrt": "sqrt", "log": "ln"}


def _parse_prefix(expr):
    pos = 0
    n = len(expr)

    def parse_node():
        nonlocal pos
        while pos < n and expr[pos] == " ":
            pos += 1
        start = pos
        while pos < n and expr[pos] not in "(),":
            pos += 1
        token = expr[start:pos].strip()
        while pos < n and expr[pos] == " ":
            pos += 1
        if pos < n and expr[pos] == "(":
            pos += 1
            args = [parse_node()]
            while pos < n and expr[pos] == ",":
                pos += 1
                args.append(parse_node())
            if pos < n and expr[pos] == ")":
                pos += 1
            return (token, args)
        return token

    return parse_node()


def _render(node):
    if isinstance(node, str):
        return node
    func, args = node
    if func in _BINARY_OPS and len(args) == 2:
        return f"({_render(args[0])} {_BINARY_OPS[func]} {_render(args[1])})"
    if func in _UNARY_OPS and len(args) == 1:
        return f"{_UNARY_OPS[func]}({_render(args[0])})"
    return f"{func}({', '.join(_render(a) for a in args)})"


def humanize_expression(expr):
    rendered = _render(_parse_prefix(expr))
    if rendered.startswith("(") and rendered.endswith(")"):
        depth = 0
        for i, ch in enumerate(rendered):
            depth += 1 if ch == "(" else -1 if ch == ")" else 0
            if depth == 0 and i != len(rendered) - 1:
                break
        else:
            rendered = rendered[1:-1]
    return rendered
