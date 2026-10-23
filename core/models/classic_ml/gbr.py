from sklearn.ensemble import GradientBoostingRegressor
from core.models.base import BaseModel


class GBRModel(BaseModel):
    name = "gbr"
    group = "ensemble"

    def __init__(self, n_estimators=400, max_depth=1, learning_rate=0.1):
        self._m = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=1337)

    def fit(self, X, y):
        self._m.fit(X, y); return self

    def set_seed(self, seed):
        self._m.set_params(random_state=seed)
        return self

    def predict(self, X):
        return self._m.predict(X)
