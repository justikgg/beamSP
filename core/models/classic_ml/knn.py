from core.models.base import BaseModel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from core.data.schema import DatasetError

class KNNModel(BaseModel):
    name = "knn"
    group = "neighbors"

    def __init__(self, n_neighbors=55, weights="distance"):
        self._m = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))

    def fit(self, X, y):
        self.validate_input(X, y)
        self._m.fit(X, y); return self

    def validate_input(self, X, y=None):
        k = self._m.steps[-1][1].n_neighbors
        if y is not None and len(y) < k:
            raise DatasetError(f"Метод knn: в train fold {len(y)} строк, но n_neighbors={k}. "
                               "Уменьшите число соседей или увеличьте обучающую выборку.")

    def predict(self, X):
        return self._m.predict(X)
