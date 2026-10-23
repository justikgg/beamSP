from core.models.base import BaseModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


def _build_kernel(kernel_type, n_features):
    ard = kernel_type.endswith("_ard")
    base = kernel_type[:-4] if ard else kernel_type
    length_scale = [1.0] * n_features if ard else 1.0

    if base == "rbf":
        core = RBF(length_scale=length_scale)
    elif base == "matern1.5":
        core = Matern(length_scale=length_scale, nu=1.5)
    elif base == "matern2.5":
        core = Matern(length_scale=length_scale, nu=2.5)
    elif base == "rq":
        core = RationalQuadratic(length_scale=1.0)
    else:
        raise ValueError(f"Неизвестный kernel_type: {kernel_type}")
    return ConstantKernel() * core + WhiteKernel()


class GPRModel(BaseModel):
    name = "gpr"
    group = "kernel"
    supports_uncertainty = True

    def __init__(self, kernel_type="matern1.5_ard", n_restarts_optimizer=5):
        self.kernel_type = kernel_type
        self.n_restarts_optimizer = n_restarts_optimizer
        self._scaler = StandardScaler()
        self._gp = None
        self._seed = 42

    def set_seed(self, seed):
        self._seed = seed
        return self

    def fit(self, X, y):
        Xs = self._scaler.fit_transform(X)
        kernel = _build_kernel(self.kernel_type, Xs.shape[1])
        self._gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer, random_state=self._seed,
        )
        self._gp.fit(Xs, y)
        return self

    def predict(self, X):
        return self._gp.predict(self._scaler.transform(X))

    def predict_std(self, X):
        _, std = self._gp.predict(self._scaler.transform(X), return_std=True)
        return std
