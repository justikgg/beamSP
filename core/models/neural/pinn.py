import torch.nn as nn
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from core.models.base import BaseModel
from core.data.schema import DatasetError


class _PINNNet(nn.Module):
    def __init__(self, n_features, hidden=(64, 64)):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PINNModel(BaseModel):
    name = "pinn"
    group = "neural"

    def __init__(self, hidden=(64, 64), epochs=1500, lr=5e-3,
                 phys_weight=1.0, seed=1337):
        self._hidden = hidden
        self._epochs = epochs
        self._lr = lr
        self._phys_weight = phys_weight
        self._seed = seed
        self._net = None
        self._xs = None
        self._ys = None

    def set_seed(self, seed):
        self._seed = seed
        return self

    def fit(self, X, y):
        torch.manual_seed(self._seed)
        X = np.asarray(X, float)
        self.validate_input(X, y)
        schema = getattr(self, "dataset_schema", None)
        monotonic = [(i, schema.spec(n).monotonic) for i, n in enumerate(self.names_for(X))
                     if schema is not None and schema.spec(n).monotonic]
        y = np.asarray(y, float).reshape(-1, 1)

        self._xs = StandardScaler().fit(X)
        self._ys = StandardScaler().fit(y)
        Xt = torch.tensor(self._xs.transform(X), dtype=torch.float32)
        yt = torch.tensor(self._ys.transform(y), dtype=torch.float32)

        self._net = _PINNNet(X.shape[1], self._hidden)
        opt = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        lo = Xt.min(0).values.detach()
        hi = Xt.max(0).values.detach()
        zero_std = float(-self._ys.mean_[0] / self._ys.scale_[0])

        for _ in range(self._epochs):
            opt.zero_grad()
            data_loss = ((self._net(Xt) - yt) ** 2).mean()

            coll = lo + (hi - lo) * torch.rand(64, Xt.shape[1])
            coll.requires_grad_(True)
            q = self._net(coll)
            grad = torch.autograd.grad(q.sum(), coll, create_graph=True)[0]

            if monotonic:
                idx, signs = zip(*monotonic)
                mono = grad[:, list(idx)] * torch.tensor(signs, dtype=grad.dtype)
                mono_loss = torch.relu(-mono).pow(2).mean()
            else:
                mono_loss = q.new_tensor(0.0)
            nonneg_loss = torch.relu(zero_std - q).pow(2).mean()
            phys_loss = mono_loss + nonneg_loss

            (data_loss + self._phys_weight * phys_loss).backward()
            opt.step()
        return self

    def validate_input(self, X, y=None):
        self.names_for(X)
        if y is not None and np.any(np.asarray(y) < 0):
            raise DatasetError("Метод pinn предполагает неотрицательный target (физический штраф Q ≥ 0).")

    def predict(self, X):
        X = np.asarray(X, float)
        Xt = torch.tensor(self._xs.transform(X), dtype=torch.float32)
        with torch.no_grad():
            pred = self._net(Xt).numpy()
        return self._ys.inverse_transform(pred).ravel()
