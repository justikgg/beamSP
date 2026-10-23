from core.models.baseline.elastic_net import ElasticNetModel
from core.models.baseline.linear import LinearModel
from core.models.baseline.ridge import RidgeModel
from core.models.baseline.lasso import LassoModel
from core.models.baseline.log_linear_ls import LogLinearLSModel
from core.models.baseline.log_linear_trf import LogLinearTRFModel
from core.models.classic_ml.gbr import GBRModel
from core.models.classic_ml.gpr import GPRModel
from core.models.classic_ml.knn import KNNModel
from core.models.classic_ml.svr import SVRModel
from core.models.bioinspired.formula_search import bio_models
from core.models.experimental.rl import ActorCriticRegressor
from core.models.neural.mlp import MLPModel
from core.models.neural.pinn import PINNModel
from core.models.sym_regression.baesian_symbolic_regression import BayesianSymbolicRegressionModel
from core.models.sym_regression.symbolic_regression import SymbolicRegressionModel

_BASELINE = [LinearModel, RidgeModel, LassoModel, ElasticNetModel]
_CLASSIC = [GBRModel, SVRModel, KNNModel, GPRModel]
_FORMULA = [SymbolicRegressionModel, BayesianSymbolicRegressionModel,
            LogLinearLSModel, LogLinearTRFModel]
_NEURAL = [PINNModel, MLPModel]
_EXPERIMENTAL = [ActorCriticRegressor]

_REGISTRY = {cls.name: cls for cls in (*_BASELINE, *_CLASSIC, *_FORMULA, *_NEURAL,*_EXPERIMENTAL)}
_REGISTRY.update(bio_models())


def model_class(name):
    if name not in _REGISTRY:
        raise KeyError(f"Модель '{name}' не зарегистрирована. "
                       f"Доступны: {list(_REGISTRY)}")
    return _REGISTRY[name]


def get_model(name):
    return model_class(name)()


def available_models():
    return list(_REGISTRY)


def register_model(cls, *, replace=False):
    """Register a BaseModel class; the Research GUI lists it automatically."""
    from core.models.base import BaseModel
    if not issubclass(cls, BaseModel) or cls.name == "base":
        raise ValueError("Зарегистрируйте класс BaseModel с уникальным name.")
    if cls.name in _REGISTRY and not replace:
        raise ValueError(f"Модель {cls.name} уже зарегистрирована.")
    _REGISTRY[cls.name] = cls
    return cls
