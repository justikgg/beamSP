from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    name: str = "base"
    group: str = "other"
    supports_uncertainty: bool = False
    stochastic: bool = False

    def configure_features(self, feature_names, schema=None) -> "BaseModel":
        """Supply runtime names/capabilities without changing fit(X, y)."""
        self.feature_names = list(feature_names)
        self.dataset_schema = schema
        return self

    def names_for(self, X) -> list[str]:
        names = getattr(self, "feature_names", None)
        if names is not None and len(names) != X.shape[1]:
            raise ValueError(f"Метод {self.name}: количество имён features не совпадает с X.")
        return names or [f"x{i}" for i in range(X.shape[1])]

    def validate_input(self, X, y=None) -> None:
        """Override for mathematical restrictions; called for each train/test fold."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def set_seed(self, seed: int) -> "BaseModel":
        return self

    def predict_std(self, X: np.ndarray):
        return None
