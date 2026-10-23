"""Explicit Formula exclusions for the partner's mathematical test cases."""
import numpy as np

from core.data.schema import DatasetSchema, FeatureSpec


def fit_excluding(model, X, y, excluded):
    X = np.asarray(X, float)
    names = [f"x{i}" for i in range(X.shape[1])]
    schema = DatasetSchema([FeatureSpec(name, formula_allowed=i not in excluded)
                            for i, name in enumerate(names)])
    return model.configure_features(names, schema).fit(X, y)
