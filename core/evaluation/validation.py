import numpy as np
import pandas as pd
from core.data.schema import DatasetError

def leave_one_group_out(model_factory, X, y, groups, is_synth=None, *, fold_diagnostics=None):
    preds = np.full(len(y), np.nan)
    if is_synth is None:
        is_synth = np.zeros(len(y), dtype=bool)
    is_synth = np.asarray(is_synth, bool)

    groups = np.asarray(groups)
    if len(groups) != len(y) or len(X) != len(y) or len(is_synth) != len(y):
        raise DatasetError("Размеры X, y, groups и is_synth не совпадают.")
    if pd.isna(groups).any():
        raise DatasetError("Group содержит пропущенные значения.")
    if len(pd.unique(groups[~is_synth])) < 2:
        raise DatasetError("Для LOGO нужны минимум две реальные группы.")
    for g in pd.unique(groups):
        test_mask = (groups == g) & (~is_synth)
        train_mask = groups != g
        if test_mask.sum() == 0:
            continue
        m = model_factory()
        if hasattr(m, "validate_input"):
            m.validate_input(X[train_mask], y[train_mask])
            m.validate_input(X[test_mask])
        m.fit(X[train_mask], y[train_mask])
        if fold_diagnostics is not None and hasattr(m, "fitted_metadata"):
            fold_diagnostics.append({
                "held_out_group": g.item() if isinstance(g, np.generic) else g,
                "train_rows": int(train_mask.sum()), "test_real_rows": int(test_mask.sum()),
                **m.fitted_metadata(),
            })
        values = np.asarray(m.predict(X[test_mask]), float)
        if not np.isfinite(values).all():
            raise DatasetError(f"Метод {getattr(m, 'name', type(m).__name__)} вернул NaN / inf для группы {g}.")
        preds[test_mask] = values
    return preds
