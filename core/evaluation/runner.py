"""Runtime dataset/schema -> fold-local models -> metrics and provenance."""
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from core.data.loader import load_dataset_bundle, schema_for_frame
from core.data.legacy import legacy_profile_label
from core.data.schema import (ColumnRole, Dataset, DatasetError, DatasetSchema, group_ids,
                              validate_dataset, validate_feature_sources)
from core.data.synth import synthesize
from core.evaluation.run_result import RunResult
from core.models.registry import get_model
from core.evaluation.validation import leave_one_group_out
from core.evaluation.metrics import compute_metrics


def _overfit_metrics(model, X, y, real, logo_r2):
    model.validate_input(X, y)
    model.fit(X, y)
    pred_train = model.predict(X[real])
    r2_train = float(r2_score(y[real], pred_train))
    return r2_train, r2_train - logo_r2


def _environment_versions():
    versions = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "pandas", "scikit-learn"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def per_fold_rmse(y, preds, groups, real):
    return {g: float(np.sqrt(np.mean((y[mask] - preds[mask]) ** 2)))
            for g in pd.unique(groups) if (mask := real & (groups == g)).any()}


def _profile_labels(df, groups, is_synth, schema):
    labels = {}
    for g in pd.unique(groups):
        sub = df[(groups == g) & ~is_synth]
        label = legacy_profile_label(sub.iloc[0]) if schema.legacy and len(sub) else None
        labels[g] = f"{label} [профиль {g}]" if label else f"профиль {g}"
    return labels


def selected_schema(df, base, feature_cols=None, targets=None):
    """Adapt historical run arguments while preserving per-column specifications."""
    features = list(feature_cols) if feature_cols is not None else base.features
    targets = dict(targets) if targets is not None else base.targets
    # Do not allow an existing target/group to become a feature by an accidental override.
    illegal = set(features) & {c.name for c in base.columns if c.role in (ColumnRole.TARGET, ColumnRole.GROUP)}
    if illegal:
        raise DatasetError(f"Пересечение ролей: {', '.join(sorted(illegal))}. Target и group нельзя включать в features.")
    if set(base.grouping_columns) & {c.name for c in base.columns if c.role == ColumnRole.TARGET}:
        raise DatasetError("Target нельзя использовать для определения профиля.")
    validate_feature_sources(base, features)
    schema = DatasetSchema.from_mapping(df.columns, features, targets, base.group_column, base.grouping_columns)
    specs = {c.name: c for c in base.columns}
    schema.columns = [replace(specs[c.name], role=c.role, enabled=True, target_key=c.target_key)
                      if c.name in specs else c for c in schema.columns]
    schema.legacy = base.legacy
    return schema


def prepare(df, feature_cols=None, synth_cfg=None, seed=1337, schema=None):
    """Validate/augment a frame and freeze profile IDs before any perturbation."""
    schema = schema or schema_for_frame(df)
    if feature_cols is not None and list(feature_cols) != schema.features:
        schema = selected_schema(df, schema, feature_cols)
    df = validate_dataset(df, schema).reset_index(drop=True)
    groups = group_ids(df, schema)
    synth_cfg = synth_cfg or {}
    if synth_cfg.get("enabled"):
        df = synthesize(df, synth_cfg.get("samples_per_profile", 15),
                        synth_cfg.get("noise", schema.noise), schema.features, seed,
                        schema=schema, groups=groups)
        groups = np.asarray(df.attrs["beamsp_groups"])
    else:
        if "is_synth" not in df:
            df["is_synth"] = 0
        if "profile_id" not in df:
            df["profile_id"] = groups
        df.attrs.update(beamsp_synthetic=[False] * len(df), beamsp_groups=groups.tolist())
    return df, groups


def run(model_names, feature_cols=None, targets=None, data_path=None, synth_cfg=None,
        seed=1337, progress=None, *, schema=None, dataset=None, model_factories=None):
    """Evaluate registered models with arbitrary runtime features and targets.

    Existing positional callers are supported. GUI may pass a validated Dataset
    and per-run model factories (fixed bio variants) without mutating the registry.
    """
    if dataset is None:
        if data_path is None:
            raise DatasetError("Укажите путь датасета или Dataset.")
        dataset = load_dataset_bundle(data_path, schema=schema, validate=False)
    elif isinstance(dataset, pd.DataFrame):
        source = dataset.attrs.get("beamsp_source")
        dataset = Dataset(dataset, schema or schema_for_frame(dataset), Path(source) if source else None)
    base = schema or dataset.schema
    if schema is not None and ((feature_cols is not None and list(feature_cols) != schema.features)
                               or (targets is not None and dict(targets) != schema.targets)):
        raise DatasetError("Переданные features/targets не совпадают с runtime schema.")
    active = selected_schema(dataset.frame, base, feature_cols, targets)
    df, groups = prepare(dataset.frame, synth_cfg=synth_cfg, seed=seed, schema=active)
    features, targets = active.features, active.targets
    X = df[features].to_numpy(float)
    is_synth = np.asarray(df.attrs["beamsp_synthetic"], bool)
    real = ~is_synth
    result = RunResult()
    source = dataset.source or (Path(data_path).resolve() if data_path else None)
    result.metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": _environment_versions(),
        "dataset_path": str(source) if source else None,
        "dataset_name": source.name if source else None,
        "feature_columns": features, "targets": targets,
        "group_column": active.group_column, "grouping_columns": active.grouping_columns,
        "real_rows": int(real.sum()), "synthetic_rows": int(is_synth.sum()),
        "groups": len(pd.unique(groups[real])), "seed": seed,
        "schema": active.to_dict(), "synthesis": {
            "enabled": bool((synth_cfg or {}).get("enabled", False)),
            "samples_per_profile": (synth_cfg or {}).get("samples_per_profile", 15),
            "noise": (synth_cfg or {}).get("noise", active.noise),
        },
        "models": list(model_names), "model_config": {}, "per_target": {},
    }
    factories = model_factories or {}
    label_map = _profile_labels(df, groups, is_synth, active)
    total, done = len(targets) * len(model_names), 0
    for tkey, tcol in targets.items():
        y = df[tcol].to_numpy(float)
        rows, preds_by_model, formulas = [], {}, {}
        result.fold_rmse[tkey] = {}
        result.metadata["per_target"][tkey] = {"target": tcol, "unit": active.spec(tcol).unit,
                                               "models": {}}
        for name in model_names:
            def make_model(n=name):
                model = factories[n]() if n in factories else get_model(n)
                return model.set_seed(seed).configure_features(features, active)
            fold_details = []
            preds = leave_one_group_out(make_model, X, y, groups, is_synth,
                                       fold_diagnostics=fold_details)
            m = compute_metrics(y[real], preds[real])
            pf = per_fold_rmse(y, preds, groups, real)
            m["RMSE_worst"] = max(pf.values()) if pf else 0.0
            result.fold_rmse[tkey][name] = {label_map[g]: v for g, v in pf.items()}
            model = make_model()
            result.metadata["model_config"][name] = {"class": type(model).__name__}
            if hasattr(model, "_form"):
                result.metadata["model_config"][name]["fixed_exponents"] = dict(model._form._fixed_exponents)
            m["R2_train"], m["overfit"] = _overfit_metrics(model, X, y, real, m["R2"])
            if hasattr(model, "fitted_metadata"):
                result.metadata["per_target"][tkey]["models"][name] = {
                    **model.fitted_metadata(), "folds": fold_details,
                }
            if hasattr(model, "formula"):
                expr = model.formula(features)
                formulas[name] = tcol + " = " + expr.split(" = ", 1)[1] if " = " in expr else expr
            m["model"] = name
            rows.append(m)
            preds_by_model[name] = {"y_true": y[real], "y_pred": preds[real]}
            done += 1
            if progress:
                progress(done, total, f"{name} / {tkey}")
        col_order = ["Qexp/Qpred_mean", "CV", "within15", "MAPE", "MAE", "MedianAE",
                     "RMSE", "RMSE_worst", "MaxError", "pct_negative", "pct_conservative",
                     "R2", "R2_train", "overfit"]
        if rows:
            table = pd.DataFrame(rows).set_index("model")
            result.metrics[tkey] = table[[c for c in col_order if c in table]].sort_values("RMSE")
        else:
            result.metrics[tkey] = pd.DataFrame()
        result.predictions[tkey], result.formulas[tkey] = preds_by_model, formulas
    return result
