"""Header-based CSV/XLSX loading with an isolated legacy workbook adapter."""
from pathlib import Path
from copy import deepcopy
import pandas as pd
from .legacy import is_legacy_workbook, legacy_schema, parse_legacy
from .material import add_material_feature, is_material_column, steel_feature_spec
from .schema import (ColumnRole, ColumnSpec, Dataset, DatasetError, DatasetSchema,
                     group_ids, mapping_path, read_schema, validate_headers)


def infer_schema(df: pd.DataFrame) -> DatasetSchema:
    """Suggest common roles; arbitrary targets require an explicit mapping."""
    specs = []
    for name in df.columns:
        lower = name.lower()
        role = ColumnRole.IGNORE
        if name == "is_steel":
            specs.append(steel_feature_spec())
            continue
        if lower in ("profile_id", "group_id", "group"):
            role = ColumnRole.GROUP
        elif lower.startswith(("target", "q_", "qдв_")):
            role = ColumnRole.TARGET
        elif is_material_column(name):
            role = ColumnRole.METADATA
        elif lower in ("id", "shifr", "шифр", "is_synth") or lower.endswith("_id"):
            role = ColumnRole.METADATA
        elif pd.api.types.is_numeric_dtype(df[name]):
            role = ColumnRole.FEATURE
        specs.append(ColumnSpec(name, role, dtype="numeric" if
                                role in (ColumnRole.FEATURE, ColumnRole.TARGET) else "categorical"))
    return DatasetSchema(specs)


def load_dataset_bundle(path, schema: DatasetSchema | None = None,
                        validate: bool = True) -> Dataset:
    """Load a common Dataset; sidecar mappings override inferred roles."""
    path = Path(path)
    legacy = False
    try:
        if path.suffix.lower() == ".xlsx":
            raw = pd.read_excel(path, header=None)
            if is_legacy_workbook(raw):
                df, inferred = parse_legacy(raw), legacy_schema()
                legacy = True
            else:
                if raw.empty:
                    raise DatasetError("Датасет пуст.")
                validate_headers(raw.iloc[0].tolist())
                df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            header = pd.read_csv(path, header=None, nrows=1, encoding="utf-8-sig")
            validate_headers(header.iloc[0].tolist())
            df = pd.read_csv(path, encoding="utf-8-sig")
        else:
            raise DatasetError("Поддерживаются файлы .xlsx и .csv.")
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("Датасет пуст.") from exc
    except (OSError, ValueError, ImportError) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"Не удалось загрузить {path}: {exc}") from exc
    validate_headers(df.columns)
    if df.empty:
        raise DatasetError("Датасет пуст.")
    sidecar = mapping_path(path)
    configured = deepcopy(schema or (read_schema(sidecar) if sidecar.exists() else None))
    material_source = None
    if not legacy:
        df, material_source = add_material_feature(df, configured)
        inferred = infer_schema(df)
    resolved = configured or inferred
    known = {c.name for c in resolved.columns}
    # A saved mapping stays authoritative. Newly added columns are visible for
    # assignment, but cannot silently introduce an unknown target into X.
    # The known material indicator has an explicit, safe default Feature role.
    resolved.columns.extend(steel_feature_spec(material_source) if name == "is_steel" else
                            ColumnSpec(name, ColumnRole.METADATA if is_material_column(name) else ColumnRole.IGNORE,
                                       dtype="numeric" if pd.api.types.is_numeric_dtype(df[name]) else "categorical")
                            for name in df.columns if name not in known)
    if material_source is not None:
        resolved.spec("is_steel").derived_from = material_source
    df.attrs.update(beamsp_schema=resolved.to_dict(), beamsp_source=str(path.resolve()))
    dataset = Dataset(df, resolved, path.resolve())
    if validate:
        dataset.validate()
    return dataset


def load_dataset(path) -> pd.DataFrame:
    """Compatibility API returning a frame with its runtime schema in attrs."""
    return load_dataset_bundle(path).frame


def schema_for_frame(df: pd.DataFrame) -> DatasetSchema:
    stored = df.attrs.get("beamsp_schema")
    return DatasetSchema.from_dict(stored) if stored else infer_schema(df)


def profile_ids(df, feature_cols=None, schema=None):
    """Compatibility wrapper; profile definitions are independent of features."""
    return group_ids(df, schema or schema_for_frame(df))
