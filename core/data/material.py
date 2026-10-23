"""Deterministic material encoding for header-based beam datasets."""
import pandas as pd

from .schema import ColumnRole, DatasetError, DatasetSchema, FeatureSpec

MATERIAL_COLUMN_NAMES = frozenset({
    "material", "material type", "материал", "тип материала",
    "тип материала двутавра", "материал двутавра",
})
MATERIAL_VALUES = {
    "сталь": 1, "стальной": 1, "steel": 1, "1": 1,
    "композит": 0, "композитный": 0, "composite": 0, "0": 0,
}


def normalized_material_name(name: str) -> str:
    """Normalize casing, underscores and whitespace for known column aliases."""
    return " ".join(name.replace("_", " ").casefold().split())


def is_material_column(name: str) -> bool:
    return normalized_material_name(name) in MATERIAL_COLUMN_NAMES


def steel_feature_spec(source: str | None = None) -> FeatureSpec:
    """Defaults for an explicit or computed binary material indicator."""
    return FeatureSpec("is_steel", dtype="binary", formula_allowed=False,
                       derived_from=source)


def _binary_values(values: pd.Series) -> pd.Series:
    missing = int(values.isna().sum())
    if missing:
        raise DatasetError(f"Колонка is_steel содержит {missing} пропущенных значений.")
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DatasetError("Колонка is_steel должна содержать числовые значения 0 или 1.") from exc
    invalid = ~numeric.isin([0, 1])
    if invalid.any():
        raise DatasetError(f"Колонка is_steel содержит {int(invalid.sum())} значений, отличных от 0 и 1.")
    return numeric.astype(int)


def add_material_feature(df: pd.DataFrame, schema: DatasetSchema | None = None
                         ) -> tuple[pd.DataFrame, str | None]:
    """Create/verify is_steel from a named material column, retaining the source.

    This fixed mapping learns nothing from the dataset or target. Unknown and
    missing materials are errors, never an implicit composite class. Explicit
    target/group roles do not generate an input feature from their values.
    """
    out = df.copy()
    if "is_steel" in out:
        out["is_steel"] = _binary_values(out["is_steel"])
    sources = [name for name in out.columns if is_material_column(name)]
    if schema is not None:
        protected = {c.name for c in schema.columns
                     if c.role in (ColumnRole.TARGET, ColumnRole.GROUP)}
        sources = [name for name in sources if name not in protected]
    if not sources:
        return out, None
    if len(sources) > 1:
        raise DatasetError("Неоднозначный источник is_steel: найдены колонки материала "
                           f"{', '.join(sources)}. Оставьте одну колонку материала.")
    source = sources[0]
    missing = int(out[source].isna().sum())
    if missing:
        raise DatasetError(f"Колонка {source} содержит {missing} пропущенных значений материала.")
    labels = out[source].astype(str).map(lambda value: " ".join(value.casefold().split()))
    # Numeric columns may be read as floats by Excel/pandas.
    labels = labels.replace({"0.0": "0", "1.0": "1"})
    encoded = labels.map(MATERIAL_VALUES)
    invalid = encoded.isna()
    if invalid.any():
        examples = ", ".join(repr(v) for v in pd.unique(out.loc[invalid, source])[:5])
        raise DatasetError(f"Колонка {source} содержит {int(invalid.sum())} неизвестных значений материала: "
                           f"{examples}. Допустимы сталь/стальной/steel и композит/композитный/composite (или 1/0).")
    encoded = encoded.astype(int)
    if "is_steel" in out:
        mismatch = out["is_steel"] != encoded
        if mismatch.any():
            raise DatasetError(f"Колонки {source} и is_steel противоречат друг другу в "
                               f"{int(mismatch.sum())} строках. Исправьте материал или числовой признак.")
    else:
        out["is_steel"] = encoded
    return out, source
