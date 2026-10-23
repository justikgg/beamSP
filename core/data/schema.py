"""Runtime column roles, validation and portable dataset mappings."""
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path

import numpy as np
import pandas as pd


class DatasetError(ValueError):
    """A user-facing dataset or mapping error."""


class ColumnRole(str, Enum):
    FEATURE = "feature"
    TARGET = "target"
    GROUP = "group"
    METADATA = "metadata"
    IGNORE = "ignore"


@dataclass
class ColumnSpec:
    name: str
    role: ColumnRole = ColumnRole.IGNORE
    display_name: str | None = None
    dtype: str = "numeric"
    unit: str = ""
    enabled: bool = True
    minimum: float | None = None
    maximum: float | None = None
    synthetic_noise: float | None = None
    formula_allowed: bool = True
    positive_only: bool = False
    monotonic: int = 0
    target_key: str | None = None
    derived_from: str | None = None

    def __post_init__(self):
        self.role = ColumnRole(self.role)


@dataclass
class FeatureSpec(ColumnSpec):
    role: ColumnRole = ColumnRole.FEATURE


@dataclass
class TargetSpec(ColumnSpec):
    role: ColumnRole = ColumnRole.TARGET


@dataclass
class DatasetSchema:
    columns: list[ColumnSpec] = field(default_factory=list)
    grouping_columns: list[str] = field(default_factory=list)
    legacy: bool = False

    @property
    def features(self) -> list[str]:
        return [c.name for c in self.columns if c.enabled and c.role == ColumnRole.FEATURE]

    @property
    def targets(self) -> dict[str, str]:
        return {c.target_key or c.name: c.name for c in self.columns
                if c.enabled and c.role == ColumnRole.TARGET}

    @property
    def group_column(self) -> str | None:
        return next((c.name for c in self.columns
                     if c.enabled and c.role == ColumnRole.GROUP), None)

    @property
    def noise(self) -> dict[str, float]:
        return {c.name: c.synthetic_noise for c in self.columns
                if c.name in self.features and c.synthetic_noise is not None}

    def spec(self, name: str) -> ColumnSpec:
        return next(c for c in self.columns if c.name == name)

    def to_dict(self) -> dict:
        return {"version": 1, "columns": [asdict(c) for c in self.columns],
                "grouping_columns": list(self.grouping_columns), "legacy": self.legacy}

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetSchema":
        if data.get("version", 1) != 1:
            raise DatasetError("Неподдерживаемая версия схемы датасета.")
        return cls([ColumnSpec(**c) for c in data["columns"]],
                   list(data.get("grouping_columns", [])), bool(data.get("legacy", False)))

    @classmethod
    def from_mapping(cls, columns, features, targets, group_column=None,
                     grouping_columns=None) -> "DatasetSchema":
        """Build a schema, rejecting overlapping or duplicate selections."""
        features = list(features)
        targets = dict(targets) if isinstance(targets, dict) else {n: n for n in targets}
        selected = features + list(targets.values()) + ([group_column] if group_column else [])
        duplicate = pd.Index(selected)[pd.Index(selected).duplicated()].tolist()
        if duplicate:
            raise DatasetError(f"Пересечение ролей или повторный выбор колонки: {', '.join(duplicate)}. "
                               "Target и group нельзя включать в features.")
        missing = set(selected) - set(columns)
        if missing:
            raise DatasetError(f"Колонки отсутствуют в датасете: {', '.join(sorted(missing))}.")
        keys = {v: k for k, v in targets.items()}
        specs = [FeatureSpec(n) for n in features]
        specs += [TargetSpec(n, target_key=keys[n]) for n in targets.values()]
        if group_column:
            specs.append(ColumnSpec(group_column, ColumnRole.GROUP, dtype="categorical"))
        specs += [ColumnSpec(n) for n in columns if n not in selected]
        return cls(specs, list(grouping_columns or []))


@dataclass
class Dataset:
    frame: pd.DataFrame
    schema: DatasetSchema
    source: Path | None = None

    def validate(self, require_logo: bool = True) -> "Dataset":
        self.frame = validate_dataset(self.frame, self.schema, require_logo)
        return self


def validate_headers(columns) -> None:
    """Reject missing and duplicate names before pandas can rename them."""
    names = list(columns)
    if any(pd.isna(n) or not isinstance(n, str) or not n.strip() for n in names):
        raise DatasetError("В строке заголовков есть пустое или нестроковое название колонки.")
    duplicates = pd.Index(names)[pd.Index(names).duplicated()].tolist()
    if duplicates:
        raise DatasetError(f"Повторяющиеся названия колонок: {', '.join(duplicates)}.")


def group_ids(df: pd.DataFrame, schema: DatasetSchema) -> np.ndarray:
    """Prefer explicit IDs; otherwise factorize user-selected profile columns."""
    columns = [schema.group_column] if schema.group_column else schema.grouping_columns
    if not columns:
        raise DatasetError("Для LOGO выберите Group или колонки, определяющие профиль.")
    for name in columns:
        if name not in df:
            raise DatasetError(f"Колонка группы {name} отсутствует в датасете.")
        missing = int(df[name].isna().sum())
        if missing:
            raise DatasetError(f"Колонка {name} содержит {missing} пропущенных значений.")
        if pd.api.types.is_numeric_dtype(df[name]) and not np.isfinite(df[name].to_numpy(float)).all():
            raise DatasetError(f"Колонка группы {name} содержит inf.")
    if schema.group_column:
        return df[schema.group_column].to_numpy(copy=True)
    return pd.factorize(pd.MultiIndex.from_frame(df[columns]), sort=False)[0]


def validate_dataset(df: pd.DataFrame, schema: DatasetSchema,
                     require_logo: bool = True) -> pd.DataFrame:
    """Validate and return a numeric copy; never drop or impute bad rows."""
    validate_headers(df.columns)
    if df.empty:
        raise DatasetError("Датасет пуст.")
    names = [c.name for c in schema.columns]
    if len(names) != len(set(names)):
        raise DatasetError("Пересечение ролей: колонка повторяется в schema. Target и group нельзя включать в features.")
    missing = set(names) - set(df.columns)
    if missing:
        raise DatasetError(f"Колонки отсутствуют в датасете: {', '.join(sorted(missing))}.")
    if not schema.features:
        raise DatasetError("Выберите хотя бы один Feature.")
    if not schema.targets:
        raise DatasetError("Выберите хотя бы один Target.")
    target_specs = [c for c in schema.columns if c.enabled and c.role == ColumnRole.TARGET]
    if len(schema.targets) != len(target_specs):
        raise DatasetError("Ключи targets должны быть уникальными.")
    if sum(c.enabled and c.role == ColumnRole.GROUP for c in schema.columns) > 1:
        raise DatasetError("Выберите только одну колонку Group.")
    if set(schema.grouping_columns) & {c.name for c in schema.columns if c.role == ColumnRole.TARGET}:
        raise DatasetError("Target нельзя использовать для определения профиля.")
    if len(schema.grouping_columns) != len(set(schema.grouping_columns)):
        raise DatasetError("Колонки, определяющие профиль, не должны повторяться.")
    validate_feature_sources(schema)
    out = df.copy()
    for name in schema.features + list(schema.targets.values()):
        spec = schema.spec(name)
        for field_name in ("minimum", "maximum", "synthetic_noise"):
            value = getattr(spec, field_name)
            if value is not None and (not isinstance(value, (int, float, np.number)) or not np.isfinite(value)):
                raise DatasetError(f"Колонка {name}: {field_name} должен быть конечным числом.")
        if spec.minimum is not None and spec.maximum is not None and spec.minimum > spec.maximum:
            raise DatasetError(f"Колонка {name}: min больше max.")
        if spec.dtype not in ("numeric", "binary"):
            raise DatasetError(f"Колонка {name}: требуются числовые features/targets; "
                               "для категорий нужна отдельная стратегия кодирования.")
        missing_count = int(out[name].isna().sum())
        if missing_count:
            raise DatasetError(f"Колонка {name} содержит {missing_count} пропущенных значений.")
        try:
            values = pd.to_numeric(out[name], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise DatasetError(f"Колонка {name} содержит нечисловые значения; требуется float.") from exc
        if not np.isfinite(values).all():
            raise DatasetError(f"Колонка {name} содержит {int((~np.isfinite(values)).sum())} значений NaN / inf.")
        if spec.positive_only and np.any(values <= 0):
            raise DatasetError(f"Колонка {name} должна содержать только положительные значения.")
        if spec.minimum is not None and np.any(values < spec.minimum):
            raise DatasetError(f"Колонка {name} содержит значения ниже min={spec.minimum}.")
        if spec.maximum is not None and np.any(values > spec.maximum):
            raise DatasetError(f"Колонка {name} содержит значения выше max={spec.maximum}.")
        if spec.synthetic_noise is not None and (not np.isfinite(spec.synthetic_noise) or spec.synthetic_noise < 0):
            raise DatasetError(f"Шум колонки {name} должен быть конечным неотрицательным числом.")
        if spec.monotonic not in (-1, 0, 1):
            raise DatasetError(f"Колонка {name}: monotonic должен быть -1, 0 или 1.")
        out[name] = values
    if require_logo and len(pd.unique(group_ids(out, schema))) < 2:
        raise DatasetError("Для LOGO нужны минимум две группы.")
    out.attrs["beamsp_schema"] = schema.to_dict()
    return out


def validate_feature_sources(schema: DatasetSchema, features=None) -> None:
    """Prevent computed features from exposing an explicitly protected column."""
    specs = {c.name: c for c in schema.columns}
    for name in schema.features if features is None else features:
        spec = specs.get(name)
        if spec is None or spec.derived_from is None:
            continue
        source = specs.get(spec.derived_from)
        if source is None:
            raise DatasetError(f"Feature {name}: источник {spec.derived_from} отсутствует в schema.")
        if source.role in (ColumnRole.TARGET, ColumnRole.GROUP):
            raise DatasetError(f"Feature {name} получен из колонки {source.name} с ролью {source.role.value}. "
                               "Target и group нельзя подавать модели через производные признаки; "
                               f"исключите {name} из features или измените роль источника.")


def mapping_path(path) -> Path:
    return Path(path).with_suffix(".beamsp.json")


def save_schema(schema: DatasetSchema, path) -> Path:
    """Save the mapping next to its dataset (or to an explicit JSON path)."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = mapping_path(path)
    path.write_text(json.dumps(schema.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_schema(path) -> DatasetSchema:
    try:
        return DatasetSchema.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (ValueError, TypeError, KeyError) as exc:
        raise DatasetError(f"Некорректный mapping {path}: {exc}") from exc
