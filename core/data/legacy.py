"""The only adapter that knows the v0.17.1 workbook layout and defaults."""
import pandas as pd
from .schema import ColumnRole, ColumnSpec, DatasetError, DatasetSchema, FeatureSpec, TargetSpec

LEGACY_FEATURES = ("a_h0", "is_steel", "H", "s", "R", "E")
LEGACY_TARGETS = {"SP63": "Qдв_СП63", "RUK78": "Qдв_рук78"}
LEGACY_PROFILE_COLUMNS = ["is_steel", "H", "s", "R", "E"]
LEGACY_NOISE = {"H": 3.0, "s": 0.3, "R": 3.0, "E": 50.0}


def legacy_schema() -> DatasetSchema:
    specs = [FeatureSpec(n, dtype="binary" if n == "is_steel" else "numeric",
                         formula_allowed=n != "is_steel", synthetic_noise=LEGACY_NOISE.get(n),
                         monotonic=1 if n in ("H", "s", "R") else 0,
                         unit="мм" if n in ("H", "s") else "МПа" if n in ("R", "E") else "")
             for n in LEGACY_FEATURES]
    specs += [TargetSpec(n, target_key=k, unit="кН") for k, n in LEGACY_TARGETS.items()]
    specs += [ColumnSpec(n, ColumnRole.METADATA, dtype="categorical") for n in ("shifr", "material")]
    return DatasetSchema(specs, list(LEGACY_PROFILE_COLUMNS), legacy=True)


def is_legacy_workbook(raw: pd.DataFrame) -> bool:
    """Recognize the old multirow Russian header, not just material values."""
    return (raw.shape[0] >= 5 and raw.shape[1] >= 11
        and raw.iat[0, 1] == "Исходные данные" and raw.iat[3, 1] == "a/h0"
        and raw.iloc[:, 3].isin(("композитный", "стальной")).any())


def parse_legacy(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in raw.iterrows():
        material = r[3]
        if material not in ("композитный", "стальной"):
            continue
        rows.append({"shifr": r[2], "a_h0": r[1], "material": material,
                     "is_steel": int(material == "стальной"),
                     "H": r[4], "s": r[5], "R": r[6], "E": r[7],
                     "Qдв_СП63": r[9], "Qдв_рук78": r[10]})
    if not rows:
        raise DatasetError("Legacy Excel не содержит строк данных.")
    return pd.DataFrame(rows)


def legacy_profile_label(row) -> str | None:
    if "is_steel" in row and "H" in row:
        material = "сталь" if row["is_steel"] else "композит"
        return f"{material} H={int(row['H'])}"
    return None
