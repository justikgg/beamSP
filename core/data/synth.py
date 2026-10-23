"""Feature-name noise with immutable targets, metadata and parent groups."""
import numpy as np
import pandas as pd
from .loader import schema_for_frame
from .schema import DatasetError, group_ids, validate_dataset


def synthesize(df, samples_per_profile, noise=None, feature_cols=None, seed=1488,
               schema=None, groups=None):
    """Augment real rows, retaining group lineage independently of perturbed X.

    The historical argument order is supported. Unspecified features are unchanged.
    """
    schema = schema or schema_for_frame(df)
    feature_cols = list(feature_cols) if feature_cols is not None else schema.features
    if not isinstance(samples_per_profile, (int, np.integer)) or samples_per_profile < 0:
        raise DatasetError("Число синтетических образцов должно быть целым и неотрицательным.")
    noise = schema.noise if noise is None else dict(noise)
    protected = set()
    if schema.group_column:
        protected.add(schema.group_column)
    for name, amp in noise.items():
        if name not in feature_cols or name not in schema.features or name in protected:
            raise DatasetError(f"Нельзя добавлять шум колонке {name}: это не изменяемый Feature "
                               "(targets и group защищены).")
        if schema.spec(name).dtype != "numeric":
            raise DatasetError(f"Нельзя добавлять шум категориальному/бинарному признаку {name}.")
        if not np.isfinite(amp) or amp < 0:
            raise DatasetError(f"Шум колонки {name} должен быть конечным неотрицательным числом.")
    # Legacy profiles are defined by geometry. Their identity is frozen BEFORE noise,
    # as in v0.17.1, including when a generic profile is defined by geometry.
    rng = np.random.default_rng(seed)
    base = df.copy().reset_index(drop=True)
    parent_groups = np.asarray(groups if groups is not None else group_ids(base, schema))
    records, lineage = [], list(parent_groups)
    for i, (_, row) in enumerate(base.iterrows()):
        for _ in range(samples_per_profile):
            new = row.copy()
            for name, amp in noise.items():
                new[name] = row[name] + rng.uniform(-amp, amp)
            records.append(new)
            lineage.append(parent_groups[i])
    if records:
        extra = pd.DataFrame(records)
        extra.attrs = {}
        base.attrs = {}
        base = pd.concat([base, extra], ignore_index=True)
    mask = [False] * len(df) + [True] * len(records)
    if "is_synth" not in df:
        base["is_synth"] = np.asarray(mask, dtype=int)  # compatibility for old tools
    if "profile_id" not in df:
        base["profile_id"] = lineage
    base.attrs.update(df.attrs)
    base.attrs.update(beamsp_synthetic=mask, beamsp_groups=lineage)
    return validate_dataset(base, schema, require_logo=False)
