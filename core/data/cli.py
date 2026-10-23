"""Shared dataset options for command-line entrypoints and research tools."""
from config import settings as cfg
from .loader import load_dataset_bundle
from .schema import DatasetError, read_schema
from core.evaluation.runner import selected_schema


def add_dataset_arguments(parser):
    parser.add_argument("--data", default=str(cfg.DATA_PATH), help="dataset .csv/.xlsx")
    parser.add_argument("--schema", help="explicit .beamsp.json mapping")
    parser.add_argument("--features", help="ordered feature names, comma-separated")
    parser.add_argument("--targets", help="target column names, comma-separated")
    parser.add_argument("--group", help="explicit group column")
    parser.add_argument("--profile-cols", help="profile definition without group, comma-separated")


def dataset_from_args(args):
    """Resolve a sidecar or command-line mapping and validate it before fitting."""
    schema = read_schema(args.schema) if args.schema else None
    dataset = load_dataset_bundle(args.data, schema=schema, validate=False)
    schema = dataset.schema
    if args.group:
        from .schema import ColumnRole
        if args.group not in dataset.frame:
            raise DatasetError(f"Колонка группы {args.group} отсутствует в датасете.")
        for spec in schema.columns:
            if spec.role == ColumnRole.GROUP:
                spec.role = ColumnRole.IGNORE
            if spec.name == args.group:
                spec.role = ColumnRole.GROUP
    if args.profile_cols:
        schema.grouping_columns = args.profile_cols.split(",")
    targets = {n: n for n in args.targets.split(",")} if args.targets else schema.targets
    features = args.features.split(",") if args.features else schema.features
    # New explicit group/target roles cannot stay in inferred X.
    if args.features is None:
        features = [n for n in features if n not in targets.values() and n != schema.group_column]
    dataset.schema = selected_schema(dataset.frame, schema, features, targets)
    return dataset.validate()


def select_targets(schema, key):
    if key == "both":
        return schema.targets
    if key not in schema.targets:
        raise DatasetError(f"Неизвестная цель {key}. Доступны: {', '.join(schema.targets)}.")
    return {key: schema.targets[key]}


def synthesis_from_args(args, schema):
    return {"enabled": not args.no_synth,
            "samples_per_profile": args.samples if args.samples is not None else cfg.SYNTH["samples_per_profile"],
            "noise": schema.noise}
