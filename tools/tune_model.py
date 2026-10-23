import argparse
import itertools

import numpy as np

from config import settings as cfg
from core.data.loader import load_dataset_bundle
from core.data.cli import add_dataset_arguments, dataset_from_args, select_targets, synthesis_from_args
from core.evaluation.runner import prepare
from core.evaluation.validation import leave_one_group_out
from core.evaluation.metrics import compute_metrics
from core.labels import method_label, target_label
from core.models.registry import model_class


def _num(s):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def parse_grid(pairs):
    grid = {}
    for pair in pairs:
        name, _, vals = pair.partition("=")
        grid[name] = [_num(v) for v in vals.split(",")]
    return grid


def _fmt(params):
    return " ".join(f"{k}={v}" for k, v in params.items()) or "(дефолт)"


def _logo(cls, params, target_col, synth_cfg, seed, dataset=None):
    dataset = dataset or load_dataset_bundle(cfg.DATA_PATH)
    df, groups = prepare(dataset.frame, synth_cfg=synth_cfg, seed=seed, schema=dataset.schema)
    X = df[dataset.schema.features].to_numpy(float)
    y = df[target_col].to_numpy(float)
    is_synth = np.asarray(df.attrs["beamsp_synthetic"], bool)
    preds = leave_one_group_out(lambda: cls(**params).set_seed(seed).configure_features(dataset.schema.features, dataset.schema), X, y, groups, is_synth)
    real = ~is_synth & np.isfinite(preds)
    m = compute_metrics(y[real], preds[real])
    return m["R2"], m["RMSE"]


def _r2_train(cls, params, target_col, synth_cfg, seed, dataset=None):
    dataset = dataset or load_dataset_bundle(cfg.DATA_PATH)
    df, _ = prepare(dataset.frame, synth_cfg=synth_cfg, seed=seed, schema=dataset.schema)
    X = df[dataset.schema.features].to_numpy(float)
    y = df[target_col].to_numpy(float)
    real = ~np.asarray(df.attrs["beamsp_synthetic"], bool)
    model = cls(**params).set_seed(seed).configure_features(dataset.schema.features, dataset.schema).fit(X, y)
    from sklearn.metrics import r2_score
    return float(r2_score(y[real], model.predict(X[real])))


def main(argv=None):
    p = argparse.ArgumentParser(description="Подбор гиперпараметров модели (LOGO)")
    p.add_argument("--model", required=True, help="имя модели из реестра, напр. gbr")
    p.add_argument("--grid", nargs="*", default=[], help="параметры: name=v1,v2,… (перебор перекрёстно)")
    p.add_argument("--target", default="both")
    add_dataset_arguments(p)
    p.add_argument("--no-synth", action="store_true")
    p.add_argument("--samples", type=int)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    args = p.parse_args(argv)
    runtime = dataset_from_args(args)
    select_targets(runtime.schema, args.target)

    cls = model_class(args.model)
    grid = parse_grid(args.grid)
    keys = list(grid)
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))] or [{}]
    targets = list(runtime.schema.targets) if args.target == "both" else [args.target]

    synth_cfg = synthesis_from_args(args, runtime.schema)

    print(f"Подбор параметров '{method_label(args.model)}' (LOGO): "
          f"конфигов {len(configs)}, seed={args.seed}")

    for tkey in targets:
        rows = []
        for params in configs:
            r2, rmse = _logo(cls, params, runtime.schema.targets[tkey], synth_cfg, args.seed, dataset=runtime)
            r2tr = _r2_train(cls, params, runtime.schema.targets[tkey], synth_cfg, args.seed, dataset=runtime)
            rows.append((params, r2, rmse, r2tr - r2))
        best = max(rows, key=lambda r: r[1])
        print(f"\n=== {target_label(tkey)} ===")
        print(f"{'параметры':<44} {'R²':>7} {'RMSE':>7} {'overfit':>8}")
        for params, r2, rmse, overfit in rows:
            tag = " лучшее" if params is best[0] else ""
            print(f"{_fmt(params):<44} {r2:>7.3f} {rmse:>7.2f} {overfit:>8.3f}{tag}")
        print(f"Лучшее: {_fmt(best[0])} → R²={best[1]:.3f}, overfit={best[3]:.3f}")


if __name__ == "__main__":
    main()
