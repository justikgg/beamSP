import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from config import settings as cfg
from core.data.loader import load_dataset_bundle
from core.data.cli import add_dataset_arguments, dataset_from_args, select_targets, synthesis_from_args
from core.evaluation.runner import prepare
from core.labels import target_label, method_label
from core.models.registry import get_model


def compute(name, target_col, synth_cfg, seed, n_repeats, dataset=None):
    dataset = dataset or load_dataset_bundle(cfg.DATA_PATH)
    df, _ = prepare(dataset.frame, synth_cfg=synth_cfg, seed=seed, schema=dataset.schema)
    X = df[dataset.schema.features].to_numpy(float)
    y = df[target_col].to_numpy(float)
    real = ~np.asarray(df.attrs["beamsp_synthetic"], bool)

    model = get_model(name).set_seed(seed).configure_features(dataset.schema.features, dataset.schema).fit(X, y)
    Xr, yr = X[real], y[real]
    baseline = r2_score(yr, model.predict(Xr))

    rng = np.random.default_rng(seed)
    n_features = Xr.shape[1]
    drops = np.zeros(n_features)
    for j in range(n_features):
        scores = np.empty(n_repeats)
        for k in range(n_repeats):
            Xp = Xr.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            scores[k] = r2_score(yr, model.predict(Xp))
        drops[j] = baseline - scores.mean()
    return drops


def build_figure(results, title, ylabel, feature_cols=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature_cols = feature_cols if feature_cols is not None else load_dataset_bundle(cfg.DATA_PATH).schema.features
    tkeys = list(results)
    n = len(feature_cols)
    y_pos = np.arange(n)
    width = 0.8 / max(len(tkeys), 1)
    colors = {"SP63": "#2a9d3f", "RUK78": "#e08a1e"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, tkey in enumerate(tkeys):
        vals = results[tkey]
        ax.barh(y_pos + (i - (len(tkeys) - 1) / 2) * width, vals, width,
                label=target_label(tkey), color=colors.get(tkey))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_cols)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Permutation importance для модели на всей выборке (оценка на обучающих реальных данных)")
    p.add_argument("--model", required=True, help="имя модели из реестра, напр. svr")
    p.add_argument("--target", default="both")
    add_dataset_arguments(p)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--no-synth", action="store_true")
    p.add_argument("--samples", type=int)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    runtime = dataset_from_args(args)
    select_targets(runtime.schema, args.target)

    synth_cfg = synthesis_from_args(args, runtime.schema)

    targets = list(runtime.schema.targets) if args.target == "both" else [args.target]
    mlabel = method_label(args.model)

    results = {}
    for tkey in targets:
        vals = compute(args.model, runtime.schema.targets[tkey], synth_cfg, args.seed, args.repeats, dataset=runtime)
        results[tkey] = vals
        print(f"\n=== {target_label(tkey)} ===")
        for feat, v in sorted(zip(runtime.schema.features, vals), key=lambda t: -t[1]):
            print(f"  {feat:<10} {v:.4f}")

    if args.plot or args.out:
        out = args.out or (cfg.RESULTS_DIR / f"{args.model}_importances.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = build_figure(results, f"Важности признаков — {mlabel}", f"падение R² при перемешивании ({args.model})", runtime.schema.features)
        fig.savefig(out, dpi=150)
        print(f"\nГрафик сохранён: {out}")

    return results


if __name__ == "__main__":
    main()
