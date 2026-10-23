import argparse
from pathlib import Path

import numpy as np

from config import settings as cfg
from core.data.loader import load_dataset_bundle
from core.data.cli import add_dataset_arguments, dataset_from_args, select_targets, synthesis_from_args
from core.evaluation.runner import prepare
from core.evaluation.validation import leave_one_group_out
from core.evaluation.metrics import compute_metrics
from core.labels import target_label, method_label
from core.models.classic_ml.knn import KNNModel

DEFAULT_K = [1, 2, 3, 5, 7, 9, 12, 15, 20, 30]
WEIGHTS = ["uniform", "distance"]


def sweep(target_col, k_values, weights, synth_cfg, seed, dataset=None):
    dataset = dataset or load_dataset_bundle(cfg.DATA_PATH)
    df, groups = prepare(dataset.frame, synth_cfg=synth_cfg, seed=seed, schema=dataset.schema)
    X = df[dataset.schema.features].to_numpy(float)
    y = df[target_col].to_numpy(float)
    is_synth = np.asarray(df.attrs["beamsp_synthetic"], bool)

    out = []
    for w in weights:
        for k in k_values:
            preds = leave_one_group_out(
                lambda k=k, w=w: KNNModel(n_neighbors=k, weights=w), X, y, groups, is_synth)
            real = ~is_synth & np.isfinite(preds)
            m = compute_metrics(y[real], preds[real])
            out.append((k, w, m["R2"], m["RMSE"]))
    return out


def _print_table(tkey, rows):
    best = max(rows, key=lambda r: r[2])
    print(f"\n=== {target_label(tkey)} ===")
    print(f"{'k':>4}  {'weights':>9}  {'R²':>7}  {'RMSE':>7}")
    for k, w, r2, rmse in rows:
        tag = " лучшее" if (k, w, r2, rmse) == best else ""
        print(f"{k:>4}  {w:>9}  {r2:>7.3f}  {rmse:>7.2f}{tag}")
    print(f"Лучшее: k={best[0]}, weights={best[1]} (R²={best[2]:.3f}, RMSE={best[3]:.2f})")
    return best


def build_figure(results, k_values, weights, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tkeys = list(results)
    fig, axes = plt.subplots(1, len(tkeys), figsize=(6 * len(tkeys), 4.5), squeeze=False)
    x = np.arange(len(k_values))
    colors = {"uniform": "#2166ac", "distance": "#e08a1e"}
    for ax, tkey in zip(axes[0], tkeys):
        rows = {w: [r for r in results[tkey] if r[1] == w] for w in weights}
        for w in weights:
            r2 = [r[2] for r in rows[w]]
            ax.plot(x, r2, "o-", color=colors.get(w), label=f"weights={w}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel("n_neighbors")
        ax.set_ylabel("R²")
        ax.set_title(target_label(tkey))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Подбор гиперпараметров KNN (LOGO)")
    p.add_argument("--k", help="список n_neighbors через запятую, напр. 1,3,5,7")
    p.add_argument("--weights", help="список через запятую (uniform,distance)")
    p.add_argument("--target", default="both")
    add_dataset_arguments(p)
    p.add_argument("--no-synth", action="store_true")
    p.add_argument("--samples", type=int)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    runtime = dataset_from_args(args)
    select_targets(runtime.schema, args.target)

    k_values = ([int(v) for v in args.k.split(",")] if args.k else DEFAULT_K)
    weights = (args.weights.split(",") if args.weights else WEIGHTS)
    targets = (list(runtime.schema.targets) if args.target == "both" else [args.target])

    synth_cfg = synthesis_from_args(args, runtime.schema)

    mlabel = method_label("knn")
    print(f"Подбор n_neighbors/weights для {mlabel} (схема Leave-One-Group-Out)")
    print(f"seed={args.seed}, синтез={'вкл' if synth_cfg['enabled'] else 'выкл'}")

    results = {}
    best = {}
    for tkey in targets:
        rows = sweep(runtime.schema.targets[tkey], k_values, weights, synth_cfg, args.seed, dataset=runtime)
        results[tkey] = rows
        best[tkey] = _print_table(tkey, rows)

    if len(best) > 1:
        print("\nИтог по целям:")
        for tkey, b in best.items():
            print(f"  {target_label(tkey)}: k={b[0]}, weights={b[1]}")

    if args.plot or args.out:
        out = args.out or (cfg.RESULTS_DIR / "knn_k_sweep.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = build_figure(results, k_values, weights, f"Подбор k — {mlabel}")
        fig.savefig(out, dpi=150)
        print(f"\nГрафик сохранён: {out}")

    return best


if __name__ == "__main__":
    main()
