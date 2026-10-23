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
from core.models.baseline.ridge import RidgeModel
from core.models.baseline.lasso import LassoModel
from core.models.baseline.elastic_net import ElasticNetModel

MODELS = {"ridge": RidgeModel, "lasso": LassoModel, "elasticnet": ElasticNetModel}

SELECTS_FEATURES = {"lasso", "elasticnet"}

DEFAULT_ALPHAS = {
    "ridge": [0, 0.1, 1, 3, 10, 30, 50, 100, 150, 200, 300, 1000],
    "lasso": [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10],
    "elasticnet": [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 20],
}

def _n_nonzero(make, X, y, alpha):
    inner = make(alpha).fit(X, y)._m.steps[-1][1]
    return int(np.sum(np.abs(inner.coef_) > 1e-8))

def sweep(make, target_col, alphas, synth_cfg, seed, count_features=False, dataset=None):
    dataset = dataset or load_dataset_bundle(cfg.DATA_PATH)
    df, groups = prepare(dataset.frame, synth_cfg=synth_cfg, seed=seed, schema=dataset.schema)
    X = df[dataset.schema.features].to_numpy(float)
    y = df[target_col].to_numpy(float)
    is_synth = np.asarray(df.attrs["beamsp_synthetic"], bool)

    out = []
    for a in alphas:
        preds = leave_one_group_out(lambda a=a: make(a), X, y, groups, is_synth)
        real = ~is_synth & np.isfinite(preds)
        m = compute_metrics(y[real], preds[real])
        nf = _n_nonzero(make, X, y, a) if count_features else None
        out.append((a, m["R2"], m["RMSE"], nf))
    return out


def _print_table(tkey, rows, show_features):
    best = max(rows, key=lambda r: r[1])
    print(f"\n=== {target_label(tkey)} ===")
    head = f"{'alpha':>8}  {'R²':>7}  {'RMSE':>7}"
    if show_features: head += f"  {'призн.':>6}"
    print(head)
    for a, r2, rmse, nf in rows:
        line = f"{a:>8}  {r2:>7.3f}  {rmse:>7.2f}"
        if show_features: line += f"  {nf:>6}"
        line += " лучшее" if (a, r2, rmse, nf) == best else ""
        print(line)
    print(f"Лучшая alpha по R²: {best[0]} (R²={best[1]:.3f}, RMSE={best[2]:.2f})")
    return best


def build_figure(results, alphas, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tkeys = list(results)
    fig, axes = plt.subplots(1, len(tkeys), figsize=(6 * len(tkeys), 4.5), squeeze=False)
    x = np.arange(len(alphas))
    labels = [f"{a:g}" for a in alphas]
    for ax, tkey in zip(axes[0], tkeys):
        rows = results[tkey]
        r2 = [r[1] for r in rows]
        rmse = [r[2] for r in rows]
        best_i = max(range(len(rows)), key=lambda i: rows[i][1])

        ax.plot(x, r2, "o-", color="#2166ac", label="R²")
        ax.set_ylabel("R²", color="#2166ac")
        ax.tick_params(axis="y", labelcolor="#2166ac")

        ax2 = ax.twinx()
        ax2.plot(x, rmse, "s--", color="#e08a1e", label="RMSE")
        ax2.set_ylabel("RMSE, кН", color="#e08a1e")
        ax2.tick_params(axis="y", labelcolor="#e08a1e")

        ax.axvline(best_i, color="#2a9d3f", ls=":", lw=1.5)
        ax.annotate(f"α = {alphas[best_i]:g}", xy=(best_i, r2[best_i]),
                    xytext=(4, -12), textcoords="offset points", color="#2a9d3f", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_xlabel("alpha")
        ax.set_title(target_label(tkey))
        ax.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Подбор alpha для Ridge/Lasso/ElasticNet (LOGO)")
    p.add_argument("--model", choices=list(MODELS), default="ridge")
    p.add_argument("--target", default="both")
    add_dataset_arguments(p)
    p.add_argument("--alphas", help="список через запятую, напр. 0.1,0.3,1")
    p.add_argument("--l1-ratio", type=float, default=0.5, help="баланс L1/L2 для ElasticNet (0..1, по умолчанию 0.5)")
    p.add_argument("--no-synth", action="store_true", help="отключить синтез")
    p.add_argument("--samples", type=int, help="образцов синтеза на профиль")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--plot", action="store_true", help="сохранить график")
    p.add_argument("--out", type=Path, help="путь для графика (подразумевает --plot)")
    args = p.parse_args(argv)
    runtime = dataset_from_args(args)
    select_targets(runtime.schema, args.target)

    model_cls = MODELS[args.model]
    show_features = args.model in SELECTS_FEATURES
    is_enet = args.model == "elasticnet"

    def make(alpha):
        if is_enet: return model_cls(alpha=alpha, l1_ratio=args.l1_ratio)
        return model_cls(alpha=alpha)

    alphas = ([float(x) for x in args.alphas.split(",")] if args.alphas else DEFAULT_ALPHAS[args.model])
    targets = (list(runtime.schema.targets) if args.target == "both" else [args.target])

    synth_cfg = synthesis_from_args(args, runtime.schema)

    mlabel = method_label(args.model) or args.model
    suffix = f", l1_ratio={args.l1_ratio}" if is_enet else ""
    print(f"Подбор alpha для {mlabel}{suffix} (схема Leave-One-Group-Out)")
    print(f"seed={args.seed}, синтез={'вкл' if synth_cfg['enabled'] else 'выкл'}")

    results = {}
    best = {}
    for tkey in targets:
        rows = sweep(make, runtime.schema.targets[tkey], alphas, synth_cfg, args.seed, count_features=show_features, dataset=runtime)
        results[tkey] = rows
        best[tkey] = _print_table(tkey, rows, show_features)[0]

    if len(best) > 1:
        print("\nИтог по целям:")
        for tkey, a in best.items():
            print(f"  {target_label(tkey)}: alpha = {a}")

    if args.plot or args.out:
        out = args.out or (cfg.RESULTS_DIR / f"alpha_sweep_{args.model}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = build_figure(results, alphas, f"Подбор alpha — {mlabel}{suffix}")
        fig.savefig(out, dpi=150)
        print(f"\nГрафик сохранён: {out}")

    return best


if __name__ == "__main__":
    main()
