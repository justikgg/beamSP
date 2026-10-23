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
from core.models.classic_ml.svr import SVRModel

DEFAULT_VALUES = {
    "C": [0.1, 0.3, 1, 3, 10, 30, 50, 100, 300, 1000],
    "epsilon": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.03, 0.1, 0.3, 1],
}


def _num(s):
    if not isinstance(s, str) or s in ("scale", "auto"):
        return s
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def sweep(param, values, fixed, target_col, synth_cfg, seed, dataset=None):
    dataset = dataset or load_dataset_bundle(cfg.DATA_PATH)
    df, groups = prepare(dataset.frame, synth_cfg=synth_cfg, seed=seed, schema=dataset.schema)
    X = df[dataset.schema.features].to_numpy(float)
    y = df[target_col].to_numpy(float)
    is_synth = np.asarray(df.attrs["beamsp_synthetic"], bool)

    def make(v):
        kwargs = dict(fixed)
        kwargs[param] = v
        return SVRModel(**kwargs)

    out = []
    for v in values:
        preds = leave_one_group_out(lambda v=v: make(v), X, y, groups, is_synth)
        real = ~is_synth & np.isfinite(preds)
        m = compute_metrics(y[real], preds[real])
        out.append((v, m["R2"], m["RMSE"]))
    return out


def _print_table(tkey, rows, param):
    best = max(rows, key=lambda r: r[1])
    print(f"\n=== {target_label(tkey)} ===")
    print(f"{param:>8}  {'R²':>7}  {'RMSE':>7}")
    for v, r2, rmse in rows:
        tag = " лучшее" if (v, r2, rmse) == best else ""
        print(f"{v!s:>8}  {r2:>7.3f}  {rmse:>7.2f}{tag}")
    print(f"Лучшее {param} по R²: {best[0]} (R²={best[1]:.3f}, RMSE={best[2]:.2f})")
    return best


def build_figure(results, values, param, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tkeys = list(results)
    fig, axes = plt.subplots(1, len(tkeys), figsize=(6 * len(tkeys), 4.5), squeeze=False)
    x = np.arange(len(values))
    labels = [v if isinstance(v, str) else f"{v:g}" for v in values]
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
        ax.annotate(f"{param} = {labels[best_i]}", xy=(best_i, r2[best_i]),
                    xytext=(4, -12), textcoords="offset points", color="#2a9d3f", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_xlabel(param)
        ax.set_title(target_label(tkey))
        ax.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Подбор гиперпараметров SVR (LOGO)")
    p.add_argument("--param", choices=list(DEFAULT_VALUES), default="C",
                    help="какой гиперпараметр перебирать")
    p.add_argument("--values", help="список через запятую, напр. 1,3,10,30")
    p.add_argument("--C", default=10.0, help="фиксированное C, если не он перебирается")
    p.add_argument("--epsilon", default=0.1, help="фиксированный epsilon, если не он перебирается")
    p.add_argument("--gamma", default="scale", help="фиксированный gamma, если не он перебирается")
    p.add_argument("--target", default="both")
    add_dataset_arguments(p)
    p.add_argument("--no-synth", action="store_true", help="отключить синтез")
    p.add_argument("--samples", type=int, help="образцов синтеза на профиль")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--plot", action="store_true", help="сохранить график")
    p.add_argument("--out", type=Path, help="путь для графика (подразумевает --plot)")
    args = p.parse_args(argv)
    runtime = dataset_from_args(args)
    select_targets(runtime.schema, args.target)

    values = ([_num(v) for v in args.values.split(",")] if args.values
              else DEFAULT_VALUES[args.param])
    fixed = {"C": _num(args.C), "epsilon": _num(args.epsilon), "gamma": _num(args.gamma)}
    fixed.pop(args.param)
    targets = (list(runtime.schema.targets) if args.target == "both" else [args.target])

    synth_cfg = synthesis_from_args(args, runtime.schema)

    mlabel = method_label("svr")
    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
    print(f"Подбор {args.param} для {mlabel} (фиксировано: {fixed_str}), схема Leave-One-Group-Out")
    print(f"seed={args.seed}, синтез={'вкл' if synth_cfg['enabled'] else 'выкл'}")

    results = {}
    best = {}
    for tkey in targets:
        rows = sweep(args.param, values, fixed, runtime.schema.targets[tkey], synth_cfg, args.seed, dataset=runtime)
        results[tkey] = rows
        best[tkey] = _print_table(tkey, rows, args.param)[0]

    if len(best) > 1:
        print("\nИтог по целям:")
        for tkey, v in best.items():
            print(f"  {target_label(tkey)}: {args.param} = {v}")

    if args.plot or args.out:
        out = args.out or (cfg.RESULTS_DIR / f"svr_{args.param}_sweep.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = build_figure(results, values, args.param, f"Подбор {args.param} — {mlabel} ({fixed_str})")
        fig.savefig(out, dpi=150)
        print(f"\nГрафик сохранён: {out}")

    return best


if __name__ == "__main__":
    main()
