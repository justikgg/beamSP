from datetime import datetime
import json
from pathlib import Path
import re
import numpy as np
from core.formula.latexify import to_mathtext
from core.labels import method_label, target_label, Q_UNIT


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _target_label(result, key):
    column = result.metadata.get("targets", {}).get(key)
    specs = result.metadata.get("schema", {}).get("columns", [])
    display = next((s.get("display_name") for s in specs if s["name"] == column), None)
    return display or target_label(key)


def _unit(result, key):
    unit = result.metadata.get("per_target", {}).get(key, {}).get("unit", "")
    return f", {unit}" if unit else ""


def _safe_filename(value):
    return re.sub(r'[^\w.\-]', '_', str(value))[:100] or "target"


def model_notes(result, key):
    notes = []
    models = result.metadata.get("per_target", {}).get(key, {}).get("models", {})
    for name, detail in models.items():
        prefix = f"[{method_label(name)}] "
        excluded = detail.get("excluded_features", [])
        if excluded:
            notes.append(prefix + "Исключены из формулы (Formula выключен): " +
                         ", ".join(c["name"] for c in excluded))
        notes.extend(prefix + message for message in detail.get("diagnostics", {}).get("warnings", []))
        deficient = [str(f["held_out_group"]) for f in detail.get("folds", [])
                     if not f.get("diagnostics", {}).get("identifiable", True)]
        if deficient:
            notes.append(prefix + "Неоднозначные показатели в LOGO при исключении групп: " +
                         ", ".join(deficient))
    return notes

def print_report(result):
    for tkey, table in result.metrics.items():
        print(f"\n=== {_target_label(result, tkey)} ===")
        if table.empty: print("  (нет моделей — только методы вывода формулы)")
        else:
            disp = table.copy()
            disp.index = [method_label(m) for m in disp.index]
            print(disp.round(4).to_string())

        formulas = result.formulas.get(tkey, {})
        if formulas:
            print("\n  Выведенные формулы:")
            for name, expr in formulas.items():
                print(f"  [{method_label(name)}] {expr}")
        for note in model_notes(result, tkey):
            print("  " + note)


def build_comparison_figure(result, tkey, scatter_model=None):
    plt = _pyplot()
    preds = result.predictions.get(tkey, {})
    metrics_df = result.metrics.get(tkey)
    fig = plt.figure(figsize=(11, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    tlabel = _target_label(result, tkey)
    unit = _unit(result, tkey)

    if scatter_model is None and metrics_df is not None and not metrics_df.empty:
        scatter_model = metrics_df["RMSE"].idxmin()

    if scatter_model in preds:
        yt = preds[scatter_model]["y_true"]
        yp = preds[scatter_model]["y_pred"]
        ax1.scatter(yt, yp, s=30)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax1.plot([lo, hi], [lo, hi], "--", color="gray", label="идеал")
        ax1.set_xlabel(f"Эксперимент{unit}")
        ax1.set_ylabel(f"Предсказание{unit}")
        ax1.set_title(f"{method_label(scatter_model)}\n{tlabel}", fontsize=10)
        ax1.legend(fontsize=8)

    if metrics_df is not None and not metrics_df.empty and "RMSE" in metrics_df:
        df = metrics_df.sort_values("RMSE", ascending=False)
        labels = [method_label(m) for m in df.index]
        bars = ax2.barh(range(len(labels)), df["RMSE"].to_numpy())
        bars[-1].set_color("#2a9d3f")
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel(f"RMSE{unit} (меньше — лучше)")
        ax2.set_title(f"Сравнение методов / {tlabel}", fontsize=10)
    fig.tight_layout()
    return fig

def build_fold_rmse_figure(result, tkey):
    fold = result.fold_rmse.get(tkey, {})
    if not fold: return None
    plt = _pyplot()
    methods = list(fold)
    profiles = list(next(iter(fold.values())).keys())
    x = np.arange(len(profiles))
    n = len(methods)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(max(7, len(profiles) * 1.2), 4.5))
    for i, name in enumerate(methods):
        vals = [fold[name].get(p, 0.0) for p in profiles]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=method_label(name) or name)
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"RMSE профиля{_unit(result, tkey)}")
    ax.set_title(f"Ошибка по отложенным профилям / {_target_label(result, tkey)}")
    if n > 1: ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def export_results(result, out_dir, stamp=None):
    plt = _pyplot()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []
    metadata_path = out_dir / f"metadata_{stamp}.json"
    metadata_path.write_text(json.dumps({**result.metadata, "formulas": result.formulas},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    saved.append(metadata_path.name)

    for index, (tkey, df) in enumerate(result.metrics.items()):
        if df.empty: continue
        safe_key = _safe_filename(tkey)
        # Preserve legacy filenames; suffix collisions for arbitrary column names.
        if sum(_safe_filename(k) == safe_key for k in result.metrics) > 1:
            safe_key += f"_{index}"
        path = out_dir / f"metrics_{safe_key}_{stamp}.csv"
        df.to_csv(path, encoding="utf-8-sig")
        saved.append(path.name)
        fig = build_comparison_figure(result, tkey)
        fig_path = out_dir / f"plots_{safe_key}_{stamp}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        saved.append(fig_path.name)

        ff = build_fold_rmse_figure(result, tkey)
        if ff is not None:
            ff_path = out_dir / f"folds_{safe_key}_{stamp}.png"
            ff.savefig(ff_path, dpi=150)
            plt.close(ff)
            saved.append(ff_path.name)

    if any(result.formulas.values()):
        fig = build_formulas_figure(result)
        fpath = out_dir / f"formulas_{stamp}.png"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fpath.name)
    return saved
def build_formulas_figure(result):
    plt = _pyplot()
    specs = result.metadata.get("schema", {}).get("columns", [])
    features = set(result.metadata.get("feature_columns", []))
    descriptions = [f"{s['name']}" + (f" ({s['unit']})" if s.get("unit") else "")
                    for s in specs if s["name"] in features]
    items = [("plain", "Признаки: " + ", ".join(descriptions))]
    for tkey, formulas in result.formulas.items():
        items.append(("target", _target_label(result, tkey)))
        for name, expr in formulas.items():
            items.append(("label", method_label(name)))
            items.append(("formula", expr))

    n = max(len(items), 1)
    line_h = 1.0 / (n + 1)
    fig = plt.figure(figsize=(11, max(4, n * 0.42)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    y = 1.0 - line_h
    for kind, text in items:
        if kind == "target":
            ax.text(0.01, y, text, fontsize=13, fontweight="bold")
        elif kind == "label":
            ax.text(0.02, y, text, fontsize=10, style="italic", color="#333")
        elif kind == "formula":
            fs = 11 if len(text) < 70 else 9 if len(text) < 130 else 7
            if not result.metadata.get("schema", {}).get("legacy", False):
                ax.text(0.03, y, text, fontsize=fs, family="monospace", parse_math=False)
                y -= line_h
                continue
            try:
                from matplotlib.mathtext import MathTextParser
                math = to_mathtext(text)
                MathTextParser("path").parse(math)
                ax.text(0.03, y, math, fontsize=fs)
            except ValueError:
                ax.text(0.03, y, text, fontsize=7, family="monospace", parse_math=False)
        else:
            ax.text(0.02, y, text, fontsize=9, color="#666")
        y -= line_h
    return fig
