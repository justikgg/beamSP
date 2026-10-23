import argparse

from config import settings as cfg
from core.evaluation.runner import run
from core.reporting import print_report, export_results
from core.data.cli import add_dataset_arguments, dataset_from_args, select_targets, synthesis_from_args


def run_method(title, model_names=None, argv=None):

    p = argparse.ArgumentParser(description=title)
    p.add_argument("--target", default="both", help="ключ цели из schema или both")
    add_dataset_arguments(p)
    p.add_argument("--no-synth", action="store_true", help="отключить синтез")
    p.add_argument("--samples", type=int, help="образцов синтеза на профиль")
    p.add_argument("--export", action="store_true", help="сохранить в results/")
    args = p.parse_args(argv)
    dataset = dataset_from_args(args)

    print("=" * 70)
    print(title)
    print("=" * 70)

    targets = select_targets(dataset.schema, args.target)
    synth_cfg = synthesis_from_args(args, dataset.schema)

    def progress(done, total, label):
        print(f"[{done}/{total}] {label}")

    result = run(
        model_names=model_names or [], targets=targets,
        dataset=dataset, synth_cfg=synth_cfg, seed=cfg.SEED,
        progress=progress,
    )
    print_report(result)

    if args.export:
        saved = export_results(result, cfg.RESULTS_DIR)
        print("\nСохранено в results/:")
        for name in saved:
            print(f"{name}")
    return result
