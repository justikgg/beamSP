"""Recheck the historical fits and test web-area models on the same profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.loader import load_dataset_bundle  # noqa: E402
from core.data.schema import Dataset, group_ids  # noqa: E402
from core.evaluation.validation import leave_one_group_out  # noqa: E402
from core.models.registry import get_model  # noqa: E402
from research.engineering_formula import run_formula_search as previous  # noqa: E402

OUT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
CATALOG_SOURCE = (
    "https://meganorm.ru/mega_doc/norm_update_04112024/gost_gosudarstvennyj-standart/"
    "0/gost_35087-2024_mezhgosudarstvennyy_standart_dvutavry.html"
)
GUIDE_SOURCE = "https://files.stroyinf.ru/Data2/1/4293850/4293850413.htm"
# ГОСТ 35087-2024, Table 1: nominal H, s -> designation, flange thickness t.
CATALOG = {(140.0, 4.7): ("14Б2", 6.9), (160.0, 5.0): ("16Б2", 7.4),
           (200.0, 5.5): ("20Б1", 8.0)}
SEEDS = (1337, 42, 20261120)
REGISTRY_MODELS = ("linear", "bayes_symreg", "symreg", "bio_ga", "bio_de", "bio_pso", "bio_cmaes")


@dataclass(frozen=True)
class Formula:
    name: str
    terms: tuple[str, ...] = ()
    loss: str = "absolute"
    penalty: float = 0.0
    penalized_terms: int = 1
    offset: str | None = None
    factor: float = 0.0
    digits: int | None = None
    catalog: bool = False
    diagnostic: bool = False
    quantize: bool = False
    original: previous.Candidate | None = None


def formulas() -> list[Formula]:
    original = [Formula(spec.name, original=spec) for spec in previous.candidates()]
    return original + [
        Formula("nominal_full_height", offset="X", factor=0.8),
        Formula("stress_area", ("X",)),
        Formula("material_area", ("X_c", "X_s")),
        Formula("material_area_paper", ("X_c", "X_s"), digits=3),
        Formula("shared_depth", ("X", "T")),
        Formula("shared_flange_ratio", ("X", "Ts")),
        Formula("shared_slenderness", ("X", "q_H_over_s")),
        Formula("shared_area_curvature", ("X", "q_Hs")),
        Formula("material_with_depth", ("X_c", "X_s", "T")),
        Formula("material_with_flange_ratio", ("X_c", "X_s", "Ts")),
        Formula("material_with_slenderness", ("X_c", "X_s", "q_H_over_s")),
        Formula("material_with_area_curvature", ("X_c", "X_s", "q_Hs")),
        Formula("steel_depth", ("X_c", "X_s", "T_s_centered")),
        Formula("steel_depth_paper", ("X_c", "X_s", "T_s_centered"), digits=4),
        Formula("steel_depth_ridge", ("X_c", "X_s", "T_s_centered"), penalty=1e-5),
        Formula("steel_height", ("X_c", "X_s", "q_H_s")),
        Formula("steel_height_ridge", ("X_c", "X_s", "q_H_s"), penalty=1e-4),
        Formula("steel_thickness_ridge", ("X_c", "X_s", "q_s_s"), penalty=1e-4),
        Formula("separate_depth_ridge", ("X_c", "X_s", "T_c_centered", "T_s_centered"),
                penalty=1e-4, penalized_terms=2),
        Formula("separate_flange_ratio_ridge", ("X_c", "X_s", "Ts_c", "Ts_s"),
                penalty=1e-4, penalized_terms=2),
        Formula("HSR_relative", ("log_H", "log_s", "log_R"), loss="relative_power"),
        Formula("HSR_robust", ("log_H", "log_s", "log_R"), loss="robust_power"),
        Formula("HSR_shrink_to_area", ("log_H", "log_s", "log_R"), loss="shrink_power"),
        Formula("HR_height_only", ("log_H", "log_R"), loss="area_power"),
        Formula("sR_thickness_only", ("log_s", "log_R"), loss="area_power"),
        Formula("HR_common_material", ("log_H",), loss="stress_power"),
        Formula("web_proxy_nominal", offset="P", factor=0.8),
        Formula("web_proxy_calibrated", ("P",)),
        Formula("web_nominal", offset="W", factor=0.8, catalog=True),
        Formula("web_calibrated", ("W",), catalog=True),
        Formula("web_separate_material", ("W_c", "W_s"), catalog=True),
        Formula("web_fit_composite_flange", ("W", "Ts_c"), catalog=True),
        Formula("web_fit_composite_nominal", ("Ts_c",), offset="W", factor=0.8, catalog=True),
        Formula("web_with_rounding", offset="W", factor=0.8, catalog=True,
                diagnostic=True, quantize=True),
        Formula("web_proxy_with_rounding", offset="P", factor=0.8,
                diagnostic=True, quantize=True),
    ]


def web_height(frame: pd.DataFrame) -> np.ndarray:
    """Use a nominal section match for steel and t=s as a composite proxy."""
    height = frame["H"].to_numpy(float)
    thickness = frame["s"].to_numpy(float)
    steel = frame["is_steel"].to_numpy(bool)
    flange = thickness.copy()
    for index in np.flatnonzero(steel):
        shape = height[index], thickness[index]
        if shape not in CATALOG:
            raise ValueError(f"No catalog section in this study for H={shape[0]}, s={shape[1]}")
        flange[index] = CATALOG[shape][1]
    result = height - 2 * flange
    if np.any(result <= 0):
        raise ValueError("Clear web height must be positive")
    return result


def basis(frame: pd.DataFrame, *, catalog: bool = False) -> dict[str, np.ndarray]:
    h, s, r, m = frame[["H", "s", "R", "is_steel"]].to_numpy(float).T
    if not np.isfinite([h, s, r, m]).all() or np.any(np.minimum(np.minimum(h, s), r) <= 0):
        raise ValueError("H, s and R must be finite and positive")
    if not np.isin(m, [0, 1]).all():
        raise ValueError("is_steel must be 0 or 1")
    t = r * s / 1000
    x = t * h
    comp = 1 - m
    values = {
        "A": h * s / 1000, "X": x, "X_c": x * comp, "X_s": x * m, "T": t, "Ts": t * s,
        "P": t * (h - (2 + m) * s),
        "Ts_c": t * s * comp, "Ts_s": t * s * m,
        "T_c_centered": t * (150 - h) * comp, "T_s_centered": t * (150 - h) * m,
        "q_H_s": x * (h / 150 - 1) * m, "q_s_s": x * (s / 6 - 1) * m,
        "q_H_over_s": x * ((h / s) / 25 - 1), "q_Hs": x * (h * s / 900 - 1),
        "log_H": np.log(h / 150), "log_s": np.log(s / 6), "log_R": np.log(r / 30),
    }
    if catalog:
        w = t * web_height(frame)
        values.update(W=w, W_c=w * comp, W_s=w * m)
    return values


def rounding_grid(y: np.ndarray) -> float:
    cents = [Decimal(str(value)) * 100 for value in y]
    if any(value != value.to_integral_value() for value in cents):
        raise ValueError("The rounding diagnostic assumes values stored to centi-kN")
    return math.gcd(*(int(value) for value in cents)) / 100


def fit_power(y: np.ndarray, spec: Formula, values: dict[str, np.ndarray]) -> list[float]:
    design = np.column_stack([np.ones(len(y)), *(values[term] for term in spec.terms)])
    anchor = np.zeros(len(y))
    if spec.loss in {"shrink_power", "area_power", "stress_power"}:
        anchor = np.log(values["X"] if spec.loss == "stress_power" else values["A"])
    initial = np.linalg.lstsq(design, np.log(y) - anchor, rcond=None)[0]
    if spec.loss == "relative_power":
        fit = least_squares(lambda params: np.exp(anchor + design @ params) / y - 1, initial)
    elif spec.loss == "robust_power":
        fit = least_squares(lambda params: np.exp(anchor + design @ params) - y,
                            initial, loss="soft_l1", f_scale=0.25)
    elif spec.loss == "shrink_power":
        # Penalize changes to the geometry powers around the unit area exponents.
        augmented = np.vstack([design, np.diag([0., 0.05, 0.05, 0.])])
        response = np.concatenate([np.log(y) - anchor, np.zeros(4)])
        return np.linalg.lstsq(augmented, response, rcond=None)[0].tolist()
    else:
        return initial.tolist()
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x.tolist()


def fit(frame: pd.DataFrame, y: np.ndarray, spec: Formula) -> dict[str, Any]:
    if spec.original:
        fitted = previous.fit_candidate(frame, y, spec.original)
        if not fitted.pop("success"):
            raise RuntimeError(f"{spec.name} failed to converge")
        fitted.pop("seconds")
        if not math.isfinite(fitted["condition"]):
            fitted["condition"] = None
        return fitted
    if not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("The comparison requires finite positive targets")
    values = basis(frame, catalog=spec.catalog)
    if spec.loss.endswith("power"):
        return {"coefficients": fit_power(y, spec, values)}
    result = {"coefficients": [], "rank": 0}
    if spec.offset:
        y_residual = y - spec.factor * values[spec.offset]
    else:
        y_residual = y
    if spec.terms:
        matrix = np.column_stack([values[term] for term in spec.terms])
        result["rank"] = int(np.linalg.matrix_rank(matrix))
        if spec.penalty:
            weights = np.zeros(len(spec.terms))
            weights[-spec.penalized_terms:] = np.sqrt(spec.penalty)
            matrix = np.vstack([matrix / values["X"][:, None], np.diag(weights)])
            y_residual = np.concatenate([y_residual / values["X"], np.zeros(len(weights))])
        coefficients = np.linalg.lstsq(matrix, y_residual, rcond=None)[0]
        result["coefficients"] = ([float(f"{c:.{spec.digits}g}") for c in coefficients]
                                  if spec.digits else coefficients.tolist())
    if spec.quantize:
        result["rounding_grid_kN"] = rounding_grid(y)
    return result


def predict(frame: pd.DataFrame, spec: Formula, fitted: dict[str, Any]) -> np.ndarray:
    if spec.original:
        return previous.predict(frame, fitted)
    values = basis(frame, catalog=spec.catalog)
    if ("P" in spec.terms or spec.offset == "P") and np.any(values["P"] <= 0):
        raise ValueError("Approximate clear web height must be positive")
    coefficients = fitted["coefficients"]
    if spec.loss.endswith("power"):
        pred = np.full(len(frame), coefficients[0])
        for term, coefficient in zip(spec.terms, coefficients[1:], strict=True):
            pred += coefficient * values[term]
        pred = np.exp(pred)
        if spec.loss in {"shrink_power", "area_power", "stress_power"}:
            pred *= values["X"] if spec.loss == "stress_power" else values["A"]
    else:
        pred = spec.factor * values[spec.offset] if spec.offset else np.zeros(len(frame))
        for term, coefficient in zip(spec.terms, coefficients, strict=True):
            pred += coefficient * values[term]
    if spec.quantize:
        step = fitted["rounding_grid_kN"]
        pred = step * np.floor(pred / step + 0.5)
    if not np.isfinite(pred).all():
        raise RuntimeError(f"Nonfinite predictions from {spec.name}")
    return pred


def supported(fitted: dict[str, Any], spec: Formula) -> bool:
    # A geometric coefficient can lose rank when only one steel profile remains.
    return spec.penalty > 0 or fitted.get("rank", len(spec.terms)) >= len(spec.terms)


def evaluate(frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
             spec: Formula, held_count: int = 1) -> dict[str, Any]:
    folds = []
    actual, predicted = [], []
    for held_groups in combinations(np.unique(groups).tolist(), held_count):
        held = np.isin(groups, held_groups)
        fitted = fit(frame.loc[~held], y[~held], spec)
        fold_pred = predict(frame.loc[held], spec, fitted)
        actual.extend(y[held])
        predicted.extend(fold_pred)
        folds.append({"held_groups": held_groups, "row_ids": np.flatnonzero(held).tolist(),
                      "predict_kN": fold_pred.tolist(), "fit": fitted,
                      "rank_supported": supported(fitted, spec)})
    return {"metrics": previous.metrics(np.asarray(actual), np.asarray(predicted)),
            "rank_supported_folds": sum(fold["rank_supported"] for fold in folds),
            "folds": folds}


def nested(frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
           pool: list[Formula]) -> dict[str, Any]:
    folds, predictions = [], np.full(len(y), np.nan)
    for held in np.unique(groups):
        test = groups == held
        scores = {}
        for spec in pool:
            cv = evaluate(frame.loc[~test], y[~test], groups[~test], spec)
            if cv["rank_supported_folds"] == len(cv["folds"]):
                scores[spec.name] = cv["metrics"]["RMSE_kN"]
        best = min(scores.values())
        tolerance = max(1e-7, 1e-6 * best)
        winner = min((spec for spec in pool if scores.get(spec.name, np.inf) <= best + tolerance),
                     key=lambda spec: len(spec.original.features) if spec.original else len(spec.terms))
        fitted = fit(frame.loc[~test], y[~test], winner)
        predictions[test] = predict(frame.loc[test], winner, fitted)
        folds.append({"held_group": int(held), "selected": winner.name,
                      "inner_RMSE_kN": scores, "fit": fitted})
    return {"metrics": previous.metrics(y, predictions), "predictions_kN": predictions.tolist(),
            "folds": folds}


def registry_benchmarks(dataset: Dataset, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    schema = dataset.schema
    names = schema.features
    x = dataset.frame[names].to_numpy(float)
    result = {}
    for name in REGISTRY_MODELS:
        seeds = SEEDS if get_model(name).stochastic else SEEDS[:1]
        runs = []
        for seed in seeds:
            def create():
                return get_model(name).set_seed(seed).configure_features(names, schema)

            pred = leave_one_group_out(create, x, y, groups)
            model = create().fit(x, y)
            runs.append({"seed": seed, "metrics": previous.metrics(y, pred),
                         "predictions_kN": pred.tolist(), "formula_all_profiles": model.formula(names)})
        result[name] = runs
        LOG.info("%s %s", name, [round(run["metrics"]["RMSE_kN"], 4) for run in runs])
    return result


def run(output: Path, *, registry: bool = True) -> None:
    source = ROOT / "config/data.xlsx"
    dataset = load_dataset_bundle(source)
    frame = dataset.frame
    groups = group_ids(frame, dataset.schema)
    old_path = OUT / "results.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    mapping_hash = hashlib.sha256(source.with_suffix(".beamsp.json").read_bytes()).hexdigest()
    if (source_hash, mapping_hash) != (old["source_sha256"], old["mapping_sha256"]):
        raise ValueError("Source or mapping differs from the archived experiment")

    specs = formulas()
    original = [spec for spec in specs if spec.original and not spec.original.baseline]
    general = [spec for spec in specs if not spec.catalog and not spec.diagnostic
               and (spec.original is None or not spec.original.baseline)]
    with_catalog = general + [spec for spec in specs if spec.catalog and not spec.diagnostic]
    payload: dict[str, Any] = {
        "source": str(source.relative_to(ROOT)), "source_sha256": source_hash,
        "mapping_sha256": mapping_hash, "versions": {"python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__},
        "catalog_source": CATALOG_SOURCE, "guide_source": GUIDE_SOURCE,
        "catalog": [{"H": h, "s": s, "designation": name, "t_f": t}
                    for (h, s), (name, t) in CATALOG.items()],
        "rows": len(frame), "profiles": len(np.unique(groups)), "synthetic_rows": 0,
        "candidates": [asdict(spec) for spec in specs], "targets": {},
    }
    records = []
    for target, column in dataset.schema.targets.items():
        y = frame[column].to_numpy(float)
        entries = {}
        for spec in specs:
            if spec.quantize and target != "RUK78":
                continue
            logo = evaluate(frame, y, groups, spec)
            entry = {"LOGO": logo, "leave_two": evaluate(frame, y, groups, spec, held_count=2),
                     "all_profile_fit": fit(frame, y, spec)}
            if spec.original:
                archived = old["targets"][target]["candidates"][spec.name]["LOGO"]
                entry["archive_RMSE_difference_kN"] = logo["metrics"]["RMSE_kN"] - archived["RMSE_kN"]
            for fold in logo["folds"]:
                for index, prediction in zip(fold["row_ids"], fold["predict_kN"], strict=True):
                    records.append({"target": target, "candidate": spec.name,
                        "profile": int(groups[index]), "excel_row": dataset.export_context["rows"][index],
                        "a_h0": frame.iloc[index]["a_h0"], "Q_table_kN": y[index],
                        "Q_oof_kN": prediction, "error_kN": prediction - y[index]})
            entries[spec.name] = entry
        transfer = {}
        for name in ("nominal_full_height", "stress_area", "HSR_trf", "web_nominal",
                     "web_calibrated", "web_fit_composite_nominal", "web_proxy_nominal",
                     "web_with_rounding", "web_proxy_with_rounding"):
            spec = next(item for item in specs if item.name == name)
            if spec.quantize and target != "RUK78":
                continue
            transfer[name] = []
            material = frame["is_steel"].to_numpy(bool)
            for held_steel in (False, True):
                test = material == held_steel
                fitted = fit(frame.loc[~test], y[~test], spec)
                pred = predict(frame.loc[test], spec, fitted)
                transfer[name].append({"held_steel": held_steel, "fit": fitted,
                    "rank_supported": fitted.get("rank", 0) >= fitted.get("n_parameters", len(spec.terms)),
                    "metrics": previous.metrics(y[test], pred)})
        payload["targets"][target] = {
            "column": column, "candidates": entries, "material_transfer": transfer,
            "nested": {"original": nested(frame, y, groups, original),
                       "general": nested(frame, y, groups, general),
                       "with_catalog": nested(frame, y, groups, with_catalog)},
            "registry": registry_benchmarks(dataset, y, groups) if registry else {},
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "recheck_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(records).to_csv(output / "recheck_predictions.csv", index=False, float_format="%.15g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--skip-registry", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.output, registry=not args.skip_registry)
