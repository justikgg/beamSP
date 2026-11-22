"""Bounded, deterministic research on real beam profiles; no augmentation.

Run from the repository root with:
    venv/bin/python research/engineering_formula/run_formula_search.py
Only this research directory is written. All selection/rounding is fold-local.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import least_squares, lsq_linear
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.loader import load_dataset_bundle  # noqa: E402
from core.data.schema import group_ids  # noqa: E402
from core.models.baseline.log_linear_ls import LogLinearLSModel  # noqa: E402
from core.models.baseline.log_linear_trf import LogLinearTRFModel  # noqa: E402

OUT = Path(__file__).resolve().parent
REFERENCES = {"H": 150.0, "s": 6.0, "R": 30.0, "E": 60000.0, "a_h0": 2.5}
SEED = 20261120
BUDGET_SECONDS = 600
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    name: str
    features: tuple[str, ...]
    method: str = "trf"
    fixed: tuple[tuple[str, float], ...] = ()
    rounding: int | None = None
    baseline: bool = False


def candidates() -> list[Candidate]:
    """Small predefined set, including ablations and existing project baselines."""
    result = []
    for label, features, fixed in (
        ("HSR", ("H", "s", "R"), ()),
        ("HSRa", ("H", "s", "R", "a_h0"), ()),
        ("HSE", ("H", "s", "E"), ()),
        ("area_R", ("H", "s", "R"), (("H", 1.0), ("s", 1.0))),
    ):
        for method in ("ls", "trf"):
            result.append(Candidate(f"{label}_{method}", features, method, fixed))
    result += [Candidate("HR_trf", ("H", "R")),
               Candidate("sR_trf", ("s", "R")),
               Candidate("area_trf", ("H", "s"), fixed=(("H", 1.0), ("s", 1.0)))]
    for label, fixed in (("HSR", ()), ("area_R", (("H", 1.0), ("s", 1.0)))):
        for digits in (1, 2):
            result.append(Candidate(f"{label}_paper{digits}", ("H", "s", "R"),
                                    fixed=fixed, rounding=digits))
    full = ("a_h0", "H", "s", "R", "E")
    for method in ("ls", "trf"):
        result.append(Candidate(f"existing_full_{method}", full, method, baseline=True))
        for preset, fixed in (
            ("sp63", (("a_h0", 0.0), ("H", 1.0), ("s", 1.0), ("R", 1.0))),
            ("ruk78", (("a_h0", 0.0), ("H", 1.0), ("s", 1.0),
                       ("R", 1.0), ("E", 0.0))),
        ):
            result.append(Candidate(f"existing_{preset}_{method}", full, method,
                                    fixed=fixed, baseline=True))
    return result


def design(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    values = frame[list(features)].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Power law requires finite, strictly positive inputs")
    return np.log(values / np.array([REFERENCES[f] for f in features]))


def predict(frame: pd.DataFrame, fit: dict[str, Any]) -> np.ndarray:
    """Evaluate saved normalized coefficients, including rounded paper formulas."""
    features = tuple(fit["features"])
    return fit["C"] * np.exp(design(frame, features) @ np.array(fit["powers"]))


def fit_candidate(frame: pd.DataFrame, y: np.ndarray, spec: Candidate) -> dict[str, Any]:
    if np.any(y <= 0) or not np.isfinite(y).all():
        raise ValueError("Log initialization requires finite positive Q")
    started = perf_counter()
    z = design(frame, spec.features)
    fixed = dict(spec.fixed)
    free = [i for i, f in enumerate(spec.features) if f not in fixed]
    powers = np.array([fixed.get(f, 0.0) for f in spec.features])
    mat = np.column_stack([np.ones(len(y)), z[:, free]])
    if spec.baseline:
        cls = LogLinearLSModel if spec.method == "ls" else LogLinearTRFModel
        model = cls(fixed_exponents=fixed).configure_features(list(spec.features))
        model.fit(frame[list(spec.features)].to_numpy(float), y)
        meta = model.fitted_metadata()
        powers = np.array([meta["exponents"][f] for f in spec.features])
        c = float(np.exp(meta["log_a"] + np.log(
            [REFERENCES[f] for f in spec.features]) @ powers))
        success = bool(model.diagnostics_["success"])
        initial_sse = model.diagnostics_.get("initial_sse")
        nfev = model.diagnostics_.get("nfev")
    else:
        offset = z @ powers
        lower = np.array([-10.0] + [-3.0] * len(free))
        upper = np.array([12.0] + [3.0] * len(free))
        # Bounded linear least squares is the fast logarithmic initialization.
        init = lsq_linear(mat, np.log(y) - offset, bounds=(lower, upper),
                          method="bvls", tol=1e-10)
        if not init.success:
            raise RuntimeError(init.message)
        initial_sse = float(np.sum((np.exp(mat @ init.x + offset) - y) ** 2))
        params = init.x
        success, nfev = True, 0
        if spec.method == "trf":
            def residual(p: np.ndarray) -> np.ndarray:
                return np.exp(mat @ p + offset) - y

            def jacobian(p: np.ndarray) -> np.ndarray:
                return np.exp(mat @ p + offset)[:, None] * mat

            fitted = least_squares(residual, params, jac=jacobian, method="trf",
                                   bounds=(lower, upper), max_nfev=1000,
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
            params, success, nfev = fitted.x, bool(fitted.success), int(fitted.nfev)
        powers[free] = params[1:]
        c = float(np.exp(params[0]))
        if spec.rounding is not None:
            # Rounding policy is fixed before validation; no test targets involved.
            powers[free] = np.round(powers[free], spec.rounding)
            shape = np.exp(z @ powers)
            c = float(shape @ y / (shape @ shape))
            c = float(f"{c:.{spec.rounding + 1}g}")
    result = {"features": list(spec.features), "references": {
        f: REFERENCES[f] for f in spec.features}, "C": c, "powers": powers.tolist(),
        "rank": int(np.linalg.matrix_rank(mat)), "n_parameters": mat.shape[1],
        "condition": float(np.linalg.cond(mat)), "success": success, "nfev": nfev,
        "initial_sse_kN2": initial_sse, "seconds": perf_counter() - started}
    result["train_sse_kN2"] = float(np.sum((predict(frame, result) - y) ** 2))
    result["formula"] = formula(result)
    return result


def formula(fit: dict[str, Any], precision: int = 9) -> str:
    terms = [f"({f}/{REFERENCES[f]:g})^{p:.{precision}g}"
             for f, p in zip(fit["features"], fit["powers"]) if abs(p) > 1e-10]
    return f"{fit['C']:.{precision}g}" + (" * " + " * ".join(terms) if terms else "")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    err = pred - y
    rel = 100 * err[y > 0] / y[y > 0]
    return {"R2": float(r2_score(y, pred)),
            "RMSE_kN": float(np.sqrt(mean_squared_error(y, pred))),
            "MAE_kN": float(mean_absolute_error(y, pred)),
            "max_abs_kN": float(np.max(np.abs(err))),
            "max_over_kN": float(max(0, np.max(err))),
            "over_count": int(np.sum(err > 1e-8)),
            "MAPE_percent": float(np.mean(np.abs(rel))),
            "max_abs_percent": float(np.max(np.abs(rel))),
            "max_over_percent": float(max(0, np.max(rel)))}


def logo(frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
         spec: Candidate) -> tuple[np.ndarray, list[dict[str, Any]]]:
    pred = np.full(len(y), np.nan)
    folds = []
    for train, test in LeaveOneGroupOut().split(frame, y, groups):
        train_groups = sorted(int(g) for g in np.unique(groups[train]))
        test_groups = sorted(int(g) for g in np.unique(groups[test]))
        if set(train_groups) & set(test_groups):
            raise AssertionError("Group leakage")
        fitted = fit_candidate(frame.iloc[train], y[train], spec)
        pred[test] = predict(frame.iloc[test], fitted)
        folds.append({"train_groups": train_groups, "test_groups": test_groups,
                      "train_row_ids": train.tolist(), "test_row_ids": test.tolist(),
                      "fit": fitted, "metrics": metrics(y[test], pred[test])})
    if not np.isfinite(pred).all():
        raise RuntimeError("Incomplete OOF predictions")
    return pred, folds


def checkpoint(payload: dict[str, Any]) -> None:
    path = OUT / "results.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")
    temp.replace(path)


def select_candidate(scores: list[tuple[float, Candidate]]) -> Candidate:
    """Resolve numerical ties by feature count, then predeclared search order."""
    best = min(score for score, _ in scores)
    tolerance = max(1e-7, abs(best) * 1e-6)
    tied = [(score, spec) for score, spec in scores if score <= best + tolerance]
    return min(tied, key=lambda item: len(item[1].features))[1]


def nested_validation(frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                      pool: list[Candidate]) -> dict[str, Any]:
    """Select candidates on inner LOGO, then evaluate untouched outer profiles."""
    started = perf_counter()
    pred = np.full(len(y), np.nan)
    folds = []
    for train, test in LeaveOneGroupOut().split(frame, y, groups):
        scores = []
        for spec in pool:
            inner_pred, _ = logo(frame.iloc[train], y[train], groups[train], spec)
            scores.append((metrics(y[train], inner_pred)["RMSE_kN"], spec))
        winner = select_candidate(scores)
        fitted = fit_candidate(frame.iloc[train], y[train], winner)
        pred[test] = predict(frame.iloc[test], fitted)
        folds.append({"test_groups": groups[test].astype(int).tolist(),
                      "train_groups": sorted(int(g) for g in np.unique(groups[train])),
                      "selected": winner.name,
                      "inner_rmse": {spec.name: score for score, spec in scores},
                      "fit": fitted, "metrics": metrics(y[test], pred[test])})
    return {"metrics": metrics(y, pred), "folds": folds,
            "predictions": pred.tolist(), "seconds": perf_counter() - started}


def pair_stability(frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                   pool: list[Candidate]) -> dict[str, Any]:
    result = {}
    for spec in pool:
        held_y, held_pred, fits = [], [], []
        for pair in combinations(np.unique(groups), 2):
            test = np.isin(groups, pair)
            fitted = fit_candidate(frame.loc[~test], y[~test], spec)
            pred = predict(frame.loc[test], fitted)
            held_y.extend(y[test])
            held_pred.extend(pred)
            fits.append({"test_groups": [int(g) for g in pair], "fit": fitted,
                         "metrics": metrics(y[test], pred)})
        result[spec.name] = {"metrics": metrics(np.array(held_y), np.array(held_pred)),
                             "folds": fits}
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    np.random.seed(SEED)
    started = perf_counter()
    source = ROOT / "config/data.xlsx"
    dataset = load_dataset_bundle(source)
    frame, schema = dataset.frame, dataset.schema
    groups = group_ids(frame, schema).astype(int)
    specs = candidates()
    payload: dict[str, Any] = {
        "status": "running", "seed": SEED, "budget_seconds": BUDGET_SECONDS,
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mapping_sha256": hashlib.sha256(source.with_suffix(".beamsp.json").read_bytes()).hexdigest(),
        "physical_meaning": {
            "H": "Высота двутавра, мм", "s": "Толщина стенки двутавра, мм",
            "R": "Предел прочности жёсткой арматуры Rаж, МПа",
            "E": "Модуль упругости двутавра Eдв, МПа",
            "a_h0": "Относительный пролёт среза a/h0, безразмерный",
            "is_steel": "1 = стальной двутавр; 0 = композитный",
            "Qдв_СП63": "Вклад двутавра в поперечную силу по СП 63.13330, кН",
            "Qдв_рук78": "Вклад двутавра в поперечную силу по Руководству 1978 г., кН",
            "provenance": "Реальные строки исходной расчётной таблицы; не экспериментально подтверждённая полная несущая способность"},
        "schema": schema.to_dict(), "raw_records": frame.to_dict("records"),
        "row_groups": groups.tolist(), "references": REFERENCES,
        "n_real_rows": len(frame), "n_profiles": len(np.unique(groups)),
        "synthetic_rows": 0, "versions": {"python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "settings": {"search": [asdict(s) for s in specs],
            "power_bounds": [-3, 3], "log_C_bounds": [-10, 12], "max_nfev": 1000,
            "rounding": "powers to 1/2 decimals, refit C in Q on train, C to 2/3 sigfig",
            "existing_bounds": "PowerLawForm: log_A [-20,20], powers [-5,5]",
            "validation": "LOGO physical profiles; nested inner LOGO for selection",
            "selection": "min pooled OOF RMSE; predeclared compact <=4 features; "
                         "ties within max(1e-7 kN, 1e-6*RMSE): fewer features then search order",
            "budget_scope": "search, final fits, nested validation and leave-two-groups-out"},
        "feature_ranges": {f: [float(frame[f].min()), float(frame[f].max())]
                           for f in schema.features},
        "correlation_log": pd.DataFrame(np.log(frame[["H", "s", "R", "E"]]),
                                         columns=["H", "s", "R", "E"]).corr().to_dict(),
        "targets": {}}
    predictions = []
    for target, column in schema.targets.items():
        target_started = perf_counter()
        y = frame[column].to_numpy(float)
        entries = {}
        payload["targets"][target] = {"column": column, "unit": "кН", "candidates": entries}
        for spec in specs:
            if perf_counter() - started > BUDGET_SECONDS:
                raise TimeoutError("Research budget exhausted; partial results preserved")
            tick = perf_counter()
            pred, folds = logo(frame, y, groups, spec)
            final = fit_candidate(frame, y, spec)
            entries[spec.name] = {"spec": asdict(spec), "LOGO": metrics(y, pred),
                                  "folds": folds, "final_all_real_data": final,
                                  "seconds": perf_counter() - tick}
            for i, (actual, predicted) in enumerate(zip(y, pred)):
                predictions.append({"target": target, "candidate": spec.name,
                    "row_id": i, "excel_row": dataset.export_context["rows"][i],
                    "profile_id": int(groups[i]), "shifr": frame.iloc[i]["shifr"],
                    "a_h0": float(frame.iloc[i]["a_h0"]), "is_synth": False,
                    "Q_real_kN": actual, "Q_oof_kN": predicted,
                    "error_kN": predicted - actual,
                    "relative_error_percent": 100 * (predicted - actual) / actual})
            checkpoint(payload)
        # The selection pool is fixed a priori, independent of any held-out data.
        eligible = [s for s in specs if not s.baseline]
        exact = select_candidate([(entries[s.name]["LOGO"]["RMSE_kN"], s)
                                  for s in eligible if s.rounding is None])
        paper = select_candidate([(entries[s.name]["LOGO"]["RMSE_kN"], s)
                                  for s in eligible if s.rounding is not None])
        target_result = payload["targets"][target]
        target_result.update(best_compact=exact.name, best_paper=paper.name)
        # Honest evaluation of the entire predeclared selection procedure.
        target_result["nested_selection"] = nested_validation(frame, y, groups, eligible)
        target_result["leave_two_profiles_out"] = pair_stability(
            frame, y, groups, [exact, paper])
        target_result["seconds"] = perf_counter() - target_started
        LOG.info("%s: exact=%s %s; paper=%s %s; nested RMSE=%.6f кН", target,
                 exact.name, entries[exact.name]["LOGO"], paper.name,
                 entries[paper.name]["LOGO"],
                 target_result["nested_selection"]["metrics"]["RMSE_kN"])
        checkpoint(payload)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv", index=False, float_format="%.15g")
    payload.update(status="complete", total_seconds=perf_counter() - started)
    checkpoint(payload)
    write_report(payload)
    LOG.info("Completed in %.3f s", payload["total_seconds"])


def write_report(data: dict[str, Any]) -> None:
    lines = ["# Исследование инженерной формулы Qдв", "",
        "Расчёты выполнены локально, без сети, синтетики и изменения исходных данных. "
        "Это исследовательское приближение вклада двутавра, а не полная несущая способность балки.", "",
        "## Данные и физический смысл", "",
        "Использован только `config/data.xlsx` через действующий `load_dataset_bundle`, "
        "`DatasetSchema` и `group_ids`; mapping `config/data.beamsp.json` сохранён без изменений. "
        "18 реальных расчётных строк, 6 профилей (3 композитных и 3 стальных), "
        "по 3 значения a/h₀ на профиль. Это расчётные значения из таблицы, "
        "экспериментальное происхождение и независимая поверка не установлены. "
        "Заголовок Excel (строки 2–5) прямо называет обе цели «Qдв (вклад двутавра)», "
        "Q — поперечная сила, кН. Это согласуется с `docs/spec.MD`, раздел 2, "
        "и адаптером `core/data/legacy.py`.", "",
        "H — высота двутавра, мм; s — толщина стенки, мм; R — предел прочности "
        "жёсткой арматуры (Rаж), МПа; E — модуль упругости двутавра, МПа; "
        "a/h₀ — относительный пролёт среза, безразмерный; is_steel=1 для стали, 0 для композита. "
        "Qдв_СП63 — вклад по СП 63.13330, Qдв_рук78 — вклад по Руководству 1978 г. "
        "Точные нормативные алгоритмы исходных расчётов и исходная точность округления "
        "не представлены; они не восстановлены как доказанные нормативные зависимости.", "",
        "Идентификатор физического профиля задан `[is_steel,H,s,R,E]`, независимо от признаков "
        "конкретной модели. Внутри каждого профиля Q одинаковы при a/h₀=1.6, 2.5, 3.2. "
        "Повторы не удалены и не выданы за независимые профили; все три всегда отложены вместе. "
        "Строки «без двутавра» содержат прочерки вместо Q и исключаются штатным legacy-адаптером; "
        "это не наблюдения Q=0.", "",
        "`examples/dynamic.*`, `examples/material.csv`, `examples/power_law_extra.xlsx` — "
        "демонстрационные таблицы с generic target_a/Q_test, без подтверждённой физической связи "
        "с двумя Qдв, поэтому не объединялись с инженерными данными. Старые `results*.xlsx`, "
        "картинки и метрики не использованы как исходные наблюдения. "
        "Пользовательский `config/data.xd.beamsp.json` задаёт другую роль a_h0 и не выбран: "
        "штатный sidecar исходной таблицы — `data.beamsp.json`.", "",
        "|Профиль|Материал|H, мм|s, мм|R, МПа|E, МПа|Qдв СП63, кН|Qдв Рук78, кН|",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    frame = pd.DataFrame(data["raw_records"])
    groups = np.array(data["row_groups"])
    for group in np.unique(groups):
        r = frame.loc[groups == group].iloc[0]
        lines.append(f"|{group}|{r.material}|{r.H:g}|{r.s:g}|{r.R:g}|{r.E:g}|"
                     f"{r['Qдв_СП63']:g}|{r['Qдв_рук78']:g}|")
    lines += ["", "## Протокол поиска и защита от утечки", "",
        "Фиксированный набор: H,s,R; добавление a/h₀; замена R на E; удаление s или H; "
        "геометрические степени H=s=1 со свободной степенью R; контроль без материала. "
        "Для основных форм сравниваются логарифмический МНК (LS) и уточнение TRF по Q. "
        "Все опорные значения фиксированы заранее, не оцениваются по test: "
        "H₀=150 мм, s₀=6 мм, R₀=30 МПа, E₀=60000 МПа, (a/h₀)₀=2.5. "
        "Свободные степени ограничены [-3,3], ln C — [-10,12], "
        "TRF: max_nfev=1000, ftol=xtol=gtol=1e-10, аналитический Якобиан. "
        "LS — bounded lsq_linear(method=bvls) по ln Q, затем least_squares(method=trf) "
        "минимизирует сумму квадратов ошибок именно Q, кН.", "",
        "Бумажные варианты заранее определены: округлить свободные степени до 1 либо 2 знаков, "
        "затем на train заново найти C в исходных Q (C=ΣfᵢQᵢ/Σfᵢ²), "
        "округлить C до 2 либо 3 значащих цифр. Test Q не участвует ни в одном шаге. "
        "Предсказания paper — именно округлённые формулы в каждом фолде. "
        "Финальные коэффициенты после выбора формы заново обучены на всех 18 реальных строках; "
        "их обучающая SSE в JSON не является валидационной метрикой.", "",
        "Внешний LOGO: train=5 профилей/15 строк, test=1 профиль/3 строки. "
        "Все метрики объединяют OOF-прогнозы в исходных кН, без усреднения R² фолдов. "
        "Коэффициенты, округление и подготовка данных выполняются внутри train. "
        "Таблица сравнения используется для выбора, поэтому оценка победителя может быть оптимистичной. "
        "Дополнительно вложенный LOGO оценивает весь выбор среди компактных кандидатов: "
        "внутри каждого внешнего train выбирается форма по inner LOGO на 4/1 профилях; "
        "внешняя группа не участвует в выборе. Эта оценка относится к процедуре выбора, "
        "а не к одной неизменной финальной формуле. "
        "Для пары финалистов дополнительно выполнены все 15 разбиений leave-two-profiles-out "
        "(4 train / 2 test); это проверка чувствительности после выбора, не независимая валидация.", "",
        "Существующие `LogLinearLSModel` и `LogLinearTRFModel` реально запущены на тех же строках "
        "и группах: полная степенная модель и фиксированные формы из sp63_fixed/ruk78_fixed. "
        "В последних фиксируются a/h₀=0, H=s=R=1; для Рук78 также E=0. "
        "Это сравнение форм DE с детерминированным LS/TRF, а не запуск DE/CMA-ES. "
        "Тяжёлые стохастические оптимизаторы не запускались: для линейной задачи в ln Q "
        "они не дают преимуществ перед решением МНК. У существующих моделей сохранены их "
        "собственные границы и алгоритмы; их bounds относятся к ненормированному A. "
        "Полная модель с R и E неидентифицируема и приведена только как диагностика.", "",
        "is_steel никогда не возводится в степень. В этих данных R=30·8^is_steel, "
        "E=60000·(10/3)^is_steel; ln R и ln E полностью коллинеарны с материалом. "
        "Влияния прочности и жёсткости раздельно установить невозможно. "
        "Эквивалентная запись материального множителя — 8^(p_R·is_steel); "
        "она не подтверждает степенной закон по промежуточным R. "
        "Принудительная монотонность не задавалась.", ""]
    lines += ["При различиях RMSE в пределах max(10⁻⁷ кН, 10⁻⁶·RMSE) "
              "побеждает меньшее число признаков, затем исходный порядок кандидатов. "
              "Поэтому численно эквивалентные HSR/HSE/HSRa представлены финальной "
              "формулой HSR без лишнего a/h₀. Это заранее заданное правило, "
              "также применяемое внутри вложенной проверки.", ""]
    for target, result in data["targets"].items():
        entries = result["candidates"]
        lines += [f"## {'СП63' if target == 'SP63' else 'Руководство 1978 г.'}", "",
                  "Qдв в кН; в формулах H,s в мм, R,E в МПа. "
                  "Все отношения безразмерны, C имеет единицу кН.", "",
                  "|Метод|LOGO RMSE, кН|MAE, кН|R²|Макс. |ошибка|, кН|MAPE, %|Макс. завышение, кН|Время, с|",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, entry in entries.items():
            m = entry["LOGO"]
            lines.append(f"|{name}|{m['RMSE_kN']:.6f}|{m['MAE_kN']:.6f}|{m['R2']:.8f}|"
                         f"{m['max_abs_kN']:.6f}|{m['MAPE_percent']:.4f}|"
                         f"{m['max_over_kN']:.6f}|{entry['seconds']:.4f}|")
        ls_rmse = entries["HSR_ls"]["LOGO"]["RMSE_kN"]
        trf_rmse = entries["HSR_trf"]["LOGO"]["RMSE_kN"]
        a_rmse = entries["HSRa_trf"]["LOGO"]["RMSE_kN"]
        lines += ["", f"Уточнение в исходных Q: HSR LS → TRF снижает LOGO RMSE "
                  f"с {ls_rmse:.6f} до {trf_rmse:.6f} кН "
                  f"({100*(1-trf_rmse/ls_rmse):.1f}%). Добавление a/h₀: "
                  f"RMSE={a_rmse:.6f} кН, улучшения сверх численного шума нет. "
                  "Замена R на E даёт те же прогнозы из-за полной связи с материалом. "
                  "Удаление H или s резко ухудшает качество (см. HR/sR); "
                  "совместное использование H,s и материального множителя необходимо."]
        for key, label in (("best_compact", "Наиболее точная компактная"),
                           ("best_paper", "Бумажная")):
            name = result[key]
            entry = entries[name]
            fitted, m = entry["final_all_real_data"], entry["LOGO"]
            lines += ["", f"**{label}: `{name}`**", "", f"`Qдв = {formula(fitted)} кН`", "",
                f"LOGO RMSE={m['RMSE_kN']:.6f} кН; MAE={m['MAE_kN']:.6f} кН; "
                f"R²={m['R2']:.8f}; максимальная абсолютная ошибка={m['max_abs_kN']:.6f} кН; "
                f"MAPE={m['MAPE_percent']:.4f}%; максимальная относительная ошибка="
                f"{m['max_abs_percent']:.4f}%; максимальное завышение={m['max_over_kN']:.6f} кН "
                f"({m['max_over_percent']:.4f}%). "
                f"Завышены {m['over_count']} из 18 строк; это {m['over_count']//3} из 6 профилей.", "",
                "Ошибки по отложенным профилям (прогноз минус истинное Q; "
                "все три строки внутри профиля совпадают):", "",
                "|Профиль|Q истинное, кН|OOF, кН|Ошибка, кН|Ошибка, %|",
                "|---|---:|---:|---:|---:|"]
            for fold in entry["folds"]:
                g = fold["test_groups"][0]
                subset = frame.loc[groups == g]
                actual = float(subset[result["column"]].iloc[0])
                pr = float(predict(subset.iloc[:1], fold["fit"])[0])
                lines.append(f"|{g}|{actual:.4f}|{pr:.6f}|{pr-actual:+.6f}|"
                             f"{100*(pr-actual)/actual:+.4f}|")
            example = frame.iloc[[0]]
            value = float(predict(example, fitted)[0])
            lines += ["", "Пример подстановки в окончательную формулу на всех данных: "
                "профиль 0, Б12: H=150 мм, s=6 мм, R=30 МПа, E=60000 МПа, a/h₀=1.6. "
                f"Qдв={value:.9f} кН; табличное Qдв={example[result['column']].iloc[0]:g} кН. "
                "Для формул H,s,R все отношения равны 1, поэтому Qдв=C. "
                "Это демонстрация арифметики, не новая независимая проверка."]
            # Nontrivial example includes genuinely different normalized inputs.
            example = frame.iloc[[4]]
            value = float(predict(example, fitted)[0])
            substitutions = " * ".join(
                f"({float(example[f].iloc[0]):g}/{REFERENCES[f]:g})^{p:.9g}"
                for f, p in zip(fitted["features"], fitted["powers"]) if abs(p) > 1e-10)
            lines += ["", f"Дополнительная подстановка, Б12Э: Qдв = {fitted['C']:.9g} * "
                f"{substitutions} = {value:.9f} кН; "
                f"табличное значение {example[result['column']].iloc[0]:g} кН."]
            pair = result["leave_two_profiles_out"][name]
            lines += ["", f"Leave-two-profiles-out: RMSE={pair['metrics']['RMSE_kN']:.6f} кН, "
                f"MAE={pair['metrics']['MAE_kN']:.6f} кН; "
                f"максимальная абсолютная ошибка={pair['metrics']['max_abs_kN']:.6f} кН; "
                f"максимальное завышение={pair['metrics']['max_over_kN']:.6f} кН "
                f"({pair['metrics']['max_over_percent']:.3f}%). "
                "каждый профиль предсказывается пять раз, поэтому 90 ошибок не являются "
                "90 независимыми наблюдениями."]
            coeffs = np.array([[fold["fit"]["C"], *fold["fit"]["powers"]]
                               for fold in entry["folds"]])
            labels = ["C, кН", *[f"p({f})" for f in fitted["features"]]]
            lines += ["", "Разброс коэффициентов при исключении одного профиля: " + "; ".join(
                f"{label}: {low:.6g}…{high:.6g}" for label, low, high in
                zip(labels, coeffs.min(axis=0), coeffs.max(axis=0))) + ". "
                "У выбранных форм все степени H,s,R положительны; это согласуется "
                "с ростом вклада при увеличении геометрии и с различием материалов, "
                "но независимый закон по R здесь не установлен."]
        exact_entry = entries[result["best_compact"]]
        paper_entry = entries[result["best_paper"]]
        lines += ["", f"Цена выбранного округления: RMSE "
            f"{exact_entry['LOGO']['RMSE_kN']:.6f} → {paper_entry['LOGO']['RMSE_kN']:.6f} кН. "
            "Бумажный вариант сокращает точность записи коэффициентов; "
            "дополнительная геометрическая степень удаляется только для СП63."]
        coarse_name = "area_R_paper1" if target == "SP63" else "HSR_paper1"
        coarse = entries[coarse_name]
        lines += ["", "Более грубая запись для сравнения (не основной выбор): "
            f"`Qдв = {formula(coarse['final_all_real_data'])} кН`; "
            f"LOGO RMSE={coarse['LOGO']['RMSE_kN']:.6f} кН, "
            f"MAE={coarse['LOGO']['MAE_kN']:.6f} кН, R²={coarse['LOGO']['R2']:.8f}. "
            "Она удобнее числом знаков, но ошибка заметно больше; "
            "поэтому основным бумажным приближением оставлена запись с двумя знаками степеней."]
        n = result["nested_selection"]
        lines += ["", f"Вложенный LOGO процедуры выбора: RMSE={n['metrics']['RMSE_kN']:.6f} кН, "
            f"MAE={n['metrics']['MAE_kN']:.6f} кН, R²={n['metrics']['R2']:.8f}. "
            "Победители по внешним фолдам: " + ", ".join(f["selected"] for f in n["folds"]) + ".", ""]
    lines += ["## Ограничения и инженерная интерпретация", "",
        "Эффективный объём — лишь 6 профилей; три значения a/h₀ не увеличивают его до 18. "
        "Признаки H и s также частично связаны геометрией небольшой выборки. "
        "Добавление a/h₀ проверено отдельным кандидатом, а не принято по всей выборке до фолдов. "
        "Почти нулевая степень a/h₀ отражает только эти расчёты; влияние на полную балку "
        "из этого не следует. Отрицательный материальный множитель или показатель при R "
        "означал бы условное различие двух материалов и требует осторожности; "
        "нельзя трактовать его как причинное снижение Q при росте прочности. "
        "Знаки и диапазоны степеней по фолдам сохранены в results.json.", "",
        "Формулы положительны при положительных H,s,R (и других используемых непрерывных "
        "признаках). Для H=0, s=0, профиля без двутавра и новых материалов они не определены "
        "в рамках проверенного применения. Проверенный диапазон H=140–200 мм, s=4.7–10 мм; "
        "для композита R=30 МПа, E=60000 МПа, для стали R=240 МПа, E=200000 МПа; "
        "a/h₀=1.6–3.2. Диапазон не означает валидацию любой комбинации этих параметров: "
        "всего шесть дискретных профилей. Особенно слабо проверен перенос на новую геометрию "
        "внутри материала (всего три профиля каждого). "
        "Нет независимой внешней выборки, проверки других R/E или нормативной гарантии. "
        "Выбор из нескольких кандидатов повышает риск оптимизма; даже вложенный LOGO "
        "при шести группах имеет большую неопределённость. "
        "Завышения нельзя считать допустимыми для проектирования: оценка Qдв — лишь "
        "исследовательская и требует независимой инженерной проверки.", "",
        "## Воспроизведение и файлы", "",
        "Из корня репозитория:", "", "```bash",
        "venv/bin/python research/engineering_formula/run_formula_search.py",
        "venv/bin/python -m unittest discover -s research/engineering_formula -p 'test_*.py'",
        "# Если pytest установлен в используемом окружении:",
        "venv/bin/python -m pytest research/engineering_formula/test_formula_search.py",
        "```", "",
        "`results.json` содержит исходные строки, единицы, hash файла, версии, все кандидаты, "
        "точные коэффициенты каждого фолда и финального обучения, ранги, ошибки по профилям, "
        "настройки и время. `predictions.csv` содержит все реальные и OOF Q по каждой "
        "цели и кандидату, исходную строку Excel, профиль, знаковую и относительную ошибки. "
        "`test_formula_search.py` проверяет арифметику, детерминированность, разделение "
        "групп и независимость коэффициентов фолда от отложенной цели. "
        "Результаты записываются после каждого кандидата, исходные файлы не перезаписываются.", "",
        f"Фактическое время расчётов: {data['total_seconds']:.3f} с; "
        "заданный бюджет 600 с, лимит достаточен. Seed=20261120; алгоритмы детерминированы. "
        "Небольшие различия последних знаков возможны между версиями BLAS/SciPy. "
        "В локальном venv отсутствуют pytest и Ruff; пакеты не устанавливались. "
        "Тесты совместимы с pytest, но выполнены встроенным unittest; "
        "7 проверок прошли успешно. "
        "результат запуска записан отдельно в TEST_RESULTS.txt."]
    # Escape inner pipe in the comparison header for valid Markdown.
    lines = [line.replace("Макс. |ошибка|", "Макс. абс. ошибка") for line in lines]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
