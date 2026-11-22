"""Small research checks, runnable with pytest or standard-library unittest."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.engineering_formula.run_formula_search import (
    ROOT,
    Candidate,
    candidates,
    fit_candidate,
    logo,
    metrics,
    nested_validation,
    predict,
    select_candidate,
)
from core.data.loader import load_dataset_bundle
from core.data.schema import group_ids


class FormulaSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset = load_dataset_bundle(ROOT / "config/data.xlsx")
        cls.frame = dataset.frame
        cls.groups = group_ids(dataset.frame, dataset.schema)
        cls.y = dataset.frame["Qдв_СП63"].to_numpy(float)

    def test_formula_arithmetic_and_domain(self) -> None:
        frame = pd.DataFrame({"H": [300.0], "s": [3.0], "R": [120.0]})
        fit = {"C": 20.0, "features": ["H", "s", "R"], "powers": [1, 2, 0.5]}
        self.assertAlmostEqual(float(predict(frame, fit)[0]), 20 * 2 * 0.5**2 * 4**0.5)
        for invalid in (0, -1, np.nan):
            frame.loc[0, "H"] = invalid
            with self.assertRaises(ValueError):
                predict(frame, fit)

    def test_reproducible_and_trf_improves_original_q_sse(self) -> None:
        spec = Candidate("test", ("H", "s", "R"))
        first = fit_candidate(self.frame, self.y, spec)
        second = fit_candidate(self.frame, self.y, spec)
        np.testing.assert_array_equal(predict(self.frame, first), predict(self.frame, second))
        self.assertLessEqual(first["train_sse_kN2"], first["initial_sse_kN2"] + 1e-10)

    def test_rounding_recalibrates_only_on_train(self) -> None:
        train = self.groups != 1
        spec = Candidate("paper", ("H", "s", "R"), rounding=2)
        fit = fit_candidate(self.frame.loc[train], self.y[train], spec)
        powers = np.array(fit["powers"])
        np.testing.assert_array_equal(powers, np.round(powers, 2))
        shape_fit = {**fit, "C": 1.0}
        shape = predict(self.frame.loc[train], shape_fit)
        c = float(shape @ self.y[train] / (shape @ shape))
        self.assertEqual(fit["C"], float(f"{c:.3g}"))

    def test_logo_groups_and_held_target_independence(self) -> None:
        for spec in (Candidate("raw", ("H", "s", "R")),
                     Candidate("paper", ("H", "s", "R"), rounding=2)):
            with self.subTest(spec=spec.name):
                pred, folds = logo(self.frame, self.y, self.groups, spec)
                self.assertEqual(len(folds), 6)
                seen = []
                for fold in folds:
                    self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
                    self.assertEqual(len(fold["test_row_ids"]), 3)
                    self.assertEqual(len(fold["train_row_ids"]), 15)
                    seen.extend(fold["test_row_ids"])
                self.assertEqual(sorted(seen), list(range(18)))
                changed = self.y.copy()
                held = self.groups == 0
                changed[held] *= 100
                other_pred, other_folds = logo(self.frame, changed, self.groups, spec)
                np.testing.assert_array_equal(pred[held], other_pred[held])
                self.assertEqual(folds[0]["fit"]["C"], other_folds[0]["fit"]["C"])

    def test_numerical_ties_prefer_fewer_features(self) -> None:
        long = Candidate("long", ("H", "s", "R", "a_h0"))
        short = Candidate("short", ("H", "s", "R"))
        self.assertEqual(select_candidate([(0.1, long), (0.100000001, short)]), short)

    def test_nested_selection_does_not_see_outer_target(self) -> None:
        pool = [Candidate("HSR", ("H", "s", "R")),
                Candidate("paper", ("H", "s", "R"), rounding=2)]
        first = nested_validation(self.frame, self.y, self.groups, pool)
        changed = self.y.copy()
        held = self.groups == 0
        changed[held] *= 100
        second = nested_validation(self.frame, changed, self.groups, pool)
        self.assertEqual(first["folds"][0]["selected"], second["folds"][0]["selected"])
        self.assertEqual(first["folds"][0]["inner_rmse"], second["folds"][0]["inner_rmse"])
        np.testing.assert_array_equal(np.array(first["predictions"])[held],
                                      np.array(second["predictions"])[held])

    def test_saved_results_and_oof_metrics(self) -> None:
        folder = Path(__file__).resolve().parent
        saved = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "complete")
        predictions = pd.read_csv(folder / "predictions.csv")
        self.assertEqual(len(predictions), 18 * 2 * len(candidates()))
        self.assertFalse(predictions["is_synth"].any())
        for target, result in saved["targets"].items():
            for spec in candidates():
                with self.subTest(target=target, candidate=spec.name):
                    rows = predictions.loc[(predictions.target == target)
                                           & (predictions.candidate == spec.name)]
                    self.assertEqual(len(rows), 18)
                    self.assertEqual(rows.profile_id.nunique(), 6)
                    actual = metrics(rows.Q_real_kN.to_numpy(), rows.Q_oof_kN.to_numpy())
                    for key in ("RMSE_kN", "MAE_kN", "R2", "max_abs_kN"):
                        self.assertAlmostEqual(actual[key], result["candidates"][spec.name]["LOGO"][key],
                                               places=10)
                    entry = result["candidates"][spec.name]
                    self.assertTrue(entry["final_all_real_data"]["success"])
                    for fold in entry["folds"]:
                        self.assertTrue(fold["fit"]["success"])
                    fitted = entry["final_all_real_data"]
                    manual = np.full(len(self.frame), fitted["C"])
                    for feature, power in zip(fitted["features"], fitted["powers"]):
                        manual *= (self.frame[feature].to_numpy()
                                   / fitted["references"][feature]) ** power
                    np.testing.assert_allclose(predict(self.frame, fitted), manual,
                                               rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
