"""Regression checks for the repeated grouped formula comparison."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.data.loader import load_dataset_bundle
from core.data.schema import group_ids
from research.engineering_formula.run_formula_recheck import (
    ROOT,
    evaluate,
    fit,
    formulas,
    nested,
    predict,
    rounding_grid,
    web_height,
)
from research.engineering_formula.run_formula_search import metrics


class FormulaRecheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset = load_dataset_bundle(ROOT / "config/data.xlsx")
        cls.frame = dataset.frame
        cls.groups = group_ids(cls.frame, dataset.schema)
        cls.targets = {key: cls.frame[column].to_numpy(float)
                       for key, column in dataset.schema.targets.items()}
        cls.spec = {spec.name: spec for spec in formulas()}

    def test_catalog_uses_material_and_nominal_dimensions(self) -> None:
        sample = self.frame.iloc[[0, 3, 4, 5]]
        np.testing.assert_allclose(web_height(sample), [138.0, 126.2, 184.0, 145.2])
        sample = sample.copy()
        sample.loc[sample["is_steel"] == 1, "s"] = 9
        with self.assertRaisesRegex(ValueError, "No catalog section"):
            web_height(sample)

    def test_nominal_web_formula_arithmetic_and_quantization(self) -> None:
        y = self.targets["RUK78"]
        nominal = self.spec["web_nominal"]
        fit_nominal = fit(self.frame, y, nominal)
        np.testing.assert_allclose(predict(self.frame.iloc[:6], nominal, fit_nominal),
                                   [19.872, 43.2, 18.0, 113.88288, 194.304, 139.392])
        self.assertEqual(rounding_grid(y), 1.96)
        self.assertEqual(rounding_grid(np.array([19.6, 43.12])), 3.92)
        for name in ("web_with_rounding", "web_proxy_with_rounding"):
            result = evaluate(self.frame, y, self.groups, self.spec[name], held_count=2)
            self.assertEqual(len(result["folds"]), 15)
            self.assertLess(result["metrics"]["RMSE_kN"], 1e-12)
            for fold in result["folds"]:
                self.assertEqual(fold["fit"]["rounding_grid_kN"], 1.96)

    def test_rounding_grid_uses_training_targets_only(self) -> None:
        y = self.targets["RUK78"].copy()
        held = self.groups == 0
        y[held] = 19.61
        self.assertEqual(rounding_grid(y), 0.01)
        result = evaluate(self.frame, y, self.groups, self.spec["web_proxy_with_rounding"])
        self.assertEqual(result["folds"][0]["fit"]["rounding_grid_kN"], 1.96)
        np.testing.assert_allclose(result["folds"][0]["predict_kN"], 19.6)

    def test_pair_ranks_and_geometry_shrinkage(self) -> None:
        y = self.targets["SP63"]
        regularized = evaluate(self.frame, y, self.groups, self.spec["steel_depth_ridge"], 2)
        unconstrained = evaluate(self.frame, y, self.groups, self.spec["steel_depth"], 2)
        self.assertEqual(regularized["rank_supported_folds"], 15)
        self.assertEqual(unconstrained["rank_supported_folds"], 12)
        self.assertLess(regularized["metrics"]["RMSE_kN"], 0.25)

    def test_held_out_target_does_not_change_nested_selection_or_prediction(self) -> None:
        y = self.targets["RUK78"]
        pool = [self.spec[name] for name in ("web_calibrated", "HSR_trf", "web_proxy_nominal")]
        first = nested(self.frame, y, self.groups, pool)
        held = self.groups == 0
        changed = y * np.where(held, 1.25, 1)
        second = nested(self.frame, changed, self.groups, pool)
        self.assertEqual(first["folds"][0], second["folds"][0])
        np.testing.assert_array_equal(np.asarray(first["predictions_kN"])[held],
                                      np.asarray(second["predictions_kN"])[held])

    def test_export_reproduces_archived_metrics_and_contains_every_real_row(self) -> None:
        folder = Path(__file__).resolve().parent
        payload = json.loads((folder / "recheck_results.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(folder / "recheck_predictions.csv")
        self.assertEqual(payload["synthetic_rows"], 0)
        for target, data in payload["targets"].items():
            for name, entry in data["candidates"].items():
                subset = predictions.loc[(predictions.target == target) & (predictions.candidate == name)]
                self.assertEqual(subset.excel_row.nunique(), 18)
                self.assertEqual(subset.profile.nunique(), 6)
                computed = metrics(subset.Q_table_kN.to_numpy(), subset.Q_oof_kN.to_numpy())
                self.assertAlmostEqual(computed["RMSE_kN"], entry["LOGO"]["metrics"]["RMSE_kN"], places=10)
                if "archive_RMSE_difference_kN" in entry:
                    self.assertAlmostEqual(entry["archive_RMSE_difference_kN"], 0, places=9)


if __name__ == "__main__":
    unittest.main()
