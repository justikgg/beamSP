"""Automatic material features across loading, mappings, LOGO and synthesis."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from core.data.loader import load_dataset_bundle
from core.data.schema import ColumnRole, DatasetError, DatasetSchema, save_schema
from core.evaluation.runner import prepare, run


def material_frame(column="material"):
    rng = np.random.default_rng(41)
    steel = np.repeat([0, 1, 0, 1], 4)
    x = rng.uniform(1, 5, 16)
    return pd.DataFrame({
        "profile_id": np.repeat(["A", "B", "C", "D"], 4),
        column: np.where(steel, "стальной", "композитный"),
        "x1": x, "target_a": 10 + 100 * steel + 2 * x,
    })


class MaterialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "material.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_csv_and_xlsx_create_a_binary_feature_used_in_logo(self):
        frame = material_frame()
        for ext in ("csv", "xlsx"):
            with self.subTest(format=ext):
                path = self.path.with_suffix("." + ext)
                if ext == "csv":
                    frame.to_csv(path, index=False)
                else:
                    frame.to_excel(path, index=False)
                dataset = load_dataset_bundle(path)
                np.testing.assert_array_equal(dataset.frame.is_steel, np.repeat([0, 1, 0, 1], 4))
                self.assertIn("is_steel", dataset.schema.features)
                spec = dataset.schema.spec("is_steel")
                self.assertEqual(spec.dtype, "binary")
                self.assertFalse(spec.formula_allowed)
                self.assertIsNone(spec.synthetic_noise)
                self.assertEqual(spec.derived_from, "material")
                self.assertEqual(dataset.schema.spec("material").role, ColumnRole.METADATA)
                self.assertNotIn("material", dataset.schema.features)
                result = run(["linear", "ridge", "gbr"], dataset=dataset)
                self.assertGreater(result.metrics["target_a"].loc["linear", "R2"], 0.999999)
                self.assertIn("is_steel", result.formulas["target_a"]["linear"])
                actual = DatasetSchema.from_dict(result.metadata["schema"])
                self.assertEqual(actual.features, dataset.schema.features)
                self.assertEqual(actual.spec("is_steel").dtype, "binary")
                self.assertEqual(actual.spec("is_steel").derived_from, "material")

    def test_column_aliases_casing_whitespace_and_values(self):
        labels = [" СтАлЬ ", " КОМПОЗИТНЫЙ ", "steel", "COMPOSITE"] * 4
        for column in ("Материал", "Тип материала двутавра", "material_type", "Material", "тип\nматериала"):
            with self.subTest(column=column):
                frame = material_frame(column)
                frame[column] = labels
                frame.to_csv(self.path, index=False)
                dataset = load_dataset_bundle(self.path)
                np.testing.assert_array_equal(dataset.frame.is_steel, [1, 0, 1, 0] * 4)
                self.assertEqual(dataset.schema.spec("is_steel").derived_from, column)

    def test_existing_mapping_missing_flag_gets_new_material_feature(self):
        frame = material_frame()
        frame.to_csv(self.path, index=False)
        old = DatasetSchema.from_mapping(frame.columns, ["x1"], ["target_a"], "profile_id")
        save_schema(old, self.path)
        dataset = load_dataset_bundle(self.path)
        self.assertEqual(dataset.schema.features, ["x1", "is_steel"])
        save_schema(dataset.schema, self.path)
        again = load_dataset_bundle(self.path)
        self.assertEqual(again.schema.to_dict(), dataset.schema.to_dict())
        np.testing.assert_array_equal(again.frame.is_steel, dataset.frame.is_steel)

    def test_saved_disable_or_ignore_is_respected(self):
        material_frame().to_csv(self.path, index=False)
        dataset = load_dataset_bundle(self.path)
        for change in ("disable", "ignore"):
            with self.subTest(setting=change):
                schema = DatasetSchema.from_dict(dataset.schema.to_dict())
                spec = schema.spec("is_steel")
                if change == "disable":
                    spec.enabled = False
                else:
                    spec.role = ColumnRole.IGNORE
                save_schema(schema, self.path)
                loaded = load_dataset_bundle(self.path)
                self.assertIn("is_steel", loaded.frame)
                self.assertNotIn("is_steel", loaded.schema.features)

    def test_existing_numeric_flag_is_checked_not_overwritten(self):
        frame = material_frame()
        expected = np.repeat([0, 1, 0, 1], 4)
        frame["is_steel"] = expected
        frame.to_csv(self.path, index=False)
        dataset = load_dataset_bundle(self.path)
        np.testing.assert_array_equal(dataset.frame.is_steel, expected)
        frame.loc[0, "is_steel"] = 1
        frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(DatasetError, "противоречат.*1 строках"):
            load_dataset_bundle(self.path)

    def test_existing_flag_without_material_gets_binary_defaults(self):
        frame = material_frame().drop(columns="material")
        frame["is_steel"] = np.repeat([0, 1, 0, 1], 4)
        frame.to_csv(self.path, index=False)
        dataset = load_dataset_bundle(self.path)
        self.assertEqual(dataset.schema.spec("is_steel").dtype, "binary")
        self.assertFalse(dataset.schema.spec("is_steel").formula_allowed)
        self.assertIn("is_steel", dataset.schema.features)
        frame.loc[0, "is_steel"] = 2
        frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(DatasetError, "is_steel.*отличных от 0 и 1"):
            load_dataset_bundle(self.path)

    def test_unknown_and_missing_material_are_not_composite(self):
        for label, message in (("алюминий", "material.*неизвестных.*алюминий"),
                               (None, "material.*пропущенных"), (" ", "неизвестных")):
            with self.subTest(material=label):
                frame = material_frame()
                frame.loc[0, "material"] = label
                frame.to_csv(self.path, index=False)
                with self.assertRaisesRegex(DatasetError, message):
                    load_dataset_bundle(self.path, validate=False)

    def test_two_material_sources_require_disambiguation(self):
        frame = material_frame()
        frame["Материал"] = frame.material
        frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(DatasetError, "Неоднозначный источник"):
            load_dataset_bundle(self.path)

    def test_material_indicator_is_immutable_in_synthesis(self):
        material_frame().to_csv(self.path, index=False)
        dataset = load_dataset_bundle(self.path)
        df, groups = prepare(dataset.frame, schema=dataset.schema,
                             synth_cfg={"enabled": True, "samples_per_profile": 1, "noise": {"x1": 0.1}})
        np.testing.assert_array_equal(df.is_steel.iloc[:16], df.is_steel.iloc[16:])
        np.testing.assert_array_equal(df.material.iloc[:16], df.material.iloc[16:])
        np.testing.assert_array_equal(groups[:16], groups[16:])
        with self.assertRaisesRegex(DatasetError, "категориальному/бинарному"):
            prepare(dataset.frame, schema=dataset.schema,
                    synth_cfg={"enabled": True, "samples_per_profile": 1, "noise": {"is_steel": 0.1}})

    def test_material_target_or_group_cannot_generate_an_input(self):
        frame = material_frame()
        frame["material"] = np.repeat([0, 1, 0, 1], 4)
        frame.to_csv(self.path, index=False)
        for role in (ColumnRole.TARGET, ColumnRole.GROUP):
            with self.subTest(role=role):
                schema = DatasetSchema.from_mapping(frame.columns, ["x1"], ["target_a"], "profile_id")
                schema.spec("material").role = role
                if role == ColumnRole.GROUP:
                    schema.spec("profile_id").role = ColumnRole.METADATA
                loaded = load_dataset_bundle(self.path, schema=schema)
                self.assertNotIn("is_steel", loaded.frame)
                self.assertNotIn("is_steel", loaded.schema.features)

    def test_gui_role_changes_cannot_leak_a_derived_source(self):
        material_frame().to_csv(self.path, index=False)
        dataset = load_dataset_bundle(self.path)
        dataset.schema.spec("material").role = ColumnRole.TARGET
        with self.assertRaisesRegex(DatasetError, "is_steel.*material.*target"):
            run(["linear"], dataset=dataset, targets={"target_a": "target_a"})

    def test_power_models_exclude_only_the_binary_formula_input(self):
        from functools import partial
        from core.models.bioinspired.formula_search import FormulaSearchModel
        from core.models.bioinspired.optimizers import de
        material_frame().to_csv(self.path, index=False)
        dataset = load_dataset_bundle(self.path)
        result = run(["quick_power"], dataset=dataset, model_factories={
            "quick_power": lambda: FormulaSearchModel(partial(de, maxiter=2, popsize=3, polish=False))})
        self.assertIn("is_steel", result.metadata["feature_columns"])
        self.assertNotIn("is_steel^", result.formulas["target_a"]["quick_power"])
        self.assertIn("x1^", result.formulas["target_a"]["quick_power"])


if __name__ == "__main__":
    unittest.main()
