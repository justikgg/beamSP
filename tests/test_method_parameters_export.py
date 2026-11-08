"""Settings propagation and source-preserving Excel export."""

from copy import copy
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from test_pipeline import dynamic

from core.data.loader import load_dataset_bundle
from core.evaluation.runner import run
from core.excel_export import export_workbook
from core.models.parameters import (
    configured_model,
    parameter_defaults,
    validate_parameters,
)
from core.models.registry import available_models


def test_defaults_for_every_method():
    for name in available_models():
        defaults = parameter_defaults(name)
        assert validate_parameters(name, defaults) == defaults
        assert configured_model(name, defaults).name == name


@pytest.mark.parametrize(
    "name,values",
    [
        ("ridge", {"alpha": -1}),
        ("svr", {"C": float("nan")}),
        ("mlp", {"hidden": (0, 3)}),
        ("bio_de", {"mutation": (1.5, 0.5)}),
        ("bio_pso", {"w_lo": 1, "w_hi": 0.2}),
        ("bio_ga", {"elite_frac": 1}),
        ("knn", {"weights": "bad"}),
        ("ridge", {"unknown": 1}),
    ],
)
def test_invalid_settings(name, values):
    with pytest.raises(ValueError):
        validate_parameters(name, values)


def test_settings_reach_each_fold_and_metadata():
    from core.evaluation import runner

    original = runner.configured_model
    calls = []

    def build(name, values, fixed):
        calls.append(dict(values))
        return original(name, values, fixed)

    with patch.object(runner, "configured_model", side_effect=build):
        result = run(
            ["ridge"],
            dataset=dynamic(),
            targets={"a": "target_a"},
            model_parameters={"ridge": {"alpha": 7}},
        )
    assert len(calls) == 5  # four held-out groups and final fit
    assert all(values["alpha"] == 7 for values in calls)
    assert result.metadata["model_config"]["ridge"]["parameters"]["alpha"] == 7


def test_bio_fixed_parameters():
    ds = dynamic()
    result = run(
        ["bio_pso", "bio_pso_fixed"],
        dataset=ds,
        targets={"a": "target_a"},
        fixed_exponents={"x1": 1},
        model_parameters={"bio_pso": {"n_particles": 3, "iters": 2}},
    )
    assert result.metadata["model_config"]["bio_pso_fixed"]["parameters"]["iters"] == 2
    assert result.metadata["model_config"]["bio_pso_fixed"]["fixed_exponents"] == {
        "x1": 1
    }


def test_workbook_preserves_source_and_snapshot(tmp_path):
    ds = dynamic()
    path = tmp_path / "source.xlsx"
    ds.frame.to_excel(path, index=False)
    book = load_workbook(path)
    book.active["A2"].font = Font(bold=True, color="FF0000")
    other = book.create_sheet("Notes")
    other.merge_cells("A1:C1")
    other["A1"] = "=1+2"
    book.save(path)
    before = path.read_bytes()
    loaded = load_dataset_bundle(path, schema=ds.schema)
    result = run(["linear", "ridge"], dataset=loaded)
    path.write_bytes(b"changed after calculation")
    output = tmp_path / "result.xlsx"
    export_workbook(result, output)
    assert path.read_bytes() == b"changed after calculation"
    exported = load_workbook(output)
    original = load_workbook(BytesIO(before))
    assert exported.sheetnames == original.sheetnames
    assert exported["Notes"]["A1"].value == "=1+2"
    assert str(exported["Notes"].merged_cells) == "A1:C1"
    assert copy(exported.active["A2"].font) == copy(original.active["A2"].font)
    column = original.active.max_column + 1
    for methods in result.predictions.values():
        for values in methods.values():
            np.testing.assert_allclose(
                [exported.active.cell(r, column).value for r in range(2, 34)],
                values["y_pred"],
            )
            column += 1
    with pytest.raises(FileExistsError):
        export_workbook(result, output)


def test_legacy_row_mapping_and_headers(tmp_path):
    book = Workbook()
    sheet = book.active
    sheet["B1"] = "Исходные данные"
    sheet.merge_cells("B1:K1")
    sheet["B4"] = "a/h0"
    for index, row in enumerate((5, 7, 8, 10)):
        values = [
            1,
            "same-id",
            "стальной",
            100 + index * 10,
            3,
            200,
            200000,
            "=1+2",
            40 + index,
            50 + index,
        ]
        for col, value in enumerate(values, 2):
            sheet.cell(row, col, value)
    source = tmp_path / "legacy.xlsx"
    book.save(source)
    before = source.read_bytes()
    dataset = load_dataset_bundle(source)
    result = run(
        ["linear"],
        dataset=dataset,
        synth_cfg={"enabled": True, "samples_per_profile": 1},
    )
    output = tmp_path / "export.xlsx"
    export_workbook(result, output)
    exported = load_workbook(output).active
    assert source.read_bytes() == before
    assert exported["L4"].value.endswith("LOGO (кН)")
    assert exported["L6"].value is None
    np.testing.assert_allclose(
        [exported.cell(row, 12).value for row in (5, 7, 8, 10)],
        result.predictions["SP63"]["linear"]["y_pred"],
    )
    assert exported["I5"].value == "=1+2"


def test_csv_and_gui_editor(tmp_path):
    from PyQt6.QtWidgets import QApplication

    from entrypoint.method_parameters import MethodParameters

    app = QApplication.instance() or QApplication([])
    panel = MethodParameters(["ridge", "mlp"])
    panel._editors["alpha"].setValue(9)
    panel.methods.setCurrentIndex(1)
    panel._editors["hidden"].setText("8, 4")
    assert panel.snapshot(["ridge", "mlp"])["mlp"]["hidden"] == (8, 4)
    panel.methods.setCurrentIndex(0)
    assert panel._editors["alpha"].value() == 9
    panel.reset()
    assert panel._editors["alpha"].value() == 100
    panel.methods.setCurrentIndex(1)
    panel._editors["hidden"].setText("0, bad")
    with pytest.raises(ValueError):
        panel.snapshot(["mlp"])
    assert panel.error.text()
    panel.close()
    app.processEvents()
    ds = dynamic()
    source = tmp_path / "input.csv"
    ds.frame.to_csv(source, index=False)
    result = run(["linear"], dataset=load_dataset_bundle(source, schema=ds.schema))
    output = tmp_path / "output.xlsx"
    export_workbook(result, output)
    sheet = load_workbook(output).active
    assert [
        sheet.cell(1, c).value for c in range(1, len(ds.frame.columns) + 1)
    ] == list(ds.frame.columns)


def test_excel_only_creates_no_other_files_or_figures(tmp_path):
    from core.reporting import export_results

    result = run(["linear"], dataset=dynamic())
    with patch("core.reporting._pyplot") as pyplot:
        saved = export_results(result, tmp_path, stamp="test", excel_only=True)
        pyplot.assert_not_called()
    assert saved == ["results_test.xlsx"]
    assert sorted(path.name for path in tmp_path.iterdir()) == saved
    sheet = load_workbook(tmp_path / saved[0]).active
    assert sheet.cell(1, len(dynamic().frame.columns) + 1).value.endswith("LOGO")


def test_excel_only_without_source_fails_without_files(tmp_path):
    from core.evaluation.run_result import RunResult
    from core.reporting import export_results

    with pytest.raises(ValueError, match="исходных данных"):
        export_results(RunResult(), tmp_path, excel_only=True)
    assert not list(tmp_path.iterdir())
