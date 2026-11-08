"""Export LOGO predictions alongside the original workbook cells."""

from copy import copy
from io import BytesIO
from pathlib import Path

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils.dataframe import dataframe_to_rows

from core.evaluation.run_result import RunResult
from core.labels import method_label


def export_workbook(result: RunResult, path: Path) -> None:
    context = result.export_context
    if context is None:
        raise ValueError("Результат не содержит исходных данных для Excel")
    if "workbook" in context:
        workbook = load_workbook(BytesIO(context["workbook"]))
        sheet = workbook[context["sheet"]]
    else:
        workbook = Workbook()
        sheet = workbook.active
        for row in dataframe_to_rows(context["frame"], index=False, header=True):
            sheet.append([None if _missing(value) else value for value in row])
    try:
        rows = context["rows"]
        header = context["header_row"]
        column = sheet.max_column + 1
        for target, methods in result.predictions.items():
            info = result.metadata.get("per_target", {}).get(target, {})
            target_name = result.metadata.get("targets", {}).get(target, target)
            unit = info.get("unit", "")
            for name, values in methods.items():
                predictions = values["y_pred"]
                if len(predictions) != len(rows):
                    raise ValueError(
                        "Число предсказаний не совпадает с исходными строками"
                    )
                title = f"{target_name} | {method_label(name)} | LOGO"
                if unit:
                    title += f" ({unit})"
                cell = sheet.cell(header, column, title)
                cell.font = Font(bold=True)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                for row, value in zip(rows, predictions):
                    if not np.isfinite(value):
                        raise ValueError(
                            f"{name}: нечисловое предсказание в строке {row}"
                        )
                    cell = sheet.cell(row, column, float(value))
                    cell._style = copy(sheet.cell(row, column - 1)._style)
                    cell.number_format = "0.000000"
                column += 1
        # Exclusive creation prevents accidental replacement, including source files.
        with path.open("xb") as output:
            workbook.save(output)
    finally:
        workbook.close()


def _missing(value) -> bool:
    import pandas as pd

    return bool(pd.isna(value))
