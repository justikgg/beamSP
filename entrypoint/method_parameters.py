"""Qt editor for the model settings defined in core.models.parameters."""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from core.models.registry import available_models

from core.labels import method_label
from core.models.parameters import (
    CHOICES,
    LABELS,
    parameter_defaults,
    validate_parameters,
)

# LS/TRF accept fixed exponents only; these are model constraints, not tuning
# settings. All other methods expose the real constructor/optimizer parameters.
TUNABLE_METHODS = {
    name for name in available_models()
    if name not in {"log_linear_ls", "log_linear_trf"} and parameter_defaults(name)
}


class MethodParameters(QWidget):
    def __init__(self, names: list[str], parent=None):
        super().__init__(parent)
        self._values: dict[str, dict[str, Any]] = {}
        self._editors: dict[str, QWidget] = {}
        self._tune_editors: dict[str, tuple] = {}
        self._name = None
        layout = QVBoxLayout(self)
        self.methods = QComboBox()
        for name in names:
            self.methods.addItem(method_label(name), name)
        layout.addWidget(self.methods)
        self.form = QFormLayout()
        layout.addLayout(self.form)
        self.tune_mode = QCheckBox("Подбор параметров по диапазонам")
        self.tune_mode.toggled.connect(self._render_tuning)
        layout.addWidget(self.tune_mode)
        self.tune_form = QFormLayout()
        layout.addLayout(self.tune_form)
        self.tune_limit = QSpinBox()
        self.tune_limit.setRange(1, 500)
        self.tune_limit.setValue(100)
        self.tune_form.addRow("Максимум комбинаций", self.tune_limit)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        reset = QPushButton("Сбросить параметры метода")
        reset.clicked.connect(self.reset)
        layout.addWidget(reset)
        self.methods.currentIndexChanged.connect(self._switch)
        self._switch()

    def _read(self) -> dict[str, Any]:
        result = {}
        for key, editor in self._editors.items():
            if isinstance(editor, QCheckBox):
                value = editor.isChecked()
            elif isinstance(editor, QComboBox):
                value = editor.currentData()
            elif isinstance(editor, QLineEdit):
                value = editor.text()
            else:
                value = editor.value()
            result[key] = value
        return result

    def _switch(self, *_args) -> None:
        if self._name is not None:
            self._values[self._name] = self._read()
        self._name = self.methods.currentData()
        self._render()

    def _render(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self._editors = {}
        defaults = parameter_defaults(self._name)
        values = self._values.get(self._name, defaults)
        for key, default in defaults.items():
            value = values[key]
            if key in CHOICES:
                editor = QComboBox()
                for option in CHOICES[key]:
                    editor.addItem(option, option)
                editor.setCurrentIndex(editor.findData(value))
            elif isinstance(default, bool):
                editor = QCheckBox()
                editor.setChecked(value)
            elif isinstance(default, tuple):
                editor = QLineEdit(
                    value
                    if isinstance(value, str)
                    else ", ".join(str(v) for v in value)
                )
                editor.setToolTip(
                    "Числа через запятую; размеры слоёв — целые положительные"
                )
            elif isinstance(default, int):
                editor = QSpinBox()
                editor.setRange(0, 10000000)
                editor.setValue(value)
            else:
                editor = QDoubleSpinBox()
                editor.setDecimals(10)
                editor.setRange(0, 1e12)
                editor.setValue(value)
            editor.setToolTip(f"{key}; по умолчанию: {default}")
            self._editors[key] = editor
            self.form.addRow(LABELS.get(key, key), editor)
        if not defaults:
            self.form.addRow(QLabel("У метода нет настраиваемых параметров"))
        self.error.clear()
        self._render_tuning()

    def _render_tuning(self, *_args) -> None:
        while self.tune_form.rowCount() > 1:
            self.tune_form.removeRow(1)
        self._tune_editors = {}
        if not self.tune_mode.isChecked():
            return
        if self._name not in TUNABLE_METHODS:
            reason = ("LS/TRF настраиваются фиксацией показателей степени; доступно обычное обучение."
                      if self._name in {"log_linear_ls", "log_linear_trf"} else
                      "У метода нет настраиваемых гиперпараметров; доступно обычное обучение.")
            self.tune_form.addRow(QLabel(reason))
            return
        for key, default in parameter_defaults(self._name).items():
            if key in CHOICES or isinstance(default, bool):
                box = QListWidget()
                options = CHOICES[key] if key in CHOICES else (False, True)
                for option in options:
                    item = QListWidgetItem(str(option))
                    item.setData(Qt.ItemDataRole.UserRole, option)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked if option == default or isinstance(default, bool) else Qt.CheckState.Unchecked)
                    box.addItem(item)
                box.setMaximumHeight(64)
                self._tune_editors[key] = ("choices", box)
                self.tune_form.addRow(LABELS.get(key, key), box)
            elif isinstance(default, tuple):
                edit = QLineEdit(", ".join(str(v) for v in default))
                edit.setPlaceholderText("готовые наборы через ; например 32,16;64,32")
                self._tune_editors[key] = ("tuple", edit)
                self.tune_form.addRow(LABELS.get(key, key), edit)
            else:
                row = QWidget(); line = QHBoxLayout(row); line.setContentsMargins(0, 0, 0, 0)
                lo, hi, step = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
                for spin in (lo, hi):
                    spin.setDecimals(8); spin.setRange(0, 1e12); spin.setValue(float(default))
                step.setDecimals(8); step.setRange(1e-8, 1e12); step.setValue(max(float(default) / 2, 0.1))
                log = QCheckBox("log")
                for widget in (QLabel("от"), lo, QLabel("до"), hi, QLabel("шаг/множ."), step, log): line.addWidget(widget)
                self._tune_editors[key] = ("number", lo, hi, step, log, default)
                self.tune_form.addRow(LABELS.get(key, key), row)

    def reset(self) -> None:
        self._values.pop(self._name, None)
        self._render()

    def snapshot(self, names: list[str]) -> dict[str, dict[str, Any]]:
        self._values[self._name] = self._read()
        result = {}
        for name in names:
            values = dict(self._values.get(name, parameter_defaults(name)))
            try:
                for key, default in parameter_defaults(name).items():
                    if isinstance(default, tuple) and isinstance(values[key], str):
                        cast = int if key == "hidden" else float
                        values[key] = tuple(
                            cast(v.strip()) for v in values[key].split(",")
                        )
                result[name] = validate_parameters(name, values)
            except ValueError as exc:
                self.methods.setCurrentIndex(self.methods.findData(name))
                self.error.setText(f"Ошибка параметров: {exc}")
                raise ValueError(f"{method_label(name)}: {exc}") from exc
        return result

    def tuning_snapshot(self, names: list[str]) -> dict[str, dict[str, Any]]:
        self._values[self._name] = self._read()
        if not self.tune_mode.isChecked():
            return {}
        if self._name not in names or self._name not in TUNABLE_METHODS:
            raise ValueError("Выберите поддерживаемый метод для подбора")
        values = {}
        for key, editor in self._tune_editors.items():
            if editor[0] == "choices":
                box = editor[1]
                selected = [box.item(i).data(Qt.ItemDataRole.UserRole) for i in range(box.count()) if box.item(i).checkState() == Qt.CheckState.Checked]
                if not selected:
                    raise ValueError(f"Не выбраны варианты для «{key}»")
                values[key] = selected
            elif editor[0] == "tuple":
                cast = int if key == "hidden" else float
                values[key] = [tuple(cast(v.strip()) for v in raw.split(",") if v.strip()) for raw in editor[1].text().split(";")]
            else:
                _, lo, hi, step, log, _default = editor
                from core.evaluation.tuning import numeric_range
                values[key] = numeric_range(lo.value(), hi.value(), step.value(), logarithmic=log.isChecked())
        return {self._name: values}
