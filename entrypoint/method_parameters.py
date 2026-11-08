"""Qt editor for the model settings defined in core.models.parameters."""

from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.labels import method_label
from core.models.parameters import (
    CHOICES,
    LABELS,
    parameter_defaults,
    validate_parameters,
)


class MethodParameters(QWidget):
    def __init__(self, names: list[str], parent=None):
        super().__init__(parent)
        self._values: dict[str, dict[str, Any]] = {}
        self._editors: dict[str, QWidget] = {}
        self._name = None
        layout = QVBoxLayout(self)
        self.methods = QComboBox()
        for name in names:
            self.methods.addItem(method_label(name), name)
        layout.addWidget(self.methods)
        self.form = QFormLayout()
        layout.addLayout(self.form)
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
