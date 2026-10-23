"""Dataset mapping editor; loading and validation are delegated to core.data."""
from copy import deepcopy
from dataclasses import replace

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QTableWidget, QTableWidgetItem, QComboBox, QCheckBox,
                             QDoubleSpinBox, QFileDialog, QMessageBox)

from core.data.loader import load_dataset_bundle
from core.data.schema import ColumnRole, Dataset, DatasetError, DatasetSchema, group_ids, save_schema


class DatasetPanel(QGroupBox):
    schema_edited = pyqtSignal(object)
    dataset_changed = pyqtSignal(object)
    file_loaded = pyqtSignal(bool)
    message = pyqtSignal(str)

    def __init__(self, dataset, parent=None):
        super().__init__("Датасет", parent)
        self.dataset = dataset
        dl = QVBoxLayout(self)
        load_btn = QPushButton("Загрузить датасет")
        load_btn.clicked.connect(self.load_dataset)
        dl.addWidget(load_btn)
        self._path_label = QLabel()
        self._path_label.setWordWrap(True)
        dl.addWidget(self._path_label)
        self._dataset_info = QLabel()
        self._dataset_info.setWordWrap(True)
        dl.addWidget(self._dataset_info)
        self._columns_table = QTableWidget()
        self._columns_table.setColumnCount(9)
        self._columns_table.setHorizontalHeaderLabels([
            "Column", "Role", "Use", "Unit", "Noise ±", "Type", "Formula", "Profile key", "Monotonic"])
        self._columns_table.setMinimumHeight(200)
        self._columns_table.setMaximumHeight(270)
        dl.addWidget(self._columns_table)
        buttons = QHBoxLayout()
        apply_btn = QPushButton("Применить роли")
        apply_btn.clicked.connect(self.apply_roles)
        save_btn = QPushButton("Сохранить mapping")
        save_btn.clicked.connect(self.save_mapping)
        buttons.addWidget(apply_btn)
        buttons.addWidget(save_btn)
        dl.addLayout(buttons)
        note = QLabel("Без Group отметьте Profile key. Noise задаётся только для числовых Feature. "
                      "Formula разрешает признак в степенных моделях (bio, LS, TRF); "
                      "Monotonic задаёт ограничение PINN.")
        note.setWordWrap(True)
        dl.addWidget(note)
        self._populate()

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить датасет", "", "Датасеты (*.xlsx *.csv)")
        if not path:
            return
        try:
            dataset = load_dataset_bundle(path, validate=False)
        except (DatasetError, OSError) as exc:
            QMessageBox.warning(self, "Ошибка датасета", str(exc))
            return
        self.dataset = dataset
        self.dataset_changed.emit(dataset)
        self._populate()
        self.file_loaded.emit(dataset.schema.legacy)
        # Incomplete mappings remain editable; the validation error is visible.
        self.apply_roles()

    def _populate(self):
        self._populating = True
        schema = self.dataset.schema
        self._path_label.setText(str(self.dataset.source))
        for row in range(self._columns_table.rowCount()):
            for column in range(self._columns_table.columnCount()):
                old = self._columns_table.cellWidget(row, column)
                if old is not None:
                    old.hide()
        self._columns_table.clearContents()
        self._columns_table.setRowCount(len(schema.columns))
        self._column_specs = deepcopy(schema.columns)
        for i, spec in enumerate(self._column_specs):
            item = QTableWidgetItem(spec.name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._columns_table.setItem(i, 0, item)
            role = QComboBox()
            for value in ColumnRole:
                role.addItem(value.value.title(), value.value)
            role.setCurrentIndex(role.findData(spec.role.value))
            self._columns_table.setCellWidget(i, 1, role)
            for j, checked in ((2, spec.enabled), (6, spec.formula_allowed),
                               (7, spec.name in schema.grouping_columns)):
                cb = QCheckBox()
                cb.setChecked(checked)
                self._columns_table.setCellWidget(i, j, cb)
            self._columns_table.setItem(i, 3, QTableWidgetItem(spec.unit))
            noise = QDoubleSpinBox()
            noise.setDecimals(6)
            noise.setRange(0, 1e12)
            noise.setSpecialValueText("нет")
            noise.setValue(spec.synthetic_noise or 0)
            self._columns_table.setCellWidget(i, 4, noise)
            dtype = QComboBox()
            dtype.addItems(["numeric", "binary", "categorical"])
            dtype.setCurrentText(spec.dtype)
            self._columns_table.setCellWidget(i, 5, dtype)
            mono = QComboBox()
            for text, value in (("нет", 0), ("↑", 1), ("↓", -1)):
                mono.addItem(text, value)
            mono.setCurrentIndex(mono.findData(spec.monotonic))
            self._columns_table.setCellWidget(i, 8, mono)
            role.currentIndexChanged.connect(self._roles_edited)
            self._columns_table.cellWidget(i, 2).toggled.connect(self._roles_edited)
        self._columns_table.resizeColumnsToContents()
        self._populating = False
        self.schema_edited.emit(schema)
        self._update_dataset_info(schema)

    def schema_from_ui(self):
        specs, grouping = [], []
        for i, original in enumerate(self._column_specs):
            role = ColumnRole(self._columns_table.cellWidget(i, 1).currentData())
            amp = self._columns_table.cellWidget(i, 4).value()
            specs.append(replace(original, role=role,
                                 enabled=self._columns_table.cellWidget(i, 2).isChecked(),
                                 unit=self._columns_table.item(i, 3).text(),
                                 synthetic_noise=amp or None,
                                 dtype=self._columns_table.cellWidget(i, 5).currentText(),
                                 formula_allowed=self._columns_table.cellWidget(i, 6).isChecked(),
                                 monotonic=self._columns_table.cellWidget(i, 8).currentData(),
                                 target_key=original.target_key if role == ColumnRole.TARGET else None))
            if self._columns_table.cellWidget(i, 7).isChecked():
                grouping.append(original.name)
        return DatasetSchema(specs, grouping, self.dataset.schema.legacy)

    def _roles_edited(self, *_args):
        if getattr(self, "_populating", False):
            return
        self.schema_edited.emit(self.schema_from_ui())

    def _update_dataset_info(self, schema):
        try:
            import pandas as pd
            groups = len(pd.unique(group_ids(self.dataset.frame, schema)))
        except DatasetError:
            groups = "не заданы"
        rows, cols = self.dataset.frame.shape
        self._dataset_info.setText(f"Строк: {rows} · колонок: {cols} · групп: {groups}")
        self.setTitle(f"Датасет · {rows} строк · {len(schema.features)} features")

    def apply_roles(self):
        try:
            schema = self.schema_from_ui()
            checked = Dataset(self.dataset.frame, schema, self.dataset.source).validate()
        except DatasetError as exc:
            self._dataset_info.setText(str(exc))
            self.message.emit(str(exc))
            return False
        self.dataset = checked
        self.dataset_changed.emit(checked)
        self.schema_edited.emit(schema)
        self._update_dataset_info(schema)
        self.message.emit("Роли датасета применены")
        return True

    def save_mapping(self):
        if not self.apply_roles():
            return
        try:
            path = save_schema(self.dataset.schema, self.dataset.source)
        except OSError as exc:
            QMessageBox.warning(self, "Ошибка сохранения", str(exc))
            return
        self.message.emit(f"Mapping сохранён: {path}")
