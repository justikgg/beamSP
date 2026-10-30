"""Dataset mapping editor; loading and validation are delegated to core.data."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import re

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QTableWidget, QTableWidgetItem, QComboBox, QCheckBox,
                             QDoubleSpinBox, QFileDialog, QMessageBox, QLineEdit,
                             QDialog, QDialogButtonBox, QFrame)

from core.data.loader import load_dataset_bundle
from core.data.schema import (ColumnRole, Dataset, DatasetError, DatasetSchema,
                              group_ids, mapping_path, read_schema, save_schema)


class TableHeightHandle(QFrame):
    """Resize the role table, allowing its scrollable page to grow with it."""

    def __init__(self, table, parent=None):
        super().__init__(parent)
        self._table = table
        self._drag_start = None
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(12)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip("Потяните вверх или вниз, чтобы изменить высоту таблицы. "
                        "Двойной щелчок — автоматическая высота.")
        self.setAccessibleName("Изменить высоту таблицы ролей")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = (event.globalPosition().y(), self._table.height())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            start_y, height = self._drag_start
            self._table.setFixedHeight(max(200, round(height + event.globalPosition().y() - start_y)))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self._table.setMinimumHeight(200)
            self._table.setMaximumHeight(16777215)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


class MappingColumnsDialog(QDialog):
    """Explicit column matching when a mapping comes from another dataset."""

    def __init__(self, schema, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сопоставить колонки разметки")
        self.resize(650, 420)
        layout = QVBoxLayout(self)
        note = QLabel("Выберите колонки текущего датасета для исходной разметки. "
                      "Лишние колонки можно пропустить; новые останутся Ignore.")
        note.setWordWrap(True)
        layout.addWidget(note)
        table = QTableWidget(len(schema.columns), 3)
        table.setHorizontalHeaderLabels(["Колонка разметки", "Роль", "Колонка датасета"])
        self._matches = {}
        for row, spec in enumerate(schema.columns):
            for col, text in ((0, spec.name), (1, spec.role.value)):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row, col, item)
            combo = QComboBox()
            combo.addItem("Пропустить", None)
            for name in columns:
                combo.addItem(name, name)
            combo.setCurrentIndex(max(0, combo.findData(spec.name)))
            table.setCellWidget(row, 2, combo)
            self._matches[spec.name] = combo
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._schema, self._columns = schema, list(columns)

    def column_mapping(self):
        return {name: combo.currentData() for name, combo in self._matches.items()}

    def accept(self):
        try:
            self._schema.remap(self._columns, self.column_mapping())
        except DatasetError as exc:
            QMessageBox.warning(self, "Ошибка сопоставления", str(exc))
            return
        super().accept()


class DatasetPanel(QGroupBox):
    schema_edited = pyqtSignal(object)
    dataset_changed = pyqtSignal(object)
    file_loaded = pyqtSignal(bool)
    message = pyqtSignal(str)

    def __init__(self, dataset, parent=None):
        super().__init__("Датасет", parent)
        self.dataset = dataset
        sidecar = mapping_path(dataset.source) if dataset.source else None
        self._mapping_path = sidecar if sidecar and sidecar.exists() else None
        self._external_mappings = set()
        dl = QVBoxLayout(self)
        load_btn = QPushButton("Загрузить датасет")
        load_btn.clicked.connect(self.load_dataset)
        dl.addWidget(load_btn)
        self._path_label = QLabel()
        self._path_label.setWordWrap(True)
        dl.addWidget(self._path_label)
        mappings = QHBoxLayout()
        mappings.addWidget(QLabel("Разметка:"))
        self._mapping_combo = QComboBox()
        self._mapping_combo.setMinimumContentsLength(15)
        self._mapping_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._mapping_combo.currentIndexChanged.connect(self._select_mapping)
        mappings.addWidget(self._mapping_combo, 1)
        dl.addLayout(mappings)
        self._load_mapping_btn = QPushButton("Загрузить разметку…")
        self._load_mapping_btn.clicked.connect(self.load_mapping)
        dl.addWidget(self._load_mapping_btn)
        self._mapping_name = QLineEdit()
        self._mapping_name.setPlaceholderText("Название разметки")
        dl.addWidget(self._mapping_name)
        self._dataset_info = QLabel()
        self._dataset_info.setWordWrap(True)
        dl.addWidget(self._dataset_info)
        self._columns_table = QTableWidget()
        self._columns_table.setColumnCount(9)
        self._columns_table.setHorizontalHeaderLabels([
            "Column", "Role", "Use", "Unit", "Noise ±", "Type", "Formula", "Profile key", "Monotonic"])
        self._columns_table.setMinimumHeight(200)
        dl.addWidget(self._columns_table, 1)
        self._table_resize_handle = TableHeightHandle(self._columns_table)
        dl.addWidget(self._table_resize_handle)
        buttons = QVBoxLayout()
        apply_btn = QPushButton("Применить роли")
        apply_btn.clicked.connect(self.apply_roles)
        save_btn = QPushButton("Сохранить разметку как…")
        save_btn.clicked.connect(self.save_mapping)
        buttons.addWidget(apply_btn)
        buttons.addWidget(save_btn)
        dl.addLayout(buttons)
        note = QLabel("Без Group отметьте Profile key. Noise задаётся только для числовых Feature. "
                      "Formula разрешает признак в степенных моделях (bio, LS, TRF); "
                      "Monotonic задаёт ограничение PINN.")
        note.setWordWrap(True)
        dl.addWidget(note)
        dl.addStretch()
        self._refresh_mappings()
        self._populate()

    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить датасет", "", "Датасеты (*.xlsx *.csv)")
        if not path:
            return
        sidecar = mapping_path(path)
        selected_mapping = None
        mapping_error = None
        try:
            try:
                dataset = load_dataset_bundle(path, validate=False)
                if sidecar.exists():
                    selected_mapping = sidecar
                    if {c.name for c in dataset.schema.columns} - set(dataset.frame.columns):
                        mapped = self._mapped_dataset(dataset, sidecar)
                        if mapped is None:
                            dataset = load_dataset_bundle(path, validate=False, use_sidecar=False)
                            selected_mapping = None
                        else:
                            dataset = mapped
            except (DatasetError, OSError) as exc:
                # A broken sidecar must not prevent opening the dataset.
                if not sidecar.exists():
                    raise
                mapping_error = str(exc)
                selected_mapping = None
                dataset = load_dataset_bundle(path, validate=False, use_sidecar=False)
        except (DatasetError, OSError) as exc:
            QMessageBox.warning(self, "Ошибка датасета", str(exc))
            return
        self.dataset = dataset
        self._mapping_path = selected_mapping
        self._refresh_mappings()
        self.dataset_changed.emit(dataset)
        self._populate()
        self.file_loaded.emit(dataset.schema.legacy)
        # Incomplete mappings remain editable; the validation error is visible.
        self.apply_roles()
        if mapping_error:
            QMessageBox.warning(self, "Ошибка разметки", mapping_error)

    def _refresh_mappings(self):
        self._mapping_combo.blockSignals(True)
        self._mapping_combo.clear()
        self._mapping_combo.addItem("Автоматическая разметка", None)
        paths = set(self._external_mappings)
        if self.dataset.source:
            paths.update(self.dataset.source.parent.glob("*.beamsp.json"))
        if self._mapping_path:
            paths.add(self._mapping_path)
        for path in sorted(paths, key=lambda p: str(p).casefold()):
            try:
                name = read_schema(path).name or path.name
            except (DatasetError, OSError):
                name = path.name
            self._mapping_combo.addItem(f"{name} ({path.name})" if name != path.name else name, str(path))
            self._mapping_combo.setItemData(self._mapping_combo.count() - 1, str(path), Qt.ItemDataRole.ToolTipRole)
        self._mapping_combo.setCurrentIndex(max(0, self._mapping_combo.findData(
            str(self._mapping_path) if self._mapping_path else None)))
        self._mapping_combo.blockSignals(False)

    def _mapped_dataset(self, dataset, path):
        schema = read_schema(path)
        missing = {c.name for c in schema.columns} - set(dataset.frame.columns)
        mapping = None
        if missing:
            dialog = MappingColumnsDialog(schema, dataset.frame.columns, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            mapping = dialog.column_mapping()
        schema = schema.remap(dataset.frame.columns, mapping)
        # Keep the provenance of generated columns even when they are ignored.
        sources = {c.name: c.derived_from for c in dataset.schema.columns if c.derived_from}
        for spec in schema.columns:
            if spec.name in sources:
                spec.derived_from = sources[spec.name]
        # Legacy describes the dataset format, not the mapping's origin.
        schema.legacy = dataset.schema.legacy
        frame = dataset.frame.copy()
        frame.attrs["beamsp_schema"] = schema.to_dict()
        return Dataset(frame, schema, dataset.source)

    def _select_mapping(self, _index):
        self.load_mapping(self._mapping_combo.currentData())

    def load_mapping(self, path=False):
        if path is False:  # QPushButton.clicked passes a boolean.
            path, _ = QFileDialog.getOpenFileName(
                self, "Загрузить разметку", str(self.dataset.source.parent) if self.dataset.source else "",
                "Разметки (*.beamsp.json *.json);;Все файлы (*)")
            if not path:
                return
        try:
            if path is None:
                dataset = load_dataset_bundle(self.dataset.source, validate=False, use_sidecar=False)
            else:
                path = Path(path).resolve()
                dataset = self._mapped_dataset(self.dataset, path)
                if dataset is None:
                    self._refresh_mappings()
                    return
        except (DatasetError, OSError) as exc:
            self._refresh_mappings()
            QMessageBox.warning(self, "Ошибка разметки", str(exc))
            return
        self.dataset = dataset
        self._mapping_path = path
        if path:
            self._external_mappings.add(path)
        self._refresh_mappings()
        self.dataset_changed.emit(dataset)
        self._populate()
        self.apply_roles()

    def _populate(self):
        self._populating = True
        schema = self.dataset.schema
        self._mapping_name.setText(schema.name)
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
        return DatasetSchema(specs, grouping, self.dataset.schema.legacy,
                             self._mapping_name.text().strip())

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
        name = self.dataset.schema.name or "Разметка"
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .") or "Разметка"
        safe_name = safe_name.encode("utf-8")[:100].decode("utf-8", errors="ignore")
        source = self.dataset.source or Path.cwd() / "dataset.csv"
        # Always suggest a new file, including when saving an edited mapping.
        stem = source.stem.encode("utf-8")[:80].decode("utf-8", errors="ignore")
        base = f"{stem}.{safe_name}"
        proposed = source.parent / f"{base}.beamsp.json"
        number = 2
        while proposed.exists():
            proposed = source.parent / f"{base} ({number}).beamsp.json"
            number += 1
        selected, _ = QFileDialog.getSaveFileName(
            self, "Сохранить разметку как", str(proposed), "Разметки (*.beamsp.json);;JSON (*.json)")
        if not selected:
            return
        try:
            path = Path(selected)
            if path.suffix.lower() != ".json":
                path = Path(str(path) + ".beamsp.json")
                if path.exists() and QMessageBox.question(
                        self, "Заменить разметку?", f"Файл {path.name} уже существует. Заменить?"
                        ) != QMessageBox.StandardButton.Yes:
                    return
            schema = deepcopy(self.dataset.schema)
            if not schema.name:
                schema.name = path.name.removesuffix(".json").removesuffix(".beamsp")
            path = save_schema(schema, path)
        except OSError as exc:
            QMessageBox.warning(self, "Ошибка сохранения", str(exc))
            return
        self.dataset.schema = schema
        self.dataset.frame.attrs["beamsp_schema"] = schema.to_dict()
        self._mapping_name.setText(schema.name)
        self._mapping_path = path.resolve()
        self._external_mappings.add(self._mapping_path)
        self._refresh_mappings()
        self.message.emit(f"Разметка сохранена: {path}")
