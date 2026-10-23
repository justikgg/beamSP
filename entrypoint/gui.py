import sys
from copy import deepcopy

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QComboBox
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QScrollArea
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QCheckBox, QTreeWidget, QTreeWidgetItem, QSpinBox, QDoubleSpinBox, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QTabWidget, QProgressBar, QSplitter,
    QMessageBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from config import settings as cfg
from core import reporting
from core.evaluation.runner import run
from core.formula.power_law import PowerLawForm
from core.labels import GROUP_LABELS, GROUP_ORDER, method_label, target_label
from core.models.registry import available_models, model_class
from core.models.bioinspired.formula_search import FormulaSearchModel
from core.models.bioinspired.optimizers import OPTIMIZERS
from core.data.loader import load_dataset_bundle
from core.data.schema import Dataset
from entrypoint.dataset_panel import DatasetPanel



BIO_METHOD_NAMES = {f"bio_{opt}" for opt in OPTIMIZERS}

HELP_HTML = """
<h3>Как пользоваться</h3>
<ol>
<li><b>Датасет</b> — загрузите CSV/XLSX с заголовками, выберите Feature/Target/Group,
задайте единицы и шум. Без Group отметьте Profile key. Сохраните mapping, чтобы
использовать его при следующем открытии файла.</li>
<li><b>Целевая величина</b> — выберите одну или несколько целей из текущей схемы.
Для старого файла доступны СП 63.13330 и руководство 1978 г.</li>
<li><b>Методы</b> — можно выбрать несколько сразу, просто кликая по нужным
строкам (клик в любом месте строки ставит или снимает галочку — Ctrl/Shift
не нужны), тогда в результатах они будут показаны рядом для сравнения.
Список сгруппирован по типу метода, группы сворачиваются кликом по
заголовку.</li>
<li><b>Фиксация показателей степени</b> — появляется, если среди выбранных
методов есть биоинспирированные (генетический алгоритм / дифференциальная
эволюция / рой частиц / CMA-ES). Задайте нужные значения (например
H = s = R = 1, E = 0), затем отметьте галочкой «+ фикс.» те методы, для
которых хотите дополнительно прогнать зафиксированную версию — обычная
(свободная) версия остаётся в прогоне, обе строки можно сравнить в
результатах.</li>
<li><b>Синтез образцов</b> — добавляет к исходным строкам искусственные
образцы с небольшим случайным разбросом входов (Q<sub>дв</sub> при этом не
пересчитывается) для устойчивости моделей, чувствительных к размеру выборки.
Можно отключить или изменить число образцов на профиль.</li>
<li><b>Запустить</b> — прогон по схеме Leave-One-Group-Out (обучение на остальных
профилях, проверка на отложенном). Результат: таблица «метод × метрики»,
графики предсказание/RMSE и разбивка по профилям, а для методов, выводящих
явную формулу — отдельная вкладка «Формулы».</li>
<li><b>Экспорт в results/</b> — становится активной после первого прогона.
Сохраняет сведения о данных в JSON, а результаты рядом с программой (в папку <code>results/</code>) CSV с таблицей
метрик и PNG с графиками по каждой посчитанной целевой величине, а если
среди методов были формульные — ещё и PNG со списком формул.</li>
</ol>
<h3>Метрики</h3>
<ul>
<li><b>Qexp/Qpred_mean</b> — среднее отношение «эксперимент / предсказание»
по всем образцам (в идеале ≈ 1); систематическое завышение или занижение
метода сразу видно по отклонению от 1.</li>
<li><b>CV</b> — коэффициент вариации этого отношения (разброс/среднее):
насколько стабильно метод предсказывает от образца к образцу.</li>
<li><b>within15</b> — доля образцов (%), где отношение Qэксп/Qпред попало
в коридор ±15% от идеального.</li>
<li><b>MAPE</b> — средняя абсолютная процентная ошибка относительно
Qэксп.</li>
<li><b>MAE</b> — средняя абсолютная ошибка в кН.</li>
<li><b>MedianAE</b> — медианная абсолютная ошибка в кН (устойчивее к
выбросам, чем MAE).</li>
<li><b>RMSE</b> — среднеквадратичная ошибка в кН; таблица методов
отсортирована по этому столбцу.</li>
<li><b>RMSE_worst</b> — RMSE на самом сложном из отложенных профилей
(худший случай схемы Leave-One-Group-Out).</li>
<li><b>MaxError</b> — наибольшая абсолютная ошибка среди всех образцов, кН.</li>
<li><b>pct_negative</b> — доля предсказаний (%), ушедших в отрицательные
значения — физически невозможный результат, признак поломки метода на
конкретных данных.</li>
<li><b>pct_conservative</b> — доля предсказаний (%), не превышающих
экспериментальное значение (Qпред ≤ Qэксп) — то есть «в запас», а не
в опасную сторону завышения несущей способности.</li>
<li><b>R2</b> — коэффициент детерминации на проверке LOGO (по отложенным
профилям).</li>
<li><b>R2_train</b> — тот же R2, но посчитанный на данных, на которых
финальная модель обучалась (вся выборка) — показывает, насколько хорошо
метод в принципе способен описать данные.</li>
<li><b>overfit</b> = R2_train − R2(LOGO) — насколько модель переобучилась:
большая разница означает, что метод хорошо запоминает исходные профили,
но плохо обобщается на новый.</li>
</ul>
"""


def _fixed_bio_class(opt_name, fixed_exponents):
    optimizer = OPTIMIZERS[opt_name]

    class _FixedPowerLaw(FormulaSearchModel):
        name = f"bio_{opt_name}_fixed"

        def __init__(self, seed=cfg.SEED):
            super().__init__(optimizer, form=PowerLawForm(fixed_exponents=fixed_exponents), seed=seed)

    return _FixedPowerLaw


class CollapsibleSection(QWidget):

    def __init__(self, title, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._title = title
        self._summary = ""

        self._toggle = QPushButton()
        self._toggle.setCheckable(True)
        self._toggle.setFlat(True)
        self._toggle.setStyleSheet("QPushButton { text-align: left; padding: 6px 2px; font-weight: 600; }")
        outer.addWidget(self._toggle)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 2, 4, 8)
        self._body.setVisible(False)
        outer.addWidget(self._body)

        self._toggle.toggled.connect(self._body.setVisible)
        self._toggle.toggled.connect(lambda _checked: self._refresh_text())
        self._refresh_text()

    def _refresh_text(self):
        arrow = "▾" if self._toggle.isChecked() else "▸"
        suffix = f"   [{self._summary}]" if self._summary else ""
        self._toggle.setText(f"{arrow}  {self._title}{suffix}")

    def set_summary(self, text):
        self._summary = text
        self._refresh_text()

    def set_open(self, is_open):
        self._toggle.setChecked(is_open)

    def layout_body(self):
        return self._body_layout


class RunWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model_names, targets, synth_cfg, dataset, factories=None):
        super().__init__()
        self._model_names = model_names
        self._targets = targets
        self._synth_cfg = synth_cfg
        self._dataset = dataset
        self._factories = factories

    def run(self):
        try:
            result = run(
                model_names=self._model_names, dataset=self._dataset,
                targets=self._targets, model_factories=self._factories,
                synth_cfg=self._synth_cfg, seed=cfg.SEED,
                progress=lambda d, t, label: self.progress.emit(d, t, label),
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Вклад двутавра Qдв — ML и биоинспирированная оптимизация")
        self.resize(1220, 760)
        self._worker = None
        self._last_result = None
        self._dataset = load_dataset_bundle(cfg.DATA_PATH)
        self._build_ui()

    def _build_ui(self):
        controls = QWidget()
        col = QVBoxLayout(controls)

        self._dataset_panel = DatasetPanel(self._dataset)
        self._dataset_panel.schema_edited.connect(self._refresh_dataset_controls)
        self._dataset_panel.dataset_changed.connect(lambda ds: setattr(self, "_dataset", ds))
        self._dataset_panel.file_loaded.connect(lambda legacy: self._synth_cb.setChecked(legacy))
        self._dataset_panel.message.connect(lambda msg: self.statusBar().showMessage(msg))
        col.addWidget(self._dataset_panel)

        tgt_box = QGroupBox("Целевая величина")
        tl = QVBoxLayout(tgt_box)
        self._target_boxes = {}
        self._target_layout = tl
        col.addWidget(tgt_box)

        m_box = QGroupBox("Методы (можно несколько)")
        ml = QVBoxLayout(m_box)
        self._method_tree = QTreeWidget()
        self._method_tree.setHeaderHidden(True)
        self._method_tree.setIndentation(14)

        by_group = {}
        for name in available_models():
            by_group.setdefault(model_class(name).group, []).append(name)
        groups_in_order = GROUP_ORDER + [g for g in by_group if g not in GROUP_ORDER]

        first_item = None
        first_group_item = None
        for group in groups_in_order:
            names = by_group.get(group)
            if not names:
                continue
            group_item = QTreeWidgetItem([f"{GROUP_LABELS.get(group, group)} ({len(names)})"])
            font = group_item.font(0)
            font.setBold(True)
            group_item.setFont(0, font)
            self._method_tree.addTopLevelItem(group_item)
            for name in names:
                item = QTreeWidgetItem([method_label(name)])
                item.setCheckState(0, Qt.CheckState.Unchecked)
                
                
                
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                item.setData(0, Qt.ItemDataRole.UserRole, name)
                group_item.addChild(item)
                if first_item is None:
                    first_item = item
                    first_group_item = group_item
        if first_item is not None:
            first_item.setCheckState(0, Qt.CheckState.Checked)
            first_item.setFlags(first_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        if first_group_item is not None:
            self._method_tree.expandItem(first_group_item)
        ml.addWidget(self._method_tree)
        col.addWidget(m_box, stretch=1)

        self._fix_section = CollapsibleSection("Фиксация показателей степени (для bio_*)")
        note = QLabel("Значения ниже применяются только к тем bio_*-методам, что отмечены "
                       "галочкой «+ фикс.» — обычная (свободная) версия при этом остаётся "
                       "в прогоне, обе строки можно сравнить в результатах.")
        note.setWordWrap(True)
        self._fix_section.layout_body().addWidget(note)
        self._fix_rows = {}
        self._fix_features = QWidget()
        self._fix_features_layout = QVBoxLayout(self._fix_features)
        self._fix_section.layout_body().addWidget(self._fix_features)

        twin_label = QLabel("Прогнать зафиксированную версию рядом с обычной:")
        twin_label.setWordWrap(True)
        self._fix_section.layout_body().addWidget(twin_label)
        self._fix_twin_container = QWidget()
        self._fix_twin_layout = QVBoxLayout(self._fix_twin_container)
        self._fix_twin_layout.setContentsMargins(0, 0, 0, 0)
        self._fix_section.layout_body().addWidget(self._fix_twin_container)
        self._fix_twin_checks = {}

        self._fix_section.set_open(True)
        col.addWidget(self._fix_section)

        self._method_tree.itemChanged.connect(self._update_fix_box_visibility)
        self._method_tree.itemClicked.connect(self._on_tree_item_clicked)
        self._update_fix_box_visibility()
        self._refresh_fix_summary()

        self._synth_section = CollapsibleSection("Синтез образцов")
        self._synth_cb = QCheckBox("Включить синтез")
        self._synth_cb.setChecked(cfg.SYNTH["enabled"])
        self._synth_cb.toggled.connect(self._refresh_synth_summary)
        self._synth_section.layout_body().addWidget(self._synth_cb)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Копий каждой строки:"))
        self._samples_spin = QSpinBox()
        self._samples_spin.setRange(0, 500)
        self._samples_spin.setValue(cfg.SYNTH["samples_per_profile"])
        self._samples_spin.valueChanged.connect(self._refresh_synth_summary)
        srow.addWidget(self._samples_spin)
        self._synth_section.layout_body().addLayout(srow)
        col.addWidget(self._synth_section)
        self._refresh_synth_summary()

        self._run_btn = QPushButton("▶  Запустить")
        self._run_btn.clicked.connect(self._on_run)
        col.addWidget(self._run_btn)
        self._progress = QProgressBar()
        col.addWidget(self._progress)

        self._export_btn = QPushButton("💾  Экспорт в results/")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        col.addWidget(self._export_btn)

        self._results = QTabWidget()
        placeholder = QLabel("Выберите методы и цели, затем «Запустить».")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._results.addTab(placeholder, "Результаты")
        self._results.addTab(self._build_help_tab(), "Помощь")

        splitter = QSplitter()
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setWidget(controls)
        splitter.addWidget(control_scroll)
        splitter.addWidget(self._results)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 890])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Готов")
        self._refresh_dataset_controls(self._dataset.schema)

    def _selected_targets(self):
        return {k: self._dataset.schema.targets[k] for k, cb in self._target_boxes.items()
                if cb.isChecked()}

    def _refresh_dataset_controls(self, schema):
        selected = {k: cb.isChecked() for k, cb in self._target_boxes.items()}
        while self._target_layout.count():
            old = self._target_layout.takeAt(0).widget()
            old.setParent(None)
            old.deleteLater()
        self._target_boxes = {}
        for key in schema.targets:
            cb = QCheckBox(target_label(key))
            cb.setChecked(selected.get(key, True))
            self._target_boxes[key] = cb
            self._target_layout.addWidget(cb)
        previous = {n: (cb.isChecked(), spin.value()) for n, (cb, spin) in self._fix_rows.items()}
        while self._fix_features_layout.count():
            old = self._fix_features_layout.takeAt(0).widget()
            old.setParent(None)
            old.deleteLater()
        self._fix_rows = {}
        for name in schema.features:
            spec = schema.spec(name)
            holder = QWidget()
            row = QHBoxLayout(holder)
            cb = QCheckBox(spec.display_name or name)
            spin = QDoubleSpinBox()
            spin.setRange(-5, 5)
            spin.setSingleStep(0.5)
            checked, value = previous.get(name, (False, 0.0))
            cb.setChecked(checked)
            spin.setValue(value)
            spin.setEnabled(checked)
            cb.toggled.connect(spin.setEnabled)
            cb.toggled.connect(self._refresh_fix_summary)
            spin.valueChanged.connect(self._refresh_fix_summary)
            row.addWidget(cb)
            row.addWidget(spin)
            self._fix_features_layout.addWidget(holder)
            self._fix_rows[name] = (cb, spin)
        self._refresh_fix_summary()

    def _selected_methods(self):
        names = []
        root = self._method_tree.invisibleRootItem()
        for gi in range(root.childCount()):
            group_item = root.child(gi)
            for ci in range(group_item.childCount()):
                item = group_item.child(ci)
                if item.checkState(0) == Qt.CheckState.Checked:
                    names.append(item.data(0, Qt.ItemDataRole.UserRole))
        return names

    def _on_tree_item_clicked(self, item, _column):
        if item.parent() is None:
            return  
        state = Qt.CheckState.Unchecked if item.checkState(0) == Qt.CheckState.Checked else Qt.CheckState.Checked
        item.setCheckState(0, state)
        
        
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)

    def _update_fix_box_visibility(self, *_args):
        selected_bio = [n for n in self._selected_methods() if n in BIO_METHOD_NAMES]
        self._fix_section.setVisible(bool(selected_bio))
        self._rebuild_fix_twin_rows(selected_bio)

    def _rebuild_fix_twin_rows(self, selected_bio):
        wanted = set(selected_bio)
        existing = set(self._fix_twin_checks)
        for name in existing - wanted:
            cb = self._fix_twin_checks.pop(name)
            self._fix_twin_layout.removeWidget(cb)
            cb.deleteLater()
        for name in wanted - existing:
            cb = QCheckBox(f"+ фикс.: {method_label(name)}")
            self._fix_twin_layout.addWidget(cb)
            self._fix_twin_checks[name] = cb

    def _refresh_fix_summary(self):
        checked = [(name, spin.value()) for name, (cb, spin) in self._fix_rows.items() if cb.isChecked()]
        if not checked:
            summary = "не задано"
        elif len(checked) <= 2:
            summary = " · ".join(f"{name}={v:g}" for name, v in checked)
        else:
            head = " · ".join(f"{name}={v:g}" for name, v in checked[:2])
            summary = f"{head} +{len(checked) - 2}"
        self._fix_section.set_summary(summary)

    def _refresh_synth_summary(self):
        if self._synth_cb.isChecked():
            self._synth_section.set_summary(f"вкл · {self._samples_spin.value()}/профиль")
        else:
            self._synth_section.set_summary("выкл")

    def _build_help_tab(self):
        label = QLabel(HELP_HTML)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        label.setTextFormat(Qt.TextFormat.RichText)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        return scroll

    def _fixed_exponents(self):
        fixed = {}
        for feat_name, (cb, spin) in self._fix_rows.items():
            if cb.isChecked():
                fixed[feat_name] = spin.value()
        return fixed

    def _apply_fixation(self, methods, fixed_exponents):
        if not fixed_exponents:
            return methods, {}
        extra = []
        factories = {}
        for name, cb in self._fix_twin_checks.items():
            if not cb.isChecked() or name not in methods:
                continue
            opt_name = name[len("bio_"):]
            fixed_name = f"bio_{opt_name}_fixed"
            factories[fixed_name] = _fixed_bio_class(opt_name, fixed_exponents)
            extra.append(fixed_name)
        return methods + extra, factories

    def _on_run(self):
        if not self._dataset_panel.apply_roles():
            return
        methods = self._selected_methods()
        targets = self._selected_targets()
        if not methods:
            self.statusBar().showMessage("Выберите хотя бы один метод")
            return
        if not targets:
            self.statusBar().showMessage("Выберите хотя бы одну цель")
            return

        methods, factories = self._apply_fixation(methods, self._fixed_exponents())

        synth_cfg = dict(cfg.SYNTH)
        synth_cfg["noise"] = self._dataset.schema.noise
        synth_cfg["enabled"] = self._synth_cb.isChecked()
        synth_cfg["samples_per_profile"] = self._samples_spin.value()

        self._run_btn.setEnabled(False)
        self._progress.setRange(0, 0)
        self.statusBar().showMessage("Запуск…")

        snapshot = Dataset(self._dataset.frame.copy(), deepcopy(self._dataset.schema), self._dataset.source)
        self._worker = RunWorker(methods, targets, synth_cfg, snapshot, factories)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, done, total, label):
        self._progress.setRange(0, total)
        self._progress.setValue(done)
        self.statusBar().showMessage(f"[{done}/{total}] {label}")

    def _on_done(self, result):
        self._run_btn.setEnabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self.statusBar().showMessage("Готово")
        self._last_result = result
        self._export_btn.setEnabled(True)
        self._show_results(result)

    def _on_failed(self, msg):
        self._run_btn.setEnabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self.statusBar().showMessage(f"Ошибка: {msg}")
        QMessageBox.warning(self, "Ошибка расчёта", msg)

    def _on_export(self):
        if self._last_result is None:
            return
        try:
            saved = reporting.export_results(self._last_result, cfg.RESULTS_DIR)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Ошибка экспорта", str(exc))
            return
        self.statusBar().showMessage(f"Сохранено в {cfg.RESULTS_DIR}: {len(saved)} файлов")

    def _show_results(self, result):
        self._results.clear()
        for tkey, table in result.metrics.items():
            if table.empty: continue
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.addWidget(self._metrics_table(table))
            notes = reporting.model_notes(result, tkey)
            if notes:
                label = QLabel("\n".join(notes))
                label.setWordWrap(True)
                layout.addWidget(label)
            figs = QTabWidget()
            figs.addTab(self._prediction_tab(result, tkey), "Предсказание / RMSE")
            fold = reporting.build_fold_rmse_figure(result, tkey)
            if fold is not None: figs.addTab(self._canvas(fold), "По профилям")
            layout.addWidget(figs, stretch=1)
            self._results.addTab(tab, target_label(tkey))

        if any(result.formulas.values()):
            fig = reporting.build_formulas_figure(result)
            canvas = self._canvas(fig)
            canvas.setMinimumSize(canvas.sizeHint())
            scroll = QScrollArea()
            scroll.setWidgetResizable(False)
            scroll.setWidget(canvas)
            self._results.addTab(scroll, "Формулы")

        self._results.addTab(self._build_help_tab(), "Помощь")

    def _metrics_table(self, df):
        disp = df.round(3)
        table = QTableWidget(len(disp.index), len(disp.columns))
        table.setHorizontalHeaderLabels([str(c) for c in disp.columns])
        table.setVerticalHeaderLabels([method_label(m) for m in disp.index])
        for i, (_, values) in enumerate(disp.iterrows()):
            for j, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(i, j, item)
        table.resizeColumnsToContents()
        table.setMaximumHeight(46 + 30 * len(disp.index))
        return table

    def _canvas(self, fig):
        return FigureCanvasQTAgg(fig)

    def _prediction_tab(self, result, tkey):
        container = QWidget()
        v = QVBoxLayout(container)

        metrics_df = result.metrics.get(tkey)
        model_names = list(metrics_df.index) if metrics_df is not None else []
        default_model = (metrics_df["RMSE"].idxmin()
                         if metrics_df is not None and not metrics_df.empty else None)

        combo = QComboBox()
        for name in model_names:
            combo.addItem(method_label(name), name)
        if default_model in model_names:
            combo.setCurrentIndex(model_names.index(default_model))

        v.addWidget(combo)
        canvas_holder = QVBoxLayout()
        v.addLayout(canvas_holder)

        state = {"canvas": None, "fig": None}

        def rebuild(model_name):
            if state["canvas"] is not None:
                canvas_holder.removeWidget(state["canvas"])
                state["canvas"].deleteLater()
            if state["fig"] is not None:
                plt.close(state["fig"])
            fig = reporting.build_comparison_figure(result, tkey, scatter_model=model_name)
            canvas = self._canvas(fig)
            canvas_holder.addWidget(canvas)
            state["canvas"] = canvas
            state["fig"] = fig

        combo.currentIndexChanged.connect(lambda i: rebuild(combo.itemData(i)))
        rebuild(default_model)

        return container


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
