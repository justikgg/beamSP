"""Offscreen integration of dataset editing, fixed powers and the Run button."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('MPLCONFIGDIR','/tmp/beamsp-mpl-tests')
import tempfile
import subprocess
import sys
from pathlib import Path
from time import monotonic
import unittest
from unittest.mock import patch

try:
    from PyQt6.QtCore import QEventLoop, QPoint, QTimer, Qt
    from PyQt6.QtWidgets import QApplication, QDialog, QScrollArea
    from PyQt6.QtTest import QTest
    from entrypoint.gui import MainWindow
    from entrypoint.dataset_panel import MappingColumnsDialog
    QT_AVAILABLE=True
except ImportError:
    QT_AVAILABLE=False

from core.data.loader import load_dataset_bundle
from core.data.schema import ColumnRole, mapping_path, read_schema, save_schema
from test_pipeline import dynamic
from test_material import material_frame


@unittest.skipUnless(QT_AVAILABLE,'PyQt6 is optional; install requirements-gui.txt')
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.window=MainWindow()
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'dynamic.csv'
        dynamic().frame.to_csv(self.path,index=False)

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.tmp.cleanup()

    def load(self):
        with patch('entrypoint.dataset_panel.QFileDialog.getOpenFileName',return_value=(str(self.path),'csv')):
            self.window._dataset_panel.load_dataset()

    def method_item(self, name):
        tree = self.window._method_tree
        for i in range(tree.topLevelItemCount()):
            group = tree.topLevelItem(i)
            for j in range(group.childCount()):
                item = group.child(j)
                if item.data(0, Qt.ItemDataRole.UserRole) == name:
                    return item
        self.fail(f'Метод {name} отсутствует в дереве')

    def test_visible_checks_toggle_once_and_single_method_button(self):
        w = self.window
        tree = w._method_tree
        self.assertEqual(w._selected_method_count.text(), 'Выбрано методов: 0')
        self.assertEqual(w._selected_methods(), [])
        linear = self.method_item('linear')
        pso = self.method_item('bio_pso')
        self.assertEqual(linear.checkState(0), Qt.CheckState.Unchecked)
        self.assertEqual(pso.checkState(0), Qt.CheckState.Unchecked)
        self.assertTrue(pso.flags() & Qt.ItemFlag.ItemIsUserCheckable)
        tree.expandItem(pso.parent())
        w.show()
        self.app.processEvents()
        rect = tree.visualItemRect(pso)
        QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton,
                         pos=QPoint(rect.left() + 8, rect.center().y()))
        self.assertEqual(pso.checkState(0), Qt.CheckState.Checked)
        self.assertEqual(w._selected_method_count.text(), 'Выбрано методов: 1')
        QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton,
                         pos=QPoint(rect.left() + 8, rect.center().y()))
        self.assertEqual(pso.checkState(0), Qt.CheckState.Unchecked)
        self.assertEqual(w._selected_method_count.text(), 'Выбрано методов: 0')
        QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton,
                         pos=QPoint(rect.left() + 60, rect.center().y()))
        self.assertEqual(pso.checkState(0), Qt.CheckState.Checked)
        QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton,
                         pos=QPoint(rect.left() + 60, rect.center().y()))
        self.assertEqual(pso.checkState(0), Qt.CheckState.Unchecked)
        tree.setCurrentItem(pso)
        w._only_method_btn.click()
        self.assertEqual(w._selected_methods(), ['bio_pso'])
        self.assertEqual(linear.checkState(0), Qt.CheckState.Unchecked)
        self.assertEqual(w._selected_method_count.text(), 'Выбрано методов: 1')

    def test_tuning_lists_multiple_methods_and_ignores_single_value_of_other_method(self):
        self.load()
        w = self.window
        self.method_item('linear').setCheckState(0, Qt.CheckState.Checked)
        pso = self.method_item('bio_pso')
        pso.setCheckState(0, Qt.CheckState.Checked)
        w._parameters.tune_mode.setChecked(True)
        with patch('entrypoint.gui.QMessageBox.warning') as warning:
            w._on_run()
        message = warning.call_args.args[2]
        self.assertIn('Линейная регрессия', message)
        self.assertIn('рой частиц', message)
        w._method_tree.setCurrentItem(pso)
        w._only_method_btn.click()
        w._parameters._values['pinn'] = {'hidden': 'bad'}
        editors = w._parameters._tune_editors
        editors['n_particles'][1].setValue(3)
        editors['n_particles'][2].setValue(3)
        editors['iters'][1].setValue(1)
        editors['iters'][2].setValue(2)
        editors['iters'][3].setValue(1)
        w._synth_cb.setChecked(False)
        for key, cb in w._target_boxes.items():
            cb.setChecked(key == 'target_a')
        with patch('entrypoint.gui.QMessageBox.warning') as warning:
            w._on_run()
            deadline = monotonic() + 15
            while w._run_btn.isEnabled() is False and monotonic() < deadline:
                self.app.processEvents()
                QTest.qWait(10)
        self.assertFalse(warning.called, warning.call_args)
        self.assertIsNotNone(w._last_result)
        self.assertEqual(w._last_result.metadata['tuning']['target_a']['combinations'], 2)

    def test_run_with_no_methods_shows_warning_and_does_not_start_worker(self):
        w = self.window
        self.assertEqual(w._selected_methods(), [])
        self.assertEqual(w._selected_method_count.text(), 'Выбрано методов: 0')
        with patch('entrypoint.gui.QMessageBox.warning') as warning:
            w._on_run()
        warning.assert_called_once_with(
            w, 'Метод не выбран', 'Выберите хотя бы один метод')
        self.assertIsNone(w._worker)
        self.assertIsNone(w._tune_worker)

        self.method_item('linear').setCheckState(0, Qt.CheckState.Checked)
        self.method_item('linear').setCheckState(0, Qt.CheckState.Unchecked)
        self.assertEqual(w._selected_methods(), [])
        self.assertEqual(w._selected_method_count.text(), 'Выбрано методов: 0')

    def test_fixed_pso_tuning_keeps_fixed_formula_in_final_result(self):
        self.load()
        w = self.window
        pso = self.method_item('bio_pso')
        w._method_tree.setCurrentItem(pso)
        w._only_method_btn.click()
        w._parameters.tune_mode.setChecked(True)
        editors = w._parameters._tune_editors
        for key, value in (('n_particles', 3), ('iters', 1)):
            editors[key][1].setValue(value)
            editors[key][2].setValue(value)
        w._fix_rows['x1'][0].setChecked(True)
        w._fix_rows['x1'][1].setValue(1)
        w._fix_twin_checks['bio_pso'].setChecked(True)
        w._synth_cb.setChecked(False)
        for key, cb in w._target_boxes.items():
            cb.setChecked(key == 'target_a')
        with patch('entrypoint.gui.QMessageBox.warning') as warning:
            w._on_run()
            deadline = monotonic() + 15
            while not w._run_btn.isEnabled() and monotonic() < deadline:
                self.app.processEvents()
                QTest.qWait(10)
        self.assertFalse(warning.called, warning.call_args)
        config = w._last_result.metadata['model_config']['target_a']['bio_pso_fixed']
        self.assertEqual(config['fixed_exponents'], {'x1': 1.0})
        self.assertIn('bio_pso_fixed', w._last_result.formulas['target_a'])

    def test_tuning_editor_exposes_new_methods_and_explains_ls(self):
        panel = self.window._parameters
        for name in ('bio_pso', 'bio_cmaes', 'symreg', 'bayes_symreg', 'pinn', 'rl_ac'):
            panel.methods.setCurrentIndex(panel.methods.findData(name))
            panel.tune_mode.setChecked(True)
            self.assertTrue(panel._tune_editors, name)
            self.assertEqual(set(panel.tuning_snapshot([name])[name]),
                             set(panel._tune_editors))
        panel.methods.setCurrentIndex(panel.methods.findData('log_linear_ls'))
        self.assertFalse(panel._tune_editors)
        self.assertIn('обычное обучение',
                      panel.tune_form.itemAt(panel.tune_form.count() - 1).widget().text())

    def test_export_to_selected_directory_and_remember_it(self):
        w = self.window
        w._last_result = object()
        chosen = Path(self.tmp.name) / 'chosen'
        with patch('entrypoint.gui.QFileDialog.getExistingDirectory',
                   return_value=str(chosen)) as dialog, patch(
                       'entrypoint.gui.reporting.export_results',
                       return_value=['result.xlsx']) as export:
            w._on_export()
            export.assert_called_once_with(w._last_result, chosen)
            self.assertIn(str(chosen), w.statusBar().currentMessage())
            w._on_export()
            self.assertEqual(dialog.call_args.args[2], str(chosen))

    def test_export_excel_only_to_selected_directory(self):
        w = self.window
        w._last_result = object()
        w._export_mode.setCurrentIndex(w._export_mode.findData(True))
        with patch('entrypoint.gui.QFileDialog.getExistingDirectory',
                   return_value=self.tmp.name), patch(
                       'entrypoint.gui.reporting.export_results',
                       return_value=['result.xlsx']) as export:
            w._on_export()
            export.assert_called_once_with(
                w._last_result, Path(self.tmp.name), excel_only=True)

    def test_export_cancel_does_not_save(self):
        w = self.window
        w._last_result = object()
        previous = w._export_dir
        with patch('entrypoint.gui.QFileDialog.getExistingDirectory',
                   return_value=''), patch(
                       'entrypoint.gui.reporting.export_results') as export:
            w._on_export()
            export.assert_not_called()
        self.assertEqual(w._export_dir, previous)

    def test_export_permission_error_is_shown(self):
        w = self.window
        w._last_result = object()
        previous = w._export_dir
        with patch('entrypoint.gui.QFileDialog.getExistingDirectory',
                   return_value=self.tmp.name), patch(
                       'entrypoint.gui.reporting.export_results',
                       side_effect=PermissionError('Нет доступа')), patch(
                           'entrypoint.gui.QMessageBox.warning') as warning:
            w._on_export()
            warning.assert_called_once_with(w, 'Ошибка экспорта', 'Нет доступа')
        self.assertEqual(w._export_dir, previous)

    def test_load_roles_targets_units_noise_and_worker_run(self):
        self.load()
        w=self.window
        self.method_item('linear').setCheckState(0, Qt.CheckState.Checked)
        self.assertEqual(w._dataset_panel._columns_table.rowCount(),11)
        self.assertEqual(list(w._fix_rows),[f'x{i}' for i in range(1,9)])
        self.assertEqual(set(w._target_boxes),{'target_a','target_b'})
        row=next(i for i,s in enumerate(w._dataset_panel._column_specs) if s.name=='x8')
        combo=w._dataset_panel._columns_table.cellWidget(row,1)
        combo.setCurrentIndex(combo.findData('ignore'))
        self.assertNotIn('x8',w._fix_rows)
        row=next(i for i,s in enumerate(w._dataset_panel._column_specs) if s.name=='x2')
        w._dataset_panel._columns_table.item(row,3).setText('mm')
        w._dataset_panel._columns_table.cellWidget(row,4).setValue(0.1)
        w._dataset_panel._columns_table.cellWidget(row,8).setCurrentIndex(1)
        self.assertTrue(w._dataset_panel.apply_roles())
        with patch('entrypoint.dataset_panel.QFileDialog.getSaveFileName',
                   return_value=(str(mapping_path(self.path)), 'json')):
            w._dataset_panel.save_mapping()
        loaded=load_dataset_bundle(self.path)
        self.assertEqual(loaded.schema.spec('x2').unit,'mm')
        self.assertEqual(loaded.schema.spec('x2').synthetic_noise,0.1)
        self.assertEqual(loaded.schema.spec('x2').monotonic,1)
        w._target_boxes['target_b'].setChecked(False)
        w._synth_cb.setChecked(True)
        w._samples_spin.setValue(1)
        errors=[]
        with patch('entrypoint.gui.QMessageBox.warning',side_effect=lambda *args:errors.append(args)):
            w._on_run()
            loop=QEventLoop()
            w._worker.finished.connect(loop.quit)
            QTimer.singleShot(15000,loop.quit)
            loop.exec()
            self.assertTrue(w._worker.wait(1000))
            self.app.processEvents()
        self.assertFalse(errors,errors)
        self.assertIsNotNone(w._last_result)
        self.assertEqual(w._last_result.metadata['feature_columns'],[f'x{i}' for i in range(1,8)])
        self.assertEqual(w._last_result.metadata['synthetic_rows'],32)
        self.assertEqual(set(w._last_result.metrics),{'target_a'})
        self.assertTrue(w._export_btn.isEnabled())

    def test_new_column_with_existing_mapping_is_visible_and_selectable(self):
        ds=dynamic()
        save_schema(ds.schema,self.path)
        ds.frame['x9']=ds.frame.x1*2
        ds.frame.to_csv(self.path,index=False)
        self.load()
        w=self.window
        self.assertEqual(w._dataset_panel._columns_table.rowCount(),12)
        row=next(i for i,s in enumerate(w._dataset_panel._column_specs) if s.name=='x9')
        combo=w._dataset_panel._columns_table.cellWidget(row,1)
        self.assertEqual(combo.currentData(),'ignore')
        combo.setCurrentIndex(combo.findData('feature'))
        self.assertIn('x9',w._fix_rows)
        self.assertTrue(w._dataset_panel.apply_roles())
        self.assertEqual(len(w._dataset.schema.features),9)
        w._fix_rows['x9'][0].setChecked(True)
        w._fix_rows['x9'][1].setValue(1)
        self.assertEqual(w._fixed_exponents(),{'x9':1})

    def test_bad_role_mapping_is_visible_and_does_not_start_worker(self):
        self.load()
        w=self.window
        self.method_item('linear').setCheckState(0, Qt.CheckState.Checked)
        for i,spec in enumerate(w._dataset_panel._column_specs):
            if spec.role==ColumnRole.TARGET:
                role=w._dataset_panel._columns_table.cellWidget(i,1)
                role.setCurrentIndex(role.findData('feature'))
        w._on_run()
        self.assertIsNone(w._worker)
        self.assertIn('Target',w._dataset_panel._dataset_info.text())

    def test_ls_trf_are_selectable_and_run_with_dynamic_features(self):
        self.load()
        w = self.window
        wanted = {'log_linear_ls', 'log_linear_trf'}
        found = set()
        for i in range(w._method_tree.topLevelItemCount()):
            group = w._method_tree.topLevelItem(i)
            for j in range(group.childCount()):
                item = group.child(j)
                name = item.data(0, Qt.ItemDataRole.UserRole)
                item.setCheckState(0, Qt.CheckState.Checked if name in wanted else Qt.CheckState.Unchecked)
                if name in wanted:
                    found.add(name)
        self.assertEqual(found, wanted)
        w._synth_cb.setChecked(False)
        errors = []
        with patch('entrypoint.gui.QMessageBox.warning', side_effect=lambda *args: errors.append(args)):
            w._on_run()
            loop = QEventLoop()
            w._worker.finished.connect(loop.quit)
            QTimer.singleShot(15000, loop.quit)
            loop.exec()
            self.assertTrue(w._worker.wait(1000))
            self.app.processEvents()
        self.assertFalse(errors, errors)
        self.assertIsNotNone(w._last_result)
        for target, formulas in w._last_result.formulas.items():
            self.assertEqual(set(formulas), wanted)
            self.assertTrue(all('x8^' in f for f in formulas.values()))

    def test_material_is_automatically_selected_and_survives_mapping(self):
        material_frame('Тип материала двутавра').to_csv(self.path,index=False)
        self.load()
        w=self.window
        self.assertIn('is_steel',w._fix_rows)
        row=next(i for i,s in enumerate(w._dataset_panel._column_specs) if s.name=='is_steel')
        table=w._dataset_panel._columns_table
        self.assertEqual(table.cellWidget(row,1).currentData(),'feature')
        self.assertTrue(table.cellWidget(row,2).isChecked())
        self.assertEqual(table.cellWidget(row,5).currentText(),'binary')
        self.assertFalse(table.cellWidget(row,6).isChecked())
        self.assertTrue(w._dataset_panel.apply_roles())
        with patch('entrypoint.dataset_panel.QFileDialog.getSaveFileName',
                   return_value=(str(mapping_path(self.path)), 'json')):
            w._dataset_panel.save_mapping()
        loaded=load_dataset_bundle(self.path)
        self.assertIn('is_steel',loaded.schema.features)
        self.assertEqual(loaded.schema.spec('is_steel').derived_from,'Тип материала двутавра')

    def test_named_copies_keep_original_and_can_be_selected(self):
        original = save_schema(dynamic().schema, self.path)
        original_bytes = original.read_bytes()
        self.load()
        panel = self.window._dataset_panel
        panel._mapping_name.setText('Моя разметка / вариант: α')
        row = next(i for i, s in enumerate(panel._column_specs) if s.name == 'x8')
        role = panel._columns_table.cellWidget(row, 1)
        role.setCurrentIndex(role.findData('ignore'))
        saved = []

        def choose_path(_parent, _title, proposed, _filter):
            saved.append(Path(proposed))
            return proposed, 'json'

        with patch('entrypoint.dataset_panel.QFileDialog.getSaveFileName', side_effect=choose_path):
            panel.save_mapping()
            panel.save_mapping()
        self.assertNotEqual(saved[0], saved[1])
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assertEqual(read_schema(saved[0]).name, 'Моя разметка / вариант: α')
        self.assertNotIn('x8', read_schema(saved[0]).features)
        panel._mapping_combo.setCurrentIndex(panel._mapping_combo.findData(str(original)))
        self.assertIn('x8', self.window._fix_rows)
        panel._mapping_combo.setCurrentIndex(panel._mapping_combo.findData(str(saved[0])))
        self.assertNotIn('x8', self.window._fix_rows)
        self.assertEqual(panel._mapping_name.text(), 'Моя разметка / вариант: α')
        panel._mapping_combo.setCurrentIndex(0)
        self.assertIsNone(panel._mapping_path)
        self.assertIn('x8', self.window._fix_rows)

    def test_external_mapping_matches_renamed_columns_and_ignores_new_ones(self):
        ds = dynamic()
        ds.schema.name = 'Исходная разметка'
        ds.schema.spec('x1').unit = 'mm'
        ds.schema.spec('x1').synthetic_noise = 0.2
        external = Path(self.tmp.name) / 'external.json'
        save_schema(ds.schema, external)
        ds.frame.rename(columns={'x1': 'height'}, inplace=True)
        ds.frame['extra'] = ds.frame.x2 * 2
        ds.frame.to_csv(self.path, index=False)
        self.load()
        panel = self.window._dataset_panel
        original_bytes = external.read_bytes()

        def match(dialog):
            self.assertNotIn('height', dialog._matches)
            combo = dialog._matches['x1']
            combo.setCurrentIndex(combo.findData('height'))
            return QDialog.DialogCode.Accepted

        with patch.object(MappingColumnsDialog, 'exec', match):
            panel.load_mapping(str(external))
        self.assertEqual(self.window._dataset.schema.features[0], 'height')
        self.assertEqual(self.window._dataset.schema.spec('height').unit, 'mm')
        self.assertEqual(self.window._dataset.schema.spec('height').synthetic_noise, 0.2)
        self.assertEqual(self.window._dataset.schema.spec('extra').role, ColumnRole.IGNORE)
        self.assertEqual(self.window._dataset.schema.name, 'Исходная разметка')
        self.assertEqual(self.window._dataset.source, self.path.resolve())
        self.assertEqual(external.read_bytes(), original_bytes)
        self.assertTrue(panel.apply_roles())

    def test_cancel_and_invalid_mapping_preserve_current_dataset(self):
        self.load()
        panel = self.window._dataset_panel
        dataset = panel.dataset
        invalid = Path(self.tmp.name) / 'bad.json'
        invalid.write_text('[]')
        with patch('entrypoint.dataset_panel.QMessageBox.warning') as warning:
            panel.load_mapping(str(invalid))
        warning.assert_called_once()
        self.assertIs(panel.dataset, dataset)
        ds = dynamic()
        ds.schema.spec('x1').name = 'different'
        external = save_schema(ds.schema, Path(self.tmp.name) / 'foreign.json')
        with patch.object(MappingColumnsDialog, 'exec', return_value=QDialog.DialogCode.Rejected):
            panel.load_mapping(str(external))
        self.assertIs(panel.dataset, dataset)
        with patch('entrypoint.dataset_panel.QFileDialog.getSaveFileName', return_value=('', '')):
            panel.save_mapping()
        self.assertFalse(mapping_path(self.path).exists())

    def test_broken_sidecar_does_not_prevent_opening_dataset(self):
        mapping_path(self.path).write_text('{broken', encoding='utf-8')
        with patch('entrypoint.dataset_panel.QMessageBox.warning') as warning:
            self.load()
        warning.assert_called_once()
        self.assertEqual(self.window._dataset.source, self.path.resolve())
        self.assertIsNone(self.window._dataset_panel._mapping_path)
        self.assertTrue(self.window._dataset_panel.apply_roles())

    def test_matching_dialog_rejects_duplicate_assignments(self):
        ds = dynamic()
        dialog = MappingColumnsDialog(ds.schema, ds.frame.columns, self.window)
        combo = dialog._matches['x1']
        combo.setCurrentIndex(combo.findData('x2'))
        with patch('entrypoint.dataset_panel.QMessageBox.warning') as warning:
            dialog.accept()
        warning.assert_called_once()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        combo.setCurrentIndex(combo.findData('x1'))
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_mapping_button_is_fully_visible_in_default_sidebar(self):
        self.window.show()
        self.app.processEvents()
        panel = self.window._dataset_panel
        scroll = panel.parentWidget()
        while not isinstance(scroll, QScrollArea):
            scroll = scroll.parentWidget()
        viewport = scroll.viewport()
        for control in (panel._load_mapping_btn, panel._mapping_combo):
            position = control.mapTo(viewport, control.rect().topLeft())
            self.assertTrue(viewport.rect().contains(control.rect().translated(position)))

    def test_mapping_button_opens_file_picker_and_loads_selected_mapping(self):
        self.load()
        panel = self.window._dataset_panel
        schema = dynamic().schema
        schema.name = 'Готовая разметка'
        schema.spec('x8').role = ColumnRole.IGNORE
        path = save_schema(schema, Path(self.tmp.name) / 'ready.json')
        with patch('entrypoint.dataset_panel.QFileDialog.getOpenFileName',
                   return_value=(str(path), 'json')) as picker:
            QTest.mouseClick(panel._load_mapping_btn, Qt.MouseButton.LeftButton)
        picker.assert_called_once()
        self.assertEqual(panel._mapping_path, path.resolve())
        self.assertEqual(panel._mapping_name.text(), 'Готовая разметка')
        self.assertNotIn('x8', self.window._fix_rows)

    def test_direct_launch_prefers_checkout_over_another_installed_copy(self):
        root = Path(__file__).resolve().parents[1]
        foreign = Path(self.tmp.name) / 'foreign'
        for package in ('entrypoint', 'core', 'config'):
            directory = foreign / package
            directory.mkdir(parents=True)
            (directory / '__init__.py').write_text('raise RuntimeError("Wrong project copy")')
        code = '''
import pathlib, runpy, sys
root = pathlib.Path(sys.argv[1])
foreign = sys.argv[2]
sys.path = [str(root / 'entrypoint'), foreign] + [
    value for value in sys.path if value and pathlib.Path(value).resolve() != root]
runpy.run_path(str(root / 'entrypoint' / 'gui.py'), run_name='direct_launch_check')
import entrypoint.dataset_panel, core.data.loader, config.settings
for module in (entrypoint.dataset_panel, core.data.loader, config.settings):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root), module.__file__
'''
        result = subprocess.run([sys.executable, '-c', code, str(root), str(foreign)],
                                cwd=self.tmp.name, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__=='__main__': unittest.main()
