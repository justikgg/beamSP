"""Offscreen integration of dataset editing, fixed powers and the Run button."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('MPLCONFIGDIR','/tmp/beamsp-mpl-tests')
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtWidgets import QApplication
    from entrypoint.gui import MainWindow
    QT_AVAILABLE=True
except ImportError:
    QT_AVAILABLE=False

from core.data.loader import load_dataset_bundle
from core.data.schema import ColumnRole, save_schema
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

    def test_load_roles_targets_units_noise_and_worker_run(self):
        self.load()
        w=self.window
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
        w._dataset_panel.save_mapping()
        loaded=load_dataset_bundle(self.path)
        self.assertIn('is_steel',loaded.schema.features)
        self.assertEqual(loaded.schema.spec('is_steel').derived_from,'Тип материала двутавра')


if __name__=='__main__': unittest.main()
