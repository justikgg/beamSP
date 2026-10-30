"""Portable named mappings, explicit column matching and safe persistence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core.data.loader import load_dataset_bundle
from core.data.schema import (ColumnRole, ColumnSpec, Dataset, DatasetError, DatasetSchema,
                              mapping_path, read_schema, save_schema)


class SchemaMappingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'data.csv'
        self.frame = pd.DataFrame({'profile_id': ['a', 'b'], 'x': [1, 2], 'target': [3, 4]})
        self.frame.to_csv(self.path, index=False)
        self.schema = DatasetSchema.from_mapping(
            self.frame.columns, ['x'], {'result': 'target'}, 'profile_id')

    def test_named_mapping_is_optional_and_backward_compatible(self):
        old = self.schema.to_dict()
        self.assertNotIn('name', old)
        self.assertEqual(DatasetSchema.from_dict(old).name, '')
        save_schema(self.schema, self.path)
        original = mapping_path(self.path).read_bytes()
        self.schema.name = 'Произвольное название / α'
        named = save_schema(self.schema, self.path.parent / 'alternative.json')
        self.assertEqual(read_schema(named).to_dict(), self.schema.to_dict())
        self.assertEqual(mapping_path(self.path).read_bytes(), original)
        self.assertEqual(load_dataset_bundle(self.path).schema.name, '')
        self.assertEqual(load_dataset_bundle(self.path, schema=read_schema(named)).schema.name,
                         self.schema.name)

    def test_sidecar_can_be_bypassed_without_modifying_it(self):
        sidecar = mapping_path(self.path)
        sidecar.write_text('broken', encoding='utf-8')
        loaded = load_dataset_bundle(self.path, use_sidecar=False)
        self.assertEqual(loaded.schema.features, ['x'])
        self.assertEqual(sidecar.read_text(encoding='utf-8'), 'broken')

    def test_remapping_preserves_settings_order_keys_and_profile_columns(self):
        self.schema.name = 'Original'
        self.schema.grouping_columns = ['x']
        self.schema.spec('x').unit = 'mm'
        self.schema.spec('x').synthetic_noise = 0.1
        self.schema.spec('x').minimum = 0
        self.schema.spec('x').formula_allowed = False
        self.schema.spec('x').monotonic = 1
        original = deepcopy(self.schema.to_dict())
        frame = self.frame.rename(columns={'x': 'height', 'target': 'response', 'profile_id': 'batch'})
        frame['extra'] = [4, 5]
        matched = self.schema.remap(frame.columns,
                                   {'x': 'height', 'target': 'response', 'profile_id': 'batch'})
        checked = Dataset(frame, matched).validate()
        self.assertEqual(checked.schema.features, ['height'])
        self.assertEqual(checked.schema.targets, {'result': 'response'})
        self.assertEqual(checked.schema.group_column, 'batch')
        self.assertEqual(checked.schema.grouping_columns, ['height'])
        self.assertEqual(matched.spec('height').unit, 'mm')
        self.assertEqual(matched.spec('height').synthetic_noise, 0.1)
        self.assertEqual(matched.spec('height').minimum, 0)
        self.assertFalse(matched.spec('height').formula_allowed)
        self.assertEqual(matched.spec('height').monotonic, 1)
        self.assertEqual(matched.spec('extra').role, ColumnRole.IGNORE)
        self.assertEqual(matched.name, 'Original')
        self.assertEqual(self.schema.to_dict(), original)

    def test_missing_and_duplicate_matches_are_rejected(self):
        for mapping in ({'x': 'missing'}, {'x': 'target'}):
            with self.subTest(mapping=mapping), self.assertRaises(DatasetError):
                self.schema.remap(self.frame.columns, mapping)

    def test_skipped_columns_and_profile_keys_are_removed(self):
        self.schema.columns.append(ColumnSpec('old_metadata', ColumnRole.METADATA))
        self.schema.grouping_columns = ['old_metadata']
        matched = self.schema.remap(self.frame.columns, {'old_metadata': None})
        self.assertNotIn('old_metadata', [c.name for c in matched.columns])
        self.assertEqual(matched.grouping_columns, [])
        Dataset(self.frame, matched).validate()

    def test_derived_provenance_is_remapped_and_cannot_be_silently_dropped(self):
        self.schema.spec('x').derived_from = 'profile_id'
        matched = self.schema.remap(['batch', 'height', 'target'],
                                   {'x': 'height', 'profile_id': 'batch'})
        self.assertEqual(matched.spec('height').derived_from, 'batch')
        with self.assertRaisesRegex(DatasetError, 'источник'):
            self.schema.remap(self.frame.columns, {'profile_id': None})
        frame = self.frame.rename(columns={'profile_id': 'batch', 'x': 'height'})
        with self.assertRaisesRegex(DatasetError, 'производные признаки'):
            Dataset(frame, matched).validate()

    def test_malformed_mapping_reports_dataset_error(self):
        good = self.schema.to_dict()
        cases = [[], None, {'columns': 'wrong'}, {**good, 'name': 3},
                 {**good, 'columns': [good['columns'][0]] * 2},
                 {**good, 'columns': [{**good['columns'][0], 'unit': {}}]},
                 {**good, 'columns': [{**good['columns'][0], 'enabled': []}]},
                 {**good, 'columns': [{**good['columns'][0], 'dtype': 'unknown'}]},
                 {**good, 'columns': [{**good['columns'][0], 'synthetic_noise': -1}]},
                 {**good, 'columns': [{**good['columns'][0], 'synthetic_noise': 'bad'}]}]
        for payload in cases:
            with self.subTest(payload=payload):
                path = self.path.parent / 'bad.json'
                path.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaises(DatasetError):
                    read_schema(path)

    def test_failed_atomic_save_keeps_existing_mapping(self):
        path = save_schema(self.schema, self.path)
        original = path.read_bytes()
        self.schema.name = 'New name'
        with patch('core.data.schema.Path.replace', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                save_schema(self.schema, path)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(path.parent.glob('*.tmp')))

    def test_mapping_name_survives_model_selection_and_result_metadata(self):
        from core.evaluation.runner import run
        self.schema.name = 'Мой вариант'
        result = run(['linear'], dataset=Dataset(self.frame, self.schema),
                     synth_cfg={'enabled': False})
        self.assertEqual(result.metadata['schema']['name'], 'Мой вариант')


if __name__ == '__main__':
    unittest.main()
