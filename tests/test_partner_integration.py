"""Acceptance checks from the LS/TRF handoff, including the whole data path."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from config import settings as cfg
from core.data.loader import load_dataset_bundle, infer_schema
from core.data.schema import ColumnRole, DatasetError, FeatureSpec, group_ids, save_schema
from core.evaluation.runner import run
from core.models.registry import get_model
from core.reporting import export_results, model_notes
from entrypoint._common import run_method


METHODS = ['log_linear_ls', 'log_linear_trf']


def extended_legacy():
    ds = load_dataset_bundle(cfg.DATA_PATH)
    rng = np.random.default_rng(103)
    # Changes inside every existing physical profile, independently of geometry.
    ds.frame['x_extra'] = np.exp(rng.uniform(-1, 1, len(ds.frame)))
    ds.schema.columns.append(FeatureSpec('x_extra', unit='mm'))
    return ds


class PartnerIntegrationTests(unittest.TestCase):
    def test_legacy_both_targets_six_groups_and_precise_export(self):
        result = run(METHODS, data_path=cfg.DATA_PATH, synth_cfg=cfg.SYNTH)
        self.assertEqual(result.metadata['groups'], 6)
        self.assertEqual(result.metadata['real_rows'], 18)
        self.assertEqual(result.metadata['synthetic_rows'], 270)
        self.assertIn('scipy', result.metadata['environment'])
        self.assertIn('created_at', result.metadata)
        with tempfile.TemporaryDirectory() as tmp:
            export_results(result, tmp, stamp='partner')
            meta = json.loads((Path(tmp)/'metadata_partner.json').read_text())
        for target in ('SP63', 'RUK78'):
            for method in METHODS:
                detail = meta['per_target'][target]['models'][method]
                self.assertEqual(detail['feature_columns'], result.metadata['feature_columns'])
                self.assertNotIn('is_steel', detail['used_feature_columns'])
                self.assertEqual(detail['excluded_features'][0]['name'], 'is_steel')
                self.assertEqual(len(detail['folds']), 6)
                self.assertEqual(len(result.predictions[target][method]['y_true']), 18)
                if method == 'log_linear_trf':
                    diag = detail['diagnostics']
                    self.assertLessEqual(diag['final_sse'], diag['initial_sse'] + 1e-8)
                # Fold coefficients reproduce actual held-out predictions.
                ds = load_dataset_bundle(cfg.DATA_PATH)
                groups = group_ids(ds.frame, ds.schema)
                restored = np.empty(len(ds.frame))
                for fold in detail['folds']:
                    mask = groups == fold['held_out_group']
                    names = fold['used_feature_columns']
                    powers = [fold['exponents'][n] for n in names]
                    restored[mask] = np.exp(np.clip(fold['log_a'] +
                        np.log(ds.frame.loc[mask, names].to_numpy(float)) @ powers, -50, 50))
                    self.assertEqual(fold['test_real_rows'], int(mask.sum()))
                    self.assertEqual(fold['train_rows'], (18 - int(mask.sum())) * 16)
                np.testing.assert_allclose(restored, result.predictions[target][method]['y_pred'])

    def test_xlsx_new_feature_roles_order_and_unchanged_physical_groups(self):
        ds = extended_legacy()
        original_groups = group_ids(ds.frame, ds.schema)
        # A mapping with deliberately different order from the physical worksheet.
        ds.schema.columns.reverse()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'extended.xlsx'
            ds.frame.to_excel(path, index=False)
            save_schema(ds.schema, path)
            loaded = load_dataset_bundle(path)
            np.testing.assert_array_equal(group_ids(loaded.frame, loaded.schema), original_groups)
            self.assertEqual(loaded.schema.features[0], 'x_extra')
            result = run(METHODS, dataset=loaded,
                         targets={'SP63': ds.schema.targets['SP63']},
                         synth_cfg={'enabled': True, 'samples_per_profile': 2,
                                    'noise': {'x_extra': 0.01}})
        self.assertEqual(result.metadata['feature_columns'], loaded.schema.features)
        self.assertEqual(result.metadata['groups'], 6)
        self.assertEqual(result.metadata['synthetic_rows'], 36)
        for method in METHODS:
            self.assertIn('x_extra^', result.formulas['SP63'][method])
            self.assertNotIn('Qдв_рук78^', result.formulas['SP63'][method])
            detail = result.metadata['per_target']['SP63']['models'][method]
            self.assertEqual(detail['used_feature_columns'][0], 'x_extra')

    def test_independent_artificial_control_recovers_extra_power_after_reordering(self):
        rng = np.random.default_rng(51)
        for width in (1, 3, 4, 6):
            logX = rng.uniform(-1, 1, (90, width))
            names = [f'x{i}' for i in range(width - 1)] + ['x_extra']
            powers = np.r_[np.linspace(-0.5, 0.8, width - 1), 0.4]
            y = np.exp(0.7 + logX @ powers)
            order = rng.permutation(width)
            reordered_names = [names[i] for i in order]
            for method in METHODS:
                with self.subTest(width=width, method=method):
                    model = get_model(method).configure_features(reordered_names)
                    model.fit(np.exp(logX[:, order]), y)
                    detail = model.fitted_metadata()
                    self.assertTrue(detail['diagnostics']['identifiable'])
                    self.assertAlmostEqual(detail['exponents']['x_extra'], 0.4, places=8)
                    np.testing.assert_allclose(model.predict(np.exp(logX[:, order])), y, rtol=1e-9)
                    changed = np.exp(logX[:, order])
                    changed[:, reordered_names.index('x_extra')] *= 2
                    np.testing.assert_allclose(model.predict(changed), y * 2**0.4, rtol=1e-9)

    def test_named_fixed_powers_exclusion_and_failed_refit(self):
        ds = extended_legacy()
        ds.schema.columns.reverse()
        names = ds.schema.features
        X = ds.frame[names].to_numpy(float)
        y = 2 * ds.frame.x_extra.to_numpy() ** 0.4 * ds.frame.H.to_numpy() ** 0.7
        for method in METHODS:
            with self.subTest(method=method):
                model = type(get_model(method))(fixed_exponents={'x_extra': 0.4, 'H': 0.7})
                model.configure_features(names, ds.schema).fit(X, y)
                detail = model.fitted_metadata()
                self.assertEqual(detail['exponents']['x_extra'], 0.4)
                self.assertIn('x_extra^0.400', model.formula())
                np.testing.assert_allclose(model.predict(X), y, rtol=1e-8)
                with self.assertRaises(ValueError):
                    model.fit(X, np.full(len(X), np.nan))
                with self.assertRaises(RuntimeError):
                    model.predict(X)

    def test_nonpositive_held_out_feature_fails_until_explicitly_excluded(self):
        ds = extended_legacy()
        ds.frame.loc[0, 'x_extra'] = 0
        for method in METHODS:
            with self.subTest(method=method), self.assertRaisesRegex(DatasetError, 'x_extra.*неположительные'):
                run([method], dataset=ds)
        ds.schema.spec('x_extra').formula_allowed = False
        result = run(METHODS, dataset=ds)
        for method in METHODS:
            self.assertNotIn('x_extra^', result.formulas['SP63'][method])
        self.assertTrue(any('x_extra' in note for note in model_notes(result, 'SP63')))

    def test_numeric_service_columns_and_both_targets_are_protected(self):
        frame = pd.DataFrame({name: [1, 2] for name in
                              ['profile_id', 'id', 'shifr', 'is_synth', 'sample_id',
                               'x_extra', 'Q_SP', 'Q_RUK']})
        schema = infer_schema(frame)
        self.assertEqual(schema.features, ['x_extra'])
        self.assertEqual(set(schema.targets), {'Q_SP', 'Q_RUK'})
        for name in ('id', 'shifr', 'is_synth', 'sample_id'):
            self.assertEqual(schema.spec(name).role, ColumnRole.METADATA)

    def test_cli_runtime_mapping_for_each_method(self):
        ds = extended_legacy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'extra.csv'
            ds.frame.to_csv(path, index=False)
            save_schema(ds.schema, path)
            for method in METHODS:
                result = run_method('partner acceptance', [method],
                                    argv=['--data', str(path), '--target', 'SP63', '--no-synth'])
                self.assertEqual(result.metadata['groups'], 6)
                self.assertIn('x_extra^', result.formulas['SP63'][method])


if __name__ == '__main__':
    unittest.main()
