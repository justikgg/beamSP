"""Regression, dynamic schema, model capability and leakage coverage."""
from copy import deepcopy
from functools import partial
import json
import os
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import settings as cfg
from core.data.loader import load_dataset, load_dataset_bundle
from core.data.schema import (ColumnRole, ColumnSpec, Dataset, DatasetError, DatasetSchema,
                              FeatureSpec, TargetSpec, group_ids, save_schema, validate_dataset)
from core.evaluation.runner import prepare, run
from core.evaluation.validation import leave_one_group_out
from core.models.base import BaseModel
from core.models.baseline.linear import LinearModel
from core.models.registry import available_models, get_model, register_model
from core.models.bioinspired.formula_search import FormulaSearchModel
from core.models.bioinspired.optimizers import ga, de, pso, cmaes
from core.formula.power_law import PowerLawForm
from core.reporting import export_results

ROOT = Path(__file__).resolve().parents[1]


def dynamic(n_features=8):
    rng = np.random.default_rng(27)
    X = rng.uniform(1, 4, (32, n_features))
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(1, n_features + 1)])
    df.insert(0, 'profile_id', np.repeat(['north', 'south', 'east', 'west'], 8))
    df['target_a'] = 10 + 2 * X[:, 0] + 3 * X[:, 1]
    df['target_b'] = 20 + X[:, 2] * X[:, 3] if n_features >= 4 else 20 + X[:, 0]
    schema = DatasetSchema.from_mapping(df.columns, list(df.columns[1:n_features+1]),
                                        ['target_a', 'target_b'], 'profile_id')
    return Dataset(df, schema)


class LegacyTests(unittest.TestCase):
    def test_legacy_export_with_target_subscripts(self):
        result = run(['linear'], data_path=cfg.DATA_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            saved = export_results(result, tmp, stamp='legacy')
            self.assertIn('formulas_legacy.png', saved)
            self.assertIn('metrics_SP63_legacy.csv', saved)
            self.assertIn('metrics_RUK78_legacy.csv', saved)
            self.assertTrue(all((Path(tmp)/name).stat().st_size > 0 for name in saved))

    def test_legacy_loader_and_mappings(self):
        ds = load_dataset_bundle(cfg.DATA_PATH)
        self.assertEqual(ds.frame.shape, (18, 10))
        self.assertEqual(ds.schema.features, ['a_h0', 'is_steel', 'H', 's', 'R', 'E'])
        self.assertEqual(ds.schema.targets, {'SP63': 'Qдв_СП63', 'RUK78': 'Qдв_рук78'})
        self.assertEqual(len(pd.unique(group_ids(ds.frame, ds.schema))), 6)
        np.testing.assert_allclose(ds.frame[ds.schema.features].iloc[0], [1.6, 0, 150, 6, 30, 60000])
        np.testing.assert_allclose(ds.frame[list(ds.schema.targets.values())].iloc[0], [21.6, 19.6])
        self.assertIsInstance(load_dataset(cfg.DATA_PATH), pd.DataFrame)

    def test_deterministic_v0171_results(self):
        reference = json.loads((ROOT/'tests/fixtures/legacy_v0171.json').read_text())
        for enabled in (False, True):
            with self.subTest(synthesis=enabled):
                result = run(['linear', 'ridge', 'lasso', 'elasticnet', 'gbr', 'svr'],
                             data_path=cfg.DATA_PATH, synth_cfg={**cfg.SYNTH, 'enabled': enabled})
                self.assertEqual(result.metadata['real_rows'], 18)
                self.assertEqual(result.metadata['synthetic_rows'], 270 if enabled else 0)
                for target, old in reference[str(enabled)].items():
                    for model, metrics in old['metrics'].items():
                        for name, value in metrics.items():
                            np.testing.assert_allclose(result.metrics[target].loc[model, name], value,
                                                       rtol=1e-9, atol=1e-9)
                        np.testing.assert_allclose(result.predictions[target][model]['y_pred'],
                                                   old['predictions'][model], rtol=1e-9, atol=1e-9)


class DynamicTests(unittest.TestCase):
    def test_csv_and_xlsx_eight_features_logo_metrics(self):
        ds = dynamic()
        with tempfile.TemporaryDirectory() as tmp:
            for extension in ('csv', 'xlsx'):
                with self.subTest(extension=extension):
                    path = Path(tmp)/f'data.{extension}'
                    if extension == 'csv':
                        ds.frame.to_csv(path, index=False)
                    else:
                        ds.frame.to_excel(path, index=False)
                    loaded = load_dataset_bundle(path, schema=ds.schema)
                    self.assertEqual(loaded.schema.features, [f'x{i}' for i in range(1,9)])
                    result = run(['linear', 'ridge', 'gbr'], dataset=loaded)
                    self.assertEqual(set(result.metrics), {'target_a','target_b'})
                    for target in result.metrics:
                        self.assertEqual(len(result.fold_rmse[target]['linear']), 4)
                        self.assertTrue(np.isfinite(result.metrics[target][['RMSE','MAE','R2']]).all().all())
                        self.assertEqual(len(result.predictions[target]['linear']['y_pred']), 32)
                    self.assertGreater(result.metrics['target_a'].loc['linear','R2'], 0.999999)
                    self.assertIn('x1', result.formulas['target_a']['linear'])
                    self.assertNotIn('H', result.formulas['target_a']['linear'])

    def test_ninth_feature_no_model_changes(self):
        for n in (2, 9, 15, 30):
            with self.subTest(features=n):
                result = run(['linear','ridge','svr'], dataset=dynamic(n))
                self.assertEqual(len(result.metadata['feature_columns']), n)
                self.assertTrue(np.isfinite(result.metrics['target_a']['RMSE']).all())

    def test_schema_sidecar_roundtrip_and_feature_order(self):
        ds = dynamic(9)
        ds.schema.columns.reverse()
        ds.schema.spec('x7').unit = 'mm'
        ds.schema.spec('x7').synthetic_noise = 0.2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.csv'
            ds.frame.to_csv(path,index=False)
            save_schema(ds.schema,path)
            loaded = load_dataset_bundle(path)
            self.assertEqual(loaded.schema.to_dict(), ds.schema.to_dict())
            self.assertEqual(loaded.schema.features, [f'x{i}' for i in range(9,0,-1)])

    def test_explicit_groups_and_fallback_without_geometry_names(self):
        ds = dynamic()
        expected = ds.frame.profile_id.to_numpy()
        ds.schema.grouping_columns = ['x1']
        np.testing.assert_array_equal(group_ids(ds.frame,ds.schema),expected)
        ds.schema.spec('profile_id').role = ColumnRole.METADATA
        ds.schema.grouping_columns = ['profile_id']
        ds.validate()
        self.assertEqual(len(pd.unique(group_ids(ds.frame,ds.schema))),4)
        run(['linear'],dataset=ds)

    def test_fallback_keys_do_not_collide_on_delimiters(self):
        frame = pd.DataFrame({'a':['a|b','a'], 'b':['c','b|c']})
        schema = DatasetSchema(grouping_columns=['a','b'])
        np.testing.assert_array_equal(group_ids(frame,schema),[0,1])

    def test_mixed_group_ids(self):
        ds=dynamic()
        ds.frame['profile_id']=np.repeat(np.array([1,'one',2,'two'],dtype=object),8)
        result=run(['linear'],dataset=ds)
        self.assertEqual(result.metadata['groups'],4)

    def test_export_provenance_and_arbitrary_names(self):
        ds=dynamic()
        ds.frame.rename(columns={'target_a':'target/a','x1':'feature_with_two_underscores'},inplace=True)
        ds.schema.spec('target_a').name='target/a'
        ds.schema.spec('target/a').target_key='target/a'
        ds.schema.spec('x1').name='feature_with_two_underscores'
        result=run(['linear'],dataset=ds,seed=21)
        with tempfile.TemporaryDirectory() as tmp:
            saved=export_results(result,Path(tmp),stamp='test')
            self.assertTrue(all((Path(tmp)/name).is_file() for name in saved))
            meta=json.loads((Path(tmp)/'metadata_test.json').read_text())
            self.assertEqual(meta['seed'],21)
            self.assertEqual(meta['real_rows'],32)
            self.assertEqual(meta['groups'],4)
            self.assertEqual(meta['per_target']['target/a']['target'],'target/a')
            self.assertEqual(meta['schema'],ds.schema.to_dict())
            self.assertIn('feature_with_two_underscores',meta['formulas']['target/a']['linear'])


class ValidationTests(unittest.TestCase):
    def test_target_feature_overlap(self):
        ds=dynamic()
        with self.assertRaisesRegex(DatasetError,'Пересечение ролей'):
            DatasetSchema.from_mapping(ds.frame.columns,['x1','target_a'],['target_a'],'profile_id')
        with self.assertRaisesRegex(DatasetError,'Пересечение ролей'):
            run(['linear'],feature_cols=['x1','target_a'],dataset=ds)
        ds.schema.columns.append(FeatureSpec('target_a'))
        with self.assertRaisesRegex(DatasetError,'Пересечение ролей'):
            ds.validate()

    def test_group_feature_overlap(self):
        ds=dynamic()
        with self.assertRaisesRegex(DatasetError,'Пересечение ролей'):
            DatasetSchema.from_mapping(ds.frame.columns,['x1','profile_id'],['target_a'],'profile_id')

    def test_invalid_numeric(self):
        ds=dynamic()
        ds.frame['x5']=ds.frame.x5.astype(object)
        ds.frame.loc[0,'x5']='invalid'
        with self.assertRaisesRegex(DatasetError,'x5.*нечисловые'):
            ds.validate()

    def test_missing_nan_inf_empty_groups_and_columns(self):
        cases=[('nan','пропущенных'),('inf','NaN / inf'),('empty','пуст'),
               ('missing','отсутствуют'),('one_group','две группы'),
               ('no_group','выберите Group'),('group_nan','пропущенных'),
               ('target_numeric','target_a.*нечисловые'),('target_grouping','Target нельзя')]
        for case, message in cases:
            with self.subTest(case=case):
                ds=dynamic()
                if case=='nan': ds.frame.loc[:2,'x4']=np.nan
                elif case=='inf': ds.frame.loc[0,'x3']=np.inf
                elif case=='empty': ds.frame=ds.frame.iloc[:0]
                elif case=='missing': ds.frame.drop(columns='x6',inplace=True)
                elif case=='one_group': ds.frame['profile_id']='only'
                elif case=='no_group': ds.schema.spec('profile_id').role=ColumnRole.IGNORE
                elif case=='group_nan': ds.frame.loc[0,'profile_id']=None
                elif case=='target_numeric': ds.frame['target_a']='bad'
                elif case=='target_grouping': ds.schema.grouping_columns=['target_a']
                with self.assertRaisesRegex(DatasetError,message): ds.validate()

    def test_duplicate_header_before_pandas_mangling(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ext in ('csv','xlsx'):
                path=Path(tmp)/f'duplicate.{ext}'
                frame=pd.DataFrame([[1,2,3]],columns=['x1','x1','target_a'])
                if ext=='csv': frame.to_csv(path,index=False)
                else: frame.to_excel(path,index=False)
                with self.assertRaisesRegex(DatasetError,'Повторяющиеся'):
                    load_dataset_bundle(path)

    def test_runtime_schema_is_authoritative(self):
        ds=dynamic()
        with self.assertRaisesRegex(DatasetError,'не совпадают'):
            run(['linear'],feature_cols=['x1'],dataset=ds,schema=ds.schema)


class SyntheticAndLeakageTests(unittest.TestCase):
    def test_noise_only_selected_numeric_features_and_parent_groups(self):
        ds=dynamic()
        ds.frame['comment']='metadata'
        ds.schema.columns.append(ColumnSpec('comment',ColumnRole.METADATA,dtype='categorical'))
        config={'enabled':True,'samples_per_profile':2,'noise':{'x2':0.1}}
        df,groups=prepare(ds.frame,synth_cfg=config,schema=ds.schema,seed=8)
        self.assertEqual(len(df),96)
        source=ds.frame.iloc[np.repeat(np.arange(32),2)].reset_index(drop=True)
        synth=df.iloc[32:].reset_index(drop=True)
        for column in ds.frame.columns.difference(['x2']):
            np.testing.assert_array_equal(synth[column],source[column])
        self.assertTrue((np.abs(synth.x2-source.x2)<=0.1).all())
        self.assertTrue((synth.x2!=source.x2).any())
        np.testing.assert_array_equal(groups[32:],source.profile_id)
        again,_=prepare(ds.frame,synth_cfg=config,schema=ds.schema,seed=8)
        pd.testing.assert_frame_equal(df,again)
        for column in ('target_a','profile_id','comment'):
            with self.assertRaisesRegex(DatasetError,'Нельзя добавлять шум'):
                prepare(ds.frame,synth_cfg={**config,'noise':{column:0.1}},schema=ds.schema)
        ds.schema.spec('x2').dtype='binary'
        with self.assertRaisesRegex(DatasetError,'категориальному/бинарному'):
            prepare(ds.frame,synth_cfg=config,schema=ds.schema)

    def test_unconfigured_noise_changes_nothing(self):
        ds=dynamic()
        df,_=prepare(ds.frame,synth_cfg={'enabled':True,'samples_per_profile':1},schema=ds.schema)
        np.testing.assert_array_equal(df[ds.schema.features].iloc[32:],ds.frame[ds.schema.features])

    def test_logo_excludes_held_out_real_and_synthetic_and_scaler_fit(self):
        ds=dynamic()
        ds.frame['x1']=np.repeat([100.,200.,300.,400.],8)
        df,groups=prepare(ds.frame,synth_cfg={'enabled':True,'samples_per_profile':2,'noise':{'x2':0.1}},schema=ds.schema)
        X=df[ds.schema.features].to_numpy(float)
        y=df.target_a.to_numpy(float)
        records=[]
        class AuditModel(LinearModel):
            def fit(self,X,y):
                self.train=X.copy()
                super().fit(X,y)
                np.testing.assert_allclose(self._m.steps[0][1].mean_,X.mean(axis=0))
                return self
            def predict(self,X):
                train_ids=set(self.train[:,0])
                test_ids=set(X[:,0])
                if train_ids & test_ids:
                    raise AssertionError('Held-out group leaked into train')
                records.append((self.train.copy(),X.copy()))
                return super().predict(X)
        synth=np.asarray(df.attrs['beamsp_synthetic'],bool)
        preds=leave_one_group_out(AuditModel,X,y,groups,synth)
        self.assertEqual(len(records),4)
        self.assertTrue(np.isfinite(preds[~synth]).all())
        self.assertTrue(np.isnan(preds[synth]).all())
        for train,test in records:
            self.assertEqual(len(train),72)
            self.assertEqual(len(test),8)
            self.assertEqual(len(set(train[:,0])),3)

    def test_fallback_geometry_noise_preserves_parent_group(self):
        ds=dynamic()
        ds.frame['x1']=np.repeat([1.,2.,3.,4.],8)
        ds.schema.spec('profile_id').role=ColumnRole.METADATA
        ds.schema.grouping_columns=['x1']
        df,groups=prepare(ds.frame,schema=ds.schema,synth_cfg={'enabled':True,'samples_per_profile':1,'noise':{'x1':0.2}})
        np.testing.assert_array_equal(groups[:32],groups[32:])
        self.assertTrue((df.x1.iloc[:32].to_numpy()!=df.x1.iloc[32:].to_numpy()).any())

    def test_metadata_named_is_synth_is_preserved(self):
        ds=dynamic()
        ds.frame['is_synth']='original annotation'
        ds.schema.columns.append(ColumnSpec('is_synth',ColumnRole.METADATA,dtype='categorical'))
        result=run(['linear'],dataset=ds,synth_cfg={'enabled':True,'samples_per_profile':1})
        self.assertEqual(result.metadata['real_rows'],32)
        self.assertEqual(result.metadata['synthetic_rows'],32)
        self.assertTrue((ds.frame.is_synth=='original annotation').all())


class ModelTests(unittest.TestCase):
    def test_all_registered_models_with_nine_runtime_features(self):
        from core.models.classic_ml.gpr import GPRModel
        from core.models.classic_ml.knn import KNNModel
        from core.models.neural.pinn import PINNModel
        from core.models.experimental.rl import ActorCriticRegressor
        from core.models.sym_regression.symbolic_regression import SymbolicRegressionModel
        ds=dynamic(9).validate()
        X=ds.frame[ds.schema.features].to_numpy(float)
        y=ds.frame.target_a.to_numpy(float)
        quick={'gpr':lambda:GPRModel(n_restarts_optimizer=0),
               'knn':lambda:KNNModel(n_neighbors=3),
               'pinn':lambda:PINNModel(hidden=(4,),epochs=2),
               'rl_ac':lambda:ActorCriticRegressor(hidden=(4,),epochs=2),
               'symreg':lambda:SymbolicRegressionModel(population_size=30,generations=2)}
        optimizers={'ga':partial(ga,pop_size=10,generations=2),
                    'de':partial(de,maxiter=2,popsize=3,polish=False),
                    'pso':partial(pso,n_particles=10,iters=2),
                    'cmaes':partial(cmaes,maxiter=2)}
        for name in available_models():
            with self.subTest(model=name):
                model=quick[name]() if name in quick else get_model(name)
                if name.startswith('bio_'): model._optimizer=optimizers[name[4:]]
                model.set_seed(42).configure_features(ds.schema.features,ds.schema)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model.fit(X,y)
                pred=model.predict(X[:4])
                self.assertEqual(pred.shape,(4,))
                self.assertTrue(np.isfinite(pred).all())
                if hasattr(model,'formula'):
                    formula=model.formula(ds.schema.features)
                    self.assertNotIn('a_h0',formula)
                    self.assertNotIn('is_steel',formula)

    def test_power_law_positive_capability_and_explicit_exclusion(self):
        ds=dynamic()
        ds.frame.loc[0,'x5']=0
        with self.assertRaisesRegex(DatasetError,'bio_de.*x5.*неположительные'):
            run(['bio_de'],dataset=ds)
        ds.schema.spec('x5').formula_allowed=False
        model=FormulaSearchModel(partial(de,maxiter=2,popsize=3,polish=False))
        result=run(['test_power'],dataset=ds,model_factories={'test_power':lambda:deepcopy(model)})
        self.assertNotIn('x5^',result.formulas['target_a']['test_power'])
        ds.frame.loc[0,'target_a']=0
        with self.assertRaisesRegex(DatasetError,'положительные.*target'):
            run(['test_power'],dataset=ds,model_factories={'test_power':lambda:deepcopy(model)})

    def test_named_fixed_exponents_follow_runtime_order(self):
        ds=dynamic()
        ds.schema.columns.reverse()
        names=ds.schema.features
        model=FormulaSearchModel(partial(de,maxiter=2,popsize=3,polish=False),
                                 form=PowerLawForm({'x1':1.,'x8':0.}))
        model.configure_features(names,ds.schema).fit(ds.frame[names].to_numpy(float),ds.frame.target_a.to_numpy(float))
        self.assertIn('x1^1.000',model.formula(names))
        self.assertIn('x8^0.000',model.formula(names))
        missing=FormulaSearchModel(de,form=PowerLawForm({'missing':1.}))
        with self.assertRaisesRegex(DatasetError,'missing.*отсутствует'):
            missing.configure_features(names,ds.schema).fit(ds.frame[names].to_numpy(float),ds.frame.target_a.to_numpy(float))

    def test_knn_and_pinn_restrictions(self):
        ds=dynamic()
        with self.assertRaisesRegex(DatasetError,'knn.*n_neighbors=55'):
            run(['knn'],dataset=ds)
        ds.frame['target_a']=-1
        with self.assertRaisesRegex(DatasetError,'pinn.*неотрицательный'):
            run(['pinn'],dataset=ds)

    def test_register_model(self):
        import core.models.registry as registry
        class ExampleLinear(LinearModel): name='example_test_linear'
        try:
            register_model(ExampleLinear)
            self.assertIn(ExampleLinear.name,available_models())
            run([ExampleLinear.name],dataset=dynamic())
            with self.assertRaisesRegex(ValueError,'уже зарегистрирована'):
                register_model(ExampleLinear)
        finally:
            registry._REGISTRY.pop(ExampleLinear.name,None)


if __name__=='__main__':
    unittest.main()
