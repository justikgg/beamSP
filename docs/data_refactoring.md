# Рефакторинг данных beamSP v0.17.1

Работа выполнена в существующей архитектуре: BaseModel.fit(X, y)/predict(X),
реестр, LOGO, формулы, метрики, старые точки входа и формат старого Excel сохранены.

## Изменённая структура и ответственность

```text
config/settings.py                     legacy aliases, прежние defaults
core/data/
  schema.py                            новые dataclass/роли, валидация, группы, JSON
  legacy.py                            layout Excel v0.17.1, профиль/единицы/defaults
  material.py                          material → is_steel по имени колонки, проверки
  loader.py                            generic CSV/XLSX и выбор legacy adapter
  synth.py                             шум по именам, сохранение группы родителя
  cli.py                               общие параметры датасета для CLI/tools
core/evaluation/
  runner.py                            runtime schema, модели, метрики, provenance
  validation.py                        LOGO, проверки folds и предсказаний
  run_result.py                        metadata результатов
core/models/
  base.py                              runtime feature names и validate_input hook
  registry.py                          публичная register_model
  classic_ml/{gbr,gpr,knn}.py           seed и ограничение размера train KNN
  neural/pinn.py                       монотонность по schema, произвольная ширина
  sym_regression/*.py                   runtime имена и тип binary
  bioinspired/formula_search.py         валидация области определения
  bioinspired/{sp63_fixed,ruk78_fixed,cold_start}.py
                                       степени по именам вместо глобальных индексов
core/formula/power_law.py               выбранные признаки, именованные степени
core/formula/latexify.py                корректные индексы имён legacy targets
core/reporting.py                      dynamic единицы/подписи, metadata JSON
core/labels.py                         подпись fixed варианта без изменения globals
entrypoint/
  dataset_panel.py                     новый независимый редактор mapping
  gui.py                               клиент Dataset, dynamic цели/степени, worker
  _common.py, interactive.py            произвольный датасет через CLI
tools/{find_alpha,importances,tune_knn,tune_model,tune_optimizer,tune_svr}.py
                                       runtime schema вместо cfg.FEATURES/TARGETS
examples/dynamic.{csv,xlsx,beamsp.json}  9 произвольных признаков, 2 цели, 4 группы
tests/test_pipeline.py                 headless data/model/LOGO/export проверки
tests/test_gui.py                      offscreen интеграция GUI
tests/test_material.py                 материал, binary Feature, mapping и защита источника
examples/material.csv                 новый CSV с материалом без числового флага
tests/fixtures/legacy_v0171.json        результаты до рефакторинга для регрессии
README.md                              новый формат, workflow, CLI, ограничения
```

GUI не содержит Excel parsing, preprocessing или ML: редактор собирает schema,
core валидирует Dataset, worker вызывает runner. Фиксированные варианты bio
передаются как фабрики отдельного запуска; изменение глобального registry или
MODEL_LABELS для такого прогона больше не требуется.

## Аудит hardcoded assumptions

| Прежняя привязка | Изменение |
|---|---|
| FEATURES/TARGETS в config как единственный источник | DatasetSchema единственный runtime источник; aliases остались только для совместимости внешних скриптов |
| r[1]/r[3]/r[4]…r[10] | Только legacy.py; generic loader получает заголовки файла |
| Все X, кроме a_h0, определяют профиль | Explicit Group либо отдельно выбранные grouping_columns; старое определение хранится только в legacy schema |
| Склейка профиля через разделитель `\|` | MultiIndex/factorize по значениям колонок, без коллизий строк |
| Синтез перезаписывает profile_id | Explicit IDs остаются неизменными; группа вычисляется до шума и копируется от родителя |
| Шум по произвольным колонкам без проверки роли | Только выбранные numeric Feature, остальные роли и binary/categorical защищены; отсутствие noise означает отсутствие изменения |
| Runner требует is_steel/H для подписей | Legacy label adapter либо универсальная подпись с group ID; подписи различают группы даже с одинаковым legacy текстом |
| FIXABLE_FEATURES и .index в GUI | Поля строятся из выбранных X, степени передаются по feature name |
| PINN индексы H/s/R из cfg.FEATURES | Monotonic по schema и runtime names; входной слой берёт X.shape[1] |
| symreg/bayes_symreg используют 6 глобальных имён | configure_features; generic defaults x0… только при прямом fit без схемы |
| Bayes специальная проверка имени is_steel | Декларативный dtype=binary в legacy schema |
| PowerLaw молча удаляет все неположительные признаки | Понятная ошибка метода с именем входа; исключение разрешается явно через formula_allowed=False |
| Фиксированные presets хранят индексы глобального X | Хранят физические имена; missing feature сообщает предметное ограничение |
| Отчёт предполагает кН и H/s/R/E | Units и описания берутся из schema; generic формулы отображаются без предположений о LaTeX-именах |
| CSV/PNG без происхождения данных | Сопутствующий metadata JSON, включая schema, seed и fixed degrees |
| Research tools привязаны к cfg.FEATURES/TARGETS | Те же dataset options и runtime schema, что у CLI |

Оставшиеся имена физических параметров намеренно находятся в legacy adapter,
совместимых предметных presets и необязательных обозначениях latexify. Общий
pipeline их не требует. Оболочки baseline/ML уже использовали sklearn Pipeline;
GPR строил ARD kernel из X.shape[1], MLP и RL строили слои по размерности X.
Их математическая архитектура сохранена.

## Защита от leakage

До синтеза фиксируются группы реальных строк. Для каждого g LOGO использует:
`train = groups != g`, `test = (groups == g) & ~is_synth`.
Все синтетические копии g исключены из train. Модель и её preprocessing создаются
заново для каждого fold. Ни scaler, ни learned transformations не fit на полном
датасете до LOGO. Предварительная проверка значений/ролей не обучает transformations.
После LOGO отдельно обучается финальная модель на всех данных для формулы и
R2_train; её predictions не используются как LOGO predictions.

## Проверки и практические ограничения

До изменения кода сохранены predictions/метрики Linear, Ridge, Lasso, ElasticNet,
GBR и SVR для обеих legacy целей, с синтезом и без него. Fixture включён в tests.
Регрессионные проверки требуют совпадения при rtol=atol=1e-9. Legacy содержит
18 реальных строк, 6 профилей; стандартный синтез добавляет 270 строк.

Проверены все 17 зарегистрированных методов на 9 новых features. Дорогие
stochastic/нейронные методы в smoke-тестах используют уменьшенные число эпох,
популяции и итерации; это проверка интерфейса и размерности, не научная оценка
качества или сходимости. LOGO/метрики подробно проверены на Linear/Ridge/GBR/SVR.
GUI проверяется offscreen через загрузку файла, редактирование ролей, noise/units,
сохранение mapping, worker и построение метрик; offscreen снимки окон legacy и dynamic также осмотрены. Ручной
запуск desktop с живым дисплеем здесь не проводился. Сборка бинарного PyInstaller release не входит в проверку.

Математические ограничения сохранены и описаны в README: положительная область
bio power-law, Q ≥ 0 у PINN, размер train у KNN и физические имена legacy presets.
Общий encoder для произвольных категорий и новые ML-алгоритмы не добавлены.
Для известного параметра материала отдельно реализовано автоматическое
кодирование в бинарный is_steel.
GPR/GBR теперь действительно используют set_seed переданный runner; default seed
GBR не изменился. GPR раньше всегда использовал 42 для случайных рестартов, теперь
использует seed запуска — возможны небольшие различия в stochastic оптимизации.

## Итоговый результат тестов

Команда: `QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/beamsp-mpl venv/bin/python -m unittest discover -s tests -v`.

```text
Ran 29 tests in 6.950s
OK
```

Также успешно выполнены legacy CLI (SP63, без синтеза), dynamic baseline CLI,
все шесть research tools с новым датасетом и compileall. Legacy CSV/PNG/JSON,
включая рисунок формул с индексами целей, проверены отдельным тестом.

## Дополнение: автоматический признак материала

14 октября 2026 года добавлен `core/data/material.py`. Новые CSV/XLSX с колонкой
`material`, `material_type`, «Материал», «Тип материала», «Материал двутавра» или
«Тип материала двутавра» автоматически получают числовой is_steel: сталь → 1,
композит → 0. Признак включается как binary Feature; исходный текст сохраняется
в Metadata. Поле derived_from хранит имя источника в schema и экспорте.

Если mapping ещё не описывает is_steel, он добавляется как Feature. Сохранённое
явное отключение учитывается. Неизвестный материал, пропуски, значения флага
вне 0/1 и противоречия с существующим флагом вызывают ошибки. Источник с ролью
Target/Group не преобразуется в X; смена роли уже использованного источника
также проверяется. Преобразование детерминированное, без fit на данных.

Синтез не меняет бинарный флаг. Для степенных bio-моделей Formula по умолчанию
отключён из-за log(0); остальные модели получают признак. GUI отображает
включённый Feature автоматически. Готовый пример: examples/material.csv.

Новый итоговый прогон после дополнения:

```text
Ran 42 tests in 7.330s
OK
```

Прежние legacy-регрессионные проверки также прошли.
