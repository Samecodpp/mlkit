# Стандарт ноутбуків лабораторних

Кожна лабораторна — це серія порівняльних запусків: одна базова модель і кілька конфігурацій, що відрізняються одним параметром (оптимізатор, BatchNorm, обсяг даних тощо). Тому всі ноутбуки мають однакову структуру, а від лаби до лаби змінюється лише вміст клітинок.

## Загальні правила

- **Лише клітинки з кодом.** Без markdown-розділів і пояснень: опис і висновки пишуться у звіті.
- **Назва файлу:** `labs/labN/labN.M.ipynb` (наприклад, `labs/lab2/lab2.2.ipynb`).
- **Шляхи від кореня проєкту:** `ROOT = Path("../..")`, далі `ROOT / "datasets/raw/..."`.
- **Константи ВЕЛИКИМИ літерами** і лише в клітинці 0. Жодних «магічних чисел» у решті клітинок.
- **Код, потрібний удруге, переноситься в `mlkit`.** Якщо клас чи функція з ноутбука знадобилися в іншій лабі, їм місце в бібліотеці, а не в копії ноутбука.
- **Результати:** графіки через `FigureSaver` (у `labs/labN/artifacts/<RUN_NAME>/` і в MLflow), метрики й параметри в MLflow.

## Клітинки

### 0. Налаштування

- імпорти: спершу стандартна бібліотека, потім сторонні (`torch`, `pandas`), потім `mlkit`. `mlkit.utils.tracking` імпортується вище за `mlflow`;
- `setup_logging()`, `use_style()`;
- `setup_mlflow(ROOT, "labN.M")`;
- константи: `SEED`, шляхи до даних, гіперпараметри (`BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`…);
- `RUN_NAME = f"labN.M_{datetime.now():%Y%m%d_%H%M%S}"` і `saver = FigureSaver(Path("artifacts") / RUN_NAME)`.

*Розділ звіту:* параметри навчання.

### 1. Опис даних

- завантаження сирих даних;
- кількість прикладів, класи і їхній розподіл, розміри та формат;
- сітка прикладів (`plot_image_grid`) → `saver.save(...)`.

*Розділ звіту:* опис набору даних.

### 2. Підготовка даних

- трансформації (розмір, нормалізація, аугментація лише для train);
- поділ (`StratifiedSplitter` / `RandomSplitter` + `TransformSubset`);
- кешування, якщо дані з диска декодуються повільно;
- `DataLoader` для train / val / test;
- приклади аугментації, якщо вона є.

*Розділ звіту:* попередня обробка.

### 3. Модель

- конфіги моделей (`CNNConfig`, `GeneratorConfig` …) з блоків `ConvBlockConfig`, `LinearBlockConfig`, `ConvTransposeBlockConfig`;
- `print(model)` і кількість параметрів.

*Розділ звіту:* структура моделі.

### 4. Експеримент

- `VARIANTS`: словник «назва → те, що відрізняє конфігурацію» (фабрика оптимізатора, конфіг моделі, частка даних тощо);
- `RunResult`: dataclass з результатами одного запуску (задача, `History`, тестові метрики, час, інше потрібне);
- `run(name, ...) -> RunResult`: один запуск від початку до кінця.

Шаблон `run`:

```python
def run(name: str, variant: ...) -> RunResult:
    seed_everything(SEED)
    task = ...                                            # модель + задача з параметрами варіанту
    with mlflow.start_run(run_name=name):
        mlflow.log_params(flatten_params(model_config) | {...})
        trainer = Trainer(callbacks=[ProgressLogger(...), MLflowLogger(), ...])
        start = time.perf_counter()
        history = trainer.fit(task, train_loader, val_loader, epochs=EPOCHS)
        seconds = time.perf_counter() - start
        test = trainer.evaluate(task, test_loader)
        mlflow.log_metrics({f"test/{name}": value for name, value in test.items()})
    return RunResult(task, history, test, seconds)
```

Обов'язково:
- `seed_everything(SEED)` **на початку кожного запуску**, щоб усі конфігурації мали однакові початкові ваги, батчі й аугментації;
- різниця між запусками — **лише** в параметрі, що досліджується.

*Розділ звіту:* індивідуальне завдання.

### 5. Запуск

```python
results: dict[str, RunResult] = {name: run(name, variant) for name, variant in VARIANTS.items()}
```

Найдовша клітинка: після неї решту можна перезапускати без повторного навчання.

### 6. Результати

- криві: `plot_comparison(histories, [...])` для train і val;
- таблиця підсумків: `pd.DataFrame` з тестових метрик і `History` кожного запуску → `summary.csv` у теці `saver.directory`.

*Розділ звіту:* результати навчання і тестування.

### 7+. Аналіз

Специфічний для лаби:
- **класифікація:** confusion matrix (`confusion_matrix`, `plot_confusion_matrix`), метрики за класами, приклади помилок (`plot_image_grid`);
- **генерація (GAN):** сітки `SampleGrid` на різних епохах, графіки `d_loss`, `g_loss`, `d_real`, `d_fake`, ознаки mode collapse;
- **регресія:** прогноз проти факту, похибки.

*Розділ звіту:* аналіз результатів, висновки.

## Відмінності за типом лаби

| Тип | Що змінюється у шаблоні |
|---|---|
| класифікація (lab1, lab3, lab4.2 LSTM/GRU) | шаблон як є |
| GAN (lab2.2) | немає val і test: `fit(task, train_loader, epochs=...)` без `val_data` і без `evaluate`; callback `SampleGrid`; метрик немає, лише втрати та `d_real` / `d_fake` |
| дифузія (lab2.4) | val з частини train, `evaluate` на test дає метрики оцінки x0 за один крок; `SampleGrid(lambda: task.generate(...))`; після навчання — генерація з однакового шуму, проміжні етапи (`remaining`) і `restore` з метриками `image_metrics()` для кожного $t_0$ |
| PINN (lab4.1) | замість `DataLoader` — семплер колокаційних точок; два `fit` поспіль (Adam → L-BFGS) з однією `History` |
| регресія (lab5, часові ряди lab3) | `RegressionTask`, `regression_metrics()`; аналіз — прогноз проти факту |
| вимога TensorBoard (lab1) | локальний callback `TensorBoardScalars` і `save_figure` з `writer.add_figure` замість MLflow |
