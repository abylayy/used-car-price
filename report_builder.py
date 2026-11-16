"""Сборка письменного отчёта outputs/report/REPORT.md и REPORT.docx.

Числа и таблицы берутся из файлов outputs/results/, созданных запуском проекта, поэтому отчёт
не расходится с результатами. Пояснения (почему получились такие результаты) написаны авторами.
"""
import json
import shutil
import subprocess

import pandas as pd

import config


def _load_json(name):
    path = config.RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    """DataFrame -> таблица Markdown."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _find_pandoc():
    """Путь к pandoc: системный или из пакета pypandoc-binary. None, если конвертера нет."""
    path = shutil.which("pandoc")
    if path is None:
        try:
            import pypandoc
            path = pypandoc.get_pandoc_path()
        except (ImportError, OSError):
            path = None
    return path


def _pending(stage: str) -> str:
    return f"> Результата ещё нет: выполните `python main.py --stage {stage} report`.\n"


def _fig(name: str, caption: str) -> str:
    """Ссылка на рисунок (путь относительно outputs/report/); если файла нет — пояснение."""
    if (config.FIGURES_DIR / name).exists():
        return f"![{caption}](../figures/{name})\n\n*Рисунок: {caption}.*\n"
    return f"> Рисунок «{caption}» появится после этапа `eda` (файл `outputs/figures/{name}`).\n"


def build_report() -> list:
    audit = _load_json("audit_raw.json")
    bench = _load_json("loading_benchmark.json")
    clean = _load_json("cleaning_log.json")
    db = _load_json("db_summary.json")
    findings = _load_json("findings.json")
    meta = None
    if config.MODEL_META_PATH.exists():
        with open(config.MODEL_META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    comp_path = config.RESULTS_DIR / "model_comparison.csv"
    comp = pd.read_csv(comp_path) if comp_path.exists() else None

    P = []
    P.append("# Отчёт по итоговому проекту: предсказание стоимости подержанных автомобилей\n")
    P.append("**Дисциплина:** Programming for Data Analysis and Database (Applied Artificial Intelligence, "
             "магистратура). **Тема №35:** Used Car Price Prediction.\n")
    P.append("**Авторы:** " + "; ".join(f"{m['name']} ({m['group']})" for m in config.TEAM) + ".\n")

    P.append("## 1. Постановка задачи\n")
    P.append("Требуется по характеристикам объявления (год, пробег, марка, модель, состояние, тип кузова, "
             "регион и др.) предсказать цену подержанного автомобиля и выявить факторы, сильнее всего "
             "влияющие на цену. Это задача регрессии. Качество измеряется средней абсолютной ошибкой "
             "(MAE, Mean Absolute Error) и среднеквадратичной ошибкой (RMSE, Root Mean Squared Error) в долларах; "
             "дополнительно приводятся R² и медианная абсолютная ошибка.\n")

    P.append("## 2. Данные\n")
    P.append("Источник: Craigslist Cars & Trucks (Kaggle, `austinreese/craigslist-carstrucks-data`). "
             "Целевая переменная — цена объявления `price`; модель обучается на log(1+price), "
             "так как распределение цены имеет тяжёлый правый хвост.\n")
    if audit:
        P.append(f"Загружено строк: **{audit['n_rows']:,}**, столбцов в работе: {audit['n_cols']}. "
                 + (f"Период публикации объявлений: {audit.get('posting_date_min')} — {audit.get('posting_date_max')} "
                    f"({audit.get('posting_span_days')} дн.).\n" if audit.get("posting_date_min") else "\n"))
        if bench:
            csv_mb = bench["loading_benchmark"][0]["file_mb"]
            P.append(f"Требование к объёму данных (не менее 500 тыс. строк или 200 МБ) выполнено по размеру файла: "
                     f"CSV занимает {csv_mb:,.0f} МБ. Строк в текущей версии набора меньше 500 тыс., поэтому "
                     f"обработка построена так, как для большого файла: чтение по частям и Parquet.\n")
        miss = pd.Series(audit["missing_pct"]).sort_values(ascending=False).head(10)
        P.append("Пропуски, % (10 столбцов с наибольшей долей):\n")
        P.append(_table(miss.reset_index().rename(columns={"index": "столбец", 0: "пропуски, %"})) + "\n")
        P.append("Некорректные значения до очистки:\n")
        inv = pd.DataFrame(list(audit["invalid_values"].items()), columns=["проверка", "число строк"])
        P.append(_table(inv) + "\n")
        P.append(f"Дубликаты: по корректному VIN — {audit['duplicates_by_valid_vin']:,}; "
                 f"полные (без id) — {audit['duplicates_full_without_id']:,}.\n")
        iq = audit["iqr_bounds"]["price"]
        P.append(f"Справочно, границы выбросов цены по правилу IQR: [{iq['low']:,.0f}; {iq['high']:,.0f}], "
                 f"выбросов по этому правилу: {iq['n_outliers']:,}. Правило IQR к цене не применялось "
                 f"(дорогие автомобили — не ошибки данных); применены доменные границы, см. раздел 4.\n")
    else:
        P.append(_pending("prepare"))

    P.append("## 3. Загрузка данных и оптимизация памяти\n")
    P.append("Применены: чтение только нужных столбцов (`usecols`), чтение по частям (`chunksize`), "
             "понижение разрядности чисел (`downcast`), тип `category` для строк с малым числом значений, "
             "формат Parquet.\n")
    if bench:
        table = pd.DataFrame(bench["loading_benchmark"])
        table.columns = ["Способ", "Время чтения, с", "DataFrame в памяти, МБ", "Файл на диске, МБ"]
        P.append(_table(table) + "\n")
        P.append("Оба способа чтения CSV тратят почти одинаковое время: оно уходит на разбор текста, и чтение "
                 "по частям его не сокращает. Выигрыш чанков и оптимизации типов — в памяти: числа хранятся "
                 "в типах меньшей разрядности, а повторяющиеся строки — как категории. Parquet хранит данные "
                 "по столбцам, в сжатом виде и с готовыми типами, поэтому читается во много раз быстрее; "
                 "все следующие этапы работают с ним, а не с CSV.\n")
    else:
        P.append(_pending("prepare"))

    P.append("## 4. Очистка данных\n")
    P.append(f"Правила (пороги заданы в `config.py`):\n\n"
             f"* цена вне [{config.PRICE_MIN}; {config.PRICE_MAX:,}] и цены-заглушки (11111, 123456) удаляются: "
             f"это «договорная цена», первый взнос или опечатка;\n"
             f"* год выпуска вне [{config.YEAR_MIN}; {config.YEAR_MAX}] удаляется;\n"
             f"* пробег больше {config.ODOMETER_MAX:,} миль удаляется, а пробег меньше {config.ODOMETER_MIN_USED} миль "
             f"у автомобиля старше года считается пропуском (продавцы вписывают 0, 1 или 10 вместо настоящего значения);\n"
             f"* строки, где цена совпадает с пробегом, удаляются: одно число вписано в оба поля;\n"
             f"* повторные объявления удаляются до разбиения на выборки: сначала по VIN, затем по совпадению "
             f"года, марки, модели и точного пробега (только объявления без VIN: разные VIN означают разные "
             f"автомобили). Из повторов остаётся объявление с VIN, а если его нет — самое свежее;\n"
             f"* категории приводятся к нижнему регистру, пропуски заменяются на `unknown`.\n\n"
             f"Правило IQR к цене не применяется: оно отнесло бы к выбросам все дорогие автомобили, "
             f"а это реальные объекты, а не ошибки.\n")
    if clean:
        log = pd.DataFrame(clean["log"])[["rule", "removed", "remaining"]].iloc[2:]   # без служебных шагов
        log.columns = ["Правило", "Удалено строк", "Осталось строк"]
        P.append(_table(log) + "\n")
        P.append(f"Итог: {clean['n_clean']:,} строк, {clean['memory_mb_clean']} МБ в памяти.\n")
    else:
        P.append(_pending("clean"))

    P.append("## 5. База данных SQLite и SQL-запросы\n")
    P.append("Схема в третьей нормальной форме: `manufacturers` → `models` → `listings` ← `regions`.\n")
    P.append(_fig("er_diagram.png", "ER-диаграмма базы данных"))
    if db:
        P.append(f"Число строк в таблицах: {db['table_counts']}.\n")
        P.append(_table(pd.DataFrame(db["queries"])[["query", "desc", "rows", "seconds"]], "{:,.3f}") + "\n")
        P.append("Сравнение Pandas и SQL на одной задаче (средняя цена по маркам):\n")
        P.append(_table(pd.DataFrame(db["pandas_vs_sql"]), "{:,.4f}") + "\n")
        P.append("Результаты запросов сохранены в `outputs/results/sql/` и использованы при построении "
                 "графиков 08 и 09 (цена по году выпуска и по пробегу).\n")
    else:
        P.append(_pending("db"))

    P.append("## 6. Исследовательский анализ данных (EDA)\n")
    stats_path = config.RESULTS_DIR / "descriptive_stats.csv"
    if stats_path.exists():
        st = pd.read_csv(stats_path, index_col=0)
        P.append("Описательная статистика числовых признаков:\n")
        P.append(_table(st[["count", "mean", "std", "min", "50%", "max", "skew"]].reset_index()
                        .rename(columns={"index": "признак"}), "{:,.2f}") + "\n")
    for name, cap in [("01_price_distribution.png", "Распределение цены и log(1+цена)"),
                      ("02_missing_values.png", "Доля пропусков по столбцам (сырые данные)"),
                      ("11_outliers_raw.png", "Выбросы цены и пробега до очистки"),
                      ("03_price_by_age.png", "Цена в зависимости от возраста"),
                      ("04_price_vs_odometer.png", "Цена и пробег"),
                      ("05_median_price_by_brand.png", "Медианная цена по маркам"),
                      ("06_correlation_heatmap.png", "Корреляции числовых признаков"),
                      ("07_price_by_categories.png", "Цена по категориальным признакам"),
                      ("08_sql_price_by_year.png", "Средняя цена по году выпуска (результат SQL-запроса q02)"),
                      ("09_sql_price_by_odometer.png", "Цена по корзинам пробега (результат SQL-запроса q09)")]:
        P.append(_fig(name, cap))
    P.append("Интерактивный график: `outputs/figures/10_interactive_price_odometer.html` (открыть в браузере).\n")
    ef = _load_json("eda_findings.json")
    if ef:
        P.append("### Выводы по EDA (рассчитаны по данным)\n")
        for it in ef["items"]:
            P.append(f"- **{it['topic']}.** {it['text']}")
        P.append("")
    else:
        P.append(_pending("eda"))
    P.append("### Интерпретация\n")
    P.append("Цена падает с возрастом и пробегом, потому что автомобиль изнашивается, у него заканчивается "
             "гарантия и сокращается оставшийся срок службы. Падение самое быстрое в первые годы, дальше кривая "
             "выравнивается: старый автомобиль дешевеет медленно. Возраст и пробег сильно связаны между собой, "
             "поэтому линейной модели трудно разделить их вклад.\n")
    P.append("Пикапы, грузовики, полный привод и дизель дороже остальных: такие автомобили дороже новыми "
             "и нужны для работы, поэтому дольше сохраняют цену.\n")
    P.append("Признак `condition` заполняет сам продавец, и он ненадёжен. Значение «salvage» должно означать "
             "автомобиль после серьёзной аварии, но медианная цена у этой группы высокая. Проверка показала, "
             "что так размечены в основном объявления дилеров коммерческих грузовиков (Ford F-550, Ram ProMaster) "
             "с чистыми документами. Поэтому в модели есть и `condition`, и `title_status`: второй признак "
             "точнее описывает юридическое состояние автомобиля.\n")

    P.append("## 7. Методы\n")
    P.append(f"* **Разбиение:** обучающая выборка {1 - config.TEST_SIZE:.0%} / тестовая {config.TEST_SIZE:.0%}, "
             f"`random_state={config.RANDOM_STATE}`; разбиение выполняется до обучения любых преобразований.\n"
             "* **Предобработка (`ColumnTransformer` в `Pipeline`):** числа — медианная импутация, индикаторы "
             "пропусков, стандартизация; категории с малым числом значений — One-Hot; категории с большим "
             "числом значений (марка, модель, марка+модель, регион) — целевое кодирование (`TargetEncoder`, "
             "кросс-фиттинг).\n"
             "* **Модели:** baseline (медиана цены), линейная модель Ridge и два варианта градиентного "
             "бустинга: HistGradientBoosting (scikit-learn) и LightGBM.\n"
             f"* **Подбор гиперпараметров:** `RandomizedSearchCV`, {config.TUNE_N_ITER} комбинаций, "
             f"{config.CV_FOLDS}-блочная кросс-валидация на подвыборке обучающей выборки "
             f"({config.TUNE_SAMPLE:,} объектов); итоговая модель обучается на всей обучающей выборке.\n"
             "* **Выбор лучшей модели** — по CV-метрике (RMSE в лог-шкале) на обучающей выборке; тест "
             "используется один раз для итоговой оценки.\n"
             "* **Защита от утечки данных:** повторы удалены до разбиения; все «обучаемые» преобразования "
             "находятся внутри Pipeline и обучаются только на обучающих данных.\n")
    if audit and audit.get("posting_span_days") is not None:
        P.append(f"Разбиение случайное, а не по времени: все объявления опубликованы за {audit['posting_span_days']} "
                 f"дн., за такой срок рынок не меняется, и разбиение по дате не дало бы более честной оценки.\n")

    P.append("## 8. Результаты сравнения моделей\n")
    if comp is not None:
        view = comp[["model", "cv_rmse_log", "MAE", "RMSE", "R2", "tune_seconds", "fit_seconds",
                     "peak_memory_mb", "model_size_mb", "predict_ms_per_1000"]]
        P.append(_table(view) + "\n")
        P.append("Обозначения: `cv_rmse_log` — RMSE кросс-валидации в лог-шкале; MAE, RMSE, R² — на тесте, "
                 "в долларах; `peak_memory_mb` — пиковый прирост памяти процесса при обучении; "
                 "`predict_ms_per_1000` — время предсказания для 1000 объектов, мс.\n")
        P.append("Гиперпараметры лучших моделей:\n")
        P.append(_table(comp[["model", "best_params"]]) + "\n")
    else:
        P.append(_pending("train"))

    P.append("## 9. Интерпретация и анализ ошибок\n")
    imp_path = config.RESULTS_DIR / "permutation_importance.csv"
    if imp_path.exists():
        P.append("Важность признаков (Permutation Importance, топ-10):\n")
        P.append(_table(pd.read_csv(imp_path).head(10), "{:,.4f}") + "\n")
        P.append(_fig("permutation_importance.png", "Важность признаков"))
        P.append(_fig("residuals.png", "Факт vs прогноз, остатки и распределение ошибок"))
        for name, title in [("error_by_price", "по ценовым диапазонам"), ("error_by_age", "по возрасту автомобиля"),
                            ("error_by_brand_worst", "марки с наибольшей MAE (n ≥ 100)"),
                            ("error_worst_predictions", "30 наибольших ошибок")]:
            p = config.RESULTS_DIR / f"{name}.csv"
            if p.exists():
                P.append(f"**Ошибки {title}:**\n")
                P.append(_table(pd.read_csv(p), "{:,.1f}") + "\n")
    else:
        P.append(_pending("train"))
    if findings:
        P.append("### Автоматически сформированные выводы (на основе таблиц выше)\n")
        for it in findings["items"]:
            P.append(f"- **{it['topic']}.** {it['text']}")
        P.append("")
    P.append("### Почему модель ошибается\n")
    P.append("**Дорогие автомобили занижаются.** Причин три. Первая: дорогих объявлений мало, и модель тянет "
             "прогноз к типичной цене. Вторая: модель обучается на логарифме цены, а обратное преобразование "
             "даёт оценку ближе к медиане, чем к среднему, то есть ниже для распределения с правым хвостом. "
             "Третья: цену таких автомобилей определяют комплектация и редкие версии (например, Dodge Demon 170 "
             "или Mercedes G-класса), а в данных есть только марка и название модели.\n")
    P.append("**Дешёвые автомобили завышаются.** У автомобиля дешевле $5,000 обычно есть дефекты, которые "
             "продавец описывает в тексте объявления, а не в полях. Кроме того, часть дилеров указывает "
             "в поле цены первый взнос, а не полную стоимость.\n")
    P.append("**Старые автомобили.** Среди автомобилей старше 20 лет есть и коллекционные, и доживающие свой "
             "срок. По году и пробегу они неразличимы, а цены отличаются в разы.\n")
    top10 = set(pd.read_csv(imp_path).head(10)["feature"]) if imp_path.exists() else set()
    coords = [name for col, name in (("long", "долгота"), ("lat", "широта")) if col in top10]
    if coords:
        P.append(f"**Важность координат.** Среди десяти самых важных признаков есть координаты объявления "
                 f"({', '.join(coords)}): цены "
                 "различаются по регионам (в сельских штатах больше пикапов и выше средняя цена), и координаты "
                 "описывают место точнее, чем название штата.\n")
    P.append("**Самые большие ошибки** приходятся на редкие спортивные и люксовые автомобили (часто с незаполненной "
             "маркой), на коммерческую технику и на объявления с ошибкой в цене. "
             "Это видно в таблице худших прогнозов: модель предсказывает обычную цену для своей марки, "
             "а в объявлении стоит цена в несколько раз выше.\n")

    P.append("## 10. Прогноз и интервал цены\n")
    P.append("Прогноз выполняется сохранённой обученной моделью (`models/best_model.joblib`): "
             "консольное меню `python app_console.py` (пункт 7) или веб-интерфейс "
             "`streamlit run app_streamlit.py`. Обязателен только год выпуска; пробег, координаты и категории "
             "можно не указывать (пробег заменяется медианой обучающей выборки, категории — `unknown`). "
             "Некорректный ввод (текст вместо числа, год или пробег вне допустимого диапазона) "
             "отклоняется с понятным сообщением.\n")
    if meta:
        P.append(f"Сохранённая модель: **{meta['best_model']}**, обучена {meta['trained_at']}; "
                 f"метрики на тесте: MAE ${meta['metrics']['MAE']:,.0f}, RMSE ${meta['metrics']['RMSE']:,.0f}, "
                 f"R² {meta['metrics']['R2']:.3f}.\n")
    else:
        P.append(_pending("train"))
    P.append("**Расширение проекта: интервал цены.** Одно число создаёт ложное ощущение точности, поэтому "
             "вместе с прогнозом приложение показывает интервал. Его границы дают две модели квантильной "
             f"регрессии (HistGradientBoosting с квантилями {config.INTERVAL_QUANTILES[0]} и "
             f"{config.INTERVAL_QUANTILES[1]}), обученные на той же обучающей выборке. Если пользователь вводит "
             "цену из объявления, приложение сообщает, ниже она рынка, выше или внутри интервала.\n")
    quality = _load_json("interval_quality.json")
    if quality:
        P.append(f"Проверка на тесте: интервал должен накрывать {quality['target_coverage']:.0%} объявлений, "
                 f"фактически внутри него {quality['coverage']:.1%} (ниже {quality['share_below']:.1%}, "
                 f"выше {quality['share_above']:.1%}). Медианная ширина — ${quality['median_width_usd']:,.0f}, "
                 f"или {quality['median_width_pct_of_price']:.0f}% от цены. Интервал широкий: это честная мера "
                 "того, насколько цену объявления можно предсказать по имеющимся признакам.\n")

    P.append("## 11. Ограничения\n")
    P.append("* Цены в объявлениях — это цены предложения, а не фактические цены сделок.\n"
             "* Повторные объявления удалены до разбиения, но полностью исключить утечку нельзя: продавец "
             "может разместить то же авто повторно, изменив пробег.\n"
             "* В данных нет информации о состоянии кузова, комплектации, истории обслуживания и фотографиях.\n"
             "* Пороги очистки (config.py) выбраны экспертно и влияют на метрики.\n"
             "* Подбор гиперпараметров выполнен на подвыборке обучающей выборки ради времени вычислений.\n"
             "* Прогноз для редких моделей опирается на сглаженное целевое кодирование и менее надёжен.\n")

    P.append("## 12. Выводы\n")
    if findings and comp is not None:
        for it in findings["items"][:4]:
            P.append(f"- {it['text']}")
        P.append("")
        base = comp[comp["model"] == "baseline_median"].iloc[0]
        best = comp[comp["model"] == meta["best_model"]].iloc[0] if meta else comp.loc[comp["cv_rmse_log"].idxmin()]
        P.append(f"**Итог.** Цену объявления удаётся предсказать со средней ошибкой ${best['MAE']:,.0f} против "
                 f"${base['MAE']:,.0f} у baseline; половина прогнозов ошибается меньше чем на ${best['MedAE']:,.0f}. "
                 "Градиентный бустинг заметно точнее линейной модели, потому что цена зависит от возраста и пробега "
                 "нелинейно, а влияние признаков различается для разных марок. Главные факторы цены — возраст "
                 "и пробег. Модель подходит для быстрой оценки массовых автомобилей "
                 "среднего ценового диапазона; для дорогих, редких и очень старых автомобилей её прогноз "
                 "следует считать ориентиром, и приложение показывает это шириной интервала.\n")
    else:
        P.append(_pending("train"))

    P.append("## 13. Вклад участников\n")
    P.append(_table(pd.DataFrame([[m["name"], m["group"], m["role"]] for m in config.TEAM],
                                 columns=["Участник", "Группа", "Вклад"])) + "\n")
    P.append("Итоговая проверка проекта и подготовка к защите выполнены совместно.\n")

    P.append("## 14. Воспроизводимость\n")
    if meta:
        P.append(f"Версии при обучении: {meta['versions']}.\n")
    P.append("Запуск: `pip install -r requirements.txt`, затем `python main.py --stage all`. "
             "Манифест запуска со списком созданных файлов: `outputs/run_manifest.json`.\n")

    P.append("## 15. Источники\n"
             "* Craigslist Cars & Trucks Data, Kaggle: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data\n"
             "* Документация scikit-learn (Pipeline, TargetEncoder, HistGradientBoostingRegressor, "
             "permutation_importance): https://scikit-learn.org/stable/\n"
             "* Документация LightGBM: https://lightgbm.readthedocs.io/\n"
             "* Документация pandas, SQLite (оконные функции), Plotly, Streamlit.\n")

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_MD.write_text("\n".join(P), encoding="utf-8")
    paths = [str(config.REPORT_MD)]
    pandoc = _find_pandoc()
    if pandoc is None:
        print("  REPORT.docx не создан: нет pandoc. Установите его командой pip install pypandoc-binary")
        return paths
    try:
        subprocess.run([pandoc, config.REPORT_MD.name, "-o", config.REPORT_DOCX.name],
                       cwd=config.REPORT_DIR, check=True, capture_output=True, timeout=180)
        paths.append(str(config.REPORT_DOCX))
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"  REPORT.docx не создан: {exc}")
    return paths