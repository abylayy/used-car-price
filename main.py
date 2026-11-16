"""Точка входа: запускает этапы проекта по порядку или по отдельности.

Примеры:
    python main.py --stage all                # весь пайплайн (данные → модель → отчёт → презентация)
    python main.py --stage prepare clean db   # только выбранные этапы
    python main.py --csv путь/к/vehicles.csv  # другой путь к данным
    python main.py --nrows 50000              # быстрый прогон на части данных
"""
import argparse
import json
import os
import platform
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

import pandas as pd
import sklearn

import config
import cleaning
import data_loader
import database
import eda
import interpretation
import modeling
import predict
import report_builder
import build_presentation

STAGES = ["prepare", "clean", "db", "eda", "train", "report", "presentation"]


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stage_prepare(csv_path, nrows, benchmark=True):
    """Этап 1: загрузка CSV по чанкам, оптимизация типов, Parquet, аудит качества."""
    print("[1/7] Загрузка и аудит данных")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Файл {csv_path} не найден. Скачайте vehicles.csv с Kaggle "
            "(austinreese/craigslist-carstrucks-data) и положите в data/raw/ — см. README.")
    if benchmark:
        bench = data_loader.benchmark_loading(csv_path, nrows)
        cleaning.save_json({"loading_benchmark": bench}, config.RESULTS_DIR / "loading_benchmark.json")
        for row in bench:
            print(f"  {row['method']}: {row['seconds']} с, {row['memory_mb']} МБ, файл {row['file_mb']} МБ")
        df = pd.read_parquet(config.RAW_PARQUET)
    else:
        df = data_loader.load_csv_chunked(csv_path, nrows=nrows)
        data_loader.save_parquet(df)
    audit = cleaning.audit_quality(df)
    audit["coerced_non_numeric_values"] = data_loader.LAST_LOAD_COERCED
    cleaning.save_json(audit, config.RESULTS_DIR / "audit_raw.json")
    print(f"  строк: {audit['n_rows']:,}; дубликатов по VIN: {audit['duplicates_by_valid_vin']:,}")
    return df


def stage_clean():
    """Этап 2: очистка по правилам, сохранение очищенного Parquet."""
    print("[2/7] Очистка данных")
    df = pd.read_parquet(config.RAW_PARQUET)
    clean, log = cleaning.clean_data(df)
    for step in log:
        print(f"  {step['step']}: удалено {step['removed']:,}, осталось {step['remaining']:,}")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(config.CLEAN_PARQUET, index=False)
    cleaning.save_json({"log": log, "n_clean": len(clean),
                        "memory_mb_clean": round(data_loader.memory_mb(clean), 1)},
                       config.RESULTS_DIR / "cleaning_log.json")
    return clean


def stage_db():
    """Этап 3: SQLite, ER-диаграмма, SQL-запросы, сравнение Pandas и SQL."""
    print("[3/7] База данных SQLite и SQL-запросы")
    df = pd.read_parquet(config.CLEAN_PARQUET)
    counts = database.build_database(df)
    print(f"  таблицы: {counts}")
    database.draw_er_diagram()
    results, summary = database.run_all_queries()
    for s in summary:
        print(f"  {s['query']}: строк={s['rows']}, {s['seconds']} с" + (f" ОШИБКА {s['error']}" if s["error"] else ""))
    cmp_ = database.compare_pandas_vs_sql(df)
    cleaning.save_json({"table_counts": counts, "queries": summary, "pandas_vs_sql": cmp_},
                       config.RESULTS_DIR / "db_summary.json")
    return results


def stage_eda():
    """Этап 4: EDA и визуализации (использует результаты SQL-запросов)."""
    print("[4/7] EDA")
    df = pd.read_parquet(config.CLEAN_PARQUET)
    sql_results = {}
    for path in config.SQL_RESULTS_DIR.glob("*.csv"):
        sql_results[path.stem] = pd.read_csv(path)
    raw_missing = _read_json(config.RESULTS_DIR / "audit_raw.json")["missing_pct"] \
        if (config.RESULTS_DIR / "audit_raw.json").exists() else None
    raw = pd.read_parquet(config.RAW_PARQUET, columns=["price", "odometer"]) if config.RAW_PARQUET.exists() else None
    out = eda.run_eda(df, sql_results, raw_missing, raw)
    out["stats"].to_csv(config.RESULTS_DIR / "descriptive_stats.csv")
    print(f"  графиков: {len(out['figures'])}")


def stage_train():
    """Этап 5: обучение, сравнение моделей, интерпретация, сохранение."""
    print("[5/7] Обучение и сравнение моделей")
    df = pd.read_parquet(config.CLEAN_PARQUET)
    out = modeling.train_and_compare(df)
    results, best = out["results"], out["best_name"]
    print(f"  лучшая модель по CV: {best}")
    results.to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    results.to_parquet(config.RESULTS_DIR / "model_comparison.parquet", index=False)
    with closing(sqlite3.connect(config.DB_PATH)) as con:       # результаты экспериментов — в SQLite
        results.to_sql("experiments", con, if_exists="replace", index=False)

    pipeline = out["models"][best]
    pred_df = interpretation.build_predictions(pipeline, out["X_test"], out["y_test"])
    pred_df.to_parquet(config.RESULTS_DIR / "test_predictions.parquet", index=False)
    importance = interpretation.permutation_importance_report(pipeline, out["X_test"], out["y_test"])
    importance.to_csv(config.RESULTS_DIR / "permutation_importance.csv", index=False)
    interpretation.plot_diagnostics(pred_df, importance, best)
    errors = interpretation.error_analysis(pred_df)
    for name, table in errors.items():
        table.to_csv(config.RESULTS_DIR / f"error_{name}.csv", index=False)
    print("  интервал цены (квантильная регрессия)")
    interval_models = modeling.fit_price_interval(out["X_train"], out["y_train"])
    interval_quality = modeling.evaluate_interval(interval_models, out["X_test"], out["y_test"])
    cleaning.save_json(interval_quality, config.RESULTS_DIR / "interval_quality.json")
    print(f"    внутри интервала {interval_quality['coverage']:.1%} объявлений теста")

    interpretation.generate_findings(results, importance, errors, interval_quality)
    predict.save_model_bundle(pipeline, interval_models, best, df, results, interval_quality)
    print(f"  модель сохранена: {config.MODEL_PATH}")


def stage_report():
    """Этап 6: отчёт REPORT.md с реальными числами из файлов результатов."""
    print("[6/7] Отчёт")
    interpretation.regenerate_findings()      # выводы пересчитываются из сохранённых таблиц (без переобучения)
    for path in report_builder.build_report():
        print(f"  отчёт: {path}")


def stage_presentation():
    """Этап 7: презентация PPTX с таблицами и рисунками из outputs/."""
    print("[7/7] Презентация")
    print(f"  презентация: {build_presentation.build()}")


def write_manifest(stages, elapsed):
    """Манифест запуска: что запускалось, версии, какие файлы созданы (воспроизводимость)."""
    files = {}
    for folder in (config.PROCESSED_DIR, config.OUTPUT_DIR, config.MODELS_DIR, config.PRESENTATION_DIR):
        if folder.exists():
            for p in sorted(folder.rglob("*")):
                if p.is_file() and p.name != ".gitkeep" and p != config.MANIFEST_PATH:
                    files[str(p.relative_to(config.BASE_DIR))] = round(p.stat().st_size / 1024 ** 2, 3)
    cleaning.save_json({
        "stages": stages, "elapsed_seconds": round(elapsed, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(), "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "smoke_mode": os.environ.get("SMOKE") == "1",
        "files_mb": files}, config.MANIFEST_PATH)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Проект: предсказание цены подержанных авто")
    parser.add_argument("--stage", nargs="+", default=["all"], choices=["all"] + STAGES)
    parser.add_argument("--csv", default=str(config.RAW_CSV), help="путь к vehicles.csv")
    parser.add_argument("--nrows", type=int, default=None, help="читать только первые N строк")
    parser.add_argument("--skip-benchmark", action="store_true", help="не сравнивать способы загрузки")
    args = parser.parse_args(argv)
    stages = STAGES if "all" in args.stage else args.stage
    csv_path = Path(args.csv)
    t0 = time.perf_counter()
    try:
        if "prepare" in stages:
            stage_prepare(csv_path, args.nrows, benchmark=not args.skip_benchmark)
        if "clean" in stages:
            stage_clean()
        if "db" in stages:
            stage_db()
        if "eda" in stages:
            stage_eda()
        if "train" in stages:
            stage_train()
        if "report" in stages:
            stage_report()
        if "presentation" in stages:
            stage_presentation()
    except (FileNotFoundError, KeyError) as exc:
        print(f"ОШИБКА: {exc}")
        return 1
    write_manifest(stages, time.perf_counter() - t0)
    print(f"Готово за {time.perf_counter() - t0:.0f} с. Манифест запуска: {config.MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())