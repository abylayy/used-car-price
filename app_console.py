"""Консольное меню проекта (требование 7). Запуск: python app_console.py"""
from pathlib import Path

import pandas as pd

import config
import database
import main as pipeline
import predict

MENU = """
===== Предсказание цены подержанных авто =====
1 – Подготовить данные (загрузка CSV, Parquet, аудит)
2 – Очистить данные
3 – Создать БД SQLite и выполнить SQL-запросы
4 – Показать результат SQL-запроса
5 – Построить EDA-графики
6 – Обучить и сравнить модели
7 – Получить прогноз цены
8 – Показать таблицу сравнения моделей
0 – Выход
"""


def ask_float(prompt: str, low: float, high: float) -> float:
    """Повторяет ввод, пока пользователь не введёт число в диапазоне (цикл while + try/except)."""
    while True:
        try:
            value = float(input(prompt).replace(",", "."))
        except ValueError:
            print("  Нужно ввести число.")
            continue
        if low <= value <= high:
            return value
        print(f"  Значение должно быть от {low} до {high}.")


def ask_optional_float(prompt: str, low: float, high: float):
    """Как ask_float, но пустой ввод означает «значение неизвестно» (None)."""
    while True:
        raw = input(prompt).strip()
        if not raw:
            return None
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            print("  Нужно ввести число или нажать Enter.")
            continue
        if low <= value <= high:
            return value
        print(f"  Значение должно быть от {low} до {high}.")


def ask_choice(prompt: str, options: list, default: str = "unknown") -> str:
    """Выбор значения из списка допустимых; пустой ввод — значение по умолчанию."""
    print(f"  Допустимые значения: {', '.join(options[:15])}{' ...' if len(options) > 15 else ''}")
    value = input(f"{prompt} [Enter = {default}]: ").strip().lower()
    return value or default


def show_sql():
    """Показывает список запросов и выполняет выбранный (результат загружается в Pandas)."""
    queries = database.load_queries()
    names = list(queries)
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name} — {queries[name]['desc']}")
    try:
        idx = int(input("Номер запроса: ")) - 1
        result = database.run_query(config.DB_PATH, queries[names[idx]]["sql"])
    except (ValueError, IndexError):
        print("  Неверный номер запроса.")
        return
    except Exception as exc:  # например, БД ещё не создана
        print(f"  Не удалось выполнить запрос: {exc}")
        return
    print(result.to_string(index=False, max_rows=30))


def do_prediction():
    """Диалог ввода признаков и вывод прогноза."""
    try:
        bundle, meta = predict.load_model_bundle()
    except FileNotFoundError as exc:
        print(f"  {exc}")
        return
    ch = meta["choices"]
    lo, hi = meta["ranges"]["year"]
    values = {
        "year": ask_float(f"Год выпуска ({lo}–{hi}): ", config.YEAR_MIN, config.YEAR_MAX),
        "odometer": ask_optional_float("Пробег, миль (Enter — неизвестен): ", 0, config.ODOMETER_MAX),
        "manufacturer": ask_choice("Марка", ch["manufacturer"]),
    }
    brand = values["manufacturer"]
    models = meta["models_by_brand"].get(brand, [])
    values["model"] = ask_choice("Модель", models) if models else input("Модель: ").strip().lower() or "unknown"
    for col, label in [("condition", "Состояние"), ("fuel", "Топливо"), ("transmission", "КПП"),
                       ("drive", "Привод"), ("type", "Тип кузова"), ("paint_color", "Цвет"),
                       ("cylinders", "Цилиндры"), ("title_status", "Статус документов"), ("state", "Штат")]:
        values[col] = ask_choice(label, ch[col])
    values["region"] = "unknown"
    values["lat"], values["long"] = None, None
    try:
        result = predict.predict_price(bundle, values)
    except ValueError as exc:
        print(f"  Ошибка ввода: {exc}")
        return
    print(f"\n  Прогноз цены: ${result['price']:,.0f}")
    print(f"  Интервал: ${result['low']:,.0f} – ${result['high']:,.0f} "
          f"(в него попадает около {meta['interval']['coverage']:.0%} похожих объявлений)")
    print(f"  Модель {meta['best_model']}, обучена {meta['trained_at']}; "
          f"средняя ошибка на тесте MAE ≈ ${meta['metrics']['MAE']:,.0f}")
    if values["odometer"] is None:
        print("  Внимание: пробег не указан, использована медиана обучающей выборки — прогноз менее точен.")
    asking = ask_optional_float("Цена в объявлении, $ (Enter — пропустить): ", 0, 10 * config.PRICE_MAX)
    if asking is not None:
        print(f"  Цена ${asking:,.0f} — {predict.rate_asking_price(asking, result)}.")


def show_comparison():
    path = config.RESULTS_DIR / "model_comparison.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("  Таблица ещё не создана — выполните пункт 6.")
        return
    print(df[["model", "MAE", "RMSE", "R2", "fit_seconds", "model_size_mb"]].round(2).to_string(index=False))


def run_step(func, *args):
    """Запускает этап пайплайна, не позволяя ошибке завершить меню."""
    try:
        func(*args)
    except (FileNotFoundError, KeyError) as exc:
        print(f"  ОШИБКА: {exc}")


def main():
    actions = {
        "1": lambda: run_step(pipeline.stage_prepare, Path(config.RAW_CSV), None, True),
        "2": lambda: run_step(pipeline.stage_clean),
        "3": lambda: run_step(pipeline.stage_db),
        "4": show_sql,
        "5": lambda: run_step(pipeline.stage_eda),
        "6": lambda: run_step(pipeline.stage_train),
        "7": do_prediction,
        "8": show_comparison,
    }
    while True:
        print(MENU)
        choice = input("Выберите пункт: ").strip()
        if choice == "0":
            print("До свидания!")
            break
        elif choice in actions:
            actions[choice]()
        else:
            print("  Неизвестный пункт меню.")


if __name__ == "__main__":
    main()
