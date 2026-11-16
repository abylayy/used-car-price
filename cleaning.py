"""Аудит качества данных и очистка.

Принцип: сначала измеряем проблемы (audit_quality), затем применяем правила
(clean_data) и фиксируем в журнале, сколько строк удалено на каждом шаге.
"""
import json

import numpy as np
import pandas as pd

import config
from data_loader import optimize_dtypes


def _as_text(s: pd.Series) -> pd.Series:
    """Приводит столбец (в т.ч. category) к обычным строкам с NaN."""
    return s.astype("object")


def is_placeholder_price(p: float) -> bool:
    """Цена-заглушка: одна и та же цифра (11111, 999999), последовательность 12345..., 1234567890."""
    text = str(int(p))
    if len(text) >= 5 and len(set(text)) == 1:
        return True
    return text in {"12345", "123456", "1234567", "12345678", "123456789", "1234567890"}


def placeholder_mask(price: pd.Series) -> pd.Series:
    """Булева маска заглушек; считаем по уникальным ценам, чтобы было быстро."""
    bad = {p for p in price.dropna().unique() if is_placeholder_price(p)}
    return price.isin(bad)


def iqr_bounds(s: pd.Series, k: float = 1.5) -> dict:
    """Границы выбросов по правилу межквартильного размаха (IQR)."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return {"q1": float(q1), "q3": float(q3), "low": float(low), "high": float(high),
            "n_outliers": int(((s < low) | (s > high)).sum())}


def derive_year_params(df: pd.DataFrame):
    """Год отсчёта и верхняя граница года выпуска — из дат публикации в данных (а не «зашиты» в код).

    reference_year — год 99-го процентиля даты публикации; year_max = reference_year + 1
    (в конце года продаются автомобили следующего модельного года). Результат сохраняется
    в data/processed/data_params.json и подхватывается config.py при следующих запусках.
    """
    if "posting_date" not in df or not df["posting_date"].notna().any():
        return None
    ref = int(df["posting_date"].quantile(0.99).year)
    params = {"reference_year": ref, "year_max": ref + 1, "source": "posting_date"}
    config.REFERENCE_YEAR, config.YEAR_MAX = ref, ref + 1
    save_json(params, config.PROCESSED_DIR / "data_params.json")
    return params


def audit_quality(df: pd.DataFrame) -> dict:
    """Аудит качества сырых данных: пропуски, дубликаты, некорректные значения, выбросы."""
    derive_year_params(df)
    n = len(df)
    report = {"n_rows": n, "n_cols": df.shape[1]}
    report["missing_pct"] = (df.isna().mean() * 100).round(2).to_dict()

    # Дубликаты: полные (без id) и по VIN (только корректные 17-символьные VIN)
    report["duplicates_full_without_id"] = int(df.drop(columns=["id"]).duplicated().sum())
    vin = _as_text(df["VIN"])
    valid_vin = vin.where(vin.str.len() == 17)
    report["duplicates_by_valid_vin"] = int(valid_vin.dropna().duplicated().sum())

    price, odo, year = df["price"], df["odometer"], df["year"]
    report["invalid_values"] = {
        "price_missing_or_leq_0": int((price.isna() | (price <= 0)).sum()),
        f"price_below_{config.PRICE_MIN}": int((price < config.PRICE_MIN).sum()),
        f"price_above_{config.PRICE_MAX}": int((price > config.PRICE_MAX).sum()),
        "price_placeholder_values": int(placeholder_mask(price).sum()),
        "odometer_negative": int((odo < 0).sum()),
        f"odometer_below_{config.ODOMETER_MIN_USED}": int((odo < config.ODOMETER_MIN_USED).sum()),
        "price_equals_odometer": int((price == odo).sum()),
        f"odometer_above_{config.ODOMETER_MAX}": int((odo > config.ODOMETER_MAX).sum()),
        f"year_outside_{config.YEAR_MIN}_{config.YEAR_MAX}":
            int(((year < config.YEAR_MIN) | (year > config.YEAR_MAX)).sum()),
    }
    # IQR считаем справочно (применять его к цене рискованно: дорогие авто — не ошибки)
    report["iqr_bounds"] = {
        "price": iqr_bounds(price.dropna()),
        "odometer": iqr_bounds(odo.dropna()),
        "year": iqr_bounds(year.dropna()),
    }
    if "posting_date" in df and df["posting_date"].notna().any():
        report["posting_date_min"] = str(df["posting_date"].min())
        report["posting_date_max"] = str(df["posting_date"].max())
        report["posting_span_days"] = int((df["posting_date"].max() - df["posting_date"].min()).days)
    return report


def clean_data(df: pd.DataFrame) -> tuple:
    """Применяет правила очистки. Возвращает (очищенный DataFrame, журнал шагов)."""
    log = []
    params = derive_year_params(df)
    if params:
        log.append({"step": "year_params", "rule": f"год отсчёта {params['reference_year']}, максимальный год выпуска "
                    f"{params['year_max']} — из дат публикации", "removed": 0, "remaining": len(df)})

    def record(step: str, rule: str, before: int, after: int):
        log.append({"step": step, "rule": rule, "removed": before - after, "remaining": after})

    df = df.copy()

    # 0. Удаляем столбцы с чрезмерной долей пропусков (по умолчанию — size)
    drop_cols = [c for c in df.columns if df[c].isna().mean() > config.MISSING_DROP_THRESHOLD
                 and c not in ("VIN", "posting_date")]
    df = df.drop(columns=drop_cols)
    log.append({"step": "drop_columns", "rule": f"доля пропусков > {config.MISSING_DROP_THRESHOLD:.0%}",
                "removed": len(drop_cols), "remaining": df.shape[1], "columns": drop_cols})

    # 1. Нет цены или года — строку использовать нельзя
    n0 = len(df)
    df = df.dropna(subset=["id", "price", "year"])
    df["id"] = df["id"].astype("int64")
    record("missing_target", "удалены строки без id, price или year (в т.ч. «сдвинутые» строки с текстом "
           "вместо числа)", n0, len(df))

    # 2. Некорректная и «фиктивная» цена
    n0 = len(df)
    df = df[(df["price"] >= config.PRICE_MIN) & (df["price"] <= config.PRICE_MAX)]
    record("price_range", f"цена в диапазоне [{config.PRICE_MIN}; {config.PRICE_MAX}]", n0, len(df))
    n0 = len(df)
    df = df[~placeholder_mask(df["price"])]
    record("price_placeholders", "удалены цены-заглушки (11111, 123456 и т.п.)", n0, len(df))

    # 3. Год выпуска
    n0 = len(df)
    df = df[(df["year"] >= config.YEAR_MIN) & (df["year"] <= config.YEAR_MAX)]
    record("year_range", f"год в диапазоне [{config.YEAR_MIN}; {config.YEAR_MAX}]", n0, len(df))

    # 4. Пробег: аномально большой — удаляем; заглушки (0, 1, 10 миль) у подержанных авто — считаем пропуском
    n0 = len(df)
    df = df[~((df["odometer"] < 0) | (df["odometer"] > config.ODOMETER_MAX))]
    record("odometer_range", f"пробег в диапазоне [0; {config.ODOMETER_MAX}] или пропуск", n0, len(df))
    placeholder_odo = (df["odometer"] < config.ODOMETER_MIN_USED) & (df["year"] < config.REFERENCE_YEAR - 1)
    df.loc[placeholder_odo, "odometer"] = np.nan
    log.append({"step": "odometer_placeholder",
                "rule": f"пробег < {config.ODOMETER_MIN_USED} миль у авто старше 1 года -> пропуск (NaN): "
                        f"{int(placeholder_odo.sum())} значений",
                "removed": 0, "remaining": len(df), "converted_to_nan": int(placeholder_odo.sum())})

    # 5. Цена равна пробегу — продавец вписал одно число в оба поля, настоящая цена неизвестна
    n0 = len(df)
    df = df[~(df["price"] == df["odometer"])]
    record("price_equals_odometer", "удалены строки, где цена совпадает с пробегом (ошибка ввода)", n0, len(df))

    # 6. Дубликаты. Одно авто часто размещают повторно: в соседних регионах или с немного другой ценой.
    #    Оставляем самое свежее объявление. Удаляем ДО разбиения на train/test, иначе будет утечка.
    df = df.sort_values("posting_date")
    n0 = len(df)
    vin = _as_text(df["VIN"])
    valid_vin = vin.where(vin.str.len() == 17).str.upper()
    df = df[~(valid_vin.notna() & valid_vin.duplicated(keep="last"))]
    record("duplicates_vin", "удалены повторы по корректному VIN (одно авто — несколько объявлений)",
           n0, len(df))
    n0 = len(df)
    # Объявления с VIN уже уникальны (разные VIN — разные автомобили), поэтому здесь удаляются только
    # объявления без VIN. При поиске повторов строки с VIN ставим в конец (порядок по дате внутри сохраняется):
    # если у того же авто есть объявление с VIN, остаётся оно, иначе — самое свежее.
    has_vin = valid_vin.loc[df.index].notna()
    order = has_vin.sort_values(kind="stable").index
    same_car = df.loc[order].duplicated(subset=["year", "manufacturer", "model", "odometer"], keep="last")
    df = df[~(same_car.loc[df.index] & ~has_vin & df["odometer"].notna())]
    record("duplicates_same_car", "удалены объявления без VIN, повторяющие год+марку+модель+точный пробег "
           "другого объявления (то же авто)", n0, len(df))

    # 7. Нормализация категорий: нижний регистр, пробелы, пропуски -> 'unknown'
    for col in config.CATEGORICAL_RAW:
        if col in df.columns:
            s = _as_text(df[col]).str.strip().str.lower()
            df[col] = s.where(s.notna() & (s != ""), "unknown")
    df = df.drop(columns=["VIN"])
    df = optimize_dtypes(df.reset_index(drop=True))
    log.append({"step": "normalize", "rule": "нижний регистр, strip, пропуски категорий -> 'unknown'",
                "removed": 0, "remaining": len(df)})
    return df, log


def save_json(obj: dict, path) -> None:
    """Сохраняет словарь в JSON (работа с файлами, требование 5)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)