"""Загрузка данных: чтение CSV по частям (chunksize), оптимизация типов, Parquet.

Файл весит больше 1 ГБ, поэтому читаем его порциями, сразу уменьшаем типы данных
и дальше работаем с Parquet. Память и время измеряются в benchmark_loading.
"""
import time
from pathlib import Path

import pandas as pd
from pandas.api.types import union_categoricals

import config

NUMERIC_RAW = ["id", "price", "year", "odometer", "lat", "long"]
LAST_LOAD_COERCED = 0   # сколько нечисловых значений в числовых столбцах заменено на пропуск при последней загрузке


def memory_mb(df: pd.DataFrame) -> float:
    """Реальный объём DataFrame в памяти (МБ), с учётом строк (deep=True)."""
    return float(df.memory_usage(deep=True).sum() / 1024 ** 2)


# Столбцы, типы которых не трогаем: id слишком велик для float32, price нужен точным (проверка заглушек)
KEEP_DTYPE = ("id", "price")


def optimize_dtypes(df: pd.DataFrame, cat_ratio: float = 0.5) -> pd.DataFrame:
    """Уменьшает память: числа -> минимально достаточные типы, строки -> category.

    cat_ratio — строковый столбец становится category, если доля уникальных значений
    меньше этого порога (иначе категории не дают выигрыша).
    """
    for col in df.columns:
        s = df[col]
        if col in KEEP_DTYPE or isinstance(s.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_float_dtype(s):
            df[col] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            n = len(s)
            if n and s.nunique(dropna=True) / n < cat_ratio:
                df[col] = s.astype("category")
    return df


def _concat_chunks(chunks: list, cat_cols: list) -> pd.DataFrame:
    """Склеивает чанки; category-столбцы объединяются через union_categoricals,
    иначе pandas превратил бы их в object и память выросла бы."""
    columns = {}
    for col in chunks[0].columns:
        parts = [ch[col] for ch in chunks]
        if col in cat_cols:
            # если в части чанков столбец стал category, а в части нет — приводим все к category
            parts = [p.astype("category") for p in parts]
            parts = [p.cat.rename_categories(p.cat.categories.astype(str)) for p in parts]
            columns[col] = pd.Series(union_categoricals(parts, ignore_order=True))
        else:
            columns[col] = pd.concat(parts, ignore_index=True)
    return pd.DataFrame(columns)


def check_columns(path: Path) -> list:
    """Проверяет, что в файле есть нужные столбцы (структура датасета не предполагается заранее)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in config.USE_COLS if c not in header]
    if missing:
        raise KeyError(f"В файле {path.name} нет столбцов: {missing}. Найдены: {header}")
    return header


def load_csv_chunked(path: Path = config.RAW_CSV, chunksize: int = config.CHUNKSIZE,
                     nrows: int = None, verbose: bool = True) -> pd.DataFrame:
    """Читает CSV порциями, оптимизируя типы каждой порции, и собирает итоговый DataFrame."""
    check_columns(path)
    cat_cols = config.CATEGORICAL_RAW + ["VIN"]
    dtypes = {c: str for c in cat_cols}   # строки как есть; в category переводим сами, единообразно
    chunks, coerced = [], 0
    reader = pd.read_csv(path, usecols=config.USE_COLS, dtype=dtypes,
                         chunksize=chunksize, nrows=nrows, low_memory=False)
    for i, chunk in enumerate(reader, start=1):
        # Дата размещения: смешанные часовые пояса -> UTC -> без tz
        chunk["posting_date"] = pd.to_datetime(
            chunk["posting_date"], errors="coerce", utc=True).dt.tz_convert(None)
        # В реальном файле встречаются строки с текстом в числовых столбцах. Без принудительного
        # приведения столбец стал бы строковым (и затем category) только в части чанков.
        for c in NUMERIC_RAW:        # мусорные значения -> NaN, их число считаем
            conv = pd.to_numeric(chunk[c], errors="coerce")
            coerced += int(conv.isna().sum() - chunk[c].isna().sum())
            chunk[c] = conv
        chunk["id"] = chunk["id"].astype("Int64")
        for c in cat_cols:
            chunk[c] = chunk[c].astype("category")
        chunk = optimize_dtypes(chunk)
        chunks.append(chunk)
        if verbose:
            print(f"  чанк {i}: {len(chunk):,} строк, {memory_mb(chunk):.1f} МБ")
    global LAST_LOAD_COERCED
    LAST_LOAD_COERCED = coerced
    if coerced:
        print(f"  Предупреждение: {coerced:,} нечисловых значений в числовых столбцах заменены на пропуск")
    return _concat_chunks(chunks, cat_cols)


def save_parquet(df: pd.DataFrame, path: Path = config.RAW_PARQUET) -> float:
    """Сохраняет DataFrame в Parquet, возвращает размер файла (МБ)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path.stat().st_size / 1024 ** 2


def benchmark_loading(csv_path: Path = config.RAW_CSV, nrows: int = None) -> list:
    """Сравнение трёх способов загрузки по времени, памяти и размеру файла.

    1) pandas.read_csv «как есть» (без оптимизации);
    2) read_csv по чанкам + оптимизация типов;
    3) чтение из Parquet (после конвертации).
    """
    results = []

    t0 = time.perf_counter()
    df_naive = pd.read_csv(csv_path, usecols=config.USE_COLS, nrows=nrows, low_memory=False)
    results.append({"method": "CSV: pandas.read_csv без оптимизации",
                    "seconds": round(time.perf_counter() - t0, 2),
                    "memory_mb": round(memory_mb(df_naive), 1),
                    "file_mb": round(csv_path.stat().st_size / 1024 ** 2, 1)})
    del df_naive

    t0 = time.perf_counter()
    df_opt = load_csv_chunked(csv_path, nrows=nrows, verbose=False)
    t_chunk = time.perf_counter() - t0
    results.append({"method": "CSV: chunksize + оптимизация типов",
                    "seconds": round(t_chunk, 2),
                    "memory_mb": round(memory_mb(df_opt), 1),
                    "file_mb": round(csv_path.stat().st_size / 1024 ** 2, 1)})

    parquet_mb = save_parquet(df_opt)
    del df_opt
    t0 = time.perf_counter()
    df_pq = pd.read_parquet(config.RAW_PARQUET)
    results.append({"method": "Parquet (после конвертации)",
                    "seconds": round(time.perf_counter() - t0, 2),
                    "memory_mb": round(memory_mb(df_pq), 1),
                    "file_mb": round(parquet_mb, 1)})
    return results