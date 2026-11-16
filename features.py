"""Подготовка признаков: производные признаки и ColumnTransformer.

Все преобразования, которые «учатся» на данных (медианы, масштаб, категории,
целевое кодирование), находятся внутри sklearn Pipeline и обучаются только на
обучающей выборке — так исключается утечка данных (data leakage).
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

import config

# Столбцы, которые пользователь вводит при прогнозе (и которые нужны модели на входе)
INPUT_COLUMNS = ["year", "odometer", "cylinders", "manufacturer", "model", "condition", "fuel",
                 "title_status", "transmission", "drive", "type", "paint_color",
                 "state", "region", "lat", "long"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки. Используется и при обучении, и при прогнозе."""
    out = df.copy()
    for col in config.LOW_CARD_CATEGORICAL + ["manufacturer", "model", "region"]:
        out[col] = out[col].astype("object").fillna("unknown").astype(str)
    out["cylinders_num"] = pd.to_numeric(
        out["cylinders"].str.extract(r"(\d+)")[0], errors="coerce")
    out["car_age"] = (config.REFERENCE_YEAR - out["year"].astype(float)).clip(lower=0)
    out["miles_per_year"] = out["odometer"].astype(float) / (out["car_age"] + 1)
    out["make_model"] = out["manufacturer"] + " " + out["model"]
    for col in ("odometer", "lat", "long"):
        out[col] = out[col].astype(float)
    return out


def build_preprocessor() -> ColumnTransformer:
    """Предобработка:
    - числа: медианная импутация (+ индикаторы пропусков) и стандартизация;
    - категории с малым числом значений: 'unknown' + One-Hot (редкие значения объединяются);
    - категории с большим числом значений (модель, регион...): целевое кодирование
      (TargetEncoder; внутри fit_transform использует кросс-фиттинг, поэтому не «подглядывает»
      в целевую переменную своей же строки).
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    low_card = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                                 sparse_output=False, dtype=np.float32)),
    ])
    # В sklearn >= 1.9 параметр random_state устарел: перемешивание задаётся через KFold
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 9):
        te_kwargs = {"cv": KFold(5, shuffle=True, random_state=config.RANDOM_STATE)}
    else:
        te_kwargs = {"cv": 5, "random_state": config.RANDOM_STATE}
    high_card = TargetEncoder(target_type="continuous", smooth="auto", **te_kwargs)
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("low", low_card, config.LOW_CARD_CATEGORICAL),
        ("high", high_card, config.HIGH_CARD_CATEGORICAL),
    ])


def feature_columns() -> list:
    """Список всех столбцов, подаваемых в ColumnTransformer."""
    return config.NUMERIC_FEATURES + config.LOW_CARD_CATEGORICAL + config.HIGH_CARD_CATEGORICAL
