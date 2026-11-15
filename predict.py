"""Сохранение и загрузка модели, прогноз по введённым пользователем значениям."""
import json
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn

import config
from features import INPUT_COLUMNS, add_derived_features, feature_columns


def save_model_bundle(pipeline, interval_models: dict, best_name: str, df_clean: pd.DataFrame,
                      results: pd.DataFrame, interval_quality: dict) -> dict:
    """Сохраняет модель и модели интервала (joblib) и метаданные (значения для формы ввода, метрики)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipeline, "interval": interval_models}, config.MODEL_PATH)
    choices = {}
    for col in config.LOW_CARD_CATEGORICAL:
        choices[col] = sorted(df_clean[col].astype("object").unique().tolist())
    manufacturers = df_clean["manufacturer"].astype("object").value_counts()
    choices["manufacturer"] = sorted(manufacturers.index.tolist())
    models_by_brand = {}
    for brand in manufacturers.head(60).index:
        sub = df_clean.loc[df_clean["manufacturer"].astype("object") == brand, "model"]
        models_by_brand[brand] = sub.astype("object").value_counts().head(100).index.tolist()
    row = results[results["model"] == best_name].iloc[0]
    meta = {
        "best_model": best_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_file": str(config.MODEL_PATH.name),
        "n_rows_clean": int(len(df_clean)),
        "metrics": {k: float(row[k]) for k in ("MAE", "RMSE", "R2", "MedAE")},
        "interval": interval_quality,
        "choices": choices,
        "models_by_brand": models_by_brand,
        "ranges": {"year": [int(df_clean["year"].min()), int(df_clean["year"].max())],
                   "odometer": [0, config.ODOMETER_MAX],
                   "price": [config.PRICE_MIN, config.PRICE_MAX]},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__},
    }
    with open(config.MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def load_model_bundle() -> tuple:
    """Загружает модели и метаданные. Бросает FileNotFoundError, если модель ещё не обучена.

    Возвращает (bundle, meta), где bundle = {"model": основная модель, "interval": модели границ цены}.
    """
    if not config.MODEL_PATH.exists() or not config.MODEL_META_PATH.exists():
        raise FileNotFoundError("Модель не найдена. Сначала обучите её: python main.py --stage train")
    bundle = joblib.load(config.MODEL_PATH)
    with open(config.MODEL_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return bundle, meta


def _is_missing(value) -> bool:
    """None, пустая строка и NaN считаются «значение не указано»."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def validate_input(values: dict) -> dict:
    """Проверяет ввод пользователя; бросает ValueError с понятным сообщением.

    Обязателен только год выпуска. Пробег, координаты и категории можно не указывать:
    пропуск пробега заменяется медианой обучающей выборки (внутри Pipeline),
    пропуск категории — значением 'unknown'.
    """
    out = dict(values)
    try:
        out["year"] = int(float(out["year"]))
    except (TypeError, ValueError, KeyError):
        raise ValueError("Год выпуска обязателен и должен быть числом.")
    if not config.YEAR_MIN <= out["year"] <= config.YEAR_MAX:
        raise ValueError(f"Год должен быть в диапазоне {config.YEAR_MIN}–{config.YEAR_MAX}.")

    if _is_missing(out.get("odometer")):
        out["odometer"] = np.nan
    else:
        try:
            out["odometer"] = float(out["odometer"])
        except (TypeError, ValueError):
            raise ValueError("Пробег должен быть числом (или оставьте пустым, если неизвестен).")
        if not 0 <= out["odometer"] <= config.ODOMETER_MAX:
            raise ValueError(f"Пробег должен быть в диапазоне 0–{config.ODOMETER_MAX} миль.")

    for col in ("lat", "long"):
        if _is_missing(out.get(col)):
            out[col] = np.nan
        else:
            try:
                out[col] = float(out[col])
            except (TypeError, ValueError):
                raise ValueError(f"Поле {col} должно быть числом.")
    for col in INPUT_COLUMNS:
        if col in ("year", "odometer", "lat", "long"):
            continue
        out[col] = "unknown" if _is_missing(out.get(col)) else str(out[col]).strip().lower()
    return out


def interval_bounds(interval_models: dict, X) -> tuple:
    """Нижняя и верхняя граница цены в долларах для каждой строки X."""
    low_q, high_q = config.INTERVAL_QUANTILES
    low = np.expm1(interval_models[low_q].predict(X))
    high = np.expm1(interval_models[high_q].predict(X))
    return np.clip(np.minimum(low, high), 0, None), np.maximum(low, high)


def predict_price(bundle: dict, values: dict) -> dict:
    """Прогноз по словарю признаков: цена и границы интервала в долларах."""
    row = validate_input(values)
    frame = add_derived_features(pd.DataFrame([row], columns=INPUT_COLUMNS))[feature_columns()]
    price = float(max(np.expm1(bundle["model"].predict(frame)[0]), 0))
    low, high = interval_bounds(bundle["interval"], frame)
    # основной прогноз и границы дают разные модели, поэтому следим, чтобы цена лежала внутри интервала
    return {"price": price, "low": float(min(low[0], price)), "high": float(max(high[0], price))}


def rate_asking_price(asking: float, prediction: dict) -> str:
    """Сравнивает цену из объявления с интервалом прогноза."""
    if asking < prediction["low"]:
        return "ниже рынка: выгодно, но проверьте состояние и документы"
    elif asking > prediction["high"]:
        return "выше рынка: есть повод торговаться"
    else:
        return "в пределах рыночного интервала"
