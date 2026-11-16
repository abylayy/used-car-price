"""Обучение и сравнение моделей: baseline, линейная модель, градиентный бустинг.

Целевая переменная — log1p(price): цена распределена с тяжёлым правым хвостом, логарифм
делает распределение ближе к нормальному и уменьшает влияние дорогих авто на обучение.
Метрики считаются в долларах (после обратного преобразования expm1).
"""
import tempfile
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

import config
from features import add_derived_features, build_preprocessor, feature_columns
from predict import interval_bounds


class MemoryProbe:
    """Контекстный менеджер: измеряет пиковый прирост RSS процесса (МБ) во время блока."""

    def __enter__(self):
        self._proc = psutil.Process()
        self._start = self._proc.memory_info().rss
        self.peak = self._start
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def _watch(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            time.sleep(0.05)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)

    @property
    def peak_delta_mb(self) -> float:
        return (self.peak - self._start) / 1024 ** 2


def split_data(df: pd.DataFrame) -> tuple:
    """Делит данные на обучающую и тестовую выборки ДО любого обучения преобразований.

    Дубликаты уже удалены на этапе очистки (VIN и ключевые признаки), поэтому одинаковые
    объявления не попадают одновременно в train и test.
    """
    data = add_derived_features(df)
    X = data[feature_columns()]
    y = np.log1p(data["price"].astype(float))
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)


def build_models() -> dict:
    """Модели и сетки гиперпараметров для RandomizedSearchCV: {имя: (Pipeline, сетка)}."""
    rs = config.RANDOM_STATE

    def pipe(est):
        return Pipeline([("prep", build_preprocessor()), ("model", est)])

    return {
        "ridge": (pipe(Ridge()), {"model__alpha": loguniform(1e-2, 1e2)}),
        "hist_gb": (
            pipe(HistGradientBoostingRegressor(random_state=rs)),
            {"model__learning_rate": [0.05, 0.1, 0.2], "model__max_iter": [200, 400],
             "model__max_leaf_nodes": [31, 63, 127], "model__l2_regularization": [0.0, 1.0, 10.0]}),
        "lightgbm": (
            pipe(LGBMRegressor(random_state=rs, n_jobs=-1, verbose=-1, subsample_freq=1)),
            {"model__n_estimators": [300, 600, 900], "model__learning_rate": [0.03, 0.06, 0.1],
             "model__num_leaves": [31, 63, 127], "model__min_child_samples": [20, 50],
             "model__subsample": [0.7, 0.9], "model__colsample_bytree": [0.7, 0.9]}),
    }


def evaluate(pipeline, X_test, y_test_log) -> dict:
    """Метрики на тестовой выборке в долларах + инференс на 1000 объектов."""
    t0 = time.perf_counter()
    pred_log = pipeline.predict(X_test)
    t_pred = time.perf_counter() - t0
    y_true = np.expm1(y_test_log)
    y_pred = np.clip(np.expm1(pred_log), 0, None)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_test_log, pred_log))),
        "predict_ms_per_1000": t_pred / max(len(X_test), 1) * 1000 * 1000,
    }


def model_size_mb(pipeline) -> float:
    """Размер сериализованной модели (МБ)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.joblib"
        joblib.dump(pipeline, p)
        return p.stat().st_size / 1024 ** 2


def train_and_compare(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Полный цикл: split -> baseline -> подбор гиперпараметров (CV) -> финальное обучение -> оценка."""
    X_train, X_test, y_train, y_test = split_data(df)
    if verbose:
        print(f"  train: {len(X_train):,}  test: {len(X_test):,}")
    rows, fitted = [], {}

    # --- Подвыборка для подбора гиперпараметров и CV ---
    n_tune = min(len(X_train), config.TUNE_SAMPLE)
    X_tune = X_train.sample(n_tune, random_state=config.RANDOM_STATE)
    y_tune = y_train.loc[X_tune.index]

    # --- Baseline: медиана цены обучающей выборки ---
    base = Pipeline([("prep", build_preprocessor()), ("model", DummyRegressor(strategy="median"))])
    t0 = time.perf_counter()
    with MemoryProbe() as mem:
        base.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    cv_base = -cross_val_score(base, X_tune, y_tune, cv=config.CV_FOLDS,
                               scoring="neg_root_mean_squared_error").mean()
    row = {"model": "baseline_median", "best_params": "{}", "cv_rmse_log": cv_base,
           "tune_seconds": 0.0, "fit_seconds": fit_time, "peak_memory_mb": mem.peak_delta_mb,
           "model_size_mb": model_size_mb(base)}
    row.update(evaluate(base, X_test, y_test))
    rows.append(row)
    fitted["baseline_median"] = base

    for name, (pipe, grid) in build_models().items():
        if verbose:
            print(f"  модель: {name}")
        search = RandomizedSearchCV(pipe, grid, n_iter=config.TUNE_N_ITER, cv=config.CV_FOLDS,
                                    scoring="neg_root_mean_squared_error",
                                    random_state=config.RANDOM_STATE, n_jobs=1, refit=False)
        t0 = time.perf_counter()
        search.fit(X_tune, y_tune)
        tune_time = time.perf_counter() - t0
        pipe.set_params(**search.best_params_)
        t0 = time.perf_counter()
        with MemoryProbe() as mem:
            pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        best_params = {k: round(float(v), 4) if isinstance(v, float) else v for k, v in search.best_params_.items()}
        row = {"model": name, "best_params": str(best_params),   # для таблицы; в модель идут точные значения
               "cv_rmse_log": -search.best_score_, "tune_seconds": tune_time,
               "fit_seconds": fit_time, "peak_memory_mb": mem.peak_delta_mb,
               "model_size_mb": model_size_mb(pipe)}
        row.update(evaluate(pipe, X_test, y_test))
        rows.append(row)
        fitted[name] = pipe
        if verbose:
            print(f"    CV RMSE(log)={row['cv_rmse_log']:.4f}  test MAE=${row['MAE']:,.0f}  "
                  f"RMSE=${row['RMSE']:,.0f}  R2={row['R2']:.3f}")

    results = pd.DataFrame(rows)
    # Лучшую модель выбираем по CV на обучающей выборке (а не по тесту — иначе утечка выбора)
    candidates = results[results["model"] != "baseline_median"]
    best_name = candidates.loc[candidates["cv_rmse_log"].idxmin(), "model"]
    return {"results": results, "models": fitted, "best_name": best_name,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def fit_price_interval(X_train, y_train) -> dict:
    """Расширение проекта: интервал цены вместо одного числа.

    Обучаются две модели квантильной регрессии (нижняя и верхняя граница). Между ними должна
    лежать цена примерно 80% объявлений — пользователь видит, насколько прогнозу можно доверять.
    """
    models = {}
    for q in config.INTERVAL_QUANTILES:
        est = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                            random_state=config.RANDOM_STATE)
        models[q] = Pipeline([("prep", build_preprocessor()), ("model", est)]).fit(X_train, y_train)
    return models


def evaluate_interval(interval_models: dict, X_test, y_test_log) -> dict:
    """Проверка интервала на тесте: доля объявлений внутри интервала и его типичная ширина."""
    low, high = interval_bounds(interval_models, X_test)
    price = np.expm1(y_test_log)
    low_q, high_q = config.INTERVAL_QUANTILES
    return {"target_coverage": round(high_q - low_q, 2),
            "coverage": float(((price >= low) & (price <= high)).mean()),
            "share_below": float((price < low).mean()),
            "share_above": float((price > high).mean()),
            "median_width_usd": float(np.median(high - low)),
            "median_width_pct_of_price": float(np.median((high - low) / price) * 100)}
