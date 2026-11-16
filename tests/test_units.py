"""Модульные тесты. Запуск из корня проекта: python -m unittest discover -s tests -v

Тесты не требуют реального датасета: проверяют ввод прогноза, правила очистки, SQL-запросы
на крошечной искусственной таблице и (если модель уже обучена) работу сохранённой модели.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cleaning, config, database, predict  # noqa: E402


def tiny_frame(n=400):
    """Маленькая таблица той же структуры, что очищенные данные (только для тестов)."""
    rng = np.random.default_rng(1)
    d = pd.DataFrame({
        "id": np.arange(n), "region": rng.choice(["a1", "b2", "c3"], n), "state": rng.choice(["ca", "tx"], n),
        "price": rng.integers(1000, 30000, n).astype(float), "year": rng.integers(2000, 2021, n).astype(float),
        "manufacturer": rng.choice(["ford", "bmw", "unknown"], n), "model": rng.choice(["m1", "m2", "m3"], n),
        "condition": "good", "cylinders": "6 cylinders", "fuel": "gas", "odometer": rng.integers(0, 200000, n).astype(float),
        "title_status": "clean", "transmission": "automatic", "drive": "fwd", "type": "sedan",
        "paint_color": "black", "lat": 40.0, "long": -100.0,
        "posting_date": pd.Timestamp("2021-04-20")})
    return d


class TestValidateInput(unittest.TestCase):
    base = {"year": 2015, "odometer": 80000, "manufacturer": "Ford", "model": "F-150"}

    def test_valid_and_normalised(self):
        out = predict.validate_input(self.base)
        self.assertEqual(out["manufacturer"], "ford")
        self.assertEqual(out["condition"], "unknown")          # отсутствующая категория
        self.assertTrue(np.isnan(out["lat"]))                  # отсутствующая координата

    def test_missing_odometer_allowed(self):
        for empty in (None, "", float("nan")):
            self.assertTrue(np.isnan(predict.validate_input({**self.base, "odometer": empty})["odometer"]))

    def test_bad_values_raise_valueerror(self):
        for bad in ({"year": "abc"}, {"year": 1800}, {"year": None}, {"odometer": "много"},
                    {"odometer": -5}, {"odometer": 10 ** 8}, {"lat": "x"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                predict.validate_input({**self.base, **bad})

    def test_missing_year_key(self):
        with self.assertRaises(ValueError):
            predict.validate_input({"odometer": 1000})


class TestCleaning(unittest.TestCase):
    def test_placeholder_prices(self):
        for p in (11111, 99999, 123456, 1234567, 123456789):
            self.assertTrue(cleaning.is_placeholder_price(p), p)
        for p in (500, 4500, 12500, 31337):
            self.assertFalse(cleaning.is_placeholder_price(p), p)

    def test_iqr(self):
        r = cleaning.iqr_bounds(pd.Series(list(range(1, 100)) + [10_000]))
        self.assertGreaterEqual(r["n_outliers"], 1)

    def test_clean_data_rules(self):
        """Каждая «плохая» строка должна быть обработана своим правилом очистки."""
        good = {"region": "a1", "state": "ca", "price": 9000.0, "year": 2015.0, "manufacturer": "Ford",
                "model": "Focus", "condition": "good", "cylinders": "4 cylinders", "fuel": "gas",
                "odometer": 80000.0, "title_status": "clean", "transmission": "automatic", "VIN": None,
                "drive": "fwd", "size": None, "type": "sedan", "paint_color": "black", "lat": 40.0, "long": -100.0}
        rows = [
            {**good},                                              # 0: нормальная строка
            {**good, "price": 1.0},                                # 1: фиктивная цена
            {**good, "price": 55555.0},                            # 2: цена-заглушка
            {**good, "year": 1950.0},                              # 3: год вне диапазона
            {**good, "odometer": 9_999_999.0},                     # 4: невозможный пробег
            {**good, "price": 70000.0, "odometer": 70000.0},       # 5: цена равна пробегу
            {**good, "model": "Fiesta", "odometer": 1.0},          # 6: пробег-заглушка -> пропуск
            {**good, "price": 9200.0, "state": "tx"},              # 7: то же авто, что и 0, с другой ценой
            {**good, "model": "Kuga", "VIN": "1FADP3F20FL123456"},  # 8 и 9: одинаковый VIN
            {**good, "model": "Kuga", "VIN": "1fadp3f20fl123456", "price": 8800.0, "odometer": 80500.0},
            {**good, "model": "Kuga", "VIN": "1FADP3F20FL654321", "odometer": 80500.0},  # 10: другой VIN — другое авто
        ]
        raw = pd.DataFrame(rows)
        raw["id"] = range(len(raw))
        raw["posting_date"] = pd.Timestamp("2021-04-01") + pd.to_timedelta(raw["id"], unit="D")
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = config.PROCESSED_DIR
            config.PROCESSED_DIR = Path(tmp)                       # чтобы тест не трогал настоящие файлы
            try:
                clean, log = cleaning.clean_data(raw)
            finally:
                config.PROCESSED_DIR = processed_dir
                config.load_data_params()
        self.assertEqual(sorted(clean["id"]), [6, 7, 9, 10])       # из повторов остаётся самое свежее объявление
        self.assertTrue(np.isnan(clean.loc[clean["id"] == 6, "odometer"].iloc[0]))
        self.assertEqual(set(clean["manufacturer"]), {"ford"})     # нижний регистр
        self.assertNotIn("size", clean.columns)                    # столбец без данных удалён
        self.assertEqual(log[-1]["remaining"], 4)


class TestSQL(unittest.TestCase):
    def test_all_queries_run_on_tiny_database(self):
        queries = database.load_queries()
        self.assertGreaterEqual(len(queries), 8)               # в проекте не менее 8 запросов
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.db"
            counts = database.build_database(tiny_frame(), db)
            self.assertEqual(counts["listings"], 400)
            for name, q in queries.items():
                res = database.run_query(db, q["sql"])
                self.assertIsInstance(res, pd.DataFrame, name)


class TestSavedModel(unittest.TestCase):
    @unittest.skipUnless(config.MODEL_PATH.exists(), "модель ещё не обучена (python main.py --stage train)")
    def test_prediction_from_saved_model(self):
        bundle, meta = predict.load_model_bundle()              # читается файл models/best_model.joblib
        v = {"year": 2015, "odometer": 80000, "manufacturer": "ford", "model": "f-150"}
        for values in (v, {**v, "odometer": None}, {**v, "manufacturer": "неизвестная"}):  # пропуск, новая категория
            result = predict.predict_price(bundle, values)
            self.assertGreater(result["price"], 0)
            self.assertLessEqual(result["low"], result["price"])
            self.assertLessEqual(result["price"], result["high"])
        self.assertIn("ниже рынка", predict.rate_asking_price(0, result))


if __name__ == "__main__":
    unittest.main()
