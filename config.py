"""Конфигурация проекта «Предсказание стоимости подержанных автомобилей».

Все пути, пороги очистки и параметры обучения собраны здесь, чтобы их можно было
менять в одном месте.
"""
import json
import os
from pathlib import Path

# ---------- Команда (подставляется в отчёт и на титульный слайд) ----------
TEAM = [
    {"name": "Нурай Кайратбеккызы", "group": "AAI-2601",
     "role": "анализ и интерфейс: eda.py, app_console.py, app_streamlit.py, тесты, отчёт и презентация"},
    {"name": "Жансая Асылбеккызы", "group": "AAI-2601",
     "role": "данные и база: data_loader.py, cleaning.py, database.py, sql/queries.sql "
             "(загрузка, Parquet, аудит и очистка, схема SQLite, SQL-запросы)"},
    {"name": "Абылай Амандыков", "group": "AAI-2601",
     "role": "модели: features.py, modeling.py, interpretation.py, predict.py "
             "(признаки, обучение и сравнение моделей, интервал цены, анализ ошибок)"},
]

BASE_DIR = Path(__file__).resolve().parent

# ---------- Пути ----------
# Путь к исходному CSV можно переопределить переменной окружения CAR_CSV
RAW_CSV = Path(os.environ.get("CAR_CSV", BASE_DIR / "data" / "raw" / "vehicles.csv"))
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_PARQUET = PROCESSED_DIR / "vehicles_raw.parquet"      # результат конвертации CSV -> Parquet
CLEAN_PARQUET = PROCESSED_DIR / "vehicles_clean.parquet"  # очищенные данные
DB_PATH = PROCESSED_DIR / "cars.db"                       # база SQLite
SQL_FILE = BASE_DIR / "sql" / "queries.sql"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
SQL_RESULTS_DIR = RESULTS_DIR / "sql"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_META_PATH = MODELS_DIR / "model_meta.json"

# ---------- Чтение данных ----------
# Столбцы, которые читаем. Не читаем: url, region_url, image_url (ссылки),
# description (длинный текст — основная часть веса файла), county (полностью пустой).
USE_COLS = [
    "id", "region", "price", "year", "manufacturer", "model", "condition",
    "cylinders", "fuel", "odometer", "title_status", "transmission", "VIN",
    "drive", "size", "type", "paint_color", "state", "lat", "long", "posting_date",
]
CATEGORICAL_RAW = [
    "region", "manufacturer", "model", "condition", "cylinders", "fuel",
    "title_status", "transmission", "drive", "size", "type", "paint_color", "state",
]
CHUNKSIZE = 100_000

# ---------- Правила очистки (обоснование — в README и отчёте) ----------
PRICE_MIN = 500          # цены ниже — чаще всего «фиктивные» ($1, $0, «договорная»)
PRICE_MAX = 150_000      # цены выше — опечатки и заглушки (123456789 и т.п.)
ODOMETER_MAX = 500_000   # пробег свыше 500 тыс. миль — ошибка ввода
ODOMETER_MIN_USED = 100  # пробег меньше 100 миль у авто старше года — заглушка (1, 10...), считаем пропуском
YEAR_MIN = 1985
YEAR_MAX = 2022          # по умолчанию; при запуске = год публикации + 1 (модельный год следующего года)
REFERENCE_YEAR = 2021    # значение по умолчанию; при запуске заменяется годом публикации из данных (cleaning.derive_year_params)
MISSING_DROP_THRESHOLD = 0.5  # столбцы с долей пропусков выше порога исключаются (size)

# ---------- Обучение ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
TUNE_SAMPLE = 60_000     # размер подвыборки обучающей выборки для подбора гиперпараметров
TUNE_N_ITER = 8          # число комбинаций в RandomizedSearchCV
INTERVAL_QUANTILES = (0.1, 0.9)   # границы интервала цены (расширение проекта): 80% объявлений
PERMUTATION_SAMPLE = 5_000        # размер подвыборки теста для Permutation Importance

LOW_CARD_CATEGORICAL = [
    "condition", "cylinders", "fuel", "title_status", "transmission",
    "drive", "type", "paint_color", "state",
]
HIGH_CARD_CATEGORICAL = ["manufacturer", "model", "make_model", "region"]
NUMERIC_FEATURES = ["odometer", "cylinders_num", "car_age", "miles_per_year", "lat", "long"]

# Быстрый режим для проверки работоспособности кода (SMOKE=1): уменьшает объём подбора параметров.
if os.environ.get("SMOKE") == "1":
    TUNE_SAMPLE = 8_000
    TUNE_N_ITER = 2
    CHUNKSIZE = 20_000

# ---------- Отчёт и презентация ----------
REPORT_DIR = OUTPUT_DIR / "report"
REPORT_MD = REPORT_DIR / "REPORT.md"
REPORT_DOCX = REPORT_DIR / "REPORT.docx"
PRESENTATION_DIR = BASE_DIR / "presentation"
PRESENTATION_PPTX = PRESENTATION_DIR / "Used_Car_Price_Prediction.pptx"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"


# ---------- Параметры, вычисленные из данных ----------
# cleaning.derive_year_params сохраняет год публикации в data/processed/data_params.json,
# а здесь он подхватывается при каждом запуске: так возраст авто и допустимый год выпуска
# не зависят от того, в каком году собран датасет.
def load_data_params():
    global REFERENCE_YEAR, YEAR_MAX
    path = PROCESSED_DIR / "data_params.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            params = json.load(f)
        REFERENCE_YEAR, YEAR_MAX = params["reference_year"], params["year_max"]


load_data_params()
