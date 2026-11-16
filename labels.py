"""Русские подписи для значений признаков в интерфейсе (значения в данных остаются прежними).

Пользователь видит «Бензин», «Седан», «Пенсильвания», а в модель уходят исходные значения
(gas, sedan, pa), на которых она обучалась.
"""
import re

UNKNOWN = "unknown"
UNKNOWN_LABEL = "Не указано"

STATES = {
    "al": "Алабама", "ak": "Аляска", "az": "Аризона", "ar": "Арканзас", "ca": "Калифорния",
    "co": "Колорадо", "ct": "Коннектикут", "de": "Делавэр", "dc": "Вашингтон (округ Колумбия)",
    "fl": "Флорида", "ga": "Джорджия", "hi": "Гавайи", "id": "Айдахо", "il": "Иллинойс",
    "in": "Индиана", "ia": "Айова", "ks": "Канзас", "ky": "Кентукки", "la": "Луизиана",
    "me": "Мэн", "md": "Мэриленд", "ma": "Массачусетс", "mi": "Мичиган", "mn": "Миннесота",
    "ms": "Миссисипи", "mo": "Миссури", "mt": "Монтана", "ne": "Небраска", "nv": "Невада",
    "nh": "Нью-Гэмпшир", "nj": "Нью-Джерси", "nm": "Нью-Мексико", "ny": "Нью-Йорк",
    "nc": "Северная Каролина", "nd": "Северная Дакота", "oh": "Огайо", "ok": "Оклахома",
    "or": "Орегон", "pa": "Пенсильвания", "ri": "Род-Айленд", "sc": "Южная Каролина",
    "sd": "Южная Дакота", "tn": "Теннесси", "tx": "Техас", "ut": "Юта", "vt": "Вермонт",
    "va": "Вирджиния", "wa": "Вашингтон", "wv": "Западная Вирджиния", "wi": "Висконсин",
    "wy": "Вайоминг",
}

VALUE_LABELS = {
    "condition": {"new": "Новый", "like new": "Как новый", "excellent": "Отличное", "good": "Хорошее",
                  "fair": "Удовлетворительное", "salvage": "Аварийное (salvage)"},
    "fuel": {"gas": "Бензин", "diesel": "Дизель", "hybrid": "Гибрид", "electric": "Электро", "other": "Другое"},
    "transmission": {"automatic": "Автоматическая", "manual": "Механическая", "other": "Другая"},
    "drive": {"4wd": "Полный привод (4WD)", "fwd": "Передний привод (FWD)", "rwd": "Задний привод (RWD)"},
    "type": {"sedan": "Седан", "suv": "Внедорожник / SUV", "truck": "Грузовик", "pickup": "Пикап",
             "coupe": "Купе", "hatchback": "Хэтчбек", "wagon": "Универсал", "van": "Фургон",
             "mini-van": "Минивэн", "convertible": "Кабриолет", "bus": "Автобус",
             "offroad": "Внедорожник (offroad)", "other": "Другое"},
    "title_status": {"clean": "Чистые документы", "rebuilt": "Восстановленный после аварии (rebuilt)",
                     "salvage": "Аварийный (salvage)", "lien": "Под залогом (lien)",
                     "missing": "Документы отсутствуют", "parts only": "Только на запчасти"},
    "paint_color": {"black": "Чёрный", "white": "Белый", "silver": "Серебристый", "grey": "Серый",
                    "blue": "Синий", "red": "Красный", "green": "Зелёный", "brown": "Коричневый",
                    "custom": "Нестандартный", "yellow": "Жёлтый", "orange": "Оранжевый",
                    "purple": "Фиолетовый"},
}

# Русские названия полей формы
FIELD_TITLES = {
    "condition": "Состояние", "fuel": "Топливо", "transmission": "Коробка передач", "drive": "Привод",
    "type": "Тип кузова", "paint_color": "Цвет", "cylinders": "Число цилиндров",
    "title_status": "Статус документов", "state": "Штат США",
}

# Русские названия признаков модели (для графика важности признаков)
FEATURE_TITLES = {
    **FIELD_TITLES, "car_age": "Возраст автомобиля", "odometer": "Пробег", "manufacturer": "Марка",
    "model": "Модель", "make_model": "Марка + модель", "region": "Регион", "state": "Штат",
    "lat": "Широта", "long": "Долгота", "cylinders_num": "Число цилиндров (число)",
    "miles_per_year": "Пробег за год",
}


def _plural_cylinders(n: int) -> str:
    return "цилиндра" if n in (2, 3, 4) else "цилиндров"


def label(col: str, value: str) -> str:
    """Русская подпись значения; неизвестные значения возвращаются как есть."""
    if value == UNKNOWN:
        return UNKNOWN_LABEL
    if col == "state":
        return STATES.get(value, value.upper())
    if col == "cylinders":
        m = re.match(r"(\d+)\s*cylinders", value)
        return f"{int(m.group(1))} {_plural_cylinders(int(m.group(1)))}" if m else "Другое"
    return VALUE_LABELS.get(col, {}).get(value, value.capitalize())


def pretty(text: str) -> str:
    """Читаемое название марки/модели: 'f-150' -> 'F-150', 'silverado' -> 'Silverado', 'cr-v' -> 'CR-V'."""
    exceptions = {"kia": "Kia", "geo": "Geo"}
    if text in exceptions:
        return exceptions[text]
    out = []
    for token in text.split():
        parts = []
        for p in token.split("-"):
            parts.append(p.upper() if (len(p) <= 3 or any(ch.isdigit() for ch in p)) else p.capitalize())
        out.append("-".join(parts))
    return " ".join(out)
