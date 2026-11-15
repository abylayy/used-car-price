"""ТОЛЬКО ДЛЯ ПРОВЕРКИ КОДА (smoke test). Это НЕ реальные данные и НЕ источник результатов проекта.

Генерирует CSV с той же схемой, что и Craigslist vehicles.csv (включая «грязные» значения:
фиктивные цены, пропуски, дубликаты), чтобы проверить, что пайплайн запускается end-to-end.
Метрики на таких данных не имеют отношения к реальной задаче и в отчёт не попадают.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int = 60000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    brands = {"ford": ["f-150", "focus", "mustang"], "toyota": ["camry", "corolla", "tacoma"],
              "honda": ["civic", "accord"], "bmw": ["x5", "328i"], "chevrolet": ["silverado", "malibu"]}
    base = {"ford": 14000, "toyota": 15000, "honda": 13000, "bmw": 25000, "chevrolet": 14500}
    states = ["ca", "tx", "ny", "fl", "wa"]
    regions = [f"{s}_city{i}" for s in states for i in range(4)]
    b = rng.choice(list(brands), n)
    year = rng.integers(1995, 2022, n).astype(float)
    odo = np.clip(rng.normal((2021 - year) * 11000, 15000), 0, None)
    price = np.array([base[x] for x in b]) * np.exp(-0.07 * (2021 - year)) - 0.03 * odo \
        + rng.normal(0, 1500, n)
    price = np.clip(price, 300, None).round()
    reg = rng.choice(regions, n)
    df = pd.DataFrame({
        "id": np.arange(7000000000, 7000000000 + n),
        "url": "http://x", "region": reg, "region_url": "http://r", "price": price, "year": year,
        "manufacturer": b, "model": [rng.choice(brands[x]) for x in b],
        "condition": rng.choice(["good", "excellent", "fair", None], n),
        "cylinders": rng.choice(["4 cylinders", "6 cylinders", "8 cylinders", None], n),
        "fuel": rng.choice(["gas", "diesel", "hybrid", None], n, p=[.8, .1, .05, .05]),
        "odometer": odo.round(), "title_status": rng.choice(["clean", "rebuilt", None], n, p=[.9, .05, .05]),
        "transmission": rng.choice(["automatic", "manual", "other"], n, p=[.8, .15, .05]),
        "VIN": [None if rng.random() < .6 else "".join(rng.choice(list("ABCDEFGH123456789"), 17))
                for _ in range(n)],
        "drive": rng.choice(["fwd", "rwd", "4wd", None], n), "size": None,
        "type": rng.choice(["sedan", "truck", "SUV", None], n),
        "paint_color": rng.choice(["black", "white", "red", None], n),
        "image_url": "http://i", "description": "text", "county": None,
        "state": [r.split("_")[0] for r in reg],
        "lat": rng.uniform(25, 48, n), "long": rng.uniform(-123, -70, n),
        "posting_date": pd.Timestamp("2021-04-20", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    # «Грязь»: фиктивные цены, пропуски, заглушки, дубликаты, нелепый пробег
    idx = rng.choice(n, 600, replace=False)
    df.loc[idx[:200], "price"] = rng.choice([0, 1, 123456, 1234567, 99999999], 200)
    df.loc[idx[200:300], "odometer"] = 9999999
    df.loc[idx[300:400], "odometer"] = np.nan
    df.loc[idx[400:450], "year"] = np.nan
    df = pd.concat([df, df.sample(1500, random_state=1)], ignore_index=True)  # дубликаты (одинаковый VIN/ключ)
    return df


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/vehicles_synthetic.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(out, index=False)
    print(f"Синтетический файл (только для проверки кода): {out}")
