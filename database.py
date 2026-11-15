"""SQLite: нормализованная схема, загрузка данных, SQL-запросы, ER-диаграмма.

Схема в третьей нормальной форме (3NF):
manufacturers 1—N models 1—N listings N—1 regions.
"""
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

DDL = """
PRAGMA foreign_keys = ON;
CREATE TABLE manufacturers (
    manufacturer_id INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE
);
CREATE TABLE models (
    model_id        INTEGER PRIMARY KEY,
    manufacturer_id INTEGER NOT NULL REFERENCES manufacturers(manufacturer_id),
    model_name      TEXT NOT NULL,
    UNIQUE (manufacturer_id, model_name)
);
CREATE TABLE regions (
    region_id   INTEGER PRIMARY KEY,
    region_name TEXT NOT NULL,
    state       TEXT NOT NULL,
    UNIQUE (region_name, state)
);
CREATE TABLE listings (
    listing_id   INTEGER PRIMARY KEY,
    model_id     INTEGER NOT NULL REFERENCES models(model_id),
    region_id    INTEGER NOT NULL REFERENCES regions(region_id),
    price        REAL    NOT NULL,
    year         INTEGER NOT NULL,
    odometer     REAL,
    condition    TEXT,
    cylinders    TEXT,
    fuel         TEXT,
    title_status TEXT,
    transmission TEXT,
    drive        TEXT,
    type         TEXT,
    paint_color  TEXT,
    lat          REAL,
    long         REAL,
    posting_date TEXT
);
CREATE INDEX idx_listings_model  ON listings(model_id);
CREATE INDEX idx_listings_region ON listings(region_id);
CREATE INDEX idx_listings_year   ON listings(year);
CREATE INDEX idx_listings_price  ON listings(price);
"""


def _txt(s: pd.Series) -> pd.Series:
    return s.astype("object")


def build_database(df: pd.DataFrame, db_path: Path = config.DB_PATH, chunksize: int = 50_000) -> dict:
    """Создаёт БД с нуля и загружает очищенные данные. Возвращает число строк по таблицам."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    d = df.copy()
    d["manufacturer"] = _txt(d["manufacturer"])
    d["model"] = _txt(d["model"])
    d["region"] = _txt(d["region"])
    d["state"] = _txt(d["state"])

    manuf = pd.DataFrame({"name": sorted(d["manufacturer"].unique())})
    manuf["manufacturer_id"] = range(1, len(manuf) + 1)
    man_map = dict(zip(manuf["name"], manuf["manufacturer_id"]))

    models = d[["manufacturer", "model"]].drop_duplicates().reset_index(drop=True)
    models["manufacturer_id"] = models["manufacturer"].map(man_map)
    models["model_id"] = range(1, len(models) + 1)
    models = models.rename(columns={"model": "model_name"})
    model_map = {(r.manufacturer, r.model_name): r.model_id for r in models.itertuples()}

    regions = d[["region", "state"]].drop_duplicates().reset_index(drop=True)
    regions["region_id"] = range(1, len(regions) + 1)
    regions = regions.rename(columns={"region": "region_name"})
    region_map = {(r.region_name, r.state): r.region_id for r in regions.itertuples()}

    listings = pd.DataFrame({
        "listing_id": d["id"].astype("int64") if (d["id"].notna().all() and d["id"].is_unique)
                      else range(1, len(d) + 1),
        "model_id": [model_map[k] for k in zip(d["manufacturer"], d["model"])],
        "region_id": [region_map[k] for k in zip(d["region"], d["state"])],
        "price": d["price"].astype(float), "year": d["year"].astype(int),
        "odometer": d["odometer"].astype(float),
        "posting_date": d["posting_date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "lat": d["lat"].astype(float), "long": d["long"].astype(float),
    })
    for col in ["condition", "cylinders", "fuel", "title_status", "transmission",
                "drive", "type", "paint_color"]:
        listings[col] = _txt(d[col]).values

    with closing(sqlite3.connect(db_path)) as con:
        con.executescript(DDL)
        manuf[["manufacturer_id", "name"]].to_sql("manufacturers", con, if_exists="append", index=False)
        models[["model_id", "manufacturer_id", "model_name"]].to_sql(
            "models", con, if_exists="append", index=False, chunksize=chunksize)
        regions[["region_id", "region_name", "state"]].to_sql(
            "regions", con, if_exists="append", index=False)
        listings.to_sql("listings", con, if_exists="append", index=False, chunksize=chunksize)
        con.commit()
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("manufacturers", "models", "regions", "listings")}
    return counts


def load_queries(sql_path: Path = config.SQL_FILE) -> dict:
    """Читает sql/queries.sql в словарь {имя: {'desc': ..., 'sql': ...}}."""
    text = sql_path.read_text(encoding="utf-8")
    queries = {}
    for block in re.split(r"^-- name:\s*", text, flags=re.M)[1:]:
        name, _, rest = block.partition("\n")
        desc_match = re.search(r"^-- desc:\s*(.*)$", rest, flags=re.M)
        sql = "\n".join(line for line in rest.splitlines() if not line.startswith("-- desc:")).strip()
        queries[name.strip()] = {"desc": desc_match.group(1) if desc_match else "", "sql": sql}
    return queries


def run_query(db_path: Path, sql: str) -> pd.DataFrame:
    """Выполняет запрос и возвращает результат как DataFrame (pd.read_sql_query)."""
    with closing(sqlite3.connect(db_path)) as con:
        return pd.read_sql_query(sql, con)


def run_all_queries(db_path: Path = config.DB_PATH, save_dir: Path = config.SQL_RESULTS_DIR) -> tuple:
    """Выполняет все запросы, сохраняет результаты в CSV. Возвращает (результаты, сводка)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    results, summary = {}, []
    for name, q in load_queries().items():
        t0 = time.perf_counter()
        try:
            res = run_query(db_path, q["sql"])
        except sqlite3.Error as exc:  # ошибка в запросе не должна ронять весь пайплайн
            summary.append({"query": name, "desc": q["desc"], "rows": None,
                            "seconds": None, "error": str(exc)})
            continue
        results[name] = res
        res.to_csv(save_dir / f"{name}.csv", index=False)
        summary.append({"query": name, "desc": q["desc"], "rows": len(res),
                        "seconds": round(time.perf_counter() - t0, 3), "error": None})
    return results, summary


def compare_pandas_vs_sql(df: pd.DataFrame, db_path: Path = config.DB_PATH, repeats: int = 3) -> list:
    """Сравнение Pandas и SQL на одной задаче: средняя цена по маркам (n >= 100). Время в секундах."""
    sql = ("SELECT m.name, COUNT(*) n, AVG(l.price) avg_price FROM listings l "
           "JOIN models md ON md.model_id=l.model_id "
           "JOIN manufacturers m ON m.manufacturer_id=md.manufacturer_id "
           "GROUP BY m.name HAVING COUNT(*)>=100")
    times_sql, times_pd = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r_sql = run_query(db_path, sql)
        times_sql.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        g = df.groupby("manufacturer", observed=True)["price"].agg(["count", "mean"])
        r_pd = g[g["count"] >= 100]
        times_pd.append(time.perf_counter() - t0)
    return [
        {"method": "SQL (SQLite, JOIN + GROUP BY)", "seconds": round(sum(times_sql) / repeats, 4),
         "result_rows": len(r_sql)},
        {"method": "Pandas (groupby)", "seconds": round(sum(times_pd) / repeats, 4),
         "result_rows": len(r_pd)},
    ]


def draw_er_diagram(path: Path = config.FIGURES_DIR / "er_diagram.png") -> Path:
    """Рисует ER-диаграмму схемы БД средствами matplotlib (не требует Graphviz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = {
        "manufacturers": (0.2, 5.6, ["PK manufacturer_id", "name"]),
        "models": (4.2, 5.4, ["PK model_id", "FK manufacturer_id", "model_name"]),
        "listings": (8.6, 0.6, ["PK listing_id", "FK model_id", "FK region_id", "price", "year",
                                "odometer", "condition", "cylinders", "fuel", "title_status",
                                "transmission", "drive", "type", "paint_color",
                                "lat, long", "posting_date"]),
        "regions": (4.2, 1.2, ["PK region_id", "region_name", "state"]),
    }
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 9.2); ax.axis("off")
    boxes = {}
    for name, (x, y, cols) in tables.items():
        h = 0.45 * (len(cols) + 1)
        w = 3.4
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, fc="#f4f8ff", ec="#2b5797", lw=1.5))
        ax.add_patch(plt.Rectangle((x, y + h - 0.45), w, 0.45, fc="#2b5797", ec="#2b5797"))
        ax.text(x + w / 2, y + h - 0.225, name, ha="center", va="center", color="white",
                fontweight="bold")
        for i, c in enumerate(cols):
            ax.text(x + 0.12, y + h - 0.45 - 0.45 * (i + 0.5), c, va="center", fontsize=9,
                    fontweight="bold" if c.startswith(("PK", "FK")) else "normal")
        boxes[name] = (x, y, w, h)

    def link(a, b, xa, ya, xb, yb):
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="->", color="#b03a2e", lw=1.6))

    mx, my, mw, mh = boxes["manufacturers"]; dx, dy, dw, dh = boxes["models"]
    lx, ly, lw, lh = boxes["listings"]; rx, ry, rw, rh = boxes["regions"]
    link("m", "d", mx + mw, my + mh / 2, dx, dy + dh / 2)           # manufacturers 1 -> N models
    ax.text((mx + mw + dx) / 2, my + mh / 2 + 0.2, "1 : N", ha="center", fontsize=9)
    link("d", "l", dx + dw, dy + dh / 2, lx, ly + lh - 1.0)         # models 1 -> N listings
    ax.text((dx + dw + lx) / 2, dy + dh / 2 + 0.4, "1 : N", ha="center", fontsize=9)
    link("r", "l", rx + rw, ry + rh / 2, lx, ly + 1.2)              # regions 1 -> N listings
    ax.text((rx + rw + lx) / 2, ry + rh / 2 + 0.6, "1 : N", ha="center", fontsize=9)
    ax.set_title("ER-диаграмма базы данных cars.db (SQLite)", fontsize=13, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path