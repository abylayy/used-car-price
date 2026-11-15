"""Исследовательский анализ данных (EDA): статистики и визуализации."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

import config
from features import add_derived_features

sns.set_theme(style="whitegrid")


def _save(fig, name: str, fig_dir=config.FIGURES_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Описательная статистика числовых признаков + асимметрия цены."""
    num = df.select_dtypes("number")
    stats = num.describe().T
    stats["skew"] = num.skew()
    return stats


def run_eda(df: pd.DataFrame, sql_results: dict = None, raw_missing: dict = None, raw: pd.DataFrame = None) -> dict:
    """Строит все графики EDA. Возвращает {'figures': [...], 'stats': ...}."""
    sql_results = sql_results or {}
    d = add_derived_features(df)
    figures = []
    sample = d.sample(min(len(d), 30000), random_state=config.RANDOM_STATE)

    # 1. Распределение цены (исходная и логарифмическая шкала) — целевая переменная
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(d["price"], bins=80, color="#2b5797")
    ax[0].set(title="Распределение цены (правый хвост)", xlabel="Цена, $", ylabel="Число объявлений")
    ax[1].hist(np.log1p(d["price"]), bins=80, color="#1e8449")
    ax[1].set(title="Распределение log(1+цена): ближе к симметричному",
              xlabel="log(1+цена)", ylabel="Число объявлений")
    figures.append(_save(fig, "01_price_distribution.png"))

    # 2. Пропуски в сырых данных (до очистки)
    if raw_missing:
        m = pd.Series(raw_missing).sort_values(ascending=True)
        m = m[m > 0]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(m.index, m.values, color="#b03a2e")
        ax.set(title="Доля пропусков по столбцам (сырые данные)", xlabel="Пропуски, %")
        figures.append(_save(fig, "02_missing_values.png"))

    # 3. Цена по возрасту автомобиля (boxplot, без показа выбросов)
    fig, ax = plt.subplots(figsize=(12, 5))
    ages = sample[sample["car_age"] <= 25]
    sns.boxplot(data=ages, x=ages["car_age"].astype(int), y="price", showfliers=False, ax=ax,
                color="#7fb3d5")
    ax.set(title="Цена в зависимости от возраста автомобиля", xlabel="Возраст, лет", ylabel="Цена, $")
    figures.append(_save(fig, "03_price_by_age.png"))

    # 4. Цена и пробег
    fig, ax = plt.subplots(figsize=(8, 6))
    s = sample.dropna(subset=["odometer"])
    s = s[s["odometer"] < 400000]
    hb = ax.hexbin(s["odometer"], s["price"], gridsize=45, cmap="viridis", bins="log", mincnt=1)
    fig.colorbar(hb, label="log10(число объявлений)")
    ax.set(title="Цена и пробег", xlabel="Пробег, миль", ylabel="Цена, $")
    figures.append(_save(fig, "04_price_vs_odometer.png"))

    # 5. Медианная цена по маркам (топ-15 по числу объявлений)
    top = d["manufacturer"].value_counts().drop("unknown", errors="ignore").head(15).index
    med = d[d["manufacturer"].isin(top)].groupby("manufacturer")["price"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(med.index, med.values, color="#2b5797")
    ax.set(title="Медианная цена по маркам (15 самых частых)", xlabel="Медианная цена, $")
    figures.append(_save(fig, "05_median_price_by_brand.png"))

    # 6. Корреляции числовых признаков
    cols = ["price", "car_age", "odometer", "cylinders_num", "miles_per_year", "lat", "long"]
    corr = d[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Корреляция числовых признаков (Пирсон)")
    figures.append(_save(fig, "06_correlation_heatmap.png"))

    # 7. Цена по категориальным признакам
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, col in zip(axes, ["fuel", "drive", "condition"]):
        order = d[col].value_counts().head(8).index
        sns.boxplot(data=d[d[col].isin(order)].sample(min(len(d), 30000), random_state=1),
                    x=col, y="price", order=order, showfliers=False, ax=ax, color="#a9cce3")
        ax.set(title=f"Цена по признаку «{col}»", xlabel=col, ylabel="Цена, $")
        ax.tick_params(axis="x", rotation=30)
    figures.append(_save(fig, "07_price_by_categories.png"))

    # 8. Графики по результатам SQL-запросов (SQL используется в анализе)
    if "q02_price_by_year" in sql_results:
        q = sql_results["q02_price_by_year"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(q["year"], q["avg_price"], marker="o")
        ax.set(title="Средняя цена по году выпуска (SQL-запрос q02)", xlabel="Год выпуска",
               ylabel="Средняя цена, $")
        figures.append(_save(fig, "08_sql_price_by_year.png"))
    if "q09_odometer_bucket_moving_avg" in sql_results:
        q = sql_results["q09_odometer_bucket_moving_avg"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(q["odo_bucket"], q["avg_price"], marker="o", label="средняя цена")
        ax.plot(q["odo_bucket"], q["moving_avg_price"], lw=3, label="скользящее среднее (окно 3)")
        ax.set(title="Цена по корзинам пробега (SQL-запрос q09)", xlabel="Пробег (начало корзины), миль",
               ylabel="Цена, $")
        ax.legend()
        figures.append(_save(fig, "09_sql_price_by_odometer.png"))

    # 9. Интерактивный график Plotly (HTML)
    inter = sample.dropna(subset=["odometer"]).head(15000)
    inter = inter[inter["manufacturer"].isin(top)]
    fig_i = px.scatter(inter, x="odometer", y="price", color="manufacturer",
                       hover_data=["model", "year", "condition"], opacity=0.5,
                       title="Интерактивно: цена, пробег и марка (выборка)",
                       labels={"odometer": "Пробег, миль", "price": "Цена, $", "manufacturer": "Марка"})
    html_path = config.FIGURES_DIR / "10_interactive_price_odometer.html"
    fig_i.write_html(html_path, include_plotlyjs="cdn")
    figures.append(html_path)

    # 10. Выбросы в СЫРЫХ данных (до очистки): boxplot цены и пробега в логарифмической шкале
    if raw is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
        for ax, col, title in zip(axes, ["price", "odometer"], ["Цена, $", "Пробег, миль"]):
            vals = np.log10(raw[col].dropna().astype(float).clip(lower=1))
            ax.boxplot(vals, vert=False, flierprops={"markersize": 2, "alpha": 0.3})
            ticks = [t for t in range(0, int(vals.max()) + 2)]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{10 ** t:,.0f}" for t in ticks])
            ax.set(title=f"Выбросы до очистки: {title} (логарифмическая шкала)", xlabel=title, yticks=[])
        figures.append(_save(fig, "11_outliers_raw.png"))

    findings = eda_findings(d, sql_results, raw_missing, raw)
    return {"figures": figures, "stats": descriptive_stats(d), "findings": findings}


def eda_findings(d: pd.DataFrame, sql_results: dict, raw_missing: dict, raw: pd.DataFrame = None) -> list:
    """Выводы по EDA, посчитанные из данных: каждое предложение содержит фактические числа.
    Сохраняются в outputs/results/eda_findings.json|md и попадают в отчёт и презентацию."""
    items = []

    def add(topic, text):
        items.append({"topic": topic, "text": text})

    p = d["price"]
    skew = p.skew()
    add("Распределение цены",
        f"Асимметрия цены равна {skew:.2f}; медиана ${p.median():,.0f}, среднее ${p.mean():,.0f}"
        + (", то есть распределение имеет тяжёлый правый хвост" if skew > 1 else "")
        + f". После логарифмирования асимметрия равна {np.log1p(p).skew():.2f}, поэтому модель обучается на log(1+цена).")
    corr = d[["price", "car_age", "odometer"]].corr(method="spearman")["price"]
    add("Связь цены с возрастом и пробегом",
        f"Ранговая корреляция Спирмена цены с возрастом автомобиля {corr['car_age']:.2f}, с пробегом {corr['odometer']:.2f}; "
        f"связь с {'возрастом' if abs(corr['car_age']) >= abs(corr['odometer']) else 'пробегом'} сильнее.")
    age_bin = pd.cut(d["car_age"], [-1, 3, 7, 12, 20, 100], labels=["0–3", "4–7", "8–12", "13–20", ">20"])
    med = d.groupby(age_bin, observed=True)["price"].median()
    add("Цена по возрасту", "Медианная цена по возрасту автомобиля (лет): "
        + "; ".join(f"{k} — ${v:,.0f}" for k, v in med.items()) + ".")
    top = d["manufacturer"].value_counts().drop("unknown", errors="ignore").head(15).index
    bm = d[d["manufacturer"].isin(top)].groupby("manufacturer")["price"].median().sort_values()
    add("Марки", f"Среди 15 самых частых марок самая высокая медианная цена у {bm.index[-1]} (${bm.iloc[-1]:,.0f}), "
                 f"самая низкая у {bm.index[0]} (${bm.iloc[0]:,.0f}).")
    for col, name in [("fuel", "топлива"), ("drive", "привода"), ("condition", "состояния"), ("type", "типа кузова")]:
        counts = d[col].value_counts()
        ok = [v for v in counts[counts >= 500].index if v != "unknown"]
        if len(ok) >= 2:
            m = d[d[col].isin(ok)].groupby(col)["price"].median().sort_values()
            add(f"Цена по признаку «{col}»", f"Среди значений с n ≥ 500 наибольшая медианная цена у «{m.index[-1]}» "
                                  f"(${m.iloc[-1]:,.0f}), наименьшая у «{m.index[0]}» (${m.iloc[0]:,.0f}).")
    if raw_missing:
        ms = pd.Series(raw_missing)
        ms = ms[(ms > 0) & (ms < 100)].sort_values(ascending=False).head(3)
        if len(ms):
            add("Пропуски", "Больше всего пропусков в сырых данных в столбцах: "
                + ", ".join(f"{k} ({v:.1f}%)" for k, v in ms.items()) + ".")
    if raw is not None:
        lo = (raw["price"] < config.PRICE_MIN).mean()
        hi = (raw["price"] > config.PRICE_MAX).mean()
        add("Выбросы", f"В сырых данных {lo:.1%} цен ниже ${config.PRICE_MIN} и {hi:.2%} выше ${config.PRICE_MAX:,}; "
                       f"эти значения вынесены за доменные границы и удалены при очистке (график 11).")
    q3 = sql_results.get("q03_state_price_rank")
    if q3 is not None and len(q3) >= 2:
        q3 = q3.sort_values("avg_price")
        add("Штаты (SQL q03)", f"По SQL-запросу q03 самая высокая средняя цена в штате {q3.iloc[-1]['state']} "
                               f"(${q3.iloc[-1]['avg_price']:,.0f}), самая низкая — в {q3.iloc[0]['state']} (${q3.iloc[0]['avg_price']:,.0f}).")
    q11 = sql_results.get("q11_brand_pareto")
    if q11 is not None and len(q11):
        reach = q11[q11["cumulative_pct"] >= 80]
        add("Концентрация по маркам (SQL q11)",
            (f"На {int(reach.index[0]) + 1} самых частых марок приходится 80% объявлений с известной маркой." if len(reach)
             else f"{len(q11)} самых частых марок дают {q11['cumulative_pct'].iloc[-1]:.0f}% объявлений."))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "eda_findings.json", "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)
    (config.RESULTS_DIR / "eda_findings.md").write_text(
        "\n".join(f"- **{i['topic']}.** {i['text']}" for i in items) + "\n", encoding="utf-8")
    return items