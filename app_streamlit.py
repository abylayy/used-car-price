"""Streamlit-приложение: оценка стоимости авто, обзор данных, информация о модели.
Запуск: streamlit run app_streamlit.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import config
import labels
import predict

st.set_page_config(page_title="Оценка стоимости автомобиля", layout="wide")

OTHER = "Другая модель…"
NAVY, TEAL, AMBER = "#14213D", "#1F7A8C", "#F2A900"     # цвета графиков, как в презентации


@st.cache_resource
def get_model():
    return predict.load_model_bundle()


@st.cache_data
def get_data():
    d = pd.read_parquet(config.CLEAN_PARQUET)
    for c in ("manufacturer", "model", "state"):
        d[c] = d[c].astype(str)
    return d


@st.cache_data
def get_catalog():
    """Что можно выбрать: марки, модели каждой марки и годы, за которые модель есть в данных.
    Списки строятся из тех же данных, на которых училась модель, чтобы нельзя было выбрать
    комбинацию, о которой модель ничего не знает (например, модель, которой в этом году не было)."""
    df = get_data()
    brand_n = df["manufacturer"].value_counts()
    brands = sorted(b for b, n in brand_n.items() if n >= 10 and b != "unknown")
    pairs = df.groupby(["manufacturer", "model"], observed=True).size().reset_index(name="n")
    models = {}
    for b in brands:
        # порог мягче для редких марок, чтобы они не пропадали из списка
        thr = 30 if brand_n[b] >= 1000 else 10 if brand_n[b] >= 200 else 3
        sub = pairs[(pairs["manufacturer"] == b) & (pairs["model"] != "unknown") & (pairs["n"] >= thr)]
        models[b] = sorted(sub.sort_values("n", ascending=False).head(150)["model"])
    ymodel = df.groupby(["manufacturer", "model", "year"], observed=True).size()
    years = {}
    for (b, m, y), n in ymodel.items():
        years.setdefault((b, m), {})[int(y)] = int(n)
    extra = {}
    for col in labels.FIELD_TITLES:
        vals = [v for v in df[col].astype(str).value_counts().index if v != labels.UNKNOWN]
        extra[col] = sorted(vals, key=lambda v: labels.label("state", v)) if col == "state" else vals
    return brands, models, years, extra


def available_years(years, brand, model):
    """Все годы от первого до последнего, что встречаются у модели в данных (пропуски внутри диапазона —
    это скорее нехватка объявлений, чем отсутствие машины). Для неизвестной модели — любые годы."""
    counts = years.get((brand, model), {})
    if not counts:
        return list(range(config.YEAR_MAX, config.YEAR_MIN - 1, -1))
    return list(range(max(counts), min(counts) - 1, -1))


st.title("Оценка стоимости автомобиля")
st.caption("Прогноз цены подержанного автомобиля по объявлениям Craigslist (США). Цены в долларах, пробег в милях.")
tab_pred, tab_data, tab_model = st.tabs(["Оценка", "Данные", "О модели"])

# ------------------------------------------------------------------ Оценка
with tab_pred:
    try:
        bundle, meta = get_model()
        brands, models_by_brand, years_by_model, extra = get_catalog()
    except FileNotFoundError as exc:
        st.error(f"{exc} Нужны обученная модель и очищенные данные: python main.py --stage all")
        st.stop()

    left, right = st.columns([3, 2], gap="large")
    with left:
        a, b = st.columns(2)
        brand = a.selectbox("Марка", brands, index=brands.index("toyota") if "toyota" in brands else 0,
                            format_func=labels.pretty)
        options = models_by_brand.get(brand, []) + [OTHER]
        choice = b.selectbox("Модель", options, key=f"model_{brand}",
                             index=options.index("camry") if "camry" in options else 0,
                             format_func=lambda m: m if m == OTHER else labels.pretty(m))
        model_name = choice
        if choice == OTHER:
            model_name = b.text_input("Название модели", key=f"custom_{brand}").strip().lower() or labels.UNKNOWN

        c, d = st.columns(2)
        year_options = available_years(years_by_model, brand, model_name)
        year_counts = years_by_model.get((brand, model_name), {})
        # по умолчанию — год, за который объявлений больше всего
        best_year = max(year_counts, key=year_counts.get) if year_counts else year_options[0]
        year = c.selectbox("Год выпуска", year_options, key=f"year_{brand}_{model_name}",
                           index=year_options.index(best_year),
                           format_func=lambda y: f"{y} · мало данных" if year_counts and year_counts.get(y, 0) < 3 else str(y))
        low_data = bool(year_counts) and year_counts.get(year, 0) < 3

        hist = get_data()
        same = hist[(hist["manufacturer"] == brand) & (hist["model"] == model_name) & (hist["year"] == year)]
        typical = int(same["odometer"].median()) if same["odometer"].notna().sum() >= 3 else 80000
        typical = min(max(typical, 0), config.ODOMETER_MAX)
        odometer = d.number_input("Пробег, миль", 0, config.ODOMETER_MAX, typical, step=5000,
                                  key=f"odo_{brand}_{model_name}_{year}",
                                  help="1 миля ≈ 1,6 км. Подставлен типичный пробег для такого авто.")
        unknown_odo = d.checkbox("Пробег неизвестен", key=f"unk_{brand}_{model_name}_{year}")

        values = {"manufacturer": brand, "model": model_name, "year": year,
                  "odometer": None if unknown_odo else odometer,
                  "region": labels.UNKNOWN, "lat": None, "long": None}
        with st.expander("Дополнительные параметры"):
            cols = st.columns(3)
            for i, col in enumerate(labels.FIELD_TITLES):
                with cols[i % 3]:
                    values[col] = st.selectbox(labels.FIELD_TITLES[col], [labels.UNKNOWN] + extra[col],
                                               format_func=lambda v, c=col: labels.label(c, v), key=f"x_{col}")

    with right:
        with st.container(border=True):
            try:
                result = predict.predict_price(bundle, values)
                st.caption(f"Примерная цена · {labels.pretty(brand)} {labels.pretty(model_name)}, {year}")
                st.markdown(f"<div style='font-size:3.4rem;font-weight:700;line-height:1.1;color:{NAVY}'>"
                            f"${result['price']:,.0f}</div>", unsafe_allow_html=True)
                st.markdown(f"Обычно такие авто стоят **${result['low']:,.0f} – ${result['high']:,.0f}**")
                asking = st.number_input("Цена в объявлении, $ (необязательно)", 0, 10 * config.PRICE_MAX, 0, step=500,
                                         help="Введите цену продавца, чтобы сравнить её с рынком.")
                if asking:
                    verdict = predict.rate_asking_price(asking, result)
                    # цвет сообщения: зелёный — ниже рынка, жёлтый — выше, синий — в пределах интервала
                    if verdict.startswith("ниже"):
                        show = st.success
                    elif verdict.startswith("выше"):
                        show = st.warning
                    else:
                        show = st.info
                    show(f"Цена ${asking:,.0f} — {verdict}.")

                # Где прогноз находится среди настоящих объявлений о таких же автомобилях
                similar = hist[(hist["manufacturer"] == brand) & (hist["model"] == model_name)
                               & hist["year"].between(year - 1, year + 1)]
                if len(similar) >= 10:
                    fig = px.histogram(similar, x="price", nbins=30, color_discrete_sequence=["#9FB3C8"],
                                       labels={"price": "Цена, $"})
                    fig.add_vrect(x0=result["low"], x1=result["high"], fillcolor=TEAL, opacity=0.15, line_width=0)
                    fig.add_vline(x=result["price"], line_color=NAVY, line_width=3, annotation_text="прогноз")
                    if asking:
                        fig.add_vline(x=asking, line_color=AMBER, line_width=3, line_dash="dash",
                                      annotation_text="объявление", annotation_position="bottom right")
                    fig.update_layout(height=240, margin=dict(l=0, r=0, t=25, b=0), yaxis_title="Объявлений")
                    st.plotly_chart(fig, width="stretch")
                    st.caption(f"Цены {len(similar):,} похожих объявлений ({year - 1}–{year + 1} гг.). "
                               f"Закрашенная полоса — интервал прогноза.")
                st.caption(f"В интервал попадает около {meta['interval']['coverage']:.0%} похожих объявлений. "
                           f"Средняя ошибка модели — ${meta['metrics']['MAE']:,.0f}, у дорогих авто она больше.")
                if low_data:
                    st.caption("Для этого года мало объявлений — оценка менее точна.")
                if unknown_odo:
                    st.caption("Без пробега оценка менее точна.")
            except ValueError as exc:
                st.error(str(exc))

# ------------------------------------------------------------------ Данные
with tab_data:
    try:
        df = get_data()
        brand_list = get_catalog()[0]
    except FileNotFoundError:
        st.warning("Очищенные данные не найдены: python main.py --stage prepare clean")
        st.stop()
    f1, f2, f3 = st.columns([2, 2, 2])
    sel = f1.multiselect("Марка", brand_list, format_func=labels.pretty, placeholder="Все марки")
    y_lo, y_hi = int(df["year"].min()), int(df["year"].max())
    years = f2.slider("Год выпуска", y_lo, y_hi, (y_lo, y_hi))
    p_lo, p_hi = f3.slider("Цена, $", config.PRICE_MIN, config.PRICE_MAX, (config.PRICE_MIN, 60000))
    states = st.multiselect("Штат", sorted(df["state"].unique(), key=lambda v: labels.label("state", v)),
                            format_func=lambda v: labels.label("state", v), placeholder="Все штаты")
    view = df[df["year"].between(*years) & df["price"].between(p_lo, p_hi)]
    if sel:
        view = view[view["manufacturer"].isin(sel)]
    if states:
        view = view[view["state"].isin(states)]
    if len(view) == 0:
        st.info("Ничего не найдено — расширьте фильтры.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Объявлений", f"{len(view):,}")
        m2.metric("Медианная цена", f"${view['price'].median():,.0f}")
        m3.metric("Медианный пробег", f"{view['odometer'].median():,.0f} миль")
        g1, g2 = st.columns(2)
        s = view.dropna(subset=["odometer"])
        s = s.sample(min(len(s), 12000), random_state=1).copy()
        s["Марка"] = s["manufacturer"].map(labels.pretty)
        g1.markdown("**Цена и пробег** · одна точка — одно объявление")
        g1.plotly_chart(px.scatter(s, x="odometer", y="price", color="Марка" if 0 < len(sel) <= 8 else None,
                                   opacity=0.5, color_discrete_sequence=[TEAL] if not 0 < len(sel) <= 8 else None,
                                   labels={"odometer": "Пробег, миль", "price": "Цена, $"}),
                        width="stretch")
        g2.markdown("**Цена по годам выпуска** · медиана")
        by_year = view.groupby("year", observed=True)["price"].median().reset_index()
        g2.plotly_chart(px.line(by_year, x="year", y="price", markers=True, color_discrete_sequence=[TEAL],
                                labels={"year": "Год выпуска", "price": "Медианная цена, $"}), width="stretch")

# ------------------------------------------------------------------ О модели
with tab_model:
    try:
        _, meta = get_model()
        m = meta["metrics"]
        a, b, c, d = st.columns(4)
        a.metric("Средняя ошибка (MAE)", f"${m['MAE']:,.0f}")
        b.metric("Половина прогнозов точнее", f"${m['MedAE']:,.0f}")
        c.metric("R²", f"{m['R2']:.2f}")
        d.metric("Цена внутри интервала", f"{meta['interval']['coverage']:.0%}")
        st.caption(f"Модель: {meta['best_model']}, обучена на {meta['n_rows_clean']:,} объявлений. "
                   f"Метрики посчитаны на объявлениях, которых модель не видела при обучении.")
    except FileNotFoundError:
        st.info("Модель ещё не обучена: python main.py --stage train")
    left, right = st.columns(2)
    path = config.RESULTS_DIR / "model_comparison.csv"
    if path.exists():
        comp = pd.read_csv(path)[["model", "MAE", "RMSE", "R2", "fit_seconds"]]
        comp.columns = ["Модель", "MAE, $", "RMSE, $", "R²", "Обучение, с"]
        left.markdown("**Сравнение моделей** · средняя ошибка на тесте, чем меньше, тем лучше")
        left.plotly_chart(px.bar(comp, x="Модель", y="MAE, $", text_auto=",.0f", color_discrete_sequence=[TEAL]),
                          width="stretch")
    imp_path = config.RESULTS_DIR / "permutation_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path).head(10).iloc[::-1]
        imp["Признак"] = imp["feature"].map(lambda f: labels.FEATURE_TITLES.get(f, f))
        right.markdown("**Что сильнее всего влияет на цену** · 10 главных признаков")
        right.plotly_chart(px.bar(imp, x="importance", y="Признак", orientation="h", color_discrete_sequence=[TEAL],
                                  labels={"importance": "Рост ошибки, если признак перемешать"}), width="stretch")
    if path.exists():
        st.dataframe(comp.round(2), hide_index=True, width="stretch")