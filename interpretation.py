"""Интерпретация модели и анализ ошибок: важность признаков, остатки, ошибки по сегментам."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config


def permutation_importance_report(pipeline, X_test, y_test, n_samples: int = config.PERMUTATION_SAMPLE) -> pd.DataFrame:
    """Перестановочная важность (Permutation Importance) исходных признаков.

    Перемешиваем один исходный столбец и смотрим, насколько растёт RMSE (в лог-шкале).
    Считаем на подвыборке теста, чтобы было быстро.
    """
    idx = X_test.sample(min(n_samples, len(X_test)), random_state=config.RANDOM_STATE).index
    res = permutation_importance(pipeline, X_test.loc[idx], y_test.loc[idx], n_repeats=5,
                                 random_state=config.RANDOM_STATE,
                                 scoring="neg_root_mean_squared_error", n_jobs=1)
    out = pd.DataFrame({"feature": X_test.columns, "importance": res.importances_mean,
                        "std": res.importances_std})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def build_predictions(pipeline, X_test, y_test) -> pd.DataFrame:
    """Таблица предсказаний на тесте (в долларах) для анализа ошибок."""
    pred = np.clip(np.expm1(pipeline.predict(X_test)), 0, None)
    out = X_test.copy()
    out["price_true"] = np.expm1(y_test).values
    out["price_pred"] = pred
    out["error"] = out["price_pred"] - out["price_true"]
    out["abs_error"] = out["error"].abs()
    out["ape_pct"] = 100 * out["abs_error"] / out["price_true"]
    return out


def error_analysis(pred_df: pd.DataFrame) -> dict:
    """Ошибки по ценовым диапазонам, возрасту и маркам; худшие предсказания."""
    d = pred_df.copy()
    d["price_bin"] = pd.cut(d["price_true"], [0, 5000, 10000, 20000, 40000, 1e9],
                            labels=["<5k", "5–10k", "10–20k", "20–40k", ">40k"])
    d["age_bin"] = pd.cut(d["car_age"], [-1, 3, 7, 12, 20, 100],
                          labels=["0–3", "4–7", "8–12", "13–20", ">20"])

    def agg(col):
        g = d.groupby(col, observed=True).agg(n=("abs_error", "size"), MAE=("abs_error", "mean"),
                                              MAPE_pct=("ape_pct", "mean"), bias=("error", "mean"))
        return g.reset_index()

    by_brand = agg("manufacturer")
    by_brand = by_brand[by_brand["n"] >= 100].sort_values("MAE", ascending=False).head(15)
    worst = d.sort_values("abs_error", ascending=False).head(30)[
        ["manufacturer", "model", "car_age", "odometer", "condition", "price_true", "price_pred", "error"]]
    return {"by_price": agg("price_bin"), "by_age": agg("age_bin"),
            "by_brand_worst": by_brand, "worst_predictions": worst}


def plot_diagnostics(pred_df: pd.DataFrame, importance: pd.DataFrame, model_name: str, fig_dir=config.FIGURES_DIR):
    """Графики: факт vs прогноз, остатки, гистограмма ошибок, важность признаков."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    s = pred_df.sample(min(len(pred_df), 15000), random_state=config.RANDOM_STATE)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    lim = float(s["price_true"].quantile(0.995))
    axes[0].scatter(s["price_true"], s["price_pred"], s=4, alpha=0.3)
    axes[0].plot([0, lim], [0, lim], "r--", label="идеальный прогноз")
    axes[0].set(xlim=(0, lim), ylim=(0, lim), xlabel="Фактическая цена, $", ylabel="Прогноз, $",
                title=f"Факт vs прогноз ({model_name})")
    axes[0].legend()
    axes[1].scatter(s["price_pred"], s["error"], s=4, alpha=0.3)
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set(xlabel="Прогноз, $", ylabel="Остаток (прогноз − факт), $", title="График остатков",
                xlim=(0, lim))
    axes[2].hist(s["error"].clip(-20000, 20000), bins=60, color="#2b5797")
    axes[2].set(xlabel="Ошибка, $ (обрезано по ±20 000)", ylabel="Число объявлений",
                title="Распределение ошибок")
    fig.tight_layout()
    paths.append(fig_dir / "residuals.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#2b5797")
    ax.set(xlabel="Рост RMSE (лог-шкала) при перемешивании признака",
           title="Важность признаков (Permutation Importance)")
    fig.tight_layout()
    paths.append(fig_dir / "permutation_importance.png")
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)
    return paths


def generate_findings(results: pd.DataFrame, importance: pd.DataFrame, errors: dict,
                      interval_quality: dict = None) -> dict:
    """Формирует выводы из таблиц результатов: каждый пункт — предложение с подставленными числами,
    поэтому текст выводов не расходится с последним запуском.
    """
    items = []

    def add(topic: str, text: str):
        items.append({"topic": topic, "text": text})

    base = results[results["model"] == "baseline_median"].iloc[0]
    cand = results[results["model"] != "baseline_median"]
    best = cand.loc[cand["cv_rmse_log"].idxmin()]
    gain = 100 * (base["MAE"] - best["MAE"]) / base["MAE"]
    add("Лучшая модель",
        f"По кросс-валидации лучшей оказалась модель {best['model']}: на тесте MAE = ${best['MAE']:,.0f}, "
        f"RMSE = ${best['RMSE']:,.0f}, R² = {best['R2']:.3f}. Baseline (медиана) даёт MAE = ${base['MAE']:,.0f}, "
        f"то есть улучшение по MAE составляет {gain:.1f}%.")
    top_test = cand.loc[cand["MAE"].idxmin()]
    if top_test["model"] != best["model"]:
        add("Выбор по CV и по тесту",
            f"Наименьшую MAE на тесте показала модель {top_test['model']} (${top_test['MAE']:,.0f}), "
            f"но выбор делался по CV ({best['model']}), чтобы не подгонять модель под тест.")

    if "ridge" in set(cand["model"]):
        ridge = cand[cand["model"] == "ridge"].iloc[0]
        nonlin = cand[cand["model"] != "ridge"]
        if len(nonlin):
            nl = nonlin.loc[nonlin["MAE"].idxmin()]
            diff = 100 * (ridge["MAE"] - nl["MAE"]) / ridge["MAE"]
            verb = "снизила" if diff > 0 else "не снизила (увеличила)"
            add("Линейная модель и ансамбли",
                f"Лучшая нелинейная модель ({nl['model']}) {verb} MAE относительно линейной Ridge "
                f"на {abs(diff):.1f}% (${nl['MAE']:,.0f} против ${ridge['MAE']:,.0f}).")

    fast, slow = cand.loc[cand["fit_seconds"].idxmin()], cand.loc[cand["fit_seconds"].idxmax()]
    big = cand.loc[cand["model_size_mb"].idxmax()]
    add("Время и размер",
        f"Быстрее всех обучилась {fast['model']} ({fast['fit_seconds']:.1f} с), дольше всех — {slow['model']} "
        f"({slow['fit_seconds']:.1f} с). Самая тяжёлая модель по размеру файла — {big['model']} "
        f"({big['model_size_mb']:.1f} МБ).")

    near = cand[cand["MAE"] <= cand["MAE"].min() * 1.02]
    if len(near) >= 2:
        f2, s2 = near.loc[near["fit_seconds"].idxmin()], near.loc[near["model_size_mb"].idxmin()]
        p2 = near.loc[near["predict_ms_per_1000"].idxmin()]
        add("Обоснование выбора",
            f"Модели, у которых MAE не более чем на 2% хуже лучшей ({', '.join(near['model'])}), близки по качеству; "
            f"среди них быстрее всех обучается {f2['model']} ({f2['fit_seconds']:.1f} с), компактнее всех — "
            f"{s2['model']} ({s2['model_size_mb']:.1f} МБ), быстрее всех предсказывает {p2['model']} "
            f"({p2['predict_ms_per_1000']:.0f} мс на 1000 объявлений). Поэтому выбор между ними определяется "
            f"требованиями к скорости и размеру, а не только качеством.")

    pos = importance[importance["importance"] > 0]
    if len(pos) >= 3:
        top3 = importance.head(3)
        share = 100 * top3["importance"].sum() / pos["importance"].sum()
        names = ", ".join(f"{r.feature} ({r.importance:.3f})" for r in top3.itertuples())
        add("Главные факторы цены",
            f"Наибольшее влияние на прогноз оказывают признаки: {names}. Вместе они дают {share:.0f}% "
            f"суммарной положительной важности (рост RMSE в лог-шкале при перемешивании признака).")

    bp = errors["by_price"]
    if len(bp):
        worst_pct = bp.loc[bp["MAPE_pct"].idxmax()]
        worst_mae = bp.loc[bp["MAE"].idxmax()]
        add("Ошибки по цене",
            f"Наибольшая относительная ошибка (MAPE) — в ценовом диапазоне «{worst_pct['price_bin']}» "
            f"({worst_pct['MAPE_pct']:.0f}%), наибольшая абсолютная (MAE) — в диапазоне «{worst_mae['price_bin']}» "
            f"(${worst_mae['MAE']:,.0f}).")
        for r in bp.itertuples():
            if r.n >= 50 and abs(r.bias) > 0.1 * r.MAE:
                direction = "завышает" if r.bias > 0 else "занижает"
                add(f"Смещение: {r.price_bin}",
                    f"В диапазоне «{r.price_bin}» модель в среднем {direction} цену на ${abs(r.bias):,.0f}.")
    ba = errors["by_age"]
    if len(ba):
        w = ba.loc[ba["MAPE_pct"].idxmax()]
        add("Ошибки по возрасту",
            f"Наибольшая относительная ошибка — для автомобилей возраста «{w['age_bin']}» лет "
            f"({w['MAPE_pct']:.0f}%, n = {int(w['n'])}).")
    bb = errors["by_brand_worst"]
    if len(bb):
        w = bb.iloc[0]
        add("Ошибки по маркам",
            f"Среди марок с n ≥ 100 наибольшая MAE у {w['manufacturer']} (${w['MAE']:,.0f}, n = {int(w['n'])}).")

    if interval_quality:
        q = interval_quality
        add("Интервал цены",
            f"Интервал квантильной регрессии должен накрывать {q['target_coverage']:.0%} объявлений; на тесте внутри "
            f"него оказалось {q['coverage']:.1%} (ниже интервала {q['share_below']:.1%}, выше {q['share_above']:.1%}). "
            f"Медианная ширина интервала — ${q['median_width_usd']:,.0f} "
            f"({q['median_width_pct_of_price']:.0f}% от цены автомобиля).")

    out = {"items": items, "source_files": ["model_comparison.csv", "permutation_importance.csv",
                                            "error_by_price.csv", "error_by_age.csv",
                                            "error_by_brand_worst.csv"]}
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "findings.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    md = "\n".join(f"- **{i['topic']}.** {i['text']}" for i in items)
    (config.RESULTS_DIR / "findings.md").write_text(md + "\n", encoding="utf-8")
    return out


def regenerate_findings() -> None:
    """Пересчитывает выводы из уже сохранённых таблиц (model_comparison.csv и др.) без переобучения."""
    r = config.RESULTS_DIR
    needed = ["model_comparison.csv", "permutation_importance.csv", "error_by_price.csv",
              "error_by_age.csv", "error_by_brand_worst.csv"]
    if not all((r / n).exists() for n in needed):
        return
    interval_quality = None
    if (r / "interval_quality.json").exists():
        with open(r / "interval_quality.json", encoding="utf-8") as f:
            interval_quality = json.load(f)
    generate_findings(pd.read_csv(r / "model_comparison.csv"), pd.read_csv(r / "permutation_importance.csv"),
                      {"by_price": pd.read_csv(r / "error_by_price.csv"), "by_age": pd.read_csv(r / "error_by_age.csv"),
                       "by_brand_worst": pd.read_csv(r / "error_by_brand_worst.csv")}, interval_quality)
