"""Генерация презентации PPTX (15 слайдов).

Числа, таблицы и рисунки берутся из файлов, созданных запуском проекта (outputs/, models/),
а пороги очистки и список моделей — из config.py и modeling.py, поэтому презентация
не расходится с кодом. Если какого-то результата ещё нет, на его месте рисуется рамка
с названием файла-источника. Пересборка: python main.py --stage presentation
"""
import json

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import config
import modeling

NAVY, INK, MUTED = "14213D", "1B2430", "5B6573"
AMBER, AMBER_BG, TEAL = "F2A900", "FFF6DD", "1F7A8C"
PAPER, WHITE, LINE = "F5F7FA", "FFFFFF", "D9DEE5"
FONT = "Calibri"
SW, SH = 13.333, 7.5
TOTAL = 15


def rgb(h):
    return RGBColor.from_string(h)


# ---------------------------------------------------------------- загрузка результатов
def _json(name):
    p = config.RESULTS_DIR / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def _csv(name):
    p = config.RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


class Results:
    """Всё, что уже посчитано запуском проекта (None — этап ещё не выполнялся)."""
    def __init__(self):
        self.audit = _json("audit_raw.json")
        self.bench = _json("loading_benchmark.json")
        self.clean = _json("cleaning_log.json")
        self.db = _json("db_summary.json")
        self.findings = _json("findings.json")
        self.eda = _json("eda_findings.json")
        self.comp = _csv("model_comparison.csv")
        self.meta = json.load(open(config.MODEL_META_PATH, encoding="utf-8")) \
            if config.MODEL_META_PATH.exists() else None

    def fig(self, name):
        p = config.FIGURES_DIR / name
        return p if p.exists() else None


# ---------------------------------------------------------------- примитивы
def add_text(slide, text, x, y, w, h, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, name=None, spacing=0):
    """text — строка или список абзацев; абзац — строка или (строка, {опции})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    paras = text if isinstance(text, list) else [text]
    expanded = []                      # "\n" внутри строки -> отдельные абзацы (ровное выравнивание)
    for item in paras:
        t, o = (item, {}) if isinstance(item, str) else item
        expanded.extend((part, o) for part in t.split("\n"))
    paras = expanded
    for i, item in enumerate(paras):
        t, o = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = o.get("align", align)
        if spacing:
            p.space_after = Pt(spacing)
        r = p.add_run()
        r.text = t
        r.font.name = FONT
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("bold", bold)
        r.font.color.rgb = rgb(o.get("color", color))
    return tb


def add_shape(slide, kind, x, y, w, h, fill=WHITE, line=LINE, name=None, radius=0.08, lw=1.0):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(lw)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def card(slide, x, y, w, h, fill=WHITE, name=None):
    return add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill, name=name)


def badge(slide, x, y, label, d=0.5, fill=AMBER, color=NAVY, size=16):
    c = add_shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill=fill, line=None)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = p.add_run()
    r.text = str(label)
    r.font.name, r.font.size, r.font.bold = FONT, Pt(size), True
    r.font.color.rgb = rgb(color)
    return c


def placeholder(slide, name, x, y, w, h, what, source, size=13):
    """Заметная рамка-заглушка: что вставить и откуда взять данные."""
    s = add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=AMBER_BG, line=AMBER, name=name, lw=1.75)
    s.line.dash_style = MSO_LINE.DASH
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.15)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "ПОЯВИТСЯ ПОСЛЕ ЗАПУСКА"
    r.font.name, r.font.size, r.font.bold = FONT, Pt(size - 1), True
    r.font.color.rgb = rgb("9A6700")
    for t, b in ((what, True), (f"Источник: {source}", False)):
        p = tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = t
        r.font.name, r.font.size, r.font.bold = FONT, Pt(size), b
        r.font.color.rgb = rgb(INK if b else MUTED)
    return s


def picture_or_placeholder(slide, name, path, x, y, w, h, what, source):
    """Вставляет рисунок по центру рамки (с сохранением пропорций) или рисует заглушку."""
    if path is None:
        return placeholder(slide, name, x, y, w, h, what, source)
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    pic = slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                                   Inches(pw), Inches(ph))
    pic.name = name
    return pic


def add_table(slide, rows, x, y, w, col_w, row_h=0.4, size=12, name=None, header=True):
    nrows, ncols = len(rows), len(rows[0])
    gs = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(row_h * nrows))
    if name:
        gs.name = name
    tbl = gs.table
    total = sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(w * cw / total)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            is_head = header and i == 0
            cell.fill.fore_color.rgb = rgb(NAVY if is_head else (WHITE if i % 2 else PAPER))
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = str(val)
            r.font.name, r.font.size = FONT, Pt(size)
            r.font.bold = is_head
            r.font.color.rgb = rgb(WHITE if is_head else INK)
    return gs


def new_slide(prs, idx, title, section, notes):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(PAPER)
    badge(s, 0.55, 0.38, idx, d=0.55)
    add_text(s, section.upper(), 1.25, 0.28, 8, 0.3, size=11, bold=True, color=TEAL)
    add_text(s, title, 1.25, 0.52, 11.4, 0.7, size=28, bold=True, color=NAVY, name="Заголовок")
    add_text(s, "Used Car Price Prediction · итоговый проект", 0.55, 7.05, 6, 0.3, size=10, color=MUTED)
    add_text(s, f"{idx} / {TOTAL}", 11.8, 7.05, 1.0, 0.3, size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def bullets(slide, items, x, y, w, h, size=16, gap=8, color=INK):
    return add_text(slide, [("•  " + t, {}) for t in items], x, y, w, h, size=size, color=color, spacing=gap)


def money(v):
    return f"${v:,.0f}"


# ---------------------------------------------------------------- слайды
def build() -> str:
    R = Results()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)

    # 1. Титульный ------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(NAVY)
    badge(s, 9.4, 1.6, "$", d=3.2, fill=AMBER, color=NAVY, size=120)
    add_text(s, "Предсказание стоимости подержанных автомобилей", 0.8, 1.5, 8.2, 2.2, size=40, bold=True,
             color=WHITE, name="Заголовок")
    add_text(s, "Used Car Price Prediction", 0.8, 3.75, 8, 0.5, size=22, bold=True, color=AMBER)
    add_text(s, ["Итоговый проект · тема №35",
                 "Programming for Data Analysis and Database",
                 "Applied Artificial Intelligence, магистратура"], 0.8, 4.5, 8.2, 1.2, size=16,
             color="C9D3E3", spacing=3)
    # вклад каждого участника: коротко на слайде, подробно — в отчёте
    add_text(s, [("Авторы и вклад", {"size": 13, "bold": True, "color": AMBER})]
             + [f"{m['name']} ({m['group']}) — {m['role'].split(' (')[0]}" for m in config.TEAM],
             0.8, 5.75, 8.4, 1.3, size=11, color="C9D3E3", spacing=2)
    s.notes_slide.notes_text_frame.text = "Вклад участников: " + "; ".join(
        f"{m['name']} — {m['role']}" for m in config.TEAM)

    # 2. Проблема --------------------------------------------------------------
    s = new_slide(prs, 2, "Проблема: как оценить справедливую цену авто с пробегом", "Введение",
                  "Задача практичная: цену определяют много факторов, а данные объявлений шумные.")
    items = [("1", "Много факторов цены", "Возраст, пробег, марка и модель, состояние, тип кузова и топлива, регион — "
                                          "человеку трудно учесть всё сразу."),
             ("2", "Шумные данные", "В объявлениях встречаются фиктивные цены (например, 1 или 123456789), "
                                    "пропуски и повторные размещения одного авто."),
             ("3", "Нужен инструмент", "Воспроизводимая модель, которая даёт ориентировочную цену и показывает, "
                                       "какие признаки на неё влияют.")]
    for i, (n, h, t) in enumerate(items):
        x = 0.6 + i * 4.1
        card(s, x, 1.7, 3.8, 3.9)
        badge(s, x + 0.3, 2.0, n, d=0.6)
        add_text(s, h, x + 0.3, 2.8, 3.2, 0.6, size=20, bold=True, color=NAVY)
        add_text(s, t, x + 0.3, 3.5, 3.2, 2.3, size=15, color=INK)
    add_text(s, "Задача машинного обучения: регрессия (прогноз числа — цены в долларах).", 0.6, 5.95, 12, 0.5,
             size=16, bold=True, color=TEAL)

    # 3. Цель и задачи ---------------------------------------------------------
    s = new_slide(prs, 3, "Цель и задачи проекта", "Введение",
                  "Цель одна, задачи соответствуют этапам пайплайна main.py (prepare → clean → db → eda → train → report).")
    card(s, 0.6, 1.7, 4.6, 4.9, fill=NAVY)
    add_text(s, "Цель", 0.9, 1.95, 4, 0.5, size=14, bold=True, color=AMBER)
    add_text(s, "Построить модель, которая по характеристикам объявления прогнозирует цену подержанного "
                "автомобиля, и выявить главные факторы цены.", 0.9, 2.5, 4.0, 3.5, size=20, color=WHITE)
    tasks = ["Загрузить большой CSV по частям, оптимизировать память, сохранить в Parquet",
             "Провести аудит качества и очистить данные",
             "Загрузить данные в SQLite, выполнить 11 SQL-запросов",
             "Выполнить EDA и построить визуализации",
             "Подготовить признаки без утечки данных",
             "Сравнить baseline и модели с подбором гиперпараметров (CV)",
             "Интерпретировать модель, проанализировать ошибки, сделать интерфейс прогноза"]
    for i, t in enumerate(tasks):
        y = 1.7 + i * 0.7
        badge(s, 5.6, y, i + 1, d=0.45, size=14)
        add_text(s, t, 6.25, y - 0.02, 6.5, 0.6, size=15, anchor=MSO_ANCHOR.MIDDLE)

    # 4. Датасет ---------------------------------------------------------------
    s = new_slide(prs, 4, "Датасет: Craigslist Cars & Trucks", "Данные",
                  "Числа справа — из аудита сырых данных (audit_raw.json). Требование к объёму выполнено "
                  "по размеру файла: он больше 1 ГБ, поэтому читаем его по частям.")
    card(s, 0.6, 1.7, 6.3, 4.9)
    bullets(s, ["Источник: Kaggle, austinreese/craigslist-carstrucks-data (файл vehicles.csv)",
                "Объявления о продаже автомобилей в США",
                "Целевая переменная: price (цена объявления)",
                "Используем 21 столбец: год, пробег, марка, модель, состояние, топливо, КПП, привод, тип кузова, "
                "цвет, регион, штат, координаты, дата, VIN",
                "Не читаем: url, region_url, image_url, description, county — ссылки, длинный текст и пустой столбец"],
            0.85, 1.9, 5.8, 4.6, size=15, gap=9)
    stats = [("Строк в файле", f"{R.audit['n_rows']:,}" if R.audit else None, "audit_raw.json → n_rows"),
             ("Столбцов в работе", str(R.audit["n_cols"]) if R.audit else None, "audit_raw.json → n_cols"),
             ("Период публикации", (f"{R.audit['posting_span_days']} дн." if R.audit and "posting_span_days" in R.audit
                                    else None), "audit_raw.json → posting_span_days")]
    for i, (lab, val, src) in enumerate(stats):
        y = 1.7 + i * 1.7
        if val:
            card(s, 7.3, y, 5.4, 1.5)
            add_text(s, val, 7.5, y + 0.1, 5.0, 0.8, size=36, bold=True, color=NAVY)
            add_text(s, lab, 7.5, y + 0.9, 5.0, 0.4, size=14, color=MUTED)
        else:
            placeholder(s, f"PH_dataset_stat_{i + 1}", 7.3, y, 5.4, 1.5, lab, src)

    # 5. Архитектура -----------------------------------------------------------
    s = new_slide(prs, 5, "Архитектура проекта: модули и поток данных", "Архитектура",
                  "Каждый модуль отвечает за один этап; main.py запускает этапы по порядку, "
                  "настройки собраны в config.py, результаты лежат отдельно от кода.")
    mods = [("data_loader.py", "чанки, типы,\nParquet"), ("cleaning.py", "аудит и\nочистка"),
            ("database.py", "SQLite,\n11 запросов"), ("eda.py", "статистика,\n11 графиков"),
            ("features.py\nmodeling.py", "Pipeline,\nмодели, CV"), ("predict.py\napp_*.py", "прогноз,\nинтерфейсы")]
    bw, gap = 1.85, 0.3
    for i, (m, d) in enumerate(mods):
        x = 0.6 + i * (bw + gap)
        card(s, x, 1.8, bw, 2.1, fill=WHITE)
        add_text(s, m, x + 0.05, 1.9, bw - 0.1, 0.8, size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        add_text(s, d, x + 0.05, 2.8, bw - 0.1, 0.9, size=13, color=MUTED, align=PP_ALIGN.CENTER)
        if i < len(mods) - 1:
            add_shape(s, MSO_SHAPE.RIGHT_ARROW, x + bw + 0.03, 2.65, gap - 0.06, 0.3, fill=AMBER, line=None)
    add_text(s, "Результаты сохраняются в папки", 0.6, 4.3, 8, 0.4, size=15, bold=True, color=TEAL)
    outs = [("data/processed/", "Parquet, cars.db"), ("outputs/results/", "метрики, JSON, CSV, SQL"),
            ("outputs/figures/", "графики и ER"), ("models/", "joblib + метаданные"),
            ("outputs/report/", "REPORT.md / .docx"), ("presentation/", "эта презентация")]
    for i, (f, d) in enumerate(outs):
        x = 0.6 + (i % 3) * 4.1
        y = 4.8 + (i // 3) * 1.05
        card(s, x, y, 3.85, 0.9, fill=WHITE)
        add_text(s, f, x + 0.15, y + 0.06, 3.6, 0.4, size=14, bold=True, color=NAVY)
        add_text(s, d, x + 0.15, y + 0.46, 3.6, 0.4, size=12, color=MUTED)

    # 6. Загрузка и память -------------------------------------------------------
    s = new_slide(prs, 6, "Большие данные: чанки, типы данных и Parquet", "Данные",
                  "Чтение по частям с оптимизацией типов экономит прежде всего память. "
                  "Parquet быстрее и компактнее CSV: данные хранятся по столбцам, сжаты и с готовыми типами.")
    card(s, 0.6, 1.7, 4.6, 4.9)
    bullets(s, ["usecols — читаем только нужные столбцы", "chunksize — файл читается порциями",
                "downcast — минимально достаточные числовые типы",
                "category — для строк с малым числом значений", "Parquet — колоночный сжатый формат"],
            0.85, 1.9, 4.2, 4.6, size=15, gap=10)
    if R.bench:
        rows = [["Способ", "Время, с", "Память, МБ", "Файл, МБ"]] + [
            [b["method"], b["seconds"], b["memory_mb"], b["file_mb"]] for b in R.bench["loading_benchmark"]]
        add_table(s, rows, 5.5, 1.9, 7.2, [4, 1.3, 1.5, 1.3], row_h=0.8, size=13, name="Таблица загрузки")
    else:
        placeholder(s, "PH_loading_table", 5.5, 1.9, 7.2, 3.2, "Таблица «способ загрузки — время — память — размер файла»",
                    "outputs/results/loading_benchmark.json")

    # 7. Очистка -----------------------------------------------------------------
    s = new_slide(prs, 7, "Аудит и очистка данных", "Данные",
                  "Слева — правила (значения из config.py). Справа — журнал очистки из cleaning_log.json "
                  "(сколько строк удалено на каждом шаге).")
    rules = [["Правило", "Значение"],
             ["Цена", f"{config.PRICE_MIN}–{config.PRICE_MAX:,}"], ["Цены-заглушки", "11111, 123456…"],
             ["Год выпуска", f"{config.YEAR_MIN}–{config.YEAR_MAX}"],
             ["Пробег", f"≤ {config.ODOMETER_MAX:,}; < {config.ODOMETER_MIN_USED} у подержанных → NaN"],
             ["Цена = пробег", "удаляется (ошибка ввода)"],
             ["Дубликаты", "VIN; без VIN: год + марка + модель + пробег"], ["Пропуски категорий", "→ unknown"]]
    add_table(s, rules, 0.6, 1.8, 5.2, [2, 3], row_h=0.47, size=13, name="Правила очистки")
    add_text(s, "IQR вычисляется справочно: дорогие авто — не ошибки данных.", 0.6, 5.8, 5.2, 0.7, size=13, color=MUTED)
    if R.clean:
        log = [["Шаг", "Удалено", "Осталось"]] + [[x["step"], f"{x['removed']:,}", f"{x['remaining']:,}"]
                                                   for x in R.clean["log"][2:]]
        add_table(s, log, 6.2, 1.8, 6.5, [3, 1.3, 1.3], row_h=0.42, size=12, name="Журнал очистки")
    else:
        placeholder(s, "PH_cleaning_log", 6.2, 1.8, 6.5, 4.4, "Журнал очистки: сколько строк удалено на каждом шаге",
                    "outputs/results/cleaning_log.json")

    # 8. EDA ----------------------------------------------------------------------
    s = new_slide(prs, 8, "Исследовательский анализ данных (EDA)", "Анализ",
                  "У цены тяжёлый правый хвост, поэтому модель учится на логарифме. "
                  "Цена падает с возрастом и пробегом, быстрее всего — в первые годы.")
    figs = [("01_price_distribution.png", "Распределение цены"), ("03_price_by_age.png", "Цена и возраст"),
            ("04_price_vs_odometer.png", "Цена и пробег"), ("06_correlation_heatmap.png", "Корреляции")]
    for i, (f, cap) in enumerate(figs):
        x = 0.6 + (i % 2) * 4.05
        y = 1.55 + (i // 2) * 2.7
        card(s, x, y, 3.9, 2.55)
        picture_or_placeholder(s, f"PH_eda_{i + 1}", R.fig(f), x + 0.08, y + 0.08, 3.74, 2.0, cap, f"outputs/figures/{f}")
        add_text(s, cap, x + 0.08, y + 2.12, 3.74, 0.35, size=12, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    card(s, 8.8, 1.55, 3.95, 5.25)
    add_text(s, "Выводы по данным", 9.0, 1.65, 3.6, 0.4, size=15, bold=True, color=TEAL)
    if R.eda:
        pick = [it["text"] for it in R.eda["items"] if it["topic"] in
                ("Распределение цены", "Связь цены с возрастом и пробегом", "Цена по возрасту", "Марки")]
        add_text(s, [("•  " + t, {}) for t in pick[:4]], 9.0, 2.1, 3.6, 4.6, size=11, spacing=5)
    else:
        placeholder(s, "PH_eda_findings", 9.0, 2.15, 3.55, 4.4, "3–4 вывода по EDA с числами",
                    "outputs/results/eda_findings.md", size=12)

    # 9. SQLite ------------------------------------------------------------------
    s = new_slide(prs, 9, "SQLite: нормализованная схема и 11 SQL-запросов", "База данных",
                  "Схема в третьей нормальной форме, диаграмма соответствует DDL в database.py. "
                  "Для показа: запрос с CTE (q04) и запросы с оконными функциями (q03 — RANK, q07 — LAG).")
    tabs = [("manufacturers", ["PK manufacturer_id", "name"], 0.6, 1.6),
            ("models", ["PK model_id", "FK manufacturer_id", "model_name"], 3.2, 1.6),
            ("regions", ["PK region_id", "region_name", "state"], 3.2, 4.4),
            ("listings", ["PK listing_id", "FK model_id", "FK region_id", "price, year, odometer", "condition, fuel, …",
                          "lat, long, posting_date"], 5.9, 2.5)]
    pos = {}
    for name, cols, x, y in tabs:
        w, rh = 2.3, 0.33
        h = 0.4 + rh * len(cols)
        add_shape(s, MSO_SHAPE.RECTANGLE, x, y, w, h, fill=WHITE, line=NAVY, lw=1.5)
        hd = add_shape(s, MSO_SHAPE.RECTANGLE, x, y, w, 0.4, fill=NAVY, line=NAVY)
        add_text(s, name, x, y + 0.03, w, 0.35, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        for i, c in enumerate(cols):
            add_text(s, c, x + 0.08, y + 0.42 + i * rh, w - 0.1, rh, size=11, bold=c.startswith(("PK", "FK")))
        pos[name] = (x, y, w, h)
    for a, b, ya, yb in (("manufacturers", "models", 2.05, 2.05), ("models", "listings", 2.05, 3.0),
                         ("regions", "listings", 4.9, 4.4)):
        xa = pos[a][0] + pos[a][2]
        xb = pos[b][0]
        ln = s.shapes.add_connector(1, Inches(xa), Inches(ya), Inches(xb), Inches(yb))
        ln.line.color.rgb = rgb(AMBER)
        ln.line.width = Pt(2.25)
    add_text(s, "1 : N во всех связях", 0.6, 4.6, 2.5, 0.4, size=12, color=MUTED)
    card(s, 8.6, 1.6, 4.1, 3.2)
    add_text(s, "Типы запросов (sql/queries.sql)", 8.75, 1.7, 3.9, 0.4, size=14, bold=True, color=NAVY)
    bullets(s, ["JOIN (до трёх таблиц)", "GROUP BY + HAVING", "CTE (WITH) и подзапросы",
                "RANK, ROW_NUMBER, LAG", "SUM / AVG OVER (окно)", "Результаты → Pandas → графики"],
            8.75, 2.15, 3.9, 2.6, size=13, gap=5)
    if R.db:
        rows = [["Способ", "Время, с"]] + [[x["method"], f"{x['seconds']:.4f}"] for x in R.db["pandas_vs_sql"]]
        add_table(s, rows, 8.6, 5.0, 4.1, [3, 1.1], row_h=0.45, size=12, name="Pandas vs SQL")
    else:
        placeholder(s, "PH_pandas_vs_sql", 8.6, 5.0, 4.1, 1.5, "Сравнение времени: Pandas и SQL",
                    "outputs/results/db_summary.json", size=11)

    # 10. Признаки и утечка ------------------------------------------------------
    s = new_slide(prs, 10, "Подготовка признаков и защита от утечки данных", "Методы",
                  "Главная идея: сначала разбиваем данные, потом обучаем преобразования только на train. "
                  "TargetEncoder использует кросс-фиттинг, поэтому не видит собственную целевую переменную.")
    flow = [("Очищенные\nданные", WHITE), ("Split\ntrain / test", AMBER_BG), ("Pipeline\n(fit на train)", WHITE),
            ("Модель", WHITE)]
    for i, (t, f) in enumerate(flow):
        x = 0.6 + i * 3.15
        card(s, x, 1.7, 2.7, 1.3, fill=f)
        add_text(s, t, x, 1.8, 2.7, 1.1, size=16, bold=True, color=NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        if i < 3:
            add_shape(s, MSO_SHAPE.RIGHT_ARROW, x + 2.75, 2.2, 0.35, 0.3, fill=AMBER, line=None)
    br = [("Числовые", "медианная импутация,\nиндикаторы пропусков,\nстандартизация"),
          ("Мало категорий", "'unknown' для пропусков,\nOne-Hot\n(редкие значения объединяются)"),
          ("Много категорий", "марка, модель, марка+модель,\nрегион: TargetEncoder\n(сглаживание, cv=5)")]
    for i, (h, t) in enumerate(br):
        x = 0.6 + i * 4.1
        card(s, x, 3.4, 3.85, 1.9)
        add_text(s, h, x + 0.15, 3.5, 3.6, 0.4, size=15, bold=True, color=TEAL)
        add_text(s, t, x + 0.15, 3.95, 3.6, 1.3, size=13)
    add_text(s, ["Целевая переменная — log(1+цена): тяжёлый правый хвост.",
                 "Повторные объявления удалены ДО разбиения; лучшая модель выбирается по CV, а не по тесту."],
             0.6, 5.55, 12.1, 1.2, size=15, color=INK, spacing=6)

    # 11. Методы ML ---------------------------------------------------------------
    s = new_slide(prs, 11, "Методы ML: baseline, линейная модель, градиентный бустинг", "Методы",
                  "Список моделей и гиперпараметров читается из modeling.build_models(), поэтому совпадает с кодом.")
    grids = {k: ", ".join(p.replace("model__", "") for p in v[1]) for k, v in modeling.build_models().items()}
    role = {"ridge": "линейная модель", "hist_gb": "градиентный бустинг (sklearn)",
            "lightgbm": "градиентный бустинг (LightGBM)"}
    rows = [["Модель", "Роль", "Подбираемые гиперпараметры"], ["baseline_median", "точка отсчёта (медиана цены)", "—"]]
    for k in grids:
        rows.append([k, role.get(k, ""), grids.get(k, "")])
    add_table(s, rows, 0.6, 1.7, 12.1, [2.2, 3.3, 6.6], row_h=0.6, size=13, name="Модели")
    add_text(s, [f"Подбор: RandomizedSearchCV, {config.TUNE_N_ITER} комбинаций, {config.CV_FOLDS}-блочная CV "
                 f"на подвыборке {config.TUNE_SAMPLE:,} объектов; финальное обучение — на всей обучающей выборке.",
                 "Метрики: MAE и RMSE в долларах, R²; также время, пиковая память, размер модели, скорость инференса."],
             0.6, 5.85, 12.1, 1.1, size=14, spacing=4)

    # 12. Результаты ------------------------------------------------------------
    s = new_slide(prs, 12, "Результаты: сравнение моделей на тестовой выборке", "Результаты",
                  "Лучшая модель выбрана по кросс-валидации, тест использован один раз для итоговой оценки. "
                  "Оба варианта бустинга близки по качеству и заметно точнее линейной модели.")
    if R.comp is not None:
        c = R.comp
        rows = [["Модель", "MAE, $", "RMSE, $", "R²", "Обучение, с", "Размер, МБ"]] + [
            [r.model, f"{r.MAE:,.0f}", f"{r.RMSE:,.0f}", f"{r.R2:.3f}", f"{r.fit_seconds:.1f}", f"{r.model_size_mb:.1f}"]
            for r in c.itertuples()]
        add_table(s, rows, 0.6, 1.6, 7.3, [2.4, 1.2, 1.2, 1, 1.3, 1.2], row_h=0.5, size=12, name="Таблица результатов")
        cd = CategoryChartData()
        cd.categories = list(c["model"])
        cd.add_series("MAE, $", [round(float(v), 0) for v in c["MAE"]])
        gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(8.1), Inches(1.6), Inches(4.7), Inches(4.3), cd)
        gf.name = "Диаграмма MAE"
        ch = gf.chart
        ch.has_legend = False
        ch.has_title = True
        ch.chart_title.text_frame.text = "MAE на тесте, $"
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
        ch.plots[0].has_data_labels = True
        ch.plots[0].data_labels.font.size = Pt(10)
        ch.plots[0].data_labels.number_format = '#,##0'
        ch.plots[0].data_labels.number_format_is_linked = False
        ch.plots[0].series[0].format.fill.solid()
        ch.plots[0].series[0].format.fill.fore_color.rgb = rgb(TEAL)
        ch.category_axis.tick_labels.font.size = Pt(9)
        ch.value_axis.tick_labels.font.size = Pt(10)
        ch.value_axis.has_major_gridlines = False
        if R.findings:
            add_text(s, R.findings["items"][0]["text"], 0.6, 5.7, 12.1, 1.2, size=14, color=INK)
    else:
        placeholder(s, "PH_results_table", 0.6, 1.6, 7.3, 4.3, "Таблица: модель, MAE, RMSE, R², время обучения, размер",
                    "outputs/results/model_comparison.csv")
        placeholder(s, "PH_results_chart", 8.1, 1.6, 4.7, 4.3, "Диаграмма MAE по моделям",
                    "outputs/results/model_comparison.csv")
        placeholder(s, "PH_results_text", 0.6, 6.0, 12.2, 0.9, "Вывод: какая модель лучшая и насколько она лучше baseline",
                    "outputs/results/findings.md", size=12)

    # 13. Интерпретация ----------------------------------------------------------
    s = new_slide(prs, 13, "Интерпретация модели и анализ ошибок", "Результаты",
                  "Главные факторы цены — возраст и пробег. Дорогие автомобили модель занижает, "
                  "дешёвые завышает; сильнее всего она ошибается на редких и очень дорогих авто.")
    card(s, 0.6, 1.5, 4.6, 3.55)
    picture_or_placeholder(s, "PH_importance", R.fig("permutation_importance.png"), 0.7, 1.6, 4.4, 3.35,
                           "Важность признаков", "outputs/figures/permutation_importance.png")
    card(s, 5.4, 1.5, 7.3, 3.55)
    picture_or_placeholder(s, "PH_residuals", R.fig("residuals.png"), 5.5, 1.6, 7.1, 3.35,
                           "Факт vs прогноз, остатки, ошибки", "outputs/figures/residuals.png")
    if R.findings:
        topics = {"Главные факторы цены", "Ошибки по цене", "Ошибки по возрасту", "Ошибки по маркам"}
        txt = [it["text"] for it in R.findings["items"] if it["topic"] in topics][:3]
        add_text(s, [("•  " + t, {}) for t in txt], 0.6, 5.2, 12.1, 1.7, size=13, spacing=4)
    else:
        placeholder(s, "PH_findings", 0.6, 5.2, 12.1, 1.6, "2–3 вывода: главные признаки; где модель ошибается сильнее всего",
                    "outputs/results/findings.md", size=12)

    # 14. Прогноз -------------------------------------------------------------------
    s = new_slide(prs, 14, "Прогноз цены: консоль и веб-интерфейс", "Приложение",
                  "Живая демонстрация: ввести автомобиль, показать интервал и проверку цены объявления, "
                  "затем ввести неверные данные (текст вместо года) — программа не падает.")
    card(s, 0.6, 1.6, 6.0, 2.0)
    add_text(s, "Консоль: python app_console.py", 0.8, 1.7, 5.6, 0.4, size=15, bold=True, color=NAVY)
    add_text(s, "Меню 1–8: данные, очистка, БД, SQL, EDA, обучение, прогноз (пункт 7), таблица моделей.",
             0.8, 2.15, 5.6, 1.4, size=14)
    card(s, 0.6, 3.8, 6.0, 1.7)
    add_text(s, "Веб: streamlit run app_streamlit.py", 0.8, 3.9, 5.6, 0.4, size=15, bold=True, color=NAVY)
    add_text(s, "Вкладки: «Оценка» (марка, модель, год, пробег — цена сразу), «Данные» (фильтры и графики), «О модели».", 0.8, 4.35, 5.6, 1.1, size=14)
    bullets(s, ["Прогноз и интервал цены (квантильная регрессия)",
                "Проверка цены объявления: ниже рынка, выше или в интервале",
                "Неверный ввод → понятное сообщение, без падения"], 0.7, 5.65, 6.0, 1.3, size=13, gap=3)
    screenshot = config.PRESENTATION_DIR / "app_screenshot.png"
    picture_or_placeholder(s, "PH_screenshot", screenshot if screenshot.exists() else None, 6.9, 1.6, 5.8, 5.2,
                           "Скриншот приложения с прогнозом", "сохраните скриншот как presentation/app_screenshot.png")

    # 15. Выводы ---------------------------------------------------------------------
    s = new_slide(prs, 15, "Выводы, ограничения и дальнейшая работа", "Итоги",
                  "Модель подходит для быстрой оценки массовых автомобилей; для дорогих, редких и очень старых "
                  "прогноз — ориентир, и это видно по ширине интервала.")
    card(s, 0.6, 1.6, 6.1, 5.2)
    add_text(s, "Выводы", 0.85, 1.7, 5.6, 0.4, size=16, bold=True, color=TEAL)
    if R.findings:
        bullets(s, [it["text"] for it in R.findings["items"][:3]], 0.85, 2.15, 5.7, 4.5, size=13, gap=8)
    else:
        placeholder(s, "PH_conclusions", 0.85, 2.2, 5.6, 4.3, "3–4 вывода по реальным результатам (лучшая модель, "
                    "выигрыш над baseline, главные факторы)", "outputs/results/findings.md")
    card(s, 6.9, 1.6, 5.8, 3.1)
    add_text(s, "Ограничения", 7.15, 1.7, 5.3, 0.4, size=16, bold=True, color=TEAL)
    bullets(s, ["Цены предложения, а не сделок", "Нет данных о кузове, комплектации, истории",
                "Короткий период публикации → случайное, а не временное разбиение",
                "Пороги очистки выбраны экспертно"], 7.15, 2.15, 5.4, 2.5, size=13, gap=5)
    card(s, 6.9, 4.9, 5.8, 1.9)
    add_text(s, "Дальше", 7.15, 5.0, 5.3, 0.4, size=16, bold=True, color=TEAL)
    bullets(s, ["Текстовые признаки из описаний", "Калибровка интервала цены (conformal prediction)",
                "Временная валидация на более длинном периоде"], 7.15, 5.45, 5.4, 1.3, size=13, gap=4)

    config.PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(config.PRESENTATION_PPTX)
    return str(config.PRESENTATION_PPTX)


if __name__ == "__main__":
    print(build())