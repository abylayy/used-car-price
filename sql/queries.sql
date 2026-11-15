-- Формат: строка "-- name: <имя>" начинает новый запрос, "-- desc:" — описание.
-- Запросы читает database.load_queries(); результаты загружаются в Pandas.

-- name: q01_top_manufacturers
-- desc: [JOIN x2, GROUP BY, HAVING] Топ-20 марок по числу объявлений, средняя цена и пробег
SELECT m.name AS manufacturer,
       COUNT(*)                  AS n_listings,
       ROUND(AVG(l.price), 0)    AS avg_price,
       ROUND(AVG(l.odometer), 0) AS avg_odometer
FROM listings l
JOIN models md        ON md.model_id = l.model_id
JOIN manufacturers m  ON m.manufacturer_id = md.manufacturer_id
WHERE m.name <> 'unknown'
GROUP BY m.name
HAVING COUNT(*) >= 100
ORDER BY n_listings DESC
LIMIT 20;

-- name: q02_price_by_year
-- desc: [GROUP BY, HAVING] Средняя цена и пробег по году выпуска (кривая «обесценивания»)
SELECT year,
       COUNT(*)                AS n_listings,
       ROUND(AVG(price), 0)    AS avg_price,
       ROUND(AVG(odometer), 0) AS avg_odometer
FROM listings
GROUP BY year
HAVING COUNT(*) >= 30
ORDER BY year;

-- name: q03_state_price_rank
-- desc: [JOIN, GROUP BY, HAVING, оконная функция RANK] Рейтинг штатов по средней цене
SELECT r.state,
       COUNT(*)                                 AS n_listings,
       ROUND(AVG(l.price), 0)                   AS avg_price,
       RANK() OVER (ORDER BY AVG(l.price) DESC) AS price_rank
FROM listings l
JOIN regions r ON r.region_id = l.region_id
GROUP BY r.state
HAVING COUNT(*) >= 100
ORDER BY price_rank;

-- name: q04_model_premium
-- desc: [CTE x2, JOIN x3] Модели, цена которых сильнее всего отличается от средней по марке
WITH model_stats AS (
    SELECT md.model_id, md.model_name, md.manufacturer_id,
           COUNT(*) AS n_listings, AVG(l.price) AS avg_price
    FROM listings l
    JOIN models md ON md.model_id = l.model_id
    GROUP BY md.model_id
    HAVING COUNT(*) >= 50
),
manuf_stats AS (
    SELECT md.manufacturer_id, AVG(l.price) AS manuf_avg_price
    FROM listings l
    JOIN models md ON md.model_id = l.model_id
    GROUP BY md.manufacturer_id
)
SELECT m.name AS manufacturer, ms.model_name, ms.n_listings,
       ROUND(ms.avg_price, 0)       AS model_avg_price,
       ROUND(mf.manuf_avg_price, 0) AS manufacturer_avg_price,
       ROUND(ms.avg_price / mf.manuf_avg_price, 2) AS premium_ratio
FROM model_stats ms
JOIN manuf_stats mf   ON mf.manufacturer_id = ms.manufacturer_id
JOIN manufacturers m  ON m.manufacturer_id = ms.manufacturer_id
WHERE m.name <> 'unknown'
ORDER BY premium_ratio DESC
LIMIT 20;

-- name: q05_top3_models_per_manufacturer
-- desc: [CTE, оконная функция ROW_NUMBER с PARTITION BY] Три самые дорогие модели каждой марки
WITH model_stats AS (
    SELECT md.model_id, md.model_name, md.manufacturer_id,
           COUNT(*) AS n_listings, AVG(l.price) AS avg_price
    FROM listings l
    JOIN models md ON md.model_id = l.model_id
    GROUP BY md.model_id
    HAVING COUNT(*) >= 50
),
ranked AS (
    SELECT ms.*,
           ROW_NUMBER() OVER (PARTITION BY ms.manufacturer_id ORDER BY ms.avg_price DESC) AS rn
    FROM model_stats ms
)
SELECT m.name AS manufacturer, r.model_name, r.n_listings,
       ROUND(r.avg_price, 0) AS avg_price, r.rn AS rank_in_brand
FROM ranked r
JOIN manufacturers m ON m.manufacturer_id = r.manufacturer_id
WHERE r.rn <= 3 AND m.name <> 'unknown'
ORDER BY m.name, r.rn;

-- name: q06_overpriced_vs_model_average
-- desc: [CTE, JOIN x3, GROUP BY] Число объявлений с ценой > 2.5 средних по своей модели, по маркам
WITH model_avg AS (
    SELECT model_id, AVG(price) AS avg_price
    FROM listings
    GROUP BY model_id
    HAVING COUNT(*) >= 20
)
SELECT m.name AS manufacturer, COUNT(*) AS n_overpriced
FROM listings l
JOIN model_avg ma     ON ma.model_id = l.model_id
JOIN models md        ON md.model_id = l.model_id
JOIN manufacturers m  ON m.manufacturer_id = md.manufacturer_id
WHERE l.price > 2.5 * ma.avg_price
GROUP BY m.name
ORDER BY n_overpriced DESC
LIMIT 15;

-- name: q07_yoy_price_change
-- desc: [CTE, оконная функция LAG] Изменение средней цены по сравнению с предыдущим годом выпуска
WITH by_year AS (
    SELECT year, COUNT(*) AS n_listings, AVG(price) AS avg_price
    FROM listings
    GROUP BY year
    HAVING COUNT(*) >= 30
)
SELECT year, n_listings,
       ROUND(avg_price, 0) AS avg_price,
       ROUND(LAG(avg_price) OVER (ORDER BY year), 0) AS prev_year_avg_price,
       ROUND(100.0 * (avg_price - LAG(avg_price) OVER (ORDER BY year))
             / LAG(avg_price) OVER (ORDER BY year), 1) AS pct_change_vs_prev_year
FROM by_year
ORDER BY year;

-- name: q08_body_type_share_by_fuel
-- desc: [GROUP BY, HAVING, SUM() OVER PARTITION BY] Доля типов кузова внутри каждого вида топлива
SELECT fuel, type,
       COUNT(*) AS n_listings,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY fuel), 1) AS pct_within_fuel
FROM listings
WHERE fuel <> 'unknown' AND type <> 'unknown'
GROUP BY fuel, type
HAVING COUNT(*) >= 50
ORDER BY fuel, n_listings DESC;

-- name: q09_odometer_bucket_moving_avg
-- desc: [CTE, оконная функция AVG OVER ROWS BETWEEN] Цена по корзинам пробега со скользящим средним
WITH buckets AS (
    SELECT CAST(odometer / 25000 AS INTEGER) * 25000 AS odo_bucket,
           COUNT(*) AS n_listings, AVG(price) AS avg_price
    FROM listings
    WHERE odometer IS NOT NULL AND odometer < 300000
    GROUP BY odo_bucket
    HAVING COUNT(*) >= 30
)
SELECT odo_bucket, n_listings, ROUND(avg_price, 0) AS avg_price,
       ROUND(AVG(avg_price) OVER (ORDER BY odo_bucket
                                  ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING), 0) AS moving_avg_price
FROM buckets
ORDER BY odo_bucket;

-- name: q10_regions_above_average
-- desc: [JOIN, GROUP BY, HAVING + скалярный подзапрос] Регионы, где средняя цена выше средней по всей базе
SELECT r.region_name, r.state, COUNT(*) AS n_listings, ROUND(AVG(l.price), 0) AS avg_price
FROM listings l
JOIN regions r ON r.region_id = l.region_id
GROUP BY r.region_id
HAVING COUNT(*) >= 100
   AND AVG(l.price) > (SELECT AVG(price) FROM listings)
ORDER BY avg_price DESC
LIMIT 20;

-- name: q11_brand_pareto
-- desc: [CTE, оконная функция SUM OVER (нарастающий итог)] Накопленная доля рынка по маркам
WITH brand AS (
    SELECT m.name AS manufacturer, COUNT(*) AS n_listings
    FROM listings l
    JOIN models md       ON md.model_id = l.model_id
    JOIN manufacturers m ON m.manufacturer_id = md.manufacturer_id
    WHERE m.name <> 'unknown'
    GROUP BY m.name
)
SELECT manufacturer, n_listings,
       ROUND(100.0 * SUM(n_listings) OVER (ORDER BY n_listings DESC
                                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
             / SUM(n_listings) OVER (), 1) AS cumulative_pct
FROM brand
ORDER BY n_listings DESC
LIMIT 25;
