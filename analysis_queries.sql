-- =====================================================================
-- analysis_queries.sql
-- CARS24-style business intelligence queries against the car marketplace
-- data. Demonstrates: JOINs, CTEs, window functions, aggregate business
-- metrics — the exact SQL skill set listed in the job description.
-- Run via: python sql/run_queries.py  (executes each against SQLite)
-- =====================================================================

-- ---------------------------------------------------------------------
-- Q1. City-wise performance dashboard: sell-through rate, avg days to
--     sell, and average negotiation gap (asking vs sold price).
--     Uses a CTE + JOIN against the city dimension table.
-- ---------------------------------------------------------------------
WITH city_stats AS (
    SELECT
        l.city,
        COUNT(*)                                         AS total_listings,
        SUM(CASE WHEN l.status = 'Sold' THEN 1 ELSE 0 END) AS total_sold,
        AVG(l.days_to_sell)                               AS avg_days_to_sell,
        AVG(1.0 * (l.asking_price - l.sold_price) / l.asking_price) AS avg_negotiation_pct
    FROM car_listings l
    GROUP BY l.city
)
SELECT
    c.city,
    c.region,
    c.tier,
    cs.total_listings,
    cs.total_sold,
    ROUND(100.0 * cs.total_sold / cs.total_listings, 1)  AS sell_through_rate_pct,
    ROUND(cs.avg_days_to_sell, 1)                         AS avg_days_to_sell,
    ROUND(100.0 * cs.avg_negotiation_pct, 2)              AS avg_negotiation_pct
FROM city_stats cs
JOIN dim_city c ON c.city = cs.city
ORDER BY sell_through_rate_pct DESC;


-- ---------------------------------------------------------------------
-- Q2. Top-2 fastest-selling models within EVERY brand.
--     Window function: RANK() OVER (PARTITION BY ... ORDER BY ...)
-- ---------------------------------------------------------------------
WITH model_speed AS (
    SELECT
        brand,
        model,
        AVG(days_to_sell) AS avg_days_to_sell,
        COUNT(*)          AS units_sold,
        RANK() OVER (PARTITION BY brand ORDER BY AVG(days_to_sell) ASC) AS speed_rank
    FROM car_listings
    WHERE status = 'Sold'
    GROUP BY brand, model
)
SELECT brand, model, ROUND(avg_days_to_sell, 1) AS avg_days_to_sell, units_sold
FROM model_speed
WHERE speed_rank <= 2
ORDER BY brand, speed_rank;


-- ---------------------------------------------------------------------
-- Q3. Month-over-month sold-unit trend with growth %.
--     Window function: LAG() for period-over-period comparison.
-- ---------------------------------------------------------------------
WITH monthly AS (
    SELECT
        strftime('%Y-%m', listing_date) AS listing_month,
        COUNT(*)                        AS units_sold,
        AVG(sold_price)                 AS avg_sold_price
    FROM car_listings
    WHERE status = 'Sold'
    GROUP BY listing_month
)
SELECT
    listing_month,
    units_sold,
    ROUND(avg_sold_price, 0) AS avg_sold_price,
    LAG(units_sold) OVER (ORDER BY listing_month)      AS prev_month_units,
    ROUND(100.0 * (units_sold - LAG(units_sold) OVER (ORDER BY listing_month))
          / NULLIF(LAG(units_sold) OVER (ORDER BY listing_month), 0), 1) AS mom_growth_pct
FROM monthly
ORDER BY listing_month;


-- ---------------------------------------------------------------------
-- Q4. Inventory ageing report — listings still "Active" (unsold),
--     bucketed by how long they've been listed. Flags aging stock,
--     a real ops problem for a used-car marketplace.
-- ---------------------------------------------------------------------
SELECT
    CASE
        WHEN julianday('2025-06-30') - julianday(listing_date) <= 15 THEN '0-15 days'
        WHEN julianday('2025-06-30') - julianday(listing_date) <= 30 THEN '16-30 days'
        WHEN julianday('2025-06-30') - julianday(listing_date) <= 60 THEN '31-60 days'
        ELSE '60+ days (aging risk)'
    END AS age_bucket,
    COUNT(*)                       AS num_listings,
    ROUND(AVG(asking_price), 0)    AS avg_asking_price
FROM car_listings
WHERE status = 'Active'
GROUP BY age_bucket
ORDER BY MIN(julianday('2025-06-30') - julianday(listing_date));


-- ---------------------------------------------------------------------
-- Q5. Brand profitability proxy: which brands sell closest to asking
--     price (lowest negotiation loss) vs which need the most discounting.
--     Uses a subquery + HAVING to filter for statistically meaningful volume.
-- ---------------------------------------------------------------------
SELECT
    brand,
    COUNT(*)                                                     AS units_sold,
    ROUND(AVG(asking_price), 0)                                  AS avg_asking_price,
    ROUND(AVG(sold_price), 0)                                    AS avg_sold_price,
    ROUND(100.0 * AVG((asking_price - sold_price) * 1.0 / asking_price), 2) AS avg_discount_pct
FROM car_listings
WHERE status = 'Sold'
GROUP BY brand
HAVING COUNT(*) > 50
ORDER BY avg_discount_pct ASC;


-- ---------------------------------------------------------------------
-- Q6. Duplicate / data-quality check — listings with identical brand,
--     model, city, and listing_date (would indicate a possible
--     duplicate entry in a real production pipeline).
-- ---------------------------------------------------------------------
SELECT brand, model, city, listing_date, COUNT(*) AS duplicate_count
FROM car_listings
GROUP BY brand, model, city, listing_date
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;
