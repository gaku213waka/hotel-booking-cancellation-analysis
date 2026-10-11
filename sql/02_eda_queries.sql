-- 1. データ品質と全体キャンセル率
SELECT
    COUNT(*) AS total_bookings,
    SUM(is_canceled) AS canceled_bookings,
    ROUND(AVG(is_canceled) * 100.0, 2) AS cancellation_rate_pct,
    ROUND(AVG(lead_time), 2) AS average_lead_time_days
FROM hotel_bookings;

-- 2. Lead Time帯ごとの予約件数・キャンセル率
WITH lead_time_bands AS (
    SELECT
        CASE
            WHEN lead_time <= 7 THEN '0-7'
            WHEN lead_time <= 30 THEN '8-30'
            WHEN lead_time <= 60 THEN '31-60'
            WHEN lead_time <= 120 THEN '61-120'
            WHEN lead_time <= 180 THEN '121-180'
            WHEN lead_time <= 365 THEN '181-365'
            ELSE '366+'
        END AS lead_time_band,
        CASE
            WHEN lead_time <= 7 THEN 1
            WHEN lead_time <= 30 THEN 2
            WHEN lead_time <= 60 THEN 3
            WHEN lead_time <= 120 THEN 4
            WHEN lead_time <= 180 THEN 5
            WHEN lead_time <= 365 THEN 6
            ELSE 7
        END AS sort_order,
        is_canceled
    FROM hotel_bookings
)
SELECT
    lead_time_band,
    COUNT(*) AS total_bookings,
    SUM(is_canceled) AS canceled_bookings,
    ROUND(AVG(is_canceled) * 100.0, 2) AS cancellation_rate_pct
FROM lead_time_bands
GROUP BY lead_time_band, sort_order
ORDER BY sort_order;

-- 3. 市場セグメント別の予約件数・キャンセル率
SELECT
    market_segment,
    COUNT(*) AS total_bookings,
    SUM(is_canceled) AS canceled_bookings,
    ROUND(AVG(is_canceled) * 100.0, 2) AS cancellation_rate_pct
FROM hotel_bookings
GROUP BY market_segment
ORDER BY total_bookings DESC;

-- 4. 予約件数が多い国Top 10
SELECT
    country,
    COUNT(*) AS total_bookings,
    SUM(is_canceled) AS canceled_bookings,
    ROUND(AVG(is_canceled) * 100.0, 2) AS cancellation_rate_pct
FROM hotel_bookings
WHERE country IS NOT NULL
GROUP BY country
ORDER BY total_bookings DESC
LIMIT 10;

-- 5. リピーターと新規予約の比較
SELECT
    is_repeated_guest,
    COUNT(*) AS total_bookings,
    SUM(is_canceled) AS canceled_bookings,
    ROUND(AVG(is_canceled) * 100.0, 2) AS cancellation_rate_pct
FROM hotel_bookings
GROUP BY is_repeated_guest
ORDER BY is_repeated_guest;

-- 6. 到着月別の予約件数・キャンセル率（暦順）
SELECT
    arrival_date_month,
    COUNT(*) AS total_bookings,
    SUM(is_canceled) AS canceled_bookings,
    ROUND(AVG(is_canceled) * 100.0, 2) AS cancellation_rate_pct
FROM hotel_bookings
GROUP BY arrival_date_month
ORDER BY CASE arrival_date_month
    WHEN 'January' THEN 1
    WHEN 'February' THEN 2
    WHEN 'March' THEN 3
    WHEN 'April' THEN 4
    WHEN 'May' THEN 5
    WHEN 'June' THEN 6
    WHEN 'July' THEN 7
    WHEN 'August' THEN 8
    WHEN 'September' THEN 9
    WHEN 'October' THEN 10
    WHEN 'November' THEN 11
    WHEN 'December' THEN 12
END;

