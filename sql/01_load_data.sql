-- Dockerコンテナ内に/dataを読み取り専用でマウントして実行する。
-- CSVではchildrenの欠損が"NA"、country・agent・companyの欠損が"NULL"である。
-- 一時テーブルへ読み込み、欠損表現をSQLのNULLへ統一してから本テーブルへ登録する。
-- トランザクション内で処理するため、途中で失敗した場合は既存データを維持する。
BEGIN;

CREATE TEMP TABLE hotel_bookings_raw (LIKE hotel_bookings);
ALTER TABLE hotel_bookings_raw
    ALTER COLUMN children TYPE TEXT,
    ALTER COLUMN country TYPE TEXT,
    ALTER COLUMN agent TYPE TEXT,
    ALTER COLUMN company TYPE TEXT;

\copy hotel_bookings_raw FROM '/data/hotel_bookings.csv' WITH (FORMAT CSV, HEADER TRUE);

TRUNCATE TABLE hotel_bookings;

INSERT INTO hotel_bookings
SELECT
    hotel,
    is_canceled,
    lead_time,
    arrival_date_year,
    arrival_date_month,
    arrival_date_week_number,
    arrival_date_day_of_month,
    stays_in_weekend_nights,
    stays_in_week_nights,
    adults,
    NULLIF(NULLIF(children, 'NA'), 'NULL')::DOUBLE PRECISION,
    babies,
    meal,
    NULLIF(NULLIF(country, 'NA'), 'NULL'),
    market_segment,
    distribution_channel,
    is_repeated_guest,
    previous_cancellations,
    previous_bookings_not_canceled,
    reserved_room_type,
    assigned_room_type,
    booking_changes,
    deposit_type,
    NULLIF(NULLIF(agent, 'NA'), 'NULL')::DOUBLE PRECISION,
    NULLIF(NULLIF(company, 'NA'), 'NULL')::DOUBLE PRECISION,
    days_in_waiting_list,
    customer_type,
    adr,
    required_car_parking_spaces,
    total_of_special_requests,
    reservation_status,
    reservation_status_date
FROM hotel_bookings_raw;

COMMIT;

