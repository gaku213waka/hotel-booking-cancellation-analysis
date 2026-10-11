# SQL analysis

PostgreSQL 16で、CSVの読み込みと主要なEDA集計を再現します。

実行順序：

1. `00_create_table.sql`：テーブル作成
2. `01_load_data.sql`：`data/hotel_bookings.csv`の読み込み
3. `02_eda_queries.sql`：全体、Lead Time、市場セグメント、国、リピーター、月別集計

Docker利用時はリポジトリ直下で次を実行します。

```bash
# PowerShell
Copy-Item docker/.env.example docker/.env

# macOS/Linux
# cp docker/.env.example docker/.env

# docker/.envのパスワードを変更してから起動する
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d
docker exec hotel-booking-postgres psql -U hotel_user -d hotel_booking -f /sql/00_create_table.sql
docker exec hotel-booking-postgres psql -U hotel_user -d hotel_booking -f /sql/01_load_data.sql
docker exec hotel-booking-postgres psql -U hotel_user -d hotel_booking -f /sql/02_eda_queries.sql
```

ホスト側のポート`5434`が使用中の場合は、`docker/.env`の`POSTGRES_PORT`を変更してください。
`.env`のユーザー名・データベース名を変更した場合は、`docker exec`の引数も合わせて変更してください。

