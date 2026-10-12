"""既存のホテル予約キャンセル分析成果を表示するための共通処理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"
SOURCE_DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "main/data/2020/2020-02-11/hotels.csv"
)
EXPECTED_DATA_SHAPE = (119_390, 32)
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "final_random_forest_pipeline.joblib",
    PROJECT_ROOT / "models" / "final_model.joblib",
]

LEAD_TIME_BINS = [-1, 7, 30, 60, 120, 180, 365, np.inf]
LEAD_TIME_LABELS = ["0–7", "8–30", "31–60", "61–120", "121–180", "181–365", "366+"]

# 02_Cancellation_Prediction.ipynb の保存済みFinal Test評価値（閾値0.30）
FINAL_TEST_METRICS = {
    "Accuracy": 0.7407,
    "Precision": 0.5548,
    "Recall": 0.8935,
    "F1-score": 0.6846,
    "ROC-AUC": 0.8622,
    "PR-AUC": 0.7362,
}
FINAL_TEST_CONFUSION = {
    "True negative": 10_886,
    "False positive": 5_352,
    "False negative": 795,
    "True positive": 6_670,
}
HIGH_RISK_SUMMARY = {
    "threshold": 0.30,
    "target_bookings": 12_022,
    "target_share": 0.5072,
    "actual_cancellations": 6_670,
    "precision": 0.5548,
    "recall": 0.8935,
}

# 03_SHAP_Analysis.ipynbの保存済み集約結果。新しいSHAP計算は行わない。
SHAP_GROUPED_IMPORTANCE = pd.DataFrame(
    {
        "feature": [
            "country", "agent", "customer_type", "market_segment", "lead_time",
            "arrival_date_year", "previous_cancellations", "required_car_parking_spaces",
            "distribution_channel", "hotel",
        ],
        "mean_abs_shap": [
            0.147287, 0.066191, 0.060387, 0.055912, 0.053869,
            0.031646, 0.026653, 0.022186, 0.017488, 0.015620,
        ],
    }
)


def load_booking_data(path: Path = DATA_PATH) -> pd.DataFrame | None:
    """有効なCSVがある場合だけ読み込む。欠損時に代替データは作らない。"""
    if path.exists() and path.stat().st_size >= 1_000:
        data = pd.read_csv(path)
    else:
        # Git管理しない原データは、同一内容を公開しているTidyTuesdayから取得する。
        # 取得先とCC BY 4.0の出典はREADMEとアプリ内に明記する。
        try:
            data = pd.read_csv(SOURCE_DATA_URL)
        except Exception:
            return None
    if " " in data.columns:
        data = data.rename(columns={" ": "hotel"})
    required = {"hotel", "market_segment", "lead_time", "is_canceled"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {sorted(missing)}")
    if tuple(data.shape) != EXPECTED_DATA_SHAPE:
        raise ValueError(
            f"想定と異なるデータサイズです: {data.shape} "
            f"(expected: {EXPECTED_DATA_SHAPE})"
        )
    return data


def filter_bookings(data: pd.DataFrame, hotel: str = "全体", market_segment: str = "全体") -> pd.DataFrame:
    """OverviewとEDAで共通利用するフィルターを適用する。"""
    filtered = data.copy()
    if hotel != "全体":
        filtered = filtered.loc[filtered["hotel"].eq(hotel)]
    if market_segment != "全体":
        filtered = filtered.loc[filtered["market_segment"].eq(market_segment)]
    return filtered


def summarize_cancellations(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """区分ごとの予約件数・キャンセル件数・キャンセル率を集計する。"""
    summary = (
        data.groupby(column, dropna=False)["is_canceled"]
        .agg(予約件数="size", キャンセル件数="sum", キャンセル率="mean")
        .reset_index()
    )
    return summary.sort_values("予約件数", ascending=False)


def lead_time_summary(data: pd.DataFrame) -> pd.DataFrame:
    """既存Notebookと同じLead Time区間で集計する。"""
    bands = pd.cut(data["lead_time"], bins=LEAD_TIME_BINS, labels=LEAD_TIME_LABELS)
    return (
        data.assign(lead_time_band=bands)
        .groupby("lead_time_band", observed=False)["is_canceled"]
        .agg(予約件数="size", キャンセル件数="sum", キャンセル率="mean")
        .reset_index()
    )


def development_subset(data: pd.DataFrame) -> pd.DataFrame:
    """既存時系列分割のTrain部分を、フォーム初期値の参照範囲として返す。"""
    required = {"arrival_date_year", "arrival_date_month", "arrival_date_day_of_month", "lead_time"}
    if not required.issubset(data.columns):
        return data
    month = pd.to_datetime(data["arrival_date_month"], format="%B", errors="coerce").dt.month
    arrival = pd.to_datetime(
        {"year": data["arrival_date_year"], "month": month, "day": data["arrival_date_day_of_month"]},
        errors="coerce",
    )
    booking_date = arrival - pd.to_timedelta(data["lead_time"], unit="D")
    cutoff = booking_date.quantile(0.60, interpolation="lower")
    subset = data.loc[booking_date.le(cutoff)]
    return subset if not subset.empty else data


def find_model_path() -> Path | None:
    """既存の保存済みPipelineだけを探す。再学習は行わない。"""
    return next((path for path in MODEL_CANDIDATES if path.exists()), None)


def load_saved_model() -> Any | None:
    """保存済みのscikit-learn Pipelineを読み込む。"""
    model_path = find_model_path()
    if model_path is None:
        return None
    import joblib
    return joblib.load(model_path)


def model_feature_names(model: Any) -> list[str]:
    """保存済みPipelineが受け取る生の特徴量名を返す。"""
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else []
