"""02_Cancellation_Prediction.ipynb の最終Pipelineを保存する補助スクリプト。

NotebookでValidationまでに確定した設定（country上位17、agent上位10、
正則化済みRandom Forest）を変更せず、TrainとValidationを結合して学習する。
Final Testは保存後の評価確認にのみ使用する。
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.model_components import TopAgentGrouper, TopCountryGrouper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_random_forest_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "final_random_forest_pipeline_metrics.json"

TARGET = "is_canceled"
COUNTRY = "country"
FINAL_THRESHOLD = 0.30
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 18,
    "min_samples_leaf": 4,
    "min_samples_split": 8,
    "random_state": 42,
    "n_jobs": -1,
}


def calculate_metrics(y_true, y_pred, y_probability) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_probability),
        "PR-AUC": average_precision_score(y_true, y_probability),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH).rename(columns={" ": "hotel"})
    post_booking = ["reservation_status", "reservation_status_date", "assigned_room_type", "booking_changes", "days_in_waiting_list"]
    uncertain = ["deposit_type", "adr", "total_of_special_requests"]
    excluded = set([TARGET, COUNTRY, "agent", "company", *post_booking, *uncertain])
    baseline_features = [column for column in df.columns if column not in excluded]
    feature_columns = baseline_features + [COUNTRY, "agent"]

    arrival_month = pd.to_datetime(df["arrival_date_month"], format="%B").dt.month
    arrival_date = pd.to_datetime({"year": df["arrival_date_year"], "month": arrival_month, "day": df["arrival_date_day_of_month"]})
    booking_date = arrival_date - pd.to_timedelta(df["lead_time"], unit="D")
    train_cutoff = booking_date.quantile(0.60, interpolation="lower")
    validation_cutoff = booking_date.quantile(0.80, interpolation="lower")
    development_index = df.index[booking_date <= validation_cutoff]
    final_test_index = df.index[booking_date > validation_cutoff]

    train_baseline_data = df.loc[df.index[booking_date <= train_cutoff], baseline_features]
    numeric_columns = train_baseline_data.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = train_baseline_data.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Missing")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns),
        ("country", Pipeline([("group", TopCountryGrouper(top_n=17)), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), [COUNTRY]),
        ("agent", Pipeline([("group", TopAgentGrouper(top_n=10)), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["agent"]),
    ], remainder="drop")
    final_model = Pipeline([("preprocess", preprocessor), ("model", RandomForestClassifier(**MODEL_PARAMS))])

    final_model.fit(df.loc[development_index, feature_columns], df.loc[development_index, TARGET])
    X_final_test = df.loc[final_test_index, feature_columns]
    y_final_test = df.loc[final_test_index, TARGET]
    probability = final_model.predict_proba(X_final_test)[:, 1]
    metrics = calculate_metrics(y_final_test, (probability >= FINAL_THRESHOLD).astype(int), probability)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    reloaded_model = joblib.load(MODEL_PATH)
    reloaded_probability = reloaded_model.predict_proba(X_final_test.iloc[:5])[:, 1]
    if not np.allclose(probability[:5], reloaded_probability):
        raise RuntimeError("Reloaded Pipeline predictions do not match the saved model.")

    METRICS_PATH.write_text(json.dumps({"threshold": FINAL_THRESHOLD, "metrics": metrics, "final_test_rows": len(X_final_test)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved_model=", MODEL_PATH)
    print("model_bytes=", MODEL_PATH.stat().st_size)
    print("feature_names_in_=", list(reloaded_model.feature_names_in_))
    print("top_countries=", reloaded_model.named_steps["preprocess"].named_transformers_["country"].named_steps["group"].top_countries_)
    print("top_agents=", reloaded_model.named_steps["preprocess"].named_transformers_["agent"].named_steps["group"].top_agents_)
    print("final_test_metrics=", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
