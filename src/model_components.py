"""保存済みホテル予約キャンセルモデルで使うカスタム前処理器。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TopCountryGrouper(BaseEstimator, TransformerMixin):
    """Train内の上位国だけを残し、それ以外をOtherへ統合する。"""

    def __init__(self, top_n: int = 17):
        self.top_n = top_n

    @staticmethod
    def _to_series(X):
        values = X.iloc[:, 0].copy() if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X).reshape(-1))
        return values.fillna("Unknown").astype(str)

    def fit(self, X, y=None):
        values = self._to_series(X)
        self.top_countries_ = values[values != "Unknown"].value_counts().head(self.top_n).index.tolist()
        return self

    def transform(self, X):
        check_is_fitted(self, "top_countries_")
        values = self._to_series(X)
        return values.where(values.isin(self.top_countries_) | values.eq("Unknown"), "Other").to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["country"], dtype=object)


class TopAgentGrouper(BaseEstimator, TransformerMixin):
    """Train内の上位代理店だけを残し、それ以外をOtherへ統合する。"""

    def __init__(self, top_n: int = 10):
        self.top_n = top_n

    @staticmethod
    def _normalize_value(value):
        if pd.isna(value):
            return "Unknown"
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric_value = float(value)
            if np.isfinite(numeric_value) and numeric_value.is_integer():
                return str(int(numeric_value))
            return format(numeric_value, "g")
        return str(value).strip()

    @classmethod
    def _to_series(cls, X):
        values = X.iloc[:, 0].copy() if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X).reshape(-1))
        return values.map(cls._normalize_value)

    def fit(self, X, y=None):
        values = self._to_series(X)
        self.top_agents_ = values[values != "Unknown"].value_counts().head(self.top_n).index.tolist()
        return self

    def transform(self, X):
        check_is_fitted(self, "top_agents_")
        values = self._to_series(X)
        return values.where(values.isin(self.top_agents_) | values.eq("Unknown"), "Other").to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["agent"], dtype=object)
