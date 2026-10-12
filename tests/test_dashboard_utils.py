"""ダッシュボード用の実データ集計処理を検証する。"""

from __future__ import annotations

import unittest

import pandas as pd

from src.dashboard_utils import (
    FINAL_TEST_CONFUSION,
    FINAL_TEST_METRICS,
    filter_bookings,
    lead_time_summary,
    summarize_cancellations,
)


class DashboardUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "hotel": ["City Hotel", "Resort Hotel", "City Hotel"],
                "market_segment": ["Online TA", "Groups", "Groups"],
                "lead_time": [3, 70, 200],
                "is_canceled": [0, 1, 1],
            }
        )

    def test_filters_use_the_same_population(self) -> None:
        filtered = filter_bookings(self.data, hotel="City Hotel", market_segment="Groups")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(int(filtered["is_canceled"].sum()), 1)

    def test_summary_has_count_and_rate(self) -> None:
        summary = summarize_cancellations(self.data, "market_segment").set_index("market_segment")
        self.assertEqual(int(summary.loc["Groups", "予約件数"]), 2)
        self.assertEqual(float(summary.loc["Groups", "キャンセル率"]), 1.0)

    def test_lead_time_uses_existing_bands(self) -> None:
        summary = lead_time_summary(self.data).set_index("lead_time_band")
        self.assertEqual(int(summary.loc["0–7", "予約件数"]), 1)
        self.assertEqual(int(summary.loc["61–120", "予約件数"]), 1)
        self.assertEqual(int(summary.loc["181–365", "予約件数"]), 1)

    def test_final_test_confusion_matches_saved_metrics(self) -> None:
        tn = FINAL_TEST_CONFUSION["True negative"]
        fp = FINAL_TEST_CONFUSION["False positive"]
        fn = FINAL_TEST_CONFUSION["False negative"]
        tp = FINAL_TEST_CONFUSION["True positive"]
        self.assertAlmostEqual((tn + tp) / (tn + fp + fn + tp), FINAL_TEST_METRICS["Accuracy"], places=4)
        self.assertAlmostEqual(tp / (tp + fp), FINAL_TEST_METRICS["Precision"], places=4)
        self.assertAlmostEqual(tp / (tp + fn), FINAL_TEST_METRICS["Recall"], places=4)


if __name__ == "__main__":
    unittest.main()
