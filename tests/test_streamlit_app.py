"""閲覧ダッシュボードがデータ・モデル未配置時にも起動することを確認する。"""

from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class StreamlitDashboardTest(unittest.TestCase):
    def test_dashboard_runs_without_data_or_model(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Overview")
        options = app.sidebar.radio[0].options
        self.assertEqual(options, ["Overview", "EDA", "Prediction", "Recommendations"])

    def test_all_pages_switch_without_errors(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        for page in ["EDA", "Recommendations"]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.title[0].value, page)

        app.sidebar.radio[0].set_value("Prediction").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Prediction")
        self.assertEqual(app.metric[0].label, "Accuracy")
        self.assertEqual(app.metric[0].value, "0.7407")


if __name__ == "__main__":
    unittest.main()
