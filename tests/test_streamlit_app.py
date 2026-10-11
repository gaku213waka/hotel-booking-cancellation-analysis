"""Streamlitアプリが起動し、入力変更を反映できることを確認する。"""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitAppTest(unittest.TestCase):
    def test_app_runs_and_sliders_update_metrics(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[4].value, "¥395,400")

        app.sidebar.slider[0].set_value(0)
        app.sidebar.slider[2].set_value(0)
        app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[2].value, "¥0")
        self.assertEqual(app.metric[1].value, "0.0件")
        self.assertEqual(app.metric[6].value, "算出不可")


if __name__ == "__main__":
    unittest.main()
