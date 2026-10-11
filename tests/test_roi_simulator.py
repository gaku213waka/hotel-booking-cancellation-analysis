"""費用対効果の計算処理を検証する。"""

from __future__ import annotations

import math
import unittest

from src.roi_simulator import (
    ACTUAL_CANCELLATIONS,
    DEFAULT_PROFIT_PER_PREVENTED,
    TARGET_BOOKINGS,
    calculate_roi,
)


class ROISimulatorTest(unittest.TestCase):
    def test_existing_example_is_reproduced(self) -> None:
        result = calculate_roi()
        self.assertEqual(result.total_cost, 3_606_600)
        self.assertEqual(result.prevented_cancellations, 333.5)
        self.assertEqual(result.incremental_profit, 4_002_000)
        self.assertEqual(result.net_profit, 395_400)

    def test_zero_cost_avoids_division_by_zero(self) -> None:
        result = calculate_roi(intervention_cost=0)
        self.assertEqual(result.total_cost, 0)
        self.assertEqual(result.break_even_rate, 0)
        self.assertIsNone(result.roi)

    def test_zero_reduction_rate(self) -> None:
        result = calculate_roi(reduction_rate=0)
        self.assertEqual(result.prevented_cancellations, 0)
        self.assertEqual(result.incremental_profit, 0)
        self.assertEqual(result.net_profit, -result.total_cost)

    def test_break_even_net_profit_is_effectively_zero(self) -> None:
        break_even_rate = (TARGET_BOOKINGS * 300) / (
            ACTUAL_CANCELLATIONS * DEFAULT_PROFIT_PER_PREVENTED
        )
        result = calculate_roi(reduction_rate=break_even_rate)
        self.assertTrue(math.isclose(result.net_profit, 0, abs_tol=1e-8))

    def test_zero_profit_avoids_division_by_zero(self) -> None:
        result = calculate_roi(profit_per_prevented=0)
        self.assertIsNone(result.break_even_rate)
        self.assertEqual(result.incremental_profit, 0)

    def test_invalid_cancellation_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_roi(target_bookings=10, actual_cancellations=11)


if __name__ == "__main__":
    unittest.main()
