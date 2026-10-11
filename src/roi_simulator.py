"""ホテルのキャンセル対策に関する費用対効果の計算処理。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

import pandas as pd


# 02_Cancellation_Prediction.ipynb の確定済みモデルを Final Test に適用した集計値
TARGET_BOOKINGS = 12_022
ACTUAL_CANCELLATIONS = 6_670

# 04_Business_Recommendations.ipynb で使用した仮定値
DEFAULT_INTERVENTION_COST = 300
DEFAULT_PROFIT_PER_PREVENTED = 12_000
DEFAULT_REDUCTION_RATE = 0.05


@dataclass(frozen=True)
class ROIResult:
    """1つの施策条件に対する試算結果。"""

    target_bookings: int
    actual_cancellations: int
    intervention_cost: float
    profit_per_prevented: float
    reduction_rate: float
    prevented_cancellations: float
    total_cost: float
    incremental_profit: float
    net_profit: float
    break_even_rate: float | None
    roi: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        """DataFrameへ変換しやすい辞書を返す。"""

        return asdict(self)


def _validate_inputs(
    target_bookings: int,
    actual_cancellations: int,
    intervention_cost: float,
    profit_per_prevented: float,
    reduction_rate: float,
) -> None:
    """入力値の範囲と件数の整合性を確認する。"""

    if target_bookings < 0 or actual_cancellations < 0:
        raise ValueError("予約件数とキャンセル件数は0以上で指定してください。")
    if actual_cancellations > target_bookings:
        raise ValueError("キャンセル件数は施策対象件数以下である必要があります。")
    if intervention_cost < 0 or profit_per_prevented < 0:
        raise ValueError("費用と追加利益は0以上で指定してください。")
    if not 0 <= reduction_rate <= 1:
        raise ValueError("キャンセル削減率は0から1の範囲で指定してください。")


def calculate_roi(
    *,
    target_bookings: int = TARGET_BOOKINGS,
    actual_cancellations: int = ACTUAL_CANCELLATIONS,
    intervention_cost: float = DEFAULT_INTERVENTION_COST,
    profit_per_prevented: float = DEFAULT_PROFIT_PER_PREVENTED,
    reduction_rate: float = DEFAULT_REDUCTION_RATE,
) -> ROIResult:
    """施策の費用、追加利益、差引利益、損益分岐点、ROIを計算する。

    ROIは差引利益 ÷ 施策費用とする。施策費用が0円のときは0除算を避け、
    ROIを ``None`` として返す。
    """

    _validate_inputs(
        target_bookings,
        actual_cancellations,
        intervention_cost,
        profit_per_prevented,
        reduction_rate,
    )

    total_cost = target_bookings * intervention_cost
    prevented_cancellations = actual_cancellations * reduction_rate
    incremental_profit = prevented_cancellations * profit_per_prevented
    net_profit = incremental_profit - total_cost

    break_even_denominator = actual_cancellations * profit_per_prevented
    break_even_rate = (
        total_cost / break_even_denominator if break_even_denominator > 0 else None
    )
    roi = net_profit / total_cost if total_cost > 0 else None

    return ROIResult(
        target_bookings=target_bookings,
        actual_cancellations=actual_cancellations,
        intervention_cost=float(intervention_cost),
        profit_per_prevented=float(profit_per_prevented),
        reduction_rate=float(reduction_rate),
        prevented_cancellations=float(prevented_cancellations),
        total_cost=float(total_cost),
        incremental_profit=float(incremental_profit),
        net_profit=float(net_profit),
        break_even_rate=float(break_even_rate) if break_even_rate is not None else None,
        roi=float(roi) if roi is not None else None,
    )


def build_sensitivity_table(
    *,
    intervention_costs: Iterable[float],
    profits_per_prevented: Iterable[float],
    reduction_rates: Iterable[float],
    target_bookings: int = TARGET_BOOKINGS,
    actual_cancellations: int = ACTUAL_CANCELLATIONS,
) -> pd.DataFrame:
    """費用・追加利益・削減率の全組み合わせを比較する表を作る。"""

    rows = []
    for cost, profit, rate in product(
        intervention_costs, profits_per_prevented, reduction_rates
    ):
        result = calculate_roi(
            target_bookings=target_bookings,
            actual_cancellations=actual_cancellations,
            intervention_cost=cost,
            profit_per_prevented=profit,
            reduction_rate=rate,
        )
        row = result.to_dict()
        row["is_profitable"] = result.net_profit >= 0
        rows.append(row)

    return pd.DataFrame(rows)


def build_reduction_curve(
    *,
    intervention_cost: float,
    profit_per_prevented: float,
    reduction_rates: Iterable[float],
    target_bookings: int = TARGET_BOOKINGS,
    actual_cancellations: int = ACTUAL_CANCELLATIONS,
) -> pd.DataFrame:
    """削減率ごとの試算結果を折れ線グラフ用の表として返す。"""

    return build_sensitivity_table(
        intervention_costs=[intervention_cost],
        profits_per_prevented=[profit_per_prevented],
        reduction_rates=reduction_rates,
        target_bookings=target_bookings,
        actual_cancellations=actual_cancellations,
    )
