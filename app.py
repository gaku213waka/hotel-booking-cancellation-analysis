"""ホテル予約キャンセル対策の費用対効果シミュレーター。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.roi_simulator import (
    ACTUAL_CANCELLATIONS,
    DEFAULT_INTERVENTION_COST,
    DEFAULT_PROFIT_PER_PREVENTED,
    DEFAULT_REDUCTION_RATE,
    TARGET_BOOKINGS,
    build_reduction_curve,
    build_sensitivity_table,
    calculate_roi,
)


st.set_page_config(
    page_title="ホテルキャンセル対策 ROIシミュレーター",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 1.8rem;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    @media (max-width: 640px) {
        .block-container {padding: 1rem 0.8rem;}
        [data-testid="stMetricValue"] {font-size: 1.25rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def yen(value: float) -> str:
    """金額を3桁区切りで表示する。"""

    return f"¥{value:,.0f}"


def percent(value: float | None) -> str:
    """割合を表示する。計算不能の場合は説明文を返す。"""

    return "算出不可" if value is None else f"{value:.1%}"


def profit_color(value: float) -> str:
    """差引利益の符号に応じた色を返す。"""

    return "#13795b" if value >= 0 else "#c0392b"


st.title("🏨 ホテルキャンセル対策 ROIシミュレーター")
st.caption(
    "確定済みRandom ForestモデルのFinal Test集計を使い、施策条件を変えた場合の費用対効果を試算します。"
)

st.info(
    "この画面は過去データを使った仮想的な試算です。施策コスト、追加利益、キャンセル削減率は仮定であり、"
    "将来の効果を保証しません。実際の効果はA/Bテストで確認する必要があります。"
)

with st.sidebar:
    st.header("試算条件")
    intervention_cost = st.slider(
        "施策コスト（1予約あたり）",
        min_value=0,
        max_value=1_000,
        value=DEFAULT_INTERVENTION_COST,
        step=50,
        format="%d円",
    )
    profit_per_prevented = st.slider(
        "キャンセル防止1件あたりの追加利益",
        min_value=3_000,
        max_value=30_000,
        value=DEFAULT_PROFIT_PER_PREVENTED,
        step=1_000,
        format="%d円",
    )
    reduction_percent = st.slider(
        "キャンセル削減率",
        min_value=0,
        max_value=20,
        value=int(DEFAULT_REDUCTION_RATE * 100),
        step=1,
        format="%d%%",
    )
    st.divider()
    st.caption("過去Final Testに基づく固定値")
    st.write(f"施策対象：**{TARGET_BOOKINGS:,}件**")
    st.write(f"対象内の実キャンセル：**{ACTUAL_CANCELLATIONS:,}件**")

reduction_rate = reduction_percent / 100
result = calculate_roi(
    intervention_cost=intervention_cost,
    profit_per_prevented=profit_per_prevented,
    reduction_rate=reduction_rate,
)

st.subheader("試算結果")
row1 = st.columns(4)
row1[0].metric("施策対象件数", f"{result.target_bookings:,}件")
row1[1].metric("想定キャンセル防止件数", f"{result.prevented_cancellations:,.1f}件")
row1[2].metric("施策費用合計", yen(result.total_cost))
row1[3].metric("追加利益合計", yen(result.incremental_profit))

row2 = st.columns(3)
row2[0].metric("差引利益", yen(result.net_profit))
row2[1].metric("損益分岐キャンセル削減率", percent(result.break_even_rate))
row2[2].metric("ROI（差引利益 ÷ 施策費用）", percent(result.roi))

status_text = "黒字の試算" if result.net_profit >= 0 else "赤字の試算"
st.markdown(
    f"<div style='padding:0.85rem 1rem;border-left:6px solid {profit_color(result.net_profit)};"
    f"background:#ffffff;border-radius:8px;margin:0.8rem 0 1.2rem;'>"
    f"<strong style='color:{profit_color(result.net_profit)}'>{status_text}</strong>："
    f"差引利益は <strong>{yen(result.net_profit)}</strong> です。</div>",
    unsafe_allow_html=True,
)

st.subheader("キャンセル削減率と差引利益")
curve = build_reduction_curve(
    intervention_cost=intervention_cost,
    profit_per_prevented=profit_per_prevented,
    reduction_rates=np.linspace(0, 0.20, 41),
)

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(curve["reduction_rate"] * 100, curve["net_profit"], color="#247BA0", linewidth=2.6)
ax.axhline(0, color="#374151", linewidth=1.1)
ax.axvline(reduction_percent, color="#E07A5F", linestyle="--", label="Selected rate")
if result.break_even_rate is not None and 0 <= result.break_even_rate <= 0.20:
    ax.axvline(
        result.break_even_rate * 100,
        color="#13795b",
        linestyle=":",
        linewidth=2,
        label=f"Break-even {result.break_even_rate:.1%}",
    )
ax.fill_between(
    curve["reduction_rate"] * 100,
    curve["net_profit"],
    0,
    where=curve["net_profit"] >= 0,
    color="#84CCB5",
    alpha=0.25,
)
ax.fill_between(
    curve["reduction_rate"] * 100,
    curve["net_profit"],
    0,
    where=curve["net_profit"] < 0,
    color="#F2A6A0",
    alpha=0.25,
)
ax.set(title="Net Profit by Cancellation Reduction Rate", xlabel="Cancellation reduction rate (%)", ylabel="Net profit (JPY)")
ax.grid(alpha=0.2)
ax.legend(frameon=False)
st.pyplot(fig, width="stretch")
plt.close(fig)

st.subheader("施策コスト × 追加利益の感度分析")
st.caption(f"現在選択中のキャンセル削減率（{reduction_rate:.0%}）で比較します。緑が黒字、赤が赤字です。")

cost_options = [0, 100, 300, 500, 750, 1_000]
profit_options = [3_000, 6_000, 9_000, 12_000, 18_000, 24_000, 30_000]
heatmap_data = build_sensitivity_table(
    intervention_costs=cost_options,
    profits_per_prevented=profit_options,
    reduction_rates=[reduction_rate],
)
heatmap_pivot = heatmap_data.pivot(
    index="intervention_cost", columns="profit_per_prevented", values="net_profit"
)

fig, ax = plt.subplots(figsize=(10, 5.4))
sns.heatmap(
    heatmap_pivot / 1_000_000,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Net profit (million JPY)"},
    ax=ax,
)
ax.set(title="Net Profit Sensitivity", xlabel="Profit per prevented cancellation (JPY)", ylabel="Intervention cost per booking (JPY)")
st.pyplot(fig, width="stretch")
plt.close(fig)

st.subheader("条件別の比較")
scenario_rates = sorted({0.00, 0.05, 0.10, 0.15, 0.20, reduction_rate})
scenario_table = build_sensitivity_table(
    intervention_costs=[intervention_cost],
    profits_per_prevented=[profit_per_prevented],
    reduction_rates=scenario_rates,
)[
    [
        "reduction_rate",
        "prevented_cancellations",
        "total_cost",
        "incremental_profit",
        "net_profit",
        "roi",
        "is_profitable",
    ]
].copy()
scenario_table.columns = [
    "削減率",
    "防止件数",
    "施策費用",
    "追加利益",
    "差引利益",
    "ROI",
    "判定",
]
scenario_table["削減率"] = scenario_table["削減率"].map(lambda x: f"{x:.0%}")
scenario_table["防止件数"] = scenario_table["防止件数"].map(lambda x: f"{x:,.1f}件")
for column in ["施策費用", "追加利益", "差引利益"]:
    scenario_table[column] = scenario_table[column].map(yen)
scenario_table["ROI"] = scenario_table["ROI"].map(percent)
scenario_table["判定"] = scenario_table["判定"].map({True: "黒字", False: "赤字"})
st.dataframe(scenario_table, hide_index=True, width="stretch")

with st.expander("計算方法と分析上の注意"):
    st.markdown(
        """
        - **施策費用** = 施策対象件数 × 1件あたり施策コスト
        - **追加利益合計** = 対象内の実キャンセル件数 × キャンセル削減率 × 防止1件あたり追加利益
        - **差引利益** = 追加利益合計 − 施策費用
        - **損益分岐削減率** = 施策費用 ÷（対象内の実キャンセル件数 × 防止1件あたり追加利益）
        - **ROI** = 差引利益 ÷ 施策費用（費用0円では算出不可）

        12,022件と6,670件は過去のFinal Testから得た集計値です。正解ラベルはこの事後評価と試算だけに使い、
        将来予約の予測入力には使いません。300円・12,000円・削減率は仮定値で、ADRそのものを利益とは扱っていません。
        実運用前に、作業時間・連絡チャネル費用・限界利益・キャンセル料・再販売率を確認し、A/Bテストで削減率を検証する必要があります。
        """
    )
