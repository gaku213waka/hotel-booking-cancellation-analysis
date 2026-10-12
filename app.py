"""ホテル予約キャンセル分析の閲覧用Streamlitダッシュボード。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_utils import (
    DATA_PATH, FINAL_TEST_CONFUSION, FINAL_TEST_METRICS, HIGH_RISK_SUMMARY,
    PROJECT_ROOT, SHAP_GROUPED_IMPORTANCE, development_subset, filter_bookings,
    find_model_path, lead_time_summary, load_booking_data, load_saved_model,
    model_feature_names, summarize_cancellations,
)


BACKGROUND, CARD, ACCENT, TEXT, MUTED, GRID = "#111116", "#23232D", "#F062AD", "#FFFFFF", "#B8B8C5", "#393945"

st.set_page_config(page_title="Hotel Booking Cancellation Dashboard", page_icon="🏨", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    f"""<style>
    .stApp {{background:{BACKGROUND};color:{TEXT};}}
    [data-testid="stSidebar"] {{background:#181820;}}
    [data-testid="stMetric"] {{background:{CARD};border:1px solid #363642;border-radius:14px;padding:1rem;min-height:118px;}}
    [data-testid="stMetricLabel"] {{color:{MUTED};}}
    [data-testid="stMetricValue"] {{color:{TEXT};font-size:1.65rem;}}
    .dashboard-card {{background:{CARD};border:1px solid #363642;border-radius:14px;padding:1.2rem 1.3rem;margin-bottom:1rem;}}
    .policy-tag {{color:{ACCENT};font-weight:700;letter-spacing:.03em;}}
    .muted {{color:{MUTED};}}
    h1,h2,h3 {{color:{TEXT};}}
    @media (max-width:720px) {{.block-container {{padding:1rem .75rem;}} [data-testid="stMetricValue"] {{font-size:1.25rem;}}}}
    </style>""",
    unsafe_allow_html=True,
)


def percent(value: float) -> str:
    return f"{value:.1%}"


def chart_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    """全ページ共通の暗色Plotlyデザインを適用する。"""
    fig.update_layout(
        title=title, height=height, paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font={"color": TEXT, "family": "Arial, sans-serif"},
        margin={"l": 20, "r": 20, "t": 55, "b": 20}, legend={"font": {"color": MUTED}},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame | None:
    return load_booking_data()


@st.cache_resource(show_spinner=False)
def get_saved_model():
    return load_saved_model()


def data_notice() -> None:
    st.warning("実データを取得できませんでした。ネットワーク接続を確認して再読み込みしてください。")
    st.caption("ローカルCSVがない場合は、TidyTuesdayの公開データを取得します。架空のKPIやグラフは表示しません。")


def sidebar(data: pd.DataFrame | None) -> tuple[str, str, str]:
    with st.sidebar:
        st.markdown("## 🏨 分析ダッシュボード")
        st.caption("Hotel Booking Cancellation Analysis")
        page = st.radio("ページ", ["Overview", "EDA", "Prediction", "Recommendations"], label_visibility="collapsed")
        st.divider()
        st.caption("データ出典: António et al. (2019), CC BY 4.0 / TidyTuesday")
        st.markdown("### 共通フィルター")
        if data is None:
            st.caption("有効なCSVを配置するとフィルターが有効になります。")
            return page, "全体", "全体"
        hotels = ["全体"] + sorted(data["hotel"].dropna().astype(str).unique().tolist())
        segments = ["全体"] + sorted(data["market_segment"].dropna().astype(str).unique().tolist())
        hotel = st.selectbox("ホテルタイプ", hotels)
        segment = st.selectbox("市場セグメント", segments)
        st.caption("フィルターはOverview・EDAのKPIとグラフに共通で適用されます。")
        return page, hotel, segment


def cancellation_bar(summary: pd.DataFrame, category: str, title: str) -> go.Figure:
    data = summary.sort_values("キャンセル率")
    fig = px.bar(data, x="キャンセル率", y=category, orientation="h", color_discrete_sequence=[ACCENT], custom_data=["予約件数", "キャンセル件数"], text="予約件数")
    fig.update_traces(
        texttemplate="%{text:,}件", textposition="outside",
        hovertemplate="%{y}<br>キャンセル率: %{x:.1%}<br>予約件数: %{customdata[0]:,}件<br>キャンセル件数: %{customdata[1]:,}件<extra></extra>",
    )
    fig.update_xaxes(tickformat=".0%", rangemode="tozero")
    return chart_layout(fig, title)


def render_overview(data: pd.DataFrame | None, hotel: str, segment: str) -> None:
    st.title("Overview")
    st.caption("予約全体の規模と、ホテルタイプ・市場セグメント・Lead Time別のキャンセル傾向を確認します。")
    if data is None:
        data_notice()
        return
    filtered = filter_bookings(data, hotel, segment)
    if filtered.empty:
        st.info("現在のフィルター条件に該当する予約がありません。")
        return
    metrics = st.columns(4)
    metrics[0].metric("総予約件数", f"{len(filtered):,}件")
    metrics[1].metric("キャンセル率", percent(float(filtered["is_canceled"].mean())))
    metrics[2].metric("キャンセル件数", f"{int(filtered['is_canceled'].sum()):,}件")
    metrics[3].metric("平均リードタイム", f"{filtered['lead_time'].mean():.1f}日")
    if segment != "全体":
        st.caption(f"市場セグメントを「{segment}」に絞っているため、セグメント別グラフは単一区分です。")
    if hotel != "全体":
        st.caption(f"ホテルタイプを「{hotel}」に絞っているため、ホテルタイプ別グラフは単一区分です。")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(cancellation_bar(summarize_cancellations(filtered, "market_segment"), "market_segment", "市場セグメント別キャンセル率"), width="stretch")
    with right:
        summary = summarize_cancellations(filtered, "hotel")
        fig = px.bar(summary, x="hotel", y="キャンセル率", color_discrete_sequence=[ACCENT], custom_data=["予約件数", "キャンセル件数"], text="予約件数")
        fig.update_traces(texttemplate="%{text:,}件", hovertemplate="%{x}<br>キャンセル率: %{y:.1%}<br>予約件数: %{customdata[0]:,}件<br>キャンセル件数: %{customdata[1]:,}件<extra></extra>")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        st.plotly_chart(chart_layout(fig, "ホテルタイプ別キャンセル率"), width="stretch")
    lead = lead_time_summary(filtered)
    fig = go.Figure(go.Scatter(
        x=lead["lead_time_band"].astype(str), y=lead["キャンセル率"], mode="lines+markers+text",
        line={"color": ACCENT, "width": 3}, marker={"size": 9}, text=[f"{value:,}件" for value in lead["予約件数"]], textposition="top center",
        customdata=np.stack([lead["予約件数"], lead["キャンセル件数"]], axis=-1),
        hovertemplate="Lead Time: %{x}日<br>キャンセル率: %{y:.1%}<br>予約件数: %{customdata[0]:,}件<br>キャンセル件数: %{customdata[1]:,}件<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    st.plotly_chart(chart_layout(fig, "Lead Time区間別キャンセル率", 420), width="stretch")


def render_eda(data: pd.DataFrame | None, hotel: str, segment: str) -> None:
    st.title("EDA")
    st.caption("全期間の記述統計として、予約属性とキャンセル率の関係を確認します。因果関係を示す分析ではありません。")
    if data is None:
        data_notice()
        return
    filtered = filter_bookings(data, hotel, segment)
    if filtered.empty:
        st.info("現在のフィルター条件に該当する予約がありません。")
        return
    if "country" in filtered.columns:
        country = filtered.assign(country=filtered["country"].fillna("Unknown").astype(str))
        top = country["country"].value_counts().head(12).index
        st.plotly_chart(cancellation_bar(summarize_cancellations(country.loc[country["country"].isin(top)], "country"), "country", "予約件数上位12か国のキャンセル率"), width="stretch")
        st.caption("予約件数上位の国だけを表示しています。少数データの高い率は、件数と合わせて解釈します。")
    left, right = st.columns(2)
    with left:
        if "is_repeated_guest" in filtered.columns:
            repeat = filtered.assign(リピーター区分=np.where(filtered["is_repeated_guest"].eq(1), "リピーター", "新規客"))
            summary = summarize_cancellations(repeat, "リピーター区分")
            fig = px.bar(summary, x="リピーター区分", y="キャンセル率", color="リピーター区分", color_discrete_sequence=[ACCENT, "#B8B8C5"], custom_data=["予約件数", "キャンセル件数"], text="予約件数")
            fig.update_traces(texttemplate="%{text:,}件", hovertemplate="%{x}<br>キャンセル率: %{y:.1%}<br>予約件数: %{customdata[0]:,}件<br>キャンセル件数: %{customdata[1]:,}件<extra></extra>")
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
            st.plotly_chart(chart_layout(fig, "新規客・リピーター別キャンセル率"), width="stretch")
            st.caption("リピーター区分とキャンセル率の差は、他の属性の影響を分離していません。")
    with right:
        if {"adr", "lead_time"}.issubset(filtered.columns):
            scatter = filtered[["adr", "lead_time", "is_canceled"]].dropna()
            if len(scatter) > 10_000:
                scatter = scatter.sample(10_000, random_state=42)
            scatter = scatter.assign(キャンセル=np.where(scatter["is_canceled"].eq(1), "キャンセル", "非キャンセル"))
            fig = px.scatter(scatter, x="lead_time", y="adr", color="キャンセル", color_discrete_map={"キャンセル": ACCENT, "非キャンセル": "#7C7C8D"}, opacity=0.45)
            st.plotly_chart(chart_layout(fig, "ADR × Lead Time", 420), width="stretch")
            st.caption("元データはADRの通貨単位を明記していないため、通貨は断定していません。散布図は関連を示すだけです。")


def build_prediction_input(model, data: pd.DataFrame) -> pd.DataFrame | None:
    """保存済みPipelineの入力列に合わせた1件分のDataFrameを作る。"""
    feature_names = model_feature_names(model)
    if not feature_names:
        st.error("保存済みPipelineから入力特徴量名を取得できません。02の最終Pipelineと同じ形式のモデルが必要です。")
        return None
    defaults, values = development_subset(data), {}
    basic_names = {"hotel", "lead_time", "market_segment", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"}
    basic = [name for name in feature_names if name in basic_names]
    detail = [name for name in feature_names if name not in basic_names]
    def fields(names: list[str]) -> None:
        for name in names:
            series = defaults[name] if name in defaults.columns else pd.Series(dtype="object")
            if pd.api.types.is_numeric_dtype(series):
                default = float(series.median()) if series.notna().any() else 0.0
                values[name] = st.number_input(name.replace("_", " "), value=default, step=1.0 if np.isclose(default, round(default)) else 0.1, key=f"prediction_{name}")
            else:
                candidates = series.dropna().astype(str).value_counts().index.tolist()
                default = candidates[0] if candidates else "Unknown"
                values[name] = st.selectbox(name.replace("_", " "), candidates, key=f"prediction_{name}") if len(candidates) <= 40 else st.text_input(name.replace("_", " "), value=default, key=f"prediction_{name}")
    st.caption("初期値は、既存の時系列分割におけるTrain部分の中央値または最頻値です。")
    left, right = st.columns(2)
    split = (len(basic) + 1) // 2
    with left: fields(basic[:split])
    with right: fields(basic[split:])
    with st.expander("詳細項目"): fields(detail)
    return pd.DataFrame([{name: values[name] for name in feature_names}])


def render_prediction(data: pd.DataFrame | None) -> None:
    st.title("Prediction")
    st.caption("02で確定したRandom Forestの既存評価と、保存済みPipelineによる1件予測を表示します。")
    model_path, model = find_model_path(), get_saved_model() if find_model_path() else None
    if data is not None and model is not None:
        with st.form("prediction_form"):
            input_frame = build_prediction_input(model, data)
            submitted = st.form_submit_button("キャンセルリスクを予測")
        if submitted and input_frame is not None:
            probability = float(model.predict_proba(input_frame)[:, 1][0])
            label = "高リスク" if probability >= HIGH_RISK_SUMMARY["threshold"] else "低リスク"
            st.markdown(f"<div class='dashboard-card'><span class='policy-tag'>{label}</span><h2>{probability:.1%}</h2><p class='muted'>既存閾値 0.30 による判定</p></div>", unsafe_allow_html=True)
            st.progress(min(max(probability, 0.0), 1.0), text=f"予測キャンセル確率: {probability:.1%}")
            st.caption("予測確率は将来のキャンセルを保証するものではありません。")
    else:
        st.info("保存済みの最終Pipelineが未配置のため、個別予測フォームは無効です。モデルを再学習して代替することは行っていません。")
        st.caption("想定配置先: `models/final_random_forest_pipeline.joblib`。有効なCSVも必要です。")
    st.subheader("Final Testでの評価（既存結果）")
    metrics = st.columns(4)
    for column, name in zip(metrics, ["Accuracy", "Precision", "Recall", "F1-score"]):
        column.metric(name, f"{FINAL_TEST_METRICS[name]:.4f}")
    matrix = np.array([[FINAL_TEST_CONFUSION["True negative"], FINAL_TEST_CONFUSION["False positive"]], [FINAL_TEST_CONFUSION["False negative"], FINAL_TEST_CONFUSION["True positive"]]])
    fig = go.Figure(go.Heatmap(z=matrix, x=["予測: 非キャンセル", "予測: キャンセル"], y=["実際: 非キャンセル", "実際: キャンセル"], colorscale=[[0, "#2C2C37"], [1, ACCENT]], showscale=False, text=matrix, texttemplate="%{text:,}"))
    st.plotly_chart(chart_layout(fig, "Final Test混同行列（閾値0.30）", 360), width="stretch")
    st.subheader("SHAPによる特徴量重要度（既存結果）")
    shap_data = SHAP_GROUPED_IMPORTANCE.sort_values("mean_abs_shap")
    fig = px.bar(shap_data, x="mean_abs_shap", y="feature", orientation="h", color_discrete_sequence=[ACCENT])
    fig.update_traces(hovertemplate="%{y}<br>mean(|SHAP value|): %{x:.4f}<extra></extra>")
    st.plotly_chart(chart_layout(fig, "元特徴量単位のSHAP重要度", 440), width="stretch")
    st.caption("`country`、`agent`、`customer_type`、`market_segment`、`lead_time`の順に寄与が大きい既存結果です。SHAPはモデルの予測根拠であり、因果関係を示しません。")


def policy_card(title: str, target: str, body: str, evidence: str, test: str) -> None:
    st.markdown(f"<div class='dashboard-card'><div class='policy-tag'>{title}</div><h3>対象：{target}</h3><p>{body}</p><p class='muted'><b>分析根拠：</b>{evidence}</p><p class='muted'><b>効果検証：</b>{test}</p></div>", unsafe_allow_html=True)


def render_recommendations(data: pd.DataFrame | None) -> None:
    st.title("Recommendations")
    st.caption("04_Business_Recommendations.ipynbで整理した3施策です。施策によるキャンセル削減効果は未検証です。")
    if data is not None:
        lead_mask, groups_mask = data["lead_time"].ge(61), data["market_segment"].eq("Groups")
        evidence_1 = f"全期間で{int(lead_mask.sum()):,}件、キャンセル率{data.loc[lead_mask, 'is_canceled'].mean():.1%}。"
        evidence_2 = f"全期間で{int(groups_mask.sum()):,}件、キャンセル率{data.loc[groups_mask, 'is_canceled'].mean():.1%}。"
    else:
        evidence_1 = "有効なCSVを配置すると、全期間の対象件数・キャンセル率を表示します。"
        evidence_2 = "有効なCSVを配置すると、Groupsの対象件数・キャンセル率を表示します。"
    policy_card("施策① 早期予約者へのフォロー", "lead_time >= 61日", "予約直後に内容を案内し、宿泊30日前・7日前に自動メールで予定変更の有無を確認する。", evidence_1, "高リスク予約を施策群・比較群へ無作為に分け、キャンセル率と早期把握率を比較する。")
    policy_card("施策② Groups予約への確認", "market_segment = Groups", "予約条件、人数、部屋数、日程、支払条件を確認し、宿泊45日前・14日前に再確認する。", evidence_2 + " Groupsは市場区分であり、実際の団体人数を直接示すわけではありません。", "施策群・比較群でキャンセル率、回答率、在庫解放までの日数を比較する。")
    policy_card("施策③ 高リスク予約への優先対応", "予測キャンセル確率 >= 30%", "対象全体には自動メールを基本とし、特に確率が高い予約は個別対応を優先する。具体的な個別対応基準は未検証。", f"Final Testで{HIGH_RISK_SUMMARY['target_bookings']:,}件（{HIGH_RISK_SUMMARY['target_share']:.1%}）を対象化。対象内キャンセル率{HIGH_RISK_SUMMARY['precision']:.1%}、捕捉率{HIGH_RISK_SUMMARY['recall']:.1%}。", "確定済みモデルで対象を抽出後、施策群・比較群へ無作為に割り付け、キャンセル率と業務負荷を比較する。")
    st.info("A/Bテストでは、同じ予測ルールで抽出した対象を無作為に施策群・比較群へ分けます。キャンセル率だけでなく、連絡コスト、顧客反応、再販売状況も合わせて評価します。")


data = get_data()
page, selected_hotel, selected_segment = sidebar(data)
if page == "Overview":
    render_overview(data, selected_hotel, selected_segment)
elif page == "EDA":
    render_eda(data, selected_hotel, selected_segment)
elif page == "Prediction":
    render_prediction(data)
else:
    render_recommendations(data)
