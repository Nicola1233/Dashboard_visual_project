from __future__ import annotations

import os
import pandas as pd
import streamlit as st

from charts import (
    build_color_map,
    make_line_steps,
    make_avg_weekly_sleep_stacked_bar,
    make_donut_stress,
)

APP_TITLE = "Daily life of BDMA student"



def load_css(path: str) -> None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    df["calendarDate"] = pd.to_datetime(df["calendarDate"])
    return df


def kpi_card(title: str, subtitle1: str, value_text: str, delta_pct_val: float | None, is_good: bool | None) -> None:
    if delta_pct_val is None or is_good is None:
        delta_html = '<span class="kpi-delta neutral">—</span>'
    else:
        arrow = "▲" if delta_pct_val > 0 else ("▼" if delta_pct_val < 0 else "•")
        cls = "good" if is_good else "bad"
        delta_html = f'<span class="kpi-delta {cls}">{arrow} {abs(delta_pct_val):.1f}%</span>'

    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-subtitle">{subtitle1}</div>
          <div class="kpi-row">
            <div class="kpi-value">{value_text}</div>
            <div class="kpi-right">
              {delta_html}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_mean(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce")
    return None if s.dropna().empty else float(s.mean())


def delta_pct(current: float | None, base: float | None) -> float | None:
    if current is None or base is None or base == 0:
        return None
    return (current - base) / base * 100.0


GOOD_WHEN = {
    "sleep": "up",
    "active": "up",
    "hr": "down",
    "stress": "down",
}


def is_good_change(metric_key: str, d: float | None) -> bool | None:
    if d is None:
        return None
    want = GOOD_WHEN.get(metric_key, "up")
    if d == 0:
        return True
    return (d > 0) if want == "up" else (d < 0)

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    load_css("styles.css")

    if "page" not in st.session_state:
        st.session_state.page = "dashboard"

    if st.session_state.page == "info":
        left, right = st.columns([6, 1])
        with left:
            st.title("Info Dashboard")
        with right:
            if st.button("×", key="back_dashboard"):
                st.session_state.page = "dashboard"
                st.rerun()
        st.markdown("## Data")
        st.markdown(
            """
        The dashboard uses data collected by a Garmin watch via its health and daily activity sensors. The aim is not to obtain “clinical” measurements, but to use these values to understand how certain habits and lifestyle indicators change over time.

        The data considered includes:
        - daily sleep duration
        - kilometers and steps traveled each day
        - heart rate (minimum and maximum values)
        - stress level, divided into four states: high, medium, low, and rest (absence of stress)

        This information forms the basis of all the visualizations in the dashboard.

        To make the analysis more meaningful, I also added the geographical context, i.e., the place where I was when the data was collected. This information was reconstructed by consulting the flights taken during the year, so as to associate the data with five main periods/locations: Barcelona, Brussels, Italy, Morocco, and France.

        In this way, the dashboard not only shows “how the values change over time,” but also allows them to be compared between different contexts, better highlighting the influence of the environment on the changes observed.

        Finally, some metrics (in particular stress) are calculated by Garmin using internal models: for this reason, they should be interpreted primarily in a relative sense, i.e., as useful for comparing periods and situations, rather than as absolute measures.
        """
        )

        st.markdown("## Purpose and context")
        st.markdown(
            """
        The dashboard explores how changes in geographical and social context are associated with variations in physical activity, sleep, and stress levels.

        The development of this dashboard arose from the opportunity to observe these changes systematically, thanks to a period of intense personal mobility. Through the Erasmus Mundus BDMA program, I lived and studied at three different European universities:
        - Université Libre de Bruxelles (ULB)
        - Universitat Politècnica de Catalunya (UPC)
        - CentraleSupélec

        In addition to these periods, I have also spent time in Italy and Morocco, mainly on vacation, which introduced further differences in terms of pace of life and daily activities.

        During these trips, I observed how factors such as:
        - distance between home and university
        - quality of public transport
        - climate
        - academic workload

        have significantly influenced my daily habits, particularly physical activity and stress levels.

        The dashboard uses data collected daily to visualize these changes, allowing for comparisons between different periods and contexts.

        For example, in Barcelona, the urban layout and the distance between home and university encouraged walking, while in Brussels, the proximity to the university and frequent use of public transport, combined with the climate, led to a reduction in daily physical activity. These differences are clearly evident in the patterns shown in the visualizations.

        Similarly, sleep and stress analysis allows us to identify periods of greater irregularity or stress, suggesting how adaptation to new contexts and academic demands have affected overall well-being.

        It is important to emphasize that the dashboard does not aim to establish causal relationships, but to support reflection based on the comparison of contexts.
        """
        )

        st.markdown("## Who it is aimed at")
        st.markdown(
            """
        This dashboard is based on personal data and individual experience, and is not intended as a directly generalizable tool.

        However, it may be relevant for:
        - students involved in international mobility programs
        - people going through periods of transition (change of city, job, or routine)
        - individuals interested in self-tracking and reflecting on their lifestyle

        In this sense, the dashboard's main objective is not to provide practical recommendations, but to show how context influences habits, highlighting the concept of change rather than individual numerical values.
        """
        )

        st.stop()


    if st.button("Info"):
        st.session_state.page = "info"
        st.rerun()

    st.title(APP_TITLE)
    st.markdown(
    "*How context and location shape activity, sleep, and stress over time*"
)
    

    csv_path = st.sidebar.text_input("CSV", value="data/Dataset_User1234.csv")
    df = load_data(csv_path)

    required = ["calendarDate", "totalSteps", "total_sleep", "Label", "avgStress_TOTAL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Columns not found in CSV: {missing}")
        st.stop()

    labels = sorted(df["Label"].dropna().unique().tolist())

    st.sidebar.header("Filter")

    if "selected_labels" not in st.session_state:
        st.session_state.selected_labels = set(labels)

    def set_all_labels(val: bool):
        if val:
            st.session_state.selected_labels = set(labels)
        else:
            st.session_state.selected_labels = set()
        for lab in labels:
            st.session_state[f"chk_{lab}"] = lab in st.session_state.selected_labels

    with st.sidebar.expander("Country", expanded=True):
        colA, colB = st.columns(2)
        colA.button("Select all", use_container_width=True, on_click=set_all_labels, args=(True,))
        colB.button("Select none", use_container_width=True, on_click=set_all_labels, args=(False,))
        st.divider()

        for lab in labels:
            checked = st.checkbox(
                lab,
                value=lab in st.session_state.selected_labels,
                key=f"chk_{lab}",
            )
            if checked:
                st.session_state.selected_labels.add(lab)
            else:
                st.session_state.selected_labels.discard(lab)

    date_min = df["calendarDate"].min().date()
    date_max = df["calendarDate"].max().date()
    with st.sidebar.expander("Date range", expanded=True):
        st.date_input(
            "Select start and end",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
            key="date_range",
        )
        if st.button("Apply date filter", use_container_width=True):
            st.session_state.applied_date_range = st.session_state.date_range

    selected = sorted(st.session_state.selected_labels)

    if len(selected) == 0:
        df_f = df.iloc[0:0]
    else:
        df_f = df[df["Label"].isin(selected)].copy()

    if "applied_date_range" in st.session_state:
        start_date, end_date = st.session_state.applied_date_range
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df_f = df_f[(df_f["calendarDate"] >= start_dt) & (df_f["calendarDate"] <= end_dt)]

    if len(df_f) == 0:
        st.warning("No country selected!")
        st.stop()

    color_map = build_color_map(df, label_col="Label")

    fig_line = make_line_steps(
        df_f, color_map, line_color="#36073B", line_width=2, point_size=6
    )
    st.plotly_chart(fig_line, use_container_width=True)

    base_sleep = safe_mean(df["total_sleep"])
    base_active = safe_mean(df["activeKilocalories"]) if "activeKilocalories" in df.columns else None
    base_hr = safe_mean(df["maxHeartRate"]) if "maxHeartRate" in df.columns else None
    base_stress = safe_mean(df["avgStress_TOTAL"])

    cur_sleep = safe_mean(df_f["total_sleep"])
    cur_active = safe_mean(df_f["activeKilocalories"]) if "activeKilocalories" in df_f.columns else None
    cur_hr = safe_mean(df_f["maxHeartRate"]) if "maxHeartRate" in df_f.columns else None
    cur_stress = safe_mean(df_f["avgStress_TOTAL"])

    k1, k2, k3, k4 = st.columns([1.5, 1.5, 1.5, 1.5], gap="large")

    d_sleep = delta_pct(cur_sleep, base_sleep)
    with k1:
        kpi_card(
            "Avg Total Sleep Hours",
            "Filter Selection vs Total Avg",
            f"{cur_sleep:.2f}" if cur_sleep is not None else "—",
            d_sleep,
            is_good_change("sleep", d_sleep),
        )

    d_active = delta_pct(cur_active, base_active)
    with k2:
        kpi_card(
            "Avg Active Kilocalories",
            "Filter Selection vs Total Avg",
            f"{cur_active:.0f}" if cur_active is not None else "—",
            d_active,
            is_good_change("active", d_active),
        )

    d_hr = delta_pct(cur_hr, base_hr)
    with k3:
        kpi_card(
            "Avg Max Heart Rate",
            "Filter Selection vs Total Avg",
            f"{cur_hr:.0f}" if cur_hr is not None else "—",
            d_hr,
            is_good_change("hr", d_hr),
        )

    d_stress = delta_pct(cur_stress, base_stress)
    with k4:
        kpi_card(
            "Avg Stress",
            "Filter Selection vs Total Avg",
            f"{cur_stress:.1f}" if cur_stress is not None else "—",
            d_stress,
            is_good_change("stress", d_stress),
        )

    c1, c2 = st.columns([2.2, 1.0], gap="large")
    with c1:
        fig_bar = make_avg_weekly_sleep_stacked_bar(df_f, color_map)
        st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        fig_donut = make_donut_stress(df_f)
        st.plotly_chart(fig_donut, use_container_width=True)

if __name__ == "__main__":
    main()
