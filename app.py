from __future__ import annotations

import os
import pandas as pd
import streamlit as st

from charts import (
    build_color_map,
    make_line_steps,
    make_weekly_sleep_stacked_bar,
    make_donut_stress,
)

APP_TITLE = "Daily life of BDMA student"


def load_css(path: str) -> None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path,sep=";")
    df["calendarDate"] = pd.to_datetime(df["calendarDate"])
    return df


# def kpi_card(title: str, value: str, subtitle: str) -> None:
#     st.markdown(
#         f"""
#         <div class="kpi-card">
#           <div class="kpi-title">{title}</div>
#           <div class="kpi-value">{value}</div>
#           <div class="kpi-sub">{subtitle}</div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
def kpi_card(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    load_css("styles.css")

    st.title(APP_TITLE)
    # st.caption("Filter by the country")

    csv_path = st.sidebar.text_input("CSV", value="data/Dataset_User1234.csv")
    df = load_data(csv_path)
    required = ["calendarDate", "totalSteps", "total_sleep", "Label", "avgStress_TOTAL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Columns not found in CSV: {missing}")
        st.stop()

    labels = sorted(df["Label"].dropna().unique().tolist())

    st.sidebar.header("Filter")
    # st.sidebar.caption("sidebar disappired")

    if "selected_labels" not in st.session_state:
        st.session_state.selected_labels = set(labels) 

    with st.sidebar.expander("Country", expanded=True):
        c1, c2 = st.columns(2)
        if c1.button("Select all", use_container_width=True):
            st.session_state.selected_labels = set(labels)
        if c2.button("Select none", use_container_width=True):
            st.session_state.selected_labels = set()

        st.divider()

        for lab in labels:
            checked = lab in st.session_state.selected_labels
            new_val = st.checkbox(lab, value=checked, key=f"chk_{lab}")
            if new_val:
                st.session_state.selected_labels.add(lab)
            else:
                st.session_state.selected_labels.discard(lab)

    selected = sorted(st.session_state.selected_labels)

    if len(selected) == 0:
        df_f = df.iloc[0:0]  
    else:
        df_f = df[df["Label"].isin(selected)].copy()

    if len(df_f) == 0:
        st.warning("No country selected!")
        st.stop()

    color_map = build_color_map(df, label_col="Label", color_col="color")


    left, right = st.columns([2.2, 1.0], gap="large")

    with left:
        fig_line = make_line_steps(df_f, color_map)
        st.plotly_chart(fig_line, use_container_width=True)

        fig_bar = make_weekly_sleep_stacked_bar(df_f, color_map)
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        fig_donut = make_donut_stress(df_f)
        st.plotly_chart(fig_donut, use_container_width=True)

        # KPIs
        avg_sleep = df_f["total_sleep"].mean()
        avg_active_kcal = df_f["activeKilocalories"].mean() if "activeKilocalories" in df_f.columns else None
        avg_max_hr = df_f["maxHeartRate"].mean() if "maxHeartRate" in df_f.columns else None

        kpi_card(
            "Avg Total Sleep Hours",
            f"{avg_sleep:.2f}"
        )

        if avg_active_kcal is not None:
            kpi_card(
                "Avg Active Kilocalories",
                f"{avg_active_kcal:.2f}"
            )
        else:
            kpi_card(
                "Avg Active Kilocalories",
                "—",
                "No Active Kilocalories found from the source"
            )

        if avg_max_hr is not None:
            kpi_card(
                "Avg Max Heart Rate",
                f"{avg_max_hr:.2f}"
            )
        else:
            kpi_card(
                "Avg Max Heart Rate",
                "—",
                "No Max Heart Rate found from the source"
            )

    # with st.expander("Preview of the data", expanded=False):
    #     st.dataframe(df_f.head(50))


if __name__ == "__main__":
    main()

