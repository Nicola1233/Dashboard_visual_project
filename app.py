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
def load_logo(path: str, width: int):
    from PIL import Image

    img = Image.open(path)
    w, h = img.size
    if w == width:
        return img
    new_h = int(round(h * (width / w)))
    return img.resize((width, new_h), Image.LANCZOS)

@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    df["calendarDate"] = pd.to_datetime(df["calendarDate"])
    return df


def kpi_card(
    title: str,
    subtitle1: str,
    value_text: str,
    delta_pct_val: float | None,
    is_good: bool | None,
    info_text: str | None = None,
) -> None:
    if delta_pct_val is None or is_good is None:
        delta_html = '<span class="kpi-delta neutral">—</span>'
    else:
        arrow = "▲" if delta_pct_val > 0 else ("▼" if delta_pct_val < 0 else "•")
        cls = "good" if is_good else "bad"
        delta_html = f'<span class="kpi-delta {cls}">{arrow} {abs(delta_pct_val):.1f}%</span>'

    info_html = ""
    if info_text:
        info_html = f'<span class="kpi-info" tabindex="0">i<span class="kpi-tooltip">{info_text}</span></span>'

    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}{info_html}</div>
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
            st.title("Explanation page")
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
        st.markdown('##  Structure')
        st.markdown("""
                    I designed the dashboard as a single-page overview to keep the message clear and concise. 
                    The main page combines four KPI cards with three primary charts to provide an at-a-glance summary of the most relevant indicators. To support deeper analysis without cluttering the layout, the interactive controls are placed in a sidebar: users can filter by context/location labels and select a specific time period to focus on particular states and compare patterns across contexts.
                """)
        st.markdown('## Meta data')
        st.markdown("""
                To reduce misinterpretation, the dashboard provides supporting information such as units (hours, steps, bpm) and clear legends for categorical encodings (location/context labels and stress states). Hover tooltips show the exact date, the corresponding values, and the context label for each data point. Since some metrics, especially stress, are estimated by Garmin using internal models, and location labels were reconstructed from travel history rather than GPS day by day, short notes clarify that these values should be interpreted mainly in a relative sense. They are useful for comparing periods and contexts rather than as absolute measurements.
            """)
        st.markdown('## Visualization Representation')
        st.markdown(""" 
            - KPI Overview (cards): shows the average values of the key metrics (sleep, active kilocalories, max heart rate, stress) and the difference compared to the overall average through a trend arrow (percentage change). This provides a quick check of the effect of the selected filters and highlights whether the chosen context differs from the global baseline.
            - Average Weekly Sleep Hours (bar chart): I use weekly aggregated bars because sleep is easier to interpret as an average trend, and this aggregation reduces daily noise. The overall average line, used as a baseline, makes it immediately clear in which periods sleep was above or below the average.
            - Daily Average Stress (donut chart): for stress, I am not interested in a single number only, but in the percentage of days in which my average stress level falls into Garmin’s categories (rest/low/medium/high). A composition chart makes the relative weight of each category clear and helps identify which stress state was most frequent in the selected period or context.
            - Total Steps (time series): a time-series view is well suited to reveal trends, variability, and routine changes over time. Points are colored by city/context to make transitions between environments visible, and a reference goal line helps interpret whether activity levels are above or below a target.
            
            """)
        st.markdown('## Page layout')
        st.markdown(""" 
            The dashboard follows a clear visual hierarchy. A collapsible filter sidebar on the left separates controls from the visual analysis area, so interaction elements do not compete with the charts. The main canvas starts with a title and subtitle to frame the question, followed by a KPI row for an at-a-glance summary. The largest chart (Total Steps over time) is placed in the center as the primary view, while the two secondary views (weekly sleep and stress distribution) are positioned below to provide complementary perspectives.
            """)
        st.markdown('## Screenspace use')
        st.markdown(""" 
            Screenspace is used to emphasize the most informative views and reduce clutter. When needed, the sidebar can be collapsed to maximize space for the charts, improving readability on smaller screens. High-level indicators (KPIs) occupy a compact top row, while the time-series chart receives the widest area because it carries most of the temporal detail. Secondary charts are smaller but still readable, enabling comparison within a single screen.
            """)
        
        st.markdown('## Interaction')
        st.markdown(""" 
            Interaction is designed to support quick exploration without adding visual clutter. Users can hover over points and chart elements to see the exact values through tooltips, including the corresponding date and contextual label. This makes it easy to inspect specific moments and compare periods directly. In addition, the sidebar filters enable personalized analysis by allowing users to accurately explore how the metrics vary across different real-world contexts and across different time periods.
            """)
        st.markdown('## Color')
        st.markdown(""" 
            
    <div style="line-height:1.5">

    <p>
    Colors are used consistently across the dashboard so the same hue always refers to the same context/state,
    making cross-chart comparison easier.
    </p>

    <h4>Location / context palette</h4>
    <ul>
      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#fc2c44;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>Spain</b> (<code>#fc2c44</code>): red is a prominent national color and visually stands out in the time series.</li>

      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#ee9b00;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>Belgium</b> (<code>#ee9b00</code>): yellow is strongly associated with Belgian national identity and is also widely used in sports jerseys and the flag.</li>

      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#475de1;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>France</b> (<code>#475de1</code>): blue is one of the main national colors and provides strong contrast with Spain’s red.</li>

      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#72e5ef;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>Italy</b> (<code>#72e5ef</code>): light blue references “Azzurro,” a widely recognized Italian color (historically linked to Italy’s sports identity).</li>

      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#8c8e30;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>Morocco</b> (<code>#8c8e30</code>): an earthy green tone reflects Moroccan national symbolism and remains distinct from the other contexts.</li>

      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#ab1eb0;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>Belgium (summer)</b> (<code>#ab1eb0</code>): a separate hue highlights the seasonal sub-period while staying clearly different from the main Belgium color.</li>
    </ul>

    <p><i>
    Design choice:</i> although some contexts could also be associated with red tones, I avoided using multiple similar reds
    to preserve legibility and reduce confusion.
    </p>

    <h4>Stress states palette</h4>
    <p>Stress uses an intuitive “warmer = higher stress” scale, while rest uses a contrasting calm color.</p>
    <ul>
      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#56CCF2;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>Rest</b> (<code>#56CCF2</code>): calm blue to clearly separate “no stress” from the stress scale.</li>

      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#F2C94C;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>Low</b> (<code>#F2C94C</code>)</li>

      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#F2994A;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>Medium</b> (<code>#F2994A</code>)</li>

      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#EB5757;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>High</b> (<code>#EB5757</code>)</li>
    </ul>

      <h4>KPI trend arrows</h4>
    <ul>
      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#2ab53c;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>Green</b> indicates a positive deviation above the overall average.</li>
      <li><span style="display:inline-block;width:0.9em;height:0.9em;background:#EB5757;border:1px solid #999;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>
          <b>Red</b> indicates a negative deviation below the overall average.</li>
    </ul>

    </div>

            """, unsafe_allow_html=True)
        
        st.markdown("## Purpose and context")
        st.markdown(
            """
        The dashboard explores how changes in geographical, social, and seasonal context are associated with variations in physical activity, sleep, and stress levels.

        My master’s program, Erasmus Mundus BDMA, involved intense mobility and gave me the opportunity to live in three very different cities while studying at three European universities:
        - Université Libre de Bruxelles (ULB)
        - Universitat Politècnica de Catalunya (UPC)
        - CentraleSupélec

        In addition to these academic locations, the analysis also includes two further countries with very different “lifestyle contexts”:
        - Italy, where I spent time mainly on holiday and living with my family
        - Morocco, a holiday trip shared with friends

        With these contexts combined, the goal is to reflect on how my lifestyle changes across:
        - different urban environments,
        - different university settings,
        - different types of free time 
        - and different time periods, including seasons.

        Importantly, the dashboard does not aim to establish causal relationships. Instead, it supports reflection through the comparison of contexts—for example, it is plausible that winter conditions may reduce outdoor activity levels compared to warmer seasons.
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


    if st.button("Explanation page"):
        st.session_state.page = "info"
        st.rerun()

    title_left, title_right = st.columns([3.2, 0.8], gap="large")
    with title_left:
        st.title(APP_TITLE)
        st.markdown(
            "*How context and location shape activity, sleep, and stress over time*"
        )
    with title_right:
        pass
       
    

    csv_path = st.sidebar.text_input("CSV", value="data/Dataset_User1234.csv")
    df = load_data(csv_path)

    required = ["calendarDate", "totalSteps", "total_sleep", "Label", "avgStress_TOTAL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Columns not found in CSV: {missing}")
        st.stop()

    labels = sorted(df["Label"].dropna().unique().tolist())

    st.sidebar.header("Filter")

    exam_periods = {
        "UPC": {
            "label": "BRU",
            "start": pd.Timestamp(year=2025, month=5, day=29).date(),
            "end": pd.Timestamp(year=2025, month=6, day=10).date(),
        },
        "ULB": {
            "label": "BCN",
            "start": pd.Timestamp(year=2025, month=1, day=13).date(),
            "end": pd.Timestamp(year=2025, month=1, day=24).date(),
        },
    }

    type_map = {
        "Holiday": ["holiday"],
        "University semester": ["first semester", "second semester", "third semester"],
        "Summer school": ["summer", "summer school"],
    }
    type_categories = list(type_map.keys())

    if "selected_labels" not in st.session_state:
        st.session_state.selected_labels = set(labels)
    if "selected_exam_periods" not in st.session_state:
        st.session_state.selected_exam_periods = set()

    date_min = df["calendarDate"].min().date()
    date_max = df["calendarDate"].max().date()

    if "date_range" not in st.session_state:
        st.session_state.date_range = (date_min, date_max)
    if "applied_date_range" not in st.session_state:
        st.session_state.applied_date_range = (date_min, date_max)
    if "applied_labels" not in st.session_state:
        st.session_state.applied_labels = set(labels)
    if "applied_types" not in st.session_state:
        st.session_state.applied_types = set(type_categories)
    if "applied_exam_periods" not in st.session_state:
        st.session_state.applied_exam_periods = set()

    def reset_filters(apply_now: bool = True):
        for lab in labels:
            st.session_state.pop(f"chk_{lab}", None)
        st.session_state.selected_labels = set(labels)
        for ex in exam_periods:
            st.session_state.pop(f"chk_exam_{ex}", None)
        st.session_state.selected_exam_periods = set()
        if "type" in df.columns:
            for t in type_categories:
                st.session_state.pop(f"chk_type_{t}", None)
            st.session_state.selected_types = set(type_categories)
        st.session_state.date_range = (date_min, date_max)
        if apply_now:
            st.session_state.applied_labels = set(labels)
            st.session_state.applied_types = set(type_categories)
            st.session_state.applied_date_range = (date_min, date_max)
            st.session_state.applied_exam_periods = set()

    col_reset, col_apply = st.sidebar.columns(2)
    if col_reset.button(
        "Clean filters",
        use_container_width=True,
        key="btn_clean_filters",
    ):
        reset_filters(apply_now=True)
        st.rerun()
    if col_apply.button(
        "Apply filters",
        use_container_width=True,
        key="btn_apply_filters",
    ):
        st.session_state.applied_labels = set(st.session_state.selected_labels)
        st.session_state.applied_types = set(st.session_state.selected_types)
        st.session_state.applied_exam_periods = set(st.session_state.selected_exam_periods)
        if st.session_state.applied_exam_periods:
            starts = [exam_periods[ex]["start"] for ex in st.session_state.applied_exam_periods]
            ends = [exam_periods[ex]["end"] for ex in st.session_state.applied_exam_periods]
            zoom_start = min(starts) - pd.Timedelta(days=14)
            zoom_end = max(ends) + pd.Timedelta(days=14)
            if zoom_start < date_min:
                zoom_start = date_min
            if zoom_end > date_max:
                zoom_end = date_max
            st.session_state.applied_date_range = (zoom_start, zoom_end)
        else:
            st.session_state.applied_date_range = st.session_state.date_range
        st.rerun()

    with st.sidebar.expander("Country", expanded=True):
        for lab in labels:
            key = f"chk_{lab}"
            if key not in st.session_state:
                st.session_state[key] = lab in st.session_state.selected_labels
            st.checkbox(lab, key=key)
        st.session_state.selected_labels = {lab for lab in labels if st.session_state[f"chk_{lab}"]}

    if "type" in df.columns:
        if "selected_types" not in st.session_state:
            st.session_state.selected_types = set(type_categories)

        with st.sidebar.expander("Type", expanded=True):
            for t in type_categories:
                key = f"chk_type_{t}"
                if key not in st.session_state:
                    st.session_state[key] = t in st.session_state.selected_types
                st.checkbox(t, key=key)
            st.session_state.selected_types = {t for t in type_categories if st.session_state[f"chk_type_{t}"]}

    with st.sidebar.expander("Exam", expanded=True):
        for ex, cfg in exam_periods.items():
            key = f"chk_exam_{ex}"
            if key not in st.session_state:
                st.session_state[key] = ex in st.session_state.selected_exam_periods
            label = f"{ex} ({cfg['label']})"
            st.checkbox(label, key=key)
        st.session_state.selected_exam_periods = {
            ex for ex in exam_periods if st.session_state[f"chk_exam_{ex}"]
        }
    def set_season_range(season: str) -> None:
        year = date_max.year
        if season == "Winter":
            start = pd.Timestamp(year=year, month=12, day=1).date()
            end = pd.Timestamp(year=year + 1, month=2, day=28).date()
        elif season == "Spring":
            start = pd.Timestamp(year=year, month=3, day=1).date()
            end = pd.Timestamp(year=year, month=5, day=31).date()
        elif season == "Summer":
            start = pd.Timestamp(year=year, month=6, day=1).date()
            end = pd.Timestamp(year=year, month=8, day=31).date()
        else:  # Fall
            start = pd.Timestamp(year=year, month=9, day=1).date()
            end = pd.Timestamp(year=year, month=11, day=30).date()

        # clamp to dataset range
        if start < date_min:
            start = date_min
        if end > date_max:
            end = date_max
        st.session_state.date_range = (start, end)

    with st.sidebar.expander("Date range", expanded=True):
        st.markdown(
            """
            <div class="sidebar-info-row">
              <span class="sidebar-info-label">Disclaimer</span>
              <span class="sidebar-info-icon" tabindex="0">i
                <span class="sidebar-info-tooltip">
                  <div class="sidebar-info-title">Disclaimer</div>
                  The temporal data were reconstructed by analyzing my flights.
                  For this reason, there can be a few days of uncertainty.
                </span>
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Winter", use_container_width=True, key="btn_season_winter"):
            set_season_range("Winter")
        if st.button("Spring", use_container_width=True, key="btn_season_spring"):
            set_season_range("Spring")
        if st.button("Summer", use_container_width=True, key="btn_season_summer"):
            set_season_range("Summer")
        if st.button("Fall", use_container_width=True, key="btn_season_fall"):
            set_season_range("Fall")

        st.date_input(
            "Select start and end",
            min_value=date_min,
            max_value=date_max,
            key="date_range",
        )

    selected = sorted(st.session_state.applied_labels)

    if len(selected) == 0:
        df_f = df.iloc[0:0]
    else:
        df_f = df[df["Label"].isin(selected)].copy()

    if "type" in df.columns:
        selected_types = sorted(st.session_state.applied_types)
        if len(selected_types) == 0:
            df_f = df_f.iloc[0:0]
        else:
            allowed = set()
            for t in selected_types:
                allowed.update(type_map.get(t, []))
            df_f = df_f[df_f["type"].astype(str).str.lower().isin(allowed)]

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
            info_text=(
                "<div class='kpi-tooltip-title'>Sleep (Garmin)</div>"
                "<div class='kpi-tooltip-body'>"
                "<strong>Sleep duration</strong> and <strong>sleep stages</strong> are identified using a "
                "combination of <strong>heart rate</strong> <strong>HR</strong>, <strong>heart rate variability</strong> "
                "<strong>HRV</strong>, and <strong>body movement</strong> data. <strong>Age</strong> information and "
                "personal <strong>physiological reference values</strong> provide context for the analysis and improve "
                "the reliability of sleep detection."
                "</div>"
            ),
        )

    d_active = delta_pct(cur_active, base_active)
    with k2:
        kpi_card(
            "Avg Active Calories (Kcal)",
            "Filter Selection vs Total Avg",
            f"{cur_active:.0f}" if cur_active is not None else "—",
            d_active,
            is_good_change("active", d_active),
            info_text=(
                "<div class='kpi-tooltip-title'>Active Calories (Garmin)</div>"
                "<div class='kpi-tooltip-body'>"
                "<strong>Active calories</strong> are estimated by <strong>Garmin</strong> using "
                "<strong>FirstBeat Analytics</strong>, primarily combining <strong>heart rate</strong> "
                "data with the user profile such as <strong>age</strong> <strong>sex</strong> "
                "<strong>height</strong> and <strong>weight</strong>."
                "</div>"
            ),
        )

    d_hr = delta_pct(cur_hr, base_hr)
    with k3:
        kpi_card(
            "Avg Max Heart Rate(bpm)",
            "Filter Selection vs Total Avg",
            f"{cur_hr:.0f}" if cur_hr is not None else "—",
            d_hr,
            is_good_change("hr", d_hr),
            info_text=(
                "<div class='kpi-tooltip-title'>Heart Rate (Garmin)</div>"
                "<div class='kpi-tooltip-body'>"
                "<strong>Garmin</strong> monitors heart rate mainly through the "
                "<strong>Garmin Elevate</strong> <strong>optical sensor</strong> <strong>PPG</strong> on the back "
                "of the watch. It shines <strong>green light</strong> into the skin and measures the reflected "
                "light from blood in the vessels to detect <strong>pulsatile blood flow</strong> and estimate, "
                "<strong>heart rate</strong>, expressed in <strong>bpm<s/trong> (beats per minute)."
                "</div>"
            ),
        )

    d_stress = delta_pct(cur_stress, base_stress)
    with k4:
        kpi_card(
            "Avg Stress",
            "Filter Selection vs Total Avg",
            f"{cur_stress:.1f}" if cur_stress is not None else "—",
            d_stress,
            is_good_change("stress", d_stress),
            info_text=(
                "<div class='kpi-tooltip-title'>Stress (Garmin)</div>"
                "<div class='kpi-tooltip-body'>"
                "<strong>Garmin</strong> estimates stress on a 0 to 100 scale using "
                "<strong>FirstBeat Analytics</strong> and a combination of <strong>heart rate</strong> "
                "<strong>HR</strong> and <strong>heart rate variability</strong> <strong>HRV</strong> "
                "measured by the <strong>optical sensor</strong> on the wrist. These signals reflect "
                "activity in the <strong>autonomic nervous system</strong>. In a "
                "<strong>sympathetic</strong> state, stress tends to be higher. In a "
                "<strong>parasympathetic</strong> state, stress tends to be lower. During "
                "<strong>sleep</strong> it is usually lower. <strong>Garmin</strong> does not report stress "
                "during <strong>physical activity</strong> because exercise changes are assessed with other metrics."
                "</div>"
            ),
        )

    highlight_ranges = []
    for ex in st.session_state.applied_exam_periods:
        cfg = exam_periods.get(ex)
        if not cfg:
            continue
        highlight_ranges.append(
            {
                "name": f"Exam: {ex}",
                "start": cfg["start"],
                "end": cfg["end"],
            }
        )

    base_span_days = max((date_max - date_min).days, 1)
    if "applied_date_range" in st.session_state:
        s, e = st.session_state.applied_date_range
        if s and e:
            zoom_span_days = max((e - s).days, 1)
        else:
            zoom_span_days = base_span_days
    else:
        zoom_span_days = base_span_days

    scale = base_span_days / zoom_span_days
    scale = max(1.0, min(scale, 3.0))
    point_size = int(round(6 * scale))
    highlight_size = int(round(point_size * 2.6))
    line_width = int(round(2 * scale))
    line_width = max(2, min(line_width, 6))
    highlight_color = "#FF0000"

    try:
        fig_line = make_line_steps(
            df_f,
            color_map,
            line_color="#36073B",
            line_width=line_width,
            point_size=point_size,
            highlight_ranges=highlight_ranges,
            highlight_size=highlight_size,
            highlight_color=highlight_color,
        )
    except TypeError:
        # Fallback for older charts.py signature without highlight args.
        fig_line = make_line_steps(
            df_f,
            color_map,
            line_color="#36073B",
            line_width=line_width,
            point_size=point_size,
        )
    st.plotly_chart(fig_line, use_container_width=True)

    
    c1, c2 = st.columns([2.2, 1.0], gap="large")
    with c1:
        fig_bar = make_avg_weekly_sleep_stacked_bar(
            df_f,
            color_map,
            band_opacity=0.55,
            highlight_ranges=highlight_ranges,
            highlight_color=highlight_color,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        fig_donut = make_donut_stress(df_f)
        st.plotly_chart(fig_donut, use_container_width=True)

if __name__ == "__main__":
    main()
