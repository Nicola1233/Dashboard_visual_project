from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


COLOR_MAP = {
    "Barcelona_sem2": "#fc2c44",
    "Bruxelles_sem1": "#ee9b00",
    "france_sem3": "#475de1",
    "Italy_vac": "#72e5ef",
    "Morocco_vac": "#708e30",
}
def _periods_with_gaps(
    x_index: pd.DatetimeIndex,
    labels: pd.Series,
    step: pd.Timedelta,
) -> pd.DataFrame:
    x = pd.DatetimeIndex(pd.to_datetime(pd.Series(x_index))).sort_values()
    lab = labels.reindex(x)

    df = pd.DataFrame({"x": x, "Label": lab.values}).sort_values("x").reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame(columns=["Label", "start", "end"])

    gap = (df["x"] - df["x"].shift()) > (step * 1.5)
    new_seg = (
        df["Label"].isna()
        | df["Label"].ne(df["Label"].shift())
        | df["Label"].shift().isna()
        | gap
    )
    df["seg"] = new_seg.cumsum()

    periods = (
        df.dropna(subset=["Label"])
        .groupby(["seg", "Label"], as_index=False)
        .agg(start=("x", "min"), end=("x", "max"))
    )
    return periods[["Label", "start", "end"]]



def build_color_map(df: pd.DataFrame, label_col: str = "Label") -> dict:
    labels = set(df[label_col].dropna().unique().tolist())
    return {lab: COLOR_MAP.get(lab, "#888888") for lab in labels}


def _mode_or_nan(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    return s.mode().iloc[0]


def add_city_bands_fast(
    fig: go.Figure,
    df: pd.DataFrame,
    color_map: dict,
    x_index: pd.DatetimeIndex | pd.Series | list,
    step: pd.Timedelta,
    opacity: float = 0.12,
    mode: str = "daily",
) -> go.Figure:
    dff = df.copy()
    dff["calendarDate"] = pd.to_datetime(dff["calendarDate"])

    x_index = pd.DatetimeIndex(pd.to_datetime(pd.Series(x_index))).sort_values()
    if len(x_index) == 0:
        return fig

    if mode == "weekly":
        tmp = dff.copy()
        tmp["week"] = tmp["calendarDate"].dt.to_period("W-MON").apply(lambda p: p.start_time)
        label_by_x = tmp.groupby("week")["Label"].agg(_mode_or_nan)
    else:
        label_by_x = dff.groupby("calendarDate")["Label"].agg(_mode_or_nan)

    periods = _periods_with_gaps(x_index=x_index, labels=label_by_x, step=step)

    for _, r in periods.iterrows():
        fig.add_vrect(
            x0=r["start"],
            x1=r["end"] + step,
            fillcolor=color_map.get(r["Label"], "#999"),
            opacity=opacity,
            line_width=0,
            layer="below",
        )

    return fig
def make_line_steps(
    df: pd.DataFrame,
    color_map: dict,
    smooth_window: int = 7,
    spline_smoothing: float = 1.1,
    line_width: int = 2,
    line_color: str = "#2E86FF",
    point_size: int = 20,
) -> go.Figure:
    dff = df.copy()
    dff["calendarDate"] = pd.to_datetime(dff["calendarDate"])

    daily = (
        dff.groupby("calendarDate", as_index=False)
        .agg(
            totalSteps=("totalSteps", "mean"),
            Label=("Label", _mode_or_nan),
        )
        .sort_values("calendarDate")
    )

    fig = go.Figure()
    if len(daily) == 0:
        fig.update_layout(template="plotly_white", title="Total Steps")
        return fig

    full_idx = pd.date_range(daily["calendarDate"].min(), daily["calendarDate"].max(), freq="D")
    s = daily.set_index("calendarDate")["totalSteps"].reindex(full_idx)   # raw
    lab = daily.set_index("calendarDate")["Label"].reindex(full_idx)

    has_data = s.notna()

    if smooth_window and smooth_window > 1:
        y_smooth = s.rolling(window=smooth_window, center=True, min_periods=1).mean()
        y = y_smooth.where(has_data, np.nan)
    else:
        y = s

    fig.add_trace(go.Scatter(
        x=full_idx,
        y=y,
        mode="lines",
        name="Steps",
        line=dict(width=line_width, shape="spline", smoothing=spline_smoothing, color=line_color),
        connectgaps=True,
        hovertemplate="Date=%{x|%b %d, %Y}<br>Steps=%{y:,.0f}<extra></extra>",
        showlegend=False,
    ))

    y_raw = s.to_numpy()
    y_line = y.to_numpy()
    lab_np = lab.to_numpy()
    has_np = has_data.to_numpy()

    labels_present = sorted(pd.Series(lab.dropna().unique()).tolist())
    for city in labels_present:
        m = (lab_np == city) & has_np
        if m.sum() == 0:
            continue

        fig.add_trace(go.Scatter(
            x=full_idx[m],
            y=y_line[m], 
            mode="markers",
            name=city,
            marker=dict(
                size=point_size,
                color=color_map.get(city, "#999"),
                line=dict(width=0),
            ),
            customdata=y_raw[m],
            hovertemplate="Date=%{x|%b %d, %Y}<br>Label=%{text}<br>Steps=%{customdata:,.0f}<extra></extra>",
            text=[city] * int(m.sum()),
            showlegend=True,
        ))

    fig.update_layout(
        template="plotly_white",
        title="Total Steps",
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="City",
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Steps")
    return fig



def _add_week_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["calendarDate"] = pd.to_datetime(out["calendarDate"])
    out["week"] = out["calendarDate"].dt.to_period("W-MON").apply(lambda p: p.start_time)
    return out


def make_avg_weekly_sleep_stacked_bar(
    df: pd.DataFrame,
    color_map: dict,
    band_opacity: float = 0.35,
    add_bands: bool = True,
) -> go.Figure:
    d = _add_week_col(df)

    weekly = (
        d.groupby("week", as_index=False)
        .agg(
            week_total=("total_sleep", "mean"),
            n_obs=("total_sleep", lambda s: s.notna().mean()), 
        )
        .sort_values("week")
    )

    
    weekly = weekly[weekly["n_obs"] > 0].copy()

    baseline = float(weekly["week_total"].mean()) if len(weekly) else 0.0

    weekly["base"] = weekly["week_total"].where(weekly["week_total"] < baseline, baseline)
    weekly["height"] = (weekly["week_total"] - baseline).abs()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=weekly["week"],
        y=weekly["height"],
        base=weekly["base"],
        marker_color=[
            "#FF6B6B" if v < baseline else "#7ED957"
            for v in weekly["week_total"]
        ],
        hovertemplate="Week=%{x|%Y-%m-%d}<br>Total=%{customdata[0]:.1f}h<br>Avg=%{customdata[1]:.1f}h<extra></extra>",
        customdata=list(zip(weekly["week_total"], [baseline] * len(weekly))),
        showlegend=False,
    ))

    if add_bands and len(weekly):
        # BANDE SOLO DOVE ESISTE UNA BARRA (settimana presente nel grafico)
        add_city_bands_fast(
            fig, d, color_map,
            x_index=weekly["week"],
            step=pd.Timedelta(days=7),
            opacity=band_opacity,
            mode="weekly",
        )

    labels_present = sorted(d["Label"].dropna().unique().tolist())
    for city in labels_present:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=city,
            marker=dict(size=10, color=color_map.get(city, "#999")),
            showlegend=True,
            hoverinfo="skip",
        ))
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        annotation_text="Total avg",
        annotation_position="top left"
    )

    fig.update_layout(
        template="plotly_white",
        title="Average Weekly Sleep Hours",
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="City",
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Hours")
    return fig


STRESS_COLOR_MAP = {
    "Rest":   "#56CCF2",
    "Low":    "#F2C94C",
    "Medium":  "#F2994A",
    "High":   "#EB5757",
}


def make_donut_stress(df: pd.DataFrame) -> go.Figure:
    if "label_stress" not in df.columns:
        agg = pd.DataFrame({"label_stress": ["(missing label_stress)"], "value": [1]})
        fig = px.pie(agg, names="label_stress", values="value", hole=0.62, title="Stress Level Frequency")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), legend_title_text="Stress Level")
        return fig

    agg = (
        df.groupby("label_stress", as_index=False)
        .size()
        .rename(columns={"size": "value"})
    )

    if len(agg) == 0:
        agg = pd.DataFrame({"label_stress": ["(no data)"], "value": [1]})
        fig = px.pie(agg, names="label_stress", values="value", hole=0.62, title="Stress Level Frequency")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), legend_title_text="Stress Level")
        return fig

    order = ["Rest", "Low", "Medium", "High"]

    fig = px.pie(
        agg,
        names="label_stress",
        values="value",
        hole=0.62,
        title="Stress Level Frequency",
        color="label_stress",
        color_discrete_map=STRESS_COLOR_MAP,
        category_orders={"label_stress": order},
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="Stress Level",
    )
    return fig
