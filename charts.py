from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def build_color_map(df: pd.DataFrame, label_col: str = "Label", color_col: str = "color") -> dict:
    labels = sorted(df[label_col].dropna().unique().tolist())

    color_map = {}
    if color_col in df.columns:
        tmp = (
            df[[label_col, color_col]]
            .dropna()
            .drop_duplicates(subset=[label_col])
            .set_index(label_col)[color_col]
            .to_dict()
        )
        color_map.update(tmp)
    fallback = px.colors.qualitative.Plotly
    for i, lab in enumerate(labels):
        if lab not in color_map or not str(color_map[lab]).strip():
            color_map[lab] = fallback[i % len(fallback)]

    return color_map


def make_line_steps(df, color_map):
    fig = px.line(
        df.sort_values("calendarDate"),
        x="calendarDate",
        y="totalSteps",
        color="Label",
        color_discrete_map=color_map,
        markers=False,
        title="Total Steps",
    )

    fig.update_traces(connectgaps=True)

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="Contry",
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Steps")

    return fig



def _add_week_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week"] = out["calendarDate"].dt.to_period("W-MON").apply(lambda p: p.start_time)
    return out


def make_weekly_sleep_stacked_bar(df: pd.DataFrame, color_map: dict) -> "plotly.graph_objects.Figure":
    d = _add_week_col(df)

    g = (
        d.groupby(["week", "Label"], as_index=False)["total_sleep"]
        .sum()
        .rename(columns={"total_sleep": "weekly_sleep_sum"})
    )

    weekly_total = g.groupby("week", as_index=False)["weekly_sleep_sum"].sum()
    baseline = weekly_total["weekly_sleep_sum"].mean() if len(weekly_total) else 0.0

    fig = px.bar(
        g,
        x="week",
        y="weekly_sleep_sum",
        color="Label",
        barmode="stack",
        color_discrete_map=color_map,
        title="Total sleep "
    )

    fig.add_hline(
        y=baseline,
        line_dash="dash",
        annotation_text="Avg",
        annotation_position="top left"
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="Contry",
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Hours")
    return fig


def make_donut_stress(df: pd.DataFrame):
    if "label_stress" not in df.columns:
        agg = pd.DataFrame({"label_stress": ["(missing label_stress)"], "value": [1]})
    else:
        agg = (
            df.groupby("label_stress", as_index=False)
              .size()
              .rename(columns={"size": "value"})
        )

        if len(agg) == 0:
            agg = pd.DataFrame({"label_stress": ["(no data)"], "value": [1]})

    fig = px.pie(
        agg,
        names="label_stress",
        values="value",
        hole=0.62,
        title="Stress Level Frequency"
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="Stress Level",
    )
    return fig
