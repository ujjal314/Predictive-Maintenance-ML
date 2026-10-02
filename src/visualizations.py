"""
Interactive Plotly visualizations for Streamlit Dashboard.
"""

import plotly.express as px
import pandas as pd


def shap_bar_chart(explanation_df):
    """
    Plot SHAP contributions using Plotly.
    """

    plot_df = explanation_df.copy()

    plot_df["Impact"] = plot_df["SHAP Value"].apply(
        lambda x: "Increase Failure" if x > 0 else "Decrease Failure"
    )

    fig = px.bar(
        plot_df.head(10),
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="Impact",
        color_discrete_map={
            "Increase Failure": "#DC2626",
            "Decrease Failure": "#16A34A",
        },
        title="Top Feature Contributions",
    )

    fig.update_layout(
        height=450,
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="SHAP Contribution",
        legend_title=""
    )

    return fig