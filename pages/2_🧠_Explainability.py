import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "xgb_pipeline.pkl"

IMAGE_DIR = PROJECT_ROOT / "images"

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.explainability import (
    explain_input_sample,
    create_waterfall_figure
)

from src.visualizations import shap_bar_chart

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🧠",
    layout="wide"
)

st.title("Explainable AI Dashboard")
st.write(
    "Understand **why** the XGBoost model predicts machine failure using SHAP."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()



st.divider()

st.header("Enter Machine Sensor Values")

col1, col2 = st.columns(2)

with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L","M","H"]
    )

    air_temp = st.slider(
        "Air Temperature [K]",
        295.0,305.0,300.0,0.1
    )

    process_temp = st.slider(
        "Process Temperature [K]",
        305.0,315.0,310.0,0.1
    )

    rpm = st.number_input(
        "Rotational Speed [rpm]",
        1100,3000,1500,10
    )

with col2:

    torque = st.slider(
        "Torque [Nm]",
        3.0,80.0,40.0,0.5
    )

    tool_wear = st.slider(
        "Tool Wear [min]",
        0,260,120
    )

temperature_difference = process_temp - air_temp

st.metric(
    "Temperature Difference [K]",
    round(temperature_difference,2)
)



if st.button("Explain Prediction", type="primary", width="stretch"):

    input_df = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Temperature Difference [K]": temperature_difference,
    }])

    (
        probability,
        explanation_df,
        shap_values,
        processed_df,
        explainer,
    ) = explain_input_sample(model, input_df)

    st.divider()

    colA,colB = st.columns([1,1])

    colA.metric(
        "Failure Probability",
        f"{probability*100:.2f}%"
    )

    if probability >= 0.70:
        colB.error("High Failure Risk")

    elif probability >= 0.30:
        colB.warning("Medium Failure Risk")

    else:
        colB.success("Low Failure Risk")



    # Feature Contribution Chart
    st.subheader("SHAP Feature Contributions")

    fig = shap_bar_chart(explanation_df)

    st.plotly_chart(fig, width="stretch")


    # Waterfall Plot
    st.subheader("SHAP Waterfall Plot")

    waterfall_fig = create_waterfall_figure(
        explainer,
        shap_values,
        processed_df
    )

    st.pyplot(waterfall_fig)


    # Top Drivers Table
    st.subheader("Top Positive Drivers")

    st.dataframe(
        explanation_df.head(5),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Top Negative Drivers")

    st.dataframe(
        explanation_df.sort_values("SHAP Value").head(3),
        width="stretch",
        hide_index=True,
    )



    # Business Reccomendation Panel
    st.subheader("Maintenance Recommendations")
    from src.recommendations import generate_recommendations

    recommendations = generate_recommendations(
        explanation_df,
        probability
    )

    for rec in recommendations:

        st.success(
            f"**{rec['Priority']} Priority:** {rec['Action']}"
        )

    # Download SHAP Report
    st.divider()

    st.header("Download SHAP Explanation")

    csv = explanation_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇Download Feature Contributions CSV",
        data=csv,
        file_name="shap_feature_contributions.csv",
        mime="text/csv",
        width="stretch"
    )


# Global SHAP Section
st.divider()

st.header("Global SHAP Insights")

st.write(
    "These plots were generated during Stage 3 using the complete test dataset."
)

summary_path = IMAGE_DIR / "SHAP_Summary.png"
bar_path = IMAGE_DIR / "SHAP_Bar.png"

if summary_path.exists():
    st.subheader("SHAP Summary Plot")
    st.image(str(summary_path), width="stretch")

if bar_path.exists():
    st.subheader("Global Feature Importance")
    st.image(str(bar_path), width="stretch")



    