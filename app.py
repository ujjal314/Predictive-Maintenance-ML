import streamlit as st
from pathlib import Path
import joblib
import pandas as pd



# Page Configuration

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project Paths

PROJECT_ROOT = Path(__file__).parent

MODEL_DIR = PROJECT_ROOT / "models"
IMAGE_DIR = PROJECT_ROOT / "images"
REPORT_DIR = PROJECT_ROOT / "reports"
ASSET_DIR = PROJECT_ROOT / "assets"

MODEL_PATH = MODEL_DIR / "xgb_pipeline.pkl"

# Import Prediction Module

import sys

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.prediction import (
    predict_machine,
    get_risk_level,
    maintenance_recommendation,
)

# Load Production Model

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Sidebar

with st.sidebar:

    st.title("⚙️ Predictive Maintenance AI")

    st.markdown("Industrial Machine Failure Prediction")

    st.divider()

    st.success("Production Model Loaded")

    st.caption("models/xgb_pipeline.pkl")

    st.markdown("---")

    st.markdown("### 📂 Dashboard Pages")

    st.page_link("app.py", label="🏠 Home")

    st.page_link(
        "pages/1_📊_Model_Performance.py",
        label="📊 Model Performance"
    )

    st.page_link(
        "pages/2_🧠_Explainability.py",
        label="🧠 Explainability"
    )

    st.page_link(
        "pages/3_📁_Batch_Prediction.py",
        label="📁 Batch Prediction"
    )

    st.markdown("---")

    st.markdown("### 🧠 Tech Stack")

    st.markdown(
        """
        - XGBoost
        - SHAP
        - Scikit-learn
        - Streamlit
        - Plotly
        """
    )

    st.markdown("---")

    st.caption("Version 1.0")
# Header

st.title("AI-Based Predictive Maintenance Dashboard")

st.markdown(
    """
    Predict machine failures before they happen using an **XGBoost model**
    trained on industrial sensor data and explained using **SHAP Explainable AI**.
    """
)

st.divider()

# KPI Cards

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model", "XGBoost")
col2.metric("Problem", "Binary Classification")
col3.metric("Model Inputs", "7 Features")
col4.metric("Explainability", "SHAP")

st.divider()



# About the Project

left, right = st.columns([2,1])

with left:

    st.subheader("Project Overview")

    st.write(
        """
        This project predicts whether an industrial machine is likely to fail
        based on sensor readings such as temperature, rotational speed,
        torque and tool wear.

        The dashboard provides:

        - Real-time machine failure prediction.
        - Failure probability score.
        - SHAP-based feature explanations.
        - Maintenance recommendations.
        - Batch prediction from uploaded CSV files.
        """
    )

with right:

    st.info(
        """
        **Production Model**

        - XGBoost Classifier
        - Scikit-learn Pipeline
        - SHAP Explainability
        """
    )

st.divider()

# Model Summary

st.subheader("Production Model Summary")

summary_df = pd.DataFrame(
    {
        "Component": [
            "Algorithm",
            "Pipeline File",
            "Task",
            "Input Features",
            "Output",
        ],
        "Details": [
            "XGBoost Classifier",
            "models/xgb_pipeline.pkl",
            "Predict Machine Failure",
            "7 Machine Sensor Features",
            "Failure Probability (0–1)",
        ],
    }
)

st.dataframe(summary_df, width="stretch", hide_index=True)

st.divider()

# Expected Input Features

st.subheader("Machine Sensor Inputs")

feature_table = pd.DataFrame(
    {
        "Feature": [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Temperature Difference [K]",
        ],
        "Description": [
            "Machine quality category (L / M / H)",
            "Ambient air temperature.",
            "Machine process temperature.",
            "Machine rotational speed.",
            "Applied machine torque.",
            "Accumulated tool wear.",
            "Process Temperature − Air Temperature",
        ],
    }
)

st.dataframe(feature_table, width="stretch", hide_index=True)
st.divider()

# Dataset Summary

st.subheader("Dataset Summary")

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Dataset", "AI4I 2020")
metric2.metric("Samples", "10,000")
metric3.metric("Target", "Machine Failure")

st.divider()

# Real-Time Prediction

st.header("Predict Machine Failure")

st.write(
    "Enter machine sensor readings below and click **Predict Failure**."
)

left, right = st.columns(2)

with left:

    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"],
    )

    air_temp = st.slider(
        "Air Temperature [K]",
        295.0,
        305.0,
        300.0,
        0.1,
    )

    process_temp = st.slider(
        "Process Temperature [K]",
        305.0,
        315.0,
        310.0,
        0.1,
    )

    rpm = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1100,
        max_value=3000,
        value=1500,
        step=10,
    )

with right:

    torque = st.slider(
        "Torque [Nm]",
        3.0,
        80.0,
        40.0,
        0.5,
    )

    tool_wear = st.slider(
        "Tool Wear [min]",
        0,
        260,
        120,
    )

temperature_difference = process_temp - air_temp

st.metric(
    "Temperature Difference [K]",
    round(temperature_difference, 2),
)

predict_button = st.button(
    "Predict Machine Failure",
    type="primary",
    width="stretch",
)

if predict_button:

    machine_input = {
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Temperature Difference [K]": temperature_difference,
    }

    prediction, probability = predict_machine(machine_input)

    risk = get_risk_level(probability)

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Failure Probability",
        f"{probability * 100:.2f}%",
    )

    c2.metric(
        "Prediction",
        "Failure" if prediction else "Healthy",
    )

    c3.metric(
        "Risk Level",
        risk,
    )

    if prediction:
        st.error("🔴 Machine Failure Predicted")
    else:
        st.success("🟢 Machine Healthy")

    st.subheader("Maintenance Recommendation")

    for rec in maintenance_recommendation(machine_input, probability):
        st.write(f"• {rec}")

    st.subheader("Sensor Values Used")

    st.dataframe(
        pd.DataFrame([machine_input]),
        width="stretch",
        hide_index=True,
    )

st.success(
    "Use the sidebar to explore model performance, SHAP explainability, and batch prediction."
)


# Dashboard Footer
st.divider()

st.markdown(
    """
    ### Project Information

    **AI-Based Predictive Maintenance using XGBoost + SHAP**

    Developed as an end-to-end machine learning project including:

    - Exploratory Data Analysis
    - Feature Engineering
    - XGBoost Model Training
    - Explainable AI (SHAP)
    - Interactive Streamlit Dashboard
    - Batch Prediction Pipeline

    **Dataset:** AI4I 2020 Predictive Maintenance Dataset
    """
)

st.caption(
    "Developed by Ujjal Barman | IIT Kharagpur | Chemical Engineering"
)