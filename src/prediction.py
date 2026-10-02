"""
Prediction utilities for the Predictive Maintenance Dashboard.
Compatible with models/xgb_pipeline.pkl
"""

from pathlib import Path
import joblib
import pandas as pd

# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_pipeline.pkl"

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

_pipeline = None


def load_model():
    global _pipeline

    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)

    return _pipeline


# -------------------------------------------------------
# Predict Single Machine
# -------------------------------------------------------

def predict_machine(machine_input: dict):

    model = load_model()

    input_df = pd.DataFrame([machine_input])

    probability = model.predict_proba(input_df)[0, 1]
    prediction = int(probability >= 0.43)

    return prediction, probability


# -------------------------------------------------------
# Risk Category
# -------------------------------------------------------

def get_risk_level(probability: float):

    if probability < 0.30:
        return "🟢 Low Risk"

    elif probability < 0.70:
        return "🟡 Medium Risk"

    else:
        return "🔴 High Risk"


# -------------------------------------------------------
# Maintenance Recommendation
# -------------------------------------------------------

def maintenance_recommendation(machine_input, probability):

    recommendations = []

    if machine_input["Tool wear [min]"] > 180:
        recommendations.append("Replace worn cutting tool.")

    if machine_input["Torque [Nm]"] > 50:
        recommendations.append("Inspect spindle load and bearings.")

    if machine_input["Temperature Difference [K]"] > 12:
        recommendations.append("Inspect cooling and lubrication system.")

    if machine_input["Rotational speed [rpm]"] > 1600:
        recommendations.append("Check motor operating conditions.")

    if probability >= 0.70:
        recommendations.append("Schedule preventive maintenance immediately.")

    elif probability >= 0.30:
        recommendations.append("Monitor machine during next maintenance cycle.")

    else:
        recommendations.append("Machine operating under normal conditions.")

    return recommendations