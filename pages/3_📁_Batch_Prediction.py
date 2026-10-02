import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📁",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_pipeline.pkl"

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.prediction import get_risk_level

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📁 Batch Machine Failure Prediction")

st.write(
    "Upload a CSV containing multiple machine records."
)

sample_path = PROJECT_ROOT / "data" / "sample_input.csv"

with open(sample_path, "rb") as f:

    st.download_button(
        "⬇️ Download Sample CSV",
        data=f,
        file_name="sample_input.csv",
        mime="text/csv",
        width="stretch"
    )

uploaded_file = st.file_uploader(
    "Upload CSV",
    type="csv"
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(df.head(), width="stretch")

    if st.button("⚙️ Predict All Machines", type="primary"):

        probabilities = model.predict_proba(df)[:,1]

        predictions = (probabilities >= 0.43).astype(int)

        results = df.copy()

        results["Failure Probability"] = probabilities.round(4)

        results["Prediction"] = predictions

        results["Risk Level"] = [
            get_risk_level(p)
            for p in probabilities
        ]

        st.success("Predictions generated successfully.")

        st.subheader("Prediction Results")

        st.dataframe(results, width="stretch")

        st.subheader("🚨 Highest Risk Machines")

        st.dataframe(
            results.sort_values(
                "Failure Probability",
                ascending=False
            ).head(10),
            width="stretch"
        )

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction Results",
            csv,
            file_name="batch_predictions.csv",
            mime="text/csv",
            width="stretch"
        )