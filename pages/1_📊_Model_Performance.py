import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "images"
REPORT_DIR = PROJECT_ROOT / "reports"

st.title("📊 Model Performance Dashboard")
st.write("Performance evaluation of the production XGBoost model.")

# ============================================================
# Metrics
# ============================================================

st.header("🏆 Final Model Metrics")
st.caption("Metrics evaluated on the held-out test set using the default classification threshold of 0.50.")
metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        0.9890,
        0.9107,
        0.7500,
        0.8226,
        0.9777
    ]
})

st.dataframe(
    metrics,
    width="stretch",
    hide_index=True
)

st.divider()

# ============================================================
# Performance Images
# ============================================================

images = [
    ("ROC Curve", "ROC_Curve_XGBoost.png"),
    ("Precision–Recall Curve", "Precision_Recall_Curve_XGBoost.png"),
    ("Confusion Matrix", "XGBoost_Confusion_Matrix.png"),
    ("Threshold Optimization", "Threshold_Optimization.png"),
]

for title, filename in images:

    path = IMAGE_DIR / filename

    st.subheader(title)

    if path.exists():
        st.image(str(path), width="stretch")
    else:
        st.warning(f"{filename} not found in images folder.")

st.divider()

st.subheader("📈 Model Comparison")

comparison = IMAGE_DIR / "Model_Performance_Comparison.png"

if comparison.exists():
    st.image(str(comparison), width="stretch")