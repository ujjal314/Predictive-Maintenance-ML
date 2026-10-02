"""
config.py
=========
Central configuration file for the Predictive Maintenance ML project.

WHY THIS FILE EXISTS
---------------------
In the original notebooks, values like `test_size=0.20`, `random_state=42`,
and the dataset path were typed directly into each cell. That works for a
single notebook, but it creates two problems as a project grows:

1. Consistency risk — if the train/test split uses random_state=42 in
   Notebook 02 but a different value in Notebook 03, results silently stop
   matching, and no one notices until metrics look wrong.
2. Duplication — the same file paths, column names, and hyperparameters get
   retyped in every notebook and later in the Streamlit dashboard.

Centralizing these values in one importable module means every notebook,
script, and the dashboard reads from a single source of truth. Change a
value once here, and it is consistent everywhere.

INTERVIEW RELEVANCE
--------------------
Separating configuration from logic is a standard software engineering
practice (the "12-Factor App" principle of config separation). Interviewers
look for this because it signals you can write code meant to be maintained
by a team, not just a one-off analysis script.
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================
# PROJECT_ROOT resolves two levels up from this file (src/config.py -> project root),
# so paths work no matter which machine or working directory the code runs from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_RAW_DIR / "ai4i2020.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "processed_dataset.csv"
BEST_MODEL_PATH = MODELS_DIR / "xgb_pipeline.pkl"

# ==========================================================
# Reproducibility
# ==========================================================
# A single random_state used everywhere (train/test split, model init,
# cross-validation folds) guarantees that re-running the full pipeline
# produces identical results every time — a requirement for any experiment
# you want to defend in an interview or reproduce for a report.
RANDOM_STATE = 42

# ==========================================================
# Train / Test Split
# ==========================================================
TEST_SIZE = 0.20  # 20% held out for testing, matches Notebook 02

# ==========================================================
# Dataset Schema
# ==========================================================
TARGET_COLUMN = "Machine failure"

# Columns removed before modeling.
# UDI / Product ID: pure identifiers, carry no predictive signal.
# TWF, HDF, PWF, OSF, RNF: failure SUBTYPE indicators. These are only known
# AFTER a failure has already occurred, so including them would leak the
# answer into the model (the model would just learn "if any subtype flag
# is 1, then Machine failure is 1" instead of learning from sensor readings).
ID_COLUMNS = ["UDI", "Product ID"]
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature Difference [K]",  # engineered feature, see utils.py
]

CATEGORICAL_FEATURES = ["Type"]

# Ordinal category order for the 'Type' column.
# L (Low) < M (Medium) < H (High) reflects a genuine quality ranking, which
# is why OrdinalEncoder is preferred over OneHotEncoder here (see
# notebooks/02_Preprocessing.ipynb for the full explanation).
TYPE_CATEGORY_ORDER = ["L", "M", "H"]

# ==========================================================
# Model Hyperparameter Search Space (Random Forest)
# ==========================================================
# Matches the GridSearchCV grid used in Notebook 03. Centralizing it here
# means Stage 2's expanded tuning can import and extend this same dict
# instead of redefining it from scratch.
RF_PARAM_GRID = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
}

# ==========================================================
# Evaluation
# ==========================================================
# F1 is used as the GridSearchCV scoring metric (instead of accuracy)
# because the dataset is highly imbalanced (~3.4% failure rate). A model
# that predicts "no failure" for every machine would score ~96.6% accuracy
# while being completely useless. F1 balances Precision and Recall, so it
# rewards a model for actually catching failures without flooding
# maintenance teams with false alarms.
SCORING_METRIC = "f1"
CV_FOLDS = 5

# ==========================================================
# Streamlit Dashboard Risk Thresholds (used from Stage 4 onward)
# ==========================================================
RISK_THRESHOLDS = {
    "low": 0.30,     # probability < 0.30  -> Low risk
    "medium": 0.70,  # 0.30 <= probability < 0.70 -> Medium risk
    # probability >= 0.70 -> High risk
}
