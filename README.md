AI-Based Predictive Maintenance System
An end-to-end machine-learning project that predicts industrial machine failure from operating conditions and provides SHAP-based explanations and a Streamlit dashboard for single-machine and batch predictions.
Project Overview
The system is built around the AI4I 2020 Predictive Maintenance Dataset and follows a complete ML workflow:
Data → Preprocessing → Feature Engineering → Model Benchmarking → XGBoost Optimization → Evaluation → SHAP Explainability → Prediction Dashboard
The project is designed as a portfolio/deployment demonstration of an explainable predictive-maintenance workflow rather than as a production safety system.
Key Features

- Leakage-aware preprocessing
- Engineered Temperature Difference [K] feature
- Benchmarking of baseline, Logistic Regression, Decision Tree, Random Forest and XGBoost models
- XGBoost hyperparameter optimization using 5-fold stratified cross-validation
- F1-oriented model selection for an imbalanced failure-detection problem
- Held-out test-set evaluation
- Classification-threshold analysis
- SHAP local and global explainability
- Rule-based maintenance guidance informed by model explanations
- Interactive Streamlit dashboard
- Single-machine prediction
- Batch CSV prediction
- Downloadable SHAP contributions and batch prediction results
  Dataset
  AI4I 2020 Predictive Maintenance Dataset
- 10,000 machine records
- Target: Machine failure
- Failure rate: approximately 3.4%
- Main model inputs:
  - Type
  - Air temperature [K]
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]
  - Temperature Difference [K]
    The raw failure-subtype columns TWF, HDF, PWF, OSF, and RNF are excluded from modeling because they directly describe failure subtypes and would leak target information into the model. Identifier columns UDI and Product ID are also excluded.
    Feature Engineering
    The project adds:
    Temperature Difference [K]
    = Process temperature [K] - Air temperature [K]
    This feature represents the temperature gap between the process and surrounding air and is used as an additional model input.
    Model Development
    Several classifiers were evaluated on the held-out test set. The final candidate was an optimized XGBoost classifier inside a scikit-learn preprocessing pipeline.
    XGBoost hyperparameter search
    The search varied:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight
- gamma
  The search objective was F1 score, with 5-fold stratified cross-validation.
  Selected XGBoost configuration
  The recorded best configuration was:
  n_estimators = 200
  max_depth = 5
  learning_rate = 0.05
  subsample = 0.8
  colsample_bytree = 1.0
  min_child_weight = 3
  gamma = 0.3
  The best cross-validation F1 score recorded during the search was approximately 0.7917.
  Final Test-Set Performance
  The final tuned XGBoost pipeline achieved the following on the 2,000-row held-out test set, using the default classification threshold of 0.50:
  Metric Score
  Accuracy 98.90%
  Precision 91.07%
  Recall 75.00%
  F1 Score 82.26%
  ROC-AUC 0.9777

For the failure class specifically:

- Precision: 0.9107
- Recall: 0.7500
- F1: 0.8226
- Support: 68 failures in the test set
  Why F1 instead of accuracy?
  The dataset is highly imbalanced: only about 3.4% of records represent machine failure. Accuracy alone can therefore look high even when a model misses failures. F1 was used during model selection to balance precision and recall.
  Classification Threshold Analysis
  The notebook also evaluates different probability thresholds.
  The recorded threshold with the highest test-set F1 was:
  Threshold = 0.43
  Precision = 0.8983
  Recall = 0.7794
  F1 = 0.8346
  The dashboard uses 0.43 as the operational classification threshold.
  Methodological note: the 0.43 threshold was selected using the held-out test set in the current notebook. For a stricter production evaluation, threshold selection should be performed on validation/cross-validation predictions and the final threshold should then be evaluated once on an untouched test set.

Explainable AI
The project uses SHAP (SHapley Additive exPlanations) to explain individual XGBoost predictions.
The Explainability page provides:

- Failure probability
- SHAP feature-contribution bar chart
- SHAP waterfall plot
- Positive and negative contributing features
- Maintenance guidance
- Downloadable feature-contribution CSV
- Global SHAP summary and feature-importance visualizations
  SHAP explanations are interpreted as model contributions, not as proof of physical causality or confirmation that a component has failed.
  Maintenance Guidance
  The dashboard combines:

1. SHAP feature contributions for the individual prediction, and
2. simple rule-based operating-condition checks.
   Examples include inspection guidance for high tool wear, torque, temperature difference, or rotational speed.
   These recommendations are intended as decision-support guidance and should not be treated as an autonomous maintenance command.
   Streamlit Dashboard
   The application contains four main views:
   🏠 Home

- Project overview
- Model information
- Required inputs
- Single-machine prediction
- Failure probability
- Prediction and risk level
- Rule-based maintenance guidance
  📊 Model Performance
- Final test-set metrics
- ROC curve
- Precision–Recall curve
- Confusion matrix
- Threshold analysis
- Model comparison
  🧠 Explainability
- Individual SHAP explanations
- Waterfall plot
- Feature contributions
- Global SHAP insights
- Downloadable explanation data
  📁 Batch Prediction
- Upload multiple machine records as CSV
- Generate failure probabilities
- Apply the operational classification threshold
- Assign risk levels
- Identify highest-risk records
- Download prediction results
  Risk Levels
  The dashboard separates classification threshold from risk-category thresholds.
  Classification
  Failure prediction = probability >= 0.43
  Risk category
  Low < 0.30
  Medium 0.30 to < 0.70
  High >= 0.70
  These thresholds serve different purposes: 0.43 controls the binary prediction, while 0.30 and 0.70 provide a three-level risk display.
  Project Structure
  Predictive-Maintenance-ML/
  │
  ├── README.md
  ├── requirements.txt
  ├── requirements-lock.txt
  ├── LICENSE
  │
  ├── app.py
  │
  ├── data/
  │ ├── raw/
  │ ├── processed/
  │ └── sample*input.csv
  │
  ├── models/
  │ ├── xgb_pipeline.pkl
  │ ├── threshold.json
  │ └── feature_names.pkl
  │
  ├── notebooks/
  │ ├── 01_EDA.ipynb
  │ ├── 02_Preprocessing.ipynb
  │ ├── 03_Model_Training.ipynb
  │ └── 04_Model_Explainability.ipynb
  │
  ├── pages/
  │ ├── 1*📊*Model_Performance.py
  │ ├── 2*🧠*Explainability.py
  │ └── 3*📁_Batch_Prediction.py
  │
  ├── src/
  │ ├── config.py
  │ ├── explainability.py
  │ ├── prediction.py
  │ ├── recommendations.py
  │ ├── utils.py
  │ └── visualizations.py
  │
  ├── images/
  ├── reports/
  └── assets/
  Installation
  Create a virtual environment and install the project dependencies:
  python -m venv .venv
  Windows
  .venv\Scripts\activate
  pip install -r requirements.txt
  Then launch the dashboard:
  streamlit run app.py
  Open the local Streamlit URL shown in the terminal.
  Batch Input Format
  For batch prediction, provide the seven model input columns:
  Type
  Air temperature [K]
  Process temperature [K]
  Rotational speed [rpm]
  Torque [Nm]
  Tool wear [min]
  Temperature Difference [K]
  A sample input file is available at:
  data/sample_input.csv
  The target column Machine failure is not required for prediction.
  Reproducibility
  The training workflow uses:
- stratified train/test splitting
- random_state = 42
- 5-fold stratified cross-validation
- scikit-learn Pipelines for preprocessing and modeling
  The trained production artifact is stored as:
  models/xgb_pipeline.pkl
  The selected classification threshold is stored separately in:
  models/threshold.json
  Important Deployment Note
  The serialized model is environment-sensitive. The deployment environment should use the same library versions used to create the final model artifact.
  Before deployment:

1. Re-run the final model-training notebook through the production-save cell.
2. Confirm that models/xgb_pipeline.pkl is the tuned XGBoost pipeline.
3. Run all Streamlit pages locally.
4. Regenerate requirements-lock.txt from the tested environment.
5. Verify that the SHAP Explainability page works with the exact deployed dependency versions.
   Limitations

- The dataset is a simulated predictive-maintenance dataset rather than live plant data.
- The model predicts statistical failure risk from the available variables; it does not establish physical causation.
- The dataset is highly imbalanced.
- The current threshold-selection notebook chooses 0.43 using the held-out test set; a stricter production workflow should use validation/cross-validation data for threshold selection.
- Maintenance recommendations are rule-based decision support, not automated maintenance instructions.
- Model performance on this dataset should not be interpreted as guaranteed performance on real industrial equipment.
  Future Improvements
- Select the classification threshold using validation or out-of-fold predictions.
- Add automated input-schema validation for uploaded CSV files.
- Centralize all prediction and risk thresholds in configuration.
- Add automated model/data drift monitoring.
- Add calibration monitoring and probability-quality metrics.
- Evaluate on real plant data before operational use.
- Add automated tests for preprocessing, prediction, SHAP explanations, and CSV validation.
- Add CI/CD and containerized deployment.
  Technology Stack
  Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Plotly · Matplotlib · Seaborn · Streamlit · Joblib
  License
  This project is licensed under the MIT License.
  Author
  Ujjal Barman
  Chemical Engineering, IIT Kharagpur
