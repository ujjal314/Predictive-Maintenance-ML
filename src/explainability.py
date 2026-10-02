from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# -------------------------------------------------------
# Load Production Pipeline
# -------------------------------------------------------

def load_pipeline(model_path):
    """Load saved production pipeline."""
    return joblib.load(model_path)


# -------------------------------------------------------
# Prepare SHAP Inputs
# -------------------------------------------------------

def prepare_shap_inputs(model, X):
    """
    Transform raw dataframe using fitted preprocessor.

    Returns
    -------
    X_processed_df : DataFrame
    clean_feature_names : list
    xgb_model : trained XGBClassifier
    preprocessor : fitted preprocessor
    """

    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["model"]

    X_processed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in feature_names
    ]

    X_processed_df = pd.DataFrame(
        X_processed,
        columns=clean_feature_names,
        index=X.index,
    )

    return X_processed_df, clean_feature_names, xgb_model, preprocessor


# -------------------------------------------------------
# Create SHAP Explainer
# -------------------------------------------------------

def create_explainer(xgb_model):
    """Create TreeExplainer for XGBoost."""
    return shap.TreeExplainer(xgb_model)


# -------------------------------------------------------
# Compute SHAP Values
# -------------------------------------------------------

def compute_shap_values(explainer, X_processed_df):
    return explainer.shap_values(X_processed_df)


# -------------------------------------------------------
# Global SHAP Importance
# -------------------------------------------------------

def global_feature_importance(shap_values, feature_names):
    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_SHAP_Value": np.abs(shap_values).mean(axis=0)
    })

    return df.sort_values(
        "Mean_SHAP_Value",
        ascending=False
    ).reset_index(drop=True)


# -------------------------------------------------------
# Summary Plot
# -------------------------------------------------------

def save_summary_plot(shap_values, X_processed_df, save_path):
    plt.figure(figsize=(10,6))

    shap.summary_plot(
        shap_values,
        X_processed_df,
        show=False
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------
# Beeswarm Plot
# -------------------------------------------------------

def save_beeswarm_plot(shap_values, X_processed_df, feature_names, save_path):
    plt.figure(figsize=(10,6))

    shap.plots.beeswarm(
        shap.Explanation(
            values=shap_values,
            data=X_processed_df.values,
            feature_names=feature_names,
        ),
        show=False,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------
# Dependence Plot
# -------------------------------------------------------

def save_dependence_plot(feature, shap_values, X_processed_df, save_path):
    plt.figure(figsize=(7,5))

    shap.dependence_plot(
        feature,
        shap_values,
        X_processed_df,
        interaction_index="auto",
        show=False,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------
# Waterfall Plot
# -------------------------------------------------------

def save_waterfall_plot(explainer, shap_values, X_processed_df,
                        feature_names, sample_position, save_path):

    explanation = shap.Explanation(
        values=shap_values[sample_position],
        base_values=explainer.expected_value,
        data=X_processed_df.iloc[sample_position],
        feature_names=feature_names,
    )

    plt.figure(figsize=(9,6))

    shap.plots.waterfall(
        explanation,
        max_display=10,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------
# Force Plot (HTML)
# -------------------------------------------------------

def save_force_plot(explainer, shap_values, X_processed_df,
                    feature_names, sample_position, save_path):

    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[sample_position],
        X_processed_df.iloc[sample_position],
        feature_names=feature_names,
        matplotlib=False,
    )

    shap.save_html(str(save_path), force_plot)


# -------------------------------------------------------
# Explain One Machine
# -------------------------------------------------------

def explain_machine(model, explainer, shap_values,
                    X_processed_df, feature_names,
                    original_X, machine_index):

    # if hasattr(model, "named_steps"):
    #     model = model.named_steps["model"]

    probability = model.predict_proba(
        original_X.loc[[machine_index]]
    )[0,1]

    position = X_processed_df.index.get_loc(machine_index)

    contribution_df = pd.DataFrame({
        "Feature": feature_names,
        "Feature Value": X_processed_df.iloc[position].values,
        "SHAP Value": shap_values[position],
    })

    contribution_df = contribution_df.sort_values(
        "SHAP Value",
        ascending=False
    ).reset_index(drop=True)

    return {
        "machine_index": int(machine_index),
        "failure_probability": float(probability),
        "top_positive": contribution_df.head(3),
        "top_negative": contribution_df.tail(2),
        "all_features": contribution_df,
    }


# -------------------------------------------------------
# Export Top Explanations (JSON)
# -------------------------------------------------------

def export_local_explanations(model, explainer, shap_values,
                              X_processed_df, feature_names,
                              original_X, machine_indices, save_path):

    results = []

    for idx in machine_indices:

        explanation = explain_machine(
            model,
            explainer,
            shap_values,
            X_processed_df,
            feature_names,
            original_X,
            idx,
        )

        results.append({
            "machine_index": explanation["machine_index"],
            "failure_probability": explanation["failure_probability"],
            "top_positive_features":
                explanation["top_positive"]["Feature"].tolist(),
            "top_negative_features":
                explanation["top_negative"]["Feature"].tolist(),
        })

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


def explain_input_sample(model, input_df):
    """
    Generate SHAP explanation for a single machine input.

    Parameters
    ----------
    model : sklearn Pipeline (xgb_pipeline.pkl)
    input_df : DataFrame (1 row)

    Returns
    -------
    probability
    explanation_df
    shap_values
    processed_df
    explainer
    """

    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["model"]

    # Transform features
    processed = preprocessor.transform(input_df)

    feature_names = preprocessor.get_feature_names_out()

    clean_names = [
        f.replace("num__", "").replace("cat__", "")
        for f in feature_names
    ]

    processed_df = pd.DataFrame(
        processed,
        columns=clean_names
    )

    explainer = shap.TreeExplainer(xgb_model)

    processed_array = processed_df.to_numpy()
    explainer.model.feature_names = None
    shap_values = explainer(processed_array).values
    probability = model.predict_proba(input_df)[0,1]

    explanation_df = pd.DataFrame({
        "Feature": clean_names,
        "Feature Value": processed_df.iloc[0].values,
        "SHAP Value": shap_values[0]
    })

    explanation_df = explanation_df.sort_values(
        "SHAP Value",
        ascending=False
    ).reset_index(drop=True)

    return (
        probability,
        explanation_df,
        shap_values,
        processed_df,
        explainer
    )


def create_waterfall_figure(
    explainer,
    shap_values,
    processed_df
):
    """
    Returns a matplotlib waterfall figure.
    """

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=processed_df.iloc[0],
        feature_names=processed_df.columns.tolist()
    )

    fig = plt.figure(figsize=(8,6))

    shap.plots.waterfall(
        explanation,
        max_display=10,
        show=False
    )

    plt.tight_layout()

    return fig