"""
Maintenance recommendation engine.
"""

def generate_recommendations(explanation_df, probability):

    recommendations = []

    top_features = explanation_df.head(3)["Feature"].tolist()

    if "Tool wear [min]" in top_features:
        recommendations.append({
            "Priority":"Very High",
            "Action":"Inspect cutting-tool wear; elevated tool wear contributed to the predicted failure risk."
        })

    if "Torque [Nm]" in top_features:
        recommendations.append({
            "Priority":"High",
            "Action":"Inspect spindle load and bearings; torque contributed to the predicted failure risk."
        })

    if "Temperature Difference [K]" in top_features:
        recommendations.append({
            "Priority":"High",
            "Action":"Inspect cooling and lubrication; temperature difference contributed to the predicted failure risk."
        })

    if "Rotational speed [rpm]" in top_features:
        recommendations.append({
            "Priority":"Medium",
            "Action":"Inspect motor operating conditions; rotational speed contributed to the predicted failure risk."
        })

    if probability >= 0.70:
        recommendations.append({
            "Priority":"Critical",
            "Action":"Consider a preventive-maintenance review based on the predicted failure risk."
        })

    elif probability >= 0.30:
        recommendations.append({
            "Priority":"Medium",
            "Action":"Monitor during next maintenance cycle."
        })

    else:
        recommendations.append({
            "Priority":"Low",
            "Action":"Machine operating normally."
        })

    return recommendations