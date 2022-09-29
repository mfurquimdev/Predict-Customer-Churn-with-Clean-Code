"""This module defines helper plot functions"""
import os

import matplotlib.pyplot as plt
import numpy as np
import shap


def explainer_plot(
    model,
    X_data,
    output_path,
) -> None:
    """
    Creates and stores the explainer plot in path

    Input:
            model: model object containing best_estimator_
            X_data: pandas dataframe of X values
            output_path: path to store the figure

    Output:
             None
    """
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")

    image_name = "summary_plot"
    path_to_image = os.path.join(output_path, image_name)
    plt.savefig(path_to_image, bbox_inches="tight")


def feature_importance_plot(
    model,
    X_data,
    output_path,
) -> None:
    """
    Creates and stores the feature importances in path

    Input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_path: path to store the figure

    Output:
             None
    """
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order

    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    image_name = "feature_importance"
    path_to_image = os.path.join(output_path, image_name)
    plt.savefig(path_to_image, bbox_inches="tight")
