"""Helper functions for EDA plotting, ROC curve and classification report"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve

from .logger import logger


def plot_correlation(
    df: pd.DataFrame,
    image_folder: str,
):
    """
    Produces correlation between all features

    Input:
            df: the entire data set
            image_folder: directory to save plot

    Output:
             None
    """
    logger.info("Plot correlation with heatmap")

    df = df.rename(
        columns={
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1": "Naive_Bayes_mon_1*",
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2": "Naive_Bayes_mon_2**",
        }
    )

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax = sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    ax.tick_params(top=True, bottom=False)
    ax.tick_params(labeltop=True, labelbottom=False)
    plt.xticks(rotation=90)

    plt.xlabel(
        "*Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1\n"
        "**Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
        fontsize=8,
        labelpad=15,
        loc="right",
    )

    plt.autoscale()

    image_name = "heatmap"
    path_to_image = os.path.join(image_folder, image_name)

    logger.info(f"Saving plot on {path_to_image}")
    plt.savefig(path_to_image, bbox_inches="tight")


def plot_total_trans_ct(
    df: pd.DataFrame,
    image_folder: str,
) -> None:
    """
    Produces total transaction distribution

    Input:
            df: the entire data set
            image_folder: directory to save plot

    Output:
             None
    """
    logger.info("Plot total transaction")

    plt.figure(figsize=(10, 5))

    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)

    image_name = "total_transaction_distribution"
    path_to_image = os.path.join(image_folder, image_name)

    logger.info(f"Saving plot on {path_to_image}")
    plt.savefig(path_to_image)


def plot_marital_status_distribution(
    df: pd.DataFrame,
    image_folder: str,
) -> None:
    """
    Produces marital status distribution

    Input:
            df: the entire data set
            image_folder: directory to save plot

    Output:
             None
    """
    logger.info("Plot marital status distribution")

    plt.figure(figsize=(10, 5))
    marital_status = df.Marital_Status.value_counts("normalize")
    marital_status.sort_values(ascending=False).plot.bar(
        figsize=(10, 5),
        title="distribution Normalised of Marital Status",
        ylabel="Normalised Frequency",
        rot=0,
    )
    image_name = "marital_status_distribution"
    path_to_image = os.path.join(image_folder, image_name)

    logger.info(f"Saving plot on {path_to_image}")
    plt.savefig(path_to_image)


def plot_distribution(
    df: pd.DataFrame,
    feature: str,
    image_folder: str,
) -> None:
    """
    Produces distribution for a given feature

    Input:
            df: the entire data set
            feature: the given feature to generate the distribution
            image_folder: directory to save plot

    Output:
             None
    """
    logger.info(f"Plot distribution for {feature}")

    plt.figure(figsize=(10, 5))
    plt.title(f"distribution of {feature}")

    plt.locator_params(axis="y", integer=True)
    plt.ylabel("Frequency", fontweight="bold", labelpad=20)

    labels, counts = np.unique(df[feature], return_counts=True)
    plt.bar(labels, counts)

    image_name = f"{feature.lower()}_distribution"
    path_to_image = os.path.join(image_folder, image_name)

    logger.info(f"Saving plot on {path_to_image}")
    plt.savefig(path_to_image)


def plot_churn_distribution(
    df: pd.DataFrame,
    image_folder: str,
) -> None:
    """
    Produces distribution for churn feature

    Input:
            df: the entire data set
            image_folder: directory to save plot

    Output:
             None
    """
    logger.info("Plot churn distribution")

    plt.figure(figsize=(10, 5))
    plt.title("distribution of Churn")

    plt.locator_params(axis="y", integer=True)
    plt.ylabel("Number of people", fontweight="bold", labelpad=20)

    _, counts = np.unique(df["Churn"], return_counts=True)
    labels = ["No Churn", "Churn"]
    plt.bar(labels, counts)

    image_name = "churn_distribution"
    path_to_image = os.path.join(image_folder, image_name)

    logger.info(f"Saving plot on {path_to_image}")
    plt.savefig(path_to_image)


def plot_report(
    image_folder,
    name,
    y_train,
    y_test,
    y_test_preds,
    y_train_preds,
):
    """
    Produces classification report image for one type of trained model

    Input:
            image_folder: directory to save plot
            name: the name of the method used to generate predict the results
            y_train: training response values
            y_test: test response values
            y_train_preds_lr: training predictions from the model
            y_test_preds_rf: test predictions from the model

    Output:
             None
    """
    logger.info(f"Plot report for {name} model")

    train_classification_report = classification_report(y_train, y_train_preds)
    test_classification_report = classification_report(y_test, y_test_preds)

    plt.rc("figure", figsize=(5, 5))
    plt.text(0.01, 1.25, str(f"{name} Train"), {"fontsize": 10}, fontproperties="monospace")
    plt.text(0.01, 0.05, str(train_classification_report), {"fontsize": 10}, fontproperties="monospace")
    plt.text(0.01, 0.6, str(f"{name} Test"), {"fontsize": 10}, fontproperties="monospace")
    plt.text(0.01, 0.7, str(test_classification_report), {"fontsize": 10}, fontproperties="monospace")
    plt.axis("off")

    training_name = "_".join(name.lower().split())
    image_name = f"{training_name}_results"
    path_to_image = os.path.join(image_folder, image_name)

    logger.info(f"Saving plot on {path_to_image}")
    plt.savefig(path_to_image, bbox_inches="tight")


def plot_lrc_rfc_roc_curve(
    image_folder,
    lrc,
    rfc,
    X_test,
    y_test,
):
    """
    Produces ROC curve plot for both Random Forest Classifier and Logistic Regression

    Input:
            image_folder: directory to save plot
            lrc: the logistic regression model
            rfc: the random forest classifier model
            X_test: X testing data
            y_test: test response values

    Output:
             None
    """
    logger.info("Plot Logistic Regression and Random Forest Classifier ROC curve")
    _, ax = plt.subplots(figsize=(15, 8))
    # ax = plt.gca()

    _ = plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    _ = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)

    image_name = "roc_curve_result"
    path_to_image = os.path.join(image_folder, image_name)

    logger.info(f"Saving plot on {path_to_image}")
    plt.savefig(path_to_image, bbox_inches="tight")


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


def plot_feature_importance(
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
    logger.info("Plot feature importance")

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

    logger.info(f"Saving plot on {path_to_image}")
    plt.savefig(path_to_image, bbox_inches="tight")
