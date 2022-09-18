"""Core of the library to predict clients who are likely to churn."""
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import parameter


def import_data(
    csv_name,
):
    """
    Returns dataframe for the csv found at path

    Input:
            csv_name: the name of the csv file

    Output:
            df: pandas dataframe
    """
    path_to_data_folder = parameter.get_env("PATH_TO_DATA_FOLDER")
    path_to_csv = os.path.join(path_to_data_folder, csv_name)
    df = pd.read_csv(path_to_csv, index_col=0)

    df["Churn"] = df["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)

    return df


def perform_eda(
    df,
):
    """
    Perform eda on df and save figures to images folder

    Input:
            df: pandas dataframe

    Output:
            None
    """
    image_folder = parameter.get_env("PATH_TO_IMAGE_FOLDER")
    Path(image_folder).mkdir(parents=True, exist_ok=True)

    plt.style.use("ggplot")

    plot_churn_histogram(df, image_folder)
    plot_histogram(df, "Customer_Age", image_folder)
    plot_marital_status_histogram(df, image_folder)
    plot_total_tras_ct(df, image_folder)
    plot_correlation(df, image_folder)


def plot_correlation(df, image_folder):
    fig = plt.figure(figsize=(10, 5))

    df = df.rename(
        columns={
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1": "Naive_Bayes_mon_1*",
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2": "Naive_Bayes_mon_2**",
        }
    )

    plt.figure(figsize=(10, 5))
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

    image_name = "correlation"
    path_to_image = os.path.join(image_folder, image_name)

    plt.savefig(path_to_image, bbox_inches="tight")


def plot_total_tras_ct(df, image_folder):
    plt.figure(figsize=(10, 5))

    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)

    image_name = "total_trans_ct_histogram"
    path_to_image = os.path.join(image_folder, image_name)

    plt.savefig(path_to_image)


def plot_marital_status_histogram(df, image_folder):
    plt.figure(figsize=(10, 5))
    marital_status = df.Marital_Status.value_counts("normalize")
    marital_status.sort_values(ascending=False).plot.bar(
        figsize=(10, 5),
        title="Histogram Normalised of Marital Status",
        ylabel="Normalised Frequency",
        rot=0,
    )
    image_name = "marital_status_histogram"
    path_to_image = os.path.join(image_folder, image_name)
    plt.savefig(path_to_image)


def plot_histogram(df, col, image_folder):
    plt.figure(figsize=(10, 5))
    plt.title(f"Histogram of {col}")

    plt.locator_params(axis="y", integer=True)
    plt.ylabel("Frequency", fontweight="bold", labelpad=20)

    labels, counts = np.unique(df[col], return_counts=True)
    plt.bar(labels, counts)

    image_name = f"{col}_histogram"
    path_to_image = os.path.join(image_folder, image_name)
    plt.savefig(path_to_image)


def plot_churn_histogram(df, image_folder):
    plt.figure(figsize=(10, 5))
    plt.title("Histogram of Churn")

    plt.locator_params(axis="y", integer=True)
    plt.ylabel("Number of people", fontweight="bold", labelpad=20)

    _, counts = np.unique(df["Churn"], return_counts=True)
    labels = ["No Churn", "Churn"]
    plt.bar(labels, counts)

    image_name = "churn_histogram"
    path_to_image = os.path.join(image_folder, image_name)
    plt.savefig(path_to_image)


def encoder_helper(
    df,
    category_list,
    response,
):
    """
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    Input:
            df: pandas dataframe
            category_list: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    Output:
            df: pandas dataframe with new columns for
    """
    pass


def perform_feature_engineering(
    df,
    response,
):
    """
    Helper function to perform feature engineer on DataFrame df.

    Input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    Output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    Produces classification report for training and testing results and stores report as image
    in images folder

    Input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    Output:
             None
    """
    pass


def feature_importance_plot(
    model,
    X_data,
    output_path,
):
    """
    Creates and stores the feature importances in path

    Input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_path: path to store the figure

    Output:
             None
    """
    pass


def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train, store model results: images + scores, and store models

    Input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    Output:
              None
    """
    pass


def main():
    """Main function when the library is issued via command line"""

    sns.set()

    np_seed = parameter.get_env("NP_SEED")
    np.random.seed(np_seed)

    csv_name = "BankChurners.csv"
    df = import_data(csv_name)

    perform_eda(df)


if __name__ == "__main__":
    main()
