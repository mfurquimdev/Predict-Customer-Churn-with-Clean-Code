"""This module defines helper functions for plotting the data"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_correlation(df, image_folder):
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
