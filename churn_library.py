# library doc string
# import libraries
#
import os


def import_data(
    path,
):
    """
    Returns dataframe for the csv found at path

    Input:
            path: a path to the csv
    Output:
            df: pandas dataframe
    """
    pass


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
    pass


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
