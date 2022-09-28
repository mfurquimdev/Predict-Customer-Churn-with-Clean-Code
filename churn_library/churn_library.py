"""Core of the library to predict clients who are likely to churn."""
import os
from pathlib import Path

import joblib
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from . import parameter
from .classification import plot_classification_report
from .eda import plot_churn_histogram
from .eda import plot_correlation
from .eda import plot_histogram
from .eda import plot_marital_status_histogram
from .eda import plot_total_tras_ct
from .logger import logger
from .roc import plot_lrc_rfc_roc_curve
from .utils import display_info


@display_info
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
    logger.info(f"loaded data of shape {df.shape} from {csv_name}")

    df["Churn"] = df["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)

    return df


@display_info
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

    plot_churn_histogram(df, image_folder)
    plot_histogram(df, "Customer_Age", image_folder)
    plot_marital_status_histogram(df, image_folder)
    plot_total_tras_ct(df, image_folder)
    plot_correlation(df, image_folder)
    logger.info(f"saved images on {image_folder}")


@display_info
def encoder_helper(
    df,
    category_list,
    response=None,
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

    def encode_column(_df, col):
        df = _df.copy(deep=True)

        groups = df.groupby(col).mean()["Churn"]

        lst = []
        for val in df[col]:
            lst.append(groups.loc[val])

        df[f"{col}_Churn"] = lst

        return df

    for col in category_list:
        df = encode_column(df, col)

    logger.info(f"encoded columns {category_list}")

    return df


@display_info
def perform_feature_engineering(
    df,
    response=None,
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
    # fmt: off
    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio', 'Gender_Churn',
        'Education_Level_Churn', 'Marital_Status_Churn', 'Income_Category_Churn', 'Card_Category_Churn'
    ]
    # fmt: on

    X = pd.DataFrame()

    X[keep_cols] = df[keep_cols]
    y = df["Churn"]

    random_state = parameter.get_env("RANDOM_STATE")
    test_size = parameter.get_env("TEST_SIZE")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logger.info(
        f"split data set into train and test with random state = {random_state} and test size = {test_size*100}%"
    )

    return X_train, X_test, y_train, y_test


@display_info
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
    # plot_classification_report(
    #     classification_report(y_test, y_test_preds_rf),
    #     title="Random Forest Classification Report Test",
    # )

    # plot_classification_report(
    #     classification_report(y_train, y_train_preds_rf),
    #     title="Random Forest Classification Report Train",
    # )

    # plot_classification_report(
    #     classification_report(y_test, y_test_preds_lr),
    #     title="Logistic Regression Classification Report Test",
    # )

    # plot_classification_report(
    #     classification_report(y_train, y_train_preds_lr),
    #     title="Logistic Regression Classification Report Train",
    # )


@display_info
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
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)
    pass


@display_info
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
    # grid search
    random_state = parameter.get_env("RANDOM_STATE")

    rfc_path = Path("rfc.pkl")
    lrc_path = Path("lrc.pkl")
    cv_rfc_path = Path("cv_rfc.pkl")

    if rfc_path.is_file() and lrc_path.is_file() and cv_rfc_path.is_file():
        rfc = joblib.load("rfc.pkl")
        lrc = joblib.load("lrc.pkl")
        cv_rfc = joblib.load("cv_rfc.pkl")

    else:
        rfc = RandomForestClassifier(random_state=random_state)
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

        param_grid = {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"],
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)

        lrc.fit(X_train, y_train)

        joblib.dump(rfc, "rfc.pkl")
        joblib.dump(lrc, "lrc.pkl")
        joblib.dump(cv_rfc, "cv_rfc.pkl")

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    image_folder = parameter.get_env("PATH_TO_RESULT_IMAGE_FOLDER")
    Path(image_folder).mkdir(parents=True, exist_ok=True)

    plot_lrc_rfc_roc_curve(image_folder, lrc, cv_rfc.best_estimator_, X_test, y_test)
