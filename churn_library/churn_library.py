"""Core of the library to predict clients who are likely to churn."""
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from . import parameter
from .logger import logger
from .plots import plot_churn_histogram
from .plots import plot_correlation
from .plots import plot_feature_importance
from .plots import plot_histogram
from .plots import plot_lrc_rfc_roc_curve
from .plots import plot_marital_status_histogram
from .plots import plot_report
from .plots import plot_total_trans_ct
from .utils import display_info

# from .plots import explainer_plot


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
    plot_total_trans_ct(df, image_folder)
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
    logger.info(f"encoding columns {category_list}")

    def encode_column(_df, col):
        logger.debug(f"encoding column {col}")
        df = _df.copy(deep=True)

        groups = df.groupby(col).mean()["Churn"]

        lst = []
        for val in df[col]:
            lst.append(groups.loc[val])

        logger.debug(f"encoding column {col}: {lst}")
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
    logger.info("perform feature engineering")

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
        f"Data set split into {(1-test_size)*100}% train ({len(y_train)}) "
        f"and {test_size*100}% test ({len(y_test)}) with random state = {random_state}"
    )

    return X_train, X_test, y_train, y_test


@display_info
def classification_report_image(
    image_folder,
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

            image_folder: directory to save plot
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    Output:
             None
    """
    logger.info("classification report image")

    name = "Random Forest"
    plot_report(image_folder, name, y_train, y_test, y_test_preds_rf, y_train_preds_rf)

    name = "Logistic Regression"
    plot_report(image_folder, name, y_train, y_test, y_test_preds_lr, y_train_preds_lr)


def load_or_train_model(X_train, y_train):
    """Load Random Forest, Logistic Regression and GridSearch models or train them"""
    path_to_models = Path(parameter.get_env("PATH_TO_MODELS"))
    path_to_models.mkdir(parents=True, exist_ok=True)

    rfc_file = "rfc.pkl"
    lrc_file = "lrc.pkl"
    cv_rfc_file = "cv_rfc.pkl"

    rfc_path = path_to_models.joinpath(rfc_file)
    lrc_path = path_to_models.joinpath(lrc_file)
    cv_rfc_path = path_to_models.joinpath(cv_rfc_file)

    if rfc_path.is_file() and lrc_path.is_file() and cv_rfc_path.is_file():
        logger.info(f"Models found under {path_to_models}")

        rfc = joblib.load(rfc_path)
        lrc = joblib.load(lrc_path)
        cv_rfc = joblib.load(cv_rfc_path)

    else:
        logger.debug("Creating Random Forest Classifier")
        random_state = parameter.get_env("RANDOM_STATE")
        rfc = RandomForestClassifier(random_state=random_state)

        logger.debug("Creating Logistic Regression")
        lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

        param_grid = {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"],
        }

        logger.debug("Creating Grid Search CV with Random Forest Classifier as the estimator")
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

        logger.debug("Running Grid Search CV")
        cv_rfc.fit(X_train, y_train)

        logger.debug("Running Logistic Regression")
        lrc.fit(X_train, y_train)

        logger.info(f"Saving models under {path_to_models}")
        joblib.dump(rfc, rfc_path)
        joblib.dump(lrc, lrc_path)
        joblib.dump(cv_rfc, cv_rfc_path)

    return rfc, lrc, cv_rfc


def train_and_test_prediction(model, X_train, X_test):

    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)

    return y_train_preds, y_test_preds


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
    logger.info("train models")

    rfc, lrc, cv_rfc = load_or_train_model(X_train, y_train)

    logger.debug("Running Random Forest Classifier prediction on train and test data")
    y_train_preds_rf, y_test_preds_rf = train_and_test_prediction(cv_rfc.best_estimator_, X_train, X_test)

    logger.debug("Running Logistic Regression prediction on train and test data")
    y_train_preds_lr, y_test_preds_lr = train_and_test_prediction(lrc, X_train, X_test)

    image_folder = parameter.get_env("PATH_TO_RESULT_IMAGE_FOLDER")
    Path(image_folder).mkdir(parents=True, exist_ok=True)

    logger.info(f"Using {image_folder} directory to save result images")

    classification_report_image(
        image_folder,
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    plot_lrc_rfc_roc_curve(image_folder, lrc, cv_rfc.best_estimator_, X_test, y_test)
    plot_feature_importance(cv_rfc, X_test, image_folder)
    # explainer_plot(cv_rfc, X_test, image_folder)
