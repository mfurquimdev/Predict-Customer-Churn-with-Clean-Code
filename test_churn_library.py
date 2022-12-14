"""Test churn library"""
import io
import os
import shutil
from pathlib import Path
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import patch

import joblib
import pandas as pd
import pytest

from churn_library import classification_report_image
from churn_library import encoder_helper
from churn_library import import_data
from churn_library import load_or_train_model
from churn_library import parameter
from churn_library import perform_eda
from churn_library import perform_feature_engineering
from churn_library import train_and_test_prediction
from churn_library import train_models
from library import logger


class TestImportData:
    """Test import data function."""

    @patch.dict(os.environ, {"PATH_TO_DATA_FOLDER": "tests/data"})
    def test_fake_data(self):
        """Test loading a fake csv."""
        function_name = "import_data"
        logger.info(f"Test {function_name} with fake data")

        csv_name = "fake.csv"

        df = import_data(csv_name)

        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
            logger.info(f"{function_name} successfully imported fake data")

        except AssertionError as exc:
            logger.error(f"{function_name} could not import fake data")
            raise exc

    @patch.dict(os.environ, {"PATH_TO_DATA_FOLDER": "data"})
    def test_real_data(self):
        """Test loading real csv."""
        function_name = "import_data"
        logger.info(f"Test {function_name} with real data")

        csv_name = "bank_data.csv"

        df = import_data(csv_name)

        try:
            assert df.shape == (10127, 23)
            logger.info(f"{function_name} successfully imported real data")

        except AssertionError as exc:
            logger.error(f"{function_name} could not import real data")
            raise exc


class TestPerformEDA:
    """Test perform EDA function."""

    @pytest.fixture(scope="class")
    def df(self):
        """Load real csv into DataFrame."""

        csv_name = "bank_data.csv"
        df = import_data(csv_name)
        yield df
        del df

    @pytest.fixture(scope="class")
    def image_folder(self):
        """Destroy image directory."""

        os.environ["PATH_TO_IMAGE_FOLDER"] = "tests/images"
        image_folder = parameter.get_env("PATH_TO_IMAGE_FOLDER")
        image_path = Path(image_folder)

        shutil.rmtree(image_path, ignore_errors=True)
        assert not image_path.exists()

        yield image_folder

    @patch.dict(os.environ, {"PATH_TO_IMAGE_FOLDER": "tests/images"})
    @patch("churn_library.plot_churn_distribution")
    @patch("churn_library.plot_distribution")
    @patch("churn_library.plot_marital_status_distribution")
    @patch("churn_library.plot_total_trans_ct")
    @patch("churn_library.plot_correlation")
    def test_perform_eda(
        self,
        plot_correlation_mock,
        plot_total_trans_ct_mock,
        plot_marital_status_distribution_mock,
        plot_distribution_mock,
        plot_churn_distribution_mock,
        df,
        image_folder,
    ):
        """Test calling plot helpers and creation of image directory."""
        function_name = "perform_eda"
        logger.info(f"Test {function_name}")

        perform_eda(df)

        try:
            assert Path(image_folder).exists()
            logger.info(f"{function_name} successfully created {image_folder} image folder")

        except AssertionError as exc:
            logger.error(f"{function_name} have not created {image_folder} image folder")
            raise exc

        try:
            plot_churn_distribution_mock.assert_called_once_with(df, image_folder)
            plot_distribution_mock.assert_called_once_with(df, "Customer_Age", image_folder)
            plot_marital_status_distribution_mock.assert_called_once_with(df, image_folder)
            plot_total_trans_ct_mock.assert_called_once_with(df, image_folder)
            plot_correlation_mock.assert_called_once_with(df, image_folder)
            logger.info(f"{function_name} successfully called all its functions")

        except AssertionError as exc:
            logger.error(f"{function_name} have not called all its functions")
            raise exc


class TestEncoderHelper:
    """Test encoder helper function."""

    @pytest.fixture(scope="class")
    def df(self):
        """Load dummy DataFrame."""

        csv = io.StringIO(
            ",Category1,Churn\n\
                0,A,1\n\
                1,B,1\n\
                2,A,0\n\
                3,B,1\n"
        )

        df = pd.read_csv(csv, index_col=0)
        yield df
        del df

    def test_encoder_helper(self, df):
        """Test encoding Category1 dummy column."""
        function_name = "encoder_helper"
        logger.info("Test {function_name}")

        df = encoder_helper(df, ["Category1"])

        expected_csv = ",Category1,Churn,Category1_Churn\n0,A,1,0.5\n1,B,1,1.0\n2,A,0,0.5\n3,B,1,1.0\n"

        try:
            assert df.to_csv() == expected_csv
            logger.info(f"{function_name} successfully encoded all given features")

        except AssertionError as exc:
            logger.error(f"{function_name} did not encoded all given features")
            raise exc


class TestPerformFeatureEngineering:
    """Test Perform Feature Engineering function."""

    @pytest.fixture(scope="class")
    def df(self):
        """Load real csv into DataFrame."""

        encoded_df_path = Path("tests", "data", "encoded_df.pkl")
        df = joblib.load(encoded_df_path)
        yield df
        del df

    @patch.dict(os.environ, {"RANDOM_STATE": "42", "TEST_SIZE": "0.3"})
    def test_perform_feature_engineering(self, df):
        """Test Perform Feature Engineering function with real DataFrame."""
        function_name = "perform_feature_engineering"
        logger.info(f"Test {function_name} with real DataFrame")

        test_data_path = Path("tests", "data")

        X_train, X_test, y_train, y_test = perform_feature_engineering(df)

        expected_X_train = joblib.load(test_data_path.joinpath("X_train.pkl"))
        expected_X_test = joblib.load(test_data_path.joinpath("X_test.pkl"))
        expected_y_train = joblib.load(test_data_path.joinpath("y_train.pkl"))
        expected_y_test = joblib.load(test_data_path.joinpath("y_test.pkl"))

        try:
            assert expected_X_train.equals(X_train)
            assert expected_X_test.equals(X_test)
            assert expected_y_train.equals(y_train)
            assert expected_y_test.equals(y_test)
            logger.info(f"{function_name} successfully split test and train data sets")

        except AssertionError as exc:
            logger.error(f"{function_name} did not split test and train data sets")
            raise exc


class TestClassificationReportImage:
    """Test the production of classification report image for training and test results"""

    @patch("churn_library.plot_report")
    def test_classification_report_image(self, plot_report_mock):
        """Test calling the two plot report functions."""
        function_name = "classification_report_image"
        logger.info(f"Test {function_name}")

        test_data_path = Path("tests", "data")

        y_train_preds_rf = joblib.load(test_data_path.joinpath("y_train_preds_rf.pkl"))
        y_test_preds_rf = joblib.load(test_data_path.joinpath("y_test_preds_rf.pkl"))
        y_train_preds_lr = joblib.load(test_data_path.joinpath("y_train_preds_lr.pkl"))
        y_test_preds_lr = joblib.load(test_data_path.joinpath("y_test_preds_lr.pkl"))
        y_train = joblib.load(test_data_path.joinpath("y_train.pkl"))
        y_test = joblib.load(test_data_path.joinpath("y_test.pkl"))

        image_folder = "fake_folder"

        classification_report_image(
            image_folder,
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf,
        )

        calls = [
            call(
                "fake_folder",
                "Random Forest",
                y_train,
                y_test,
                y_test_preds_rf,
                y_train_preds_rf,
            ),
            call(
                "fake_folder",
                "Logistic Regression",
                y_train,
                y_test,
                y_test_preds_lr,
                y_train_preds_lr,
            ),
        ]

        try:
            plot_report_mock.assert_has_calls(calls)

            logger.info(f"{function_name} successfully called its two plot report functions")

        except AssertionError as exc:
            logger.error(f"{function_name} did not call its two plot report functions properly")
            raise exc


class TestLoadOrTrainModel:
    """Test Load or Train models."""

    @pytest.fixture(scope="class")
    @patch.dict(os.environ, {"PATH_TO_MODELS": "tests/data"})
    def X_train(self):
        """Load real csv into DataFrame."""
        test_data_path = Path(parameter.get_env("PATH_TO_MODELS"))
        yield joblib.load(test_data_path.joinpath("X_train.pkl"))

    @pytest.fixture(scope="class")
    @patch.dict(os.environ, {"PATH_TO_MODELS": "tests/data"})
    def y_train(self):
        """Load real csv into DataFrame."""
        test_data_path = Path(parameter.get_env("PATH_TO_MODELS"))
        yield joblib.load(test_data_path.joinpath("y_train.pkl"))

    @patch.dict(os.environ, {"PATH_TO_MODELS": "tests/images"})
    @patch("churn_library.joblib.load")
    def test_loading_models(
        self,
        joblib_load_mock,
        X_train,
        y_train,
    ):
        """Test loading models when pickle files exist."""
        function_name = "load_or_train_models"
        logger.info(f"Test {function_name}: loading models")

        test_data_path = Path(parameter.get_env("PATH_TO_MODELS"))
        test_data_path.mkdir(parents=True, exist_ok=True)

        rfc_path = Path(test_data_path, "rfc.pkl")
        lrc_path = Path(test_data_path, "lrc.pkl")
        cv_rfc_path = Path(test_data_path, "cv_rfc.pkl")

        rfc_path.touch()
        lrc_path.touch()
        cv_rfc_path.touch()

        joblib_load_mock.side_effect = ["rfc", "lrc", "cv_rfc"]
        joblib_load_calls = [
            call(rfc_path),
            call(lrc_path),
            call(cv_rfc_path),
        ]

        rfc_actual, lrc_actual, cv_rfc_actual = load_or_train_model(X_train, y_train)

        rfc_expected = "rfc"
        lrc_expected = "lrc"
        cv_rfc_expected = "cv_rfc"

        try:
            assert rfc_expected == rfc_actual
            assert lrc_expected == lrc_actual
            assert cv_rfc_expected == cv_rfc_actual
            logger.info(f"{function_name} returned the correct models")

        except AssertionError as exc:
            logger.error(f"{function_name} did not return the correct models")
            raise exc

        try:
            joblib_load_mock.assert_has_calls(joblib_load_calls)
            logger.info(f"{function_name} successfully loaded models with joblib")

        except AssertionError as exc:
            logger.error(f"{function_name} did not loaded models with joblib")
            raise exc

        try:
            assert test_data_path.exists()
            shutil.rmtree(test_data_path, ignore_errors=True)
            logger.info(f"{function_name} successfully created {test_data_path} directory")

        except AssertionError as exc:
            logger.error(f"{function_name} did not create {test_data_path} directory")
            raise exc

    @patch.dict(os.environ, {"PATH_TO_MODELS": "tests/images", "RANDOM_STATE": "42"})
    @patch("churn_library.RandomForestClassifier")
    @patch("churn_library.LogisticRegression")
    @patch("churn_library.GridSearchCV")
    @patch("churn_library.joblib.dump")
    def test_training_models(
        self,
        joblib_dump_mock,
        grid_search_cv_mock,
        logistic_regression_mock,
        random_forest_classifier_mock,
        X_train,
        y_train,
    ):
        """Test training models when pickle files does not exist."""
        function_name = "load_or_train_models"
        logger.info(f"Test {function_name}: training models")

        random_state = parameter.get_env("RANDOM_STATE")
        test_data_path = Path(parameter.get_env("PATH_TO_MODELS"))
        test_data_path.mkdir(parents=True, exist_ok=True)

        rfc_path = Path(test_data_path, "rfc.pkl")
        lrc_path = Path(test_data_path, "lrc.pkl")
        cv_rfc_path = Path(test_data_path, "cv_rfc.pkl")

        rfc_path.unlink(missing_ok=True)
        lrc_path.unlink(missing_ok=True)
        cv_rfc_path.unlink(missing_ok=True)

        rfc_mock = MagicMock()
        lrc_mock = MagicMock()
        cv_rfc_mock = MagicMock()

        random_forest_classifier_mock.return_value = rfc_mock
        random_forest_classifier_call = {"random_state": random_state}

        logistic_regression_mock.return_value = lrc_mock
        logistic_regression_call = {"solver": "lbfgs", "max_iter": 3000}

        param_grid = {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"],
        }
        grid_search_cv_mock.return_value = cv_rfc_mock
        grid_search_cv_call = {"estimator": rfc_mock, "param_grid": param_grid, "cv": 5}

        joblib_dump_calls = [
            call(rfc_mock, rfc_path),
            call(lrc_mock, lrc_path),
            call(cv_rfc_mock, cv_rfc_path),
        ]

        rfc_actual, lrc_actual, cv_rfc_actual = load_or_train_model(X_train, y_train)

        rfc_expected = rfc_mock
        lrc_expected = lrc_mock
        cv_rfc_expected = cv_rfc_mock

        try:
            assert rfc_expected == rfc_actual
            assert lrc_expected == lrc_actual
            assert cv_rfc_expected == cv_rfc_actual
            logger.info(f"{function_name} successfully returned the trained models")

        except AssertionError as exc:
            logger.error(f"{function_name} did not return the correct trained models")
            raise exc

        try:
            random_forest_classifier_mock.assert_called_once_with(**random_forest_classifier_call)
            logistic_regression_mock.assert_called_once_with(**logistic_regression_call)
            grid_search_cv_mock.assert_called_once_with(**grid_search_cv_call)
            logger.info(f"{function_name} successfully created models")

        except AssertionError as exc:
            logger.error(f"{function_name} did not create models properly")
            raise exc

        try:
            cv_rfc_mock.fit.assert_called_once_with(X_train, y_train)
            lrc_mock.fit.assert_called_once_with(X_train, y_train)
            logger.info(f"{function_name} successfully called models training")

        except AssertionError as exc:
            logger.error(f"{function_name} did not call models training properly")
            raise exc

        joblib_dump_mock.assert_has_calls(joblib_dump_calls)


class TestTrainAndTestPrediction:
    """Test Train and Test prediction function"""

    def test_train_and_test_prediction_with_fake_data(self):
        """Test model's train and test prediction function"""
        function_name = "train_and_test_prediction"
        logger.info(f"Test {function_name} with fake data")

        model_mock = MagicMock()
        model_mock.predict.side_effect = ["y_train_preds", "y_test_preds"]

        model_calls = [
            call.predict("X_train"),
            call.predict("X_test"),
        ]

        y_train_preds_actual, y_test_preds_actual = train_and_test_prediction(model_mock, "X_train", "X_test")

        try:
            model_mock.assert_has_calls(model_calls)
            logger.info(f"{function_name} successfully called prediction on train and test data sets")

        except AssertionError as exc:
            logger.error(f"{function_name} did not properly call prediction on train and test data sets")
            raise exc

        y_train_preds_expected = "y_train_preds"
        y_test_preds_expected = "y_test_preds"

        try:
            assert y_train_preds_actual == y_train_preds_expected
            assert y_test_preds_actual == y_test_preds_expected
            logger.info(f"{function_name} successfully returned train and test prediction")

        except AssertionError as exc:
            logger.error(f"{function_name} did not return the correct train and test prediction")
            raise exc


class TestTrainModels:
    """Test Train models."""

    @patch.dict(os.environ, {"PATH_TO_RESULT_IMAGE_FOLDER": "fake_image_folder"})
    @patch("churn_library.load_or_train_model")
    @patch("churn_library.train_and_test_prediction")
    @patch("churn_library.classification_report_image")
    @patch("churn_library.plot_lrc_rfc_roc_curve")
    @patch("churn_library.plot_feature_importance")
    def test_train_models_with_fake_data(
        self,
        plot_feature_importance_mock,
        plot_lrc_rfc_roc_curve_mock,
        classification_report_image_mock,
        train_and_test_prediction_mock,
        load_or_train_model_mock,
    ):
        """Test train models with fake data."""
        function_name = "train_models"
        logger.info(f"Test {function_name} with fake data")

        image_folder = "fake_image_folder"

        X_train = "X_train"
        X_test = "X_test"
        y_train = "y_train"
        y_test = "y_test"

        rfc = "rfc"
        lrc = "lrc"
        cv_rfc = MagicMock()

        y_train_preds_rf = "y_train_preds_rf"
        y_test_preds_rf = "y_test_preds_rf"
        y_train_preds_lr = "y_train_preds_lr"
        y_test_preds_lr = "y_test_preds_lr"

        load_or_train_model_mock.return_value = (rfc, lrc, cv_rfc)
        load_or_train_model_call = (X_train, y_train)

        train_and_test_prediction_mock.side_effect = [
            (y_train_preds_rf, y_test_preds_rf),
            (y_train_preds_lr, y_test_preds_lr),
        ]

        train_and_test_prediction_calls = [
            call(cv_rfc.best_estimator_, X_train, X_test),
            call(lrc, X_train, X_test),
        ]

        classification_report_image_call = (
            image_folder,
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf,
        )

        plot_lrc_rfc_roc_curve_call = (
            image_folder,
            lrc,
            cv_rfc.best_estimator_,
            X_test,
            y_test,
        )

        plot_feature_importance_call = (
            cv_rfc,
            X_test,
            image_folder,
        )

        train_models(X_train, X_test, y_train, y_test)

        try:
            load_or_train_model_mock.assert_called_once_with(*load_or_train_model_call)
            train_and_test_prediction_mock.assert_has_calls(train_and_test_prediction_calls)
            classification_report_image_mock.assert_called_once_with(*classification_report_image_call)
            plot_lrc_rfc_roc_curve_mock.assert_called_once_with(*plot_lrc_rfc_roc_curve_call)
            plot_feature_importance_mock.assert_called_once_with(*plot_feature_importance_call)
            logger.info(f"{function_name} successfully called training and plotting functions")

        except AssertionError as exc:
            logger.error(f"{function_name} did not properly call training and plotting functions")
            raise exc

        image_path = Path(image_folder)
        try:
            assert image_path.exists()
            shutil.rmtree(image_path, ignore_errors=True)
            logger.info(f"{function_name} successfully created {image_path} directory")

        except AssertionError as exc:
            logger.error(f"{function_name} did not create {image_path} directory")
            raise exc

    @patch.dict(os.environ, {"PATH_TO_RESULT_IMAGE_FOLDER": "fake_image_folder"})
    @patch("churn_library.load_or_train_model")
    @patch("churn_library.train_and_test_prediction")
    @patch("churn_library.classification_report_image")
    @patch("churn_library.plot_lrc_rfc_roc_curve")
    @patch("churn_library.plot_feature_importance")
    def test_train_models(
        self,
        plot_feature_importance_mock,
        plot_lrc_rfc_roc_curve_mock,
        classification_report_image_mock,
        train_and_test_prediction_mock,
        load_or_train_model_mock,
    ):
        """Test train models with real data."""
        function_name = "train_models"
        logger.info(f"Test {function_name} with fake data")

        test_data_path = Path("tests", "data")
        image_folder = "fake_image_folder"

        X_train = joblib.load(test_data_path.joinpath("X_train.pkl"))
        X_test = joblib.load(test_data_path.joinpath("X_test.pkl"))
        y_train = joblib.load(test_data_path.joinpath("y_train.pkl"))
        y_test = joblib.load(test_data_path.joinpath("y_test.pkl"))

        rfc = joblib.load(test_data_path.joinpath("rfc.pkl"))
        lrc = joblib.load(test_data_path.joinpath("lrc.pkl"))
        cv_rfc = joblib.load(test_data_path.joinpath("cv_rfc.pkl"))

        y_train_preds_rf = joblib.load(test_data_path.joinpath("y_train_preds_rf.pkl"))
        y_test_preds_rf = joblib.load(test_data_path.joinpath("y_test_preds_rf.pkl"))
        y_train_preds_lr = joblib.load(test_data_path.joinpath("y_train_preds_lr.pkl"))
        y_test_preds_lr = joblib.load(test_data_path.joinpath("y_test_preds_lr.pkl"))

        load_or_train_model_mock.return_value = (rfc, lrc, cv_rfc)
        load_or_train_model_call = (X_train, y_train)

        train_and_test_prediction_mock.side_effect = [
            (y_train_preds_rf, y_test_preds_rf),
            (y_train_preds_lr, y_test_preds_lr),
        ]

        train_and_test_prediction_calls = [
            call(cv_rfc.best_estimator_, X_train, X_test),
            call(lrc, X_train, X_test),
        ]

        classification_report_image_call = (
            image_folder,
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf,
        )

        plot_lrc_rfc_roc_curve_call = (
            image_folder,
            lrc,
            cv_rfc.best_estimator_,
            X_test,
            y_test,
        )

        plot_feature_importance_call = (
            cv_rfc,
            X_test,
            image_folder,
        )

        train_models(X_train, X_test, y_train, y_test)

        try:
            load_or_train_model_mock.assert_called_once_with(*load_or_train_model_call)
            train_and_test_prediction_mock.assert_has_calls(train_and_test_prediction_calls)
            classification_report_image_mock.assert_called_once_with(*classification_report_image_call)
            plot_lrc_rfc_roc_curve_mock.assert_called_once_with(*plot_lrc_rfc_roc_curve_call)
            plot_feature_importance_mock.assert_called_once_with(*plot_feature_importance_call)
            logger.info(f"{function_name} successfully called training and plotting functions")

        except AssertionError as exc:
            logger.error(f"{function_name} did not properly call training and plotting functions")
            raise exc

        image_path = Path(image_folder)
        try:
            assert image_path.exists()
            shutil.rmtree(image_path, ignore_errors=True)
            logger.info(f"{function_name} successfully created {image_path} directory")

        except AssertionError as exc:
            logger.error(f"{function_name} did not create {image_path} directory")
            raise exc
