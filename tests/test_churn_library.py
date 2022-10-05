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

from churn_library import encoder_helper
from churn_library import import_data
from churn_library import parameter
from churn_library import perform_eda
from churn_library import perform_feature_engineering
from churn_library import train_models
from churn_library.churn_library import classification_report_image


class TestImportData:
    """Test import data function."""

    @patch.dict(os.environ, {"PATH_TO_DATA_FOLDER": "tests/data"})
    def test_fake_data(self):
        """Test loading a fake csv."""

        csv_name = "fake.csv"

        df = import_data(csv_name)

        assert df.shape[0] > 0
        assert df.shape[1] > 0

    @patch.dict(os.environ, {"PATH_TO_DATA_FOLDER": "data"})
    def test_real_data(self):
        """Test loading real csv."""

        csv_name = "BankChurners.csv"

        df = import_data(csv_name)

        assert df.shape == (10127, 23)


class TestPerformEDA:
    """Test perform EDA function."""

    @pytest.fixture(scope="class")
    def df(self):
        """Load real csv into DataFrame."""

        csv_name = "BankChurners.csv"
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
    @patch("churn_library.churn_library.plot_churn_histogram")
    @patch("churn_library.churn_library.plot_histogram")
    @patch("churn_library.churn_library.plot_marital_status_histogram")
    @patch("churn_library.churn_library.plot_total_trans_ct")
    @patch("churn_library.churn_library.plot_correlation")
    def test_perform_eda(
        self,
        plot_correlation_mock,
        plot_total_trans_ct_mock,
        plot_marital_status_histogram_mock,
        plot_histogram_mock,
        plot_churn_histogram_mock,
        df,
        image_folder,
    ):
        """Test calling plot helpers and creation of image directory."""

        perform_eda(df)

        assert Path(image_folder).exists()

        plot_churn_histogram_mock.assert_called_once_with(df, image_folder)
        plot_histogram_mock.assert_called_once_with(df, "Customer_Age", image_folder)
        plot_marital_status_histogram_mock.assert_called_once_with(df, image_folder)
        plot_total_trans_ct_mock.assert_called_once_with(df, image_folder)
        plot_correlation_mock.assert_called_once_with(df, image_folder)


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

        df = encoder_helper(df, ["Category1"])

        expected_csv = ",Category1,Churn,Category1_Churn\n0,A,1,0.5\n1,B,1,1.0\n2,A,0,0.5\n3,B,1,1.0\n"

        assert df.to_csv() == expected_csv


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

        test_data_path = Path("tests", "data")

        X_train, X_test, y_train, y_test = perform_feature_engineering(df)

        expected_X_train = joblib.load(test_data_path.joinpath("X_train.pkl"))
        expected_X_test = joblib.load(test_data_path.joinpath("X_test.pkl"))
        expected_y_train = joblib.load(test_data_path.joinpath("y_train.pkl"))
        expected_y_test = joblib.load(test_data_path.joinpath("y_test.pkl"))

        assert expected_X_train.equals(X_train)
        assert expected_X_test.equals(X_test)
        assert expected_y_train.equals(y_train)
        assert expected_y_test.equals(y_test)


class TestClassificationReportImage:
    """Test the production of classification report image for training and test results"""

    @patch("churn_library.churn_library.plot_report")
    def test_classification_report_image(self, plot_report_mock):
        """Test calling the two plot report functions."""
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

        plot_report_mock.assert_has_calls(calls)

        pass


class LoadOrTrainModel:
    """Test Load or Train models."""

    def loading_models(self):
        pass

    def training_models(self):
        pass


class TestTrainModels:
    """Test Train models."""

    @patch.dict(os.environ, {"PATH_TO_RESULT_IMAGE_FOLDER": "fake_image_folder"})
    @patch("churn_library.churn_library.load_or_train_model")
    @patch("churn_library.churn_library.train_and_test_prediction")
    @patch("churn_library.churn_library.classification_report_image")
    @patch("churn_library.churn_library.plot_lrc_rfc_roc_curve")
    @patch("churn_library.churn_library.plot_feature_importance")
    def test_train_models_with_fake_data(
        self,
        plot_feature_importance_mock,
        plot_lrc_rfc_roc_curve_mock,
        classification_report_image_mock,
        train_and_test_prediction_mock,
        load_or_train_model_mock,
    ):
        """Test when there is a pickle file of a trained model."""
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

        load_or_train_model_mock.assert_called_once_with(*load_or_train_model_call)
        train_and_test_prediction_mock.assert_has_calls(train_and_test_prediction_calls)
        classification_report_image_mock.assert_called_once_with(*classification_report_image_call)
        plot_lrc_rfc_roc_curve_mock.assert_called_once_with(*plot_lrc_rfc_roc_curve_call)
        plot_feature_importance_mock.assert_called_once_with(*plot_feature_importance_call)

        image_path = Path(image_folder)
        assert image_path.exists()
        shutil.rmtree(image_path, ignore_errors=True)

    @patch.dict(os.environ, {"PATH_TO_RESULT_IMAGE_FOLDER": "fake_image_folder"})
    @patch("churn_library.churn_library.load_or_train_model")
    @patch("churn_library.churn_library.train_and_test_prediction")
    @patch("churn_library.churn_library.classification_report_image")
    @patch("churn_library.churn_library.plot_lrc_rfc_roc_curve")
    @patch("churn_library.churn_library.plot_feature_importance")
    def test_train_models(
        self,
        plot_feature_importance_mock,
        plot_lrc_rfc_roc_curve_mock,
        classification_report_image_mock,
        train_and_test_prediction_mock,
        load_or_train_model_mock,
    ):
        """Test when there is a pickle file of a trained model."""
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

        load_or_train_model_mock.assert_called_once_with(*load_or_train_model_call)
        train_and_test_prediction_mock.assert_has_calls(train_and_test_prediction_calls)
        classification_report_image_mock.assert_called_once_with(*classification_report_image_call)
        plot_lrc_rfc_roc_curve_mock.assert_called_once_with(*plot_lrc_rfc_roc_curve_call)
        plot_feature_importance_mock.assert_called_once_with(*plot_feature_importance_call)

        image_path = Path(image_folder)
        assert image_path.exists()
        shutil.rmtree(image_path, ignore_errors=True)
