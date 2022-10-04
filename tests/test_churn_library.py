import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from churn_library import import_data
from churn_library import logger
from churn_library import parameter
from churn_library import perform_eda


class TestImportData:
    @patch.dict(os.environ, {"PATH_TO_DATA_FOLDER": "tests/data"})
    def test_fake_data(self):
        csv_name = "fake.csv"

        df = import_data(csv_name)

        assert df.shape[0] > 0
        assert df.shape[1] > 0

    @patch.dict(os.environ, {"PATH_TO_DATA_FOLDER": "data"})
    def test_real_data(self):
        csv_name = "BankChurners.csv"

        df = import_data(csv_name)

        assert df.shape == (10127, 23)


class TestPerformEDA:
    @pytest.fixture(scope="class")
    def df(self):
        csv_name = "BankChurners.csv"
        df = import_data(csv_name)
        yield df
        del df

    @pytest.fixture(scope="class")
    def image_folder(self):
        os.environ["PATH_TO_IMAGE_FOLDER"] = "tests/image"
        image_folder = parameter.get_env("PATH_TO_IMAGE_FOLDER")
        image_path = Path(image_folder)

        shutil.rmtree(image_path, ignore_errors=True)
        assert not image_path.exists()

        yield image_folder

    @patch.dict(os.environ, {"PATH_TO_IMAGE_FOLDER": "tests/image"})
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
        perform_eda(df)

        assert Path(image_folder).exists()

        plot_churn_histogram_mock.assert_called_once_with(df, image_folder)
        plot_histogram_mock.assert_called_once_with(df, "Customer_Age", image_folder)
        plot_marital_status_histogram_mock.assert_called_once_with(df, image_folder)
        plot_total_trans_ct_mock.assert_called_once_with(df, image_folder)
        plot_correlation_mock.assert_called_once_with(df, image_folder)


class TestEncoderHelper:
    def test_encoder_helper(self):
        pass


class TestPerformFeatureEngineering:
    def test_perform_feature_engineering(self):
        pass


class TestTrainModels:
    def test_train_models(self):
        pass
