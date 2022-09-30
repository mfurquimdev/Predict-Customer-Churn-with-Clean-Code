import os
from unittest.mock import patch

from churn_library import import_data
from churn_library import logger


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
    @patch.dict(os.environ, {"PATH_TO_IMAGE_FOLDER": "tests/image"})
    def test_perform_eda(self):
        pass
