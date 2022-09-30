import os
from unittest.mock import patch

from churn_library import import_data
from churn_library import logger


@patch.dict(os.environ, {"PATH_TO_DATA_FOLDER": "tests/data"})
def test_import_data():
    csv_name = "fake.csv"

    df = import_data(csv_name)

    assert df.shape[0] > 0
    assert df.shape[1] > 0
