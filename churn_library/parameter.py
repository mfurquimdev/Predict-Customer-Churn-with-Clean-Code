"""Module to handle environment variables definitions and parse their values"""
import os

from .exceptions import WrongParameter
from .logger import logger


def get_env(option):
    """Get and parse environment variables data."""

    def path_to_data_folder():
        return os.getenv(
            "PATH_TO_DATA_FOLDER",
            "data/",
        )

    def path_to_image_folder():
        return os.getenv(
            "PATH_TO_IMAGE_FOLDER",
            os.path.join("images", "eda"),
        )

    def path_to_result_image_folder():
        return os.getenv(
            "PATH_TO_RESULT_IMAGE_FOLDER",
            os.path.join("images", "results"),
        )

    def np_seed():
        return int(
            os.getenv(
                "NP_SEED",
                "42",
            )
        )

    def random_state():
        return int(
            os.getenv(
                "RANDOM_STATE",
                "42",
            )
        )

    def path_to_models():
        return os.getenv(
            "PATH_TO_MODELS",
            "models",
        )

    def test_size():
        test_size = float(os.getenv("TEST_SIZE", "0.3"))
        logger.debug(f"test_size = {test_size}")

        if test_size >= 1 or test_size <= 0:
            message = f"TEST_SIZE is {test_size} but should be between 0 and 1, non-inclusive"
            logger.error(message)
            raise WrongParameter(message)

        return test_size

    options = {
        "PATH_TO_IMAGE_FOLDER": path_to_image_folder,
        "PATH_TO_DATA_FOLDER": path_to_data_folder,
        "PATH_TO_RESULT_IMAGE_FOLDER": path_to_result_image_folder,
        "PATH_TO_MODELS": path_to_models,
        "NP_SEED": np_seed,
        "RANDOM_STATE": random_state,
        "TEST_SIZE": test_size,
    }

    return options[option]()
