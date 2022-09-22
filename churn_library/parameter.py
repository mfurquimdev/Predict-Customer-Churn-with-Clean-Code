"""Module to handle environment variables definitions and parse their values"""
import os


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

    def test_size():
        return float(
            os.getenv(
                "TEST_SIZE",
                "0.3",
            )
        )

    options = {
        "PATH_TO_IMAGE_FOLDER": path_to_image_folder,
        "PATH_TO_DATA_FOLDER": path_to_data_folder,
        "NP_SEED": np_seed,
        "RANDOM_STATE": random_state,
        "TEST_SIZE": test_size,
    }

    return options[option]()
