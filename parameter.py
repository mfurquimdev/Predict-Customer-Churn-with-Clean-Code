"""Module to deal with environment variables definitions and parse their values"""
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

    options = {
        "PATH_TO_IMAGE_FOLDER": path_to_image_folder,
        "PATH_TO_DATA_FOLDER": path_to_data_folder,
        "NP_SEED": np_seed,
    }

    return options[option]()
