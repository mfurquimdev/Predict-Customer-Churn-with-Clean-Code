"""This module defines helper functions for plotting classification report."""
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from ..logger import logger
from ..utils import display_info


def plot_report(image_folder, name, y_train, y_test, y_test_preds, y_train_preds):
    plt.rc("figure", figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str(f"{name} Train"), {"fontsize": 10}, fontproperties="monospace")
    plt.text(
        0.01, 0.05, str(classification_report(y_test, y_test_preds)), {"fontsize": 10}, fontproperties="monospace"
    )  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(f"{name} Test"), {"fontsize": 10}, fontproperties="monospace")
    plt.text(
        0.01, 0.7, str(classification_report(y_train, y_train_preds)), {"fontsize": 10}, fontproperties="monospace"
    )  # approach improved by OP -> monospace!
    plt.axis("off")

    image_name = f"{name}_report"
    path_to_image = os.path.join(image_folder, image_name)
    plt.savefig(path_to_image, bbox_inches="tight")
