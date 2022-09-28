"""This module defines helper functions for plotting the ROC curve from trained models"""
import os

import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve


def plot_lrc_rfc_roc_curve(image_folder, lrc, rfc, X_test, y_test):
    fig, ax = plt.subplots(figsize=(15, 8))
    # ax = plt.gca()

    lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)

    image_name = "roc_curve"
    path_to_image = os.path.join(image_folder, image_name)
    plt.savefig(path_to_image, bbox_inches="tight")
