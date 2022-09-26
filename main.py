"""Main script to execute the churn library."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from churn_library import encoder_helper
from churn_library import import_data
from churn_library import parameter
from churn_library import perform_eda
from churn_library import perform_feature_engineering


def main():
    """Main function when the library is issued via command line"""

    sns.set()
    plt.style.use("ggplot")

    np_seed = parameter.get_env("NP_SEED")
    np.random.seed(np_seed)

    csv_name = "BankChurners.csv"
    df = import_data(csv_name)

    perform_eda(df)

    category_list = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    df = encoder_helper(df, category_list)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)


if __name__ == "__main__":
    main()
