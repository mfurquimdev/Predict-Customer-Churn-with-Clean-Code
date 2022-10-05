from setuptools import setup

setup(
    name="predict_customer_churn_with_clean_code",
    version="0.0.1",
    description="Udacity MLDevOps Engineer: Predict Customer Churn with Clean Code",
    author="Mateus Furquim",
    author_email="mateus@mfurquim.dev",
    url="https://github.com/mfurquimdev/Predict-Customer-Churn-with-Clean-Code",
    packages=[
        "churn_library",
    ],
    include_package_data=True,
    install_requires=[
        "scikit-learn",
        "shap",
        "joblib",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "loguru",
    ],
    extras_require={
        "dev": [
            "ipykernel",
            "ipdb",
            "pytest",
            "coverage",
            "pytest-cov",
            "black",
            "autopep8",
            "pre-commit",
            "pylint",
        ],
    },
)
