from setuptools import setup

setup(
    name="predict_customer_churn_with_clean_code",
    version="0.0.1",
    description="Udacity MLDevOps Engineer: Predict Customer Churn with Clean Code",
    author="Mateus Furquim",
    author_email="mateus@mfurquim.dev",
    url="https://github.com/mfurquimdev/Predict-Customer-Churn-with-Clean-Code",
    packages=[
        # "simulation_statistical",
        # "simulation_statistical.reader",
        # "simulation_statistical.writer",
        # "statistical_disag",
        # "statistical_disag.prior_knowledge",
        # "statistical_disag.retro",
        # "utils",
    ],
    include_package_data=True,
    install_requires=[
        "scikit-learn==0.24.1"
        "shap==0.40.0"
        "joblib==1.0.1"
        "pandas==1.2.4"
        "numpy==1.20.1"
        "matplotlib==3.3.4"
        "seaborn==0.11.2"
        # "pylint==2.7.4"
        # "autopep8==1.5.6"
    ],
    extras_require={
        "dev": [
            "ipykernel~=7.12.1",
            "pytest~=7.1.1",
            "coverage~=6.3.2",
            "pytest-cov~=3.0.0",
            "black~=22.3.0",
        ]
    },
)
