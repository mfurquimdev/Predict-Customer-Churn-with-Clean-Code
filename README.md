# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This project takes a jupyter notebook and splits it into python scripts to facilitate the code's maintenance, and also running and sharing with colleagues.
The notebook loads and pre-process the data set of Bank Churners, runs some Exploratory Data Analysis and generates plots.
It also trains the models and plots the ROC curve to analyze the models' performance.

## Files and data description

You should see a structure similar to the following:

```
Predict-Customer-Churn-with-Clean-Code/
├── data/
│   └── bank_data.csv
├── images/
│   ├── eda/
│   │   ├── churn_distribution.png
│   │   ├── customer_age_distribution.png
│   │   ├── heatmap.png
│   │   ├── marital_status_distribution.png
│   │   └── total_transaction_distribution.png
│   └── results/
│       ├── feature_importance.png
│       ├── logistic_regression_results.png
│       ├── random_forest_results.png
│       └── roc_curve_result.png
├── library/
│   ├── exceptions.py
│   ├── logger.py
│   ├── parameter.py
│   ├── plots.py
│   └── utils.py
├── logs/
│   └── churn_library.log
├── models/
│   ├── cv_rfc.pkl
│   ├── lrc.pkl
│   └── rfc.pkl
├── tests/
│   └── data/
│       └── ...
├── churn_library.py
├── churn_notebook.ipynb
├── env.example
├── Guide.ipynb
├── hotload_unittest.sh
├── Pipfile
├── Pipfile.lock
├── README.md
├── requirements.txt
└── test_churn_library.py

10 directories, 44 files
```

The jupyter notebook was split into a few scripts which resides part on `churn_library.py` and a few auxiliary scripts on `library`.
The `churn_library.py` script has a `main()` function which is responsible for executing each function from loading the data up to training the models.
The data is stored under `data` with the name `bank_data.csv`.
All the images are stores in the `images` directory.
If this script have already been ran, the models as stored under `models`.
The unit tests are inside the ` test_churn_library.py` script.

## Running Files

This project was developed under python 3.8.
After installing this version (you can have multiple versions with [pyenv](https://github.com/pyenv/pyenv)),
install the dependencies with:
```
pipenv install --dev
```

First, see if you can run the unit tests with the following:
```
pipenv run unittests
```

After that, you can run the script `execute` under Pipfile by issuing the following command on the terminal:
```
pipenv run execute
```

This should start the `churn_library.py` script with the `main()` function which will load the csv, generate a few EDA plots, train the model and plot the ROC curve.

Take a look at the log under `logs` and the plots under `images/eda` and `images/results`.
