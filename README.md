# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This project takes a jupyter notebook and splits it into python scripts to facilitate the code's maintenance, and also running and sharing with colleagues.
The notebook loads and pre-process the data set of Bank Churners, runs some Exploratory Data Analysis and generates plots.
It also trains the models and plots the ROC curve to analyze the models' performance.

## Files and data description

You should see a structure similar to the following:

```
├── churn_library/
│   ├── churn_library.py
│   ├── exceptions.py
│   ├── logger.py
│   ├── parameter.py
│   ├── plots.py
│   └── utils.py
├── data/
│   └── bank_data.csv
├── images/
│   ├── eda/
│   │   ├── churn_histogram.png
│   │   ├── correlation.png
│   │   ├── Customer_Age_histogram.png
│   │   ├── marital_status_histogram.png
│   │   └── total_trans_ct_histogram.png
│   └── results/
│       ├── feature_importance.png
│       ├── Logistic Regression_report.png
│       ├── Random Forest_report.png
│       └── roc_curve.png
├── models/
│   ├── cv_rfc.pkl
│   ├── lrc.pkl
│   └── rfc.pkl
├── tests/
└── main.py
```

The jupyter notebook was split into a few scripts which resides on `churn_library`.
The `main.py` script is responsible for importing and executing each function from loading the data set up to training the models.
The data is stored under `data` with the name `bank_data.csv`.
All the images are stores in the `images` directory.
If this script have already been ran, the models as stored under `models`.
The unit tests are under the `tests` directory.

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

This should start the `main.py` script which will load the csv, generate a few EDA plots, train the model and plot the ROC curve.

Take a look at the plots under `images/eda` and `images/results`.
