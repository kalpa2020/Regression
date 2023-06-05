"""
    Uses grid search to help find the best hyperparameters for each algorithm.
"""

import numpy as np
import csv

from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

## Reads a file and process the data

filename = 'OnlineNewsPopularity.csv'

# Reads a file of data and saves the labels (targets) in a list
rawData = np.genfromtxt(filename, delimiter=",", names=True)
labels = rawData.dtype.names

# Reads a file of data
rawData = np.genfromtxt(filename, delimiter=",", skip_header=1)

# Shuffle the data
rawData = shuffle(rawData)

# Saves the data of the classification column in a list
labels = rawData[:,60]

# Saves the data of the file in a list
data = rawData[:,1:60]

## Splits the data into training and testing data sets

from sklearn.model_selection import train_test_split

trainingData, testingData, trainingLabels, testingLabels = train_test_split(data, labels)

## Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

params = {
    "degree": [2, 3, 4],
    "order": ["C", "F"]
}

polynomialFeatures = PolynomialFeatures()
ridge = Ridge()

x_poly = polynomialFeatures.fit_transform(trainingData)

gridSearch = GridSearchCV(estimator=ridge, param_grid=params, scoring="neg_mean_squared_error")

gridSearch = gridSearch.fit(x_poly, trainingLabels)

print()
print("Polynomial Regression")
print("=====================")
print("Best Parameter Set: ", gridSearch.best_params_)

## Decision Trees

from sklearn.tree import DecisionTreeRegressor

params = {
    "criterion": ["squared_error", "absolute_error"],
    "max_depth": [None, 4, 8],
    "min_samples_split": [None, 2, 4]
}

gridSearch = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=params, scoring="neg_mean_squared_error")

gridSearch = gridSearch.fit(trainingData, trainingLabels)

print()
print("Decision Trees")
print("==============")
print("Best Parameter Set: ", gridSearch.best_params_)

## Random Forests

from sklearn.ensemble import RandomForestRegressor

params = {
    "n_estimators": [10, 50, 100, 150],
    "max_features": [5, "auto", "sqrt", "log2"],
    "max_samples": [None, 100, 150, 200]
}

gridSearch = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params, scoring="neg_mean_squared_error")

gridSearch = gridSearch.fit(trainingData, trainingLabels)

print()
print("Random Forests")
print("==============")
print("Best Parameter Set: ", gridSearch.best_params_)

## Support Vectors

from sklearn.svm import SVR

params = {
    "C": [1, 10, 100],
    "gamma": ["scale", 1, 10],
    "kernel": ["linear", "poly", "rbf"],
    "degree": [2, 3, 4],
    "epsilon": [0.1, 10, 100]
}

gridSearch = GridSearchCV(estimator=SVR(), param_grid=params, scoring="neg_mean_squared_error")

gridSearch = gridSearch.fit(trainingData, trainingLabels)

print()
print("Random Forests")
print("==============")
print("Best Parameter Set: ", gridSearch.best_params_)