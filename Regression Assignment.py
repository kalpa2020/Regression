"""
    Compares linear and polynomial regression to K-NN, decision trees, and
    support vector machines.
    Uses grid search to help find the best hyperparameters for each algorithm.
"""

import numpy as np
import csv

from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error

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

## Print info about the dataset

print()
print("Size of the training set: ", len(trainingData))
print("Size of the testing set: ", len(testingData))
print("Number of features: ", data.shape[1])

## Linear regression

from sklearn.linear_model import LinearRegression

linearRegression = LinearRegression();

# Testing set
results_1 = cross_validate(linearRegression, data, labels, scoring="neg_mean_squared_error", cv=8)

print()
print("Linear Regression")
print("=================")
print("Average MSE:",(-1*results_1["test_score"]).mean())
print("Minimum MSE:", (-1*results_1["test_score"]).min())
print("Maximum MSE:", (-1*results_1["test_score"]).max())

## Polynomial regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

polynomialFeatures = PolynomialFeatures();

degree = 2

x_poly = PolynomialFeatures(degree = degree).fit_transform(data)

results_2 = cross_validate(linearRegression, x_poly, labels, scoring="neg_mean_squared_error", cv=8)

print()
print("Polynomial Regression")
print("=====================")
print("Average MSE:",(-1*results_2["test_score"]).mean())
print("Minimum MSE:", (-1*results_2["test_score"]).min())
print("Maximum MSE:", (-1*results_2["test_score"]).max())

## Decision Trees

from sklearn.tree import DecisionTreeRegressor

decisionTreeRegressor = DecisionTreeRegressor(criterion='absolute_error', max_depth=4, min_samples_split=4);

results_3 = cross_validate(decisionTreeRegressor, data, labels, scoring="neg_mean_squared_error", cv=8)

print()
print("Decision Trees Regression")
print("=====================")
print("Average MSE:",(-1*results_3["test_score"]).mean())
print("Minimum MSE:", (-1*results_3["test_score"]).min())
print("Maximum MSE:", (-1*results_3["test_score"]).max())

## Random Forests

from sklearn.ensemble import RandomForestRegressor

randomForestRegressor = RandomForestRegressor()

results_4 = cross_validate(randomForestRegressor, data, labels, scoring="neg_mean_squared_error", cv=8)

print()
print("Random Forests Regression")
print("=====================")
print("Average MSE:",(-1*results_4["test_score"]).mean())
print("Minimum MSE:", (-1*results_4["test_score"]).min())
print("Maximum MSE:", (-1*results_4["test_score"]).max())

## Support Vectors

from sklearn.svm import SVR

svr = SVR();

results_5 = cross_validate(svr, data, labels, scoring="neg_mean_squared_error", cv=8)

print()
print("Support Vectors Regression")
print("=====================")
print("Average MSE:",(-1*results_5["test_score"]).mean())
print("Minimum MSE:", (-1*results_5["test_score"]).min())
print("Maximum MSE:", (-1*results_5["test_score"]).max())