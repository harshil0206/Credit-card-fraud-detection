import sys
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10000)

# load the dataset from the csv file using pandas
data = pd.read_csv("C:\\Users\\hshah\\Desktop\\"
                   "building machine learning project with python\\"
                   "Credit card fraud detection\\creditcard.csv")

# print(data.columns)
#
# print(data.shape)

# print(data.describe())

# data = data.sample(frac=0.1, random_state=1)

# print(data.shape)

# plot the histogram of each parameter

# data.hist(figsize=(20,20))
# plt.show()

fraud = data[data["Class"] == 1]
valid = data[data["Class"] == 0]

outlier_fraction = len(fraud) / float(len(valid))
print(outlier_fraction)

print("Fraud cases: {}".format(len(fraud)))
print("valid cases: {}".format(len(valid)))

# correlation matrix
# corrmat = data.corr()

# fig = plt.figure(figsize=(12,9))
# sns.heatmap(corrmat, vmax=1,square=True)
# plt.show()

# get all the columns from the dataframe
columns = data.columns.tolist()

# filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["Class"]]

# store the variable we will be predicting on
target = "Class"

X = data[columns]
y = data[target]

# print the shape of X and y
print(X.shape)
print(y.shape)

# define a random state
state = 1

# define the outlier detection methods

classifiers = {"Isolation Forest": IsolationForest(max_samples=len(X),
                                                   contamination= outlier_fraction,
                                                   random_state=state),
               "Local Outlier Factor":
                   LocalOutlierFactor(n_neighbors=20,contamination=outlier_fraction)}

# fit the model
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):

    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

    # Reshape the prediction values to a 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred!= y).sum()

    # run the classification matrix
    print("{}: {}".format(clf_name, n_errors))
    print(accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))