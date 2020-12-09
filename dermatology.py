# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 08:40:45 2020

@author: Kyle Schmidt
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# This line allows pandas to print all rows
pd.set_option('display.max_rows', None)

# Import the data from file as a pandas df
file_path = 'dermatology.data'
df = pd.read_csv(file_path, na_values='?')

# view a sample to ensure proper import
print(df.head())

# From the data description we have 366 rows of data and 34 columns
# (excluding the classification). The 34 columns include 32 ordinal, 1 binary
# and 1 discrete numerical. Lets check on the completeness and quality of the
# data.
df.info()

# All the data looks to be complete and in the correct format except Age. Age
# has 8 missing values. Now the question is whether to impute or not.
# In this case Imputation is not really viable because we don't have a way to
# Accurately predict the age. Most of the variables are ordinal which don't
# generally perform well in regression and it wouldn't make sense to impute
# the mean, median or mode age. In this situation The best way to continue
# will be to delete the rows with missing data.
df.dropna(inplace=True)
df.info()

# Split the data into features and target.
X = df.loc[:, df.columns != 'Classification']
y = df['Classification']

# Split into Train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print('Decision Tree train/test score:')
clf.score(X_test, y_test)

# K-fold cross validation Decsion Tree
cv_scores = cross_val_score(clf, X, y, cv=5)
print('Decision Tree K=5 cross validation score:')
cv_scores.mean()

# Random Forest
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print('Random Forest train/test score:')
clf.score(X_test, y_test)

# K-fold cross validation Random Forest
cv_scores = cross_val_score(clf, X, y, cv=5)
print('Random Forest K=5 cross validation score:')
cv_scores.mean()

# Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)
print('Naive Bayes train/test score:')
clf.score(X_test, y_test)

# K-fold cross validation Naive Bayes
cv_scores = cross_val_score(clf, X, y, cv=5)
print('Naive Bayes K=5 cross validation score:')
cv_scores.mean()

# Logistic Regression
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)
print('Logistic Regression train/test score:')
clf.score(X_test, y_test)

# K-fold cross validation Logistic Regression
cv_scores = cross_val_score(clf, X, y, cv=5)
print('Logistic Regression K=5 cross validation score:')
cv_scores.mean()

# xgboost need to change class code to be 0:5 rather than 1:6
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)
param = {
    'max_depth': 4,
    'eta': 0.3,
    'objective': 'multi:softmax',
    'num_class': 7}
epochs = 10

model = xgb.train(param, train, epochs)

predictions = model.predict(test)

accuracy_score(y_test, predictions)