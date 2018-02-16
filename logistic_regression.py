# -*- coding: utf-8 -*-
# Author: Hoan Bui Dang
# Python: 3.6

""" 
A quick test to see how well a logistic regression model works.
Feature reduction and normalization was not done. 
Evaluate the performance using KS chart and confusion matrix.
"""

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

df = pd.read_csv('creditcard.csv')
X = df.iloc[:,1:29]
y = df['Class']

# split data into training and testing sets
spl = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
for idx_train, idx_test in spl.split(X, y):
    X_train = X.iloc[idx_train]
    y_train = y.iloc[idx_train]
    X_test = X.iloc[idx_test]
    y_test = y.iloc[idx_test]

# train the model using train data
classifier = LogisticRegression()    
classifier.fit(X_train, y_train)

# test the model on test data
y_predict = classifier.predict(X_test)
y_score = classifier.predict_proba(X_test)
    
import evaluation as ev
ev.confusion(y_predict,y_test)

xy = ev.KS_chart(y_score[:,0],y_test)
plt.plot(xy[:,0], xy[:,1])
plt.plot(xy[:,0], xy[:,2])
plt.show()