# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:09:12 2017

@author: zaher
"""

import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs



# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=5))
# TODO: Total number of records
n_records = len(data)

# TODO: Number of records where individual's income is more than $50,000

n_greater_50k = len(filter(lambda x: x==">50K",data["income"]))

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = len(data)-len(filter(lambda x: x==">50K",data["income"]))

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = (n_greater_50k/n_records*100)

# Print the results
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_raw, transformed = True)
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
display(features_raw.head(n = 10))
# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
df=['workclass','education_level','marital-status','occupation','relationship','race','sex','native-country']
features = pd.get_dummies(features_raw)
# TODO: Encode the 'income_raw' data to numerical values
income = pd.get_dummies(income_raw)
# Print the number of features after one-hot encoding
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Uncomment the following line to see the encoded feature names
#print encoded
# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

# TODO: Calculate accuracy
from sklearn.metrics import accuracy_score 
from sklearn.metrics import fbeta_score
#from sklearn.naive_bayes import GaussianNB

#clf=GaussianNB()
#clf.fit(features,income[">50K"])
#predection=clf.predict(features)


predection=np.ones(len(income[">50K"]))
TP=0
TN=0
FP=0
FN=0
for p,i in zip(predection,income[">50K"]):
    if (p==1 and i==1):
        TP+=1
    elif (p==0 and i==0):
        TN+=1
    elif(p==1 and i==0):
        FP+=1
    else:
        FN+=1
arr=income[">50K"]
accuracy=accuracy_score(arr,predection)
accuracy1=(TP+TN)/float(len(income))
fscore1=(1+(0.5*0.5))*((TP/float(TP+FP))*(TP/float(TP+FN)))/((0.5*0.5*(TP/float(TP+FP)))+(TP/float(TP+FN)))
#TODO: Calculate F-score using the formula above for beta = 0.5
fscore = fbeta_score(income[">50K"].as_matrix(),predection,beta=0.5)

# Print the results 
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy1, fscore1)
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)
#I am not sure about this ansewr because I did not understand what do you mean by this question.