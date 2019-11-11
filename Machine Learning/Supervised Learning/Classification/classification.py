# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:16:15 2019

@author: jlang
"""

#Unsupervised Learning - Uncovering hidden patters from unlabeled data
#Group customers into distinct catergories - clustering
#Reinforcement learning - software agents interact with environment - learn how to optimize behavior, given sstem of rewards and punishments, inspiration from psychology
#AlphaGo
#Supervised Learning - Predictor variables/features and a toarget with labeled data

#feature = predictor = independent vraiable
#target variable = dependent variable = response variable
from sklearn import datasets
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data', names = ['party', 'infants', 'water', 'budget', 'physician', 'salvador', 'religious',
                                                                                                                         'satellite', 'aid', 'missile', 'immigration', 'synfuels', 'education', 'superfund',
                                                                                                                         'crime', 'duty_free_exports', 'eaa_rsa'])

df.head()
#################
#convert to int 64 besides party
#################
        
# Exploratory Data Analysis
plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

plt.figure()
sns.countplot(df.satellite, hue = 'party', data = df, palette = 'RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

plt.figure()
sns.countplot(x = 'missile', hue = 'party', data = df, palette = 'RdBu')
plt.xticks([0,1],['No', 'Yes'])
plt.show()

# Setting Classifier
# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X: y_pred
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction)) 

# Measuring model performance, split into training and test - train classifier make prediction on test sets - then compare

