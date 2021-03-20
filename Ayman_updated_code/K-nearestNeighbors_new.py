#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

#importing the dataset
dataset = pd.read_csv("datasets/phishcoop.csv")
dataset = dataset.drop('id', 1) #removing unwanted column
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1:].values

#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)

#applying grid search to find best performing parameters 
from sklearn.model_selection import GridSearchCV
parameters = [{'n_neighbors' : [3, 5, 10, 20],
               'weights' : ['uniform','distance'],
              'leaf_size' : [5, 10, 20, 30, 50, 100]}]
grid_search = GridSearchCV(KNeighborsClassifier(n_jobs = -1), parameters, cv = 5, n_jobs= -1)
grid_search.fit(x_train, y_train.ravel())
bestN_neighbors = grid_search.best_params_["n_neighbors"]
bestWeights = grid_search.best_params_["weights"]
bestLeaf_size = grid_search.best_params_["leaf_size"]

#printing best parameters 
print("Best parameters = " + str(grid_search.best_params_)) 
print("Best accuracy with above parameters (training data only) = " + str(grid_search.best_score_))


#fitting K-nearest Neighbors
classifier = KNeighborsClassifier(n_neighbors = bestN_neighbors, weights = bestWeights, leaf_size = bestLeaf_size, n_jobs = -1)
classifier.fit(x_train, y_train.ravel())

#predicting the tests set result
y_pred = classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

#performance metrics
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("Accuracy = " + str(ac))

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print("F-measure = " + str(f1))

from sklearn.metrics import precision_score
pr = precision_score(y_test, y_pred)
print("Precision = " + str(pr))

from sklearn.metrics import recall_score
rc = recall_score(y_test, y_pred)
print("Recall = " + str(rc))

from sklearn.metrics import jaccard_score
jc = jaccard_score(y_test, y_pred)
print("Jaccard = " + str(jc))

#pickle file joblib
joblib.dump(classifier, 'K-nearestNeighbors_new.pkl')


# In[ ]:




