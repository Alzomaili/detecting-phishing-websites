#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
parameters = [{'n_estimators': [100, 700],
               'max_features': ['sqrt', 'log2'],
               'criterion' :['gini', 'entropy']}]
grid_search = GridSearchCV(RandomForestClassifier(random_state = 0), parameters, cv = 5, n_jobs= -1)
grid_search.fit(x_train, y_train.ravel())
bestCriterion = grid_search.best_params_["criterion"]
bestMax_features = grid_search.best_params_["max_features"]
bestN_estimators = grid_search.best_params_["n_estimators"]

#printing best parameters 
print("Best parameters = " + str(grid_search.best_params_)) 
print("Best accuracy with above parameters (training data only) = " + str(grid_search.best_score_))

#fitting RandomForest regression with best params
classifier = RandomForestClassifier(n_estimators = bestN_estimators, criterion = bestCriterion, max_features = bestMax_features,  random_state = 0)
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
joblib.dump(classifier, 'rf_modified.pkl')


#-------------Features Importance random forestL
#names = dataset.iloc[:,:-1].columns
#importances =classifier.feature_importances_
#sorted_importances = sorted(importances, reverse=True)
#indices = np.argsort(-importances)
#var_imp = pd.DataFrame(sorted_importances, names[indices], columns=['importance'])



#-------------plotting variable importance
#plt.title("Variable Importances")
#plt.barh(np.arange(len(names)), sorted_importances, height = 0.7)
#plt.yticks(np.arange(len(names)), names[indices], fontsize=7)
#plt.xlabel('Relative Importance')
#plt.show()


# In[ ]:




