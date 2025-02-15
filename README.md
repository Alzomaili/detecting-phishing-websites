#### Ayman Alzomaili's Edits (see folder name ayman_updated_code) ####
1- Created updated versions of the three algorithms: Logistic Regression (file name: LogisticRegression_modified.py), Random Forest Classification (file name: RandomForest_modified.py), and Support Vector Machine (file name: SupportVectorMachine_modified.py).
2- Added two algorithms: K-nearest Neighbors (file name: K-nearestNeighbors_new.py), and Multilayer Perceptron (file name: MultilayerPerceptron_new.py)
3- Graph of preformance metrics of all algorithms can be found in performance_chart.png
4- The best performing algorithm of the origonal three is random forest. However, Support Vector Machine is very comparable, and possibly better with more fine tuning.
 
NOTE:
There was an issue with the original SupportVectorMachine and RandomForest code, where the results claimed by the auther could not be replecated exactly since there was one random seed that was missing in both algorithms. This resulted in varying results when running the code multiple times. I modified the algorithms by adding the missing random seed in the files named: SupportVectorMachine_modified and RandomForest_modified. The old code that was only updated can be found in the files named: RandomForest_old_from_html and SupportVectorMachine_old_from_html.
######################################################################



#### The original files were pulled from https://github.com/abhishekdid/detecting-phishing-websites ####
#### Below is the content of the original README.md ####

# PHISHCOOP phishing website detection
Detection of phishing websites is a really important safety measure for most of the online platforms. So, as to save a platform with malicious requests from such websites, it is important to have a robust phishing detection system in place.

## DATA SELECTION

The dataset is downloaded from UCI machine learning repository. The dataset contains 31 columns, with 30 features and 1 target. The dataset has 2456 observations.

## MODELS

To fit the models over the dataset the dataset is split into training and testing sets. The split ratio is 75-25.  Where in 75% accounts to training set. 

Now the training set is used to train the classifier. The classifiers chosen are:  
#### * Logistic Regression
#### * Random Forest Classification
#### * Support Vector Machine

We will see which one fits best in our dataset.

### 1.Logistic Regression

Fitting logistic regression and creating confusion matrix of predicted values and real values I was able to get 92.3 accuracy. Which was good for a logistic regression model.

### 2.Support Vector Machine

Support vector machine with a rbf kernel and using gridsearchcv to predict best parameters for svm was a really good choice, and fitting the model with predicted best parameters I was able to get 96.47 accuracy which is pretty good.

### 3.Random Forest Classification

Next model I wanted to try was random forest and I will also get features importances using it, again using gridsearchcv to get best parameters and fitting best parameters to it I got very good accuracy 97.26.

Random forest was giving very good accuracy. We can also try artificial neural network to get a improved accuracy.

## FEATURE IMPORTANCES

![FEATURE IMPORTANCE](https://raw.githubusercontent.com/abhishekdid/PHISHCOOP-phishing-website-detection/master/variable_Importances.png)

