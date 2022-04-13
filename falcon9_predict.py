import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



def plot_confusion_matrix(y, y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv')

#Creating a numpy array from column 'Class'in data, which denotes launch success or failure with 0 or 1
Y = data['Class'].to_numpy()

#Standardize data in X 
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

#Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}

#Create logistic regression object, then a GridSearchCV to find best params
parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()

g_scv = GridSearchCV(lr, parameters, scoring = 'accuracy', cv = 10)
logreg_cv = g_scv.fit(X_train, Y_train)

#Display best parameters
print("Tuned hyperparameters: (best parameters) ",logreg_cv.best_params_)
print("Logistic Regression best accuracy score (Training data): ", logreg_cv.best_score_)

#Calculate accuracy on test data
print("Logistic Regression accuracy (Test data): ", logreg_cv.score(X_test, Y_test))

#Create and plot confusion matrix for logistic regression
yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

#Create support vector machine, then a GridSearchCV to find best params
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()

g_scv = GridSearchCV(svm, parameters, scoring = 'accuracy', cv = 10)
svm_cv = g_scv.fit(X_train, Y_train)

#Display best parameters
print("Tuned hyperparameters: (best parameters) ", svm_cv.best_params_)
print("Support Vector Machine best accuracy score (Training data):", svm_cv.best_score_)

#Calculate accuracy on test data
print("Support Vector Machine accuracy (Test data): ", svm_cv.score(X_test, Y_test))

#Create and plot confusion matrix for SVM
yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

#Create decision tree classifier, then a GridSearchCV to find best params
parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()

g_scv = GridSearchCV(tree, parameters, scoring = 'accuracy', cv = 10)
tree_cv = g_scv.fit(X_train, Y_train)

#Display best parameters
print("Tuned hyperparameters: (best parameters) ", tree_cv.best_params_)
print("Decision Tree best accuracy score (Training data) :", tree_cv.best_score_)

#Calculate accuracy on test data
print("Decision Tree accuracy (Test Data): ", tree_cv.score(X_test, Y_test))

#Create and plot confusion matrix for decision tree
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)

#Create KNN classifier, then a GridSearchCV to find best params
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()

g_scv = GridSearchCV(KNN, parameters, scoring = 'accuracy', cv = 10)
knn_cv = g_scv.fit(X_train, Y_train)

#Display best parameters
print("Tuned hyperparameters: (best parameters) ", knn_cv.best_params_)
print("KNN best accuracy score (Training data) :", knn_cv.best_score_)

#Calculate accuracy on test data
print("KNN accuracy (Test data): ", knn_cv.score(X_test, Y_test))

#Create and plot confusion matrix for decision tree
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)
