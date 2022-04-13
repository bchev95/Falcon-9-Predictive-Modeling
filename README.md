# Falcon-9-Predictive-Modeling
This repository consists of a Python program to train and test machine learning algorithms to predict if a SpaceX Falcon 9 first stage booster will successfully land or not.

After reading data into a pandas dataframe, it is standardized and split into training and testing sets. For each of the algorithms, cross-validation is performed using sklearn's GridSearchCV function to determine the best hyperparameters for the training set with that algorithm.

Once the parameters have been determined, the model is trained and then used to predict on the testing set. Finally, a confusion matrix is plotted to display prediction results.

Machine Learning Algorithms used:
- Logistic Regression
- Support Vector Machine
- Decision Tree Classifier
- K Nearest Neighbors


The Python file falcon9.py is a containerized version of one of the Jupyter notebooks for my capstone project for the IBM Data Science Professional Certificate.
