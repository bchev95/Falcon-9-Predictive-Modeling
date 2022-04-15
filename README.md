# Falcon-9-Predictive-Modeling
This repository consists of a series of Jupyter notebooks to gather data from SpaceX API, format that data and use it to train and test machine learning algorithms to predict if a SpaceX Falcon 9 first stage booster will successfully land or not.

### Notebook Progression:
1. **falcon9_API_data.ipynb** - Makes request to spacexdata API, formats data and stores in **falcon9_data.csv**

2. **falcon9_analysis_feature_engineering.ipynb** - Perform exploratory data analysis to determine important attributes, then feature engineering for one-hot encoding. Exports one-hot encoded data table to **falcon9_onehot.csv**
3. **falcon9_predictive_modeling.ipynb** - Reads in .csv data, standardizes and splits data into training and testing sets. Four different machine learning algorithms are used; Cross-validation is performed to determine best hyperparameters for each algorithm, and models are trained and used to predict on testing set. Accuracy results are confusion matrices are then displayed. 


### Machine Learning Algorithms used:
- Logistic Regression
- Support Vector Machine
- Decision Tree Classifier
- K Nearest Neighbors


The Python file falcon9.py is a containerized version of one of the Jupyter notebooks for my capstone project for the IBM Data Science Professional Certificate.
