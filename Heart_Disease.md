Heart Disease Prediction 

Project Overview
This project predicts the presence of heart disease using Machine Learning techniques. The goal is to compare different classification models and improve performance through hyperparameter tuning.
Dataset
The project uses the NHANES 2017–2018 Heart Disease Prediction dataset, which contains health-related information used to predict whether a patient has heart disease.
Data Preparation
Loaded and prepared the dataset.

Split the data into training and testing sets.
Used the target variable heart_disease for prediction.

Machine Learning Models
The following models were implemented and compared:
Random Forest Classifier
Logistic Regression
Model Optimization
Hyperparameter tuning was performed using GridSearchCV with Recall as the optimization metric to improve the detection of heart disease cases.
Model Evaluation
The models were evaluated using:
Accuracy Score
Confusion Matrix
Classification Report
Cross Validation
Libraries
Python
Pandas
Scikit-learn
Conclusion
This project demonstrates a complete machine learning workflow, including model training, evaluation, hyperparameter tuning, and performance comparison for heart disease prediction.
