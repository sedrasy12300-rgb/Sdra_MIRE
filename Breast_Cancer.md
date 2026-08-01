Breast Cancer Prediction 


Project Overview

This project aims to predict whether a breast tumor is malignant or benign using Machine Learning classification algorithms.
The project includes data preprocessing, model training, hyperparameter tuning, and evaluation to select the best performing model.

Dataset
The dataset contains medical measurements of breast tumors.
The target variable:
Diagnosis 
Malignant (M)
Benign (B)
Data Preprocessing
The following steps were applied:
Converted the target variable into numerical values.
Removed unnecessary features.
Split the dataset into training and testing sets.
Machine Learning Models
The following models were trained and compared:
Random Forest Classifier
Logistic Regression
Model Optimization
Hyperparameter tuning was performed using:
GridSearchCV
Cross Validation
Recall was used as the optimization metric because detecting malignant cases correctly is important in medical prediction problems.
Model Evaluation
The models were evaluated using:
Accuracy Score
Confusion Matrix
Classification Report
Cross Validation Scores
Libraries Used
Python
Pandas
Scikit-learn
Conclusion
This project demonstrates a complete Machine Learning workflow for medical classification, including data preparation, model comparison, optimization, and evaluation.
