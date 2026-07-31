Titanic Survival Prediction 🚢

Project Overview

This project aims to predict whether a passenger survived the Titanic disaster using Machine Learning algorithms.

The project includes data preprocessing, feature engineering, model training, and evaluation to find the best performing model.
Dataset
The dataset used in this project is the Titanic dataset, which contains information about passengers such as:

Age
Sex
Passenger class
Family information
Embarkation location
Fare
Data Preprocessing
The following steps were applied:
Handling missing values in Age and Embarked.
Converting categorical features into numerical values.
Encoding passenger titles extracted from names.
Creating a new feature: Family Size.
Removing unnecessary columns.
Machine Learning Models
Several models were trained and compared:
Random Forest Classifier
Logistic Regression
Hyperparameter tuning was performed using:
GridSearchCV
Cross Validation
A Pipeline was also used with:
StandardScaler
Logistic Regression
Model Evaluation
The models were evaluated using:
Accuracy Score
Confusion Matrix
Cross Validation Score
Tools & Libraries
Python
Pandas
Scikit-learn
NumPy
Conclusion
This project demonstrates a complete Machine Learning workflow, from data preprocessing to model evaluation and selection.
