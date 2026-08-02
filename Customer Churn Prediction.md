Customer Churn Prediction

Overview
This project focuses on predicting customer churn using Machine Learning techniques.  
The goal is to identify customers who are likely to leave a service provider based on their characteristics and usage information.

Dataset
The dataset used in this project is the Telco Customer Churn Dataset.

It contains customer information such as:
- Demographic details
- Services subscribed
- Contract information
- Payment methods
- Monthly and total charges

The target variable is:

- Churn
  - 1 → Customer left the company
  - 0 → Customer stayed

Data Preprocessing
The following preprocessing steps were applied:

- Removed unnecessary columns (customerID)
- Converted the target column (Churn) into numerical values
- Converted TotalCharges into numeric format
- Handled missing values
- Converted categorical features into numerical values using One-Hot Encoding

Machine Learning Models

The following models were implemented:

1. Random Forest Classifier
- Used as the main classification model.
- Tuned using GridSearchCV.
- Optimized using Recall scoring to better detect customers who may leave.

2. Logistic Regression
- Implemented using a Pipeline with:
  - StandardScaler
  - Logistic Regression

Model Evaluation
The models were evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Cross Validation
- Recall
- Precision
- F1-score

Hyperparameter Optimization

GridSearchCV was used to find the best parameters for Random Forest:

- Number of estimators
- Maximum depth
- Class weights

Feature Importance

Feature importance was extracted from the best Random Forest model to understand which factors have the biggest impact on customer churn prediction.

Technologies Used

- Python
- Pandas
- Scikit-learn
- Machine Learning
- Random Forest
- Logistic Regression

## Project Structure
