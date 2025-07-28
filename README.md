# Customer Churn Prediction

This project provides a complete solution for predicting customer churn using machine learning.

# Pipeline

A production-ready machine learning solution for predicting customer churn with:  
✔ **Data Retrieval from static csv file**  
✔ **Automated data preprocessing**  
✔ **Model comparison & selection**  
✔ **Hyperparameter optimization**  
✔ **Comprehensive evaluation**  
✔ **Unit Test Example**  

## Key Features

### Data Processing
- **Automated handling of missing values**  
  - Median imputation for numerical features  
- **Outlier detection and treatment**  
  - Log-transform for skewed distributions  
  - 99th percentile capping for extreme values  
- **Correlation analysis**  
  - Automatic alerts for features with >0.8 correlation  
- **Class imbalance handling**  
  - Auto-SMOTE activation when churn rate <10%  
- **Feature engineering**  
  - Standard scaling (StandardScaler)  
  - One-hot encoding (OneHotEncoder)  

### Model Development
- **Cross-validated model comparison**  
  - 5-fold stratified cross-validation  
  - F1 score as primary metric (optimizes precision/recall balance)  
- **Supported algorithms**  
  - Logistic Regression 
  - Random Forest  
  - Gradient Boosting  

### Optimization & Evaluation
- **Hyperparameter tuning**  
  - GridSearchCV with predefined parameter grids  
- **Test-set metrics**
  - **Primary metric**: F1 Score 
  - Accuracy, Precision, Recall  
  - ROC-AUC  
  - Confusion matrix  
- **Model interpretability**  
  - SHAP value analysis for handoff to business/marketing 
  - Feature importance plots  

### Quality Assurance Example
- **Unit tests**  
  - Data validation (value ranges, missing values)

### Configuration Changes
- **config.py**  
  - to change default settings for models.

## Requirements

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - pytest (for testing)

