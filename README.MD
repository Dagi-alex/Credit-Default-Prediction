#  Credit Default Prediction

A machine learning project for predicting credit card payment default using
sequential feature selection, model explainability, hyperparameter tuning,
and classification model benchmarking.

##  Project Overview

The project starts with **23 predictor features** and applies a three-stage
feature-selection pipeline:

1. **Correlation Filtering**
2. **Information Value (IV)**
3. **SHAP Feature Importance**

The feature set is reduced from:

**23 -> 16 -> 15 final features**

Six machine learning classifiers are then tuned and compared using
GridSearchCV with Stratified 5-Fold Cross-Validation.

##  Models Compared

- Logistic Regression
- Random Forest
- AdaBoost
- LightGBM
- CatBoost
- XGBoost

##  Model Performance

| Model | Test ROC-AUC |
|  ---  |    ---:      |
|**CatBoost**| **0.7808**|
|XGBoost|   0.7786    |
|LightGBM| 0.7770    |
|Random Forest | 0.7746 |
|AdaBoost | 0.7683 |
|Logistic Regression | 0.7119 |

## Best Model

**CatBoost** achieved the highest Test ROC-AUC:

**0.7808**

The final model uses **15 selected features**, compared with the original
23 predictors.

## Feature Selection

### Correlation Filtering
Highly correlated predictors were identified and compared using their
individual predictive performance.

**23 -> 16 features**

### Information Value
Information Value was used to evaluate the predictive separation of the
remaining variables.

**16 -> 15 features**

### SHAP
SHAP values were used to rank the final features according to their
contribution to LightGBM predictions.

`PAY_0` was identified as the most influential feature.

## Evaluation Metrics

Models were evaluated using:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost
- SHAP

## Repository Structure

```text
Credit-Default-Prediction/
|- CreditDefaultPrediction.ipynb
|- credit_scoring.csv
|- README.md
|- requirements.txt
|- .gitignore