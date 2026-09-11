# Retail Churn Ensemble Models

Ensemble learning models predicting e-commerce customer churn for a retail analytics client, comparing tree-based methods for both accuracy and interpretability.

## Files

- **`retail_churn_ensemble_lab.ipynb`** — EDA, baseline Random Forest, AdaBoost/Gradient Boosting/XGBoost comparison, targeted `GridSearchCV` tuning, final model evaluation, and feature importance.
- **`ecommerce_customer_data.csv`** — 15,000 customer records, 14 features (usage patterns, support interactions, account details) plus the `churn` target.

## Approach

1. EDA: churn class distribution, correlation with churn
2. 80/20 train-test split (`random_state=42`)
3. Baseline `RandomForestClassifier` evaluated via cross-validated accuracy
4. Untuned AdaBoost, Gradient Boosting, and XGBoost compared on train vs. CV accuracy
5. XGBoost (largest train/CV gap → most overfitting) tuned via `GridSearchCV`
6. Final model evaluated on the test set — accuracy, classification report, confusion matrix, feature importance

## Note

Accuracy was the required metric for this lab, but given the business cost of missing an actual churner, **recall** is flagged as a stronger metric to optimize for in a production setting.