from data_preprocessing import *
from classification_regression import *
import pandas as pd

classification_df = pd.read_csv('data/magic04.data')
balanced_classification_df = balance_classes(classification_df)

X_train, X_val, X_test, y_train, y_val, y_test = split_classification_data(balanced_classification_df)

knn = train_knn(X_train, y_train, 3)
knn_results = evaluate_knn(knn, X_val, y_val)

print("Classification result\n\n")
for metric in knn_results:
    print(f'{metric} : {knn_results[metric]}')


regression_df = pd.read_csv('data/California_Houses.csv')

X_train, X_val, X_test, y_train, y_val, y_test = split_regression_data(regression_df)

linear_model = train_linear_regression(X_train, y_train)
lasso_model = train_lasso(X_train, y_train)
ridge_model = train_ridge(X_train, y_train)

print("\n")

print("\nEvaluatoin of linear model")
linear_result = evaluate_regression(linear_model, X_val, y_val)
print(f'MSE: {linear_result[0]}, MAE {linear_result[1]}')

print("\nEvaluatoin of lasso model")
lasso_result = evaluate_regression(lasso_model, X_val, y_val)
print(f'MSE: {lasso_result[0]}, MAE {lasso_result[1]}')

print("\nEvaluatoin of ridge model")
ridge_result = evaluate_regression(ridge_model, X_val, y_val)
print(f'MSE: {ridge_result[0]}, MAE {ridge_result[1]}')