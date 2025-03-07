import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, Ridge

# === training and evaluatng the models ===

def train_knn(X_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, y_train)
    return knn

def evaluate_knn(knn, X_val, y_val):
    y_pred = knn.predict(X_val)
    # in binary classification we will choose 'g' to be the positive label and 'h' to be the negative
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, pos_label='g'), 
        'recall': recall_score(y_val, y_pred, pos_label='g'),
        'f1_score': f1_score(y_val, y_pred, pos_label='g'),
        'confusion_matrix': confusion_matrix(y_val, y_pred)
    }
    return metrics

def try_different_k_values(X_train, y_train, X_val, y_val, k_values=[3, 5, 7, 9]):
    results = {}
    for k in k_values:
        knn = train_knn(X_train, y_train, k)
        results[k] = evaluate_knn(knn, X_val, y_val)
    return results

# train the data on Linear Regression and return the trained model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# train the data on Lasso Regression and return the trained model
def train_lasso(X_train, y_train, alpha=1.0):
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

# train the data on Ridge Regression and return the trained model
def train_ridge(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

# evaluate the regression model (Linear, Ridge or Lasso) and return the mean squared error (mse) and maen absolute error (mae)
def evaluate_regression(model, X_val, y_val):
    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    return mse, mae
