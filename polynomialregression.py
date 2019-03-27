import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

#Read data
dataset = pd.read_csv('datas.csv')

X = dataset.iloc[0:100, 1:6].values
#X = dataset.iloc[0:100, 2:4].values #(it takes x2 and x3's columns information)
y = dataset.iloc[0:100, 6].values

#K-folds Cross Validation
kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=False)

y_tests = []
y_preds = []

for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #poly_reg = PolynomialFeatures(degree=2)
        poly_reg = PolynomialFeatures(degree = 3)
        #Model fitting
        X_poly = poly_reg.fit_transform(X_train)
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y_train)
        # Predicting values using our trained model
        y_pred = lin_reg.predict(poly_reg.fit_transform(X_test))
        y_preds += list(y_pred)
        y_tests += list(y_test)

plt.scatter(y_tests, y_preds)
plt.plot(y_tests, y_tests, color="black")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

ex_var_score = explained_variance_score(y_tests, y_preds)
m_absolute_error = mean_absolute_error(y_tests, y_preds)
m_squared_error = mean_squared_error(y_tests, y_preds)
r_2_score = r2_score(y_tests, y_preds)

print("Explained Variance Score: "+str(ex_var_score))
print("Mean Absolute Error "+str(m_absolute_error))
print("Mean Squared Error "+str(m_squared_error))
print("R Squared Error "+str(r_2_score))
