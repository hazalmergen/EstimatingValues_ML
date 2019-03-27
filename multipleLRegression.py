import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import statsmodels.formula.api as sm
import numpy as np

#Read data
dataset = pd.read_csv('datas.csv')
X = dataset.iloc[0:100, 1:6].values #(it takes from x1 to x5 )
#X = dataset.iloc[0:100, 2:4].values #(it takes x2 and x3's columns information)
y = dataset.iloc[0:100, 6].values

#K-folds Cross Validation
kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=True)

y_tests = []
y_preds = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #Model fitting
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # Predicting values using our trained model
    y_pred = regressor.predict(X_test)
    y_preds += list(y_pred)
    y_tests += list(y_test)

# #Backward Elimination Process ( This shows the backward elimination process. I used it, to find the relevant ones. )
# X=np.append(arr = np.ones((100, 1)).astype(int), values = X, axis = 1)
#
# #Check all the variables's p values, eliminate the variable with highest value.
# X_opt = X[:,[0,1,2,3,4,5]]
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# print(regressor_OLS.summary())
#
# X_opt = X[:,[0,2,3,4,5]]
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# print(regressor_OLS.summary())
#
# X_opt = X[:,[0,2,3,5]]
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# print(regressor_OLS.summary())
#
# X_opt = X[:,[0,2,3]]
# regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# print(regressor_OLS.summary())
#
# X_Opt = X[:,[2,3]] # Meaning that x2 and x3 affects y value more than other independent variables.
#
# print(X_Opt)

plt.scatter(y_tests,y_preds)
plt.plot(y_tests, y_tests, color="black")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

ex_var_score = explained_variance_score(y_tests, y_preds, multioutput='uniform_average')
m_absolute_error = mean_absolute_error(y_tests, y_preds)
m_squared_error = mean_squared_error(y_tests, y_preds)
r_2_score = r2_score(y_tests, y_preds)

print("Explained Variance Score: "+str(ex_var_score))
print("Mean Absolute Error: "+str(m_absolute_error))
print("Mean Squared Error: "+str(m_squared_error))
print("R Squared Error: "+str(r_2_score))
print("Coefficients of independent variables",regressor.coef_)