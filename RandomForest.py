import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

#Read data
dataset = pd.read_csv('datas.csv')
X = dataset.iloc[0:100, 1:6].values
y = dataset.iloc[0:100, 6].values
X_Pred = dataset.iloc[100:120, 1:6].values

#K-folds Cross Validation
kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=False)
y_preds = []
y_tests = []

#Predictions from 101th data point to 120th data point
Y_final_pred = np.array([])

flag=0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #Model fitting
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X_train, y_train)
    # Predicting values using our trained model
    y_pred = regressor.predict(X_test)
    y_preds += list(y_pred)
    y_tests += list(y_test)
    if flag==0:
        Y_final_pred = regressor.predict(X_Pred)
        flag=1
    else:
        Y_final_pred = np.vstack([Y_final_pred, regressor.predict(X_Pred)])


#Since there are 10 y value predictions for each data point so, we take the mean of those 10 predictions for one actual y value.
print(Y_final_pred)
Y_final_pred = np.mean(Y_final_pred, axis=0)

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
print(Y_final_pred)