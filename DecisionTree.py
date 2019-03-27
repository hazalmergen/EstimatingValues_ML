import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, accuracy_score

#Read data
dataset = pd.read_csv('datas.csv')
X = dataset.iloc[0:100, 1:6].values
y = dataset.iloc[0:100, 6].values

#K-Fold cross validation
kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=False)
y_tests = []
y_preds = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #Model fitting
    regressor = DecisionTreeRegressor(max_depth=10,random_state=0)
    regressor.fit(X_train,y_train)
    # Predicting values using our trained model
    y_pred = regressor.predict(X_test)
    y_preds += list(y_pred)
    y_tests += list(y_test)

plt.scatter(y_tests,y_preds)
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
