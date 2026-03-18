import numpy as np
from algorithms.regression.linear_regression.vinit_linear_regression import VinitLinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([1,2,3,4,5])
y = np.array([2,4,5,4,5])

model = VinitLinearRegression()

model.fit(X, y)

prediction = model.predict(X)
mse = mean_squared_error(y, prediction)

print("slope: " ,model.slope)#0.6
print("intercept: ", model.intercept)#2.2
print("predictions: ", prediction)##[2.8 3.4 4.  4.6 5.2]


print("MSE:", mse)#0.48