import numpy as np

class VinitLinearRegression:

    def __init__(self):
        self.slope = None  #objects 
        self.intercept = None

    def fit(self, X, y):#.fit function

        X = np.array(X)
        y = np.array(y)

        x_mean = np.mean(X) # mean of x and y
        y_mean = np.mean(y)

        numerator = np.sum((X - x_mean) * (y - y_mean)) ## covariance 
        denominator = np.sum((X - x_mean) ** 2) ## variance 

        self.slope = numerator / denominator  # slope formula 
        self.intercept = y_mean - self.slope * x_mean # intercept formula 

    def predict(self, X):

        X = np.array(X)

        return self.slope * X + self.intercept


    

