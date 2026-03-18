import numpy as np

def vinit_mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)