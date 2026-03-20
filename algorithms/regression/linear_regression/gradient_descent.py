import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

for i in range(epochs):

    y_pred = m * X + b

    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    m = m - learning_rate * dm
    b = b - learning_rate * db

    if i % 100 == 0:
        loss = np.mean((y - y_pred) ** 2)
        print(f"Epoch {i}: Loss={loss}, m={m}, b={b}")
