import numpy as np


class LinearRegression:
    # Линейная регрессия ŷ = X @ w + b, обучение полнобатчевым градиентным спуском по MSE.

    def __init__(self, learning_rate=0.05, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = None
        self.b = None
        self.history_ = []  # loss (MSE) по эпохам

    def fit(self, X, y):
        # Подбор весов градиентным спуском.
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history_ = []

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y
            self.history_.append(float(np.mean(error ** 2)))

            dw = 2 / n_samples * (X.T @ error)  # градиент MSE по весам
            db = 2 * float(np.mean(error))  # градиент MSE по bias

            self.w -= self.learning_rate * dw  # шаг против градиента
            self.b -= self.learning_rate * db

        return self

    def predict(self, X):
        # Применение обученных параметров к новым данным.
        return X @ self.w + self.b
