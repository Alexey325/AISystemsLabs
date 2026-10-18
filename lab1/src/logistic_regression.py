import numpy as np

from metrics import log_loss


class LogisticRegression:
    # Логистическая регрессия: p = sigmoid(X @ w + b), обучение градиентным спуском по BCE.

    def __init__(self, learning_rate=0.1, epochs=2000, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold  # граница, по которой вероятность превращается в класс

        self.w = None
        self.b = None
        self.history_ = []  # loss (BCE) по эпохам

    @staticmethod
    def _sigmoid(z):
        # Сжимает любое число в диапазон (0, 1): большие z -> ближе к 1, малые -> ближе к 0.
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Подбор весов градиентным спуском по BCE.
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history_ = []

        for _ in range(self.epochs):
            p = self.predict_proba(X)
            error = p - y

            self.history_.append(log_loss(y, p))  # loss = BCE

            dw = (X.T @ error) / n_samples  # градиент BCE по весам
            db = float(np.mean(error))  # градиент BCE по bias

            self.w -= self.learning_rate * dw  # шаг против градиента
            self.b -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        # Вероятность класса 1 для каждого объекта.
        z = X @ self.w + self.b  # линейная часть (score), как в линейной регрессии
        return self._sigmoid(z)

    def predict(self, X):
        # Жёсткий ответ: вероятность выше порога -> класс 1, иначе 0.
        return (self.predict_proba(X) >= self.threshold).astype(float)
