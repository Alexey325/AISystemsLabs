import numpy as np

def train_test_split_own(n_samples, test_size=0.2, random_state=42):
    # Перемешивает номера строк с фиксированным seed и режет на train и test.
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(n_samples)

    n_test = int(round(n_samples * test_size))
    return shuffled[n_test:], shuffled[:n_test]


class Preprocessor:

    def __init__(self, numeric, binary, categorical):
        self.numeric = list(numeric)
        self.binary = list(binary)
        self.categorical = list(categorical)

        self.levels_ = {}
        self.mean_ = None
        self.std_ = None

    def fit(self, df):
        # Вычисляет статистики по обучающей выборке.
        for col in self.categorical:
            levels = np.sort(df[col].unique())
            self.levels_[col] = levels[1:]  # первый уровень — точка отсчёта (dummy trap)

        values = df[self.numeric].to_numpy(dtype=float)
        self.mean_ = values.mean(axis=0)
        self.std_ = values.std(axis=0)
        return self

    def transform(self, df):
        # Применяет вычисленные на fit статистики. Возвращает матрицу признаков (N, D).
        numeric = (df[self.numeric].to_numpy(dtype=float) - self.mean_) / self.std_
        binary = df[self.binary].to_numpy(dtype=float)
        dummies = [
            (df[col].to_numpy()[:, None] == self.levels_[col][None, :]).astype(float)
            for col in self.categorical
        ]

        return np.hstack([numeric, binary] + dummies)
