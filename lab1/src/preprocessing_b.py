import numpy as np
import pandas as pd


class PreprocessorB:
    # Импутация пропусков, log1p трат, one-hot и стандартизация.

    def __init__(self, numeric, binary, categorical, spend):
        self.numeric = list(numeric)
        self.binary = list(binary)
        self.categorical = list(categorical)
        self.spend = list(spend)

        self.modes_ = {}  # моды бинарных и категориальных признаков
        self.medians_ = {}  # медианы числовых признаков
        self.levels_ = {}
        self.mean_ = None
        self.std_ = None

    def fit(self, df):
        for col in self.binary + self.categorical:
            self.modes_[col] = df[col].mode()[0]

        # Медиана трат считается только по бодрствующим: у пассажиров в криосне траты
        # всегда нулевые, и их 63% — медиана по всем дала бы 0 и обнулила признак
        awake = df[df["CryoSleep"] == False]
        for col in self.numeric:
            self.medians_[col] = (awake[col] if col in self.spend else df[col]).median()

        filled = self._impute(df)

        for col in self.categorical:
            levels = np.sort(filled[col].unique())
            self.levels_[col] = levels[1:]

        values = self._numeric_matrix(filled)
        self.mean_ = values.mean(axis=0)
        self.std_ = values.std(axis=0)
        return self

    def _impute(self, df):
        out = df.copy()

        for col in self.binary + self.categorical:
            out[col] = out[col].fillna(self.modes_[col])

        # Пассажир в криосне физически не может ничего потратить, пропуск означает 0;
        # для бодрствующего подставляем медиану бодрствующих из train
        in_cryo = out["CryoSleep"] == True
        for col in self.spend:
            fill = pd.Series(np.where(in_cryo, 0.0, self.medians_[col]), index=out.index)
            out[col] = out[col].fillna(fill)

        out["Age"] = out["Age"].fillna(self.medians_["Age"])
        return out

    def _numeric_matrix(self, filled):
        # Числовые столбцы в матрицу, траты в логарифмическую шкалу.
        values = filled[self.numeric].to_numpy(dtype=float)
        spend_idx = [self.numeric.index(col) for col in self.spend]
        values[:, spend_idx] = np.log1p(values[:, spend_idx])
        return values

    def transform(self, df):
        filled = self._impute(df)

        numeric = (self._numeric_matrix(filled) - self.mean_) / self.std_
        binary = filled[self.binary].to_numpy(dtype=float)  # True/False -> 1.0/0.0
        dummies = [
            (filled[col].to_numpy()[:, None] == self.levels_[col][None, :]).astype(float)
            for col in self.categorical
        ]

        return np.hstack([numeric, binary] + dummies)
