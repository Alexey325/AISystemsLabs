import numpy as np

def mae(y_true, y_pred):
    # Средняя абсолютная ошибка, в единицах таргета.
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    # Корень из среднего квадрата ошибки: сильнее реагирует на крупные промахи, чем MAE.
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    # Доля разброса таргета, объяснённая моделью относительно предсказания средним.
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)
