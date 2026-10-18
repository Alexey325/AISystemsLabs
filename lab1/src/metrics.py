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


def confusion_matrix(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))  # сказали 1, и это 1
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))  # сказали 1, а это 0
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))  # сказали 0, а это 1
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))  # сказали 0, и это 0
    return tp, fp, fn, tn


def accuracy(y_true, y_pred):
    # Доля правильных ответов среди всех.
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    # Из предсказанных единиц — какая доля действительно единицы.
    tp, fp, _, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp)


def recall(y_true, y_pred):
    # Из реальных единиц — какую долю модель нашла.
    tp, _, fn, _ = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn)


def f1(y_true, y_pred):
    # Гармоническое среднее precision и recall.
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r)


def log_loss(y_true, y_proba):
    # BCE: учитывает не класс, а уверенность.
    y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
    return float(-np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba)))
