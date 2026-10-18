from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logistic_regression import LogisticRegression
from metrics import accuracy, confusion_matrix, f1, log_loss, precision, recall
from preprocessing_a import train_test_split_own
from preprocessing_b import PreprocessorB

LAB_DIR = Path(__file__).resolve().parent.parent  # папка lab1

DATA_PATH = LAB_DIR / "data/spaceship_titanic/train.csv"
PLOT_PATH = LAB_DIR / "pic/loss_b.png"
TARGET = "Transported"
TEST_SIZE = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.1
EPOCHS = 2000

DROP_SERVICE = ["PassengerId", "Name", "Cabin"]  # ID, имя и исходная каюта (разбирается на Deck и Side)

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]  # скошены, логарифмируем

NUMERIC = ["Age"] + SPEND  # стандартизуем
BINARY = ["CryoSleep", "VIP"]  # True/False -> 1/0
CATEGORICAL = ["HomePlanet", "Destination", "Deck", "Side"]  # one-hot


def split_cabin(df):
    parts = df["Cabin"].str.split("/", expand=True)  # формат каюты: палуба/номер/сторона
    return parts[0], parts[1], parts[2]


def explore_data(df):
    deck, _, side = split_cabin(df)
    rows_with_na = int(df.isna().any(axis=1).sum())
    spend_zero = (df[SPEND] == 0).mean()

    print("1. Исследование данных")
    print(f"размер {df.shape[0]}x{df.shape[1]} | строк хотя бы с одним пропуском {rows_with_na} "
          f"({100 * rows_with_na / len(df):.0f}%), по столбцам 2-3%")

    balance = df[TARGET].value_counts(normalize=True)
    print(f"таргет {TARGET}: True {balance[True]:.3f}, False {balance[False]:.3f} — классы сбалансированы")

    print(f"категории: HomePlanet {df['HomePlanet'].nunique()}, Destination {df['Destination'].nunique()}, "
          f"палуб {deck.nunique()}, сторон {side.nunique()}")
    print(f"траты: медиана {df[SPEND].median().max():.0f}, максимум {df[SPEND].max().max():.0f}, "
          f"нулевых {100 * spend_zero.min():.0f}-{100 * spend_zero.max():.0f}%")

    cryo = df.groupby("CryoSleep")[TARGET].mean()
    no_spend = df.assign(zero=df[SPEND].fillna(0).sum(axis=1) == 0).groupby("zero")[TARGET].mean()
    print(f"доля Transported: CryoSleep True {cryo[True]:.2f} / False {cryo[False]:.2f} | "
          f"нулевые траты {no_spend[True]:.2f} / {no_spend[False]:.2f}")
    print()


def build_x_y(df):
    deck, _, side = split_cabin(df)

    X_df = df.drop(columns=DROP_SERVICE + [TARGET]).assign(Deck=deck, Side=side)
    y = df[TARGET].to_numpy(dtype=float)  # bool -> 1.0 / 0.0

    print("2. Формирование X и y")
    print(f"X {X_df.shape}, y {y.shape}")
    print(f"Cabin разобран на Deck и Side, убраны {', '.join(DROP_SERVICE)}")
    print()

    return X_df, y


def split_data(X_df, y):
    train_idx, test_idx = train_test_split_own(len(X_df), TEST_SIZE, RANDOM_STATE)
    y_train, y_test = y[train_idx], y[test_idx]

    print("5. Train/test split")
    print(f"train {len(train_idx)}, test {len(test_idx)} (test_size={TEST_SIZE}, seed={RANDOM_STATE})")
    # при дисбалансе классов понадобилась бы стратификация, здесь классы и так 50/50
    print(f"доля класса 1: train {y_train.mean():.3f}, test {y_test.mean():.3f}")
    print()

    return X_df.iloc[train_idx], X_df.iloc[test_idx], y_train, y_test


def preprocess(X_train_df, X_test_df):
    prep = PreprocessorB(NUMERIC, BINARY, CATEGORICAL, SPEND).fit(X_train_df)

    X_train = prep.transform(X_train_df)
    X_test = prep.transform(X_test_df)

    print("3-4. Препроцессинг")
    print("пропуски: траты -> 0 у пассажиров в криосне, иначе медиана бодрствующих; "
          "Age -> медиана; категории и бинарные -> мода")
    print(f"медианы трат (бодрствующие): {', '.join(f'{c} {prep.medians_[c]:.0f}' for c in SPEND)}")
    print(f"log1p для {len(SPEND)} столбцов трат, one-hot для {', '.join(CATEGORICAL)}")
    print(f"X_train {X_train.shape}, X_test {X_test.shape}")
    print()

    return X_train, X_test


def train_model(X_train, y_train):
    model = LogisticRegression(learning_rate=LEARNING_RATE, epochs=EPOCHS).fit(X_train, y_train)

    print("6-10. Обучение градиентным спуском")
    # старт 0.6931 = ln(2): при нулевых весах модель всем даёт p = 0.5
    print(f"lr={LEARNING_RATE}, epochs={EPOCHS} | BCE {model.history_[0]:.4f} -> {model.history_[-1]:.4f}")
    print()

    return model


def plot_loss(model):
    plt.plot(model.history_)
    plt.title(f"BCE по эпохам (lr={LEARNING_RATE}, epochs={EPOCHS})")
    plt.xlabel("эпоха")
    plt.ylabel("BCE на train")
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")

    print("11. График loss")
    print(f"сохранён в {PLOT_PATH}")
    print()

    plt.show()


def evaluate(model, X_train, y_train, X_test, y_test):
    print("12. Метрики")

    for name, X, y_true in [("train", X_train, y_train), ("test ", X_test, y_test)]:
        y_pred, y_proba = model.predict(X), model.predict_proba(X)
        print(f"{name}: Accuracy {accuracy(y_true, y_pred):.3f}, Precision {precision(y_true, y_pred):.3f}, "
              f"Recall {recall(y_true, y_pred):.3f}, F1 {f1(y_true, y_pred):.3f}, "
              f"LogLoss {log_loss(y_true, y_proba):.4f}")

    print()

    tp, fp, fn, tn = confusion_matrix(y_test, model.predict(X_test))
    print("13. Confusion matrix (test)")
    print("                предсказали 1   предсказали 0")
    print(f"реально 1       TP = {tp:<11} FN = {fn}")
    print(f"реально 0       FP = {fp:<11} TN = {tn}")
    print()


def compare_with_baseline(model, X_test, y_train, y_test):
    major_class = float(y_train.mean() >= 0.5)  # класс, которого в train больше
    base_pred = np.full_like(y_test, major_class)  # этот класс каждому объекту test
    base_proba = np.full_like(y_test, y_train.mean())  # уверенность = доля класса 1, нужна для LogLoss

    model_pred, model_proba = model.predict(X_test), model.predict_proba(X_test)

    print("14. Сравнение с baseline")
    print(f"baseline (всегда класс {major_class:.0f}, p={y_train.mean():.3f}): "
          f"Accuracy {accuracy(y_test, base_pred):.3f}, Precision {precision(y_test, base_pred):.3f}, "
          f"Recall {recall(y_test, base_pred):.3f}, F1 {f1(y_test, base_pred):.3f}, "
          f"LogLoss {log_loss(y_test, base_proba):.4f}")
    print(f"наша модель: Accuracy {accuracy(y_test, model_pred):.3f}, "
          f"Precision {precision(y_test, model_pred):.3f}, Recall {recall(y_test, model_pred):.3f}, "
          f"F1 {f1(y_test, model_pred):.3f}, LogLoss {log_loss(y_test, model_proba):.4f}")
    print()


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    explore_data(data)

    X_df, y = build_x_y(data)
    X_train_df, X_test_df, y_train, y_test = split_data(X_df, y)
    X_train, X_test = preprocess(X_train_df, X_test_df)
    model = train_model(X_train, y_train)
    plot_loss(model)
    evaluate(model, X_train, y_train, X_test, y_test)
    compare_with_baseline(model, X_test, y_train, y_test)
