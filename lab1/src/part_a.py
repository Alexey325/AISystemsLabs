import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linear_regression import LinearRegression
from metrics import mae, r2, rmse
from preprocessing import Preprocessor, train_test_split_own

DATA_PATH = "../data/bike_sharing/day.csv"
PLOT_PATH = "../pic/loss.png"
TARGET = "cnt"
TEST_SIZE = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.05
EPOCHS = 10000
PLOT_EPOCHS = 300

DROP_LEAKAGE = ["casual", "registered"]  # части таргета: cnt = casual + registered
DROP_SERVICE = ["instant", "dteday"]  # номер строки и сырая дата

NUMERIC = ["temp", "atemp", "hum", "windspeed"]  # стандартизуем
BINARY = ["yr", "holiday", "workingday"]  # уже 0/1
CATEGORICAL = ["season", "mnth", "weekday", "weathersit"]  # one-hot


def explore_data(df):
    print("1. Исследование данных")
    print(f"размер {df.shape[0]}x{df.shape[1]} | пропусков {int(df.isna().sum().sum())} | "
          f"дубликатов {int(df.duplicated().sum())} | cnt {df[TARGET].min()}..{df[TARGET].max()}, "
          f"среднее {df[TARGET].mean():.0f}")
    print(f"cnt == casual + registered: "
          f"{bool((df['casual'] + df['registered'] == df[TARGET]).all())}")

    corr = df[NUMERIC + BINARY + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    print("корреляция с cnt:", ", ".join(f"{name} {value:.2f}" for name, value in corr.items()))
    print()


def build_x_y(df):
    X_df = df.drop(columns=DROP_LEAKAGE + DROP_SERVICE + [TARGET])
    y = df[TARGET].to_numpy(dtype=float)

    print("2. Формирование X и y")
    print(f"X {X_df.shape}, y {y.shape}")
    print(f"убраны {', '.join(DROP_LEAKAGE)} (leakage), {', '.join(DROP_SERVICE)} (служебные)")
    print()

    return X_df, y


def split_data(X_df, y):
    train_idx, test_idx = train_test_split_own(len(X_df), TEST_SIZE, RANDOM_STATE)

    print("4. Train/test split")
    print(f"train {len(train_idx)}, test {len(test_idx)} (test_size={TEST_SIZE}, seed={RANDOM_STATE})")
    print()

    return X_df.iloc[train_idx], X_df.iloc[test_idx], y[train_idx], y[test_idx]


def preprocess(X_train_df, X_test_df):
    prep = Preprocessor(NUMERIC, BINARY, CATEGORICAL).fit(X_train_df)

    X_train = prep.transform(X_train_df)
    X_test = prep.transform(X_test_df)

    print("3. Preprocessing")
    print(f"X_train {X_train.shape}, X_test {X_test.shape}")
    print()

    return X_train, X_test


def train_model(X_train, y_train):
    model = LinearRegression(learning_rate=LEARNING_RATE, epochs=EPOCHS).fit(X_train, y_train)

    print("5-8. Обучение градиентным спуском")
    print(f"lr={LEARNING_RATE}, epochs={EPOCHS} | MSE {model.history_[0]:.0f} -> {model.history_[-1]:.0f}")
    print()

    return model


def plot_loss(model):
    plt.plot(model.history_[:PLOT_EPOCHS])
    plt.title(f"Loss по эпохам (lr={LEARNING_RATE}): первые {PLOT_EPOCHS} эпох из {EPOCHS}")
    plt.xlabel("эпоха")
    plt.ylabel("MSE на train")
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")

    print("9. График loss")
    print(f"сохранён в {PLOT_PATH}")
    print()

    plt.show()


def evaluate(model, X_train, y_train, X_test, y_test):
    print("10. Метрики")

    for name, X, y_true in [("train", X_train, y_train), ("test ", X_test, y_test)]:
        y_pred = model.predict(X)
        print(f"{name}: MAE {mae(y_true, y_pred):.0f}, RMSE {rmse(y_true, y_pred):.0f}, "
              f"R2 {r2(y_true, y_pred):.3f}")

    print()


def compare_with_baseline(model, X_test, y_train, y_test):
    baseline_pred = np.full_like(y_test, y_train.mean())
    model_pred = model.predict(X_test)

    print("11. Сравнение с baseline")
    print(f"baseline (всегда {y_train.mean():.0f}): MAE {mae(y_test, baseline_pred):.0f}, "
          f"RMSE {rmse(y_test, baseline_pred):.0f}, R2 {r2(y_test, baseline_pred):.3f}")
    print(f"модель точнее baseline в {mae(y_test, baseline_pred) / mae(y_test, model_pred):.1f} раза по MAE")
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
