import numpy as np
import pandas as pd
from keras.api.layers import Dense, Flatten
from keras.api.models import Sequential
from keras.src.saving import load_model
from keras.src.saving.saving_api import save_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.settings import RESOURCES_PATH


def prepare_data(file_path, target_col, M, L):
    """Загружает данные, нормализует и формирует обучающие примеры"""
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date").sort_index()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    df_scaled.to_csv(RESOURCES_PATH / 'scaled_dataset.csv')

    def create_sequences(data, target_col, M, L):
        X, y = [], []
        for i in range(len(data) - M - L):
            X.append(data.iloc[i: i + M].values)
            y.append(data.iloc[i + M + L - 1][target_col])
        return np.array(X), np.array(y)

    X, y = create_sequences(df_scaled, target_col, M, L)
    # Разобьем набор данных в соотношении 80/20, где 80% - обучающие примеры
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    return X_train, X_test, y_train, y_test, scaler, df, df_scaled


def train_model(X_train, y_train, M, feature_count, Ns, epochs, batch_size, save_path):
    """Создает и обучает модель"""
    model = Sequential([
        Flatten(input_shape=(M, feature_count)),
        Dense(Ns, activation="relu"),
        Dense(Ns // 2, activation="relu"),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    save_model(model, save_path)
    return model


def load_trained_model(filepath) -> Sequential | None:
    """Загружает обученную модель, если такая существует"""
    if filepath.exists():
        return load_model(filepath)


def predict(model: Sequential, X_test, scaler: MinMaxScaler, df_scaled: pd.DataFrame, target_col='BTC'):
    """Выполняет предсказание и возвращает значения BTC"""
    y_pred = model.predict(X_test).flatten()
    btc_index = df_scaled.columns.get_loc(target_col)
    y_pred_real = scaler.inverse_transform(np.column_stack([y_pred] * len(df_scaled.columns)))[:, btc_index]
    return y_pred_real
