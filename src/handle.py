from pathlib import Path

import pandas as pd
from keras import Sequential
from sklearn.preprocessing import MinMaxScaler

from graph_utils import plot_graph
from model_utils import create_dataset, train_model, predict, load_trained_model
from settings import RAW_DATA_PATH, MAIN_DATA_NAME


def handle():
    model: Sequential | None = None

    while not model:
        choice = input("Выберите действие:\n"
                       "1. Обучение модели\n"
                       "2. Загрузка модели\n"
                       "3. Построить график по исходным данным\n")

        if choice == '1':
            print('Запускаем обучение модели...')

            # Загрузка данных
            data = pd.read_csv(Path(RAW_DATA_PATH, MAIN_DATA_NAME))
            prices = data['Price'].values

            # Нормализация данных
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))

            # Подготовка данных для обучения
            M = 30  # Ширина окна наблюдения
            L = 5  # Шаги вперед для предсказания

            X, y = create_dataset(scaled_prices, M, L)
            X = X.reshape(X.shape[0], X.shape[1])  # Преобразование для FFNN

            # Обучение модели
            model = train_model(X, y, hidden_units=50, epochs=50, batch_size=32)

            print('Обучение модели завершено.')

        elif choice == '2':
            t = load_trained_model()
            if t:
                model = t
                print('Модель успешно загружена.')
            else:
                print('Обученная модель не найдена.')

        elif choice == '3':
            plot_graph()

        else:
            print('Неверный вариант ответа')

    input('Введите Enter для предсказания')
    predictions = predict(model, X)
    predictions = scaler.inverse_transform(predictions)  # Обратное преобразование
    print(predictions)
