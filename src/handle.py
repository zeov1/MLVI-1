from src.settings import FEATURES_COUNT, RESOURCES_PATH, DATASET_NAME, MODEL_PATH, MODEL_NAME
from src.utils.graphs import plot_graphs, plot_predictions
from src.utils.models import prepare_data, train_model, load_trained_model, predict


def handle(L: int, M: int, Ns: int, epochs: int):
    model = None

    while not model:
        print('\nВыберите действие:\n'
              '1. Обучение модели\n'
              '2. Загрузка модели\n'
              '3. Построить график по исходным данным\n')
        choice = input('>>> ')

        if choice == '1':
            X_train, X_test, y_train, y_test, scaler, df, df_scaled = \
                prepare_data(RESOURCES_PATH / DATASET_NAME, 'BTC', M, L)
            model = train_model(X_train, y_train, M, FEATURES_COUNT, Ns, epochs, 32, MODEL_PATH / MODEL_NAME)
            if not model:
                print('Не удалось обучить модель.')
        elif choice == '2':
            X_train, X_test, y_train, y_test, scaler, df, df_scaled = \
                prepare_data(RESOURCES_PATH / DATASET_NAME, 'BTC', M, L)
            model = load_trained_model(filepath=MODEL_PATH / MODEL_NAME)
            if not model:
                print('Модель не найдена. Произведите обучение.')
        elif choice == '3':
            plot_graphs()
        else:
            print('Неверный вариант ответа')

    print('Введите Enter для начала предсказывания...')
    input('>>> ')

    test_predictions = predict(model, X_test, scaler, df_scaled)
    train_predictions = predict(model, X_train, scaler, df_scaled)
    test_dates = df_scaled.index[-len(y_test):]
    train_dates = df_scaled.index[:-len(y_test)]
    plot_predictions(df, test_dates, train_dates, test_predictions, train_predictions, L, M, Ns, epochs)
