import datetime
import time
from datetime import timedelta

import pandas as pd
from matplotlib import pyplot as plt

from src.settings import IMAGE_PATH, PREPROCESSED_DATA_PATH, CURRENCIES


def plot_graphs(path=PREPROCESSED_DATA_PATH):
    """Функция для вывода графика стоимости BTC по дням"""

    fig0, ax0 = plt.subplots(figsize=(10, 5))  # BTC
    fig1, ax1 = plt.subplots(figsize=(10, 5))  # SP500
    fig2, ax2 = plt.subplots(figsize=(10, 5))  # TWD
    fig3, ax3 = plt.subplots(figsize=(10, 5))  # HKD, CNY
    fig4, ax4 = plt.subplots(figsize=(10, 5))  # остальные валюты
    fig5, ax5 = plt.subplots(figsize=(10, 5))  # нормализованные данные

    for data_path in path.iterdir():
        filename = path / data_path
        currency = 'ERR'  # валюта не найдена :(
        for c, words in CURRENCIES.items():
            if any(word in str(data_path) for word in words):
                currency = c  # валюта найдена :)
                break
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])

        if currency == 'BTC':
            ax = ax0
        elif currency == 'SP500':
            ax = ax1
        elif currency == 'TWD':
            ax = ax2
        elif currency in ('HKD', 'CNY'):
            ax = ax3
        else:
            ax = ax4
        ax.plot(df['Date'], df['Price'], linestyle='-', label=currency)

        df['Price'] -= min(df['Price'])
        df['Price'] /= max(df['Price'])

        # Нормализованные данные:
        ax5.plot(df['Date'], df['Price'], linestyle='-', label=currency)

    # Вывод графиков
    for ax in (ax0, ax1, ax2, ax3, ax4, ax5):
        ax.set_xlabel('Дата')
        ax.grid()
        ax.legend()
    for ax in (ax0, ax1, ax2, ax3, ax4):
        ax.set_ylabel('Стоимость (USD)')
    ax5.set_ylabel('Стоимость (USD, нормализованная)')
    plt.show()


def plot_predictions(df: pd.DataFrame, test_dates, y_pred_real, L: int, M: int, Ns: int, target_col='BTC'):
    """Строит график фактических и предсказанных значений BTC"""
    plt.figure(figsize=(12, 6))

    # График реальных значений BTC (синий)
    plt.plot(df.index, df[target_col], label='Реальная стоимость BTC', color='blue', linewidth=1)

    # График предсказанных значений (красный)
    test_dates = list(map(lambda x: x - timedelta(days=L), test_dates))
    plt.plot(test_dates, y_pred_real, label='Предсказанная стоимость BTC', color='red', linestyle='-', linewidth=1)

    plt.xlim(pd.Timestamp('2017-03-01'), pd.Timestamp('2020-04-01'))
    plt.xlabel('Дата')
    plt.ylabel('Цена BTC')
    plt.title(f'Прогноз стоимости BTC (L={L}, M={M}, Ns={Ns})')
    plt.legend()
    plt.grid(True)

    plt.savefig(
        IMAGE_PATH / f'{datetime.date.today()} (L={L}, M={M}, Ns={Ns}) {round(time.time() * 2) % 100000}.png',
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


if __name__ == '__main__':
    plot_graphs()
    pass
