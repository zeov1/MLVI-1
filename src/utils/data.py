from datetime import datetime, timedelta

import pandas as pd

from src.settings import (
    RAW_DATA_PATH,
    PREPROCESSED_DATA_PATH,
    MAIN_DATA_NAME,
    DATASET_NAME,
    RESOURCES_PATH, CURRENCIES,
)

target_col1_name = 'Date'
target_col2_name = 'Price'

start_date = datetime(year=2010, month=7, day=18)
end_date = datetime(year=2019, month=11, day=14)


def merge_to_dataset():
    """Создает датасет на основе подготовленных csv-файлов"""
    df = pd.DataFrame()
    df['Date'] = pd.read_csv(PREPROCESSED_DATA_PATH / MAIN_DATA_NAME)['Date']
    for data_path in PREPROCESSED_DATA_PATH.iterdir():
        filename = data_path.name
        for currency, words in CURRENCIES.items():
            if any(word in filename for word in words):
                df[currency] = pd.read_csv(data_path)['Price']
                break

    df.to_csv(RESOURCES_PATH / DATASET_NAME, index=False)


def sorted_by_first_date() -> list:
    """
    :return: Список пар значений [filename, date], отсортированных по дате, где date -- начальная дата
    """
    res = []

    for raw_data_path in RAW_DATA_PATH.iterdir():
        filename = raw_data_path.name.split('.')[0]
        df = pd.read_csv(raw_data_path)
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        first_date = min(df[df.columns[0]])
        res.append([filename, first_date.date()])

    res.sort(key=lambda x: x[1])
    res = list(map(lambda x: [x[0], str(x[1])], res))

    return res


def sorted_by_last_date() -> list:
    """
    :return: Список пар значений (filename, date), отсортированных по дате, где date -- конечная дата
    """
    res = []

    for raw_data_path in RAW_DATA_PATH.iterdir():
        filename = raw_data_path.name.split('.')[0]
        df = pd.read_csv(raw_data_path)
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        last_date = max(df[df.columns[0]])
        res.append((filename, last_date.date()))

    res.sort(key=lambda x: x[1])
    res = list(map(lambda x: (x[0], str(x[1])), res))

    return res


def print_max_available_date_range():
    max_first_date = sorted_by_first_date()[-1]
    print(f'Начало промежутка: {max_first_date[1]} (ограничено {max_first_date[0]})')
    min_last_date = sorted_by_last_date()[0]
    print(f'Конец промежутка:  {min_last_date[1]} (ограничено {min_last_date[0]})')
    print(f'Промежуток: ({max_first_date[1]}, {min_last_date[1]})')


def _handle_price(price) -> str:
    if isinstance(price, str):
        return price.replace(',', '')
    elif isinstance(price, float):
        return str(price)


def crop_and_complete():
    """
    Функция, приводящая CSV-файлы со статистикой по
    отдельным финансовым показателям к единому виду.

    CSV-файлы хранятся по адресу RAW_DATA_PATH (см. settings.py)
    и обязательно имеют первыми двумя столбцами дату и стоимость.
    """
    for raw_data_path in RAW_DATA_PATH.iterdir():
        filename = raw_data_path.name
        df = pd.read_csv(raw_data_path)
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], yearfirst=True)

        df.sort_values(by=df.columns[0], inplace=True)

        # DEBUG
        if 'S&P' in filename:
            pass

        dates = []
        prices = []

        curr_date = start_date
        last_price = 0

        for index, row in df.iterrows():
            date_from_csv = datetime.fromisoformat(str(row[df.columns[0]]))
            if date_from_csv > end_date:
                break

            curr_price = _handle_price(row[df.columns[1]])
            if curr_price:
                last_price = curr_price

            while date_from_csv > curr_date:
                dates.append(curr_date)
                prices.append(last_price)
                curr_date += timedelta(days=1)

            if date_from_csv == curr_date:
                dates.append(curr_date)
                curr_price = _handle_price(row[df.columns[1]])
                if curr_price:
                    last_price = curr_price
                    prices.append(curr_price)
                else:
                    prices.append(last_price)
                curr_date += timedelta(days=1)

        output_df = pd.DataFrame({
            target_col1_name: dates,
            target_col2_name: prices,
        })
        output_df.to_csv(PREPROCESSED_DATA_PATH / filename, index=False)


if __name__ == '__main__':
    print_max_available_date_range()
    crop_and_complete()
    merge_to_dataset()
