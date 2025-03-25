from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

RESOURCES_PATH = ROOT_PATH / 'resources'

RAW_DATA_PATH = RESOURCES_PATH / 'data_raw'

PREPROCESSED_DATA_PATH = RESOURCES_PATH / 'data_preprocessed'
MAIN_DATA_NAME = 'bitcoin-price.csv'
DATASET_NAME = 'dataset.csv'

MODEL_PATH = RESOURCES_PATH / 'models'
MODEL_NAME = 'bitcoin_price_prediction_model_ffnn.keras'

IMAGE_PATH = RESOURCES_PATH / 'images'

CURRENCIES = {
    'BTC': ('bitcoin',),
    'CAD': ('canad',),
    'HKD': ('hong', 'kong',),
    'SP500': ('S&P',),
    'TWD': ('thaiwan',),
    'EUR': ('euro',),
    'GBP': ('pound', 'sterling',),
    'CNY': ('yuan',),
}

FEATURES_COUNT = len(CURRENCIES)
