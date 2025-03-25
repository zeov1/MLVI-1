from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

RESOURCES_PATH = ROOT_PATH / 'resources'

RAW_DATA_PATH = RESOURCES_PATH / 'data_raw'

PREPROCESSED_DATA_PATH = RESOURCES_PATH / 'data_preprocessed'
MAIN_DATA_NAME = 'bitcoin_price.csv'

MODEL_PATH = RESOURCES_PATH / 'models'
MODEL_NAME = 'bitcoin_price_prediction_model_ffnn.keras'
