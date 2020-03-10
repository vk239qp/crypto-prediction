from src.neural_network.lstm_bitcoin import LSTMBitcoin
from src.pipeline.pipeline import Pipeline
from src.utils.data_util import DataUtil
from src.utils.enums.crypto_enum import CryptoEnum
from src.utils.scrapper import Scrapper

PREV_DAYS = 30
FUTURE_DAYS = 7
LAST_SAMPLES = 1000

if __name__ == '__main__':
    scrapper = Scrapper(CryptoEnum.BITCOIN)
    # data_util = DataUtil()
    # lstm_bitcoin = LSTMBitcoin(PREV_DAYS, FUTURE_DAYS, 3)

    pipe = Pipeline()
    pipe.process(scrapper)
