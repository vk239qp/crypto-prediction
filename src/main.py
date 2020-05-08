from src.helpers.CryptoCurrency import CryptoCurrency
from src.helpers.Runner import Runner
from src.pipeline.pipeline import Pipeline
from src.stages.execution.lstm.lstm_ethereum import LSTMEthereum
from src.stages.execution.lstm.lstm_bitcoin import LSTMBitcoin
from src.stages.execution.tester import Tester
from src.stages.operation.preprocessor import Preprocessor
from src.stages.source.scrapper import Scrapper

if __name__ == '__main__':
    # 0 - train, 1 - test, 2 - comparing
    mode = 0

    crypto_currencies_to_run = [
        CryptoCurrency("Bitcoin", "BTC")
        # CryptoCurrency("Ethereum", "ETC")
    ]
    Runner(crypto_currencies_to_run, mode)
