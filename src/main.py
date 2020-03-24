from src.pipeline.pipeline import Pipeline
from src.stages.execution.lstm_bitcoin import LSTMBitcoin
from src.stages.execution.lstm_ethereum import LSTMEthereum
from src.stages.operation.preprocessor import Preprocessor
from src.stages.source.scrapper import Scrapper

if __name__ == '__main__':
    pipe = Pipeline()
    pipe.add(Scrapper("scrapper_config"))
    pipe.add(Preprocessor("preprocessor_config"))
    # pipe.add(LSTMBitcoin("bitcoin_config"))
    pipe.add(LSTMEthereum("ethereum_config"))

    pipe.process()
