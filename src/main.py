from src.pipeline.pipeline import Pipeline
from src.stages.operation.preprocessor import Preprocessor
from src.stages.source.scrapper import Scrapper

if __name__ == '__main__':
    scrapper = Scrapper("scrapper_config")
    data_util = Preprocessor("transformer_config")
    # lstm_bitcoin = LSTMBitcoin(PREV_DAYS, FUTURE_DAYS, 3)

    pipe = Pipeline()
    pipe.process(scrapper, data_util)
