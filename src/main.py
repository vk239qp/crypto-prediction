from src.pipeline.pipeline import Pipeline
from src.stages.execution.lstm.lstm_ethereum import LSTMEthereum
from src.stages.execution.tester import Tester
from src.stages.operation.preprocessor import Preprocessor
from src.stages.source.scrapper import Scrapper

if __name__ == '__main__':
    # 0 - train, 1 - test, 2 - comparing
    mode = 2
    pipe = Pipeline()
    pipe.add(Scrapper("scrapper_config"))

    if mode == 0:
        pipe.add(Preprocessor("preprocessor_config"))
        pipe.add(LSTMEthereum("ethereum_config"))
        pipe.process()
    elif mode == 1:
        pipe.add(Preprocessor("preprocessor_config"))
        pipe.add(Tester("tester_config"))
        pipe.test()
    elif mode == 2:
        pipe.add(Tester("tester_config", only_compare=True))
        pipe.test()
