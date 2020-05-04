from src.pipeline.pipeline import Pipeline
from src.stages.execution.lstm.lstm_ethereum import LSTMEthereum
from src.stages.execution.tester import Tester
from src.stages.operation.preprocessor import Preprocessor
from src.stages.source.scrapper import Scrapper

if __name__ == '__main__':
    pipe = Pipeline()
    train = False
    test = False

    pipe.add(Scrapper("scrapper_config"))

    if train:
        pipe.add(Preprocessor("preprocessor_config"))
        pipe.add(LSTMEthereum("ethereum_config"))
        pipe.process()
    elif test:
        pipe.add(Preprocessor("preprocessor_config"))
        pipe.add(Tester("tester_config"))
        pipe.test()
    else:
        pipe.add(Tester("tester_config"))
        pipe.test()
