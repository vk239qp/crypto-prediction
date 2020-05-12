import imp

from src.helpers.crypto_currency import CryptoCurrency
from src.stages.execution.tester import Tester
from src.stages.operation.preprocessor import Preprocessor

from src.pipeline.pipeline import Pipeline
from src.stages.source.scrapper import Scrapper


class Runner:
    def __init__(self, crypto_currencies, mode):
        self.crypto_currencies = crypto_currencies
        self.mode = mode
        self.run()

    def run(self):
        if len(self.crypto_currencies) < 1:
            print("You did not define any crypto currencies which you wanna train or test")
            return

        for index, crypto in enumerate(self.crypto_currencies):
            pipe = Pipeline()
            pipe.add(Scrapper("scrapper_config", crypto.key))
            self.run_crypto_pipe(pipe, crypto)

    def run_crypto_pipe(self, pipe: Pipeline, crypto: CryptoCurrency):
        if self.mode == 0:
            pipe.add(Preprocessor("preprocessor_config"))
            pipe.add(self.get_lstm_class(crypto.name))
            pipe.process()
        elif self.mode == 1:
            pipe.add(Preprocessor("preprocessor_config"))
            pipe.add(Tester("tester_config"))
            pipe.test()
        elif self.mode == 2:
            pipe.add(Tester("tester_config", only_compare=True))
            pipe.test()

    def get_lstm_class(self, crypto_name: str):
        """ Dynamic import of modules """
        crypto_name_lower = crypto_name.lower()
        module_name = "lstm_" + crypto_name_lower
        module = imp.load_source(module_name, "stages/execution/lstm/lstm_" + crypto_name_lower + ".py")
        """ Dynamically creating the model classes (LSTM) """
        return getattr(module, ("LSTM" + crypto_name))(crypto_name_lower + "_config")
