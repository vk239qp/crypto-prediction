from keras.layers import LSTM, Dense
from numpy import random
from tensorflow_core import metrics

from src.stages.execution.builder import Builder


class LSTMEthereum(Builder):

    def __init__(self, config_file: str):
        super().__init__(config_file)

    def build(self):
        random.seed(1337)
        self.model.add(LSTM(20, input_shape=(self.get_attributes('past_steps'), self.get_attributes('features'))))
        self.model.add(Dense(self.get_attributes('future_steps')))

        self.model.compile(loss='mae', optimizer='adam',
                           metrics=[metrics.RootMeanSquaredError(name='rmse')])
        self.model.summary()
