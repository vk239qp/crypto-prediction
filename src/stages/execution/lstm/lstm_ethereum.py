from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import Adam
from numpy import random
from tensorflow_core import metrics

from src.stages.execution.builder import Builder


class LSTMEthereum(Builder):

    def __init__(self, config_file: str):
        super().__init__(config_file)

    def build(self):
        random.seed(52)
        self.model.add(LSTM(256, input_shape=(self.get_attributes('past_steps'), self.get_attributes('features')),
                            return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.get_attributes('future_steps')))
        self.model.add(Activation("tanh"))

        self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.002),
                           metrics=[metrics.RootMeanSquaredError(name='rmse')])
        self.model.summary()
